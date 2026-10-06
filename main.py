"""FastAPI entry point for the automated ebook formatting platform."""

from __future__ import annotations

import io
from dataclasses import asdict
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services import auth, credits, exporter, health_jobs, packaging
from services.analyzer import analyze_manuscript
from services.auth import AuthedUser, AuthError, AuthUnavailableError
from services.cover import CoverError, CoverReport, validate_cover
from services.credits import CreditsError, CreditsUnavailableError, InsufficientCredits
from services.extractors import ManuscriptError, extract_text
from services.health_jobs import (
    HealthJobError,
    HealthJobUnavailableError,
    JobRequest,
)

app = FastAPI(
    title="Ebook Formatting Platform",
    description="Analyzes an uploaded manuscript ahead of formatting.",
    version="0.1.0",
)


class TypoLocation(BaseModel):
    """One suspected misspelling, with where it occurs."""

    word: str = Field(description="The word as it appears in the manuscript.")
    suggestions: list[str] = Field(description="Up to three suggested corrections.")
    chapter: str = Field(description="Chapter it is in, or 'Front Matter'.")
    line_number: int = Field(description="Line the word is on, 1-based.")
    context: str = Field(description="A snippet of the surrounding words.")
    likely_proper_noun: bool = Field(
        description="A capitalised word not merely opening a sentence, so a "
        "name or brand rather than necessarily a mistake."
    )


class ChapterSummary(BaseModel):
    """One chapter of a manuscript, and how much of it the chapter holds."""

    title: str = Field(description="The heading, without its Markdown marker.")
    line_number: int = Field(description="Line the heading is on, 1-based.")
    word_count: int = Field(description="Words in the chapter, heading included.")


class BookAnalysis(BaseModel):
    """Measurements taken from one manuscript."""

    word_count: int = Field(description="Total words in the manuscript.")
    chapter_count: int = Field(description="Chapter markers found in the manuscript.")
    chapters: list[ChapterSummary] = Field(
        description="Per-chapter breakdown, in document order. A 'Front Matter' "
        "entry is included when the manuscript has text before its first "
        "heading, so this can be one longer than chapter_count."
    )
    detected_language: str = Field(description="ISO language code, or 'unknown'.")
    language_name: str = Field(description="Human-readable language name.")
    text_direction: Literal["ltr", "rtl"] = Field(description="Reading direction.")
    script_type: Literal[
        "latin", "rtl", "cjk", "cyrillic", "greek", "devanagari"
    ] = Field(description="Writing system.")
    estimated_pages: int = Field(description="Estimated typeset page count.")
    token_cost: int = Field(description="Cost of formatting this manuscript.")
    typo_count: int = Field(description="Total suspected misspellings found.")
    typos: list[TypoLocation] = Field(description="The suspected misspellings.")
    typo_check_available: bool = Field(
        description="False when no dictionary exists for the detected language, "
        "in which case typo_count is 0 and says nothing about the manuscript."
    )
    typo_check_note: str | None = Field(
        description="Why the typo list is incomplete, if it is."
    )
    manuscript_text: str | None = Field(
        default=None,
        description="The extracted manuscript, returned only when include_text "
        "was requested. Lets a caller apply the typo corrections and hand the "
        "corrected text back to /api/export-book, rather than uploading twice.",
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/analyze-book",
    response_model=BookAnalysis,
    responses={
        401: {"description": "Authentication is enabled and the token is missing or invalid."},
        402: {"description": "Not enough credits for this manuscript."},
        503: {"description": "Authentication or metering is enabled but misconfigured."},
    },
)
async def analyze_book(
    file: UploadFile = File(...),
    include_text: bool = Form(default=False),
    authorization: str | None = Header(default=None),
) -> BookAnalysis:
    """Analyze an uploaded .docx, .md, or .txt manuscript.

    When authentication is configured, the caller is identified and the
    analysis is charged to their ledger. When it is not, this behaves exactly as
    it always has — no header required, nothing metered — so a deployment that
    never sets a Supabase secret loses nothing by upgrading.
    """
    data = await file.read()
    if not data.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        user = auth.current_user(authorization)
    except AuthError as exc:
        raise HTTPException(
            status_code=401, detail=str(exc), headers={"WWW-Authenticate": "Bearer"}
        ) from exc
    except AuthUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = analyze_manuscript(file.filename or "", data)
    except ManuscriptError as exc:
        # Bad input from the caller, not a server fault: unsupported extension or
        # a file too damaged to parse.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = asdict(result)
    if include_text:
        try:
            payload["manuscript_text"] = extract_text(file.filename or "", data)
        except ManuscriptError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Charged once the request is known to be completable, so a caller is never
    # billed for an analysis that then failed on the way back. The cost is the
    # analyzer's own `token_cost`, which is why it has to come after the
    # analysis rather than before it.
    if user is not None:
        await _charge_for_analysis(user, result.token_cost, file.filename or "")

    return BookAnalysis(**payload)


async def _charge_for_analysis(
    user: AuthedUser, token_cost: int, filename: str
) -> None:
    """Bill an analysis to the caller's ledger.

    Metering is skipped when it is not configured, rather than failing the
    request: a deployment that authenticates but has no ledger still formats
    books. It is *not* skipped when metering is configured but broken — that is
    a 503, because silently free work on a deployment that means to charge is a
    bug discovered on the invoice.
    """
    if not credits.credits_configured():
        return

    try:
        # The allowance is monthly and idempotent, so this is what makes a new
        # account's first analysis affordable rather than a 402 on day one.
        await credits.ensure_grant(user)
        await credits.debit(
            user,
            amount=token_cost,
            reason="analysis",
            ref=filename[:200] if filename else None,
        )
    except InsufficientCredits as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    except CreditsError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get(
    "/api/credits",
    responses={
        200: {"description": "The caller's balance, spend and period."},
        401: {"description": "Authentication is enabled and the token is missing or invalid."},
        503: {"description": "Metering is not configured on this deployment."},
    },
)
async def read_credits(authorization: str | None = Header(default=None)) -> dict[str, object]:
    """The caller's credit position.

    The browser reads this directly from Supabase too; this route exists so a
    non-browser client — a script, or a future CLI — has the same view without
    needing to speak PostgREST.
    """
    try:
        user = auth.current_user(authorization)
    except AuthError as exc:
        raise HTTPException(
            status_code=401, detail=str(exc), headers={"WWW-Authenticate": "Bearer"}
        ) from exc
    except AuthUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if user is None:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured on this deployment, so there is no account to read credits for.",
        )

    try:
        await credits.ensure_grant(user)
        position = await credits.position(user)
    except CreditsUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except CreditsError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "balance": position.balance,
        "spent": position.spent,
        "period_start": position.period_start,
    }


@app.post(
    "/api/export-book",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"application/octet-stream": {}},
            "description": "The rendered book, as a download.",
        },
        400: {"description": "Unsupported format or genre, or a bad manuscript."},
        503: {"description": "The library needed for that format is not installed."},
    },
)
async def export_book(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    genre: str = Form(default="non-fiction"),
    export_format: str = Form(default="pdf", alias="format"),
    script_type: str = Form(default="latin"),
    text_direction: str = Form(default="ltr"),
    title: str = Form(default="Untitled"),
    author: str | None = Form(default=None),
    language: str = Form(default="en"),
    custom_font: str | None = Form(default=None),
    font_size: float | None = Form(default=None),
    trim_size: str = Form(default=exporter.DEFAULT_TRIM),
) -> StreamingResponse:
    """Render a manuscript to PDF, EPUB, DOCX, RTF or TXT and stream it back.

    Supply the manuscript either as an uploaded .docx/.md/.txt file or as the
    `text` field. `script_type` and `text_direction` are the values
    /api/analyze-book returned for this book. Pass `text` when the manuscript
    has been edited since analysis; pass the file again when it has not.
    """
    manuscript = await _manuscript_text(file, text)

    try:
        result = exporter.export_book(
            exporter.ExportRequest(
                text=manuscript,
                title=title,
                author=author or None,
                genre=genre,
                format=export_format,
                script_type=script_type,
                text_direction=text_direction,
                language=language,
                custom_font=custom_font or None,
                font_size=font_size,
                trim_size=trim_size,
            )
        )
    except exporter.BackendUnavailableError as exc:
        # The request itself was fine; this deployment cannot serve that format.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except exporter.ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StreamingResponse(
        io.BytesIO(result.content),
        media_type=result.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{result.filename}"',
            "Content-Length": str(len(result.content)),
        },
    )


class CoverCheck(BaseModel):
    """One measurement taken from a cover, and what it was measured against."""

    id: str = Field(description="Stable identifier: dimensions, resolution, dpi, spine, bleed, safe-area.")
    label: str = Field(description="Human-readable name of the check.")
    status: Literal["pass", "warn", "fail"] = Field(
        description="Outcome of this check. 'warn' means it will print but is "
        "worth fixing; 'fail' means it will not print correctly."
    )
    measured: str = Field(description="What the file actually is.")
    required: str = Field(description="What it needs to be.")
    message: str = Field(description="What to do about it, in plain language.")


class CoverSpecification(BaseModel):
    """The dimensions a cover has to be, before any file is compared to them."""

    trim_width_in: float
    trim_height_in: float
    spine_width_in: float = Field(
        description="Computed from the page count and the paper stock."
    )
    bleed_in: float
    safe_area_in: float
    full_width_in: float = Field(
        description="Back cover, spine and front cover side by side, plus bleed."
    )
    full_height_in: float
    required_width_px: int
    required_height_px: int
    page_count: int
    paper: str


class CoverValidation(BaseModel):
    """The result of validating one cover."""

    status: Literal["passed", "warnings", "failed"] = Field(
        description="The worst status among the checks."
    )
    width_px: int
    height_px: int
    dpi: int | None = Field(
        description="The resolution the file declares, or null when it declares "
        "none. Distinct from the measured resolution in the checks, which is "
        "computed from the pixels and cannot be wrong."
    )
    specification: CoverSpecification
    checks: list[CoverCheck]


@app.post("/api/validate-cover", response_model=CoverValidation)
async def validate_cover_endpoint(
    file: UploadFile = File(...),
    trim_size: str = Form(default=exporter.DEFAULT_TRIM),
    page_count: int = Form(default=0),
    paper: str = Form(default="white"),
) -> CoverValidation:
    """Measure an uploaded cover against the specification for its book.

    `page_count` decides the spine width, so it comes from the analysis of the
    manuscript the cover belongs to. It is the estimated page count, which is
    why the spine figure is reported to four decimal places: it is a
    calculation, not a measurement of the artwork.
    """
    data = await file.read()

    try:
        report = validate_cover(data, trim_size, page_count, paper)
    except CoverError as exc:
        # Bad input — not an image, an unknown trim, or an empty upload — rather
        # than a cover that failed its checks. The two must not be confused.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CoverValidation(**asdict(report))


@app.post(
    "/api/package-book",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"application/zip": {}},
            "description": "Every rendered format, the cover, and the metadata.",
        },
        400: {"description": "Unsupported genre or trim, or a bad manuscript."},
        503: {"description": "No format could be rendered in this deployment."},
    },
)
async def package_book(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    cover: UploadFile | None = File(default=None),
    title: str = Form(default="Untitled"),
    author: str | None = Form(default=None),
    genre: str = Form(default="non-fiction"),
    script_type: str = Form(default="latin"),
    text_direction: str = Form(default="ltr"),
    language: str = Form(default="en"),
    custom_font: str | None = Form(default=None),
    font_size: float | None = Form(default=None),
    trim_size: str = Form(default=exporter.DEFAULT_TRIM),
    page_count: int = Form(default=0),
    paper: str = Form(default="white"),
    isbn: str | None = Form(default=None),
) -> StreamingResponse:
    """Render every format and return them as one downloadable archive.

    This is the sign-off modal's download. It renders through the same
    `/api/export-book` code path, so a package cannot contain a book that is
    different from the one the export page produces.

    A format whose library is missing is skipped and named in the archive's
    README rather than failing the request: an author should not lose the four
    formats that work because the fifth does not.
    """
    manuscript = await _manuscript_text(file, text)

    request = exporter.ExportRequest(
        text=manuscript,
        title=title,
        author=author or None,
        genre=genre,
        script_type=script_type,
        text_direction=text_direction,
        language=language,
        custom_font=custom_font or None,
        font_size=font_size,
        trim_size=trim_size,
    )

    cover_bytes: bytes | None = None
    cover_name: str | None = None
    cover_report: CoverReport | None = None

    if cover is not None and cover.filename:
        cover_bytes = await cover.read()
        if cover_bytes.strip():
            cover_name = cover.filename
            try:
                cover_report = validate_cover(
                    cover_bytes, trim_size, page_count, paper
                )
            except CoverError as exc:
                # The cover is optional to the package, so an unreadable one is
                # reported rather than fatal — the manuscript still exports.
                raise HTTPException(
                    status_code=400, detail=f"Cover: {exc}"
                ) from exc

    try:
        package = packaging.build_package(
            request,
            page_count=page_count,
            cover=cover_bytes,
            cover_name=cover_name,
            cover_report=cover_report,
            isbn=isbn or None,
        )
    except exporter.BackendUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except exporter.ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = package.result
    return StreamingResponse(
        io.BytesIO(result.content),
        media_type=result.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{result.filename}"',
            "Content-Length": str(len(result.content)),
        },
    )


async def _manuscript_text(file: UploadFile | None, text: str | None) -> str:
    """Take the manuscript from an uploaded file if there is one, else from text."""
    if file is not None and file.filename:
        data = await file.read()
        if data.strip():
            try:
                return extract_text(file.filename, data)
            except ManuscriptError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    if text and text.strip():
        return text

    raise HTTPException(
        status_code=400,
        detail="Provide a manuscript file upload or a non-empty text field.",
    )


# --------------------------------------------------------------------------
# Manuscript intelligence: the health report and the Book Doctor's source data
# --------------------------------------------------------------------------


class HealthReadingLevel(BaseModel):
    """Readability scores. Defined for English only; see HealthMetrics."""

    flesch_reading_ease: float = Field(
        description="Flesch Reading Ease. Higher is easier; 60-70 is plain English."
    )
    flesch_kincaid_grade: float = Field(
        description="US school grade level the manuscript reads at."
    )
    label: str = Field(description="The grade in plain language.")


class HealthMetrics(BaseModel):
    """How big the manuscript is and how hard it is to read."""

    word_count: int
    character_count: int = Field(description="Every character, spaces included.")
    character_count_no_spaces: int
    sentence_count: int
    paragraph_count: int
    chapter_count: int = Field(description="Excludes the Front Matter entry.")
    print_pages: int = Field(
        description="Estimated typeset pages. Uses the same words-per-page "
        "figure as the pre-scan and the exporter, so the three cannot disagree."
    )
    reading_time_minutes: int = Field(description="At 220 words per minute.")
    reading_level: HealthReadingLevel | None = Field(
        description="Null when the manuscript is not in English, in which case "
        "reading_level_note explains why. Flesch-Kincaid is a formula over "
        "English syllables and would return a meaningless number elsewhere."
    )
    reading_level_note: str | None = Field(
        description="Why no reading level is reported, when none is."
    )
    detected_genre: str = Field(
        description="Best guess at the genre, offered as a default the author "
        "can override in the genre selector."
    )
    genre_confidence: float = Field(
        description="0-1. Low means the signals were close or absent."
    )
    genre_source: str = Field(description="'heuristic', 'ai', or 'default'.")
    completeness: int = Field(
        description="0-100, from structural signals only — chapters present, "
        "chapters of sane length, an ending that finishes. Distinct from the "
        "publishing-readiness score in the UI, which measures saved-row state "
        "(title set, cover validated, sign-off signed). The two are different "
        "numbers and must not be shown as one."
    )
    completeness_notes: list[str] = Field(
        description="What is missing, one line each. Empty at 100."
    )
    estimated_processing_seconds: int = Field(
        description="How long a scan of this manuscript takes."
    )


class HealthFinding(BaseModel):
    """One thing worth the author's attention."""

    id: str = Field(
        description="Stable across re-runs of the same manuscript, so a "
        "reviewed or dismissed finding is not asked about twice."
    )
    category: str = Field(description="Which category this belongs to.")
    severity: Literal["high", "medium", "low"]
    message: str = Field(description="What is wrong, addressed to the author.")
    chapter: str = Field(description="Chapter it is in, or 'Front Matter'.")
    line_number: int = Field(description="Line in the extracted text, 1-based.")
    context: str = Field(description="A snippet of the surrounding text.")
    suggestion: str | None = Field(
        default=None, description="What to do about it, when there is advice."
    )
    original: str | None = Field(
        default=None,
        description="The exact text to replace. Null for anything that is a "
        "judgement call rather than a mechanical fix.",
    )
    replacement: str | None = Field(
        default=None, description="What to replace it with."
    )
    fixable: bool = Field(
        description="True when original and replacement are both set, so this "
        "finding may be applied automatically. Only mechanical corrections "
        "ever qualify: prose judgements never do."
    )


class HealthCategory(BaseModel):
    """One category of the report: what it is, and what it found."""

    id: str
    label: str
    group: str = Field(
        description="The Book Doctor section this is filed under: Writing "
        "health, Consistency, Dialogue, Pacing, or Typography & structure."
    )
    source: Literal["rules", "ai"] = Field(
        description="Where the findings came from. A rule finding is exact and "
        "free; an AI finding is a judgement."
    )
    fixable: bool = Field(
        description="Whether Fix All may apply findings in this category."
    )
    count: int
    available: bool = Field(
        description="False when this category could not be checked at all — no "
        "dictionary for the language, or no API key. Distinct from count 0, "
        "which means it was checked and was clean."
    )
    note: str | None = Field(
        default=None, description="Why it could not run, when it could not."
    )
    findings: list[HealthFinding]


class HealthReport(BaseModel):
    """The whole manuscript health report."""

    metrics: HealthMetrics
    categories: list[HealthCategory] = Field(
        description="Every category, whether or not it found anything, so the "
        "dashboard renders a stable breakdown."
    )
    total_findings: int
    fixable_findings: int = Field(
        description="How many Fix All would apply. Always a subset of the "
        "mechanical categories."
    )
    health_score: int = Field(
        description="0-100, from defect density rather than defect count: a "
        "long novel with thirty findings is in better shape than a short story "
        "with thirty."
    )
    ai_available: bool
    ai_note: str | None = Field(
        default=None, description="Why the AI categories are missing, when they are."
    )
    ai_model: str | None = None
    ai_calls: int
    ai_cost_usd: float = Field(
        description="What the AI pass actually cost, computed from the token "
        "usage the API reported. This is a real cost and is a different "
        "quantity from the pre-scan's token_cost, which is a platform pricing "
        "formula. They are never summed."
    )
    ai_credits: int = Field(description="ai_cost_usd expressed in platform credits.")
    scanned_at: str


class HealthJobAccepted(BaseModel):
    """The response to starting a scan. Poll the job for the report."""

    job_id: str
    status: Literal["queued", "running", "succeeded", "failed", "stale"]


class HealthJobStatus(BaseModel):
    """The state of one health scan, and its report once it has one."""

    job_id: str
    status: Literal["queued", "running", "succeeded", "failed", "stale"] = Field(
        description="'stale' means the scan stopped before finishing — the "
        "server was probably restarted mid-run, since the background runner "
        "does not survive one. The author should start it again."
    )
    stage: str | None = Field(
        default=None,
        description="queued, extracting, metrics, rules, ai, assembling, done.",
    )
    progress: int = Field(description="0-100.")
    source_filename: str | None = None
    error: str | None = Field(
        default=None, description="Why it failed, when it failed."
    )
    created_at: str | None = None
    updated_at: str | None = None
    report: HealthReport | None = Field(
        default=None, description="Present once status is 'succeeded'."
    )


@app.post(
    "/api/manuscript/analyze",
    status_code=202,
    response_model=HealthJobAccepted,
    responses={
        202: {"description": "The scan is queued. Poll GET /api/manuscript/health/{job_id}."},
        400: {"description": "No manuscript, or a file that could not be read."},
        401: {"description": "Authentication is enabled and the token is missing or invalid."},
        503: {"description": "Health scans are not configured on this deployment."},
    },
)
async def start_manuscript_analysis(
    background: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    title: str | None = Form(default=None),
    author: str | None = Form(default=None),
    genre: str = Form(default="non-fiction"),
    manuscript_id: str | None = Form(default=None),
    ignored: list[str] = Form(default=[]),
    authorization: str | None = Header(default=None),
) -> HealthJobAccepted:
    """Start a manuscript health scan and return a job to poll.

    Accepts the manuscript either as an uploaded .docx/.epub/.md/.txt file or as
    the `text` field — a caller that already has the extracted text from
    /api/analyze-book should pass that rather than uploading the book a second
    time.

    Scans are per-account because the job row is owner-scoped, so unlike
    /api/analyze-book this route needs authentication to be configured.
    """
    try:
        user = auth.current_user(authorization)
    except AuthError as exc:
        raise HTTPException(
            status_code=401, detail=str(exc), headers={"WWW-Authenticate": "Bearer"}
        ) from exc
    except AuthUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if user is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Health scans are stored per account, so they need "
                "authentication to be configured on this deployment."
            ),
        )

    if not health_jobs.jobs_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Health scans need SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
                "to store their results."
            ),
        )

    manuscript = await _manuscript_text(file, text)
    filename = file.filename if file is not None else None

    try:
        job = await health_jobs.create_job(
            user, filename=filename, manuscript_id=manuscript_id or None
        )
    except HealthJobUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HealthJobError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # The text travels in the task rather than being re-read from storage: the
    # caller already sent it, and asking them to upload a novel again just to
    # poll for its own scan would be absurd.
    background.add_task(
        health_jobs.run_job,
        JobRequest(
            job_id=job["id"],
            text=manuscript,
            title=title or None,
            author=author or None,
            genre=genre or None,
            ignored=[item for item in ignored if item.strip()],
        ),
    )

    return HealthJobAccepted(job_id=job["id"], status=job["status"])


@app.get(
    "/api/manuscript/health/{job_id}",
    response_model=HealthJobStatus,
    responses={
        401: {"description": "Authentication is enabled and the token is missing or invalid."},
        404: {"description": "No such scan, or it belongs to another account."},
        503: {"description": "Health scans are not configured on this deployment."},
    },
)
async def read_manuscript_health(
    job_id: str,
    authorization: str | None = Header(default=None),
) -> HealthJobStatus:
    """Read one health scan.

    The row is fetched with the caller's own token, so row-level security
    decides what they may see. A scan belonging to someone else reports 404
    rather than 403 — the same answer as a scan that does not exist, which
    keeps this route from confirming that a given job id is real.
    """
    try:
        user = auth.current_user(authorization)
    except AuthError as exc:
        raise HTTPException(
            status_code=401, detail=str(exc), headers={"WWW-Authenticate": "Bearer"}
        ) from exc
    except AuthUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if user is None:
        raise HTTPException(
            status_code=503,
            detail="Health scans need authentication to be configured.",
        )

    try:
        row = await health_jobs.read_job(job_id, user)
    except HealthJobUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HealthJobError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not row:
        raise HTTPException(status_code=404, detail="No such health scan.")

    return HealthJobStatus(**row)
