"""FastAPI entry point for the automated ebook formatting platform."""

from __future__ import annotations

import io
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services import auth, credits, exporter, packaging
from services.analyzer import analyze_manuscript
from services.auth import AuthedUser, AuthError, AuthUnavailableError
from services.cover import CoverError, CoverReport, validate_cover
from services.credits import CreditsError, CreditsUnavailableError, InsufficientCredits
from services.extractors import ManuscriptError, extract_text

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
