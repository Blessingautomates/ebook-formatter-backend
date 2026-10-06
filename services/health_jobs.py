"""The job lifecycle for a manuscript health scan.

A health scan is slow — it makes an API call per chunk of the book — and it
costs money. Both rule out doing it inside the request that asks for it. So the
route creates a row and returns immediately, the work happens in the background,
and the client polls.

**The honest limitation.** The background runner is
`fastapi.BackgroundTasks`, which executes in-process. It needs no new
infrastructure and is right for the current single-`uvicorn` deployment, but it
does not survive a restart: a job that was `running` when the process died stays
`running` forever, and the client spins. That is what `read_job` is for — it
notices a job that has been running longer than any scan could take and marks it
`stale`, so the author is told to try again instead of watching a progress bar
that will never move.

Swapping in a real queue means replacing this module, not its callers: the row
already has the shape a worker queue wants (`status`, `stage`, `progress`,
attempts are implicit in `created_at`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from services.analyzer import count_words
from services.auth import AuthedUser
from services.chapters import chapter_breakdown
from services.detection import detect_language
from services.health import index_lines
from services.health_ai import chunk_manuscript, run_ai_scan
from services.health_report import HealthReport, build_report, cast_for_prompt

TABLE = "manuscript_health"

#: PostgREST's code for "no such table" — the one error here that means the
#: schema was never applied, rather than that this request was wrong.
UNDEFINED_TABLE = "42P01"

#: How long a job may sit `queued` before we assume nothing ever picked it up.
#: The worker starts within a request lifetime, so this only needs to absorb a
#: slow event loop.
QUEUED_TIMEOUT_SECONDS = 120

#: How long a job may go without a progress write before we assume its worker
#: is gone. Generous on purpose: the AI stage can spend minutes inside a single
#: call, between retries and a long timeout. A false `stale` throws away work
#: the author paid for, so this errs towards patience.
RUNNING_TIMEOUT_SECONDS = 600

REQUEST_TIMEOUT_SECONDS = 20.0

#: Where each stage sits on the progress bar. The AI stage is a range rather
#: than a point, because it advances as chunks finish.
STAGE_START_PROGRESS = {
    "queued": 0,
    "extracting": 5,
    "metrics": 10,
    "rules": 30,
    "ai": 30,
    "assembling": 95,
    "done": 100,
}
AI_PROGRESS_SPAN = 60  # 30 -> 90

STATUSES = ("queued", "running", "succeeded", "failed", "stale")


class HealthJobError(Exception):
    """The job store refused the request."""


class HealthJobUnavailableError(HealthJobError):
    """Job persistence is not configured on this deployment. A 503."""


@dataclass(frozen=True)
class JobRequest:
    """Everything the background worker needs, captured at request time.

    The manuscript text is held here rather than re-read from storage, because
    the caller already has it — re-uploading a novel to poll for its own scan
    would be absurd.
    """

    job_id: str
    text: str
    title: str | None = None
    author: str | None = None
    genre: str | None = None
    #: Findings the author has already dismissed, so a re-run does not ask the
    #: same question twice.
    ignored: list[str] = field(default_factory=list)


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _url() -> str:
    base = _env("SUPABASE_URL")
    if not base:
        raise HealthJobUnavailableError(
            "Health jobs need SUPABASE_URL to reach the job store."
        )
    return base.rstrip("/")


def _service_key() -> str:
    key = _env("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        raise HealthJobUnavailableError(
            "Health jobs need SUPABASE_SERVICE_ROLE_KEY to write to the job store."
        )
    return key


def jobs_configured() -> bool:
    """Whether this deployment can persist health jobs."""
    return bool(_env("SUPABASE_URL") and _env("SUPABASE_SERVICE_ROLE_KEY"))


def _require_httpx() -> Any:
    """httpx, imported lazily so a deployment without job storage needs no HTTP client."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise HealthJobUnavailableError(
            "Health jobs need httpx. Add it to requirements.txt and rebuild."
        ) from exc
    return httpx


def _plain_error(payload: Any, status: int) -> HealthJobError:
    """Turn a PostgREST error body into something worth reading."""
    code = payload.get("code") if isinstance(payload, dict) else None
    message = payload.get("message") if isinstance(payload, dict) else None
    text = message if isinstance(message, str) and message.strip() else None

    if code == UNDEFINED_TABLE:
        return HealthJobUnavailableError(
            "The manuscript_health table is missing from the database. "
            "Apply supabase/schema.sql."
        )
    if status in (401, 403):
        return HealthJobUnavailableError(
            "The job store refused the service key. Check SUPABASE_SERVICE_ROLE_KEY."
        )
    return HealthJobError(text or f"The job store returned {status}.")


def _headers(*, token: str | None = None, service: bool = False) -> dict[str, str]:
    """PostgREST headers.

    Writes use the service role, because the worker has no user session — it
    runs in a background task, after the request that created it has been
    answered and its token discarded. Reads forward the caller's own token, so
    row-level security decides what they may see rather than this module.
    """
    if service:
        key = _service_key()
        return {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
    if not token:
        raise HealthJobUnavailableError(
            "Reading a health job needs the caller's session token."
        )
    return {
        "apikey": _env("SUPABASE_ANON_KEY") or token,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


async def _request(
    method: str,
    path: str,
    *,
    headers: dict[str, str],
    json_body: Any = None,
    params: dict[str, str] | None = None,
) -> Any:
    """One PostgREST call, with the transport failures mapped to a 503."""
    httpx = _require_httpx()
    url = f"{_url()}/rest/v1/{path}"

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.request(
                method, url, headers=headers, json=json_body, params=params
            )
    except Exception as exc:  # noqa: BLE001 - surfaced as a 503 below
        raise HealthJobUnavailableError(
            f"The job store could not be reached: {exc}"
        ) from exc

    if response.status_code >= 400:
        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - a non-JSON error body is still an error
            body = {}
        raise _plain_error(body, response.status_code)

    if not response.content:
        return None
    try:
        return response.json()
    except Exception:  # noqa: BLE001 - a 2xx with no body is not an error
        return None


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def serialize_report(report: HealthReport) -> dict[str, Any]:
    """Split a report into the columns it is stored in.

    Metrics and categories go to jsonb for the reason schema.sql already gives
    for `manuscripts.chapters`: they are read and written whole, never queried
    into, and a child table would buy nothing but joins. The headline numbers
    get their own columns because those *are* worth indexing and filtering on.
    """
    categories: list[dict[str, Any]] = []
    for category in report.categories:
        payload = asdict(category)
        for finding in payload["findings"]:
            # `fixable` is a property on Finding, so asdict does not carry it.
            # The UI needs it per finding, not just per category: a
            # misspelling on a character's name is not fixable while the one
            # beside it is, and re-deriving that from two nullable fields in
            # the client is a rule that would drift.
            finding["fixable"] = bool(
                finding.get("original") and finding.get("replacement")
            )
        categories.append(payload)

    return {
        "metrics": asdict(report.metrics),
        "categories": categories,
        "health_score": report.health_score,
        "total_findings": report.total_findings,
        "fixable_findings": report.fixable_findings,
        "ai_available": report.ai_available,
        "ai_note": report.ai_note,
        "ai_model": report.ai_model,
        "ai_calls": report.ai_calls,
        "ai_cost_usd": report.ai_cost_usd,
        "ai_credits": report.ai_credits,
        "scanned_at": report.scanned_at,
    }


def deserialize_report(row: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild the report object from a stored row, or None if there is none.

    The response is assembled rather than stored whole so that a job whose
    report was written by an older version of this code still reads back
    sensibly: each field is taken from its own column, and a missing one is
    missing rather than fatal.
    """
    if not row.get("metrics"):
        return None
    return {
        "metrics": row.get("metrics"),
        "categories": row.get("categories") or [],
        "health_score": row.get("health_score"),
        "total_findings": row.get("total_findings"),
        "fixable_findings": row.get("fixable_findings"),
        "ai_available": row.get("ai_available"),
        "ai_note": row.get("ai_note"),
        "ai_model": row.get("ai_model"),
        "ai_calls": row.get("ai_calls"),
        "ai_cost_usd": float(row.get("ai_cost_usd") or 0.0),
        "ai_credits": row.get("ai_credits") or 0,
        "scanned_at": row.get("scanned_at"),
    }


def _job_response(row: dict[str, Any]) -> dict[str, Any]:
    """The shape both job routes return."""
    return {
        "job_id": row.get("id"),
        "status": row.get("status"),
        "stage": row.get("stage"),
        "progress": row.get("progress"),
        "source_filename": row.get("source_filename"),
        "error": row.get("error"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "report": deserialize_report(row),
    }


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


async def create_job(
    user: AuthedUser | None,
    *,
    filename: str | None,
    manuscript_id: str | None = None,
) -> dict[str, Any]:
    """Insert a queued row and return it.

    Written with the service role rather than the caller's token, because
    `user_id` defaults to `auth.uid()` in the schema and a background worker
    has no session to supply it. The route has already authenticated the
    caller; this records who that was.
    """
    payload: dict[str, Any] = {
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "source_filename": filename[:255] if filename else None,
    }
    if user is not None:
        payload["user_id"] = user.id
    if manuscript_id:
        payload["manuscript_id"] = manuscript_id

    rows = await _request(
        "POST",
        TABLE,
        headers={**_headers(service=True), "Prefer": "return=representation"},
        json_body=payload,
    )
    if not rows:
        raise HealthJobError("The job store accepted the job but returned no row.")
    return rows[0]


async def update_job(job_id: str, **fields: Any) -> None:
    """Patch one job."""
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    await _request(
        "PATCH",
        TABLE,
        headers=_headers(service=True),
        json_body=fields,
        params={"id": f"eq.{job_id}"},
    )


async def _update_quietly(job_id: str, **fields: Any) -> None:
    """Patch one job, swallowing failures. For progress writes.

    Progress is a courtesy. A store blip while writing "62%" must not destroy a
    scan the author has already paid for, so only the final write is allowed to
    fail loudly.
    """
    try:
        await update_job(job_id, **fields)
    except HealthJobError:
        pass


async def run_job(request: JobRequest) -> None:
    """Run one health scan end to end. The background entry point.

    Every failure is caught and written to the row as `failed` with the reason.
    A job that vanishes without a word is the worst outcome available: the
    author cannot tell whether to wait or to try again.
    """
    job_id = request.job_id
    try:
        await update_job(
            job_id,
            status="running",
            stage="extracting",
            progress=STAGE_START_PROGRESS["extracting"],
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        text = request.text
        language = detect_language(text).code
        chapters = chapter_breakdown(text, count_words)
        lines = index_lines(text)

        await _update_quietly(
            job_id, stage="metrics", progress=STAGE_START_PROGRESS["metrics"]
        )

        await _update_quietly(
            job_id, stage="rules", progress=STAGE_START_PROGRESS["rules"]
        )

        ai = await run_ai_scan(
            text=text,
            lines=lines,
            chunks=chunk_manuscript(lines),
            chapters=chapters,
            cast=cast_for_prompt(text),
            language=language,
            genre=request.genre or "non-fiction",
            title=request.title,
            author=request.author,
            ignored=request.ignored,
            on_progress=lambda fraction: _update_quietly(
                job_id,
                stage="ai",
                progress=int(
                    STAGE_START_PROGRESS["ai"] + fraction * AI_PROGRESS_SPAN
                ),
            ),
        )

        await _update_quietly(
            job_id, stage="assembling", progress=STAGE_START_PROGRESS["assembling"]
        )

        report = build_report(
            text=text,
            chapters=chapters,
            language=language,
            ai=ai,
            title=request.title,
            author=request.author,
        )

        await update_job(
            job_id,
            status="succeeded",
            stage="done",
            progress=100,
            error=None,
            **serialize_report(report),
        )
    except Exception as exc:  # noqa: BLE001 - the reason is the whole point
        await _update_quietly(
            job_id,
            status="failed",
            error=str(exc)[:500] or exc.__class__.__name__,
        )


async def read_job(job_id: str, user: AuthedUser | None) -> dict[str, Any]:
    """Read one job, marking it stale if its worker has clearly died.

    The read is what makes an orphaned job visible. `BackgroundTasks` does not
    survive a restart, so without this a job that was mid-scan when the process
    stopped would report `running` at 30% forever.
    """
    if user is None:
        raise HealthJobError("Reading a health job needs a signed-in account.")

    rows = await _request(
        "GET",
        TABLE,
        headers=_headers(token=user.token),
        params={"id": f"eq.{job_id}", "select": "*", "limit": "1"},
    )
    if not rows:
        return {}

    row = rows[0]
    if row.get("status") in ("queued", "running") and _is_stale(row):
        await _mark_stale(job_id, row)
        row["status"] = "stale"
        row["error"] = (
            "This scan stopped before it finished — the server was probably "
            "restarted. Run it again."
        )

    return _job_response(row)


def _is_stale(row: dict[str, Any]) -> bool:
    """Whether a job has been unfinished for longer than any scan could take.

    Two different clocks, because the two ways a job dies look different. A job
    still `queued` long after it was created was never picked up — the process
    died between insert and dispatch. A job `running` with nothing written for a
    long time was picked up and then lost. Judging the second by `created_at`
    would also kill a scan that is merely slow, which is why progress writes
    keep `updated_at` fresh and buy the job more patience.
    """
    if row.get("status") == "queued":
        moment = _parse_time(row.get("created_at"))
        window = QUEUED_TIMEOUT_SECONDS
    else:
        moment = _parse_time(row.get("updated_at") or row.get("started_at"))
        window = RUNNING_TIMEOUT_SECONDS

    if moment is None:
        # No timestamp to judge by. Calling it stale would be a guess, and the
        # wrong guess throws away a scan that is still running.
        return False
    return datetime.now(timezone.utc) - moment > timedelta(seconds=window)


def _parse_time(value: Any) -> datetime | None:
    """Parse a PostgREST timestamp, or None if it is missing or unreadable."""
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


async def _mark_stale(job_id: str, row: dict[str, Any]) -> None:
    """Record the staleness, so the next poll does not have to work it out again."""
    await _update_quietly(
        job_id,
        status="stale",
        error=row.get("error")
        or "This scan stopped before it finished — the server was probably restarted.",
    )
