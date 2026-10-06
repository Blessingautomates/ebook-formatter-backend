"""The AI half of the manuscript health report.

The rule detectors in services.health catch what a regular expression can see.
Grammar, continuity, dialogue and plot cannot be checked that way — they need
something that has read the book. That is what this module is.

Three properties it is built around:

**It is optional.** No API key means `ai_configured()` is False and every AI
category is reported as unavailable-with-a-reason, exactly as
`services.typos` reports a missing dictionary. A deployment that never sets a
key still serves a useful health report.

**It never invents a location.** The model is asked to quote the text it is
complaining about, and this module finds that quote in the manuscript itself to
derive the line number. A model-reported line number is not trusted, and a
finding whose quote cannot be located is downgraded to advisory (no `original`,
so the Book Doctor's Fix All cannot touch it) rather than applied blind.

**It is priced honestly.** `ai_cost_usd` is computed from the token usage the
API actually returned, priced per model. It is a different quantity from
services.analyzer's `token_cost`, which is a synthetic platform pricing formula,
and the two are never added together or shown as one number.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from services.chapters import FRONT_MATTER, ChapterSummary
from services.health import (
    CONTINUITY,
    DIALOGUE,
    GRAMMAR,
    PLOT,
    Finding,
    Line,
    finding_id,
)

# The Anthropic API version this module speaks. Pinned rather than floating:
# the structured-output request shape below is version-specific.
ANTHROPIC_VERSION = "2023-06-01"

#: Sonnet for the per-chunk prose pass — fast, and the task is local.
DEFAULT_MODEL = "claude-sonnet-5-5"

#: Opus for continuity and plot. Those are the judgements where being wrong is
#: expensive, the input is a digest rather than the whole book, and it runs
#: once rather than once per chunk.
CONTINUITY_MODEL = "claude-opus-5-5"

# $/million tokens (input, output). Cache reads are priced at 0.1x input and
# cache writes at 1.25x, which is the API's own rule rather than a choice here.
_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-5-5": (3.0, 15.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-5-5": (4.0, 20.0),
}
_PRICE_DEFAULT = (3.0, 15.0)

#: What one platform credit is worth, so an AI cost can be expressed in the
#: same unit the credit ledger uses. A policy figure, not a technical one —
#: change it here and `ai_credits` follows.
USD_PER_CREDIT = 0.01

# Retry ladder. A 429 is someone else's traffic and is worth waiting out; a 5xx
# usually clears faster. Both are bounded so a bad afternoon at the API cannot
# hold a job open indefinitely.
ATTEMPTS = 3
RETRYABLE = frozenset({500, 502, 503, 529})
RATE_LIMITED = 429

# Chunking. A novel never fits one call, and per-chapter chunks give correct
# chapter attribution for free.
MAX_CHUNK_CHARS = 24_000
MAX_OUTPUT_TOKENS = 8_000
CONTINUITY_MAX_OUTPUT_TOKENS = 16_000

REQUEST_TIMEOUT_SECONDS = 180.0


class HealthAIError(Exception):
    """The AI scan could not be completed."""


class HealthAIUnavailableError(HealthAIError):
    """The AI scan is not available in this deployment.

    Distinct from `HealthAIError` for the same reason `CreditsUnavailableError`
    is distinct from `CreditsError`: "nobody configured this" is an operator's
    problem and a 503, whereas a refusal or a malformed response is a bug.
    """


@dataclass(frozen=True)
class Chunk:
    """A slice of the manuscript small enough to send in one call."""

    index: int
    text: str
    chapters: tuple[str, ...]
    #: Line number, in the whole manuscript, of this chunk's first line.
    first_line: int


@dataclass
class AIResult:
    """What the AI pass found, what it cost, and whether it ran at all."""

    findings: list[Finding] = field(default_factory=list)
    available: bool = False
    note: str | None = None
    cost_usd: float = 0.0
    credits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    model: str | None = None


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _api_key() -> str:
    return _env("ANTHROPIC_API_KEY")


def _base_url() -> str:
    return (_env("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")


def chunk_model() -> str:
    return _env("HEALTH_AI_MODEL") or DEFAULT_MODEL


def continuity_model() -> str:
    return _env("HEALTH_CONTINUITY_MODEL") or CONTINUITY_MODEL


def ai_configured() -> bool:
    """Whether an AI scan can run here.

    Only the key is required. The HTTP client is checked separately by
    `_require_httpx`, so a missing key reads as "not switched on" while a
    missing library reads as "broken deployment" — two different problems.
    """
    return bool(_api_key())


def _thinking() -> dict[str, str] | None:
    """The thinking block to send, or None to omit it.

    Adaptive thinking is the right default on current models, but it is not
    accepted by older ones, so a deployment that pins `HEALTH_AI_MODEL` to an
    older model can set `HEALTH_AI_THINKING=off` rather than being stuck.
    """
    if _env("HEALTH_AI_THINKING").lower() in {"off", "0", "false", "none"}:
        return None
    return {"type": "adaptive"}


def _require_httpx():
    """Import httpx on first use.

    Same reasoning as services/credits.py: a deployment with the AI scan
    switched off should not need the library installed to serve anything else.
    """
    try:
        import httpx  # noqa: PLC0415  (deliberately late)
    except ImportError as exc:
        raise HealthAIUnavailableError(
            "The AI health scan needs the 'httpx' package, which is not "
            "installed in this deployment."
        ) from exc
    return httpx


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def price_usage(model: str, usage: dict) -> float:
    """The dollar cost of one API response, from the usage it reported.

    Longest-prefix match on the model id, so a dated snapshot
    (`claude-sonnet-5-5-20260101`) prices as its family. An unrecognised model
    falls back to the default row rather than pricing at zero — a wrong number
    that looks plausible is easier to notice than a free one.
    """
    key = (model or "").lower()
    price_in, price_out = _PRICE_DEFAULT
    for name in sorted(_PRICE_PER_MTOK, key=len, reverse=True):
        if key.startswith(name):
            price_in, price_out = _PRICE_PER_MTOK[name]
            break

    def count(name: str) -> int:
        try:
            return int(usage.get(name) or 0)
        except (TypeError, ValueError):
            return 0

    return (
        count("input_tokens") * price_in
        + count("output_tokens") * price_out
        + count("cache_read_input_tokens") * price_in * 0.1
        + count("cache_creation_input_tokens") * price_in * 1.25
    ) / 1_000_000


def credits_for(cost_usd: float) -> int:
    """Convert a dollar cost to platform credits, never rounding down to zero."""
    if cost_usd <= 0:
        return 0
    return max(1, round(cost_usd / USD_PER_CREDIT))


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def chunk_manuscript(
    lines: list[Line], max_chars: int = MAX_CHUNK_CHARS
) -> list[Chunk]:
    """Split the manuscript into call-sized chunks, never mid-chapter if avoidable.

    Chapters are packed into chunks in order until the character budget is
    reached. A single chapter longer than the budget is split at paragraph
    boundaries — the one case where a chunk spans a chapter break, and it is
    recorded in `chapters` so the findings still carry the right attribution.
    """
    if not lines:
        return []

    # Group consecutive lines by chapter, then pack the groups.
    groups: list[list[Line]] = []
    for line in lines:
        if groups and groups[-1][0].chapter == line.chapter:
            groups[-1].append(line)
        else:
            groups.append([line])

    chunks: list[Chunk] = []
    current: list[Line] = []

    def flush() -> None:
        if not current:
            return
        seen: list[str] = []
        for line in current:
            if line.chapter not in seen:
                seen.append(line.chapter)
        chunks.append(
            Chunk(
                index=len(chunks),
                text="\n".join(line.text for line in current),
                chapters=tuple(seen),
                first_line=current[0].number,
            )
        )
        current.clear()

    for group in groups:
        size = sum(len(line.text) + 1 for line in group)
        if size > max_chars:
            # One chapter too large for a single call: split it on blank lines,
            # which is where a scene break is, and is the least disruptive place
            # to cut prose in half.
            flush()
            for piece in _split_group(group, max_chars):
                current.extend(piece)
                flush()
            continue

        if sum(len(line.text) + 1 for line in current) + size > max_chars:
            flush()
        current.extend(group)

    flush()
    return chunks


def _split_group(group: list[Line], max_chars: int) -> list[list[Line]]:
    """Break one oversized chapter into pieces under `max_chars`."""
    pieces: list[list[Line]] = []
    current: list[Line] = []
    size = 0
    for line in group:
        length = len(line.text) + 1
        # Prefer to cut on a blank line; fall back to cutting on length when a
        # chapter has no paragraph breaks at all.
        if current and size + length > max_chars:
            pieces.append(current)
            current, size = [], 0
        current.append(line)
        size += length
    if current:
        pieces.append(current)
    return pieces


# --------------------------------------------------------------------------
# Locating a finding
# --------------------------------------------------------------------------


def _locate(chunk: Chunk, quote: str) -> tuple[int, bool]:
    """Find `quote` in `chunk`, as (line number, found verbatim).

    The second element is what decides whether a correction may be applied
    automatically. A verbatim hit means the text is exactly where the model says
    it is, so replacing it is safe. A whitespace-insensitive hit means the model
    reflowed a line break inside the quote — the finding is real but the span is
    not exact, so it is reported without `original` and the author edits it by
    hand.
    """
    needle = quote.strip()
    if not needle:
        return chunk.first_line, False

    position = chunk.text.find(needle)
    if position >= 0:
        return chunk.first_line + chunk.text.count("\n", 0, position), True

    flat_text = re.sub(r"\s+", " ", chunk.text)
    flat_needle = re.sub(r"\s+", " ", needle)
    position = flat_text.find(flat_needle)
    if position < 0:
        return chunk.first_line, False

    # Walk the original lines, measuring how much normalised text each
    # contributes, until the match position is passed.
    consumed = 0
    for offset, line in enumerate(chunk.text.splitlines()):
        consumed += len(re.sub(r"\s+", " ", line).strip()) + 1
        if consumed > position:
            return chunk.first_line + offset, False
    return chunk.first_line, False


def _context_for(chunk: Chunk, line_number: int) -> str:
    """The line the finding sits on, as a short snippet."""
    offset = line_number - chunk.first_line
    lines = chunk.text.splitlines()
    if 0 <= offset < len(lines):
        text = lines[offset].strip()
        return text if len(text) <= 200 else text[:200] + "..."
    return ""


# --------------------------------------------------------------------------
# Prompts and schemas
# --------------------------------------------------------------------------

_FINDING_ITEM = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["grammar", "dialogue", "plot"],
        },
        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
        "message": {
            "type": "string",
            "description": "What is wrong, in one sentence, addressed to the author.",
        },
        "quote": {
            "type": "string",
            "description": "The exact text from the manuscript that is wrong, "
            "copied character for character. Must appear in the manuscript.",
        },
        "correction": {
            "type": "string",
            "description": "The corrected text, replacing the quote exactly. "
            "Omit for anything that is a judgement call rather than a fix.",
        },
    },
    "required": ["category", "severity", "message", "quote"],
    "additionalProperties": False,
}

CHUNK_SCHEMA = {
    "type": "object",
    "properties": {"findings": {"type": "array", "items": _FINDING_ITEM}},
    "required": ["findings"],
    "additionalProperties": False,
}

CONTINUITY_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["continuity", "plot", "dialogue"],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "message": {"type": "string"},
                    "quote": {
                        "type": "string",
                        "description": "Text from the manuscript that shows the "
                        "problem, or an empty string when the issue is an absence "
                        "(a character who vanishes, a thread left unresolved).",
                    },
                    "chapters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The chapters this issue spans.",
                    },
                },
                "required": ["category", "severity", "message"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}

_CHUNK_SYSTEM = (
    "You are a meticulous copy editor preparing a manuscript for publication. "
    "You report only defects you can point at in the text you were given.\n\n"
    "Rules:\n"
    "- Quote the exact text you are complaining about. Copy it character for "
    "character, including the surrounding words needed to locate it.\n"
    "- Supply `correction` only when the fix is mechanical and you are certain: "
    "a grammatical error, a punctuation error, a misused word. Leave it out for "
    "anything an author should decide — style, voice, rhythm, dialect.\n"
    "- Never report a stylistic preference as an error.\n"
    "- Never invent text that is not in the manuscript.\n"
    "- If the prose is clean, return an empty findings list. That is a valid and "
    "common answer.\n"
    "- Report at most 25 findings, most serious first."
)

_CONTINUITY_SYSTEM = (
    "You are a developmental editor reading a whole novel for the first time. "
    "You are given a digest: the cast, and the opening and closing of every "
    "chapter. You are looking for problems that span chapters.\n\n"
    "Look for:\n"
    "- A character whose name, age, appearance or established facts change.\n"
    "- A character present in one chapter who vanishes with no explanation.\n"
    "- A plot thread opened and never resolved, or resolved without setup.\n"
    "- Two characters whose names are so similar that a reader would confuse "
    "them, or one character written under two spellings.\n"
    "- Dialogue that does not match who is speaking — a child sounding like an "
    "adult, a character losing a distinctive voice.\n\n"
    "Rules:\n"
    "- You are reading a digest, not the whole book. Say when a suspicion might "
    "be an artefact of that; do not assert what you cannot see.\n"
    "- Quote the manuscript where you can. Leave `quote` empty when the problem "
    "is something missing.\n"
    "- Report at most 20 findings, most serious first."
)


def _manuscript_header(
    title: str | None,
    author: str | None,
    genre: str,
    language: str,
    cast: list[tuple[str, int]],
    ignored: list[str],
) -> str:
    """The shared context block both passes receive."""
    parts = [
        f"Title: {title or 'Untitled'}",
        f"Author: {author or 'unknown'}",
        f"Genre: {genre}",
        f"Language: {language}",
    ]
    if cast:
        listing = ", ".join(f"{name} ({count})" for name, count in cast[:40])
        parts.append(f"Characters, most-mentioned first: {listing}")
    if ignored:
        # A re-run must not re-litigate decisions the author already made.
        listed = "\n".join(f"- {item}" for item in ignored[:50])
        parts.append(
            "The author has already reviewed and dismissed these. Do not report "
            f"them again:\n{listed}"
        )
    return "\n".join(parts)


# --------------------------------------------------------------------------
# The API call
# --------------------------------------------------------------------------


@dataclass
class _Usage:
    """Token totals accumulated across every call in one scan."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, model: str, usage: dict) -> None:
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)
        self.cost_usd += price_usage(model, usage)
        self.calls += 1


def _extract_payload(response: dict) -> dict:
    """Pull the structured JSON out of an API response.

    With `output_config.format` the answer arrives as a text block containing
    JSON. Both that and a block carrying a parsed `json` field are accepted, so
    a change in which one the API returns does not break the scan.
    """
    if response.get("stop_reason") == "refusal":
        raise HealthAIError("The model declined to analyse this manuscript.")

    for block in response.get("content") or []:
        if not isinstance(block, dict):
            continue
        if isinstance(block.get("json"), dict):
            return block["json"]
        if block.get("type") == "text":
            text = (block.get("text") or "").strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise HealthAIError(
                    "The model's response was not valid JSON."
                ) from exc
            if not isinstance(parsed, dict):
                raise HealthAIError("The model's response was not a JSON object.")
            return parsed

    raise HealthAIError("The model returned no usable content.")


def _payload_of(response) -> dict:
    """The structured answer, plus the token usage that sits beside it.

    The findings arrive as JSON inside a content block, while `usage` is a
    sibling of `content` at the top level of the response. The caller needs
    both, so this puts them in one dict. Skipping the extraction here is the
    quiet failure mode: `payload.get("findings")` would return nothing on every
    call, and a scan that finds nothing looks exactly like a clean manuscript.
    """
    raw = response.json()
    if not isinstance(raw, dict):
        raise HealthAIError("The API returned an unexpected response body.")

    payload = _extract_payload(raw)
    usage = raw.get("usage")
    if isinstance(usage, dict):
        # setdefault, so a payload that carries its own usage wins.
        payload.setdefault("usage", usage)
    return payload


async def _post(client, body: dict) -> dict:
    """POST to the Messages API, retrying what is worth retrying.

    A 429 is contention and a 5xx is usually transient, so both back off and
    retry. Everything else in the 4xx range is this request being wrong — a bad
    key, a bad model, a schema the API rejects — and retrying it would only
    spend the same three attempts to reach the same answer.
    """
    headers = {
        "x-api-key": _api_key(),
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    url = f"{_base_url()}/v1/messages"
    last_error = "no attempt was made"

    for attempt in range(ATTEMPTS):
        response = await client.post(url, headers=headers, json=body)
        status = response.status_code

        if status == RATE_LIMITED:
            last_error = "the API is rate limiting this deployment"
            await asyncio.sleep((attempt + 1) * 5)
            continue
        if status in RETRYABLE:
            last_error = f"the API returned {status}"
            await asyncio.sleep((attempt + 1) * 2)
            continue
        if status >= 400:
            raise HealthAIError(
                f"The AI request was rejected ({status}): {_message(response)}"
            )
        return _payload_of(response)

    raise HealthAIUnavailableError(
        f"The AI scan could not complete after {ATTEMPTS} attempts: {last_error}."
    )


def _message(response) -> str:
    """The API's own error text, or the raw body if it is not JSON."""
    try:
        payload = response.json()
    except Exception:
        return response.text[:300]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if payload.get("message"):
            return str(payload["message"])
    return response.text[:300]


def _request_body(model: str, system: str, user: str, schema: dict, max_tokens: int) -> dict:
    body: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }
    thinking = _thinking()
    if thinking is not None:
        body["thinking"] = thinking
    return body


# --------------------------------------------------------------------------
# Finding construction
# --------------------------------------------------------------------------


def _finding_from_chunk(
    chunk: Chunk, item: dict, category_id: str, source_note: str
) -> Finding | None:
    """Build a Finding from one model-reported item, or None if unusable."""
    message = str(item.get("message") or "").strip()
    if not message:
        return None

    quote = str(item.get("quote") or "")
    line_number, exact = _locate(chunk, quote) if quote else (chunk.first_line, False)

    # Only a verbatim hit may carry a mechanical replacement, and only when the
    # model supplied one. Anything else is advice.
    correction = item.get("correction")
    original: str | None = None
    replacement: str | None = None
    if exact and isinstance(correction, str) and correction.strip() and quote.strip():
        original = quote.strip()
        replacement = correction.strip()

    severity = str(item.get("severity") or "medium")
    if severity not in {"high", "medium", "low"}:
        severity = "medium"

    # Advice survives even when the correction cannot be applied. A quote the
    # model reflowed is still a sentence the author should look at, and dropping
    # the wording it proposed would leave them with a complaint and no fix.
    advice = correction.strip() if isinstance(correction, str) else ""
    if not advice:
        advice = str(item.get("suggestion") or "").strip()

    return Finding(
        id=finding_id(category_id, line_number, original, message),
        category=category_id,
        severity=severity,
        message=message,
        chapter=chunk.chapters[0] if chunk.chapters else FRONT_MATTER,
        line_number=line_number,
        context=_context_for(chunk, line_number) or source_note,
        suggestion=replacement or advice or None,
        original=original,
        replacement=replacement,
    )


def _category_for(item: dict, allowed: set[str]) -> str | None:
    """The category id from a model item, if it is one we asked for."""
    value = str(item.get("category") or "").strip().lower()
    return value if value in allowed else None


# --------------------------------------------------------------------------
# The scan
# --------------------------------------------------------------------------


async def run_ai_scan(
    *,
    text: str,
    lines: list[Line],
    chunks: list[Chunk],
    chapters: list[ChapterSummary],
    cast: list[tuple[str, int]],
    language: str,
    genre: str,
    title: str | None = None,
    author: str | None = None,
    ignored: list[str] | None = None,
    on_progress: Callable[[float], Awaitable[None]] | None = None,
) -> AIResult:
    """Run the AI half of the health report.

    Two passes, because the categories have different shapes. Grammar and
    dialogue are local — a sentence is wrong or it is not — so they run per
    chunk on the fast model. Continuity and plot are inherently cross-chapter,
    so they run once over a digest on the stronger model, where the whole-book
    view is the entire point.

    `on_progress` receives a 0-1 fraction as chunks complete, so a long job can
    show movement rather than an indefinite spinner. It is awaited, so a caller
    that writes to the database can keep the poll response honest without
    blocking the scan.
    """
    if not ai_configured():
        return AIResult(
            available=False,
            note=(
                "No ANTHROPIC_API_KEY is set on this deployment, so grammar, "
                "continuity, dialogue and plot were not checked. Everything "
                "else in this report is unaffected."
            ),
        )

    httpx = _require_httpx()
    ignored = ignored or []
    header = _manuscript_header(title, author, genre, language, cast, ignored)
    usage = _Usage()
    findings: list[Finding] = []

    async def report(fraction: float) -> None:
        if on_progress is None:
            return
        try:
            await on_progress(max(0.0, min(1.0, fraction)))
        except Exception:
            # Progress reporting is a courtesy. A database blip while writing
            # "62%" must not destroy a scan the author is paying for.
            pass

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            total = len(chunks) + (1 if chapters else 0)
            done = 0

            for chunk in chunks:
                body = _request_body(
                    chunk_model(),
                    _CHUNK_SYSTEM,
                    f"{header}\n\n---\n\n{chunk.text}",
                    CHUNK_SCHEMA,
                    MAX_OUTPUT_TOKENS,
                )
                payload = await _post(client, body)
                usage.add(chunk_model(), payload.get("usage") or {})
                for item in payload.get("findings") or []:
                    if not isinstance(item, dict):
                        continue
                    category = _category_for(
                        item, {GRAMMAR.id, DIALOGUE.id, PLOT.id}
                    )
                    if category is None:
                        continue
                    finding = _finding_from_chunk(
                        chunk, item, category, f"Chapter: {chunk.chapters[0]}"
                    )
                    if finding is not None:
                        findings.append(finding)
                done += 1
                await report(done / total if total else 1.0)

            digest = build_digest(lines, chapters, cast)
            if digest:
                body = _request_body(
                    continuity_model(),
                    _CONTINUITY_SYSTEM,
                    f"{header}\n\n---\n\n{digest}",
                    CONTINUITY_SCHEMA,
                    CONTINUITY_MAX_OUTPUT_TOKENS,
                )
                payload = await _post(client, body)
                usage.add(continuity_model(), payload.get("usage") or {})
                for item in payload.get("findings") or []:
                    if not isinstance(item, dict):
                        continue
                    category = _category_for(
                        item, {CONTINUITY.id, PLOT.id, DIALOGUE.id}
                    )
                    if category is None:
                        continue
                    finding = _continuity_finding(item, category, lines, chapters)
                    if finding is not None:
                        findings.append(finding)
                done += 1
                await report(done / total if total else 1.0)

    except HealthAIError:
        raise
    except Exception as exc:  # network, DNS, timeout, malformed TLS
        raise HealthAIUnavailableError(
            f"The AI scan could not reach the API: {exc}"
        ) from exc

    return AIResult(
        findings=findings,
        available=True,
        note=None,
        cost_usd=round(usage.cost_usd, 6),
        credits=credits_for(usage.cost_usd),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        calls=usage.calls,
        model=chunk_model(),
    )


def build_digest(
    lines: list[Line], chapters: list[ChapterSummary], cast: list[tuple[str, int]]
) -> str:
    """A whole-book view small enough to send in one call.

    Every chapter's opening and closing, because that is where a chapter
    establishes and resolves things, and continuity faults live in the joins
    between chapters rather than in their middles.
    """
    real = [chapter for chapter in chapters if chapter.title != FRONT_MATTER]
    if not real:
        return ""

    by_number = {line.number: index for index, line in enumerate(lines)}
    parts: list[str] = []

    for position, chapter in enumerate(real):
        start_index = by_number.get(chapter.line_number, 0)
        # A chapter runs to the line before the next heading.
        if position + 1 < len(real):
            end_index = by_number.get(real[position + 1].line_number, len(lines))
        else:
            end_index = len(lines)

        body = [line.text for line in lines[start_index:end_index] if line.text.strip()]
        if not body:
            continue

        opening = "\n".join(body[:12])
        closing = "\n".join(body[-6:])
        if len(body) <= 18:
            parts.append(f"### {chapter.title}\n{opening}")
        else:
            parts.append(f"### {chapter.title}\n{opening}\n[...]\n{closing}")

    if cast:
        names = ", ".join(name for name, _ in cast[:25])
        parts.insert(0, f"Cast: {names}\n")

    return "\n\n".join(parts)


def _continuity_finding(
    item: dict,
    category: str,
    lines: list[Line],
    chapters: list[ChapterSummary],
) -> Finding | None:
    """Build a Finding from a whole-book pass item.

    Continuity findings are rarely a span of text — "this character vanishes
    after chapter nine" has nothing to replace — so they are advisory by
    construction. There is no `original`, and therefore no automatic fix, even
    when the model volunteers a quote.
    """
    message = str(item.get("message") or "").strip()
    if not message:
        return None

    named = [
        str(name)
        for name in (item.get("chapters") or [])
        if isinstance(name, str) and name.strip()
    ]
    chapter = named[0] if named else FRONT_MATTER

    # Point at the chapter's heading so the Book Doctor can jump there.
    line_number = next(
        (c.line_number for c in chapters if c.title == chapter), 1
    )

    quote = str(item.get("quote") or "").strip()
    context = quote[:200] if quote else f"Spans: {', '.join(named) or 'the manuscript'}"

    severity = str(item.get("severity") or "medium")
    if severity not in {"high", "medium", "low"}:
        severity = "medium"

    return Finding(
        id=finding_id(category, line_number, None, message),
        category=category,
        severity=severity,
        message=message,
        chapter=chapter,
        line_number=line_number,
        context=context,
        suggestion=None,
        original=None,
        replacement=None,
    )
