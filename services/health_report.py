"""Assembles the manuscript health report from its three sources.

The report is a merge of things that know different things:

* `services.analyzer` — how long the book is, what language it is in.
* `services.health` — measurements, and the defects a rule can see.
* `services.typos` — misspellings, from the platform's existing dictionary scan.
* `services.health_ai` — grammar, continuity, dialogue and plot.

The merge is where the interesting decisions live. A finding can be reported by
more than one source (an AI grammar pass and the doubled-word rule both see
"the the"), and reporting it twice would make the Book Doctor ask the author the
same question twice. So findings are deduplicated on the span they cover, with
the deterministic source winning — it is exact, it is free, and it does not
sometimes change its mind between runs.

Every category appears in the output whether or not it found anything, because
the dashboard renders a fixed breakdown and "0 spelling problems" is
information. A category that could not run is distinguished from one that ran
and found nothing, by `available` and `note` — the convention
`typo_check_available` already established.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from services.chapters import ChapterSummary
from services.health import (
    AI_CATEGORIES,
    CATEGORIES,
    Finding,
    Metrics,
    build_cast,
    index_lines,
    measure,
    run_rule_detectors,
)
from services.health_ai import AIResult
from services.typos import TypoReport, scan_typos

#: Severity weights, used only by the health score. A high-severity finding
#: costs as much as three medium ones, and a low as a third.
SEVERITY_WEIGHT = {"high": 3.0, "medium": 1.0, "low": 0.3}

#: Defects per 1000 words that a manuscript is allowed before the score starts
#: to fall. Nobody publishes a book with zero findings; the score measures how
#: far past "normal" a manuscript is, not whether it is perfect.
SCORE_TOLERANCE = 1.0

#: How much of the score one defect per 1000 words above the tolerance costs.
SCORE_PER_DENSITY_POINT = 10.0


@dataclass(frozen=True)
class CategorySummary:
    """One category of the report: what it is, and what it found."""

    id: str
    label: str
    group: str
    source: str  # "rules" | "ai"
    fixable: bool
    count: int
    available: bool
    note: str | None
    findings: list[Finding]


@dataclass(frozen=True)
class HealthReport:
    """Everything the health dashboard and the Book Doctor need."""

    metrics: Metrics
    categories: list[CategorySummary]
    total_findings: int
    fixable_findings: int
    health_score: int
    ai_available: bool
    ai_note: str | None
    ai_model: str | None
    ai_calls: int
    ai_cost_usd: float
    ai_credits: int
    scanned_at: str


def _typo_findings(report: TypoReport) -> list[Finding]:
    """Fold the spell scan into the unified finding list.

    A likely proper noun is reported but never given a replacement, so the
    Book Doctor's Fix All cannot rewrite a character's name into a dictionary
    word. That is the same distinction the existing typo drawer draws with its
    "possible names" filter; here it is enforced at the data level instead of
    relying on the interface to be careful.
    """
    findings: list[Finding] = []
    for typo in report.typos:
        suggestions = list(typo.suggestions or [])
        advice = (
            f"Did you mean {', '.join(suggestions)}?"
            if suggestions
            else "No suggestion — check this word."
        )
        if typo.likely_proper_noun:
            advice = "A name or proper noun the dictionary does not know."

        findings.append(
            Finding(
                id=f"typo-{typo.line_number}-{typo.word.lower()}",
                category="spelling",
                severity="low" if typo.likely_proper_noun else "medium",
                message=f"“{typo.word}” is not in the dictionary.",
                chapter=typo.chapter,
                line_number=typo.line_number,
                context=typo.context,
                suggestion=advice,
                # Left None for a probable name: the machine does not get to
                # decide a character is misspelled.
                original=None if typo.likely_proper_noun else typo.word,
                replacement=(suggestions[0] if suggestions else None)
                if not typo.likely_proper_noun
                else None,
            )
        )
    return findings


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """Drop findings that describe the same span of text as another.

    Two findings collide when they cover the same characters on the same line.
    The deterministic one wins: it is exact, it costs nothing, and it gives the
    same answer every run, which is what the diff viewer needs. Findings with no
    span (continuity, plot, structure) collide on their message instead.
    """
    best: dict[tuple, Finding] = {}
    for finding in findings:
        if finding.original:
            key = ("span", finding.line_number, finding.original.strip().lower())
        else:
            key = (
                "message",
                finding.category,
                finding.line_number,
                finding.message.strip().lower()[:80],
            )

        existing = best.get(key)
        if existing is None:
            best[key] = finding
            continue
        existing_is_ai = CATEGORIES[existing.category].source == "ai"
        new_is_rule = CATEGORIES[finding.category].source == "rules"
        if existing_is_ai and new_is_rule:
            best[key] = finding

    return list(best.values())


def health_score(findings: list[Finding], word_count: int) -> int:
    """A 0-100 summary of how much work the manuscript needs.

    Defect density, not defect count: a 200,000-word novel with thirty findings
    is in better shape than a 5,000-word short story with thirty, and a raw
    count would say the opposite. Weighted by severity, with a tolerance band so
    that ordinary imperfection does not move the number.
    """
    if not findings:
        return 100
    weight = sum(SEVERITY_WEIGHT.get(f.severity, 1.0) for f in findings)
    per_1000 = weight / max(1.0, word_count / 1000)
    penalty = max(0.0, per_1000 - SCORE_TOLERANCE) * SCORE_PER_DENSITY_POINT
    return int(max(0, min(100, round(100 - penalty))))


def _summarise(
    categories: dict[str, list[Finding]],
    availability: dict[str, tuple[bool, str | None]],
) -> list[CategorySummary]:
    """Build a summary for every known category, in registry order."""
    summaries: list[CategorySummary] = []
    for category_id, info in CATEGORIES.items():
        available, note = availability.get(category_id, (True, None))
        summaries.append(
            CategorySummary(
                id=info.id,
                label=info.label,
                group=info.group,
                source=info.source,
                fixable=info.fixable,
                count=len(categories.get(category_id, [])),
                available=available,
                note=note,
                findings=categories.get(category_id, []),
            )
        )
    return summaries


def _ai_availability(ai: AIResult) -> dict[str, tuple[bool, str | None]]:
    """Mark the AI categories unavailable when the pass did not run.

    Distinguishing "checked and clean" from "never checked" is the whole point.
    A report that showed zero grammar problems on a deployment with no API key
    would be actively misleading.
    """
    if ai.available:
        return {}
    note = ai.note or "The AI scan did not run."
    return {category: (False, note) for category in AI_CATEGORIES}


def build_report(
    *,
    text: str,
    chapters: list[ChapterSummary],
    language: str,
    ai: AIResult,
    title: str | None = None,
    author: str | None = None,
) -> HealthReport:
    """Run every source and merge them into one report.

    Deterministic and synchronous apart from `ai`, which the caller has already
    awaited. Keeping it that way means the whole report except the AI pass can
    be built and tested without a network.
    """
    lines = index_lines(text)
    typo_report = scan_typos(text, language)
    rule_results = run_rule_detectors(text, lines, chapters)

    metrics = measure(text, chapters, language, ai_enabled=ai.available)

    collected: list[Finding] = []
    collected.extend(_typo_findings(typo_report))
    for result in rule_results:
        collected.extend(result.findings)
    collected.extend(ai.findings)

    findings = _dedupe(collected)

    grouped: dict[str, list[Finding]] = {category: [] for category in CATEGORIES}
    for finding in findings:
        grouped.setdefault(finding.category, []).append(finding)
    for bucket in grouped.values():
        bucket.sort(key=lambda f: (f.line_number, f.category))

    availability: dict[str, tuple[bool, str | None]] = {
        "spelling": (typo_report.available, typo_report.note)
    }
    for result in rule_results:
        availability[result.category] = (result.available, result.note)
    availability.update(_ai_availability(ai))

    categories = _summarise(grouped, availability)

    return HealthReport(
        metrics=metrics,
        categories=categories,
        total_findings=len(findings),
        fixable_findings=len([f for f in findings if f.fixable]),
        health_score=health_score(findings, metrics.word_count),
        ai_available=ai.available,
        ai_note=ai.note,
        ai_model=ai.model,
        ai_calls=ai.calls,
        ai_cost_usd=ai.cost_usd,
        ai_credits=ai.credits,
        scanned_at=datetime.now(timezone.utc).isoformat(),
    )


def cast_for_prompt(text: str) -> list[tuple[str, int]]:
    """The cast list the continuity prompt is built from.

    Exposed separately because the AI pass needs it *before* the report exists
    — the whole point is that the character-name detector feeds the prompt.
    """
    return build_cast(index_lines(text))
