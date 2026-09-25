"""Chapter counting heuristics, covering the scripts the platform accepts."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# A line is only a chapter heading if it is short. Long lines are prose that
# happens to start with a number or the word "chapter".
MAX_HEADING_CHARS = 80

# Label for the text before the first chapter heading. Shared with the typo
# scanner so a finding and a chapter entry never disagree about where they are.
FRONT_MATTER = "Front Matter"

_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_MARKDOWN_PREFIX_RE = re.compile(r"^#{1,6}\s*")

# "Chapter 3", "Chapitre IV", "Capítulo 2: The Return", "Chapter One", "Глава 1".
# A trailing title is allowed after a delimiter, which keeps prose such as
# "Chapter 3 was long" from counting as a heading.
_LATIN_CHAPTER_KEYWORDS = (
    r"chapter|chapitre|cap[ií]tulo|capitolo|capítol|kapitel|hoofdstuk"
    r"|rozdział|fejezet|chương|глава"
)
_CHAPTER_NUMBER = (
    r"\d{1,4}|[ivxlcdm]{1,9}"
    r"|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
)

_KEYWORD_HEADING_RE = re.compile(
    rf"^(?:{_LATIN_CHAPTER_KEYWORDS})\s*[:.\-–—]?\s*(?:{_CHAPTER_NUMBER})"
    rf"\s*(?:[:.\-–—]\s*\S.{{0,60}})?[.:]?$",
    re.IGNORECASE,
)

# RTL headings number the chapter with a word rather than a digit ("الفصل الأول"),
# so any short trailing text is accepted after the keyword.
_RTL_CHAPTER_KEYWORDS = r"الفصل|فصل|פרק|باب"
_RTL_HEADING_RE = re.compile(rf"^(?:{_RTL_CHAPTER_KEYWORDS})\s+\S.{{0,60}}$")

# CJK markers put the number between the words ("第3章", "第三章", "제1장"), so
# they are matched anywhere in the line rather than anchored to its start.
_CJK_HEADING_RE = re.compile(
    r"第\s*[0-9０-９一二三四五六七八九十百千]+\s*[章回節节篇卷部]"
    r"|제\s*\d{1,4}\s*[장부편]"
)

# A line that is nothing but a number or roman numeral, optionally with a short
# title: "12", "12.", "12. The Return", "XII". Trailing text must be introduced
# by a delimiter, so prose such as "3 apples" is not mistaken for a heading.
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:\d{1,3})(?:\s*[.)、:\-–—]\s*\S.{0,60}|\s*[.)、:\-–—])?$"
)
_ROMAN_HEADING_RE = re.compile(
    r"^([IVXLCDM]{1,9})(?:\s*[.)、:\-–—]\s*\S.{0,60}|\s*[.)、:\-–—])?$"
)

# Used only for manuscripts with no line breaks, where sentence starts stand in
# for line starts.
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?。！？])\s+")

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_TO_ROMAN_TABLE = (
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
    (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
)


@dataclass(frozen=True)
class ChapterSummary:
    """One chapter's share of a manuscript."""

    title: str
    line_number: int
    word_count: int


def count_chapters(text: str) -> int:
    """Count the chapter markers in `text`.

    Each line counts at most once, so a heading matching several rules at once
    ("## Chapter 3") is not double counted.
    """
    count = sum(1 for line in text.splitlines() if is_chapter_heading(line))
    if count == 0:
        count = _count_without_line_breaks(text)
    return count


def chapter_breakdown(
    text: str, count_words: Callable[[str], int]
) -> list[ChapterSummary]:
    """Split `text` into its chapters, with where each starts and how long it is.

    The headings are the same ones `count_chapters` counts, so the length of
    this list matches it. Text before the first heading is reported as one
    "Front Matter" entry, and only when there is text there — a manuscript that
    opens on "Chapter 1" should not gain an empty leading section.

    The chapter's own heading counts towards its word count: it is part of the
    chapter as a reader sees it.

    `count_words` is passed in rather than imported because the CJK-aware
    counter lives in services.analyzer, which imports this module.
    """
    lines = text.splitlines()
    if not any(is_chapter_heading(line) for line in lines):
        # The same fallback count_chapters uses, so the two agree. It only
        # applies to a manuscript that arrived as one unbroken paragraph; the
        # "line numbers" are then sentence positions.
        lines = _SENTENCE_BREAK_RE.split(text)

    # (index of the line the chapter starts on, its title)
    starts = [
        (index, chapter_title(line) or line.strip())
        for index, line in enumerate(lines)
        if is_chapter_heading(line)
    ]

    sections: list[tuple[str, int, str]] = []
    if starts and any(line.strip() for line in lines[: starts[0][0]]):
        sections.append((FRONT_MATTER, 1, "\n".join(lines[: starts[0][0]])))

    for position, (start, title) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        sections.append((title, start + 1, "\n".join(lines[start:end])))

    return [
        ChapterSummary(
            title=title, line_number=line_number, word_count=count_words(body)
        )
        for title, line_number, body in sections
    ]


def is_chapter_heading(raw_line: str) -> bool:
    line = raw_line.strip()
    if not line or len(line) > MAX_HEADING_CHARS:
        return False
    if _MARKDOWN_HEADING_RE.match(line):
        return True
    if _KEYWORD_HEADING_RE.match(line) or _RTL_HEADING_RE.match(line):
        return True
    if _CJK_HEADING_RE.search(line):
        return True
    return _is_numbered_heading(line)


def _is_numbered_heading(line: str) -> bool:
    if _NUMBERED_HEADING_RE.match(line):
        return True
    match = _ROMAN_HEADING_RE.match(line)
    return bool(match) and _roman_to_int(match.group(1)) is not None


def _count_without_line_breaks(text: str) -> int:
    """Count markers in a manuscript that arrived as one unbroken paragraph.

    A .txt file with no newlines at all has no heading lines to find, so
    sentence starts are treated as line starts and the line rules are reused.
    """
    return sum(1 for part in _SENTENCE_BREAK_RE.split(text) if is_chapter_heading(part))


def chapter_title(raw_line: str) -> str:
    """The display text of a chapter heading, without its Markdown marker."""
    return _MARKDOWN_PREFIX_RE.sub("", raw_line.strip()).strip()


def _roman_to_int(token: str) -> int | None:
    """Return the value of a *canonical* roman numeral, or None if it is not one.

    Canonical form is what separates numerals from ordinary words spelled out of
    roman letters: "CIVIL" and "MIX" are not chapter headings.
    """
    total = 0
    previous = 0
    for char in reversed(token):
        value = _ROMAN_VALUES[char]
        total = total - value if value < previous else total + value
        previous = max(previous, value)
    if not 0 < total < 4000 or _to_roman(total) != token:
        return None
    return total


def _to_roman(value: int) -> str:
    parts: list[str] = []
    for amount, numeral in _TO_ROMAN_TABLE:
        while value >= amount:
            parts.append(numeral)
            value -= amount
    return "".join(parts)
