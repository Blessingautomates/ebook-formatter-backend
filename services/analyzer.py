"""Turns an uploaded manuscript into the analysis the API returns."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from services.chapters import count_chapters
from services.detection import detect_language
from services.extractors import extract_text
from services.typos import Typo, scan_typos

# A typeset trade-paperback page holds roughly this many words.
WORDS_PER_PAGE = 250

# Pricing: a flat fee per book, plus a per-1000-words rate and a per-chapter rate.
BASE_TOKEN_COST = 10
TOKENS_PER_1000_WORDS = 1
TOKENS_PER_CHAPTER = 2

# Han, kana, and hangul ranges: scripts that do not separate words with spaces.
_CJK_CHARACTER_RE = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]"
)


@dataclass(frozen=True)
class AnalysisResult:
    """Everything the API reports about one manuscript."""

    word_count: int
    chapter_count: int
    detected_language: str
    language_name: str
    text_direction: str
    script_type: str
    estimated_pages: int
    token_cost: int
    typo_count: int
    typos: list[Typo]
    typo_check_available: bool
    typo_check_note: str | None


def analyze_manuscript(filename: str, data: bytes) -> AnalysisResult:
    """Extract `data` and measure it.

    Raises ManuscriptError if the file is not a supported, readable manuscript.
    """
    text = extract_text(filename, data)
    word_count = count_words(text)
    chapter_count = count_chapters(text)
    language = detect_language(text)
    typo_report = scan_typos(text, language.code)

    return AnalysisResult(
        word_count=word_count,
        chapter_count=chapter_count,
        detected_language=language.code,
        language_name=language.name,
        text_direction=language.text_direction,
        script_type=language.script_type,
        estimated_pages=estimate_pages(word_count),
        token_cost=calculate_token_cost(word_count, chapter_count),
        typo_count=typo_report.count,
        typos=typo_report.typos,
        typo_check_available=typo_report.available,
        typo_check_note=typo_report.note,
    )


def count_words(text: str) -> int:
    """Count words in `text`, treating each CJK character as one word.

    CJK scripts do not put spaces between words, so a plain whitespace split
    would report a whole Chinese novel as a few hundred "words" and price it at
    the base fee. Text in space-delimited scripts is counted with a plain split.
    """
    cjk_character_count = len(_CJK_CHARACTER_RE.findall(text))
    if not cjk_character_count:
        return len(text.split())
    remainder = _CJK_CHARACTER_RE.sub(" ", text)
    return cjk_character_count + len(remainder.split())


def estimate_pages(word_count: int) -> int:
    return math.ceil(word_count / WORDS_PER_PAGE)


def calculate_token_cost(word_count: int, chapter_count: int) -> int:
    """10 (base fee) + (word_count // 1000) + (chapter_count * 2)."""
    return (
        BASE_TOKEN_COST
        + (word_count // 1000) * TOKENS_PER_1000_WORDS
        + chapter_count * TOKENS_PER_CHAPTER
    )
