"""Spellchecking: find suspicious words and report where they sit.

The scan is dictionary based (pyspellchecker), so it flags any word the
dictionary does not contain. In fiction that includes character and place
names, which is why every finding carries `likely_proper_noun`: a capitalised
word that is not merely opening a sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from spellchecker import SpellChecker

from services.chapters import FRONT_MATTER, chapter_title, is_chapter_heading

# Ceiling on the reported findings. A misspelled word is reported once per
# occurrence, so a book with a systematic error would otherwise return an
# unbounded response. `TypoReport.count` still holds the true total.
MAX_TYPOS = 200

# How many suggestions to return per word, and how many words of context.
MAX_SUGGESTIONS = 3
CONTEXT_WORDS = 5

# Letters with optional internal apostrophes. Hyphens are treated as separators
# so that ordinary compounds ("well-known") resolve to two known words, and the
# digit lookbehind keeps "3rd" from being read as the word "rd".
_WORD_RE = re.compile(r"(?<!\d)[^\W\d_]+(?:['’][^\W\d_]+)*")

# Context is rebuilt from the line's own whitespace tokens, which preserves the
# author's punctuation and hyphenation in the snippet.
_WHITESPACE_TOKEN_RE = re.compile(r"\S+")

# A capitalised word following one of these is merely starting a sentence, so
# its capital carries no evidence of being a proper noun.
_SENTENCE_START_CHARS = frozenset(".?!。！？…:;\"'“”‘’([{—-–")


@dataclass(frozen=True)
class Typo:
    """One occurrence of a word the dictionary does not contain."""

    word: str
    suggestions: list[str]
    chapter: str
    line_number: int
    context: str
    likely_proper_noun: bool


@dataclass(frozen=True)
class TypoReport:
    """The typo findings for one manuscript."""

    count: int
    typos: list[Typo]
    available: bool
    note: str | None


@lru_cache(maxsize=None)
def _spellchecker_for(language_code: str) -> SpellChecker | None:
    """Return a dictionary for `language_code`, or None if there is not one.

    Building a SpellChecker loads a full word list, so the instance is cached
    and reused across requests rather than rebuilt per call.
    """
    try:
        return SpellChecker(language=language_code)
    except Exception:
        # Raised for a language with no bundled dictionary, and for damaged or
        # missing dictionary data. Either way the scan simply cannot run.
        return None


def scan_typos(text: str, language_code: str) -> TypoReport:
    """Scan `text` for misspellings, locating each one in its chapter."""
    spellchecker = _spellchecker_for(language_code)
    if spellchecker is None:
        return TypoReport(
            count=0,
            typos=[],
            available=False,
            note=f"No spellcheck dictionary is available for '{language_code}'.",
        )

    lines = text.splitlines()

    # Ask the dictionary about each distinct word once for the whole document.
    # `unknown` is a set lookup, but doing it per line would repeat that work
    # for every line in the book.
    vocabulary = {word.lower() for line in lines for word in _WORD_RE.findall(line)}
    unknown = set(spellchecker.unknown(vocabulary)) if vocabulary else set()
    if not unknown:
        return TypoReport(count=0, typos=[], available=True, note=None)

    typos: list[Typo] = []
    total = 0
    chapter = FRONT_MATTER
    suggestions: dict[str, list[str]] = {}

    for line_number, line in enumerate(lines, start=1):
        if is_chapter_heading(line):
            chapter = chapter_title(line) or chapter
        for match in _WORD_RE.finditer(line):
            word = match.group()
            lookup = word.lower()
            if lookup not in unknown:
                continue
            total += 1
            if len(typos) >= MAX_TYPOS:
                # Keep counting so `count` stays accurate; stop building records.
                continue
            if lookup not in suggestions:
                # Computing suggestions is the expensive step, so it is done
                # once per distinct misspelling rather than once per occurrence.
                suggestions[lookup] = _suggestions_for(spellchecker, lookup)
            typos.append(
                Typo(
                    word=word,
                    suggestions=suggestions[lookup],
                    chapter=chapter,
                    line_number=line_number,
                    context=_context(line, match.start()),
                    likely_proper_noun=word[:1].isupper()
                    and not _opens_sentence(line, match.start()),
                )
            )

    note = None
    if total > len(typos):
        note = f"Showing the first {len(typos)} of {total} findings."
    return TypoReport(count=total, typos=typos, available=True, note=note)


def _suggestions_for(spellchecker: SpellChecker, word: str) -> list[str]:
    """The best few corrections for `word`, most frequent first."""
    candidates = spellchecker.candidates(word) or set()
    ranked = sorted(
        candidates,
        key=lambda candidate: _frequency(spellchecker.word_frequency, candidate),
        reverse=True,
    )
    return ranked[:MAX_SUGGESTIONS]


def _frequency(word_frequency: object, candidate: str) -> int:
    """How often `candidate` occurs in the corpus, or 0 when it is unknown.

    `pyspellchecker` exposes `word_frequency` as a `WordFrequency`, which is a
    mapping but not a `dict` and has no `get`, so it is indexed directly and a
    `KeyError` marks a word the dictionary does not contain.
    """
    try:
        return word_frequency[candidate]
    except KeyError:
        return 0


def _context(line: str, start: int) -> str:
    """A snippet of `line` around the word beginning at `start`."""
    tokens = list(_WHITESPACE_TOKEN_RE.finditer(line))
    index = next(
        (i for i, token in enumerate(tokens) if token.start() <= start < token.end()),
        None,
    )
    if index is None:
        return line.strip()
    window = tokens[max(0, index - CONTEXT_WORDS) : index + CONTEXT_WORDS + 1]
    return "..." + " ".join(token.group() for token in window) + "..."


def _opens_sentence(line: str, start: int) -> bool:
    """Whether the word at `start` follows the start of the line or a full stop."""
    for index in range(start - 1, -1, -1):
        char = line[index]
        if char.isspace():
            continue
        return char in _SENTENCE_START_CHARS
    return True
