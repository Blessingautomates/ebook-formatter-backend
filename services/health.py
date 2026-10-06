"""Manuscript health: the measurements, and the detectors that need no AI.

This module is the half of the health report that is deterministic, instant and
free. It answers two questions:

* **How big is this book, and how hard is it to read?** Word and character
  counts, sentences, paragraphs, estimated print pages, eBook reading time and
  the Flesch-Kincaid readability scores.
* **What is mechanically wrong with it?** Doubled words, mixed quotation marks,
  heading-level jumps, character names spelled two ways, chapters that are wild
  outliers in length, and the formatting debris a manuscript picks up on its way
  out of a word processor.

The other half — grammar, continuity, dialogue and plot, which need a reader's
judgement rather than a regular expression — lives in services.health_ai.

Two conventions this module keeps from the rest of the platform:

* Page estimates go through `services.analyzer.WORDS_PER_PAGE` and word counts
  through `services.analyzer.count_words`, so a health report can never disagree
  with the pre-scan or the export about how long a book is.
* A detector that cannot run says so (`available=False` and a note) rather than
  returning an empty list, which would read as "nothing wrong here". The spell
  scan already sets this precedent with `typo_check_available`.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from statistics import median

from services.analyzer import count_words, estimate_pages
from services.chapters import (
    FRONT_MATTER,
    ChapterSummary,
    chapter_title,
    is_chapter_heading,
)

# Average adult reading speed for prose, in words per minute. Used for the
# eBook reading-time estimate; print pages use the analyzer's own rate.
WORDS_PER_MINUTE = 220

# A chapter is flagged as a pacing outlier when it falls outside this multiple
# of the manuscript's median chapter length. Deliberately loose: chapters are
# allowed to differ, and a detector that fires on ordinary variation is one an
# author learns to ignore.
PACING_LOW_RATIO = 0.4
PACING_HIGH_RATIO = 2.5

# A capitalised word has to appear at least this often, mid-sentence, before it
# is treated as a character rather than a one-off proper noun.
MIN_NAME_OCCURRENCES = 2

# Genre guessing. Poetry needs enough short lines to be a poem rather than a
# title page; the dialogue threshold is the share of lines carrying speech.
MIN_POETRY_LINES = 20
POETRY_WORDS_PER_LINE = 8
DIALOGUE_RATIO = 0.25

# Levenshtein distance at or below which two character names are treated as
# variants of each other rather than two different people. Two edits catches
# "Maribel"/"Maribelle" and "Jon"/"John" without merging "Anna" and "Anne"
# into "Hannah" and similar.
NAME_VARIANT_DISTANCE = 2

# Words that are capitalised mid-sentence for reasons other than being a name.
# Without this the cast list fills up with "I", months and honorifics, and the
# continuity prompt built from it is worse than useless.
_NOT_NAMES = frozenset(
    {
        "i", "i'm", "i'll", "i've", "i'd",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday", "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "god", "christ", "lord", "sir", "madam", "mr", "mrs", "ms", "dr",
        "english", "french", "spanish", "german", "american", "british",
    }
)

# The categories the health report can report against, and what a caller may do
# about each. `fixable` is what the Book Doctor's "Fix All" is allowed to touch:
# a spelling or punctuation correction is mechanical, whereas a continuity
# problem is a judgement call an author has to make.
CATEGORIES: dict[str, "CategoryInfo"] = {}


@dataclass(frozen=True)
class CategoryInfo:
    """One kind of finding, and how it should be presented and acted on."""

    id: str
    label: str
    #: The Book Doctor group this category is filed under.
    group: str
    #: "rules" when services.health computes it, "ai" when services.health_ai does.
    source: str
    fixable: bool


def _register(
    id: str, label: str, group: str, source: str, fixable: bool
) -> CategoryInfo:
    info = CategoryInfo(id, label, group, source, fixable)
    CATEGORIES[id] = info
    return info


# Spelling is computed by services.typos and folded in by services.health_report,
# so it is registered here for its metadata rather than detected here.
SPELLING = _register("spelling", "Spelling", "Writing health", "rules", True)
GRAMMAR = _register("grammar", "Grammar", "Writing health", "ai", True)
PUNCTUATION = _register("punctuation", "Punctuation", "Writing health", "rules", True)
REPEATED_WORDS = _register(
    "repeated_words", "Repeated words", "Writing health", "rules", True
)
CONTINUITY = _register("continuity", "Continuity", "Consistency", "ai", False)
CHARACTER_NAMES = _register(
    "character_names", "Character names", "Consistency", "rules", False
)
DIALOGUE = _register("dialogue", "Dialogue", "Dialogue", "ai", False)
FORMATTING = _register("formatting", "Formatting", "Typography & structure", "rules", True)
STRUCTURE = _register(
    "structure", "Structure", "Typography & structure", "rules", False
)
PACING = _register("pacing", "Pacing", "Pacing", "rules", False)
PLOT = _register("plot", "Plot", "Pacing", "ai", False)

#: The categories the AI pass is responsible for. Anything not in here is
#: computed by this module, and a report with no API key still has content.
AI_CATEGORIES: tuple[str, ...] = (GRAMMAR.id, CONTINUITY.id, DIALOGUE.id, PLOT.id)

#: The categories this module computes.
RULE_CATEGORIES: tuple[str, ...] = (
    SPELLING.id,
    PUNCTUATION.id,
    REPEATED_WORDS.id,
    CHARACTER_NAMES.id,
    FORMATTING.id,
    STRUCTURE.id,
    PACING.id,
)


@dataclass(frozen=True)
class Finding:
    """One thing worth an author's attention, and where it is.

    `line_number` indexes the *extracted* text, which is what `services.typos`
    already reports and what the corrected-text pipeline in lib/typos.ts expects,
    so a health finding and a spelling finding can be applied by the same code.

    `original` and `replacement` are set only when the correction is mechanical
    — a doubled word, a straight quote in a curly-quoted document. When they are
    None the finding is advisory and no fix may be applied automatically.
    """

    id: str
    category: str
    severity: str  # "high" | "medium" | "low"
    message: str
    chapter: str
    line_number: int
    context: str
    suggestion: str | None = None
    original: str | None = None
    replacement: str | None = None

    @property
    def fixable(self) -> bool:
        return self.original is not None and self.replacement is not None


@dataclass(frozen=True)
class ReadingLevel:
    """Readability scores. Both are computed from the same three counts."""

    flesch_reading_ease: float
    flesch_kincaid_grade: float
    label: str


@dataclass(frozen=True)
class Metrics:
    """The numbers the health report leads with."""

    word_count: int
    character_count: int
    character_count_no_spaces: int
    sentence_count: int
    paragraph_count: int
    chapter_count: int
    print_pages: int
    reading_time_minutes: int
    reading_level: ReadingLevel | None
    reading_level_note: str | None
    detected_genre: str
    genre_confidence: float
    genre_source: str
    completeness: int
    completeness_notes: list[str]
    estimated_processing_seconds: int


@dataclass(frozen=True)
class DetectorResult:
    """What one detector found, and whether it was able to run at all."""

    category: str
    findings: list[Finding]
    available: bool = True
    note: str | None = None


@dataclass(frozen=True)
class Line:
    """One line of the extracted manuscript, and the chapter it belongs to."""

    number: int
    text: str
    chapter: str


def index_lines(text: str) -> list[Line]:
    """Number every line and tag it with its chapter.

    One pass, shared by every detector, so they cannot disagree about which
    chapter a line is in. The chapter is tracked exactly as services.typos
    tracks it: a line is a heading or it is not, and anything before the first
    heading is front matter.
    """
    lines: list[Line] = []
    chapter = FRONT_MATTER
    for number, raw in enumerate(text.splitlines(), start=1):
        if is_chapter_heading(raw):
            chapter = chapter_title(raw) or chapter
        lines.append(Line(number=number, text=raw, chapter=chapter))
    return lines


def finding_id(category: str, line_number: int, original: str | None, message: str) -> str:
    """A stable identifier for a finding.

    Stable across re-runs of the same manuscript, which is what lets a Book
    Doctor decision (fixed, or ignored) survive a rescan instead of being asked
    again. It is *not* stable across edits that move the line, and it does not
    need to be: an edit invalidates the question.
    """
    digest = hashlib.sha1(
        f"{category}|{line_number}|{original or ''}|{message}".encode("utf-8")
    )
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------

_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")
_LETTERS_ONLY_RE = re.compile(r"[^a-z]")
_SENTENCE_END_RE = re.compile(r"[.!?…。！？]+[\"'”’)\]]*(?=\s|$)")
_WHITESPACE_RE = re.compile(r"\S+")


def syllable_count(word: str) -> int:
    """Approximate the syllables in `word`.

    The standard heuristic: count vowel groups, then take one back for a silent
    trailing "e". It is wrong on some words ("business" reads as three), which
    is acceptable — it is used only inside a readability average over thousands
    of words, where individual errors wash out.
    """
    letters = _LETTERS_ONLY_RE.sub("", word.lower())
    if not letters:
        return 0

    groups = len(_VOWEL_GROUP_RE.findall(letters))
    # "le" endings are syllabic ("table", "little"), and so are "ee"/"ye", so a
    # trailing "e" is only silent in the other cases.
    if (
        letters.endswith("e")
        and not letters.endswith(("le", "ee", "ye"))
        and groups > 1
    ):
        groups -= 1
    return max(1, groups)


def count_sentences(text: str) -> int:
    """Count sentences in `text`.

    A run of terminators counts once, so "What?!" is one sentence. This
    over-counts abbreviations ("Mr. Smith" reads as two) — the same class of
    error as the syllable heuristic, and it matters as little.
    """
    return len(_SENTENCE_END_RE.findall(text)) or (1 if text.strip() else 0)


def count_paragraphs(text: str) -> int:
    """Count non-empty blocks separated by blank lines."""
    blocks = re.split(r"\n\s*\n", text)
    return sum(1 for block in blocks if block.strip())


def reading_time_minutes(word_count: int) -> int:
    """Estimated minutes to read `word_count` words, rounded up to at least one."""
    return max(1, round(word_count / WORDS_PER_MINUTE))


def flesch_reading_ease(words: int, sentences: int, syllables: int) -> float:
    """Flesch Reading Ease: higher is easier. Roughly 60-70 is plain English."""
    if words == 0 or sentences == 0:
        return 0.0
    return round(
        206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words), 1
    )


def flesch_kincaid_grade(words: int, sentences: int, syllables: int) -> float:
    """Flesch-Kincaid Grade Level: the US school year the text reads at."""
    if words == 0 or sentences == 0:
        return 0.0
    return round(
        0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59, 1
    )


def reading_level_label(grade: float) -> str:
    """Plain-language name for a Flesch-Kincaid grade."""
    if grade <= 5:
        return "Very easy — around age 10"
    if grade <= 8:
        return "Easy — around age 13"
    if grade <= 12:
        return "Plain English — around age 17"
    if grade <= 16:
        return "Fairly difficult — undergraduate"
    return "Difficult — postgraduate"


def _reading_level(
    text: str, language: str, words: int, sentences: int
) -> tuple[ReadingLevel | None, str | None]:
    """The readability scores, or None and the reason they cannot be given.

    Flesch-Kincaid is a formula over English syllables. Pointing it at Spanish
    or Chinese would not fail — it would return a number, and the number would
    be meaningless. So it declines instead, in the same spirit as
    `typo_check_available`.
    """
    if not language.lower().startswith("en"):
        return (
            None,
            f"The Flesch-Kincaid readability scores are defined for English, "
            f"and this manuscript is {language or 'an unknown language'}. "
            f"No reading level is reported rather than a misleading one.",
        )
    syllables = sum(syllable_count(word) for word in _WHITESPACE_RE.findall(text))
    if syllables == 0 or sentences == 0:
        return None, "There is not enough prose here to score a reading level."
    grade = flesch_kincaid_grade(words, sentences, syllables)
    return (
        ReadingLevel(
            flesch_reading_ease=flesch_reading_ease(words, sentences, syllables),
            flesch_kincaid_grade=grade,
            label=reading_level_label(grade),
        ),
        None,
    )


# Genre keywords, scored against the manuscript. This is a guess offered as a
# default the author can override, not a classification — which is why the
# report carries `genre_confidence` and `genre_source` beside it.
_GENRE_SIGNALS: dict[str, tuple[str, ...]] = {
    "poetry": ("stanza", "verse", "canto"),
    "academic": (
        "abstract", "et al", "doi:", "references", "bibliography",
        "hypothesis", "methodology", "peer review", "issn",
    ),
    "technical": (
        "```", "function", "install", "configuration", "api", "repository",
        "command line", "syntax", "variable",
    ),
    "children": ("once upon a time", "mummy", "daddy", "kitten", "puppy"),
    "journal": ("dear diary", "day 1", "today i"),
    "comic": ("panel", "sfx", "pencils", "inks"),
}

_DIALOGUE_OPEN_RE = re.compile(r"[\"“”]")

#: Offered when nothing scores. "fiction" is the platform's own default genre
#: (schema.sql gives the column that default), so the guess agrees with it.
DEFAULT_GENRE = "fiction"


def guess_genre(text: str, chapters: list[ChapterSummary]) -> tuple[str, float, str]:
    """Guess the manuscript's genre, with a confidence and where it came from.

    Signals are cheap surface features — keyword density, how much of the text
    is dialogue, how long the lines are. The AI pass refines this when it is
    configured; on its own this is a reasonable default that an author overrides
    in the genre selector, not a verdict.
    """
    lowered = text.lower()
    word_count = count_words(text) or 1
    lines = [line for line in text.splitlines() if line.strip()]

    scores: dict[str, float] = {}
    for genre, keywords in _GENRE_SIGNALS.items():
        hits = sum(lowered.count(keyword) for keyword in keywords)
        if hits:
            # Per 10k words, so a long technical book does not simply out-score
            # a short academic one by being longer.
            scores[genre] = (hits / word_count) * 10_000

    # How much of the manuscript is speech. Only double quotes count: treating
    # every apostrophe as dialogue would score a contraction as a line of
    # speech, and nearly every line of English prose has one.
    dialogue_lines = sum(1 for line in lines if _DIALOGUE_OPEN_RE.search(line))
    dialogue_ratio = dialogue_lines / len(lines) if lines else 0.0

    # Poetry: many short lines, and no dialogue. Both gates matter. Without the
    # line count a title page reads as a poem; without the dialogue gate a
    # novel's clipped dialogue does, since terse exchanges are short lines too.
    if len(lines) >= MIN_POETRY_LINES and dialogue_ratio < 0.05:
        words_per_line = word_count / len(lines)
        if words_per_line < POETRY_WORDS_PER_LINE:
            scores["poetry"] = (POETRY_WORDS_PER_LINE - words_per_line) * 3

    # Dialogue-heavy prose is fiction; a very low dialogue ratio alongside real
    # chapters is more likely non-fiction.
    if dialogue_ratio > DIALOGUE_RATIO:
        scores["fiction"] = scores.get("fiction", 0.0) + dialogue_ratio * 10
    elif chapters and dialogue_ratio < 0.05:
        scores["non-fiction"] = scores.get("non-fiction", 0.0) + 4

    if not scores:
        return DEFAULT_GENRE, 0.0, "default"

    genre = max(scores, key=lambda key: scores[key])
    best = scores[genre]
    runner_up = max((value for key, value in scores.items() if key != genre), default=0.0)
    # Confidence from how far clear of the next-best signal the winner is. A
    # tie is a coin toss and is reported as one.
    confidence = 0.0 if best <= 0 else round(min(1.0, (best - runner_up) / best), 2)
    return genre, confidence, "heuristic"


def completeness_score(
    text: str, chapters: list[ChapterSummary], word_count: int
) -> tuple[int, list[str]]:
    """How finished the manuscript looks, from its structure alone.

    Deliberately structural. This cannot know whether the plot resolves — that
    is the AI pass's `plot` category, and ultimately the author's call. What it
    can see is whether the pieces a finished book has are present, and it
    reports the ones that are not so the number is actionable rather than
    mysterious.

    Not to be confused with lib/readiness.ts's `readinessFor`, which scores how
    far along the *publishing workflow* a saved book is (title set, cover
    validated, sign-off signed). They measure different things and must never be
    shown as one number.
    """
    notes: list[str] = []
    earned = 0
    total = 0

    def check(weight: int, ok: bool, missing: str) -> None:
        nonlocal earned, total
        total += weight
        if ok:
            earned += weight
        else:
            notes.append(missing)

    check(30, word_count >= 1000, "The manuscript is very short.")
    check(25, len(chapters) >= 3, "Fewer than three chapters were detected.")
    check(
        20,
        bool(chapters) and all(chapter.word_count >= 250 for chapter in chapters),
        "At least one chapter is under 250 words, which usually means a "
        "heading was misplaced or a chapter was left unfinished.",
    )

    # A book that stops mid-sentence is not finished. The last non-empty line
    # is the only evidence of an ending available without reading it.
    tail = next(
        (line.strip() for line in reversed(text.splitlines()) if line.strip()), ""
    )
    check(
        15,
        bool(tail) and tail[-1:] in ".!?…\"'”’)",
        "The manuscript does not end on a finished sentence.",
    )

    if chapters:
        lengths = [chapter.word_count for chapter in chapters]
        middle = median(lengths)
        check(
            10,
            middle > 0
            and all(length >= middle * PACING_LOW_RATIO for length in lengths),
            "At least one chapter is far shorter than the rest.",
        )
    else:
        check(10, False, "No chapters were detected, so pacing cannot be assessed.")

    return (round((earned / total) * 100) if total else 0), notes


# How long a health run takes. The rules pass is fast and linear; the AI pass
# dominates, and it is bounded by how many chunks the book splits into.
BASE_SECONDS = 2
AI_CHUNK_WORDS = 4_000
AI_SECONDS_PER_CHUNK = 7


def estimate_processing_seconds(word_count: int, ai_enabled: bool) -> int:
    """Estimated wall-clock seconds for a health run on this manuscript."""
    if not ai_enabled:
        return BASE_SECONDS + max(1, word_count // 20_000)
    chunks = max(1, -(-word_count // AI_CHUNK_WORDS))  # ceiling division
    return BASE_SECONDS + chunks * AI_SECONDS_PER_CHUNK


def measure(
    text: str,
    chapters: list[ChapterSummary],
    language: str,
    ai_enabled: bool,
) -> Metrics:
    """Compute every measurement for one manuscript."""
    words = count_words(text)
    sentences = count_sentences(text)
    level, note = _reading_level(text, language, words, sentences)
    genre, confidence, source = guess_genre(text, chapters)
    completeness, missing = completeness_score(text, chapters, words)

    return Metrics(
        word_count=words,
        character_count=len(text),
        character_count_no_spaces=sum(1 for char in text if not char.isspace()),
        sentence_count=sentences,
        paragraph_count=count_paragraphs(text),
        chapter_count=len([c for c in chapters if c.title != FRONT_MATTER]),
        print_pages=estimate_pages(words),
        reading_time_minutes=reading_time_minutes(words),
        reading_level=level,
        reading_level_note=note,
        detected_genre=genre,
        genre_confidence=confidence,
        genre_source=source,
        completeness=completeness,
        completeness_notes=missing,
        estimated_processing_seconds=estimate_processing_seconds(words, ai_enabled),
    )


# --------------------------------------------------------------------------
# Rule-based detectors
# --------------------------------------------------------------------------

# A word repeated immediately: "the the". Case-insensitive, and the second one
# must be a whole word so "had had" is caught but "that thatcher" is not.
_DOUBLED_WORD_RE = re.compile(r"\b(\w+)(\s+)(\1)\b", re.IGNORECASE)

# Crutch words. Frequent in a first draft and never wrong individually, so they
# are reported as a density note rather than line by line.
_CRUTCH_WORDS = ("very", "really", "just", "quite", "suddenly", "somehow")

_STRAIGHT_QUOTES = ('"', "'")
_CURLY_QUOTES = ("“", "”", "‘", "’")

#: A single quote between two letters is an apostrophe ("don't"), not a
#: quotation mark. Almost every English sentence has one, so counting them as
#: quotation marks would report any curly-quoted book with a contraction as
#: "mixing styles" — a false positive on nearly every manuscript. The same goes
#: for the curly apostrophe, which is what most word processors insert.
_APOSTROPHE_RE = re.compile(r"(?<=[^\W\d_])['‘’](?=[^\W\d_])")

_DOUBLED_PUNCTUATION_RE = re.compile(r"([!?])\1+")
_MISSING_SPACE_RE = re.compile(r"[,;:](?=[A-Za-z])")
_SPACED_ELLIPSIS_RE = re.compile(r"\.\s+\.\s+\.")

_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
_TAB_RE = re.compile(r"\t")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$")
_SCENE_BREAK_RE = re.compile(r"^\s*(?:\*\s*\*\s*\*|—{3,}|-{3,}|#{3,}\s*$)\s*$")


def _context(line: str, start: int, end: int, window: int = 5) -> str:
    """A snippet of `line` around the span [start, end)."""
    tokens = list(_WHITESPACE_RE.finditer(line))
    index = next(
        (i for i, token in enumerate(tokens) if token.start() <= start < token.end()),
        None,
    )
    if index is None:
        return line.strip()
    low = max(0, index - window)
    high = min(len(tokens), index + window + 1)
    # Bracket the span's own end too, so a finding near the line's end is not
    # truncated to the point of being unrecognisable.
    high = max(high, next(
        (i + 1 for i, token in enumerate(tokens) if token.start() < end <= token.end()),
        high,
    ))
    return "..." + " ".join(token.group() for token in tokens[low:high]) + "..."


def _starts_sentence(line: str, start: int) -> bool:
    """Whether the word at `start` follows the line start or a full stop.

    Mirrors the helper in services.typos: a capital there is grammar, not
    evidence of a proper noun.
    """
    for index in range(start - 1, -1, -1):
        char = line[index]
        if char.isspace():
            continue
        return char in ".?!。！？…:;\"'“”‘’([{—-–"
    return True


def detect_repeated_words(lines: list[Line]) -> DetectorResult:
    """Find words doubled back to back, and note crutch-word density."""
    findings: list[Finding] = []
    crutch_total = 0
    word_total = 0

    for line in lines:
        word_total += len(_WHITESPACE_RE.findall(line.text))
        for match in _DOUBLED_WORD_RE.finditer(line.text):
            word = match.group(1)
            # "had had" and "that that" are legitimate English. Only the
            # clearly accidental cases are offered as mechanical fixes.
            if word.lower() in {"had", "that", "is", "was"}:
                continue
            findings.append(
                Finding(
                    id=finding_id(
                        REPEATED_WORDS.id, line.number, match.group(0),
                        f"Doubled word “{word}”",
                    ),
                    category=REPEATED_WORDS.id,
                    severity="medium",
                    message=f"“{word}” is repeated.",
                    chapter=line.chapter,
                    line_number=line.number,
                    context=_context(line.text, match.start(), match.end()),
                    suggestion=f"Delete the second “{word}”.",
                    original=match.group(0),
                    replacement=word,
                )
            )
        lowered = line.text.lower()
        crutch_total += sum(lowered.count(f" {word} ") for word in _CRUTCH_WORDS)

    note = None
    if word_total and crutch_total / word_total > 0.01:
        note = (
            f"Crutch words (very, really, just, …) make up "
            f"{crutch_total / word_total:.1%} of this manuscript."
        )
    return DetectorResult(REPEATED_WORDS.id, findings, True, note)


def detect_punctuation(lines: list[Line], text: str) -> DetectorResult:
    """Find punctuation that is inconsistent or doubled."""
    findings: list[Finding] = []

    # Quotation style is judged across the *whole* document, not per line: the
    # fault is mixing the two, and a single straight quote in a curly-quoted
    # book is the finding, not the line it happens to sit on.
    #
    # Apostrophes are blanked out rather than removed, so every index into the
    # masked text still points at the same character as in the original. That
    # lets the per-line search below work on the mask and report a context
    # sliced from the real line.
    masked = _APOSTROPHE_RE.sub(" ", text)
    straight = sum(masked.count(char) for char in _STRAIGHT_QUOTES)
    curly = sum(masked.count(char) for char in _CURLY_QUOTES)
    # The losing style is the one flagged, so a fix moves the manuscript
    # towards the convention it already mostly follows. A tie resolves against
    # the straight quotes: curly is what a printed book uses, so an even split
    # is a reason to change the straight ones, not the curly ones.
    wrong_style = "straight" if straight <= curly else "curly"
    right_style = "curly" if wrong_style == "straight" else "straight"
    mixed_quotes = straight > 0 and curly > 0
    if mixed_quotes:
        wrong = _STRAIGHT_QUOTES if wrong_style == "straight" else _CURLY_QUOTES
        for line in lines:
            masked_line = _APOSTROPHE_RE.sub(" ", line.text)
            for char in wrong:
                start = masked_line.find(char)
                if start < 0:
                    continue
                if straight == curly:
                    # Saying "mostly uses curly" here would be false, and a
                    # finding that misdescribes the manuscript is worse than no
                    # finding at all.
                    message = (
                        "This manuscript splits evenly between straight and "
                        "curly quotation marks. This one is straight; curly is "
                        "the convention for a printed book."
                    )
                else:
                    message = (
                        f"This manuscript mostly uses {right_style} quotation "
                        f"marks, but this one is {wrong_style}."
                    )
                findings.append(
                    Finding(
                        id=finding_id(
                            PUNCTUATION.id, line.number, char,
                            f"Mixed quotation marks ({wrong_style})",
                        ),
                        category=PUNCTUATION.id,
                        severity="low",
                        message=message,
                        chapter=line.chapter,
                        line_number=line.number,
                        context=_context(line.text, start, start + 1),
                        suggestion="Match the rest of the manuscript.",
                    )
                )
                # One finding per line. A line with both quote characters, or a
                # quoted phrase, has one problem to look at, not three.
                break

    for line in lines:
        for match in _DOUBLED_PUNCTUATION_RE.finditer(line.text):
            findings.append(
                Finding(
                    id=finding_id(
                        PUNCTUATION.id, line.number, match.group(0),
                        "Doubled punctuation",
                    ),
                    category=PUNCTUATION.id,
                    severity="low",
                    message=f"“{match.group(0)}” uses repeated punctuation.",
                    chapter=line.chapter,
                    line_number=line.number,
                    context=_context(line.text, match.start(), match.end()),
                    suggestion=f"Use a single “{match.group(1)}”.",
                    original=match.group(0),
                    replacement=match.group(1),
                )
            )
        for match in _MISSING_SPACE_RE.finditer(line.text):
            # A comma or colon pressed against the next word, but not a number
            # ("1,000") and not a time ("3:30").
            following = line.text[match.end() : match.end() + 1]
            if not following.isalpha():
                continue
            findings.append(
                Finding(
                    id=finding_id(
                        PUNCTUATION.id, line.number, match.group(0),
                        "Missing space after punctuation",
                    ),
                    category=PUNCTUATION.id,
                    severity="medium",
                    message=f"No space after “{match.group(0)}”.",
                    chapter=line.chapter,
                    line_number=line.number,
                    context=_context(line.text, match.start(), match.end() + 1),
                    suggestion=f"Add a space: “{match.group(0)} ”.",
                    original=match.group(0) + following,
                    replacement=match.group(0) + " " + following,
                )
            )
        for match in _SPACED_ELLIPSIS_RE.finditer(line.text):
            findings.append(
                Finding(
                    id=finding_id(
                        PUNCTUATION.id, line.number, match.group(0),
                        "Spaced ellipsis",
                    ),
                    category=PUNCTUATION.id,
                    severity="low",
                    message="This ellipsis is spaced out.",
                    chapter=line.chapter,
                    line_number=line.number,
                    context=_context(line.text, match.start(), match.end()),
                    suggestion="Use “…” or three unspaced dots.",
                )
            )

    note = None
    if mixed_quotes:
        note = (
            f"This manuscript mixes straight and curly quotation marks "
            f"({straight} straight, {curly} curly). Only the {wrong_style} ones "
            f"are listed."
        )
    return DetectorResult(PUNCTUATION.id, findings, True, note)


def detect_structure(lines: list[Line]) -> DetectorResult:
    """Find heading-level jumps and repeated scene breaks.

    A document that goes from `#` to `###` loses a level, which the exporter
    renders as a real structural gap. The check is on Markdown headings only:
    chapter headings in other styles are counted by services.chapters, which
    has no notion of nesting.
    """
    findings: list[Finding] = []
    previous_level = 0
    previous_level_line = 0
    scene_breaks: list[int] = []

    for line in lines:
        if _SCENE_BREAK_RE.match(line.text):
            scene_breaks.append(line.number)

        match = _HEADING_RE.match(line.text)
        if not match:
            continue
        level = len(match.group(1))
        # Level 1 is where a document starts, so the first heading is never a
        # jump however deep it goes.
        if previous_level and level > previous_level + 1:
            findings.append(
                Finding(
                    id=finding_id(
                        STRUCTURE.id, line.number, line.text.strip(),
                        "Heading level skipped",
                    ),
                    category=STRUCTURE.id,
                    severity="medium",
                    message=(
                        f"This heading is level {level}, but the one before it "
                        f"(line {previous_level_line}) is level {previous_level}. "
                        f"Level {previous_level + 1} is missing."
                    ),
                    chapter=line.chapter,
                    line_number=line.number,
                    context=line.text.strip(),
                    suggestion=(
                        f"Use {'#' * (previous_level + 1)} if this is a "
                        f"subsection, or {'#' * previous_level} if it is not."
                    ),
                )
            )
        previous_level = level
        previous_level_line = line.number

    # Three or more identical scene breaks in a row is a paste artefact, not a
    # deliberate pause.
    run = 1
    for index in range(1, len(scene_breaks)):
        if scene_breaks[index] == scene_breaks[index - 1] + 1:
            run += 1
            if run == 3:
                findings.append(
                    Finding(
                        id=finding_id(
                            STRUCTURE.id, scene_breaks[index], None,
                            "Repeated scene break",
                        ),
                        category=STRUCTURE.id,
                        severity="low",
                        message="Three scene breaks in a row.",
                        chapter=next(
                            (l.chapter for l in lines if l.number == scene_breaks[index]),
                            FRONT_MATTER,
                        ),
                        line_number=scene_breaks[index],
                        context="(consecutive scene-break lines)",
                        suggestion="Keep one.",
                    )
                )
        else:
            run = 1

    return DetectorResult(STRUCTURE.id, findings)


def detect_formatting(lines: list[Line]) -> DetectorResult:
    """Find word-processor debris: tabs, trailing space runs."""
    findings: list[Finding] = []
    tab_lines = [line for line in lines if _TAB_RE.search(line.text)]
    trailing_lines = [line for line in lines if _TRAILING_SPACE_RE.search(line.text)]

    # Reported as one finding per kind rather than one per line. A manuscript
    # imported from a word processor has hundreds of these, and a list of
    # hundreds buries the findings that need thought.
    if tab_lines:
        findings.append(
            Finding(
                id=finding_id(
                    FORMATTING.id, tab_lines[0].number, None, "Tab characters"
                ),
                category=FORMATTING.id,
                severity="low",
                message=(
                    f"{len(tab_lines)} line{'' if len(tab_lines) == 1 else 's'} "
                    f"contain tab characters. The exporter indents with styles, "
                    f"so these render as inconsistent spacing."
                ),
                chapter=tab_lines[0].chapter,
                line_number=tab_lines[0].number,
                context=_context(tab_lines[0].text, 0, len(tab_lines[0].text)),
                suggestion="Replace tabs with spaces, or remove them.",
            )
        )
    if trailing_lines:
        findings.append(
            Finding(
                id=finding_id(
                    FORMATTING.id, trailing_lines[0].number, None, "Trailing whitespace"
                ),
                category=FORMATTING.id,
                severity="low",
                message=(
                    f"{len(trailing_lines)} line"
                    f"{'' if len(trailing_lines) == 1 else 's'} end in whitespace."
                ),
                chapter=trailing_lines[0].chapter,
                line_number=trailing_lines[0].number,
                context=repr(trailing_lines[0].text[-40:]),
                suggestion="Strip trailing whitespace.",
            )
        )

    return DetectorResult(FORMATTING.id, findings)


def build_cast(lines: list[Line]) -> list[tuple[str, int]]:
    """The character names in the manuscript, most-mentioned first.

    A name is a capitalised word appearing *mid-sentence*, at least
    `MIN_NAME_OCCURRENCES` times. Mid-sentence is the whole trick: "The", "He"
    and "She" open sentences constantly and would otherwise crowd out the cast
    of any book, and a character who never appears anywhere but the start of a
    sentence is not one a continuity check can say much about anyway.

    The list primes the AI's continuity prompt. It is not authoritative, and is
    not presented to the author as though it were.
    """
    counts: Counter[str] = Counter()
    for line in lines:
        for match in re.finditer(r"\b[A-Z][a-z'’-]{1,}\b", line.text):
            word = match.group()
            if word.lower() in _NOT_NAMES:
                continue
            if _starts_sentence(line.text, match.start()):
                continue
            counts[word] += 1

    return [
        (name, count)
        for name, count in counts.most_common()
        if count >= MIN_NAME_OCCURRENCES
    ]


def _edit_distance(left: str, right: str) -> int:
    """Levenshtein distance between `left` and `right`."""
    if left == right:
        return 0
    if abs(len(left) - len(right)) > NAME_VARIANT_DISTANCE:
        return NAME_VARIANT_DISTANCE + 1
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (left_char != right_char),  # substitution
                )
            )
        previous = current
    return previous[-1]


def detect_character_names(lines: list[Line], cast: list[tuple[str, int]]) -> DetectorResult:
    """Find character names that are spelled more than one way.

    "Maribel" in chapter two and "Maribelle" in chapter nine is either a typo or
    a different person, and only the author knows which. The detector's job is
    to raise it, not to decide.
    """
    findings: list[Finding] = []
    names = [name for name, _ in cast]
    seen_pairs: set[tuple[str, str]] = set()

    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if abs(len(name) - len(other)) > NAME_VARIANT_DISTANCE:
                continue
            # Names shorter than four letters collide too easily ("Ana"/"Ann"
            # against "Ava"), and the shorter the name the less the distance
            # means.
            if min(len(name), len(other)) < 4:
                continue
            if _edit_distance(name.lower(), other.lower()) > NAME_VARIANT_DISTANCE:
                continue
            key = (name, other) if name < other else (other, name)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            first = next((line for line in lines if name in line.text), lines[0])
            findings.append(
                Finding(
                    id=finding_id(
                        CHARACTER_NAMES.id, first.number, None,
                        f"Name variants: {key[0]} / {key[1]}",
                    ),
                    category=CHARACTER_NAMES.id,
                    severity="high",
                    message=(
                        f"“{key[0]}” and “{key[1]}” both appear and differ by "
                        f"one or two letters. Either one is misspelled, or they "
                        f"are two characters with confusingly similar names."
                    ),
                    chapter=first.chapter,
                    line_number=first.number,
                    context=f"First use of “{name}”: {first.text.strip()[:80]}",
                    suggestion=f"Keep one spelling, unless they are two people.",
                )
            )

    return DetectorResult(CHARACTER_NAMES.id, findings)


def detect_pacing(chapters: list[ChapterSummary]) -> DetectorResult:
    """Flag chapters that are outliers against the manuscript's own median.

    Compared to the book's own median rather than a fixed word count, because a
    40,000-word novella and a 200,000-word epic have very different ideas of a
    normal chapter.
    """
    # Front matter is not a chapter and its length says nothing about pacing.
    real = [chapter for chapter in chapters if chapter.title != FRONT_MATTER]
    if len(real) < 3:
        return DetectorResult(
            PACING.id,
            [],
            available=False,
            note=(
                "Pacing is judged across chapters, and fewer than three were "
                "detected."
            ),
        )

    lengths = [chapter.word_count for chapter in real]
    middle = median(lengths)
    if middle <= 0:
        return DetectorResult(
            PACING.id, [], available=False, note="The chapters are empty."
        )

    findings: list[Finding] = []
    for chapter in real:
        ratio = chapter.word_count / middle
        if PACING_LOW_RATIO <= ratio <= PACING_HIGH_RATIO:
            continue
        direction = "shorter" if ratio < 1 else "longer"
        findings.append(
            Finding(
                id=finding_id(
                    PACING.id, chapter.line_number, chapter.title, "Chapter pacing"
                ),
                category=PACING.id,
                severity="medium" if ratio < PACING_LOW_RATIO else "low",
                message=(
                    f"“{chapter.title}” is {chapter.word_count:,} words — "
                    f"{abs(ratio - 1):.0%} {direction} than the median chapter "
                    f"({middle:,.0f} words)."
                ),
                chapter=chapter.title,
                line_number=chapter.line_number,
                context=f"Chapter of {chapter.word_count:,} words.",
                suggestion=(
                    "Split it, or check that nothing is missing."
                    if ratio > 1
                    else "Check that this chapter is complete."
                ),
            )
        )

    return DetectorResult(PACING.id, findings)


def run_rule_detectors(
    text: str, lines: list[Line], chapters: list[ChapterSummary]
) -> list[DetectorResult]:
    """Run every deterministic detector, in a stable order."""
    return [
        detect_repeated_words(lines),
        detect_punctuation(lines, text),
        detect_character_names(lines, build_cast(lines)),
        detect_structure(lines),
        detect_formatting(lines),
        detect_pacing(chapters),
    ]
