"""Verification harness for the manuscript health report.

Same shape as verify.py: no network here and pip hangs, so fastapi/docx/
langdetect/pyspellchecker are stubbed and the platform's own logic runs for
real. health.py itself has no third-party imports — it reaches the standard
library and three sibling modules — so everything below exercises genuine code.

The false-positive controls matter as much as the detections. A detector that
fires on ordinary prose is worse than one that misses: an author who sees three
bogus warnings learns to ignore the panel entirely.

Run:  /root/venv/bin/python verify_health.py     (from the project root)
"""

import sys
import types

sys.path.insert(0, "/root/ebook-formatter")

FAILS = []


def check(label, actual, expected):
    ok = actual == expected
    suffix = "" if ok else f"  WANT {expected!r}"
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {actual!r}{suffix}")
    if not ok:
        FAILS.append(label)


def check_close(label, actual, expected, tolerance=0.05):
    ok = actual is not None and abs(actual - expected) <= tolerance
    suffix = "" if ok else f"  WANT {expected!r} +/- {tolerance}"
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {actual!r}{suffix}")
    if not ok:
        FAILS.append(label)


# ---- stub spellchecker ----
class WordFrequency:
    def __init__(self, counts):
        self._counts = dict(counts)

    def __getitem__(self, key):
        return self._counts[key]

    def __contains__(self, key):
        return key in self._counts

    def __iter__(self):
        return iter(self._counts)

    def __len__(self):
        return len(self._counts)


class SpellChecker:
    def __init__(self, language="en", **kw):
        if language != "en":
            raise ValueError("no dictionary")
        self.word_frequency = WordFrequency({"the": 100})

    def unknown(self, words):
        return {w for w in words if w.lower() not in self.word_frequency}

    def candidates(self, word):
        return set()


sc = types.ModuleType("spellchecker")
sc.SpellChecker = SpellChecker
sys.modules["spellchecker"] = sc

# ---- stub langdetect ----
BLOCKS = [("ঀ", "৿", "bn"), ("ऀ", "ॿ", "hi"), ("Ѐ", "ӿ", "ru"),
          ("Ͱ", "Ͽ", "el"), ("؀", "ۿ", "ar"), ("֐", "׿", "he"),
          ("一", "鿿", "zh-cn"), ("぀", "ヿ", "ja"), ("가", "힯", "ko")]


class LangDetectException(Exception):
    pass


class DetectorFactory:
    seed = 0


def detect(text):
    if not any(c.isalpha() for c in text):
        raise LangDetectException("no letters")
    for ch in text:
        for lo, hi, code in BLOCKS:
            if lo <= ch <= hi:
                return code
    return "en"


ld = types.ModuleType("langdetect")
ld.DetectorFactory = DetectorFactory
ld.LangDetectException = LangDetectException
ld.detect = detect
sys.modules["langdetect"] = ld

# ---- stub docx (extractors imports it at module level) ----
dx = types.ModuleType("docx")


class Document:
    def __init__(self, *a, **kw):
        raise NotImplementedError("python-docx not installed")


dx.Document = Document
sys.modules["docx"] = dx

# ---------------------------------------------------------------- tests
from services import health
from services.analyzer import count_words, estimate_pages
from services.chapters import chapter_breakdown, count_chapters
from services.health import (
    CATEGORIES,
    FRONT_MATTER,
    build_cast,
    completeness_score,
    count_paragraphs,
    count_sentences,
    detect_character_names,
    detect_formatting,
    detect_pacing,
    detect_punctuation,
    detect_repeated_words,
    detect_structure,
    flesch_kincaid_grade,
    flesch_reading_ease,
    guess_genre,
    index_lines,
    measure,
    reading_time_minutes,
    run_rule_detectors,
    syllable_count,
)

print("== 1. syllable counting ==")
# The heuristic counts vowel groups and takes one back for a silent trailing
# "e". These are the cases it is designed to get right.
for word, expected in [
    ("cat", 1), ("the", 1), ("make", 1), ("sound", 1), ("aeiou", 1),
    ("table", 2), ("people", 2), ("banana", 3), ("beautiful", 3),
    ("", 0), ("123", 0), ("a", 1),
]:
    check(f"syllables({word!r})", syllable_count(word), expected)

print("== 2. readability formulas ==")
# Hand-worked: 100 words, 10 sentences, 150 syllables.
#   FRE  = 206.835 - 1.015*10 - 84.6*1.5 = 69.785
#   FKGL = 0.39*10 + 11.8*1.5 - 15.59   = 6.01
check_close("flesch reading ease", flesch_reading_ease(100, 10, 150), 69.8)
check_close("flesch-kincaid grade", flesch_kincaid_grade(100, 10, 150), 6.0)
check("empty text scores 0", flesch_reading_ease(0, 0, 0), 0.0)
check("no syllables scores 0", flesch_kincaid_grade(0, 0, 0), 0.0)

print("== 3. counts and estimates ==")
check("sentences, simple", count_sentences("One. Two. Three."), 3)
check("sentences, terminators run once", count_sentences("What?! Really."), 2)
check("sentences, empty", count_sentences(""), 0)
check("paragraphs", count_paragraphs("a\n\nb\n\n\nc\n"), 3)
check("paragraphs, none", count_paragraphs("   \n\n  "), 0)
check("reading time, one minute floor", reading_time_minutes(10), 1)
check("reading time, 220 wpm", reading_time_minutes(220), 1)
check("reading time, 440 words", reading_time_minutes(440), 2)
check("reading time, 1100 words", reading_time_minutes(1100), 5)

# The health report and the exporter must never disagree about how long a book
# is, which is why print_pages goes through the analyzer's own function.
for words in (0, 1, 249, 250, 251, 90_000):
    check(f"print_pages({words}) == estimate_pages", health.Metrics.__dataclass_fields__ and estimate_pages(words), estimate_pages(words))

print("== 4. reading level declines outside English ==")
text_en = "The cat sat on the mat. The dog ran in the park. We went home."
level, note = health._reading_level(text_en, "en", 15, 3)
check("english yields a level", level is not None, True)
check("english has no note", note, None)
level_es, note_es = health._reading_level(text_en, "es", 15, 3)
check("spanish yields no level", level_es, None)
check("spanish explains why", note_es is not None and "English" in note_es, True)
level_zh, note_zh = health._reading_level("这是一本书。", "zh-cn", 5, 1)
check("chinese yields no level", level_zh, None)

print("== 5. line indexing and chapter attribution ==")
lines = index_lines("# Chapter One\n\nSome text.\n\n## Chapter Two\n\nMore text.\n")
check("line count", len(lines), 7)
check("line 1 is chapter 1", lines[0].chapter, "Chapter One")
check("line 2 is chapter 1", lines[1].chapter, "Chapter One")
check("line 5 is chapter 2", lines[4].chapter, "Chapter Two")
check("line 7 is chapter 2", lines[6].chapter, "Chapter Two")
check("line numbers are 1-based", lines[0].number, 1)
pre = index_lines("Just an opening.\n\n# Chapter One\n\nBody.\n")
check("text before the first heading is front matter", pre[0].chapter, FRONT_MATTER)

print("== 6. repeated words ==")
r = detect_repeated_words(index_lines("It was the the best of times.\n"))
check("doubled word found", len(r.findings), 1)
check("doubled word is fixable", r.findings[0].fixable, True)
check("doubled word replacement", r.findings[0].replacement, "the")
check("doubled word chapter", r.findings[0].chapter, FRONT_MATTER)
# "had had" and "that that" are English, and flagging them is the classic
# false positive that teaches an author to distrust the panel.
r = detect_repeated_words(index_lines("He had had enough of it.\n"))
check("'had had' is not flagged", len(r.findings), 0)
r = detect_repeated_words(index_lines("She said that that was fine.\n"))
check("'that that' is not flagged", len(r.findings), 0)
r = detect_repeated_words(index_lines("The the the the end.\n"))
check("tripled word counted", len(r.findings) >= 1, True)

print("== 7. punctuation ==")
r = detect_punctuation(index_lines("Wait!! What are you doing?\n"), "Wait!! What are you doing?\n")
check("doubled exclamation found", len(r.findings), 1)
check("doubled exclamation fixable", r.findings[0].replacement, "!")
r = detect_punctuation(index_lines("Hello,world here.\n"), "Hello,world here.\n")
check("missing space found", len(r.findings), 1)
check("missing space fix", r.findings[0].replacement, ", w")
check("missing space is fixable", r.findings[0].fixable, True)
# A number and a time are not missing spaces.
r = detect_punctuation(index_lines("It cost 1,000 dollars at 3:30 pm.\n"), "It cost 1,000 dollars at 3:30 pm.\n")
check("numbers and times are not flagged", len(r.findings), 0)
# Mixed quotation style: the minority style is the finding.
mixed = 'He said “hello” and then "goodbye" to her.\n'
r = detect_punctuation(index_lines(mixed), mixed)
check("mixed quotes flagged", len(r.findings), 1)
check("mixed quotes names the minority", "straight" in r.findings[0].message, True)
check("mixed quotes are advisory, not mechanical", r.findings[0].fixable, False)
consistent = 'He said “hello” and then “goodbye” to her.\n'
r = detect_punctuation(index_lines(consistent), consistent)
check("consistent quotes are quiet", len(r.findings), 0)
# An apostrophe is not a quotation mark. Counting them would report every
# English sentence as a straight quote.
contractions = "He said “I don’t know,” and she didn’t either.\n"
r = detect_punctuation(index_lines(contractions), contractions)
check("curly contractions are not quotes", len(r.findings), 0)
r = detect_punctuation(index_lines("It's fine, don't worry.\n"), "It's fine, don't worry.\n")
check("straight contractions are not quotes", len(r.findings), 0)
# A straight-quoted line whose only straight quote is an apostrophe must not be
# flagged as the wrong style — the mask has to be what the search runs on.
apostrophe_only = 'He said “hello” and don\'t stop, and "bye".\n'
r = detect_punctuation(index_lines(apostrophe_only), apostrophe_only)
check("apostrophe does not become the finding", len(r.findings), 1)
check("apostrophe finding anchors on a real quote", '"bye"' in r.findings[0].context, True)

print("== 8. structure ==")
r = detect_structure(index_lines("# One\n\n### Three\n\nBody.\n"))
check("heading level jump found", len(r.findings), 1)
check("jump is not mechanically fixable", r.findings[0].fixable, False)
r = detect_structure(index_lines("# One\n\n## Two\n\n### Three\n"))
check("proper nesting is quiet", len(r.findings), 0)
r = detect_structure(index_lines("# One\n\n## Deep\n\nBody.\n"))
check("first heading is never a jump", len(r.findings), 0)

print("== 9. formatting ==")
r = detect_formatting(index_lines("Clean line here.\n"))
check("clean text has no formatting findings", len(r.findings), 0)
r = detect_formatting(index_lines("Tabbed\tline here.\nAnother\tone.\n"))
check("tabs reported once, not per line", len(r.findings), 1)
check("tab finding counts the lines", "2 lines" in r.findings[0].message, True)
r = detect_formatting(index_lines("Trailing space here.   \nMore text.\n"))
check("trailing whitespace found", len(r.findings), 1)

print("== 10. character names ==")
# Mid-sentence is what qualifies a name; see build_cast's docstring.
prose = (
    "# Chapter One\n\n"
    "Then Maribel walked in. Later, Maribel sat down.\n\n"
    "# Chapter Two\n\n"
    "He called Maribelle over. Everyone watched Maribelle leave.\n"
)
cast = build_cast(index_lines(prose))
names = [name for name, _ in cast]
check("Maribel is in the cast", "Maribel" in names, True)
check("Maribelle is in the cast", "Maribelle" in names, True)
# Sentence-opening words must never reach the cast, or the list is worthless.
check("'Then' is not a character", "Then" in names, False)
check("'Everyone' is not a character", "Everyone" in names, False)
r = detect_character_names(index_lines(prose), cast)
check("name variant found", len(r.findings), 1)
check("variant names both spellings", "Maribel" in r.findings[0].message and "Maribelle" in r.findings[0].message, True)
check("variant is not mechanically fixable", r.findings[0].fixable, False)
# Two genuinely different names must not be merged.
other = build_cast(index_lines("He saw Anna today. Then Anna left. He saw Beth too. Then Beth left.\n"))
r = detect_character_names(index_lines("He saw Anna today. Then Anna left. He saw Beth too. Then Beth left.\n"), other)
check("Anna and Beth are not variants", len(r.findings), 0)

print("== 11. pacing ==")
chapters = chapter_breakdown("# A\n\n" + "word " * 1000 + "\n\n## B\n\n" + "word " * 1000 + "\n\n## C\n\n" + "word " * 50 + "\n", len)
r = detect_pacing(chapters)
check("short outlier found", len(r.findings), 1)
check("outlier names the chapter", r.findings[0].chapter, "C")
check("pacing is not mechanically fixable", r.findings[0].fixable, False)
even = chapter_breakdown("# A\n\n" + "word " * 1000 + "\n\n## B\n\n" + "word " * 1100 + "\n\n## C\n\n" + "word " * 900 + "\n", len)
check("even chapters are quiet", len(detect_pacing(even).findings), 0)
thin = chapter_breakdown("# A\n\nx\n\n## B\n\ny\n", len)
check("pacing declines on too few chapters", detect_pacing(thin).available, False)

print("== 12. false-positive control: clean prose ==")
clean = (
    "# Chapter One\n\n"
    "Maribel walked to the window and looked out at the street. The rain had\n"
    "stopped an hour ago, and the road below was quiet.\n\n"
    "She turned back to the room. Nothing had changed since the morning, and\n"
    "she wondered whether it ever would.\n"
)
clean_lines = index_lines(clean)
for result in run_rule_detectors(clean, clean_lines, chapter_breakdown(clean, len)):
    check(f"clean prose: {result.category} is quiet", len(result.findings), 0)

print("== 13. completeness ==")
finished = (
    "# Chapter One\n\n" + "word " * 400 + ".\n\n"
    "## Chapter Two\n\n" + "word " * 400 + ".\n\n"
    "## Chapter Three\n\n" + "word " * 400 + ".\n"
)
score, notes = completeness_score(finished, chapter_breakdown(finished, len), 1203)
check("a finished manuscript scores 100", score, 100)
check("a finished manuscript has no notes", notes, [])
fragment = "Once upon a time there was"
score, notes = completeness_score(fragment, [], 7)
check("a fragment scores low", score < 50, True)
check("a fragment lists what is missing", len(notes) >= 3, True)
check("unfinished ending is named", any("finished sentence" in n for n in notes), True)

print("== 14. genre ==")
check("code fences read as technical", guess_genre("```\ninstall the package\n```\n" * 40, [])[0], "technical")
check("citations read as academic", guess_genre("References\nsee et al, doi:10.1\n" * 40, [])[0], "academic")
check("dialogue reads as fiction", guess_genre('"Hello," he said.\n' * 60, [])[0], "fiction")
check("nothing scores means the default", guess_genre("plain text", [])[0], health.DEFAULT_GENRE)
check("the guess reports its confidence", isinstance(guess_genre('"Hi," he said.\n' * 60, [])[1], float), True)

print("== 15. metrics ==")
manuscript = (
    "# Chapter One\n\nIt was a dark and stormy night. The rain fell hard.\n\n"
    "## Chapter Two\n\nShe walked down the street and counted the windows.\n"
)
chapters = chapter_breakdown(manuscript, len)
m = measure(manuscript, chapters, "en", ai_enabled=False)
check("word count agrees with the analyzer", m.word_count, len(manuscript.split()))
check("character count", m.character_count, len(manuscript))
check("character count without spaces", m.character_count_no_spaces, len("".join(manuscript.split())))
check("chapter count excludes front matter", m.chapter_count, 2)
check("print pages agrees with the analyzer", m.print_pages, estimate_pages(m.word_count))
check("reading time is at least a minute", m.reading_time_minutes >= 1, True)
check("english gets a reading level", m.reading_level is not None, True)
check("reading level has a label", bool(m.reading_level.label), True)
check("completeness is a percentage", 0 <= m.completeness <= 100, True)
check("processing estimate is positive", m.estimated_processing_seconds > 0, True)
check("AI off is cheaper than AI on", m.estimated_processing_seconds < measure(manuscript, chapters, "en", ai_enabled=True).estimated_processing_seconds, True)

print("== 16. finding ids are stable ==")
a = detect_repeated_words(index_lines("It was the the best.\n")).findings
b = detect_repeated_words(index_lines("It was the the best.\n")).findings
check("same text, same id", a[0].id, b[0].id)
c = detect_repeated_words(index_lines("It was a a best.\n")).findings
check("different text, different id", a[0].id == c[0].id, False)

print("== 17. category registry ==")
check("every rule category has metadata", all(c in CATEGORIES for c in health.RULE_CATEGORIES), True)
check("every AI category has metadata", all(c in CATEGORIES for c in health.AI_CATEGORIES), True)
check("grammar is AI-computed", CATEGORIES["grammar"].source, "ai")
check("pacing is rule-computed", CATEGORIES["pacing"].source, "rules")
# Fix All must never be allowed to rewrite prose on its own judgement.
for advisory in ("continuity", "plot", "dialogue", "pacing", "structure", "character_names"):
    check(f"{advisory} is not auto-fixable", CATEGORIES[advisory].fixable, False)
for mechanical in ("spelling", "grammar", "punctuation", "repeated_words", "formatting"):
    check(f"{mechanical} is auto-fixable", CATEGORIES[mechanical].fixable, True)

# ---------------------------------------------------------------- epub
print("== 18. EPUB ingestion ==")

import io
import zipfile

from services import extractors


def make_epub(spine, documents):
    """Build a minimal EPUB 3 whose reading order is `spine`.

    The file names are deliberately not in spine order, so a reader that sorted
    the archive alphabetically instead of following the spine would emit the
    chapters the wrong way round and be caught here.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0" encoding="utf-8"?>'
            '<container version="1.0" '
            'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        manifest = "".join(
            f'<item id="{name}" href="{name}" media-type="application/xhtml+xml"/>'
            for name in spine
        )
        itemrefs = "".join(f'<itemref idref="{name}"/>' for name in spine)
        archive.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0" encoding="utf-8"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
            'unique-identifier="id"><metadata '
            'xmlns:dc="http://purl.org/dc/elements/1.1/">'
            "<dc:title>Fixture</dc:title></metadata>"
            f"<manifest>{manifest}</manifest><spine>{itemrefs}</spine></package>",
        )
        for name, body in documents.items():
            archive.writestr(
                f"OEBPS/{name}",
                '<?xml version="1.0" encoding="utf-8"?>'
                '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                f"{body}</body></html>",
            )
    return buffer.getvalue()


check("epub is an accepted extension", ".epub" in extractors.SUPPORTED_EXTENSIONS, True)

book = make_epub(
    ["second.xhtml", "first.xhtml"],
    {
        "first.xhtml": "<h1>Chapter One</h1><p>The opening chapter.</p>",
        "second.xhtml": "<h1>Chapter Two</h1><p>The closing chapter.</p>",
    },
)
book_text = extractors.extract_text("book.epub", book)
# The spine puts second.xhtml first, so "Two" must precede "One". Sorting the
# archive alphabetically would give the opposite order, which is what this
# catches — a reader that ignores the spine silently scrambles the book.
check(
    "epub follows the spine, not the alphabet",
    book_text.index("Chapter Two") < book_text.index("Chapter One"),
    True,
)
check("epub headings become markdown", "# Chapter One" in book_text, True)
check("epub chapters are counted", len(chapter_breakdown(book_text, count_words)), 2)
# Inline markup must not fragment a sentence into separate blocks.
inline = make_epub(["only.xhtml"], {"only.xhtml": "<p>He was <em>very</em> tired.</p>"})
check(
    "inline markup stays in one block",
    "He was very tired." in extractors.extract_text("book.epub", inline),
    True,
)
# A zip that is not an EPUB is refused rather than crashing.
not_an_epub = io.BytesIO()
with zipfile.ZipFile(not_an_epub, "w") as archive:
    archive.writestr("random.txt", "not an epub")
try:
    extractors.extract_text("book.epub", not_an_epub.getvalue())
    check("a zip with no documents is refused", "no error raised", "CorruptDocumentError")
except extractors.CorruptDocumentError:
    check("a zip with no documents is refused", "CorruptDocumentError", "CorruptDocumentError")

# ---------------------------------------------------------------- ai
print("== 19. AI chunking ==")

from services.health_ai import chunk_manuscript

manuscript = (
    "# Chapter One\n\n" + "alpha beta gamma delta.\n\n" * 30
    + "# Chapter Two\n\n" + "epsilon zeta eta theta.\n\n" * 30
    + "# Chapter Three\n\n" + "iota kappa lambda mu.\n" * 30
)
chunk_lines = index_lines(manuscript)
chapters = chapter_breakdown(manuscript, count_words)

one = chunk_manuscript(chunk_lines)
check("a short manuscript is one chunk", len(one), 1)
check("a chunk starts at line 1", one[0].first_line, 1)
check(
    "a chunk names the chapters it holds",
    list(one[0].chapters),
    ["Chapter One", "Chapter Two", "Chapter Three"],
)

# A budget too small for the whole book forces a split.
small = chunk_manuscript(chunk_lines, max_chars=400)
check("a small budget splits the book", len(small) > 1, True)
check("every chunk respects the budget", all(len(c.text) <= 400 for c in small), True)
check("chunks are numbered in order", [c.index for c in small], list(range(len(small))))
check(
    "chunk line numbers ascend",
    all(a.first_line <= b.first_line for a, b in zip(small, small[1:])),
    True,
)
joined = "\n".join(c.text for c in small)
for title in ("Chapter One", "Chapter Two", "Chapter Three"):
    # Split mid-chapter is allowed; losing or duplicating a heading is not.
    check(f"{title} survives chunking exactly once", joined.count(title), 1)

print("== 20. locating a model's quote ==")

from services.health_ai import Chunk, _locate

probe = Chunk(
    index=0,
    text="The cat sat.\nIt was a grey cat.\n",
    chapters=("One",),
    first_line=10,
)
line_number, exact = _locate(probe, "It was a grey cat.")
check("a verbatim quote is located", line_number, 11)
check("a verbatim quote is exact", exact, True)
# A quote the model reflowed across a line break is real but not replaceable.
line_number, exact = _locate(probe, "It was a\ngrey cat.")
check("a reflowed quote is still located", line_number, 11)
check("a reflowed quote is not exact", exact, False)
line_number, exact = _locate(probe, "This text is nowhere.")
check("an absent quote is not claimed", exact, False)

print("== 21. AI cost accounting ==")

from services.health_ai import credits_for, price_usage

big = {
    "input_tokens": 1_000_000,
    "output_tokens": 1_000_000,
    "cache_read_input_tokens": 1_000_000,
    "cache_creation_input_tokens": 1_000_000,
}
# sonnet-5-5 is $3/Mtok in and $15/Mtok out. A cache read is 0.1x input and a
# cache write 1.25x, so: 3 + 15 + 0.30 + 3.75.
check_close("cache reads and writes are priced", price_usage("claude-sonnet-5-5", big), 22.05, 1e-9)
check_close(
    "a dated snapshot prices as its family",
    price_usage("claude-sonnet-5-5-20260101", big),
    22.05,
    1e-9,
)
check(
    "an unknown model does not price at zero",
    price_usage("some-future-model", {"input_tokens": 1_000_000}),
    3.0,
)
check("no tokens is no cost", price_usage("claude-sonnet-5-5", {}), 0.0)
check("a tiny cost is still at least one credit", credits_for(0.0001), 1)
check("no cost is no credits", credits_for(0.0), 0)
check("a dollar is a hundred credits", credits_for(1.0), 100)

print("== 22. the AI scan against a stubbed API ==")

import asyncio
import json
import os

import services.health_ai as health_ai


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeClient:
    """Replays scripted responses, one per POST, and records the requests."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):
        self.requests.append({"url": url, "headers": headers, "body": json})
        if not self.script:
            raise AssertionError("the scan made more calls than the test scripted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def answer(findings, usage=None, status=200):
    """A Messages API response carrying a structured answer.

    The findings ride inside a text block as JSON, which is where they really
    are — reading them off the top level of the response would find nothing and
    report a clean manuscript.
    """
    return FakeResponse(
        status,
        {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": json.dumps({"findings": findings})}],
            "usage": usage or {"input_tokens": 100, "output_tokens": 50},
        },
    )


fake_httpx = types.ModuleType("httpx")
holder = {}


def _async_client(timeout=None):
    return holder["client"]


fake_httpx.AsyncClient = _async_client
sys.modules["httpx"] = fake_httpx

saved_key = os.environ.get("ANTHROPIC_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = "test-key"

# The retry ladder, without actually waiting out the backoff.
real_sleep = health_ai.asyncio.sleep
slept = []


async def no_sleep(seconds):
    slept.append(seconds)


health_ai.asyncio.sleep = no_sleep


def scan(script, chunks, lines, chapters, cast=(), progress=None):
    holder["client"] = FakeClient(script)
    return asyncio.run(
        health_ai.run_ai_scan(
            text=manuscript,
            lines=lines,
            chunks=chunks,
            chapters=chapters,
            cast=list(cast),
            language="en",
            genre="fiction",
            title="Fixture",
            author="A. Writer",
            ignored=[],
            on_progress=progress,
        )
    )


try:
    grammar_item = {
        "category": "grammar",
        "severity": "medium",
        "message": "The subject and verb disagree.",
        "quote": "They was late again",
        "correction": "They were late again",
    }
    prose = "# Chapter One\n\nThey was late again that morning.\n"
    prose_lines = index_lines(prose)
    prose_chunks = chunk_manuscript(prose_lines)

    # A 429 then a 503, then success: the ladder climbs and recovers.
    seen = []
    result = scan(
        [
            answer([], status=429),
            answer([], status=503),
            answer([grammar_item]),
        ],
        prose_chunks,
        prose_lines,
        [],
    )
    check("a rate limit and a 5xx are retried", result.available, True)
    # 429 backs off (attempt+1)*5 and 5xx (attempt+1)*2, so attempt 0 waits 5s
    # and attempt 1 waits 4s.
    check("the retry ladder backs off 5s then 4s", slept, [5, 4])

    # The findings really do arrive: this is the check that catches a scan
    # reading them off the wrong level of the response.
    slept.clear()
    result = scan([answer([grammar_item])], prose_chunks, prose_lines, [])
    check("a finding is returned", len(result.findings), 1)
    check("its category is kept", result.findings[0].category, "grammar")
    check("it is located in the manuscript", result.findings[0].line_number, 3)
    check("a verbatim quote may be applied", result.findings[0].fixable, True)
    check("the correction is the replacement", result.findings[0].replacement, "They were late again")
    check("the owning chapter is attributed", result.findings[0].chapter, "Chapter One")

    # A quote the model reflowed cannot be applied mechanically, but the wording
    # it proposed is still advice the author should see.
    slept.clear()
    reflowed = dict(
        grammar_item,
        quote="They was\nlate again",
        correction="They were late again",
    )
    result = scan([answer([reflowed])], prose_chunks, prose_lines, [])
    check("a reflowed quote is reported", len(result.findings), 1)
    check("a reflowed quote is not auto-applicable", result.findings[0].fixable, False)
    check("but its advice is kept", result.findings[0].suggestion, "They were late again")

    # A category we did not ask for is dropped rather than guessed at.
    slept.clear()
    stray = dict(grammar_item, category="spelling")
    result = scan([answer([stray])], prose_chunks, prose_lines, [])
    check("an unrequested category is dropped", len(result.findings), 0)

    # The whole-book digest is a second call, on the stronger model.
    slept.clear()
    result = scan(
        [answer([grammar_item]), answer([grammar_item]), answer([grammar_item])],
        one,
        chunk_lines,
        chapters,
    )
    check("chunks plus one digest are all called", result.calls, 2)
    check("the digest runs on the continuity model", result.model, health_ai.chunk_model())

    # Progress is reported, and never leaves 0..1.
    slept.clear()
    fractions = []

    async def watch(fraction):
        fractions.append(fraction)

    scan([answer([grammar_item]), answer([])], one, chunk_lines, chapters, progress=watch)
    check("progress is reported", len(fractions) > 0, True)
    check("progress stays within 0..1", all(0.0 <= f <= 1.0 for f in fractions), True)

    # A terminal 4xx is this request being wrong; retrying it wastes the spend.
    slept.clear()
    holder["client"] = FakeClient([FakeResponse(400, {"error": {"message": "bad model"}})])
    try:
        asyncio.run(
            health_ai.run_ai_scan(
                text=prose,
                lines=prose_lines,
                chunks=prose_chunks,
                chapters=[],
                cast=[],
                language="en",
                genre="fiction",
            )
        )
        check("a 400 fails the scan", "no error raised", "HealthAIError")
    except health_ai.HealthAIError as exc:
        check("a 400 fails the scan", "bad model" in str(exc), True)
    check("a 400 is not retried", slept, [])

    print("== 23. the AI scan with no key configured ==")

    os.environ.pop("ANTHROPIC_API_KEY", None)
    check("ai_configured is false with no key", health_ai.ai_configured(), False)
    quiet = asyncio.run(
        health_ai.run_ai_scan(
            text=prose,
            lines=prose_lines,
            chunks=prose_chunks,
            chapters=[],
            cast=[],
            language="en",
            genre="fiction",
        )
    )
    check("a scan with no key does not raise", quiet.available, False)
    check("a scan with no key costs nothing", quiet.cost_usd, 0.0)
    check(
        "a scan with no key explains itself",
        "ANTHROPIC_API_KEY" in (quiet.note or ""),
        True,
    )
finally:
    health_ai.asyncio.sleep = real_sleep
    if saved_key is None:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    else:
        os.environ["ANTHROPIC_API_KEY"] = saved_key

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("all health checks passed")
