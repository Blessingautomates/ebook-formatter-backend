"""Verification harness for the ebook-formatter changes.

No network here and pip hangs, so fastapi/docx/langdetect/pyspellchecker cannot
be installed. This stubs those four and runs the platform's own logic for real.
pydantic is genuine (from /root/venv), so the API model is really validated.

Run:  /root/venv/bin/python verify.py     (from the project root)
"""

import asyncio
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


# ---- stub fastapi (pydantic is real) ----
fm = types.ModuleType("fastapi")


class FastAPI:
    def __init__(self, **kw): pass
    def get(self, p, **kw): return lambda f: f
    def post(self, p, **kw): return lambda f: f


class HTTPException(Exception):
    def __init__(self, status_code, detail=""):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class UploadFile:
    def __init__(self, filename, data):
        self.filename = filename
        self._data = data
    async def read(self):
        return self._data


fm.FastAPI, fm.HTTPException, fm.UploadFile = FastAPI, HTTPException, UploadFile
fm.File = lambda default=None, **kw: default
fm.Form = lambda default=None, **kw: default
sys.modules["fastapi"] = fm

# main.py imports StreamingResponse from this submodule, so it has to exist too.
fr = types.ModuleType("fastapi.responses")


class StreamingResponse:
    def __init__(self, content=None, media_type=None, headers=None, **kw):
        self.content = content
        self.media_type = media_type
        self.headers = headers


fr.StreamingResponse = StreamingResponse
fm.responses = fr
sys.modules["fastapi.responses"] = fr

# ---- stub spellchecker ----
DICT = {w: 10 for w in (
    "the tech ten then there rain fell hard and he walked down street before stopping to "
    "watch it fall today she counted windows of every house on block found them all dark "
    "waved at crowd went back inside empty hall chapter one two beginnings middles a was "
    "day saw thing well known as page book sample"
).split()}
DICT["the"] = 1000


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class WordFrequency:
    """Mirrors pyspellchecker's `WordFrequency`: a mapping, but not a `dict`.

    It is subscriptable and iterable and defines `__contains__`, yet it has no
    `get`. Stubbing this as a plain `dict` hid an AttributeError that only
    showed up against the installed library, so the real shape is reproduced.
    """

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
        self.word_frequency = WordFrequency(DICT)
    def unknown(self, words):
        return {w for w in words if w.lower() not in self.word_frequency}
    def candidates(self, word):
        return {c for c in self.word_frequency if _lev(word, c) <= 2}


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
ld.DetectorFactory, ld.LangDetectException, ld.detect = DetectorFactory, LangDetectException, detect
sys.modules["langdetect"] = ld

# ---- stub docx ----
dx = types.ModuleType("docx")


class Document:
    def __init__(self, *a, **kw):
        raise NotImplementedError("python-docx not installed")


dx.Document = Document
sys.modules["docx"] = dx

# ---------------------------------------------------------------- tests
from dataclasses import asdict

import main
from services.analyzer import analyze_manuscript
from services.chapters import chapter_breakdown, count_chapters
from services.detection import _direction_and_script, detect_language
from services.extractors import UnsupportedFormatError
from services.typos import MAX_TYPOS, scan_typos

print("== 1. script classification ==")
for code in ("ru", "bg", "uk"):
    check(f"{code}->cyrillic", _direction_and_script(code), ("ltr", "cyrillic"))
check("el->greek", _direction_and_script("el"), ("ltr", "greek"))
for code in ("hi", "mr", "ne", "bn"):
    check(f"{code}->devanagari", _direction_and_script(code), ("ltr", "devanagari"))
for code in ("ar", "he", "ur", "fa"):
    check(f"{code}->rtl", _direction_and_script(code), ("rtl", "rtl"))
for code in ("zh-cn", "ja", "ko"):
    check(f"{code}->cjk", _direction_and_script(code), ("ltr", "cjk"))
for code in ("en", "es", "fr", "nl"):
    check(f"{code}->latin", _direction_and_script(code), ("ltr", "latin"))
check("unknown->latin", _direction_and_script("xx"), ("ltr", "latin"))
check("russian detected", detect_language("Привет мир").script_type, "cyrillic")
check("greek name", detect_language("Καλημέρα κόσμε").name, "Greek")
check("hindi script", detect_language("नमस्ते दुनिया").script_type, "devanagari")
check("bengali name", detect_language("হ্যালো বিশ্ব").name, "Bengali")

print("== 2. chapter counting ==")
check("english", count_chapters("# T\n\n## Chapter 1: A\n\nx\n\n## Chapter 2: B\n\nx\n"), 3)
check("arabic", count_chapters("الفصل الأول\n\nنص.\n\nالفصل الثاني\n\nنص.\n"), 2)
check("chinese", count_chapters("第一章\n\n文字。\n\n第二章\n\n文字。\n\n第三章\n\n文。\n"), 3)
check("prose ignored", count_chapters("Chapter 3 was long.\n3 apples fell.\nCIVIL war.\n"), 0)
check("roman+numbered", count_chapters("I\n\nII\n\nIII\n\n12. The Return\n\n13 - Aftermath\n"), 5)

split = "Preface words here.\n\n# Chapter One\n\nA b c.\n\n# Chapter Two\n\nD e.\n"


def words(text):
    """Stands in for analyzer.count_words, which is the same split for ASCII."""
    return len(text.split())


bd = chapter_breakdown(split, words)
check("breakdown titles", [c.title for c in bd], ["Front Matter", "Chapter One", "Chapter Two"])
check("breakdown lines", [c.line_number for c in bd], [1, 3, 7])
check("breakdown words", [c.word_count for c in bd], [3, 6, 5])
check("breakdown is count+front matter", len(bd), count_chapters(split) + 1)
check("breakdown sums to the whole text", sum(c.word_count for c in bd), words(split))
opens_on_chapter = "# Chapter One\n\nA b.\n"
check("no empty front matter",
      [c.title for c in chapter_breakdown(opens_on_chapter, words)], ["Chapter One"])
check("breakdown matches count",
      len(chapter_breakdown(opens_on_chapter, words)), count_chapters(opens_on_chapter))

print("== 3. typo scan ==")
book = "\n".join([
    "The rain fell hard and he walked down teh street before stopping to watch it fall today.",
    "",
    "# Chapter One: Beginnings",
    "",
    "She counted the windows of every house on the block and found them all dark.",
    "",
    "## Chapter Two: Middles",
    "",
    "Marcus waved at teh crowd and then went back inside the empty hall.",
])
rep = scan_typos(book, "en")
check("count", rep.count, 3)
check("available", rep.available, True)
check("note", rep.note, None)
check("returned", len(rep.typos), 3)
t0 = rep.typos[0]
check("word", t0.word, "teh")
check("chapter", t0.chapter, "Front Matter")
check("line_number", t0.line_number, 1)
check("context", t0.context, "...hard and he walked down teh street before stopping to watch...")
check("top suggestion", t0.suggestions[0], "the")
check("<=3 suggestions", len(t0.suggestions) <= 3, True)
check("not proper noun", t0.likely_proper_noun, False)
check("2nd word", rep.typos[1].word, "Marcus")
check("2nd proper noun", rep.typos[1].likely_proper_noun, True)
check("2nd chapter (md stripped)", rep.typos[1].chapter, "Chapter Two: Middles")
check("2nd line", rep.typos[1].line_number, 9)
check("3rd chapter", rep.typos[2].chapter, "Chapter Two: Middles")
check("digits not words", scan_typos("It was the 3rd day", "en").count, 0)
check("hyphens split", scan_typos("It was a well known page", "en").count, 0)
un = scan_typos("Καλημέρα κόσμε", "el")
check("unsupported available", un.available, False)
check("unsupported note", un.note, "No spellcheck dictionary is available for 'el'.")
cap = scan_typos("\n".join("he saw teh thing" for _ in range(250)), "en")
check("capped count", cap.count, 250)
check("capped list", len(cap.typos), MAX_TYPOS)
check("capped note", cap.note, f"Showing the first {MAX_TYPOS} of 250 findings.")

print("== 4. pipeline ==")
res = analyze_manuscript("book.md", book.encode())
check("fields", set(asdict(res)), {
    "word_count", "chapter_count", "chapters", "detected_language", "language_name",
    "text_direction", "script_type", "estimated_pages", "token_cost",
    "typo_count", "typos", "typo_check_available", "typo_check_note"})
check("words", res.word_count, len(book.split()))
check("chapters", res.chapter_count, 2)
check("chapter titles", [c.title for c in res.chapters], ["Front Matter", "Chapter One: Beginnings", "Chapter Two: Middles"])
check("chapter fields", set(asdict(res)["chapters"][0]), {"title", "line_number", "word_count"})
check("script", res.script_type, "latin")
check("typo_count", res.typo_count, 3)
check("typo dict fields", set(asdict(res)["typos"][0]),
      {"word", "suggestions", "chapter", "line_number", "context", "likely_proper_noun"})
try:
    analyze_manuscript("book.pdf", b"%PDF")
    check("pdf rejected", "no error", "UnsupportedFormatError")
except UnsupportedFormatError as e:
    print(f"PASS  pdf rejected: {e}")

print(f"== 5. API layer (pydantic {__import__('pydantic').VERSION}) ==")
r = asyncio.run(main.analyze_book(UploadFile("book.md", book.encode())))
check("model type", type(r).__name__, "BookAnalysis")
check("api typo_count", r.typo_count, 3)
check("api serialise", r.model_dump()["typos"][0]["word"], "teh")
check("api chapters serialise", r.model_dump()["chapters"][1]["title"], "Chapter One: Beginnings")
check("api chapter word count", r.model_dump()["chapters"][1]["word_count"] > 0, True)
check("api script", r.script_type, "latin")
ru = asyncio.run(main.analyze_book(UploadFile("книга.txt", "Привет мир как дела".encode())))
check("api accepts cyrillic", ru.script_type, "cyrillic")
check("api flag", ru.typo_check_available, False)
for name, data in (("empty.txt", b"  \n "), ("book.pdf", b"%PDF")):
    try:
        asyncio.run(main.analyze_book(UploadFile(name, data)))
        check(f"{name} rejected", "no error", "HTTPException")
    except HTTPException as e:
        check(f"{name} rejected 400", e.status_code, 400)

print()
print(f"{len(FAILS)} failure(s)" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
