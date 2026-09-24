"""Tests for the export engine that need no third-party libraries.

services/exporter.py imports weasyprint/ebooklib/python-docx lazily, so the
block parser, HTML renderer, stylesheet loader, RTF and TXT writers can all run
for real here. The PDF/EPUB/DOCX paths are covered only as far as their
graceful-degradation behaviour, since those libraries are not installed.

Run:  python3 verify_export.py     (from the project root)
"""

import sys

sys.path.insert(0, "/root/ebook-formatter")

from services import exporter

FAILS = []


def check(label, actual, expected):
    ok = actual == expected
    suffix = "" if ok else f"  WANT {expected!r}"
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {actual!r}{suffix}")
    if not ok:
        FAILS.append(label)


def ok(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'' if condition else f'  {detail}'}")
    if not condition:
        FAILS.append(label)


MANUSCRIPT = """A note from the publisher.

# Chapter One: Beginnings

The rain fell hard and he walked down the street.

She counted the windows and found them all dark.

***

## A Subsection

Some **bold** and *italic* text.

## Chapter Two: Middles

Marcus waved at the crowd.
"""

print("== 1. block parsing ==")
blocks = exporter.parse_blocks(MANUSCRIPT)
kinds = [(b.kind, b.text) for b in blocks]
check("block count", len(blocks), 7)
check("front matter is a paragraph", kinds[0][0], "p")
check("chapter one is h1", kinds[1], ("h1", "Chapter One: Beginnings"))
check("subsection is h2", kinds[4], ("h2", "A Subsection"))
check("scene break", kinds[3][0], "break")
check("chapter two is h1", kinds[6], ("h1", "Chapter Two: Middles"))

# A numbered heading with no markdown marker is still a chapter.
numbered = exporter.parse_blocks("12. The Return\n\nBody text here.\n")
check("numbered heading", numbered[0].kind, "h1")
check("numbered heading text", numbered[0].text, "12. The Return")

print("== 2. HTML rendering ==")
body = exporter.render_body_html(blocks)
ok("front matter section", '<section class="front-matter">' in body, body[:120])
ok("chapter heading", "<h1>Chapter One: Beginnings</h1>" in body)
ok("scene break markup", '<p class="break">* * *</p>' in body)
ok("bold converted", "<strong>bold</strong>" in body)
ok("italic converted", "<em>italic</em>" in body)
ok("html escaped", exporter.render_body_html(exporter.parse_blocks("a < b & c")) and "&lt;" in
   exporter.render_body_html(exporter.parse_blocks("a < b & c")))

request = exporter.ExportRequest(
    text=MANUSCRIPT, title="The Sample Book", author="A. Writer", genre="fiction"
)
document = exporter.render_document_html(request, blocks, "/* css */")
ok("doctype", document.startswith("<!DOCTYPE html>"))
ok("title in head", "<title>The Sample Book</title>" in document)
ok("genre attribute", 'data-genre="fiction"' in document)
ok("script attribute", 'data-script="latin"' in document)
ok("direction attribute", 'dir="ltr"' in document)
ok("book title element", 'class="book-title"' in document)

rtl_request = exporter.ExportRequest(
    text=MANUSCRIPT, title="كتاب", script_type="rtl", text_direction="rtl"
)
rtl_document = exporter.render_document_html(rtl_request, blocks, "")
ok("rtl dir attribute", 'dir="rtl"' in rtl_document)
ok("rtl script attribute", 'data-script="rtl"' in rtl_document)

print("== 3. stylesheet loading ==")
css = exporter.load_stylesheet(request)
ok("base sheet included", "@page :right" in css)
ok("genre sheet included", 'data-genre="fiction"' in css)

cjk = exporter.load_stylesheet(
    exporter.ExportRequest(text=MANUSCRIPT, script_type="cjk", genre="technical")
)
ok("script sheet included", 'data-script="cjk"' in cjk)
ok("cjk disables hyphens", "hyphens: none" in cjk)

for genre in exporter.GENRES:
    sheet = exporter.load_stylesheet(exporter.ExportRequest(text="x", genre=genre))
    ok(f"genre sheet exists and applies: {genre}", f'data-genre="{genre}"' in sheet)

print("== 4. runtime overrides and font sanitising ==")
styled = exporter.load_stylesheet(
    exporter.ExportRequest(text="x", custom_font="Garamond", font_size=12)
)
ok("custom font applied", '--book-font: "Garamond", serif;' in styled)
ok("font size applied", "--base-font-size: 12pt;" in styled)

# A font name is interpolated into a CSS declaration, so injection must fail.
for hostile in (
    'Evil"; } body { display: none; } .x {',
    "Bad</style><script>alert(1)</script>",
    "Font; color: red",
    "Font{}",
):
    cleaned = exporter._sanitize_font(hostile)
    ok(f"font sanitised: {hostile[:28]!r}", not any(c in cleaned for c in '{};<>"'), repr(cleaned))
    injected = exporter.load_stylesheet(
        exporter.ExportRequest(text="x", custom_font=hostile)
    )
    ok(f"no injection via font: {hostile[:28]!r}", "display: none" not in injected)

print("== 5. validation ==")
for bad, label in (
    ({"format": "mobi"}, "bad format"),
    ({"genre": "romance"}, "bad genre"),
    ({"script_type": "klingon"}, "bad script"),
    ({"text_direction": "upside-down"}, "bad direction"),
    ({"font_size": 400}, "font size too large"),
    ({"font_size": 0.5}, "font size too small"),
):
    try:
        exporter.export_book(exporter.ExportRequest(text="hello", **bad))
        check(f"{label} rejected", "no error", "ExportError")
    except exporter.ExportError as exc:
        ok(f"{label} rejected", isinstance(exc, exporter.UnsupportedExportError), type(exc).__name__)

try:
    exporter.export_book(exporter.ExportRequest(text="   "))
    check("empty text rejected", "no error", "ExportError")
except exporter.ExportError as exc:
    ok("empty text rejected", not isinstance(exc, exporter.UnsupportedExportError), type(exc).__name__)

print("== 6. graceful degradation for missing backends ==")
for fmt, package in (("pdf", "weasyprint"), ("epub", "ebooklib"), ("docx", "docx")):
    try:
        exporter.export_book(exporter.ExportRequest(text=MANUSCRIPT, format=fmt))
        print(f"      note: {fmt} backend IS installed, skipping degradation check")
    except exporter.BackendUnavailableError as exc:
        ok(f"{fmt} reports missing backend", package in str(exc).lower() or "pip install" in str(exc), str(exc)[:80])

print("== 7. RTF export ==")
rtf = exporter.export_rtf(exporter.ExportRequest(text=MANUSCRIPT, title="Book"))
ok("rtf is bytes", isinstance(rtf, bytes))
ok("rtf header", rtf.startswith(b"{\\rtf1"))
ok("rtf closed", rtf.rstrip().endswith(b"}"))
ok("rtf page size 6x9", b"\\paperw8640\\paperh12960" in rtf)
ok("rtf inner margin on left for ltr", b"\\margl1260\\margr720" in rtf)
text = rtf.decode("ascii", errors="replace")
ok("rtf has chapter text", "Chapter One" in text)
ok("rtf bold marker", "\\b " in text)

rtf_rtl = exporter.export_rtf(
    exporter.ExportRequest(text=MANUSCRIPT, text_direction="rtl", script_type="rtl")
)
ok("rtl gutter flips to the right", b"\\margl720\\margr1260" in rtf_rtl)
ok("rtl paragraph direction", b"\\rtlpar" in rtf_rtl)

# Non-ASCII must be encoded as \uN, and astral characters cannot be represented.
rtf_unicode = exporter.export_rtf(exporter.ExportRequest(text="Привет мир", title="T"))
ok("cyrillic escaped as \\uN", b"\\u1055?" in rtf_unicode, rtf_unicode[:120])

print("== 8. TXT export ==")
txt = exporter.export_txt(exporter.ExportRequest(text=MANUSCRIPT, title="Book"))
plain = txt.decode("utf-8")
ok("txt is utf-8 bytes", isinstance(txt, bytes))
ok("heading underlined", "Chapter One: Beginnings\n=====================" in plain)
ok("markers stripped", "**bold**" not in plain and "<" not in plain)
ok("emphasis text kept", "bold" in plain and "italic" in plain)
ok("scene break kept", "* * *" in plain)
ok("no lines over 72 chars", all(len(line) <= 72 for line in plain.splitlines()),
   max((len(line) for line in plain.splitlines()), default=0))

print("== 9. dispatch ==")
result = exporter.export_book(exporter.ExportRequest(text=MANUSCRIPT, format="txt", title="My Book!"))
check("media type", result.media_type, "text/plain; charset=utf-8")
check("filename", result.filename, "my-book.txt")
ok("content returned", len(result.content) > 0)

rtf_result = exporter.export_book(exporter.ExportRequest(text=MANUSCRIPT, format="rtf", title="كتاب عربي"))
check("rtf media type", rtf_result.media_type, "application/rtf")
check("non-latin title slugged to ascii", rtf_result.filename, "book.rtf")

print()
print(f"{len(FAILS)} failure(s)" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
