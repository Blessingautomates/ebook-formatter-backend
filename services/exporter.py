"""Render a manuscript into PDF, EPUB, DOCX, RTF or TXT.

Each backend is imported lazily, so a missing library disables only the formats
it serves -- the rest of the API keeps working and the caller gets a 503 naming
the missing package instead of the app failing to start.

On print resolution: a PDF produced here is vector, so "300 DPI" has no meaning
for text and rules -- it only describes embedded raster images, which WeasyPrint
places at their own native resolution. Supply images at 300 DPI and they stay at
300 DPI.

On trim size: the print templates are authored for 6x9in. Other trims are
applied as a generated override sheet (see _trim_css) for the paged formats.
EPUB is reflowable, so its trim has no meaning and is ignored there.
"""

from __future__ import annotations

import html
import importlib
import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from services.chapters import chapter_title, is_chapter_heading

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "templates" / "css"

GENRES = (
    "fiction",
    "non-fiction",
    "academic",
    "journal",
    "comic",
    "minimal",
    "poetry",
    "technical",
    "children",
)

EXPORT_FORMATS = ("pdf", "epub", "docx", "rtf", "txt")

SCRIPT_TYPES = ("latin", "rtl", "cjk", "cyrillic", "greek", "devanagari")
TEXT_DIRECTIONS = ("ltr", "rtl")

MEDIA_TYPES = {
    "pdf": "application/pdf",
    "epub": "application/epub+zip",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "rtf": "application/rtf",
    "txt": "text/plain; charset=utf-8",
}

_TWIPS_PER_INCH = 1440

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SCENE_BREAK_RE = re.compile(r"^(?:\*\s*\*\s*\*|\*\*\*+|⁂|-{3,}|_{3,})$")
_EMPHASIS_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_STRONG_RE = re.compile(r"\*\*(.+?)\*\*")
_SLUG_RE = re.compile(r"[^a-z0-9]+")

# A font name reaches a CSS declaration, so anything that could close the string
# or the rule is stripped rather than escaped.
_FONT_ALLOWED_RE = re.compile(r"[^A-Za-z0-9 \-]")

MIN_FONT_SIZE = 6.0
MAX_FONT_SIZE = 24.0

# The trim the print templates are authored for. This is the default value of
# ExportRequest.trim_size, so it has to be bound before that dataclass is
# defined: a default is evaluated while the class body runs, not at call time.
DEFAULT_TRIM = "6x9"


class ExportError(ValueError):
    """The requested export cannot be produced."""


class UnsupportedExportError(ExportError):
    """The format, genre, script or direction is not one we support."""


class BackendUnavailableError(ExportError):
    """A library needed for this format is not installed."""


@dataclass(frozen=True)
class ExportRequest:
    """Everything needed to render one book."""

    text: str
    title: str = "Untitled"
    author: str | None = None
    genre: str = "non-fiction"
    format: str = "pdf"
    script_type: str = "latin"
    text_direction: str = "ltr"
    language: str = "en"
    custom_font: str | None = None
    font_size: float | None = None
    trim_size: str = DEFAULT_TRIM


@dataclass(frozen=True)
class ExportResult:
    """A rendered book, ready to stream back to the caller."""

    content: bytes
    media_type: str
    filename: str


@dataclass(frozen=True)
class Block:
    """One structural unit of the manuscript."""

    kind: str      # "h1".."h6", "p" or "break"
    text: str


@dataclass(frozen=True)
class TrimSize:
    """A page size and the margins that suit it, in inches.

    `inner` is the binding edge and `outer` the trimmed edge; which side each
    falls on depends on whether the page is recto or verso, and flips again for
    a right-to-left book. The print templates are written for the 6x9 default,
    so any other trim is applied as a later override (see _trim_css).
    """

    width: float
    height: float
    inner: float
    outer: float
    top: float
    bottom: float


# The gutter scales with the page, so a narrower trim gets a slightly smaller
# binding margin rather than an unchanged one that would eat the text block.
TRIM_SIZES: dict[str, TrimSize] = {
    "6x9": TrimSize(width=6.0, height=9.0, inner=0.875, outer=0.5, top=0.75, bottom=0.7),
    "5.5x8.5": TrimSize(width=5.5, height=8.5, inner=0.8, outer=0.45, top=0.7, bottom=0.65),
    # A5 is 148x210mm.
    "a5": TrimSize(width=5.827, height=8.268, inner=0.85, outer=0.5, top=0.72, bottom=0.67),
}


# --------------------------------------------------------------------- public

def export_book(request: ExportRequest) -> ExportResult:
    """Render `request` into its target format.

    Raises UnsupportedExportError for a format we do not serve, and
    BackendUnavailableError when the library for that format is missing.
    """
    _validate(request)
    renderer = _RENDERERS[request.format]
    content = renderer(request)
    return ExportResult(
        content=content,
        media_type=MEDIA_TYPES[request.format],
        filename=f"{_slug(request.title)}.{request.format}",
    )


def _validate(request: ExportRequest) -> None:
    if request.format not in EXPORT_FORMATS:
        raise UnsupportedExportError(
            f"Unsupported format '{request.format}'. Supported: {', '.join(EXPORT_FORMATS)}."
        )
    if request.genre not in GENRES:
        raise UnsupportedExportError(
            f"Unsupported genre '{request.genre}'. Supported: {', '.join(GENRES)}."
        )
    if request.script_type not in SCRIPT_TYPES:
        raise UnsupportedExportError(
            f"Unsupported script_type '{request.script_type}'. "
            f"Supported: {', '.join(SCRIPT_TYPES)}."
        )
    if request.text_direction not in TEXT_DIRECTIONS:
        raise UnsupportedExportError(
            f"Unsupported text_direction '{request.text_direction}'. "
            f"Supported: {', '.join(TEXT_DIRECTIONS)}."
        )
    if request.trim_size not in TRIM_SIZES:
        raise UnsupportedExportError(
            f"Unsupported trim_size '{request.trim_size}'. "
            f"Supported: {', '.join(TRIM_SIZES)}."
        )
    if request.font_size is not None and not (
        MIN_FONT_SIZE <= request.font_size <= MAX_FONT_SIZE
    ):
        raise UnsupportedExportError(
            f"font_size must be between {MIN_FONT_SIZE:g} and {MAX_FONT_SIZE:g} points."
        )
    if not request.text.strip():
        raise ExportError("There is no manuscript content to export.")


# ---------------------------------------------------------------- manuscript

def parse_blocks(text: str) -> list[Block]:
    """Split the manuscript into headings, paragraphs and scene breaks.

    Headings are recognised the same way /api/analyze-book counts chapters, so
    the export and the analysis agree on where the chapter boundaries are.
    """
    blocks: list[Block] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            blocks.append(Block("p", " ".join(buffer).strip()))
            buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue

        heading = _MD_HEADING_RE.match(line)
        if heading:
            flush()
            level = min(len(heading.group(1)), 6)
            blocks.append(Block(f"h{level}", heading.group(2).strip()))
            continue

        if is_chapter_heading(line):
            flush()
            blocks.append(Block("h1", chapter_title(line)))
            continue

        if _SCENE_BREAK_RE.match(line):
            flush()
            blocks.append(Block("break", "* * *"))
            continue

        buffer.append(line)

    flush()
    return blocks


def _chapters(blocks: list[Block]) -> list[tuple[str, list[Block]]]:
    """Group blocks under their top-level heading, for the EPUB spine."""
    chapters: list[tuple[str, list[Block]]] = []
    current_title = "Front Matter"
    current: list[Block] = []

    for block in blocks:
        if block.kind == "h1":
            if current or chapters:
                chapters.append((current_title, current))
            current_title = block.text or current_title
            current = [block]
            continue
        current.append(block)

    if current:
        chapters.append((current_title, current))
    return [(title, items) for title, items in chapters if items]


def _inline_markup(text: str) -> str:
    """Escape `text` for HTML, then apply the inline emphasis marks."""
    escaped = html.escape(text)
    escaped = _STRONG_RE.sub(r"<strong>\1</strong>", escaped)
    return _EMPHASIS_RE.sub(r"<em>\1</em>", escaped)


def render_body_html(blocks: list[Block]) -> str:
    """The manuscript as HTML, with front matter marked for roman folios."""
    first_chapter = next(
        (index for index, block in enumerate(blocks) if block.kind == "h1"),
        len(blocks),
    )

    parts: list[str] = []
    if first_chapter:
        parts.append('<section class="front-matter">')
        parts.extend(_block_html(block) for block in blocks[:first_chapter])
        parts.append("</section>")
    parts.extend(_block_html(block) for block in blocks[first_chapter:])
    return "\n".join(parts)


def _block_html(block: Block) -> str:
    if block.kind == "break":
        return '<p class="break">* * *</p>'
    if block.kind.startswith("h"):
        return f"<{block.kind}>{_inline_markup(block.text)}</{block.kind}>"
    return f"<p>{_inline_markup(block.text)}</p>"


def render_document_html(request: ExportRequest, blocks: list[Block], css: str) -> str:
    """A complete standalone HTML document for the print and EPUB backends."""
    direction = request.text_direction
    author = (
        f'<p class="author">{html.escape(request.author)}</p>' if request.author else ""
    )
    return f"""<!DOCTYPE html>
<html lang="{html.escape(request.language)}" dir="{direction}">
<head>
<meta charset="utf-8">
<title>{html.escape(request.title)}</title>
<style>
{css}
</style>
</head>
<body>
<div class="book" data-genre="{request.genre}" data-script="{request.script_type}" dir="{direction}">
<section class="title-page">
<h1 class="book-title">{html.escape(request.title)}</h1>
{author}
</section>
{render_body_html(blocks)}
</div>
</body>
</html>
"""


# ----------------------------------------------------------------------- CSS

def load_stylesheet(request: ExportRequest) -> str:
    """Base sheet, then the script rules, then the genre sheet, then overrides."""
    parts = [
        _read_template("base-print.css"),
        _read_template("scripts.css"),
        _read_template(f"genres/{request.genre}.css"),
        _runtime_css(request),
    ]
    return "\n\n".join(part for part in parts if part)


def _read_template(relative_path: str) -> str:
    path = TEMPLATE_DIR / relative_path
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"Missing print template {relative_path}: {exc}") from exc


def _runtime_css(request: ExportRequest) -> str:
    """The per-book overrides: page geometry, font, size, page-number face."""
    parts: list[str] = []

    trim = _trim_css(request)
    if trim:
        parts.append(trim)

    declarations: list[str] = []
    if request.custom_font:
        font = _sanitize_font(request.custom_font)
        if font:
            # The family is quoted and the fallbacks are kept, so an unknown or
            # unavailable font degrades to the genre's own stack.
            declarations.append(f'  --book-font: "{font}", serif;')
            declarations.append(f'  --running-head-font: "{font}", serif;')
    if request.font_size:
        declarations.append(f"  --base-font-size: {request.font_size:g}pt;")

    if declarations:
        parts.append(":root {\n" + "\n".join(declarations) + "\n}")

    return "\n\n".join(parts)


def _trim_css(request: ExportRequest) -> str:
    """Page-geometry overrides for a trim other than the 6x9 default.

    These come last in the sheet, so they win over base-print.css at equal
    specificity. `size` needs setting only on the unnamed @page rule: the named
    page variants (rtl-book, caseless-book, front-matter) inherit it and differ
    only in which edge the gutter falls on.
    """
    if request.trim_size == DEFAULT_TRIM:
        return ""

    trim = TRIM_SIZES[request.trim_size]
    inner, outer = trim.inner, trim.outer
    vertical = f"margin-top: {trim.top}in; margin-bottom: {trim.bottom}in;"

    return "\n".join(
        [
            "@page {",
            f"  size: {trim.width}in {trim.height}in;",
            f"  margin: {trim.top}in {outer}in {trim.bottom}in {outer}in;",
            "}",
            # Recto binds left, verso binds right; the RTL variants are the
            # mirror of each, as in the base sheet.
            f"@page :right {{ {vertical} margin-left: {inner}in; margin-right: {outer}in; }}",
            f"@page :left {{ {vertical} margin-left: {outer}in; margin-right: {inner}in; }}",
            f"@page rtl-book:right {{ {vertical} margin-left: {outer}in; margin-right: {inner}in; }}",
            f"@page rtl-book:left {{ {vertical} margin-left: {inner}in; margin-right: {outer}in; }}",
            f"@page caseless-book:right {{ {vertical} margin-left: {inner}in; margin-right: {outer}in; }}",
            f"@page caseless-book:left {{ {vertical} margin-left: {outer}in; margin-right: {inner}in; }}",
        ]
    )


def _sanitize_font(name: str) -> str:
    """Reduce a caller-supplied font name to something safe in a declaration.

    A font name is interpolated into CSS, so quotes, braces, semicolons and
    anything else that could close the string or the rule are removed.
    """
    cleaned = _FONT_ALLOWED_RE.sub("", unicodedata.normalize("NFKC", name))
    return " ".join(cleaned.split())[:64]


# ----------------------------------------------------------------------- PDF

def export_pdf(request: ExportRequest) -> bytes:
    weasyprint = _require(
        "weasyprint", "PDF export needs WeasyPrint (pip install weasyprint)"
    )
    blocks = parse_blocks(request.text)
    document = render_document_html(request, blocks, load_stylesheet(request))

    rendered = weasyprint.HTML(string=document, base_url=str(BASE_DIR)).render()
    return rendered.write_pdf()


# ---------------------------------------------------------------------- EPUB

def export_epub(request: ExportRequest) -> bytes:
    epub = _require(
        "ebooklib", "EPUB export needs ebooklib (pip install EbookLib)"
    )
    from ebooklib import epub as epub_module

    blocks = parse_blocks(request.text)
    book = epub_module.EpubBook()
    book.set_identifier(f"urn:uuid:{_slug(request.title)}-{abs(hash(request.text)):x}")
    book.set_title(request.title)
    book.set_language(request.language)
    if request.author:
        book.add_author(request.author)

    # A right-to-left book declares its page progression so readers open it the
    # correct way round.
    if request.text_direction == "rtl":
        book.set_direction("rtl")

    style = epub_module.EpubItem(
        uid="style",
        file_name="style/book.css",
        media_type="text/css",
        content=load_stylesheet(request),
    )
    book.add_item(style)

    chapters = []
    for index, (title, chapter_blocks) in enumerate(_chapters(blocks), start=1):
        item = epub_module.EpubHtml(
            title=title or f"Chapter {index}",
            file_name=f"chapter_{index:03d}.xhtml",
            lang=request.language,
        )
        body = "\n".join(_block_html(block) for block in chapter_blocks)
        item.content = (
            f'<div class="book" data-genre="{request.genre}" '
            f'data-script="{request.script_type}" dir="{request.text_direction}">'
            f"{body}</div>"
        )
        item.add_item(style)
        book.add_item(item)
        chapters.append(item)

    book.toc = tuple(chapters)
    book.add_item(epub_module.EpubNcx())
    book.add_item(epub_module.EpubNav())
    book.spine = ["nav", *chapters]

    buffer = io.BytesIO()
    epub_module.write_epub(buffer, book)
    return buffer.getvalue()


# ---------------------------------------------------------------------- DOCX

def export_docx(request: ExportRequest) -> bytes:
    docx = _require(
        "docx", "DOCX export needs python-docx (pip install python-docx)"
    )
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    document = docx.Document()
    section = document.sections[0]
    trim = TRIM_SIZES[request.trim_size]
    section.page_width = Inches(trim.width)
    section.page_height = Inches(trim.height)
    # Mirrored margins are not exposed by python-docx, so the gutter is applied
    # to both sides; Word's own "mirror margins" setting can adjust it.
    section.left_margin = Inches(trim.inner)
    section.right_margin = Inches(trim.inner)
    section.top_margin = Inches(trim.top)
    section.bottom_margin = Inches(trim.bottom)

    base_style = document.styles["Normal"]
    base_style.font.name = request.custom_font or "Georgia"
    base_style.font.size = Pt(request.font_size or 11)

    alignment = (
        WD_ALIGN_PARAGRAPH.RIGHT
        if request.text_direction == "rtl"
        else WD_ALIGN_PARAGRAPH.JUSTIFY
    )

    for block in parse_blocks(request.text):
        if block.kind.startswith("h"):
            level = int(block.kind[1])
            heading = document.add_heading(block.text, level=min(level, 4))
            heading.alignment = (
                WD_ALIGN_PARAGRAPH.RIGHT
                if request.text_direction == "rtl"
                else WD_ALIGN_PARAGRAPH.LEFT
            )
            continue
        paragraph = document.add_paragraph(_plain_text(block.text))
        paragraph.alignment = alignment

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# ----------------------------------------------------------------------- RTF

def export_rtf(request: ExportRequest) -> bytes:
    """Write RTF directly: the format is simple enough not to need a library."""
    rtl = request.text_direction == "rtl"
    font = _sanitize_font(request.custom_font or "") or "Georgia"
    font_size = int((request.font_size or 11) * 2)   # RTF sizes are half-points
    trim = TRIM_SIZES[request.trim_size]

    header = [
        r"{\rtf1\ansi\deff0\uc1",
        r"{\fonttbl{\f0 " + _rtf_escape(font) + r";}}",
        r"{\colortbl;\red17\green17\blue17;}",
        f"\\paperw{_twips(trim.width)}\\paperh{_twips(trim.height)}",
    ]

    # The gutter sits on the binding edge, which is the right for an RTL book.
    if rtl:
        header.append(f"\\margl{_twips(trim.outer)}\\margr{_twips(trim.inner)}")
    else:
        header.append(f"\\margl{_twips(trim.inner)}\\margr{_twips(trim.outer)}")
    header.append(f"\\margt{_twips(trim.top)}\\margb{_twips(trim.bottom)}")
    if rtl:
        header.append(r"\rtlpar\qr")

    body: list[str] = []
    for block in parse_blocks(request.text):
        text = _rtf_escape(_plain_text(block.text))
        if block.kind == "break":
            body.append(r"\pard\qc\f0\fs" + str(font_size) + r" * * *\par")
            continue
        if block.kind.startswith("h"):
            level = int(block.kind[1])
            size = max(font_size + (14 - level * 2), font_size)
            body.append(
                r"\pard\sb240\sa120\b\f0\fs"
                + str(size)
                + " "
                + text
                + r"\b0\par"
            )
            continue
        body.append(r"\pard\fi360\f0\fs" + str(font_size) + " " + text + r"\par")

    document = "\n".join(header) + "\n" + "\n".join(body) + "\n}"
    return document.encode("ascii", errors="replace")


def _rtf_escape(text: str) -> str:
    """Escape RTF control characters and encode everything above ASCII."""
    out: list[str] = []
    for character in text:
        code = ord(character)
        if character in "\\{}":
            out.append("\\" + character)
        elif code < 128:
            out.append(character)
        elif code <= 0xFFFF:
            # RTF wants a signed 16-bit value in \uN.
            signed = code if code < 0x8000 else code - 0x10000
            out.append(f"\\u{signed}?")
        else:
            out.append("?")   # astral plane: not representable in RTF
    return "".join(out)


# ----------------------------------------------------------------------- TXT

def export_txt(request: ExportRequest) -> bytes:
    """Plain text, with the structure the markers implied made explicit."""
    lines: list[str] = []
    for block in parse_blocks(request.text):
        if block.kind == "break":
            lines.append("* * *")
        elif block.kind == "h1":
            lines.append(_underline(block.text))
        elif block.kind.startswith("h"):
            lines.append(_plain_text(block.text).upper() if block.kind == "h2" else _plain_text(block.text))
        else:
            lines.append(_wrap(_plain_text(block.text)))
        lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _underline(text: str) -> str:
    line = _plain_text(text)
    return f"{line}\n{'=' * len(line)}"


def _wrap(text: str, width: int = 72) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        addition = len(word) + (1 if current else 0)
        if length + addition > width and current:
            lines.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += addition
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


# -------------------------------------------------------------------- shared

def _twips(inches: float) -> int:
    """Inches to twips, the twentieth-of-a-point unit RTF measures in."""
    return int(round(inches * _TWIPS_PER_INCH))


def _plain_text(text: str) -> str:
    """Drop the inline emphasis markers, keeping the words."""
    return _STRONG_RE.sub(r"\1", _EMPHASIS_RE.sub(r"\1", text))


def _slug(title: str) -> str:
    slug = _SLUG_RE.sub("-", title.strip().lower()).strip("-")
    return slug[:64] or "book"


def _require(module_name: str, hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise BackendUnavailableError(f"{hint}. ({exc})") from exc


_RENDERERS = {
    "pdf": export_pdf,
    "epub": export_epub,
    "docx": export_docx,
    "rtf": export_rtf,
    "txt": export_txt,
}
