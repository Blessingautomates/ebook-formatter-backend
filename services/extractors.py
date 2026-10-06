"""Text extraction for the manuscript formats the platform accepts."""

from __future__ import annotations

import codecs
import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from docx import Document

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".docx", ".epub", ".md", ".txt"})

_HEADING_LEVEL_RE = re.compile(r"(\d+)")

# XHTML in an uploaded EPUB is untrusted XML. Two attacks matter and both are
# already closed by the standard library, so the safety here is in *not*
# switching them off:
#
#   * External entities (XXE): ElementTree's expat parser is created without
#     external entity resolution, so a DOCTYPE pointing at file:///etc/passwd
#     is never fetched.
#   * Entity expansion ("billion laughs"): ElementTree installs its own
#     EntityDeclHandler that raises ParseError on any entity declaration, so
#     the bomb never expands.
#
# Both would be reopened by installing a permissive handler on the parser, so
# this module installs none. Only headings and text are read out.

# HTML elements that end a block of text. EPUB chapter files are a single line
# of markup, so without inserting breaks at these the whole chapter arrives as
# one enormous line and the line-oriented chapter counter sees nothing.
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "br", "li", "tr", "td", "th", "section", "article",
        "blockquote", "figure", "figcaption", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "footer", "aside", "nav", "pre", "hr", "dt", "dd",
    }
)

#: Elements whose text is never manuscript content: `script` and `style` hold
#: code, and the rest are document furniture. Without this, `<script>` is a leaf
#: with text and would be emitted as though it were prose.
_SKIP_TAGS = frozenset({"script", "style", "head", "title", "meta", "link"})

_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


class ManuscriptError(ValueError):
    """Base class for an uploaded file the platform cannot read."""


class UnsupportedFormatError(ManuscriptError):
    """The file is not one of the supported manuscript formats."""


class CorruptDocumentError(ManuscriptError):
    """The file claims a supported format but could not be parsed."""


def extract_text(filename: str, data: bytes) -> str:
    """Return the plain text of `data`, dispatching on `filename`'s extension."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        shown = extension or (filename or "the uploaded file")
        raise UnsupportedFormatError(
            f"Unsupported file type '{shown}'. Supported types: {supported}."
        )
    if extension == ".docx":
        return _extract_docx(data)
    if extension == ".epub":
        return _extract_epub(data)
    return _decode_text(data)


def _decode_text(data: bytes) -> str:
    """Decode a .md/.txt payload, tolerating the encodings manuscripts arrive in."""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16")
    # UTF-8 first (it is the common case and rejects invalid input rather than
    # mangling it), then the Windows codepage, then latin-1, which never fails.
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise CorruptDocumentError("Could not decode the manuscript text.")


def _extract_docx(data: bytes) -> str:
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:  # python-docx raises a range of errors for bad input
        raise CorruptDocumentError(f"Could not read the .docx file: {exc}") from exc

    blocks: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append(_with_heading_marker(paragraph, text))
    return "\n\n".join(blocks)


def _with_heading_marker(paragraph, text: str) -> str:
    """Render a Word heading as a Markdown heading so chapter detection sees it.

    Word marks headings with a paragraph style rather than a marker character,
    so without this a .docx manuscript would look like an unbroken run of body
    paragraphs to the line-oriented chapter counter.
    """
    style = getattr(paragraph, "style", None)
    style_name = (getattr(style, "name", "") or "").strip()
    if not style_name.lower().startswith("heading"):
        return text
    match = _HEADING_LEVEL_RE.search(style_name)
    level = min(int(match.group(1)), 6) if match else 1
    return f"{'#' * level} {text}"


def _extract_epub(data: bytes) -> str:
    """Read an EPUB's spine in order and return it as Markdown-ish text.

    The spine is the reading order — the manifest is a bag of files and walking
    it alphabetically would scramble the book. Each document's `<h1>`/`<h2>`
    becomes a `#`/`##` heading so `is_chapter_heading` recognises it without
    changes, and every other block element ends a line so paragraphs stay
    separate. Without those breaks an EPUB chapter arrives as one unbroken line
    of text and the whole layout pipeline has nothing to work with.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise CorruptDocumentError("Could not read the .epub file: it is not a valid archive.") from exc

    try:
        with archive:
            documents = _epub_spine(archive)
            if not documents:
                raise CorruptDocumentError(
                    "The .epub file contains no readable text documents."
                )
            blocks: list[str] = []
            for name in documents:
                try:
                    markup = archive.read(name)
                except KeyError:
                    # A spine entry pointing at a file the archive does not
                    # contain. Skipping it keeps the rest of the book readable.
                    continue
                blocks.extend(_html_blocks(markup))
    except CorruptDocumentError:
        raise
    except Exception as exc:  # zipfile and ElementTree raise a range of errors
        raise CorruptDocumentError(f"Could not read the .epub file: {exc}") from exc

    return "\n\n".join(blocks)


# Namespace prefixes differ between EPUB producers, so elements are matched on
# their local name rather than a fully qualified one.
def _local_name(tag: str) -> str:
    """The element name without its `{namespace}` prefix."""
    return tag.rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _epub_spine(archive: zipfile.ZipFile) -> list[str]:
    """The XHTML documents of an EPUB, in reading order.

    Falls back to scanning the archive when the package document cannot be
    read, so an EPUB with a malformed OPF still yields its text rather than
    nothing at all.
    """
    try:
        container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        rootfile = next(
            (
                element.get("full-path")
                for element in container.iter()
                if _local_name(element.tag) == "rootfile" and element.get("full-path")
            ),
            None,
        )
    except (KeyError, ElementTree.ParseError):
        rootfile = None

    if rootfile:
        try:
            package = ElementTree.fromstring(archive.read(rootfile))
        except (KeyError, ElementTree.ParseError):
            package = None
        if package is not None:
            documents = _spine_from_package(package, rootfile)
            if documents:
                return documents

    return sorted(
        name
        for name in archive.namelist()
        if name.lower().endswith((".xhtml", ".html", ".htm"))
    )


def _spine_from_package(package, rootfile: str) -> list[str]:
    """Resolve the OPF's spine ids against its manifest, in spine order."""
    manifest: dict[str, str] = {}
    for element in package.iter():
        if _local_name(element.tag) == "item" and element.get("id"):
            href = element.get("href")
            if href:
                manifest[element.get("id")] = href

    # The OPF lives in a subdirectory in most EPUBs; hrefs are relative to it.
    base = Path(rootfile).parent

    documents: list[str] = []
    for element in package.iter():
        if _local_name(element.tag) != "itemref":
            continue
        href = manifest.get(element.get("idref") or "")
        if not href:
            continue
        resolved = str((base / href).as_posix()) if str(base) != "." else href
        documents.append(resolved)
    return documents


def _html_blocks(markup: bytes) -> list[str]:
    """Turn one XHTML document into a list of text blocks.

    Headings come back with their Markdown marker; everything else is a plain
    paragraph. Text is taken per *block*, not per element, so a sentence broken
    up by `<em>` or a link arrives whole rather than in fragments — and a `<p>`
    nested inside a `<div>` is emitted once, by the `<p>`, rather than twice.
    """
    try:
        root = ElementTree.fromstring(markup)
    except ElementTree.ParseError:
        # Not well-formed XML. Many EPUBs in the wild are really HTML, so a
        # tolerant second pass is worth attempting before giving up on the file.
        return _html_blocks_lenient(markup)

    blocks: list[str] = []
    _collect_blocks(root, blocks)
    return blocks


def _normalise(text: str | None) -> str:
    """Collapse runs of whitespace, as XHTML source is mostly indentation."""
    return re.sub(r"\s+", " ", text or "").strip()


def _collect_blocks(element, blocks: list[str]) -> None:
    """Append `element`'s text blocks, in document order."""
    name = _local_name(element.tag)
    if name in _SKIP_TAGS:
        return

    if name in _HEADING_TAGS:
        text = _element_text(element)
        if text:
            blocks.append(f"{'#' * _HEADING_TAGS[name]} {text}")
        return

    children = list(element)
    if not children:
        # A leaf. It is reached either directly, as a paragraph, or by
        # descending through a wrapper. Either way its text is manuscript
        # content, so it is emitted: `<body>Bare text</body>` has no block-level
        # child to carry it and must not come back empty.
        text = _element_text(element)
        if text:
            blocks.append(text)
        return

    if not any(_local_name(child.tag) in _BLOCK_TAGS for child in children):
        if name in _BLOCK_TAGS:
            # A single block broken up by inline markup:
            # `<p>He was <em>very</em> tired.</p>` is one sentence, not three
            # fragments.
            text = _element_text(element)
            if text:
                blocks.append(text)
            return
        # A wrapper that is not itself a block — `<html>`, `<body>`, `<table>`,
        # `<ul>`. Its children are containers or inline markup, so descend.
        # Returning here instead is how a whole EPUB imports as nothing: the
        # book's text is inside `<body>`, which is not a block tag.
        for child in children:
            _collect_blocks(child, blocks)
        return

    # A container with real blocks inside it. Its own direct text is a paragraph
    # before its children — the mixed-content case, `<div>Lead-in<p>then a
    # paragraph</p>` — so it is emitted first.
    own = _normalise(element.text)
    if own:
        blocks.append(own)

    for child in children:
        child_name = _local_name(child.tag)
        _collect_blocks(child, blocks)
        # Text after a block child sits between paragraphs rather than inside
        # one, so it becomes its own block.
        if child_name in _BLOCK_TAGS:
            tail = _normalise(child.tail)
            if tail:
                blocks.append(tail)


def _element_text(element) -> str:
    """All the text under `element`, with runs of whitespace collapsed."""
    parts: list[str] = []
    for piece in element.itertext():
        if piece:
            parts.append(piece)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _html_blocks_lenient(markup: bytes) -> list[str]:
    """Last-resort extraction for XHTML that is not well-formed XML.

    Tags are stripped and the remaining text is split on blank lines. This
    loses heading levels, so a book that only parses this way may show fewer
    chapters — which is why it is the fallback and not the first choice.
    """
    try:
        text = markup.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - decode with errors= cannot raise
        return []

    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
    # Block-level ends become line breaks; every other tag is simply removed.
    text = re.sub(
        r"(?i)</?(?:p|div|br|li|tr|h[1-6]|section|article|blockquote)\b[^>]*>",
        "\n",
        text,
    )
    text = re.sub(r"<[^>]+>", "", text)
    text = _unescape(text)
    return [line.strip() for line in text.splitlines() if line.strip()]


def _unescape(text: str) -> str:
    """Resolve the five XML entities that survive tag stripping."""
    for entity, char in (
        ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
        ("&apos;", "'"), ("&nbsp;", " "), ("&amp;", "&"),
    ):
        text = text.replace(entity, char)
    return text
