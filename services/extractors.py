"""Text extraction for the manuscript formats the platform accepts."""

from __future__ import annotations

import codecs
import io
import re
from pathlib import Path

from docx import Document

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".docx", ".md", ".txt"})

_HEADING_LEVEL_RE = re.compile(r"(\d+)")


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
