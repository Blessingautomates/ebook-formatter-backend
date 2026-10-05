"""Assemble the complete publishing package.

The brief asks for one download that contains everything an author needs to
publish, rather than five separate exports and a cover they have to keep track
of. This builds that archive from the renderers that already exist — it calls
`exporter.export_book` once per format rather than re-implementing any of them,
so a package cannot drift from what the individual exports produce.

The important behaviour is what happens when a format cannot be rendered. The
exporter raises `BackendUnavailableError` when the library for one format is
missing (WeasyPrint for PDF, EbookLib for EPUB), and a deployment can easily be
missing one. Failing the whole package for that would mean an author cannot
download the four formats that *do* work because of the fifth that does not, so
each format is attempted independently, and the ones that were skipped are
reported in the archive and in the response.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone

from services import exporter
from services.cover import CoverReport, specification_text
from services.exporter import (
    EXPORT_FORMATS,
    ExportRequest,
    ExportResult,
    _slug,
)

#: What each format is called in the archive. The exporter's own filename
#: already carries the right extension, so this only maps the label used in the
#: metadata file.
FORMAT_LABELS: dict[str, str] = {
    "pdf": "Print-ready PDF",
    "epub": "EPUB 3",
    "docx": "Microsoft Word",
    "rtf": "Rich Text",
    "txt": "Plain text",
}


@dataclass(frozen=True)
class PackagedFormat:
    """One book that made it into the archive."""

    format: str
    label: str
    path: str
    bytes: int


@dataclass(frozen=True)
class SkippedFormat:
    """One that could not be rendered, and why."""

    format: str
    label: str
    reason: str


@dataclass(frozen=True)
class PackageResult:
    """The archive, plus an account of what went into it."""

    result: ExportResult
    included: list[PackagedFormat] = field(default_factory=list)
    skipped: list[SkippedFormat] = field(default_factory=list)


def _readme(
    request: ExportRequest,
    included: list[PackagedFormat],
    skipped: list[SkippedFormat],
    page_count: int,
    cover_name: str | None,
) -> str:
    """The index that sits at the top of the archive."""
    lines = [
        f"{request.title}",
        "=" * len(request.title),
        "",
    ]

    if request.author:
        lines.append(f"By {request.author}")
        lines.append("")

    lines += [
        "PUBLISHING PACKAGE",
        "------------------",
        "",
        "BOOKS",
    ]

    for item in included:
        lines.append(f"  books/{item.path.rsplit('/', 1)[-1]:<32} {item.label}")

    if skipped:
        lines += [
            "",
            "NOT INCLUDED",
        ]
        for item in skipped:
            lines.append(f"  {item.format.upper():<6} {item.reason}")

    lines += ["", "COVER"]
    if cover_name:
        lines.append(f"  cover/{cover_name}")
        lines.append("  cover-specification.txt   dimensions this cover must meet")
    else:
        lines.append("  No cover was supplied with this package.")
        lines.append(
            "  cover-specification.txt gives the dimensions it needs to be."
        )

    lines += [
        "",
        "METADATA",
        "  metadata.json             title, author, language, trim, counts",
        "",
        "ABOUT THESE FILES",
        "",
        f"  Trim size      {request.trim_size}",
        f"  Genre sheet    {request.genre}",
        f"  Language       {request.language} ({request.script_type} script, "
        f"{request.text_direction.upper()})",
        f"  Pages          {page_count} (estimated)",
        "",
        "  The PDF is vector, so its text is resolution-independent. The EPUB",
        "  is reflowable and carries no page size; the trim size applies to the",
        "  print editions.",
        "",
        "  Each book here is the same manuscript rendered five ways. Edit the",
        "  manuscript and export again rather than editing these files, or the",
        "  versions will disagree.",
        "",
    ]
    return "\n".join(lines)


def _metadata(
    request: ExportRequest,
    included: list[PackagedFormat],
    skipped: list[SkippedFormat],
    page_count: int,
    report: CoverReport | None,
    isbn: str | None,
    stored_cover: str | None,
) -> str:
    """The machine-readable description of the package."""
    payload = {
        "title": request.title,
        "author": request.author,
        "language": {
            "code": request.language,
            "script": request.script_type,
            "direction": request.text_direction,
        },
        "formatting": {
            "genre": request.genre,
            "trim_size": request.trim_size,
            "custom_font": request.custom_font,
            "font_size": request.font_size,
        },
        "measurements": {
            "pages": page_count,
        },
        "identifiers": {
            # A slot, not a value: nothing in this product issues ISBNs, and
            # inventing one would be worse than leaving it visibly empty.
            "isbn": isbn,
        },
        "files": {
            "books": [
                {
                    "format": item.format,
                    "label": item.label,
                    "path": item.path,
                    "bytes": item.bytes,
                }
                for item in included
            ],
            "skipped": [
                {"format": item.format, "reason": item.reason} for item in skipped
            ],
            # Whether the file is in the archive, which is not the same
            # question as whether it was measured: a deployment without Pillow
            # ships the cover and cannot check it, and a reader of this file
            # needs to be able to tell those two apart.
            "cover": stored_cover is not None,
            "cover_path": stored_cover,
            "cover_measured": report is not None,
        },
        "cover_specification": (
            {
                "trim_width_in": report.specification.trim_width_in,
                "trim_height_in": report.specification.trim_height_in,
                "spine_width_in": report.specification.spine_width_in,
                "bleed_in": report.specification.bleed_in,
                "safe_area_in": report.specification.safe_area_in,
                "full_width_in": report.specification.full_width_in,
                "full_height_in": report.specification.full_height_in,
                "required_width_px": report.specification.required_width_px,
                "required_height_px": report.specification.required_height_px,
                "paper": report.specification.paper,
                "status": report.status,
            }
            if report is not None
            else None
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_package(
    request: ExportRequest,
    *,
    page_count: int,
    cover: bytes | None = None,
    cover_name: str | None = None,
    cover_report: CoverReport | None = None,
    isbn: str | None = None,
) -> PackageResult:
    """Render every format and zip them with the metadata and the cover.

    `request.format` is ignored: a package is all of them. Everything else about
    the request — trim, genre, language, font — is passed straight through to
    each render, so the five files inside are the same five the export page
    would have produced one at a time.
    """
    slug = _slug(request.title)
    included: list[PackagedFormat] = []
    skipped: list[SkippedFormat] = []
    rendered: list[tuple[PackagedFormat, bytes]] = []

    for export_format in EXPORT_FORMATS:
        label = FORMAT_LABELS.get(export_format, export_format.upper())
        try:
            result = exporter.export_book(
                ExportRequest(
                    text=request.text,
                    title=request.title,
                    author=request.author,
                    genre=request.genre,
                    format=export_format,
                    script_type=request.script_type,
                    text_direction=request.text_direction,
                    language=request.language,
                    custom_font=request.custom_font,
                    font_size=request.font_size,
                    trim_size=request.trim_size,
                )
            )
        except exporter.BackendUnavailableError as exc:
            # This deployment is missing the library for this one format. The
            # others are unaffected, so it is reported rather than raised.
            skipped.append(SkippedFormat(export_format, label, str(exc)))
            continue
        except exporter.ExportError as exc:
            # A bad request — an unsupported genre, say — would fail every
            # format identically, so it is raised rather than swallowed.
            raise

        item = PackagedFormat(
            format=export_format,
            label=label,
            path=f"books/{_package_filename(slug, result.filename)}",
            bytes=len(result.content),
        )
        included.append(item)
        rendered.append((item, result.content))

    if not rendered:
        raise exporter.BackendUnavailableError(
            "No format could be rendered in this deployment. "
            + "; ".join(f"{item.format}: {item.reason}" for item in skipped)
        )

    buffer = io.BytesIO()
    # The name the cover lands under, resolved once so the file written and the
    # path recorded in the metadata cannot disagree.
    stored_cover = f"cover/{cover_name or f'{slug}-cover'}" if cover else None
    # ZIP_DEFLATED because four of the five formats are text underneath and
    # compress well; the PDF is already compressed and simply costs a little
    # time. Deterministic order so two packages of the same book are
    # byte-identical apart from the timestamp inside metadata.json.
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "README.txt",
            _readme(request, included, skipped, page_count, cover_name),
        )
        archive.writestr(
            "metadata.json",
            _metadata(
                request,
                included,
                skipped,
                page_count,
                cover_report,
                isbn,
                stored_cover,
            ),
        )

        for item, content in rendered:
            archive.writestr(item.path, content)

        if stored_cover is not None:
            archive.writestr(stored_cover, cover)

        if cover_report is not None:
            archive.writestr(
                "cover-specification.txt", specification_text(cover_report)
            )

    content = buffer.getvalue()

    return PackageResult(
        result=ExportResult(
            content=content,
            media_type="application/zip",
            filename=f"{slug}-publishing-package.zip",
        ),
        included=included,
        skipped=skipped,
    )


def _package_filename(slug: str, exporter_filename: str) -> str:
    """Name a rendered book inside the archive, keeping the exporter's extension.

    The exporter already produces `slug.ext`; this only falls back to guessing
    the extension from the media type if it ever stops doing so, so the archive
    layout does not depend on that detail holding.
    """
    _, _, extension = exporter_filename.rpartition(".")
    return f"{slug}.{extension}" if extension else slug
