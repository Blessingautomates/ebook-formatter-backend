"""Cover validation against a print-on-demand specification.

Every check here is a measurement, not a decoration. The brief asks for
dimensions, DPI, spine width, bleed and safe area, and each of those is
answerable from the file itself:

* The pixel dimensions come from the image.
* The effective DPI is computed from the pixels and the size the cover has to
  be — ``width_px / required_width_in`` — rather than read from metadata. That
  matters because the metadata is frequently absent (a PNG re-exported from a
  design tool often carries none) or wrong (a 72-DPI tag on a file that is
  plainly 3000 px wide), and the pixels cannot lie about how much detail there
  is.
* The spine width follows from the page count and the paper stock.
* Bleed and safe area are then constraints on how large the file has to be.

What is *not* checked is whether the artwork actually fills the bleed, or
whether anything important sits inside the safe area. A flat raster cannot say:
an image that is entirely white in its outer eighth of an inch passes every
size check. So the report says what was measured and the UI draws the guide
lines, rather than claiming a judgement the file cannot support.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from services.exporter import TRIM_SIZES, ExportError

#: Paper thickness in inches per page, as used by Kindle Direct Publishing for
#: its standard stocks. Black-and-white printing on white or cream, and colour.
PAPER_THICKNESS: dict[str, float] = {
    "white": 0.002252,
    "cream": 0.0025,
    "colour": 0.002347,
}

DEFAULT_PAPER = "white"

#: How far artwork must extend past the trim edge, in inches.
BLEED_IN = 0.125

#: How far inside the trim edge text must stay, in inches.
SAFE_AREA_IN = 0.25

#: The DPI a print-ready cover needs, and the floor below which it is rejected.
TARGET_DPI = 300
MIN_DPI = 200

#: A file within this much of the required size is treated as exact. Design
#: tools round to whole pixels, and one pixel over is not a defect.
DIMENSION_TOLERANCE = 0.01

#: The largest upload accepted, in bytes. A print-ready cover is a few tens of
#: megabytes; beyond this it is a mistake or an attack.
MAX_COVER_BYTES = 40 * 1024 * 1024


class CoverError(ExportError):
    """The uploaded file is not an image, or cannot be read as one."""


@dataclass(frozen=True)
class CoverCheck:
    """One measurement, and what it was measured against."""

    id: str
    label: str
    #: "pass", "warn" or "fail" — the three states the drawer renders.
    status: str
    measured: str
    required: str
    message: str


@dataclass(frozen=True)
class CoverSpecification:
    """The dimensions the cover has to be, before any file is compared to them."""

    trim_width_in: float
    trim_height_in: float
    spine_width_in: float
    bleed_in: float
    safe_area_in: float
    full_width_in: float
    full_height_in: float
    required_width_px: int
    required_height_px: int
    page_count: int
    paper: str


@dataclass(frozen=True)
class CoverReport:
    """The result of validating one cover."""

    #: "passed", "warnings" or "failed" — the worst status among the checks.
    status: str
    width_px: int
    height_px: int
    dpi: int | None
    specification: CoverSpecification
    checks: list[CoverCheck] = field(default_factory=list)


def spine_width_in(page_count: int, paper: str = DEFAULT_PAPER) -> float:
    """The spine width for a book of this many pages, in inches."""
    thickness = PAPER_THICKNESS.get(paper, PAPER_THICKNESS[DEFAULT_PAPER])
    # A book has at least one page; a negative or zero count would produce a
    # spine narrower than the paper could physically be.
    return max(1, int(page_count)) * thickness


def specification_for(
    trim_size: str,
    page_count: int,
    paper: str = DEFAULT_PAPER,
) -> CoverSpecification:
    """The cover dimensions a book of this trim and length requires.

    This is the calculation the UI also quotes in `cover-specification.txt`, and
    it is deliberately independent of any uploaded file — a cover that has not
    been designed yet still has a size it has to be.
    """
    if trim_size not in TRIM_SIZES:
        raise CoverError(
            f"Unknown trim size {trim_size!r}. "
            f"Choose one of: {', '.join(sorted(TRIM_SIZES))}."
        )

    trim = TRIM_SIZES[trim_size]
    spine = spine_width_in(page_count, paper)

    # A paperback cover is the back cover, the spine and the front cover side by
    # side, plus bleed on all four outer edges.
    full_width = (trim.width * 2) + spine + (BLEED_IN * 2)
    full_height = trim.height + (BLEED_IN * 2)

    return CoverSpecification(
        trim_width_in=trim.width,
        trim_height_in=trim.height,
        spine_width_in=round(spine, 4),
        bleed_in=BLEED_IN,
        safe_area_in=SAFE_AREA_IN,
        full_width_in=round(full_width, 4),
        full_height_in=round(full_height, 4),
        required_width_px=int(round(full_width * TARGET_DPI)),
        required_height_px=int(round(full_height * TARGET_DPI)),
        page_count=max(1, int(page_count)),
        paper=paper,
    )


def _worst(statuses: list[str]) -> str:
    """The overall status: fail beats warn beats pass."""
    if "fail" in statuses:
        return "failed"
    if "warn" in statuses:
        return "warnings"
    return "passed"


def _inches(pixels: int, dpi: float) -> float:
    """Pixels as inches at a given resolution, or 0 when the DPI is unknown."""
    return pixels / dpi if dpi > 0 else 0.0


def validate_cover(
    data: bytes,
    trim_size: str,
    page_count: int,
    paper: str = DEFAULT_PAPER,
    *,
    require_exact_size: bool = True,
) -> CoverReport:
    """Measure an uploaded cover against the specification for its book.

    Raises `CoverError` when the bytes are not a readable image or the trim size
    is unknown — those are bad input, not a failed check, and conflating them
    would report "your cover failed the bleed check" for a PDF that Pillow
    cannot open.
    """
    if not data:
        raise CoverError("The uploaded cover is empty.")
    if len(data) > MAX_COVER_BYTES:
        raise CoverError(
            f"The cover is larger than {MAX_COVER_BYTES // (1024 * 1024)} MB."
        )

    spec = specification_for(trim_size, page_count, paper)

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise CoverError(
            "Cover validation needs Pillow, which is not installed in this "
            "deployment. Add it to requirements.txt and rebuild."
        ) from exc

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width_px, height_px = image.size
            metadata_dpi = _read_dpi(image)
    except CoverError:
        raise
    except Exception as exc:
        raise CoverError(f"That file could not be read as an image: {exc}") from exc

    checks: list[CoverCheck] = []

    # ---- dimensions --------------------------------------------------------
    width_ratio = width_px / spec.required_width_px
    height_ratio = height_px / spec.required_height_px
    exact = (
        abs(width_ratio - 1) <= DIMENSION_TOLERANCE
        and abs(height_ratio - 1) <= DIMENSION_TOLERANCE
    )
    large_enough = width_ratio >= 1 and height_ratio >= 1

    if exact:
        dimension_status = "pass"
        dimension_message = "The file matches the required cover size exactly."
    elif large_enough:
        dimension_status = "warn" if require_exact_size else "pass"
        dimension_message = (
            "Larger than required. The extra will be trimmed, so keep the "
            "artwork centred."
        )
    else:
        dimension_status = "fail"
        dimension_message = (
            "Smaller than required in at least one direction. Scaling it up "
            "will reduce the effective resolution."
        )

    checks.append(
        CoverCheck(
            id="dimensions",
            label="Dimensions",
            status=dimension_status,
            measured=f"{width_px} × {height_px} px",
            required=f"{spec.required_width_px} × {spec.required_height_px} px",
            message=dimension_message,
        )
    )

    # ---- effective resolution ---------------------------------------------
    # Computed from the pixels rather than read from metadata, so it is true of
    # the file whatever the file says about itself.
    effective_dpi = width_px / spec.full_width_in if spec.full_width_in else 0
    effective_dpi_int = int(round(effective_dpi))

    if effective_dpi >= TARGET_DPI:
        dpi_status = "pass"
        dpi_message = "Enough detail for print at this size."
    elif effective_dpi >= MIN_DPI:
        dpi_status = "warn"
        dpi_message = (
            "Below the 300 DPI print standard. It will print, but fine detail "
            "may look soft."
        )
    else:
        dpi_status = "fail"
        dpi_message = (
            "Too little detail for print at this size. Supply a larger image."
        )

    checks.append(
        CoverCheck(
            id="resolution",
            label="Resolution",
            status=dpi_status,
            measured=f"{effective_dpi_int} DPI",
            required=f"{TARGET_DPI} DPI",
            message=dpi_message,
        )
    )

    # ---- declared DPI ------------------------------------------------------
    # Reported separately from the measurement above because the two can
    # disagree, and a wrong tag is what a printer's own preflight will flag.
    if metadata_dpi is None:
        checks.append(
            CoverCheck(
                id="dpi",
                label="DPI metadata",
                status="warn",
                measured="not set",
                required=f"{TARGET_DPI} DPI",
                message=(
                    "The file declares no resolution. The measured resolution "
                    "above is what counts, but tagging it 300 DPI avoids "
                    "questions at the printer."
                ),
            )
        )
    else:
        metadata_ok = metadata_dpi >= TARGET_DPI
        checks.append(
            CoverCheck(
                id="dpi",
                label="DPI metadata",
                status="pass" if metadata_ok else "warn",
                measured=f"{metadata_dpi} DPI",
                required=f"{TARGET_DPI} DPI",
                message=(
                    "The file declares a print-ready resolution."
                    if metadata_ok
                    else "The file is tagged below 300 DPI even though its "
                    "pixels may be sufficient. Re-export it at 300 DPI."
                ),
            )
        )

    # ---- spine -------------------------------------------------------------
    checks.append(
        CoverCheck(
            id="spine",
            label="Spine width",
            status="pass",
            measured=f"{spec.spine_width_in:.4f} in for {spec.page_count} pages",
            required=f"{spec.spine_width_in:.4f} in",
            message=(
                f"Computed from {spec.page_count} pages on {spec.paper} paper. "
                "The spine is the middle band of the cover, between the two "
                "trim columns."
            ),
        )
    )

    # ---- bleed -------------------------------------------------------------
    # The file has to extend BLEED_IN past the trim on every outer edge, which
    # is exactly the difference between the full cover size and the page size.
    # Each axis's slack is split between its two edges, so the bleed the file
    # actually has is half the smaller of the two.
    bleed_have_x = _inches(width_px, TARGET_DPI) - ((spec.trim_width_in * 2) + spec.spine_width_in)
    bleed_have_y = _inches(height_px, TARGET_DPI) - spec.trim_height_in
    bleed_have = min(bleed_have_x, bleed_have_y) / 2

    if bleed_have >= BLEED_IN - 1e-6:
        bleed_status = "pass"
        bleed_message = "Artwork extends far enough past the trim on every edge."
    elif bleed_have > 0:
        bleed_status = "fail"
        bleed_message = (
            "There is not enough artwork outside the trim. Anything short of "
            "the bleed leaves a white sliver when the book is trimmed."
        )
    else:
        bleed_status = "fail"
        bleed_message = "The file is smaller than the trimmed page size."

    checks.append(
        CoverCheck(
            id="bleed",
            label="Bleed",
            status=bleed_status,
            measured=f"{max(bleed_have, 0):.3f} in",
            required=f"{BLEED_IN:.3f} in",
            message=bleed_message,
        )
    )

    # ---- safe area ---------------------------------------------------------
    # A flat raster cannot show where the text is, so this checks the one thing
    # it can: that there is room for the safe margin inside the trim.
    safe_have_x = (spec.trim_width_in * 2 + spec.spine_width_in) - (SAFE_AREA_IN * 2)
    safe_have_y = spec.trim_height_in - (SAFE_AREA_IN * 2)
    fits = safe_have_x > 0 and safe_have_y > 0

    checks.append(
        CoverCheck(
            id="safe-area",
            label="Safe area",
            status="pass" if fits else "fail",
            measured=f"{SAFE_AREA_IN:.2f} in margin",
            required=f"{SAFE_AREA_IN:.2f} in",
            message=(
                "Keep text and logos at least "
                f"{SAFE_AREA_IN:.2f} in inside the trim edge. This is a "
                "requirement on the artwork, which a flat image cannot be "
                "checked for automatically — use the guides in the preview."
                if fits
                else "The page is too small to hold the safe area."
            ),
        )
    )

    return CoverReport(
        status=_worst([check.status for check in checks]),
        width_px=width_px,
        height_px=height_px,
        dpi=metadata_dpi,
        specification=spec,
        checks=checks,
    )


def _read_dpi(image) -> int | None:
    """The resolution the file declares, if it declares one.

    JPEG and TIFF carry it in `info["dpi"]`; PNG carries it in `info["dpi"]`
    too when it was written with one. Anything else returns None, which the
    caller reports as "not set" rather than inventing a number.
    """
    dpi = image.info.get("dpi")
    if not dpi:
        return None

    # Pillow returns (x, y) for most formats and a bare number for some.
    horizontal = dpi[0] if isinstance(dpi, (tuple, list)) else dpi
    try:
        value = float(horizontal)
    except (TypeError, ValueError):
        return None

    # Some writers store dots per centimetre. 300 DPI is 118 DPCM, and nothing
    # legitimate sits between the two readings, so the ambiguity is resolvable.
    if 0 < value < 200:
        value *= 2.54

    return int(round(value)) if value > 0 else None


def specification_text(report: CoverReport) -> str:
    """The specification as a plain-text file, for the publishing package."""
    spec = report.specification
    lines = [
        "COVER SPECIFICATION",
        "===================",
        "",
        f"Trim size            {spec.trim_width_in} x {spec.trim_height_in} in",
        f"Page count           {spec.page_count}",
        f"Paper stock          {spec.paper}",
        f"Spine width          {spec.spine_width_in:.4f} in",
        f"Bleed                {spec.bleed_in:.3f} in on all outer edges",
        f"Safe area            {spec.safe_area_in:.2f} in inside the trim",
        "",
        "Full cover size",
        f"  {spec.full_width_in:.4f} x {spec.full_height_in:.4f} in",
        f"  {spec.required_width_px} x {spec.required_height_px} px at {TARGET_DPI} DPI",
        "",
        "The full cover is the back cover, the spine and the front cover side by",
        "side, plus bleed on all four outer edges.",
        "",
        "VALIDATION",
        "----------",
    ]

    for check in report.checks:
        lines.append(
            f"[{check.status.upper():4}] {check.label}: "
            f"{check.measured} (required {check.required})"
        )
        lines.append(f"       {check.message}")

    lines += [
        "",
        f"Overall: {report.status.upper()}",
        "",
        "A flat image cannot be checked for where its text sits. The safe-area",
        "requirement above is on the artwork, not on the file.",
        "",
    ]
    return "\n".join(lines)
