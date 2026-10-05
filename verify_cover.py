"""Checks the cover arithmetic and the validation verdicts.

Run it with the project's own interpreter:

    python verify_cover.py

It is a script rather than a test suite because this repo has no test runner —
`verify.py` and `verify_export.py` set the precedent — and because the thing
worth checking is arithmetic: a spine width that is wrong by a factor of two
looks exactly like one that is right until a book is printed.
"""

from __future__ import annotations

import io
import sys

from services.cover import (
    BLEED_IN,
    PAPER_THICKNESS,
    TARGET_DPI,
    CoverError,
    specification_for,
    spine_width_in,
    validate_cover,
)

failures: list[str] = []


def check(label: str, actual, expected) -> None:
    if actual != expected:
        failures.append(f"{label}: expected {expected!r}, got {actual!r}")
        print(f"  FAIL  {label}: expected {expected!r}, got {actual!r}")
    else:
        print(f"  ok    {label}")


def close(label: str, actual: float, expected: float, tolerance: float = 1e-6) -> None:
    if abs(actual - expected) > tolerance:
        failures.append(f"{label}: expected {expected}, got {actual}")
        print(f"  FAIL  {label}: expected {expected}, got {actual}")
    else:
        print(f"  ok    {label} = {actual}")


print("Spine arithmetic")
# A 6x9 book of 296 pages on white stock.
check("spine for 296 pages, white", round(spine_width_in(296, "white"), 6), round(296 * 0.002252, 6))
check("spine for 296 pages, cream", round(spine_width_in(296, "cream"), 6), round(296 * 0.0025, 6))
check("spine for 296 pages, colour", round(spine_width_in(296, "colour"), 6), round(296 * 0.002347, 6))
# A zero or negative page count must not produce a negative spine.
check("spine floors at one page", spine_width_in(0, "white"), PAPER_THICKNESS["white"])
check("unknown paper falls back to white", spine_width_in(100, "unobtainium"), round(100 * PAPER_THICKNESS["white"], 6))

print()
print("Specification for a 6x9 book of 296 pages")
spec = specification_for("6x9", 296)
# 6 + 6 + 0.6666 + 0.125 + 0.125
close("full width", round(spec.full_width_in, 4), round((6 * 2) + spec.spine_width_in + (BLEED_IN * 2), 4))
close("full height", round(spec.full_height_in, 4), round(9 + (BLEED_IN * 2), 4))
check("required width px", spec.required_width_px, round(spec.full_width_in * TARGET_DPI))
check("required height px", spec.required_height_px, round(spec.full_height_in * TARGET_DPI))
check("page count is carried through", spec.page_count, 296)

print()
print("Unknown trim is rejected")
try:
    specification_for("nonsense", 100)
    failures.append("an unknown trim size was accepted")
    print("  FAIL  an unknown trim size was accepted")
except CoverError:
    print("  ok    unknown trim raises CoverError")

print()
try:
    from PIL import Image
except ImportError:
    print("Pillow is not installed, so the image checks were skipped.")
    print("Install it with: pip install -r requirements.txt")
else:
    def render(width: int, height: int, dpi: tuple[int, int] | None) -> bytes:
        buffer = io.BytesIO()
        image = Image.new("RGB", (width, height), "white")
        image.save(buffer, format="JPEG", dpi=dpi)
        return buffer.getvalue()

    print("A cover at exactly the required size, 300 DPI")
    exact = validate_cover(
        render(spec.required_width_px, spec.required_height_px, (300, 300)), "6x9", 296
    )
    check("overall status", exact.status, "passed")
    check("width", exact.width_px, spec.required_width_px)
    check("declared dpi", exact.dpi, 300)
    for item in exact.checks:
        print(f"      {item.status:4}  {item.label}: {item.measured} (needs {item.required})")

    print()
    print("A cover far too small for print")
    small = validate_cover(render(800, 600, None), "6x9", 296)
    check("overall status", small.status, "failed")
    by_id = {item.id: item for item in small.checks}
    check("dimensions fail", by_id["dimensions"].status, "fail")
    check("resolution fails", by_id["resolution"].status, "fail")
    check("bleed fails", by_id["bleed"].status, "fail")
    # No DPI tag is a warning, not a failure: the pixels are what count.
    check("missing dpi metadata warns", by_id["dpi"].status, "warn")

    print()
    print("A 72-DPI tag on a file with enough pixels")
    mislabelled = validate_cover(
        render(spec.required_width_px, spec.required_height_px, (72, 72)), "6x9", 296
    )
    mislabelled_checks = {item.id: item for item in mislabelled.checks}
    # The measured resolution is computed from the pixels, so it passes; the
    # tag is wrong and warns. That split is the point of having both.
    check("measured resolution passes", mislabelled_checks["resolution"].status, "pass")
    check("declared dpi warns", mislabelled_checks["dpi"].status, "warn")
    check("overall status", mislabelled.status, "warnings")

    print()
    print("A file that is not an image")
    try:
        validate_cover(b"this is not a jpeg", "6x9", 296)
        failures.append("non-image bytes were accepted")
        print("  FAIL  non-image bytes were accepted")
    except CoverError:
        print("  ok    non-image bytes raise CoverError")

    print()
    print("An empty upload")
    try:
        validate_cover(b"", "6x9", 296)
        failures.append("an empty upload was accepted")
        print("  FAIL  an empty upload was accepted")
    except CoverError:
        print("  ok    an empty upload raises CoverError")

print()
if failures:
    print(f"{len(failures)} check(s) failed:")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("All cover checks passed.")
