"""Checks for `services/packaging.py` — the publishing package builder.

The package is what an author uploads to a printer, so the things worth
asserting are the ones that would be discovered too late if they were wrong:
that the archive is a valid ZIP, that it names the book rather than a
placeholder, that the manuscript it contains is the text that was sent, that
the metadata describes that same book, and that a format whose library is
missing is reported rather than silently absent.

The renderers themselves are covered by `verify_export.py`. This file is about
the archive around them.

    python3 verify_package.py
"""

from __future__ import annotations

import io
import json
import zipfile

from services import cover as cover_service
from services import packaging
from services.exporter import ExportRequest

FAILS: list[str] = []


def check(label: str, actual, expected) -> None:
    ok = actual == expected
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {actual!r}")
    if not ok:
        FAILS.append(f"{label} ({actual!r} != {expected!r})")


def truthy(label: str, actual) -> None:
    ok = bool(actual)
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {actual!r}")
    if not ok:
        FAILS.append(label)


MANUSCRIPT = """Title page.

# Chapter One: Beginnings

The first paragraph, with enough words to be measured as prose.

* * *

# Chapter Two: Middles

A second chapter.
"""

REQUEST = ExportRequest(
    text=MANUSCRIPT,
    title="The Salt Road",
    author="Blessing Adeyemi",
    genre="fiction",
    script_type="latin",
    text_direction="ltr",
    language="en",
    trim_size="6x9",
)


def entries_of(package: packaging.PackageResult) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(package.result.content)) as archive:
        return sorted(archive.namelist())


def read(package: packaging.PackageResult, name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(package.result.content)) as archive:
        return archive.read(name)


print("== 1. the archive is a real zip ==")
package = packaging.build_package(REQUEST, page_count=296)
content = package.result.content
truthy("content is bytes", isinstance(content, bytes))
truthy("zip magic", content[:2] == b"PK")
check("media type", package.result.media_type, "application/zip")
check(
    "filename is slugged from the title",
    package.result.filename,
    "the-salt-road-publishing-package.zip",
)

entries = entries_of(package)

print("\n== 2. the index files are always present ==")
check("README present", "README.txt" in entries, True)
check("metadata present", "metadata.json" in entries, True)

print("\n== 3. rendered formats are named and counted ==")
# Whatever this deployment can render; RTF and TXT need no library.
rendered = {item.format for item in package.included}
skipped = {item.format for item in package.skipped}
check(
    "every format is either included or skipped",
    rendered | skipped,
    set(packaging.EXPORT_FORMATS),
)
check("no format is both", rendered & skipped, set())
check(
    "the two dependency-free formats rendered",
    {"rtf", "txt"} <= rendered,
    True,
)
for item in package.included:
    truthy(f"{item.format} is in the archive at {item.path}", item.path in entries)
    truthy(f"{item.format} recorded a byte count", item.bytes > 0)

print("\n== 4. a missing library is reported, not fatal ==")
# WeasyPrint, EbookLib and python-docx are not installed here, so the three
# formats that need them must appear as skips with a reason rather than
# taking the whole package down with them.
for item in package.skipped:
    truthy(
        f"{item.format} names the reason",
        isinstance(item.reason, str) and item.reason.strip(),
    )

print("\n== 5. the archive holds the text that was sent ==")
books = [name for name in entries if name.startswith("books/")]
truthy("at least one book in books/", books)
txt = [name for name in books if name.endswith(".txt")]
if txt:
    body = read(package, txt[0]).decode("utf-8")
    truthy("chapter one survived the round trip", "Chapter One: Beginnings" in body)
    truthy("chapter two survived the round trip", "Chapter Two: Middles" in body)
else:
    print("SKIP  no txt renderer in this deployment")

print("\n== 6. the metadata describes the same book ==")
meta = json.loads(read(package, "metadata.json").decode("utf-8"))

check("title", meta["title"], "The Salt Road")
check("author", meta["author"], "Blessing Adeyemi")
check("language code", meta["language"]["code"], "en")
check("script", meta["language"]["script"], "latin")
check("direction", meta["language"]["direction"], "ltr")
check("trim", meta["formatting"]["trim_size"], "6x9")
check("genre", meta["formatting"]["genre"], "fiction")
check("page count", meta["measurements"]["pages"], 296)
check("no ISBN invented", meta["identifiers"]["isbn"], None)
check(
    "the formats listed match what is in the archive",
    sorted(item["path"] for item in meta["files"]["books"]),
    sorted(item.path for item in package.included),
)
check(
    "the skips are recorded here too",
    sorted(item["format"] for item in meta["files"]["skipped"]),
    sorted(item.format for item in package.skipped),
)
truthy("no cover recorded", meta["files"]["cover"] is False)
truthy("no cover path recorded", meta["files"]["cover_path"] is None)
truthy("nothing was measured", meta["files"]["cover_measured"] is False)
truthy("cover specification absent", meta["cover_specification"] is None)
truthy("generated_at stamped", isinstance(meta["generated_at"], str))

print("\n== 7. a cover, when there is one ==")
# 1x1 transparent PNG — enough to be stored, and small enough that the check
# correctly fails. A failing cover must still produce a package.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)

report = None
try:
    report = cover_service.validate_cover(
        PNG, trim_size="6x9", page_count=296, paper="white"
    )
except cover_service.CoverError as exc:
    # Pillow is what reads the pixels. Without it the measurement cannot run at
    # all, and the library says so rather than guessing at the resolution — so
    # the cover is checked for storage but not for its dimensions. Note this is
    # a `CoverError`, not an `ImportError`: the missing dependency is reported
    # through the service's own error type, which is what the API turns into a
    # 503 naming the package.
    print(f"SKIP  cover validation unavailable: {exc}")

covered = packaging.build_package(
    REQUEST,
    page_count=296,
    cover=PNG,
    cover_name="front-cover.png",
    cover_report=report,
)
covered_entries = entries_of(covered)

truthy("the cover is in the archive", "cover/front-cover.png" in covered_entries)
truthy(
    "the cover file is byte-identical to what was sent",
    read(covered, "cover/front-cover.png") == PNG,
)

if report is not None:
    truthy("the specification is in the archive", "cover-specification.txt" in covered_entries)
    spec = read(covered, "cover-specification.txt").decode("utf-8")
    truthy("the specification names the spine", "spine" in spec.lower())
    truthy("the specification reports the page count", "296" in spec)
    truthy("the specification states the trim", "6" in spec and "9" in spec)
    truthy("the specification carries the verdict", report.status.upper() in spec)

    covered_meta = json.loads(read(covered, "metadata.json").decode("utf-8"))
    truthy("metadata records that a cover is present", covered_meta["files"]["cover"] is True)
    check(
        "metadata names the cover's path",
        covered_meta["files"]["cover_path"],
        "cover/front-cover.png",
    )
    truthy(
        "metadata records that it was measured",
        covered_meta["files"]["cover_measured"] is True,
    )
    truthy(
        "metadata carries the spine width",
        covered_meta["cover_specification"]["spine_width_in"] > 0,
    )
    check(
        "metadata's spine matches the report's",
        covered_meta["cover_specification"]["spine_width_in"],
        report.specification.spine_width_in,
    )
else:
    print("SKIP  specification checks need Pillow")
    # The archive still has to carry the cover, and must not claim to have
    # measured it. This is the path a deployment without Pillow actually takes.
    truthy(
        "no specification is written when nothing was measured",
        "cover-specification.txt" not in covered_entries,
    )
    covered_meta = json.loads(read(covered, "metadata.json").decode("utf-8"))
    truthy(
        "metadata still records the cover file",
        covered_meta["files"]["cover"] is True,
    )
    truthy(
        "metadata does not claim it was measured",
        covered_meta["files"]["cover_measured"] is False,
    )
    truthy("metadata claims no specification", covered_meta["cover_specification"] is None)

print("\n== 8. a book with no cover still packages ==")
bare = packaging.build_package(REQUEST, page_count=10)
bare_entries = entries_of(bare)
check(
    "no cover folder without a cover",
    any(name.startswith("cover/") for name in bare_entries),
    False,
)
truthy("README present", "README.txt" in bare_entries)
truthy(
    "the README says no cover was supplied",
    "No cover was supplied" in read(bare, "README.txt").decode("utf-8"),
)

print("\n== 9. the title never escapes the archive ==")
nasty = packaging.build_package(
    ExportRequest(text=MANUSCRIPT, title="../../etc/passwd"), page_count=10
)
for name in entries_of(nasty):
    truthy(f"no traversal in {name!r}", ".." not in name and not name.startswith("/"))

print("\n== 10. a page count of zero is not a crash ==")
# The API clamps this, but the builder is called from more than one place and a
# negative spine would be a silent corruption rather than an error.
zero = packaging.build_package(REQUEST, page_count=0)
truthy("still builds", zero.result.content[:2] == b"PK")

print()
if FAILS:
    print(f"{len(FAILS)} failure(s): {FAILS}")
    raise SystemExit(1)
print("0 failure(s)")
