"""FastAPI entry point for the automated ebook formatting platform."""

from __future__ import annotations

import io
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services import exporter
from services.analyzer import analyze_manuscript
from services.extractors import ManuscriptError, extract_text

app = FastAPI(
    title="Ebook Formatting Platform",
    description="Analyzes an uploaded manuscript ahead of formatting.",
    version="0.1.0",
)


class TypoLocation(BaseModel):
    """One suspected misspelling, with where it occurs."""

    word: str = Field(description="The word as it appears in the manuscript.")
    suggestions: list[str] = Field(description="Up to three suggested corrections.")
    chapter: str = Field(description="Chapter it is in, or 'Front Matter'.")
    line_number: int = Field(description="Line the word is on, 1-based.")
    context: str = Field(description="A snippet of the surrounding words.")
    likely_proper_noun: bool = Field(
        description="A capitalised word not merely opening a sentence, so a "
        "name or brand rather than necessarily a mistake."
    )


class BookAnalysis(BaseModel):
    """Measurements taken from one manuscript."""

    word_count: int = Field(description="Total words in the manuscript.")
    chapter_count: int = Field(description="Chapter markers found in the manuscript.")
    detected_language: str = Field(description="ISO language code, or 'unknown'.")
    language_name: str = Field(description="Human-readable language name.")
    text_direction: Literal["ltr", "rtl"] = Field(description="Reading direction.")
    script_type: Literal[
        "latin", "rtl", "cjk", "cyrillic", "greek", "devanagari"
    ] = Field(description="Writing system.")
    estimated_pages: int = Field(description="Estimated typeset page count.")
    token_cost: int = Field(description="Cost of formatting this manuscript.")
    typo_count: int = Field(description="Total suspected misspellings found.")
    typos: list[TypoLocation] = Field(description="The suspected misspellings.")
    typo_check_available: bool = Field(
        description="False when no dictionary exists for the detected language, "
        "in which case typo_count is 0 and says nothing about the manuscript."
    )
    typo_check_note: str | None = Field(
        description="Why the typo list is incomplete, if it is."
    )
    manuscript_text: str | None = Field(
        default=None,
        description="The extracted manuscript, returned only when include_text "
        "was requested. Lets a caller apply the typo corrections and hand the "
        "corrected text back to /api/export-book, rather than uploading twice.",
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze-book", response_model=BookAnalysis)
async def analyze_book(
    file: UploadFile = File(...),
    include_text: bool = Form(default=False),
) -> BookAnalysis:
    """Analyze an uploaded .docx, .md, or .txt manuscript."""
    data = await file.read()
    if not data.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = analyze_manuscript(file.filename or "", data)
    except ManuscriptError as exc:
        # Bad input from the caller, not a server fault: unsupported extension or
        # a file too damaged to parse.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = asdict(result)
    if include_text:
        try:
            payload["manuscript_text"] = extract_text(file.filename or "", data)
        except ManuscriptError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BookAnalysis(**payload)


@app.post(
    "/api/export-book",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"application/octet-stream": {}},
            "description": "The rendered book, as a download.",
        },
        400: {"description": "Unsupported format or genre, or a bad manuscript."},
        503: {"description": "The library needed for that format is not installed."},
    },
)
async def export_book(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    genre: str = Form(default="non-fiction"),
    export_format: str = Form(default="pdf", alias="format"),
    script_type: str = Form(default="latin"),
    text_direction: str = Form(default="ltr"),
    title: str = Form(default="Untitled"),
    author: str | None = Form(default=None),
    language: str = Form(default="en"),
    custom_font: str | None = Form(default=None),
    font_size: float | None = Form(default=None),
    trim_size: str = Form(default=exporter.DEFAULT_TRIM),
) -> StreamingResponse:
    """Render a manuscript to PDF, EPUB, DOCX, RTF or TXT and stream it back.

    Supply the manuscript either as an uploaded .docx/.md/.txt file or as the
    `text` field. `script_type` and `text_direction` are the values
    /api/analyze-book returned for this book. Pass `text` when the manuscript
    has been edited since analysis; pass the file again when it has not.
    """
    manuscript = await _manuscript_text(file, text)

    try:
        result = exporter.export_book(
            exporter.ExportRequest(
                text=manuscript,
                title=title,
                author=author or None,
                genre=genre,
                format=export_format,
                script_type=script_type,
                text_direction=text_direction,
                language=language,
                custom_font=custom_font or None,
                font_size=font_size,
                trim_size=trim_size,
            )
        )
    except exporter.BackendUnavailableError as exc:
        # The request itself was fine; this deployment cannot serve that format.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except exporter.ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StreamingResponse(
        io.BytesIO(result.content),
        media_type=result.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{result.filename}"',
            "Content-Length": str(len(result.content)),
        },
    )


async def _manuscript_text(file: UploadFile | None, text: str | None) -> str:
    """Take the manuscript from an uploaded file if there is one, else from text."""
    if file is not None and file.filename:
        data = await file.read()
        if data.strip():
            try:
                return extract_text(file.filename, data)
            except ManuscriptError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    if text and text.strip():
        return text

    raise HTTPException(
        status_code=400,
        detail="Provide a manuscript file upload or a non-empty text field.",
    )
