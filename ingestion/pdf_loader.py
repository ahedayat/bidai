"""PDF loading (Phase 1)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz

from core.exceptions import (
    PDFEncryptedError,
    PDFExtractionError,
    PDFInvalidFileError,
    PDFNotFoundError,
    PDFReadError,
)
from core.models import ExtractedDocument, ExtractedPage

_PDF_SUFFIXES = {".pdf"}
_HORIZONTAL_WHITESPACE_RE = re.compile(r"[^\S\n]+")
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving Persian and other Unicode content."""
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _HORIZONTAL_WHITESPACE_RE.sub(" ", normalized)
    normalized = _EXCESS_NEWLINES_RE.sub("\n\n", normalized)
    return normalized.strip()


def _validate_pdf_path(path: Path) -> None:
    if not path.exists():
        raise PDFNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise PDFInvalidFileError(f"Path is not a file: {path}")
    if path.suffix.lower() not in _PDF_SUFFIXES:
        raise PDFInvalidFileError(f"File is not a PDF (expected .pdf extension): {path}")


def load_pdf(path: str | Path, *, min_page_chars: int = 30) -> ExtractedDocument:
    """Load a PDF and extract text page by page."""
    source_path = Path(path).expanduser().resolve()
    _validate_pdf_path(source_path)

    try:
        document = fitz.open(source_path)
    except fitz.FileDataError as exc:
        raise PDFInvalidFileError(f"File is not a valid PDF: {source_path}") from exc
    except Exception as exc:
        raise PDFReadError(f"Unable to read PDF: {source_path}") from exc

    try:
        if document.needs_pass:
            raise PDFEncryptedError(f"PDF is encrypted and requires a password: {source_path}")

        pages: list[ExtractedPage] = []
        for page_index in range(document.page_count):
            page_number = page_index + 1
            try:
                raw_text = document[page_index].get_text("text")
            except Exception as exc:
                raise PDFExtractionError(
                    f"Failed to extract text from page {page_number} in {source_path}"
                ) from exc

            text = normalize_text(raw_text)
            char_count = len(text)
            is_empty = char_count == 0
            is_short = not is_empty and char_count < min_page_chars

            pages.append(
                ExtractedPage(
                    page_number=page_number,
                    text=text,
                    source_path=source_path,
                    char_count=char_count,
                    is_empty=is_empty,
                    is_short=is_short,
                )
            )

        total_char_count = sum(page.char_count for page in pages)
        return ExtractedDocument(
            source_path=source_path,
            file_name=source_path.name,
            page_count=len(pages),
            pages=tuple(pages),
            total_char_count=total_char_count,
        )
    finally:
        document.close()


def _print_debug_summary(document: ExtractedDocument) -> None:
    print(f"file_name: {document.file_name}")
    print(f"page_count: {document.page_count}")
    print(f"total_char_count: {document.total_char_count}")

    if document.pages:
        first = document.pages[0]
        last = document.pages[-1]
        print(f"first_page_sample ({first.page_number}): {first.text[:200]!r}")
        print(f"last_page_sample ({last.page_number}): {last.text[:200]!r}")

    empty_pages = document.empty_pages
    short_pages = document.short_pages
    print(f"empty_pages ({len(empty_pages)}): {[p.page_number for p in empty_pages]}")
    print(f"short_pages ({len(short_pages)}): {[p.page_number for p in short_pages]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m ingestion.pdf_loader <path-to-pdf>", file=sys.stderr)
        sys.exit(1)

    result = load_pdf(sys.argv[1])
    _print_debug_summary(result)
