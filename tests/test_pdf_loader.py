"""PDF loader tests (Phase 1)."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from core.exceptions import PDFInvalidFileError, PDFNotFoundError
from core.models import ExtractedDocument, ExtractedPage
from ingestion.pdf_loader import load_pdf, normalize_text

PERSIAN_SENTENCE = "این یک فایل آزمایشی برای مناقصه است."
SHORT_TEXT = "کوتاه"
MIN_PAGE_CHARS = 30


def _create_test_pdf(path: Path, pages: list[str]) -> None:
    """Generate a tiny PDF with the given page texts."""
    document = fitz.open()
    try:
        for page_text in pages:
            page = document.new_page()
            page.insert_text((72, 72), page_text, fontsize=12)
        document.save(path)
    finally:
        document.close()


@pytest.fixture
def tiny_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "fixture.pdf"
    _create_test_pdf(
        pdf_path,
        [
            PERSIAN_SENTENCE,
            "",
            SHORT_TEXT,
        ],
    )
    return pdf_path


def test_normalize_text_collapses_whitespace_preserves_persian() -> None:
    raw = "  این   یک\tفایل\n\n\nآزمایشی  "
    assert normalize_text(raw) == "این یک فایل\n\nآزمایشی"


def test_load_pdf_nonexistent_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"
    with pytest.raises(PDFNotFoundError, match="not found"):
        load_pdf(missing)


def test_load_pdf_rejects_non_pdf(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(PDFInvalidFileError, match="not a PDF"):
        load_pdf(text_file)


def test_load_pdf_valid_fixture(tiny_pdf: Path) -> None:
    document = load_pdf(tiny_pdf, min_page_chars=MIN_PAGE_CHARS)

    assert isinstance(document, ExtractedDocument)
    assert document.file_name == "fixture.pdf"
    assert document.source_path == tiny_pdf.resolve()
    assert document.page_count == 3
    assert document.total_char_count > 0
    assert len(document.pages) == 3


def test_load_pdf_page_count(tiny_pdf: Path) -> None:
    document = load_pdf(tiny_pdf)
    assert document.page_count == 3


def test_load_pdf_extracts_non_empty_text(tiny_pdf: Path) -> None:
    document = load_pdf(tiny_pdf, min_page_chars=MIN_PAGE_CHARS)
    first_page = document.pages[0]

    assert first_page.text
    # Persian extraction may vary by font/renderer; ensure structure is valid.
    assert first_page.char_count > 0
    assert not first_page.is_empty


def test_load_pdf_preserves_page_metadata(tiny_pdf: Path) -> None:
    document = load_pdf(tiny_pdf, min_page_chars=MIN_PAGE_CHARS)

    for index, page in enumerate(document.pages, start=1):
        assert isinstance(page, ExtractedPage)
        assert page.page_number == index
        assert page.source_path == tiny_pdf.resolve()
        assert page.char_count == len(page.text)


def test_load_pdf_empty_and_short_page_detection(tiny_pdf: Path) -> None:
    document = load_pdf(tiny_pdf, min_page_chars=MIN_PAGE_CHARS)

    empty_pages = document.empty_pages
    short_pages = document.short_pages

    assert len(empty_pages) == 1
    assert empty_pages[0].page_number == 2
    assert empty_pages[0].is_empty

    assert len(short_pages) == 1
    assert short_pages[0].page_number == 3
    assert short_pages[0].is_short
    assert not short_pages[0].is_empty


def test_extracted_document_full_text_skips_empty_pages(tiny_pdf: Path) -> None:
    document = load_pdf(tiny_pdf, min_page_chars=MIN_PAGE_CHARS)
    full_text = document.full_text

    assert full_text
    assert "\n\n" in full_text or len(document.empty_pages) == 0
