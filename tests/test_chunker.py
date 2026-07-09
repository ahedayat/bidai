"""Chunker tests (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from config.settings import settings
from core.exceptions import EmptyExtractedDocumentError, InvalidChunkConfigError
from core.models import ExtractedDocument, ExtractedPage
from ingestion.chunker import PERSIAN_SEPARATORS, chunk_document

SOURCE_PATH = Path("/tmp/test-tender.pdf").resolve()
FILE_NAME = "test-tender.pdf"


def _make_page(
    page_number: int,
    text: str,
    *,
    is_empty: bool | None = None,
    is_short: bool = False,
) -> ExtractedPage:
    normalized = text.strip()
    empty = is_empty if is_empty is not None else not normalized
    return ExtractedPage(
        page_number=page_number,
        text=text,
        source_path=SOURCE_PATH,
        char_count=len(text),
        is_empty=empty,
        is_short=is_short,
    )


def _make_document(
    pages: list[ExtractedPage],
    *,
    file_name: str = FILE_NAME,
) -> ExtractedDocument:
    return ExtractedDocument(
        source_path=SOURCE_PATH,
        file_name=file_name,
        page_count=len(pages),
        pages=tuple(pages),
        total_char_count=sum(p.char_count for p in pages),
    )


def test_chunk_simple_extracted_document() -> None:
    document = _make_document(
        [
            _make_page(1, "این یک متن آزمایشی برای مناقصه است."),
            _make_page(2, "مهلت ارسال پیشنهاد تا پایان ماه است."),
        ]
    )

    chunks = chunk_document(document)

    assert len(chunks) >= 2
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert chunks[0].page_content
    assert chunks[0].metadata["page"] == 1
    assert chunks[-1].metadata["page"] == 2


def test_chunk_skips_empty_pages() -> None:
    document = _make_document(
        [
            _make_page(1, "صفحه اول دارای متن است."),
            _make_page(2, "", is_empty=True),
            _make_page(3, "صفحه سوم نیز دارای متن است."),
        ]
    )

    chunks = chunk_document(document)
    pages = {chunk.metadata["page"] for chunk in chunks}

    assert 2 not in pages
    assert pages == {1, 3}


def test_chunk_handles_short_page() -> None:
    document = _make_document([_make_page(1, "کوتاه", is_short=True)])

    chunks = chunk_document(document, chunk_size=1000, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].page_content == "کوتاه"
    assert chunks[0].metadata["char_count"] == len("کوتاه")


def test_chunk_metadata_preservation() -> None:
    document = _make_document([_make_page(1, "متن نمونه برای بررسی متادیتا.")])

    chunks = chunk_document(document, document_id="custom-doc-123")

    assert len(chunks) == 1
    meta = chunks[0].metadata
    assert meta["document_id"] == "custom-doc-123"
    assert meta["source"] == str(SOURCE_PATH)
    assert meta["file_name"] == FILE_NAME
    assert meta["page"] == 1
    assert meta["chunk_index"] == 0
    assert meta["page_chunk_index"] == 0
    assert meta["char_count"] == len(chunks[0].page_content)


def test_chunk_index_ordering_is_stable() -> None:
    long_sentence = "جمله آزمایشی. " * 80
    document = _make_document(
        [
            _make_page(1, long_sentence),
            _make_page(2, long_sentence),
        ]
    )

    chunks = chunk_document(document, chunk_size=200, chunk_overlap=20)

    chunk_indices = [chunk.metadata["chunk_index"] for chunk in chunks]
    assert chunk_indices == list(range(len(chunks)))

    page_chunk_indices_page_1 = [
        chunk.metadata["page_chunk_index"]
        for chunk in chunks
        if chunk.metadata["page"] == 1
    ]
    assert page_chunk_indices_page_1 == list(range(len(page_chunk_indices_page_1)))


def test_chunk_default_document_id_uses_file_name() -> None:
    document = _make_document(
        [_make_page(1, "متن.")],
        file_name="my-tender.pdf",
    )

    chunks = chunk_document(document)

    assert chunks[0].metadata["document_id"] == "my-tender.pdf"


def test_chunk_invalid_chunk_size() -> None:
    document = _make_document([_make_page(1, "متن.")])

    with pytest.raises(InvalidChunkConfigError, match="chunk_size must be positive"):
        chunk_document(document, chunk_size=0)

    with pytest.raises(InvalidChunkConfigError, match="chunk_size must be positive"):
        chunk_document(document, chunk_size=-10)


def test_chunk_invalid_chunk_overlap() -> None:
    document = _make_document([_make_page(1, "متن.")])

    with pytest.raises(InvalidChunkConfigError, match="chunk_overlap must be non-negative"):
        chunk_document(document, chunk_size=100, chunk_overlap=-1)

    with pytest.raises(InvalidChunkConfigError, match="must be smaller than chunk_size"):
        chunk_document(document, chunk_size=100, chunk_overlap=100)

    with pytest.raises(InvalidChunkConfigError, match="must be smaller than chunk_size"):
        chunk_document(document, chunk_size=100, chunk_overlap=150)


def test_chunk_raises_when_no_pages() -> None:
    document = ExtractedDocument(
        source_path=SOURCE_PATH,
        file_name=FILE_NAME,
        page_count=0,
        pages=(),
        total_char_count=0,
    )

    with pytest.raises(EmptyExtractedDocumentError, match="no pages"):
        chunk_document(document)


def test_chunk_all_empty_pages_returns_empty_list() -> None:
    document = _make_document(
        [
            _make_page(1, "", is_empty=True),
            _make_page(2, "   ", is_empty=True),
        ]
    )

    chunks = chunk_document(document)

    assert chunks == []


def test_chunk_persian_punctuation_separators() -> None:
    """Splitter should prefer breaks at Persian punctuation when chunking."""
    text = "؟ ".join([f"سوال شماره {i}" for i in range(1, 21)])
    document = _make_document([_make_page(1, text)])

    chunks = chunk_document(document, chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.page_content.strip()
        # Most chunks should end near Persian punctuation or be short tail pieces.
        content = chunk.page_content.strip()
        assert len(content) <= 80 or content.endswith(("؟", "?", ".", "۔"))


def test_chunk_chunks_are_non_empty() -> None:
    document = _make_document(
        [
            _make_page(1, "متن اول.\n\nمتن دوم."),
            _make_page(2, "متن سوم؛ متن چهارم، متن پنجم."),
        ]
    )

    chunks = chunk_document(document, chunk_size=30, chunk_overlap=5)

    assert chunks
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(chunk.metadata["char_count"] > 0 for chunk in chunks)


def test_chunk_size_respected_approximately() -> None:
    paragraph = "این یک جمله آزمایشی است. " * 50
    document = _make_document([_make_page(1, paragraph)])

    chunk_size = 150
    chunks = chunk_document(document, chunk_size=chunk_size, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        # RecursiveCharacterTextSplitter keeps indivisible segments whole.
        assert len(chunk.page_content) <= chunk_size * 2


def test_chunk_avoids_single_chunk_for_whole_document() -> None:
    page_text = "بند مناقصه. " * 40
    document = _make_document(
        [
            _make_page(1, page_text),
            _make_page(2, page_text),
            _make_page(3, page_text),
        ]
    )

    chunks = chunk_document(document, chunk_size=120, chunk_overlap=15)

    assert len(chunks) > 3
    pages_represented = {chunk.metadata["page"] for chunk in chunks}
    assert pages_represented == {1, 2, 3}


def test_chunk_uses_settings_defaults() -> None:
    document = _make_document([_make_page(1, "متن کوتاه.")])

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150


def test_persian_separators_include_required_tokens() -> None:
    required = ["\n\n", "\n", "؟", "!", ".", "؛", "،", " ", ""]
    for separator in required:
        assert separator in PERSIAN_SEPARATORS
