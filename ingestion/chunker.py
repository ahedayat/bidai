"""Text chunking (Phase 2)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from config.logging_config import get_logger
from core.exceptions import EmptyExtractedDocumentError, InvalidChunkConfigError
from core.models import ExtractedDocument

# Persian-aware separators: paragraph breaks, lines, sentence/clause punctuation,
# then space and character-level fallback. Includes Arabic/Persian punctuation variants.
PERSIAN_SEPARATORS: list[str] = [
    "\n\n",
    "\n",
    "؟",
    "?",
    "!",
    "۔",  # Arabic full stop (common in Persian texts)
    ".",
    "؛",
    "،",
    " ",
    "",
]

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 150

logger = get_logger(__name__)


def _resolve_chunk_params(
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> tuple[int, int]:
    """Resolve and validate chunk size and overlap."""
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    if size <= 0:
        raise InvalidChunkConfigError(f"chunk_size must be positive, got {size}")
    if overlap < 0:
        raise InvalidChunkConfigError(f"chunk_overlap must be non-negative, got {overlap}")
    if overlap >= size:
        raise InvalidChunkConfigError(
            f"chunk_overlap ({overlap}) must be smaller than chunk_size ({size})"
        )

    return size, overlap


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=PERSIAN_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_document(
    extracted: ExtractedDocument,
    *,
    document_id: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split an extracted document into LangChain Document chunks.

    Non-empty pages are chunked independently so page metadata stays accurate.
    Empty pages are skipped. When every page is empty, returns an empty list.

    Raises:
        EmptyExtractedDocumentError: When the document has no pages at all.
        InvalidChunkConfigError: When chunk size or overlap settings are invalid.
    """
    if not extracted.pages:
        raise EmptyExtractedDocumentError(
            f"Extracted document has no pages: {extracted.source_path}"
        )

    resolved_size, resolved_overlap = _resolve_chunk_params(chunk_size, chunk_overlap)
    logger.info(
        "Chunking document %s (chunk_size=%d, overlap=%d)",
        extracted.file_name,
        resolved_size,
        resolved_overlap,
    )
    splitter = _make_splitter(resolved_size, resolved_overlap)

    resolved_document_id = document_id if document_id is not None else extracted.file_name
    source = str(extracted.source_path)

    documents: list[Document] = []
    global_chunk_index = 0

    for page in extracted.pages:
        if page.is_empty or not page.text.strip():
            continue

        page_texts = splitter.split_text(page.text)
        non_empty_texts = [text for text in page_texts if text.strip()]

        for page_chunk_index, text in enumerate(non_empty_texts):
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "document_id": resolved_document_id,
                        "source": source,
                        "file_name": extracted.file_name,
                        "page": page.page_number,
                        "chunk_index": global_chunk_index,
                        "page_chunk_index": page_chunk_index,
                        "char_count": len(text),
                    },
                )
            )
            global_chunk_index += 1

    logger.info("Chunking complete: %d chunks from %s", len(documents), extracted.file_name)
    return documents


def _build_sample_document() -> ExtractedDocument:
    """Build a tiny in-memory extracted document for CLI debugging."""
    from core.models import ExtractedPage

    source_path = Path("/tmp/sample-tender.pdf").resolve()
    pages = (
        ExtractedPage(
            page_number=1,
            text="این یک متن آزمایشی است. مهلت ارسال پیشنهاد چه زمانی است؟",
            source_path=source_path,
            char_count=0,
            is_empty=False,
            is_short=False,
        ),
        ExtractedPage(
            page_number=2,
            text="",
            source_path=source_path,
            char_count=0,
            is_empty=True,
            is_short=False,
        ),
        ExtractedPage(
            page_number=3,
            text="شرایط مناقصه در این بخش توضیح داده شده است؛ لطفاً با دقت مطالعه کنید.",
            source_path=source_path,
            char_count=0,
            is_empty=False,
            is_short=False,
        ),
    )
    return ExtractedDocument(
        source_path=source_path,
        file_name="sample-tender.pdf",
        page_count=len(pages),
        pages=pages,
        total_char_count=sum(len(p.text) for p in pages),
    )


def _main() -> None:
    sample = _build_sample_document()
    chunks = chunk_document(sample, document_id="debug-doc")
    print(f"Chunks: {len(chunks)}")
    for chunk in chunks:
        meta = chunk.metadata
        preview = chunk.page_content[:60].replace("\n", " ")
        print(
            f"  [{meta['chunk_index']}] page={meta['page']} "
            f"page_chunk={meta['page_chunk_index']} chars={meta['char_count']}: {preview}..."
        )


if __name__ == "__main__":
    _main()
