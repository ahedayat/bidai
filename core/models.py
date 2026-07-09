"""Shared domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text extracted from a single PDF page."""

    page_number: int
    text: str
    source_path: Path
    char_count: int
    is_empty: bool
    is_short: bool


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Text extracted from an entire PDF document."""

    source_path: Path
    file_name: str
    page_count: int
    pages: tuple[ExtractedPage, ...] = field(default_factory=tuple)
    total_char_count: int = 0

    @property
    def full_text(self) -> str:
        """Concatenate non-empty page texts with blank-line separators."""
        parts = [page.text for page in self.pages if not page.is_empty]
        return "\n\n".join(parts)

    @property
    def empty_pages(self) -> tuple[ExtractedPage, ...]:
        return tuple(page for page in self.pages if page.is_empty)

    @property
    def short_pages(self) -> tuple[ExtractedPage, ...]:
        return tuple(page for page in self.pages if page.is_short)


@dataclass(frozen=True, slots=True)
class IndexResult:
    """Outcome of indexing LangChain documents into Chroma."""

    document_id: str
    collection_name: str
    chunk_count: int
    persist_directory: Path
    replaced_existing: bool
    indexed_at: datetime


@dataclass(frozen=True, slots=True)
class RAGSource:
    """A retrieved chunk formatted for display or downstream use."""

    page: int
    chunk_index: int
    page_chunk_index: int | None
    source: str
    file_name: str | None
    text_preview: str


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    """Outcome of a synchronous RAG question-answering request."""

    question: str
    answer: str
    document_id: str
    sources: tuple[RAGSource, ...]
    retrieved_count: int
    model_name: str
