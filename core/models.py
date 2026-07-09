"""Shared domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
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
