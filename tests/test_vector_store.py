"""Vector store tests (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.exceptions import VectorStoreError
from retrieval.vector_store import (
    ensure_persist_directory,
    get_vector_store,
    sanitize_collection_name,
)
from tests.fake_embeddings import FakeEmbeddings


def test_sanitize_collection_name_strips_invalid_characters() -> None:
    assert sanitize_collection_name("abc/def:ghi") == "abc_def_ghi"


def test_sanitize_collection_name_prefixes_non_alnum_start() -> None:
    # After stripping invalid chars only punctuation remains -> fallback to "doc"
    assert sanitize_collection_name("!!!") == "doc"
    # Leading punctuation removed; result starts with alphanumeric
    assert sanitize_collection_name("---weird-id---") == "weird-id"


def test_sanitize_collection_name_truncates_long_ids() -> None:
    long_id = "a" * 100
    sanitized = sanitize_collection_name(long_id)
    assert len(sanitized) <= 63
    assert sanitized[0].isalnum()


def test_sanitize_collection_name_rejects_empty() -> None:
    with pytest.raises(VectorStoreError, match="non-empty"):
        sanitize_collection_name("   ")


def test_ensure_persist_directory_creates_path(tmp_path: Path) -> None:
    target = tmp_path / "chroma" / "nested"
    created = ensure_persist_directory(target)
    assert created == target
    assert target.is_dir()


def test_get_vector_store_creates_persistent_collection(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings()
    document_id = "test-doc-001"
    vector_store = get_vector_store(
        document_id,
        embeddings,
        persist_directory=tmp_path,
    )

    assert vector_store._collection.name == sanitize_collection_name(document_id)  # noqa: SLF001
    assert tmp_path.is_dir()


def test_get_vector_store_requires_embedding_function(tmp_path: Path) -> None:
    with pytest.raises(VectorStoreError, match="embedding_function is required"):
        get_vector_store("doc-1", None, persist_directory=tmp_path)
