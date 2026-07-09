"""Indexer tests (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from core.exceptions import (
    EmptyDocumentListError,
    InvalidDocumentIdError,
    MissingDocumentMetadataError,
    MissingOpenAIAPIKeyError,
)
from ingestion.indexer import (
    build_chunk_id,
    build_document_id,
    index_documents,
)
from retrieval.vector_store import get_vector_store, sanitize_collection_name
from services.openai_client import create_embeddings
from tests.fake_embeddings import FakeEmbeddings

SOURCE_PATH = Path("/tmp/test-tender.pdf").resolve()
DOCUMENT_ID = "test-doc-abc123"


def _make_doc(
    text: str,
    *,
    chunk_index: int,
    page: int = 1,
    document_id: str = DOCUMENT_ID,
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "document_id": document_id,
            "source": str(SOURCE_PATH),
            "page": page,
            "chunk_index": chunk_index,
        },
    )


def test_build_document_id_is_stable() -> None:
    first = build_document_id(SOURCE_PATH)
    second = build_document_id(SOURCE_PATH)
    assert first == second
    assert len(first) == 16


def test_build_chunk_id_is_stable() -> None:
    assert build_chunk_id(DOCUMENT_ID, 0) == f"{DOCUMENT_ID}__chunk__0"
    assert build_chunk_id(DOCUMENT_ID, 3) == f"{DOCUMENT_ID}__chunk__3"


def test_build_chunk_id_rejects_invalid_inputs() -> None:
    with pytest.raises(InvalidDocumentIdError):
        build_chunk_id("", 0)
    with pytest.raises(InvalidDocumentIdError):
        build_chunk_id(DOCUMENT_ID, -1)


def test_index_documents_creates_searchable_collection(tmp_path: Path) -> None:
    docs = [
        _make_doc("مهلت ارسال پیشنهاد تا پایان ماه است.", chunk_index=0, page=1),
        _make_doc("شرایط مناقصه در این بخش توضیح داده شده است.", chunk_index=1, page=2),
    ]
    embeddings = FakeEmbeddings()

    result = index_documents(
        docs,
        embedding_function=embeddings,
        persist_directory=tmp_path,
        replace_existing=True,
    )

    assert result.document_id == DOCUMENT_ID
    assert result.collection_name == sanitize_collection_name(DOCUMENT_ID)
    assert result.chunk_count == 2
    assert result.persist_directory == tmp_path
    assert result.replaced_existing is False

    vector_store = get_vector_store(DOCUMENT_ID, embeddings, persist_directory=tmp_path)
    collection = vector_store._collection  # noqa: SLF001
    assert collection.count() == 2

    hits = vector_store.similarity_search(docs[0].page_content, k=1)
    assert hits
    assert hits[0].metadata["chunk_index"] == 0


def test_index_documents_preserves_metadata(tmp_path: Path) -> None:
    docs = [_make_doc("متن آزمایشی برای حفظ متادیتا.", chunk_index=0, page=4)]
    embeddings = FakeEmbeddings()

    index_documents(
        docs,
        embedding_function=embeddings,
        persist_directory=tmp_path,
    )

    vector_store = get_vector_store(DOCUMENT_ID, embeddings, persist_directory=tmp_path)
    stored = vector_store.similarity_search("متن آزمایشی", k=1)[0]
    assert stored.metadata["document_id"] == DOCUMENT_ID
    assert stored.metadata["source"] == str(SOURCE_PATH)
    assert stored.metadata["page"] == 4
    assert stored.metadata["chunk_index"] == 0


def test_reindex_with_replace_existing_does_not_grow(tmp_path: Path) -> None:
    docs = [
        _make_doc("بند اول مناقصه.", chunk_index=0),
        _make_doc("بند دوم مناقصه.", chunk_index=1),
    ]
    embeddings = FakeEmbeddings()

    index_documents(docs, embedding_function=embeddings, persist_directory=tmp_path)
    index_documents(
        docs,
        embedding_function=embeddings,
        persist_directory=tmp_path,
        replace_existing=True,
    )

    vector_store = get_vector_store(DOCUMENT_ID, embeddings, persist_directory=tmp_path)
    assert vector_store._collection.count() == 2  # noqa: SLF001


def test_reindex_without_replace_skips_duplicate_ids(tmp_path: Path) -> None:
    docs = [
        _make_doc("بند اول مناقصه.", chunk_index=0),
        _make_doc("بند دوم مناقصه.", chunk_index=1),
    ]
    embeddings = FakeEmbeddings()

    index_documents(
        docs,
        embedding_function=embeddings,
        persist_directory=tmp_path,
        replace_existing=True,
    )
    index_documents(
        docs,
        embedding_function=embeddings,
        persist_directory=tmp_path,
        replace_existing=False,
    )

    vector_store = get_vector_store(DOCUMENT_ID, embeddings, persist_directory=tmp_path)
    assert vector_store._collection.count() == 2  # noqa: SLF001


def test_index_documents_infers_document_id_from_metadata(tmp_path: Path) -> None:
    docs = [_make_doc("متن استنتاج شناسه سند.", chunk_index=0)]
    result = index_documents(docs, embedding_function=FakeEmbeddings(), persist_directory=tmp_path)
    assert result.document_id == DOCUMENT_ID


def test_index_documents_raises_on_empty_list(tmp_path: Path) -> None:
    with pytest.raises(EmptyDocumentListError):
        index_documents([], embedding_function=FakeEmbeddings(), persist_directory=tmp_path)


def test_index_documents_raises_on_missing_metadata(tmp_path: Path) -> None:
    bad_doc = Document(page_content="متن بدون متادیتا.", metadata={"source": "x"})
    with pytest.raises(MissingDocumentMetadataError, match="document_id"):
        index_documents([bad_doc], embedding_function=FakeEmbeddings(), persist_directory=tmp_path)


def test_index_documents_raises_on_empty_page_content(tmp_path: Path) -> None:
    bad_doc = _make_doc("   ", chunk_index=0)
    with pytest.raises(MissingDocumentMetadataError, match="empty page_content"):
        index_documents([bad_doc], embedding_function=FakeEmbeddings(), persist_directory=tmp_path)


def test_index_documents_raises_on_conflicting_document_ids(tmp_path: Path) -> None:
    docs = [
        _make_doc("متن اول.", chunk_index=0, document_id="doc-a"),
        _make_doc("متن دوم.", chunk_index=1, document_id="doc-b"),
    ]
    with pytest.raises(InvalidDocumentIdError, match="multiple document_id"):
        index_documents(docs, embedding_function=FakeEmbeddings(), persist_directory=tmp_path)


def test_create_embeddings_requires_api_key() -> None:
    with pytest.raises(MissingOpenAIAPIKeyError):
        create_embeddings(api_key="")
    with pytest.raises(MissingOpenAIAPIKeyError):
        create_embeddings(api_key="   ")
