"""Retriever tests (Phase 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from core.exceptions import (
    CollectionNotFoundError,
    EmptyQuestionError,
    InvalidDocumentIdError,
    InvalidTopKError,
)
from ingestion.indexer import index_documents
from retrieval.retriever import get_retriever, retrieve_documents
from tests.fake_embeddings import FakeEmbeddings

SOURCE_PATH = Path("/tmp/test-tender.pdf").resolve()
DOCUMENT_ID = "test-retriever-doc"


def _make_doc(
    text: str,
    *,
    chunk_index: int,
    page: int = 1,
    page_chunk_index: int = 0,
    document_id: str = DOCUMENT_ID,
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "document_id": document_id,
            "source": str(SOURCE_PATH),
            "file_name": "test-tender.pdf",
            "page": page,
            "chunk_index": chunk_index,
            "page_chunk_index": page_chunk_index,
        },
    )


def _index_sample_docs(tmp_path: Path) -> None:
    docs = [
        _make_doc(
            "مهلت ارسال پیشنهاد تا پایان ماه است.",
            chunk_index=0,
            page=1,
            page_chunk_index=0,
        ),
        _make_doc(
            "شرایط مناقصه در این بخش توضیح داده شده است.",
            chunk_index=1,
            page=2,
            page_chunk_index=0,
        ),
    ]
    index_documents(
        docs,
        DOCUMENT_ID,
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
        replace_existing=True,
    )


def test_get_retriever_with_valid_document_id(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    retriever = get_retriever(
        DOCUMENT_ID,
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
    )
    assert retriever is not None
    assert retriever.search_kwargs["k"] == 4


def test_get_retriever_respects_top_k_override(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    retriever = get_retriever(
        DOCUMENT_ID,
        top_k=2,
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
    )
    assert retriever.search_kwargs["k"] == 2


def test_get_retriever_rejects_invalid_document_id() -> None:
    with pytest.raises(InvalidDocumentIdError):
        get_retriever("")


def test_get_retriever_rejects_invalid_top_k(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    with pytest.raises(InvalidTopKError):
        get_retriever(
            DOCUMENT_ID,
            top_k=0,
            embedding_function=FakeEmbeddings(),
            persist_directory=tmp_path,
        )


def test_get_retriever_raises_when_collection_missing(tmp_path: Path) -> None:
    with pytest.raises(CollectionNotFoundError):
        get_retriever(
            "missing-doc-id",
            embedding_function=FakeEmbeddings(),
            persist_directory=tmp_path,
        )


def test_retrieve_documents_rejects_empty_question(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    with pytest.raises(EmptyQuestionError):
        retrieve_documents(
            "   ",
            DOCUMENT_ID,
            embedding_function=FakeEmbeddings(),
            persist_directory=tmp_path,
        )


def test_retrieve_documents_returns_results(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    results = retrieve_documents(
        "مهلت ارسال پیشنهاد تا پایان ماه است.",
        DOCUMENT_ID,
        top_k=1,
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
    )
    assert len(results) == 1
    assert "مهلت" in results[0].page_content


def test_retrieve_documents_preserves_metadata(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    results = retrieve_documents(
        "شرایط مناقصه",
        DOCUMENT_ID,
        top_k=2,
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
    )
    assert len(results) >= 1
    for doc in results:
        assert doc.metadata["document_id"] == DOCUMENT_ID
        assert doc.metadata["page"] in (1, 2)
        assert "chunk_index" in doc.metadata
        assert doc.metadata["source"] == str(SOURCE_PATH)
        assert doc.metadata["file_name"] == "test-tender.pdf"
