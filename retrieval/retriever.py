"""LangChain retriever wrapper (Phase 4)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import settings
from core.exceptions import (
    CollectionNotFoundError,
    EmptyQuestionError,
    InvalidDocumentIdError,
    InvalidTopKError,
    RetrievalError,
    VectorStoreError,
)
from retrieval.vector_store import (
    collection_exists,
    get_vector_store,
    sanitize_collection_name,
)
from services.openai_client import create_embeddings


def _validate_document_id(document_id: str) -> str:
    if not document_id or not document_id.strip():
        raise InvalidDocumentIdError("document_id must be a non-empty string")
    return document_id.strip()


def _resolve_top_k(top_k: int | None) -> int:
    resolved = settings.top_k if top_k is None else top_k
    if resolved <= 0:
        raise InvalidTopKError(f"top_k must be a positive integer, got {resolved}")
    return resolved


def _resolve_embedding_function(
    embedding_function: Embeddings | None,
) -> Embeddings:
    return embedding_function if embedding_function is not None else create_embeddings()


def _ensure_collection_exists(
    document_id: str,
    persist_directory: Path | None,
) -> str:
    collection_name = sanitize_collection_name(document_id)
    directory = persist_directory or settings.chroma_persist_dir
    if not collection_exists(directory, collection_name):
        raise CollectionNotFoundError(
            f"No indexed collection found for document_id={document_id!r} "
            f"(collection={collection_name!r}). Index the document first."
        )
    return collection_name


def get_retriever(
    document_id: str,
    *,
    top_k: int | None = None,
    embedding_function: Embeddings | None = None,
    persist_directory: Path | None = None,
) -> VectorStoreRetriever:
    """Return a LangChain retriever for the indexed document collection.

    Args:
        document_id: Logical document identifier used during indexing.
        top_k: Number of chunks to retrieve; defaults to ``settings.top_k``.
        embedding_function: Optional embeddings override (useful in tests).
        persist_directory: Optional Chroma persist path override.

    Raises:
        InvalidDocumentIdError: When ``document_id`` is empty.
        InvalidTopKError: When ``top_k`` is not positive.
        CollectionNotFoundError: When no collection exists for the document.
        VectorStoreError: When Chroma initialization fails.
    """
    resolved_id = _validate_document_id(document_id)
    k = _resolve_top_k(top_k)
    _ensure_collection_exists(resolved_id, persist_directory)

    embeddings = _resolve_embedding_function(embedding_function)
    vector_store = get_vector_store(
        resolved_id,
        embeddings,
        persist_directory=persist_directory,
    )
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve_documents(
    question: str,
    document_id: str,
    *,
    top_k: int | None = None,
    embedding_function: Embeddings | None = None,
    persist_directory: Path | None = None,
) -> list[Document]:
    """Retrieve the top-k relevant chunks for a question.

    Args:
        question: User question in Persian or any language.
        document_id: Logical document identifier used during indexing.
        top_k: Number of chunks to retrieve; defaults to ``settings.top_k``.
        embedding_function: Optional embeddings override (useful in tests).
        persist_directory: Optional Chroma persist path override.

    Returns:
        LangChain ``Document`` objects with chunk metadata preserved.

    Raises:
        EmptyQuestionError: When ``question`` is empty or whitespace-only.
        InvalidDocumentIdError: When ``document_id`` is empty.
        InvalidTopKError: When ``top_k`` is not positive.
        CollectionNotFoundError: When no collection exists for the document.
        RetrievalError: When the retriever invocation fails.
    """
    if not question or not question.strip():
        raise EmptyQuestionError("question must be a non-empty string")

    retriever = get_retriever(
        document_id,
        top_k=top_k,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )

    try:
        results = retriever.invoke(question.strip())
    except Exception as exc:
        if isinstance(exc, (EmptyQuestionError, InvalidDocumentIdError, InvalidTopKError, CollectionNotFoundError)):
            raise
        raise RetrievalError(f"Failed to retrieve documents for document_id={document_id!r}: {exc}") from exc

    if not isinstance(results, list):
        results = [results]

    return results
