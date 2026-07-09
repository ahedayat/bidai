"""Embedding and Chroma indexing (Phase 3)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings
from core.exceptions import (
    EmptyDocumentListError,
    IndexingFailureError,
    InvalidDocumentIdError,
    MissingDocumentMetadataError,
)
from core.models import IndexResult
from retrieval.vector_store import (
    collection_exists,
    delete_collection,
    ensure_persist_directory,
    get_vector_store,
    sanitize_collection_name,
)
from services.openai_client import create_embeddings

_REQUIRED_METADATA_KEYS = ("document_id", "source", "page", "chunk_index")


def build_document_id(source_path: str | Path) -> str:
    """Build a stable document ID from an absolute source path."""
    resolved = str(Path(source_path).resolve())
    if not resolved:
        raise InvalidDocumentIdError("source_path must resolve to a non-empty path")
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def build_chunk_id(document_id: str, chunk_index: int) -> str:
    """Build a stable vector-store ID for a chunk."""
    if not document_id or not document_id.strip():
        raise InvalidDocumentIdError("document_id must be a non-empty string")
    if chunk_index < 0:
        raise InvalidDocumentIdError(f"chunk_index must be non-negative, got {chunk_index}")
    return f"{document_id}__chunk__{chunk_index}"


def _resolve_embedding_function(
    embedding_function: Embeddings | None,
) -> Embeddings:
    if embedding_function is not None:
        return embedding_function
    return create_embeddings()


def _validate_documents(docs: list[Document]) -> None:
    if not docs:
        raise EmptyDocumentListError("Cannot index an empty document list")

    for index, doc in enumerate(docs):
        if not doc.page_content or not doc.page_content.strip():
            raise MissingDocumentMetadataError(
                f"Document at list index {index} has empty page_content"
            )

        metadata = doc.metadata or {}
        missing = [key for key in _REQUIRED_METADATA_KEYS if key not in metadata]
        if missing:
            raise MissingDocumentMetadataError(
                f"Document at list index {index} is missing required metadata: {missing}"
            )


def _infer_document_id(docs: list[Document]) -> str:
    document_ids = {str(doc.metadata["document_id"]).strip() for doc in docs}
    document_ids.discard("")
    if not document_ids:
        raise InvalidDocumentIdError("Could not infer document_id from document metadata")

    if len(document_ids) > 1:
        raise InvalidDocumentIdError(
            f"Documents contain multiple document_id values: {sorted(document_ids)}"
        )

    source_paths = {str(doc.metadata["source"]).strip() for doc in docs}
    source_paths.discard("")
    if len(source_paths) > 1:
        raise InvalidDocumentIdError(
            f"Documents contain multiple source paths: {sorted(source_paths)}"
        )

    return document_ids.pop()


def _chunk_ids_for_documents(document_id: str, docs: list[Document]) -> list[str]:
    return [
        build_chunk_id(document_id, int(doc.metadata["chunk_index"]))
        for doc in docs
    ]


def index_documents(
    docs: list[Document],
    document_id: str | None = None,
    *,
    replace_existing: bool = True,
    embedding_function: Embeddings | None = None,
    persist_directory: Path | None = None,
) -> IndexResult:
    """Embed and persist LangChain documents into a local Chroma collection.

    Args:
        docs: Chunked LangChain documents from Phase 2.
        document_id: Optional explicit document ID; inferred from metadata when omitted.
        replace_existing: When True, delete any existing collection before indexing.
        embedding_function: Optional embeddings override (useful for tests).
        persist_directory: Optional Chroma persist directory override.

    Returns:
        IndexResult with collection metadata and indexing timestamp.
    """
    _validate_documents(docs)

    resolved_document_id = document_id.strip() if document_id else _infer_document_id(docs)
    if not resolved_document_id:
        raise InvalidDocumentIdError("document_id must be a non-empty string")

    embeddings = _resolve_embedding_function(embedding_function)
    directory = ensure_persist_directory(persist_directory)
    collection_name = sanitize_collection_name(resolved_document_id)
    chunk_ids = _chunk_ids_for_documents(resolved_document_id, docs)

    replaced_existing = False
    if replace_existing:
        replaced_existing = delete_collection(directory, collection_name)

    try:
        vector_store = get_vector_store(
            resolved_document_id,
            embeddings,
            persist_directory=directory,
        )
        if replace_existing:
            vector_store.add_documents(docs, ids=chunk_ids)
        else:
            existing_ids: set[str] = set()
            if collection_exists(directory, collection_name):
                collection = vector_store._collection  # noqa: SLF001 — MVP introspection
                existing = collection.get(ids=chunk_ids)
                existing_ids = set(existing.get("ids") or [])

            new_docs: list[Document] = []
            new_ids: list[str] = []
            for doc, chunk_id in zip(docs, chunk_ids, strict=True):
                if chunk_id in existing_ids:
                    continue
                new_docs.append(doc)
                new_ids.append(chunk_id)

            if new_docs:
                vector_store.add_documents(new_docs, ids=new_ids)
    except Exception as exc:
        if isinstance(exc, (EmptyDocumentListError, MissingDocumentMetadataError, InvalidDocumentIdError)):
            raise
        raise IndexingFailureError(
            f"Failed to index {len(docs)} chunks for document {resolved_document_id!r}: {exc}"
        ) from exc

    return IndexResult(
        document_id=resolved_document_id,
        collection_name=collection_name,
        chunk_count=len(docs),
        persist_directory=directory,
        replaced_existing=replaced_existing,
        indexed_at=datetime.now(timezone.utc),
    )


def _run_cli() -> None:
    """Minimal debug helper using deterministic fake embeddings."""
    import struct

    from langchain_core.embeddings import Embeddings

    class _DeterministicEmbeddings(Embeddings):
        def __init__(self, dimension: int = 8) -> None:
            self.dimension = dimension

        def _vectorize(self, text: str) -> list[float]:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values: list[float] = []
            for index in range(self.dimension):
                start = (index * 4) % len(digest)
                chunk = digest[start : start + 4]
                if len(chunk) < 4:
                    chunk = (chunk + digest)[:4]
                values.append(struct.unpack("!i", chunk)[0] / 2_147_483_647)
            return values

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._vectorize(text) for text in texts]

        def embed_query(self, text: str) -> list[float]:
            return self._vectorize(text)

    sample_docs = [
        Document(
            page_content="مهلت ارسال پیشنهاد تا پایان ماه است.",
            metadata={
                "document_id": "cli-sample-doc",
                "source": str(Path("sample.pdf").resolve()),
                "page": 1,
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="شرایط مناقصه در این بخش توضیح داده شده است.",
            metadata={
                "document_id": "cli-sample-doc",
                "source": str(Path("sample.pdf").resolve()),
                "page": 2,
                "chunk_index": 1,
            },
        ),
    ]

    result = index_documents(
        sample_docs,
        embedding_function=_DeterministicEmbeddings(),
        persist_directory=settings.chroma_persist_dir / "cli-debug",
        replace_existing=True,
    )
    print("Indexed sample documents:")
    print(f"  document_id: {result.document_id}")
    print(f"  collection_name: {result.collection_name}")
    print(f"  chunk_count: {result.chunk_count}")
    print(f"  persist_directory: {result.persist_directory}")
    print(f"  replaced_existing: {result.replaced_existing}")
    print(f"  indexed_at: {result.indexed_at.isoformat()}")


if __name__ == "__main__":
    _run_cli()
