"""Chroma vector store initialization (Phase 3)."""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from config.settings import settings
from core.exceptions import VectorStoreError

_COLLECTION_NAME_MAX_LEN = 63
_COLLECTION_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{2,62}$")


def sanitize_collection_name(document_id: str) -> str:
    """Convert a document ID into a stable, Chroma-safe collection name."""
    if not document_id or not document_id.strip():
        raise VectorStoreError("document_id must be a non-empty string")

    normalized = document_id.strip()
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", normalized)
    safe = safe.strip("._-")
    if not safe:
        safe = "doc"

    if not safe[0].isalnum():
        safe = f"doc_{safe}"

    if len(safe) > _COLLECTION_NAME_MAX_LEN:
        safe = safe[:_COLLECTION_NAME_MAX_LEN].rstrip("._-")

    if not _COLLECTION_NAME_PATTERN.match(safe):
        raise VectorStoreError(f"Could not derive a valid collection name from {document_id!r}")

    return safe


def ensure_persist_directory(persist_directory: Path | None = None) -> Path:
    """Create the Chroma persist directory when it does not exist."""
    directory = persist_directory or settings.chroma_persist_dir
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VectorStoreError(
            f"Failed to create Chroma persist directory at {directory}: {exc}"
        ) from exc
    return directory


def collection_exists(persist_directory: Path, collection_name: str) -> bool:
    """Return True when a collection already exists in the persist directory."""
    client = chromadb.PersistentClient(path=str(persist_directory))
    existing = {collection.name for collection in client.list_collections()}
    return collection_name in existing


def delete_collection(persist_directory: Path, collection_name: str) -> bool:
    """Delete a collection when present. Returns True if it existed."""
    client = chromadb.PersistentClient(path=str(persist_directory))
    if collection_name not in {collection.name for collection in client.list_collections()}:
        return False
    client.delete_collection(collection_name)
    return True


def get_vector_store(
    document_id: str,
    embedding_function: Embeddings | None = None,
    *,
    persist_directory: Path | None = None,
) -> Chroma:
    """Return a persistent Chroma vector store for the given document ID.

    Args:
        document_id: Logical document identifier used to derive the collection name.
        embedding_function: LangChain-compatible embeddings instance.
        persist_directory: Override for the configured Chroma persist path.

    Raises:
        VectorStoreError: When initialization fails or embeddings are missing.
    """
    if embedding_function is None:
        raise VectorStoreError("embedding_function is required to initialize Chroma")

    directory = ensure_persist_directory(persist_directory)
    collection_name = sanitize_collection_name(document_id)

    try:
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=str(directory),
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to initialize Chroma collection {collection_name!r} at {directory}: {exc}"
        ) from exc
