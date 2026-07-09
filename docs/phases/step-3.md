# Phase 3 — OpenAI Embeddings + Chroma Indexing Report

## Summary

Phase 3 implements local vector indexing for chunked LangChain documents. A thin OpenAI embeddings factory reads model and API key settings from the environment, Chroma is initialized with persistent local storage per document, and the indexer validates chunk metadata, assigns stable document/chunk IDs, embeds documents, and persists vectors with idempotent re-indexing support. Unit tests use a deterministic fake embeddings class so no real OpenAI API calls or API keys are required. A minimal CLI debug helper is available via `python -m ingestion.indexer`.

No retrieval wrapper, RAG answer generation, LangGraph, PyQt UI, or OpenAI Chat API calls were added.

## Files Created or Updated

| Path | Action |
|------|--------|
| `services/openai_client.py` | Updated — `create_embeddings()` wrapper |
| `retrieval/vector_store.py` | Updated — Chroma init, collection sanitization, persistence helpers |
| `ingestion/indexer.py` | Updated — `build_document_id`, `build_chunk_id`, `index_documents`, CLI helper |
| `core/models.py` | Updated — `IndexResult` dataclass |
| `core/exceptions.py` | Updated — indexing-related custom exceptions |
| `tests/fake_embeddings.py` | Created — deterministic `FakeEmbeddings` for tests |
| `tests/test_vector_store.py` | Created — vector store unit tests |
| `tests/test_indexer.py` | Updated — full indexer unit test suite |
| `docs/phases/step-3.md` | Created — this report |

## OpenAI Embeddings Wrapper Behavior

**Entry point:** `create_embeddings(*, api_key=None, model=None) -> OpenAIEmbeddings`

- Uses `langchain_openai.OpenAIEmbeddings`.
- Reads `EMBEDDING_MODEL` from `config.settings` (default: `text-embedding-3-small`).
- Reads `OPENAI_API_KEY` from environment / `.env` via settings unless `api_key` is passed explicitly.
- Raises `MissingOpenAIAPIKeyError` when no non-empty API key is available.
- Thin factory only — no chat client, no hardcoded secrets.

## Chroma Vector Store Behavior

**Key functions in `retrieval/vector_store.py`:**

| Function | Role |
|----------|------|
| `sanitize_collection_name(document_id)` | Derive a stable, Chroma-safe collection name |
| `ensure_persist_directory(persist_directory)` | Create persist directory when missing |
| `collection_exists(persist_directory, collection_name)` | Check whether a collection exists |
| `delete_collection(persist_directory, collection_name)` | Delete an existing collection |
| `get_vector_store(document_id, embedding_function, ...)` | Return a persistent `Chroma` instance |

- Uses `langchain_chroma.Chroma` with `chromadb.PersistentClient` under the hood.
- Default persist path: `settings.chroma_persist_dir` (`data/chroma` relative to project root).
- Creates the persist directory automatically when it does not exist.
- Requires an explicit `embedding_function` at initialization time.
- Wraps Chroma initialization failures as `VectorStoreError`.

## Indexing Strategy

**Entry point:** `index_documents(docs, document_id=None, *, replace_existing=True, embedding_function=None, persist_directory=None) -> IndexResult`

1. Validate the document list is non-empty and each document has non-empty `page_content`.
2. Validate required metadata on every document: `document_id`, `source`, `page`, `chunk_index`.
3. Resolve `document_id` from the argument or infer uniformly from document metadata.
4. Resolve embeddings via `embedding_function` override or `create_embeddings()` (OpenAI).
5. Ensure the Chroma persist directory exists.
6. Build deterministic chunk IDs from `document_id` + `chunk_index`.
7. When `replace_existing=True`, delete the target collection if it already exists, then add all chunks with stable IDs.
8. When `replace_existing=False`, skip chunk IDs that already exist in the collection; add only new chunks.
9. Return `IndexResult` with collection metadata and UTC timestamp.

## Document ID and Chunk ID Strategy

### Document ID

`build_document_id(source_path)` hashes the resolved absolute path:

```
SHA-256(resolved_path)[:16]   # 16-char hex string
```

This is stable across repeated runs for the same file path. Phase 2 chunker may still use `file_name` as `document_id` in metadata; the indexer accepts either explicit IDs or metadata-inferred IDs.

### Chunk ID

`build_chunk_id(document_id, chunk_index)`:

```
{document_id}__chunk__{chunk_index}
```

Example: `test-doc-abc123__chunk__0`

These IDs are passed to `Chroma.add_documents(..., ids=...)` for idempotent storage and re-indexing.

## Collection Naming Strategy

`sanitize_collection_name(document_id)`:

1. Strip whitespace from the document ID.
2. Replace invalid characters with `_` (allowed: letters, digits, `.`, `_`, `-`).
3. Strip leading/trailing `._-`.
4. Fallback to `doc` when nothing remains.
5. Prefix `doc_` when the first character is not alphanumeric.
6. Truncate to 63 characters (Chroma collection name limit).
7. Validate against `^[a-zA-Z0-9][a-zA-Z0-9._-]{2,62}$`.

Example: `abc/def:ghi` → `abc_def_ghi`; `!!!` → `doc`.

## Persistence Directory Behavior

- Configured via `CHROMA_PERSIST_DIR` (default `data/chroma`).
- Relative paths are resolved against the project root.
- `ensure_persist_directory()` calls `mkdir(parents=True, exist_ok=True)`.
- Tests use `tmp_path` overrides so no production Chroma data is written during unit tests.
- CLI debug helper writes to `data/chroma/cli-debug` (gitignored under `data/chroma/*`).

## Re-indexing / Idempotency Behavior

| Mode | Behavior |
|------|----------|
| `replace_existing=True` | Delete existing collection (if any), recreate, insert all chunks — count stays stable on re-run |
| `replace_existing=False` | Query existing chunk IDs; skip duplicates; append only new IDs |

`IndexResult.replaced_existing` is `True` only when a prior collection was deleted during a replace run.

## Error Handling Added

| Exception | When raised |
|-----------|-------------|
| `IndexingError` | Base class for indexing failures |
| `EmptyDocumentListError` | Empty `docs` list |
| `MissingDocumentMetadataError` | Missing required metadata or empty `page_content` |
| `InvalidDocumentIdError` | Invalid / conflicting / missing document ID |
| `VectorStoreError` | Chroma init failure, bad collection name, persist dir failure |
| `IndexingFailureError` | Failure during `add_documents` |
| `MissingOpenAIAPIKeyError` | `create_embeddings()` called without a valid API key |

All inherit from `BidaiError` (indexing errors via `IndexingError`).

## Test Cases Added

### `tests/test_vector_store.py`

| Test | Coverage |
|------|----------|
| `test_sanitize_collection_name_strips_invalid_characters` | Invalid chars replaced |
| `test_sanitize_collection_name_prefixes_non_alnum_start` | Punctuation-only IDs and leading punctuation |
| `test_sanitize_collection_name_truncates_long_ids` | Max length enforcement |
| `test_sanitize_collection_name_rejects_empty` | Empty document ID rejected |
| `test_ensure_persist_directory_creates_path` | Directory creation |
| `test_get_vector_store_creates_persistent_collection` | Chroma init in temp dir |
| `test_get_vector_store_requires_embedding_function` | Missing embeddings rejected |

### `tests/test_indexer.py`

| Test | Coverage |
|------|----------|
| `test_build_document_id_is_stable` | Deterministic path-based ID |
| `test_build_chunk_id_is_stable` | Deterministic chunk IDs |
| `test_build_chunk_id_rejects_invalid_inputs` | Invalid ID / index rejected |
| `test_index_documents_creates_searchable_collection` | Index + similarity search |
| `test_index_documents_preserves_metadata` | Metadata round-trip |
| `test_reindex_with_replace_existing_does_not_grow` | Replace mode idempotency |
| `test_reindex_without_replace_skips_duplicate_ids` | Append mode deduplication |
| `test_index_documents_infers_document_id_from_metadata` | Metadata inference |
| `test_index_documents_raises_on_empty_list` | `EmptyDocumentListError` |
| `test_index_documents_raises_on_missing_metadata` | `MissingDocumentMetadataError` |
| `test_index_documents_raises_on_empty_page_content` | Empty content rejected |
| `test_index_documents_raises_on_conflicting_document_ids` | `InvalidDocumentIdError` |
| `test_create_embeddings_requires_api_key` | `MissingOpenAIAPIKeyError` |

**Fake embeddings:** `tests/fake_embeddings.FakeEmbeddings` returns deterministic 8-dimensional vectors derived from SHA-256 of the input text. No network calls.

## Commands Used for Smoke Tests and Unit Tests

### Settings import (Phase 0)

```bash
PYTHONPATH=. python -c "from config.settings import settings; print(settings)"
```

### Phase 0 smoke tests

```bash
PYTHONPATH=. pytest tests/test_smoke.py -v
```

### Phase 1 PDF loader tests

```bash
PYTHONPATH=. pytest tests/test_pdf_loader.py -v
```

### Phase 2 chunker tests

```bash
PYTHONPATH=. pytest tests/test_chunker.py -v
```

### Phase 3 vector store tests

```bash
PYTHONPATH=. pytest tests/test_vector_store.py -v
```

### Phase 3 indexer tests

```bash
PYTHONPATH=. pytest tests/test_indexer.py -v
```

### Combined Phase 0–3 run

```bash
PYTHONPATH=. pytest tests/test_smoke.py tests/test_pdf_loader.py tests/test_chunker.py tests/test_vector_store.py tests/test_indexer.py -v
```

### Full test suite

```bash
PYTHONPATH=. pytest -v
```

### Indexer CLI debug helper

```bash
PYTHONPATH=. python -m ingestion.indexer
```

## Test Results

| Command | Result |
|---------|--------|
| Settings import | **PASS** — defaults loaded (`text-embedding-3-small`, `data/chroma`, etc.) |
| `pytest tests/test_smoke.py tests/test_pdf_loader.py tests/test_chunker.py tests/test_vector_store.py tests/test_indexer.py -v` | **PASS** — 48 passed, 0 failed |
| `pytest -v` | **PASS** — 48 passed, 0 failed |
| `python -m ingestion.indexer` | **PASS** — 2 sample chunks indexed under `data/chroma/cli-debug` |

## Assumptions and Deviations

1. **Document ID from path vs. metadata** — `build_document_id()` uses a path hash for service-layer use; indexer primarily uses `document_id` from Phase 2 chunk metadata (default `file_name`). Both are supported.
2. **Collection name equals sanitized document ID** — One collection per `document_id`, not per session UUID.
3. **`replace_existing=False` dedup** — Uses Chroma `collection.get(ids=...)` to detect existing chunk IDs; does not update changed content for the same ID (acceptable for MVP append-only use).
4. **Fake embeddings for similarity tests** — Query uses exact chunk text because hash-based vectors are only identical for identical strings; this still validates Chroma search plumbing without OpenAI.
5. **CLI writes to `cli-debug` subfolder** — Keeps debug data separate from production collections; still under gitignored `data/chroma/`.
6. **`PYTHONPATH=.` for local runs** — Same approach as Phases 1–2.

## Known Limitations

- **No batching** — All chunks are embedded and added in one `add_documents` call; large documents may be slow.
- **No embedding dimension validation** — Switching embedding models on an existing collection is not handled.
- **Append mode does not upsert** — Re-indexing changed text with `replace_existing=False` and the same chunk ID will skip the update.
- **Chroma internal API** — Duplicate detection uses `vector_store._collection.get()` (documented MVP trade-off).
- **OpenAI required for production indexing** — Real indexing needs `OPENAI_API_KEY`; tests do not.

## What Remains for Phase 4

Phase 4 — **Retriever + basic RAG (no graph yet)**:

- Implement `retrieval/retriever.py` — LangChain retriever wrapper with top-k and metadata filters
- Implement `agent/prompts.py` — Persian system/user prompt templates
- Implement `services/rag_service.py` — orchestrate retrieve + generate (no LangGraph yet)
- CLI or script: ask a Persian question against an indexed document
- Add `tests/test_retriever.py` and integration tests for retrieve + generate
- Hallucination guard: answer only from retrieved context
