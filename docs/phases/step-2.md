# Phase 2 — Persian-Aware Document Chunking Report

## Summary

Phase 2 implements Persian-aware document chunking on top of the Phase 1 `ExtractedDocument` model. The chunker converts non-empty extracted pages into LangChain `Document` objects using `RecursiveCharacterTextSplitter` with Persian/Arabic punctuation separators, configurable chunk size and overlap (from settings or call-time overrides), and rich per-chunk metadata for downstream indexing and retrieval. Validation guards invalid chunk parameters and documents with no pages. Empty pages are skipped; documents where every page is empty return an empty list. Unit tests cover chunking behavior, metadata, ordering, validation, and Persian separator preferences. A minimal CLI debug helper is available via `python -m ingestion.chunker`.

No embeddings, Chroma indexing, retrieval, RAG, LangGraph, OpenAI calls, or PyQt UI were added.

## Files Created or Updated

| Path | Action |
|------|--------|
| `ingestion/chunker.py` | Updated — `chunk_document`, Persian separators, CLI helper |
| `core/exceptions.py` | Updated — `ChunkingError`, `InvalidChunkConfigError`, `EmptyExtractedDocumentError` |
| `tests/test_chunker.py` | Updated — full unit test suite |
| `docs/phases/step-2.md` | Created — this report |

## Chunking Strategy

**Entry point:** `chunk_document(extracted, *, document_id=None, chunk_size=None, chunk_overlap=None) -> list[Document]`

1. Validate that the extracted document has at least one page; raise `EmptyExtractedDocumentError` if `pages` is empty.
2. Resolve `chunk_size` and `chunk_overlap` from explicit arguments or `config.settings` (defaults: 1000 / 150).
3. Validate chunk parameters (positive size, non-negative overlap, overlap < size).
4. Build a `RecursiveCharacterTextSplitter` with Persian-aware separators.
5. Iterate pages in order; **skip empty pages** (`is_empty` or whitespace-only text).
6. Split each non-empty page independently — this keeps `page` metadata accurate and prevents one page's text from dominating chunk boundaries across the whole document.
7. Filter out empty split fragments.
8. Attach metadata to each LangChain `Document` with stable global `chunk_index` and per-page `page_chunk_index`.
9. If all pages are empty, return `[]` (no exception).

**Default `document_id`:** uses `extracted.file_name` when not provided explicitly.

## Persian-Aware Separators Used

Configured in `PERSIAN_SEPARATORS` (highest priority first):

| Separator | Role |
|-----------|------|
| `\n\n` | Paragraph break |
| `\n` | Line break |
| `؟` | Arabic question mark (Persian) |
| `?` | Latin question mark |
| `!` | Exclamation |
| `۔` | Arabic full stop (common in Persian texts) |
| `.` | Latin period |
| `؛` | Arabic semicolon |
| `،` | Arabic comma |
| ` ` | Word boundary (space) |
| `""` | Character-level fallback |

`RecursiveCharacterTextSplitter` tries larger separators first and recursively falls back to finer ones, respecting `chunk_size` and `chunk_overlap`.

## Default Chunk Size and Overlap Behavior

| Setting | Source | Default |
|---------|--------|---------|
| `chunk_size` | `CHUNK_SIZE` env / `settings.chunk_size` | `1000` |
| `chunk_overlap` | `CHUNK_OVERLAP` env / `settings.chunk_overlap` | `150` |

Call-time `chunk_size` and `chunk_overlap` override settings when provided. Internal fallbacks (`_DEFAULT_CHUNK_SIZE`, `_DEFAULT_CHUNK_OVERLAP`) mirror these values for documentation clarity; runtime resolution always goes through `settings` unless overridden.

## Metadata Included in Each LangChain Document

| Key | Description |
|-----|-------------|
| `document_id` | Caller-provided ID or `file_name` default |
| `source` | Absolute path string of the source PDF |
| `file_name` | Base file name |
| `page` | 1-based source page number |
| `chunk_index` | Stable global index across the document (0-based) |
| `page_chunk_index` | Index within the current page (0-based) |
| `char_count` | Length of `page_content` |

## Validation Rules Added

| Rule | Behavior |
|------|----------|
| `chunk_size` must be positive | Raises `InvalidChunkConfigError` |
| `chunk_overlap` must be non-negative | Raises `InvalidChunkConfigError` |
| `chunk_overlap` < `chunk_size` | Raises `InvalidChunkConfigError` |
| Extracted document has no pages | Raises `EmptyExtractedDocumentError` |
| All pages empty | Returns empty list `[]` |

## Test Cases Added

| Test | Coverage |
|------|----------|
| `test_chunk_simple_extracted_document` | Basic multi-page chunking |
| `test_chunk_skips_empty_pages` | Empty pages excluded from output |
| `test_chunk_handles_short_page` | Single short page → one chunk |
| `test_chunk_metadata_preservation` | All required metadata keys |
| `test_chunk_index_ordering_is_stable` | Monotonic `chunk_index` and per-page indices |
| `test_chunk_default_document_id_uses_file_name` | Default `document_id` |
| `test_chunk_invalid_chunk_size` | Zero/negative size rejected |
| `test_chunk_invalid_chunk_overlap` | Negative overlap and overlap ≥ size rejected |
| `test_chunk_raises_when_no_pages` | `EmptyExtractedDocumentError` |
| `test_chunk_all_empty_pages_returns_empty_list` | All-empty document → `[]` |
| `test_chunk_persian_punctuation_separators` | Multi-chunk splits on Persian `؟` |
| `test_chunk_chunks_are_non_empty` | No blank chunks |
| `test_chunk_size_respected_approximately` | Chunks stay near `chunk_size` |
| `test_chunk_avoids_single_chunk_for_whole_document` | Multi-page docs produce multiple chunks |
| `test_chunk_uses_settings_defaults` | Settings integration |
| `test_persian_separators_include_required_tokens` | Separator list completeness |

Tests construct `ExtractedDocument` / `ExtractedPage` directly in memory — no PDF fixtures required.

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

### Combined run

```bash
PYTHONPATH=. pytest tests/test_smoke.py tests/test_pdf_loader.py tests/test_chunker.py -v
```

### Full test suite

```bash
PYTHONPATH=. pytest -v
```

### Chunker CLI debug helper

```bash
PYTHONPATH=. python -m ingestion.chunker
```

## Test Results

| Command | Result |
|---------|--------|
| Settings import | **PASS** — `chunk_size=1000`, `chunk_overlap=150` |
| `pytest tests/test_smoke.py tests/test_pdf_loader.py tests/test_chunker.py -v` | **PASS** — 28 passed, 0 failed |
| `pytest -v` | **PASS** — 28 passed, 0 failed |
| `python -m ingestion.chunker` | **PASS** — 2 chunks from sample doc (empty page skipped) |

## Assumptions and Deviations

1. **Per-page chunking** — Pages are split independently rather than concatenating the full document first. This preserves accurate `page` metadata and avoids cross-page mega-chunks; slight clause context at page boundaries is recovered via overlap within each page.
2. **All-empty pages → empty list** — Chosen over raising an exception so Phase 3 indexing can treat it as a no-op gracefully.
3. **No pages at all → exception** — Distinct from all-empty: a structurally invalid document raises `EmptyExtractedDocumentError`.
4. **`document_id` default** — Uses `file_name` (not a hash); Phase 3 can supply a hash-based ID at the service layer.
5. **`PYTHONPATH=.` for local runs** — Same approach as Phase 1; editable install is configured in `pyproject.toml` but tests were run with `PYTHONPATH=.`.

## Known Limitations

- **No cross-page overlap** — Overlap applies within a single page's text only; a clause split across a page break may lose local context (acceptable for MVP; page numbers remain correct).
- **Indivisible segments** — If a segment between separators exceeds `chunk_size`, `RecursiveCharacterTextSplitter` may emit a chunk larger than `chunk_size` (handled by character-level fallback).
- **No semantic chunking** — Purely character/recursive splitting; no sentence embedding or NLP boundary detection.
- **Whitespace-only pages** — Treated as empty via `text.strip()` check, even if `is_empty` flag were incorrectly set.

## What Remains for Phase 3

Phase 3 — **Embeddings + Chroma indexing**:

- Implement `services/openai_client.py` (embedding wrapper)
- Implement `ingestion/indexer.py` — `index_documents(docs, document_id)`
- Implement `retrieval/vector_store.py` — local persistent Chroma collection per `document_id`
- Wire chunk metadata into vector store IDs for idempotent re-indexing
- Add `tests/test_indexer.py` with mocked or live embedding integration tests
