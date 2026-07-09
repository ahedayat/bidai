# Phase 4 — Retriever + Basic RAG Report

## Summary

Phase 4 implements synchronous retrieve-and-generate RAG without LangGraph. A LangChain retriever wrapper loads indexed Chroma collections per `document_id`, Persian prompt templates enforce context-only answers with citation expectations, and `RAGService` orchestrates retrieval, context formatting, and OpenAI Chat API answer generation. Unit tests use deterministic fake embeddings and a fake chat model — no real OpenAI API calls or API keys are required.

No LangGraph state/nodes/graph, PyQt UI, streaming, multi-turn memory, or additional tools were added.

## Files Created or Updated

| Path | Action |
|------|--------|
| `retrieval/retriever.py` | Updated — `get_retriever`, `retrieve_documents` |
| `agent/prompts.py` | Updated — Persian system/user RAG prompt templates |
| `services/rag_service.py` | Updated — `RAGService`, formatting helpers, optional CLI |
| `services/openai_client.py` | Updated — `create_chat_model()` |
| `core/models.py` | Updated — `RAGSource`, `RAGAnswer` |
| `core/exceptions.py` | Updated — retrieval and RAG exceptions; `MissingOpenAIAPIKeyError` moved to `BidaiError` |
| `tests/fake_chat.py` | Created — `FakeChatModel` for tests |
| `tests/test_retriever.py` | Updated — full retriever unit test suite |
| `tests/test_rag_service.py` | Created — RAG service and formatting tests |
| `docs/phases/step-4.md` | Created — this report |

## Retriever Behavior

**Entry points:** `get_retriever(document_id, *, top_k=None, ...)` and `retrieve_documents(question, document_id, *, top_k=None, ...)`

1. Validate `document_id` is non-empty (`InvalidDocumentIdError`).
2. Resolve `top_k` from argument or `settings.top_k` (default 4); reject non-positive values (`InvalidTopKError`).
3. Verify the Chroma collection exists for the document (`CollectionNotFoundError`).
4. Load embeddings via `create_embeddings()` unless an override is provided (tests use `FakeEmbeddings`).
5. Open the persistent Chroma vector store via `get_vector_store()`.
6. Return `vectorstore.as_retriever(search_kwargs={"k": top_k})`.
7. `retrieve_documents` additionally rejects empty questions (`EmptyQuestionError`) and returns LangChain `Document` objects with metadata preserved.

Optional `persist_directory` and `embedding_function` overrides support isolated test collections.

## Persian RAG Prompt Design

**Module:** `agent/prompts.py`

### System prompt (`RAG_SYSTEM_PROMPT`)

- Defines the assistant as a Persian tender/procurement document expert.
- Requires answers **only** from retrieved context.
- Mandates a clear refusal phrase when the answer is not in context: «پاسخ این پرسش در سند مناقصه ارائه‌شده یافت نشد.»
- Forbids inventing dates, numbers, deadlines, prices, requirements, or legal conditions.
- Encourages citing source page numbers when possible.
- Keeps answers concise but complete.

### User prompt (`RAG_USER_PROMPT`)

- Includes a `{context}` block (formatted retrieved chunks) and `{question}`.
- Reminds the model to refuse when the answer is absent from context.

Combined as `RAG_PROMPT` (`ChatPromptTemplate`) for use with `ChatOpenAI.invoke()`.

## RAG Service Behavior

**Class:** `RAGService`

**Entry point:** `ask(question, document_id, *, top_k=None) -> RAGAnswer`

1. Reject empty questions (`EmptyQuestionError`).
2. Retrieve top-k chunks via `retrieve_documents()`.
3. Raise `NoRetrievedDocumentsError` if retrieval returns nothing.
4. Format context with `format_context_block()` (page/chunk headers per chunk).
5. Format sources with `format_sources()` → `RAGSource` tuples.
6. Build messages from `RAG_PROMPT` and invoke `ChatOpenAI` (or injected fake model).
7. Return `RAGAnswer` with question, answer, document_id, sources, retrieved_count, and model_name.

Constructor accepts optional `chat_model`, `embedding_function`, and `persist_directory` for testing and future wiring.

**CLI (optional, manual only):**

```bash
PYTHONPATH=. python -m services.rag_service \
  --document-id <id> \
  --question "مهلت ارسال پیشنهاد چیست؟"
```

Requires a real indexed document and `OPENAI_API_KEY` in the environment.

## OpenAI Chat API Integration Approach

**Factory:** `create_chat_model(*, api_key=None, model=None, temperature=0.0) -> ChatOpenAI`

- Uses `langchain_openai.ChatOpenAI`.
- Reads `CHAT_MODEL` from settings (default: `gpt-4o-mini`).
- Reads `OPENAI_API_KEY` from environment / `.env` unless overridden.
- Raises `MissingOpenAIAPIKeyError` when no key is available.
- `temperature=0.0` for deterministic, fact-grounded answers.
- `RAGService` lazily creates the chat model on first `ask()` unless injected.
- Unit tests inject `FakeChatModel`; no network calls in CI.

## Source Formatting Strategy

**Function:** `format_source(doc) -> RAGSource`

Each source includes:

| Field | Source |
|-------|--------|
| `page` | `metadata["page"]` |
| `chunk_index` | `metadata["chunk_index"]` |
| `page_chunk_index` | `metadata["page_chunk_index"]` if present |
| `source` | `metadata["source"]` (absolute path string) |
| `file_name` | `metadata["file_name"]` if present |
| `text_preview` | Truncated `page_content` (default 200 chars, ellipsis suffix) |

`format_sources()` maps a list of retrieved documents to a tuple of `RAGSource` objects.

## Context Formatting Strategy

**Function:** `format_context_block(docs) -> str`

- Each chunk is prefixed with a Persian metadata header: `[صفحه N | بخش M | زیربخش صفحه P | فایل: name]`.
- Chunk bodies are included in full up to 4000 characters per chunk (truncation with `…` only for extreme lengths).
- Chunks are separated by `\n\n---\n\n`.
- Format is structured for reuse in LangGraph `generate` / `format_sources` nodes (Phase 5).

## Error Handling Added

| Exception | Base | When raised |
|-----------|------|-------------|
| `MissingOpenAIAPIKeyError` | `BidaiError` | No API key for embeddings or chat (moved from `IndexingError`) |
| `RetrievalError` | `BidaiError` | Base for retrieval failures |
| `EmptyQuestionError` | `RetrievalError` | Empty or whitespace-only question |
| `InvalidTopKError` | `RetrievalError` | `top_k` ≤ 0 |
| `CollectionNotFoundError` | `RetrievalError` | No indexed collection for `document_id` |
| `NoRetrievedDocumentsError` | `RetrievalError` | Retrieval returned zero chunks |
| `RAGError` | `BidaiError` | Base for answer-generation failures |
| `ChatAPIError` | `RAGError` | OpenAI Chat API invocation failure |

Existing exceptions reused: `InvalidDocumentIdError`, `VectorStoreError`, `RetrievalError` (wrapper for retriever invoke failures).

## Test Cases Added

### `tests/test_retriever.py`

| Test | Coverage |
|------|----------|
| `test_get_retriever_with_valid_document_id` | Retriever creation with indexed collection |
| `test_get_retriever_respects_top_k_override` | Custom `top_k` |
| `test_get_retriever_rejects_invalid_document_id` | Empty document ID |
| `test_get_retriever_rejects_invalid_top_k` | Non-positive `top_k` |
| `test_get_retriever_raises_when_collection_missing` | `CollectionNotFoundError` |
| `test_retrieve_documents_rejects_empty_question` | `EmptyQuestionError` |
| `test_retrieve_documents_returns_results` | Retrieval from temp Chroma with fake embeddings |
| `test_retrieve_documents_preserves_metadata` | Metadata round-trip |

### `tests/test_rag_service.py`

| Test | Coverage |
|------|----------|
| `test_format_source_includes_required_fields` | Source field mapping |
| `test_format_source_truncates_long_preview` | Preview truncation |
| `test_format_sources_returns_tuple` | Multi-source formatting |
| `test_format_context_block_includes_page_and_chunk_metadata` | Context headers |
| `test_rag_service_returns_answer_and_sources` | End-to-end with fake chat model |
| `test_rag_service_rejects_empty_question` | Empty question guard |
| `test_rag_service_unknown_document_raises` | Missing collection |
| `test_rag_service_no_context_handled_by_fake_model` | Unknown-answer refusal text |
| `test_rag_service_chat_api_failure` | `ChatAPIError` on model failure |
| `test_create_chat_model_requires_api_key` | `MissingOpenAIAPIKeyError` |

**Test helpers:** `tests/fake_embeddings.py` (Phase 3), `tests/fake_chat.py` (Phase 4).

## Commands Used for Smoke Tests and Unit Tests

### Settings import

```bash
PYTHONPATH=. python -c "from config.settings import settings; print(settings)"
```

### Phase 0 smoke tests

```bash
PYTHONPATH=. pytest tests/test_smoke.py -v
```

### Phase 1–3 regression

```bash
PYTHONPATH=. pytest tests/test_pdf_loader.py tests/test_chunker.py tests/test_vector_store.py tests/test_indexer.py -v
```

### Phase 4 tests

```bash
PYTHONPATH=. pytest tests/test_retriever.py tests/test_rag_service.py -v
```

### Full suite

```bash
PYTHONPATH=. pytest -v
```

### Optional RAG CLI (requires API key + indexed document)

```bash
PYTHONPATH=. python -m services.rag_service \
  --document-id <document_id> \
  --question "مهلت ارسال پیشنهاد چیست؟"
```

## Test Results

| Command | Result |
|---------|--------|
| Settings import | **PASS** — defaults: `gpt-4o-mini`, `top_k=4`, `data/chroma` |
| `pytest tests/test_smoke.py tests/test_pdf_loader.py tests/test_chunker.py tests/test_vector_store.py tests/test_indexer.py tests/test_retriever.py tests/test_rag_service.py -v` | **PASS** — 66 passed, 0 failed |
| `pytest -v` | **PASS** — 66 passed, 0 failed |

## Assumptions and Deviations

1. **`MissingOpenAIAPIKeyError` base class** — Moved from `IndexingError` to `BidaiError` so it applies to both embeddings and chat without coupling RAG to indexing.
2. **`RAGService` embedding override** — Constructor accepts `embedding_function` for test isolation; production path uses `create_embeddings()` automatically.
3. **Fake embedding similarity** — Retrieval tests query with exact chunk text (hash-based fake vectors are identical only for identical strings), matching the Phase 3 indexer test pattern.
4. **No `NoRetrievedDocumentsError` in normal flow** — With `top_k ≥ 1` and a non-empty collection, Chroma typically returns results; the guard exists for defensive completeness.
5. **Context chunk cap** — Per-chunk context limit of 4000 characters prevents runaway prompt size on large chunks; previews use 200 characters.
6. **`PYTHONPATH=.` for local runs** — Same approach as Phases 1–3.

## Known Limitations

- **No LangGraph** — Linear retrieve → generate logic lives in `RAGService.ask()`; Phase 5 will refactor into graph nodes.
- **No streaming** — Full answer returned synchronously after Chat API completes.
- **No multi-turn memory** — Each `ask()` is stateless; no conversation history passed to the model.
- **No hybrid / reranker search** — Pure vector similarity with fixed `top_k`.
- **Fake embeddings do not model semantic similarity** — Real OpenAI embeddings needed for meaningful Persian similarity in production.
- **Temperature fixed at 0.0** — Not configurable via settings yet.
- **CLI requires real API key and indexed document** — Not exercised in automated tests.

## What Remains for Phase 5

Phase 5 — **LangGraph agent wrapper**:

- Implement `agent/state.py` — `QAState` TypedDict
- Implement `agent/nodes.py` — `retrieve`, `generate`, `format_sources` nodes
- Implement `agent/graph.py` — linear `StateGraph` compile
- Refactor `RAGService.ask()` to call `compiled_graph.invoke()`
- Add `tests/test_graph.py` with mocked LLM/retriever
- Reuse `agent/prompts.py`, retriever, and formatting helpers from Phase 4
