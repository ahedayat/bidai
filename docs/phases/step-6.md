# Phase 6 — PyQt GUI Report

## Summary

Phase 6 implements a minimal runnable PyQt6 desktop GUI for the Persian tender-document QA MVP. The application lets users select a PDF, index it in a background thread, ask Persian questions, and view answers with retrieved source previews. All OpenAI and Chroma I/O runs off the main UI thread via `QThread` workers. `RAGService` gained a synchronous `index_pdf()` orchestration method used by the ingest worker. Unit tests cover worker instantiation, `index_pdf()` with monkeypatched pipeline components, and widget imports — no real OpenAI API calls or display server required for automated tests.

No Phase 7 polish, PyInstaller packaging, streaming responses, OCR, multi-document search, or new LangGraph tools were added.

## Files Created or Updated

| Path | Action |
|------|--------|
| `services/rag_service.py` | Updated — added `index_pdf()` orchestration method |
| `ui/main_window.py` | Updated — full main window layout and event wiring |
| `ui/widgets/file_picker.py` | Updated — PDF file selection widget |
| `ui/widgets/chat_panel.py` | Updated — RTL-friendly chat history panel |
| `ui/widgets/sources_panel.py` | Updated — retrieved sources display panel |
| `ui/workers.py` | Updated — `IngestWorker`, `QueryWorker`, error formatting |
| `main.py` | Updated — PyQt application entry point |
| `tests/test_workers.py` | Created — worker and `index_pdf()` unit tests |
| `docs/phases/step-6.md` | Created — this report |

## Main Window Behavior

**Module:** `ui/main_window.py`

Layout matches the roadmap:

```
┌─────────────────────────────────────────────┐
│ [Select PDF]  [Process/Index]  Status: ...  │
├──────────────────────────┬──────────────────┤
│ Chat history             │ Retrieved sources│
├──────────────────────────┴──────────────────┤
│ [Question input................] [Send]     │
└─────────────────────────────────────────────┘
```

**State tracked:**

- Selected PDF path (via `FilePicker`)
- `document_id` after successful indexing
- Active ingest/query thread references

**Control enablement:**

| Control | Enabled when |
|---------|--------------|
| Process/Index | PDF selected, not indexing, not querying |
| File picker | Not indexing, not querying |
| Question input / Send | Document indexed, not indexing, not querying |

**Status messages:**

- `No PDF selected` — initial state
- `PDF selected` — after file selection
- `Indexing...` — during ingest (updated with worker progress)
- `Indexing completed` — after successful index
- `Answering...` — during RAG query
- `Ready` — after answer received
- `Indexing failed` / `Query failed` — on errors

**On indexing success:** stores `document_id`, shows a Persian system message in chat, enables question controls.

**On query success:** appends user question and assistant answer to chat, populates sources panel.

## File Picker Behavior

**Module:** `ui/widgets/file_picker.py`

- `Select PDF` button opens a native file dialog filtered to `*.pdf`.
- Displays the selected file name (not the full path) in a label.
- Validates `.pdf` extension before accepting a selection.
- Emits `file_selected(str)` signal with the resolved absolute path.
- Exposes `selected_path` property and `clear()` method.

## Chat Panel Behavior

**Module:** `ui/widgets/chat_panel.py`

- Read-only `QTextEdit` with RTL layout direction.
- HTML-formatted messages with role labels in Persian:
  - `شما` (user) — blue
  - `دستیار` (assistant) — green
  - `سیستم` (system) — amber
- Methods: `add_user_message()`, `add_assistant_message()`, `add_system_message()`, `clear()`.
- Newlines preserved via HTML `<br>` tags.

## Sources Panel Behavior

**Module:** `ui/widgets/sources_panel.py`

- `QTextBrowser` with RTL layout direction.
- `set_sources(sources: list[dict])` renders each source as:
  - Header: `{index}. صفحه {page} | بخش {chunk_index} | {file_name}`
  - Body: `text_preview`
- Sources separated by `---`.
- `clear()` empties the panel.

## Worker / Threading Design

**Module:** `ui/workers.py`

Both workers are `QObject` subclasses moved to dedicated `QThread` instances from `MainWindow`. `RAGService` stays synchronous; workers call it from the background thread.

### `IngestWorker`

| Signal | Payload | When |
|--------|---------|------|
| `progress` | `str` | Status updates during `index_pdf()` |
| `finished` | `str` (document_id) | Indexing succeeded |
| `error` | `str` | User-friendly error message |

Calls `RAGService.index_pdf(path, on_progress=...)`.

### `QueryWorker`

| Signal | Payload | When |
|--------|---------|------|
| `finished` | `(answer: str, sources: list[dict])` | RAG query succeeded |
| `error` | `str` | User-friendly error message |

Calls `RAGService.ask(question, document_id)` and converts `RAGSource` tuples to dicts via `dataclasses.asdict`.

### Error formatting

`format_user_error()` maps application exceptions to readable messages:

- `MissingOpenAIAPIKeyError` → instructions to set `OPENAI_API_KEY`
- `PDFError` → PDF-specific message
- `IndexingError` → indexing failure
- `RetrievalError` → retrieval failure
- `RAGError` / `ChatAPIError` / `GraphInvocationError` → answer generation failure

## How Indexing Is Triggered from the GUI

1. User clicks **Process/Index**.
2. `MainWindow` validates a PDF is selected; shows warning dialog if not.
3. Clears chat/sources, disables controls, sets status to `Indexing...`.
4. Creates `IngestWorker` + `QThread`, connects signals, starts thread.
5. Worker calls `RAGService.index_pdf()` which orchestrates:
   - `load_pdf()` → `chunk_document()` → `index_documents()`
6. Progress signals update the status label.
7. On success: `document_id` stored, status `Indexing completed`, chat enabled.
8. On error: `QMessageBox.critical` with formatted message.

## How Querying Is Triggered from the GUI

1. User types a question and clicks **Send** (or presses Enter).
2. Validates non-empty question and indexed `document_id`.
3. Appends user message to chat, clears sources, sets status `Answering...`.
4. Creates `QueryWorker` + `QThread`, starts thread.
5. Worker calls `RAGService.ask()` → LangGraph retrieve → generate → format_sources.
6. On success: assistant answer in chat, sources in side panel, status `Ready`.
7. On error: `QMessageBox.critical` with formatted message.

## How the GUI Integrates with `RAGService`

**New method:** `RAGService.index_pdf(path, *, on_progress=None) -> IndexResult`

1. Resolves path and builds stable `document_id` via `build_document_id()`.
2. Loads PDF with `load_pdf()`.
3. Rejects PDFs with fewer than 50 total characters (scanned/image-only guard).
4. Chunks with `chunk_document(extracted, document_id=document_id)`.
5. Indexes with `index_documents(docs, document_id=..., embedding_function=..., persist_directory=...)`.
6. Optional `on_progress` callback for worker status updates.

**Existing method:** `RAGService.ask(question, document_id) -> RAGAnswer` — unchanged from Phase 5; invoked by `QueryWorker`.

`MainWindow` constructs a single `RAGService()` instance shared by both workers. No PyQt imports exist in `RAGService`.

## RTL / Persian Display Considerations

- Chat panel and sources panel use `setLayoutDirection(Qt.LayoutDirection.RightToLeft)`.
- Question input field is RTL-oriented.
- Chat messages use `dir="rtl"` in HTML blocks.
- Persian labels for roles (`شما`, `دستیار`, `سیستم`) and placeholders.
- Unicode logical order preserved; no bidi reshaping library needed for MVP.
- Minimal styling (bold headers, role colors) for readability without over-engineering.

## Error Handling Behavior

| Scenario | Behavior |
|----------|----------|
| No PDF selected on index | Warning dialog: "Please select a PDF file first." |
| Invalid / non-PDF file | File picker rejects non-`.pdf` extensions |
| PDF not found / read error | Critical dialog via `format_user_error(PDFError)` |
| Scanned/image PDF (< 50 chars) | `PDFError` with OCR-not-supported message |
| No indexable chunks | `EmptyDocumentListError` dialog |
| Missing OpenAI API key | Dialog with `.env` setup instructions |
| Indexing failure | Critical dialog, status `Indexing failed` |
| Empty question | Warning dialog |
| Query without indexed doc | Warning dialog |
| RAG/retrieval/chat failure | Critical dialog, status `Query failed` |

## Test Cases Added

### `tests/test_workers.py`

| Test | Coverage |
|------|----------|
| `test_format_user_error_missing_api_key` | API key error message |
| `test_format_user_error_pdf_error` | PDF error message |
| `test_ingest_worker_can_be_instantiated` | Worker construction |
| `test_query_worker_can_be_instantiated` | Worker construction |
| `test_gui_widgets_can_be_imported` | Import `MainWindow`, all widgets |
| `test_rag_service_index_pdf_with_monkeypatched_pipeline` | Full `index_pdf()` with mocked loader/chunker/indexer |
| `test_rag_service_index_pdf_rejects_low_text` | Scanned PDF guard (< 50 chars) |
| `test_query_worker_emits_finished_with_sources` | Worker `finished` signal with answer + sources |
| `test_ingest_worker_emits_error_on_failure` | Worker `error` signal on PDF failure |

**GUI testing limitation:** No full Qt widget interaction tests (no `pytest-qt`). Automated tests verify imports and worker logic only. Manual E2E testing is required for visual/interaction validation.

## Commands Used for Smoke Tests and Unit Tests

### Settings import

```bash
PYTHONPATH=. python -c "from config.settings import settings; print(settings)"
```

### Phase 0 smoke tests

```bash
PYTHONPATH=. pytest tests/test_smoke.py -v
```

### Phase 1–5 regression

```bash
PYTHONPATH=. pytest tests/test_pdf_loader.py tests/test_chunker.py tests/test_vector_store.py tests/test_indexer.py tests/test_retriever.py tests/test_rag_service.py tests/test_graph.py -v
```

### Phase 6 worker tests

```bash
PYTHONPATH=. pytest tests/test_workers.py -v
```

### Full suite

```bash
PYTHONPATH=. pytest -v
```

### Manual GUI launch

```bash
PYTHONPATH=. python main.py
```

## Test Results

| Command | Result |
|---------|--------|
| Settings import | **PASS** — defaults: `gpt-4o-mini`, `top_k=4`, `data/chroma` |
| `pytest -v` (full suite) | **PASS** — 89 passed, 0 failed |
| `python main.py` (automated environment) | **NOT RUN** — requires a display server; manual launch documented below |

## Manual E2E Test Checklist

- [ ] Run `python main.py`.
- [ ] Select a Persian tender PDF.
- [ ] Click **Process/Index**.
- [ ] Confirm the UI remains responsive during indexing.
- [ ] Confirm status changes to indexing/completed.
- [ ] Ask a question such as: `مهلت ارسال پیشنهاد چیست؟`
- [ ] Confirm the answer appears in the chat.
- [ ] Confirm retrieved sources appear with page numbers.
- [ ] Try an invalid PDF or missing API key and confirm error handling.

## Assumptions and Deviations

1. **50-character minimum for indexing** — Added a guard in `index_pdf()` to reject scanned/image PDFs early with a clear error; not explicitly in the roadmap but aligns with the OCR-not-supported risk mitigation.
2. **`on_progress` callback on `index_pdf()`** — Optional callback for granular status updates without adding PyQt to the service layer.
3. **File name display** — File picker shows file name only, not full path, for cleaner layout.
4. **Single `RAGService` instance** — Shared across workers; graph compiled lazily on first `ask()`.
5. **No `pytest-qt`** — GUI interaction tests deferred; import and worker unit tests only.
6. **`PYTHONPATH=.` for local runs** — Same approach as Phases 1–5.

## Known Limitations

- **No streaming** — Full answer appears after RAG completes.
- **No multi-turn memory** — Each question is independent; chat history is display-only.
- **No OCR** — Scanned PDFs rejected when text extraction yields < 50 characters.
- **Single document per session** — Re-indexing clears chat; no multi-document search.
- **No automated GUI tests** — Requires manual E2E validation with a display server.
- **No progress bar** — Status label text only during indexing.
- **Chat history not sent to model** — Display panel only; model sees one question at a time.
- **Re-indexing same file** — Uses path-hash `document_id`; replace mode in indexer overwrites collection.

## What Remains for Phase 7

Phase 7 — **Polish & MVP hardening**:

- README with setup/run steps
- `.env.example` documentation
- Basic logging to file/console
- Optional: clear index / new document button
- Fresh-machine setup validation
- Demo script for stakeholders
