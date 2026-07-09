# Phase 0 — Project Bootstrap Report

## Summary

Phase 0 bootstrapped the Persian tender-document QA MVP project skeleton. The repository now has dependency configuration, environment variable templates, settings loading, empty module stubs matching the roadmap layout, a minimal `main.py` entry stub, data directories, and smoke tests to verify imports and default configuration.

No business logic, PDF extraction, RAG, LangGraph, or PyQt UI was implemented.

## Files and Directories Created or Updated

### Configuration and project root

| Path | Action |
|------|--------|
| `pyproject.toml` | Created |
| `.env.example` | Created |
| `.gitignore` | Created |
| `main.py` | Created |
| `config/__init__.py` | Created |
| `config/settings.py` | Created |

### Core modules (empty stubs)

| Path | Action |
|------|--------|
| `core/__init__.py` | Created |
| `core/models.py` | Created |
| `core/exceptions.py` | Created |

### Ingestion (empty stubs)

| Path | Action |
|------|--------|
| `ingestion/__init__.py` | Created |
| `ingestion/pdf_loader.py` | Created |
| `ingestion/chunker.py` | Created |
| `ingestion/indexer.py` | Created |

### Retrieval (empty stubs)

| Path | Action |
|------|--------|
| `retrieval/__init__.py` | Created |
| `retrieval/vector_store.py` | Created |
| `retrieval/retriever.py` | Created |

### Agent (empty stubs)

| Path | Action |
|------|--------|
| `agent/__init__.py` | Created |
| `agent/state.py` | Created |
| `agent/nodes.py` | Created |
| `agent/graph.py` | Created |
| `agent/prompts.py` | Created |

### Services (empty stubs)

| Path | Action |
|------|--------|
| `services/__init__.py` | Created |
| `services/openai_client.py` | Created |
| `services/rag_service.py` | Created |

### UI (empty stubs)

| Path | Action |
|------|--------|
| `ui/__init__.py` | Created |
| `ui/main_window.py` | Created |
| `ui/workers.py` | Created |
| `ui/widgets/__init__.py` | Created |
| `ui/widgets/file_picker.py` | Created |
| `ui/widgets/chat_panel.py` | Created |
| `ui/widgets/sources_panel.py` | Created |

### Data directories

| Path | Action |
|------|--------|
| `data/chroma/.gitkeep` | Created |
| `data/uploads/.gitkeep` | Created |

### Tests

| Path | Action |
|------|--------|
| `tests/__init__.py` | Created |
| `tests/test_smoke.py` | Created |
| `tests/test_pdf_loader.py` | Created (placeholder) |
| `tests/test_chunker.py` | Created (placeholder) |
| `tests/test_indexer.py` | Created (placeholder) |
| `tests/test_retriever.py` | Created (placeholder) |
| `tests/test_graph.py` | Created (placeholder) |

## Dependencies Added

Declared in `pyproject.toml`:

- `python-dotenv`
- `pymupdf`
- `langchain`
- `langchain-openai`
- `langchain-chroma`
- `langchain-text-splitters`
- `chromadb`
- `langgraph`
- `PyQt6`

Optional dev dependency: `pytest` (for running tests once installed).

Install with:

```bash
pip install -e .
# or
pip install -e ".[dev]"
```

## Environment Variables (`.env.example`)

| Variable | Default in `.env.example` |
|----------|---------------------------|
| `OPENAI_API_KEY` | (empty — user must set) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` |
| `CHAT_MODEL` | `gpt-4o-mini` |
| `CHROMA_PERSIST_DIR` | `data/chroma` |
| `CHUNK_SIZE` | `1000` |
| `CHUNK_OVERLAP` | `150` |
| `TOP_K` | `4` |

## Smoke Test Commands

### Roadmap import check

```bash
python -c "from config.settings import settings; print(settings.embedding_model)"
```

### Full smoke test (no pytest required)

```bash
python -c "
from tests.test_smoke import test_settings_import, test_settings_defaults, test_project_imports
test_settings_import()
test_settings_defaults()
test_project_imports()
print('All smoke tests passed')
"
```

### With pytest (after `pip install -e ".[dev]"`)

```bash
pytest tests/test_smoke.py -v
```

### Main stub

```bash
python main.py
```

## Smoke Test Results

| Command | Result |
|---------|--------|
| `python -c "from config.settings import settings; ..."` | **PASS** — defaults loaded: `text-embedding-3-small`, `gpt-4o-mini`, `chunk_size=1000` |
| Full smoke test (import all packages) | **PASS** — `All smoke tests passed` |
| `python main.py` | **PASS** — prints bootstrap message |
| `pytest tests/test_smoke.py -v` | **SKIPPED** — `pytest` not installed in the current environment; smoke logic verified via direct Python invocation |

## Assumptions and Deviations

1. **No `README.md` yet** — Phase 0 roadmap lists it in the target structure, but Phase 0 deliverables focus on bootstrap files; README is planned for Phase 7.
2. **Placeholder test files** — Empty test modules from the roadmap were created as stubs; only `tests/test_smoke.py` contains runnable assertions.
3. **`data/chroma/` and `data/uploads/` contents are gitignored** — `.gitkeep` files preserve directory structure in version control.
4. **Default values** — `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=150`, `TOP_K=4` align with roadmap MVP suggestions; tunable via `.env`.
5. **Dependencies not installed in this session** — `pyproject.toml` declares them; `python-dotenv` was already available in the environment for the smoke import test. Full dependency install is left to the developer (`pip install -e .`).

## What Remains for Phase 1

Phase 1 — **Persian PDF extraction**:

- Implement `ingestion/pdf_loader.py` with PyMuPDF (`fitz`)
- Add `core/models.py` types: `ExtractedPage`, `ExtractedDocument`
- Implement `load_pdf(path) -> ExtractedDocument`
- Preserve `page_number`, `source_path`; normalize whitespace; keep Persian characters
- Add `tests/test_pdf_loader.py` with a small fixture PDF
- Validate on 1–2 real tender PDFs (readable Persian, correct page count)
