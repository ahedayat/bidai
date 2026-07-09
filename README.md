# Bidai — Persian Tender Document QA

A desktop MVP for asking questions in Persian about tender/procurement PDF documents. Select a PDF, index it into a local Chroma vector store, and query it with RAG powered by OpenAI embeddings, OpenAI chat models, and a LangGraph workflow — all through a PyQt6 GUI.

## MVP Features

- Select a Persian PDF tender document
- Extract text (PyMuPDF)
- Chunk text with Persian-aware separators
- Index chunks into local Chroma
- Ask questions using RAG
- OpenAI embeddings and chat models
- LangGraph for the RAG workflow (`retrieve → generate → format_sources`)
- PyQt6 desktop GUI with background workers

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  PyQt6 GUI (main.py)                                        │
│  FilePicker │ ChatPanel │ SourcesPanel │ QThread workers    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  RAGService (services/rag_service.py)                       │
│  index_pdf() │ ask() │ clear_document_index()               │
└──────┬───────────────────────────────┬──────────────────────┘
       │ ingest                        │ query
┌──────▼──────────┐            ┌───────▼──────────────────────┐
│ ingestion/      │            │ agent/ (LangGraph)             │
│ pdf_loader      │            │ retrieve → generate → sources  │
│ chunker         │            └───────┬──────────────────────┘
│ indexer         │                    │
└──────┬──────────┘            ┌───────▼──────────────────────┐
       │                       │ retrieval/retriever.py       │
       │                       └───────┬──────────────────────┘
       │                               │
       └───────────────┬───────────────┘
                       ▼
              ┌─────────────────┐
              │ Chroma (local)  │
              │ data/chroma/    │
              └─────────────────┘
```

**Data flow — indexing:** PDF → extract pages → chunk → embed (OpenAI) → persist to Chroma.

**Data flow — Q&A:** Question → retrieve top-k chunks → Persian prompt + Chat API → answer + source previews.

## Project Structure

```
bidai/
├── main.py                 # GUI entry point
├── config/
│   ├── settings.py         # Environment-based configuration
│   └── logging_config.py   # Console/file logging setup
├── core/                   # Models and exceptions
├── ingestion/              # PDF load, chunk, index
├── retrieval/              # Chroma vector store and retriever
├── agent/                  # LangGraph state, nodes, graph
├── services/               # OpenAI client, RAG service, helpers
├── ui/                     # PyQt main window, widgets, workers
├── scripts/
│   └── demo_check.py       # Environment smoke check
├── data/
│   ├── chroma/             # Local vector DB (gitignored)
│   └── uploads/            # Optional uploads (gitignored)
├── tests/                  # Unit and integration tests
└── docs/phases/            # Phase implementation reports
```

## Requirements

- **Python 3.10+** (see `pyproject.toml`)
- **OpenAI API key** for real indexing and querying
- **Display server** for the PyQt GUI (macOS, Windows, or Linux with X11/Wayland)

## Setup

### 1. Clone and enter the project

```bash
cd bidai
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -e .
# or with dev tools (pytest):
pip install -e ".[dev]"
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-key-here
```

See `.env.example` for all available options (models, chunk size, Chroma path, logging).

### 5. Verify setup (optional)

```bash
python scripts/demo_check.py
```

## Running the Application

```bash
python main.py
```

### Manual demo flow

1. Click **Select PDF** and choose a Persian tender PDF.
2. Click **Process/Index** and wait for indexing to complete.
3. Type a question in Persian, e.g. `مهلت ارسال پیشنهاد چیست؟`
4. Click **Send** and review the answer in the chat panel.
5. Inspect retrieved sources in the side panel (page numbers and text previews).
6. Click **New Document** to reset the session and optionally clear the indexed collection.

## Running Tests

```bash
pytest -v
```

Smoke check without pytest:

```bash
python scripts/demo_check.py
python -c "from config.settings import settings; print(settings)"
python -c "from config.logging_config import setup_logging; setup_logging(); print('logging ok')"
```

## Known Limitations

- **Scanned/image-only PDFs** are not supported. The app rejects PDFs with very little extractable text (< 50 characters).
- **OCR** is out of scope for this MVP.
- **Complex tables and multi-column layouts** may extract imperfectly; page numbers help locate content.
- **Local Chroma data** (`data/chroma/`) should not be committed — it is gitignored.
- **OpenAI API key** is required for real indexing and querying.
- **Single document per session** — no multi-document cross-search.
- **No streaming** — answers appear after generation completes.
- **No multi-turn memory** — each question is independent; chat history is display-only.

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| **Missing OpenAI API key** | Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, restart the app. |
| **Invalid PDF** | Ensure the file is a valid, unencrypted `.pdf`. Try opening it in a PDF viewer. |
| **Empty extracted text** | The PDF may be scanned/image-only. OCR is not supported in v1. |
| **Chroma persistence issues** | Delete `data/chroma/` contents and re-index. Check write permissions on `data/`. |
| **PyQt display issues** | Ensure a display server is available. On headless Linux, use X11 forwarding or run locally. |
| **Import errors** | Activate the virtualenv and run `pip install -e .`. |

## Next Steps After MVP

- OCR pipeline for scanned tender PDFs
- Multi-document search across collections
- Streaming answers in the chat UI
- Hybrid retrieval (BM25 + vector) and reranking
- Multi-turn conversation memory
- PyInstaller packaging for distribution
- CI/CD with GitHub Actions

## License

Internal MVP — see project maintainers for usage terms.
