# Phase 1 — Persian PDF Extraction Report

## Summary

Phase 1 implements Persian-aware PDF text extraction using PyMuPDF (`fitz`). The loader validates input paths, extracts text page by page with 1-based page numbering, normalizes whitespace without stripping Persian/Arabic content, and returns structured `ExtractedDocument` / `ExtractedPage` models with empty/short page flags. Custom exceptions cover common PDF failure modes. Unit tests generate a tiny in-memory PDF fixture (including a Persian sentence) and verify loader behavior end to end. A minimal CLI helper is available via `python -m ingestion.pdf_loader`.

No chunking, embeddings, Chroma, RAG, LangGraph, OpenAI, or PyQt work was added.

## Files Created or Updated

| Path | Action |
|------|--------|
| `core/models.py` | Updated — `ExtractedPage`, `ExtractedDocument` |
| `core/exceptions.py` | Updated — PDF-related custom exceptions |
| `ingestion/pdf_loader.py` | Updated — `load_pdf`, `normalize_text`, CLI helper |
| `tests/test_pdf_loader.py` | Updated — full unit test suite |
| `docs/phases/step-1.md` | Created — this report |

## Data Models Added

### `ExtractedPage`

| Field | Type | Description |
|-------|------|-------------|
| `page_number` | `int` | 1-based page index |
| `text` | `str` | Normalized page text |
| `source_path` | `Path` | Resolved absolute path to source PDF |
| `char_count` | `int` | Length of normalized text |
| `is_empty` | `bool` | `True` when text is empty after normalization |
| `is_short` | `bool` | `True` when non-empty text is below `min_page_chars` |

### `ExtractedDocument`

| Field / property | Type | Description |
|------------------|------|-------------|
| `source_path` | `Path` | Resolved absolute path |
| `file_name` | `str` | Base file name |
| `page_count` | `int` | Number of pages |
| `pages` | `tuple[ExtractedPage, ...]` | Per-page extraction results |
| `total_char_count` | `int` | Sum of page `char_count` values |
| `full_text` | property | Non-empty pages joined with `\n\n` |
| `empty_pages` | property | Pages where `is_empty` is `True` |
| `short_pages` | property | Pages where `is_short` is `True` |

Both models are frozen dataclasses with `slots=True`.

## PDF Loader Behavior

**Entry point:** `load_pdf(path: str | Path, *, min_page_chars: int = 30) -> ExtractedDocument`

1. Resolve and validate the path (must exist, be a file, have `.pdf` extension).
2. Open the PDF with `fitz.open()`.
3. Reject encrypted/password-protected PDFs (`document.needs_pass`).
4. Iterate pages in order; extract text with `page.get_text("text")`.
5. Normalize whitespace per page.
6. Build `ExtractedPage` objects with metadata and empty/short flags.
7. Return `ExtractedDocument` with aggregated counts.
8. Always close the PDF in a `finally` block.

**CLI helper:** `python -m ingestion.pdf_loader <path-to-pdf>` prints file name, page count, total character count, first/last page samples, and empty/short page lists.

## Text Normalization Approach

`normalize_text()` in `ingestion/pdf_loader.py`:

1. Convert `\r\n` and `\r` to `\n`.
2. Collapse horizontal whitespace (spaces, tabs) to a single space — **without** touching newlines.
3. Collapse runs of 3+ newlines to `\n\n`.
4. Strip leading/trailing whitespace from the page.

Persian, Arabic, English, digits, punctuation, and tender-related symbols are preserved. Only layout whitespace is normalized.

## Error Handling Added

| Exception | When raised |
|-----------|-------------|
| `PDFNotFoundError` | Path does not exist |
| `PDFInvalidFileError` | Not a file, wrong extension, or invalid PDF data |
| `PDFReadError` | PDF cannot be opened/read |
| `PDFEncryptedError` | PDF requires a password |
| `PDFExtractionError` | Per-page text extraction fails |

All inherit from `PDFError` → `BidaiError`.

## Test Cases Added

| Test | Coverage |
|------|----------|
| `test_normalize_text_collapses_whitespace_preserves_persian` | Whitespace normalization keeps Persian text |
| `test_load_pdf_nonexistent_path` | `PDFNotFoundError` for missing file |
| `test_load_pdf_rejects_non_pdf` | `PDFInvalidFileError` for `.txt` file |
| `test_load_pdf_valid_fixture` | Valid PDF returns `ExtractedDocument` with metadata |
| `test_load_pdf_page_count` | Page count matches fixture (3 pages) |
| `test_load_pdf_extracts_non_empty_text` | First page has non-empty extracted text |
| `test_load_pdf_preserves_page_metadata` | 1-based numbering, `source_path`, `char_count` |
| `test_load_pdf_empty_and_short_page_detection` | Empty page 2, short page 3 detected |
| `test_extracted_document_full_text_skips_empty_pages` | `full_text` omits empty pages |

**Fixture strategy:** PDFs are generated in tests with PyMuPDF — no binary fixture committed. Fixture pages:

1. Persian sentence: `این یک فایل آزمایشی برای مناقصه است.`
2. Empty page
3. Short text: `کوتاه` (below default `min_page_chars=30`)

Tests assert loader structure and metadata rather than exact Persian glyph rendering, since PyMuPDF font extraction can vary by environment.

## Commands Used for Smoke Tests and Unit Tests

### Settings import (Phase 0 smoke)

```bash
PYTHONPATH=. python -c "from config.settings import settings; print(settings.embedding_model)"
```

### Phase 0 smoke tests

```bash
PYTHONPATH=. pytest tests/test_smoke.py -v
```

### Phase 1 PDF loader tests

```bash
PYTHONPATH=. pytest tests/test_pdf_loader.py -v
```

### Combined run

```bash
PYTHONPATH=. pytest tests/test_smoke.py tests/test_pdf_loader.py -v
```

### CLI helper

```bash
PYTHONPATH=. python -m ingestion.pdf_loader path/to/file.pdf
```

**Note:** `pip install -e .` was not required for testing; `PYTHONPATH=.` was used because the project has no package discovery config in `pyproject.toml` yet. `pymupdf` and `pytest` were installed directly for this session.

## Test Results

| Command | Result |
|---------|--------|
| Settings import | **PASS** — `text-embedding-3-small` |
| `pytest tests/test_smoke.py tests/test_pdf_loader.py -v` | **PASS** — 12 passed, 0 failed |

## Assumptions and Deviations

1. **PDF validation by extension** — Non-`.pdf` files are rejected before opening. Corrupt files with a `.pdf` extension raise `PDFInvalidFileError` via `fitz.FileDataError`.
2. **Encrypted PDF detection** — Uses `document.needs_pass` rather than attempting decryption.
3. **Persian extraction fidelity** — Tests validate non-empty extraction and metadata; exact Persian glyph fidelity is not guaranteed with default PyMuPDF fonts in generated fixtures.
4. **`PYTHONPATH=.` for local runs** — Editable install was not configured in Phase 1; tests run from project root with `PYTHONPATH=.`.
5. **No binary PDF committed** — Fixture PDFs are generated in `tmp_path` during tests.

## Known Limitations

- **Scanned/image-only PDFs** are not supported; extraction will yield empty or near-empty pages flagged as `is_empty` / `is_short`.
- **OCR** is intentionally out of scope for Phase 1.
- **Complex layouts** (multi-column, tables, headers/footers) may produce shuffled or imperfect text order.
- **Encrypted PDFs** are rejected; no password support.
- **Font-dependent Persian rendering** in generated test PDFs may not round-trip exact Unicode; real tender PDFs with embedded fonts should behave better.

## What Remains for Phase 2

Phase 2 — **Chunking**:

- Implement `ingestion/chunker.py`
- Convert `ExtractedDocument` pages → LangChain `Document` objects
- Split with overlap using Persian-aware separators (`۔`, `؟`, `\n\n`, `.`)
- Attach metadata: `page`, `source`, `chunk_index`, `document_id`
- Add `tests/test_chunker.py`
- Suggested params: `chunk_size=800–1200`, `overlap=150–200`
