"""Phase 6 worker and GUI import tests."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.exceptions import MissingOpenAIAPIKeyError, PDFNotFoundError
from core.models import ExtractedDocument, ExtractedPage, IndexResult, RAGAnswer, RAGSource
from services.rag_service import RAGService
from ui.workers import IngestWorker, QueryWorker, format_user_error


def test_format_user_error_missing_api_key() -> None:
    message = format_user_error(MissingOpenAIAPIKeyError("missing"))
    assert "OpenAI API key" in message


def test_format_user_error_pdf_error() -> None:
    message = format_user_error(PDFNotFoundError("not found"))
    assert "PDF error" in message


def test_ingest_worker_can_be_instantiated() -> None:
    worker = IngestWorker("/tmp/sample.pdf", RAGService())
    assert worker is not None


def test_query_worker_can_be_instantiated() -> None:
    worker = QueryWorker("سؤال", "doc-id", RAGService())
    assert worker is not None


def test_gui_widgets_can_be_imported() -> None:
    from ui.main_window import MainWindow
    from ui.widgets.chat_panel import ChatPanel
    from ui.widgets.file_picker import FilePicker
    from ui.widgets.sources_panel import SourcesPanel

    assert MainWindow is not None
    assert ChatPanel is not None
    assert FilePicker is not None
    assert SourcesPanel is not None


def test_rag_service_index_pdf_with_monkeypatched_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "tender.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    source_path = pdf_path.resolve()
    extracted = ExtractedDocument(
        source_path=source_path,
        file_name=pdf_path.name,
        page_count=1,
        pages=(
            ExtractedPage(
                page_number=1,
                text="مهلت ارسال پیشنهاد تا پایان ماه است. لطفاً مدارک را به موقع ارسال کنید.",
                source_path=source_path,
                char_count=70,
                is_empty=False,
                is_short=False,
            ),
        ),
        total_char_count=70,
    )

    fake_doc = MagicMock()
    fake_doc.metadata = {
        "document_id": "abc123",
        "source": str(source_path),
        "page": 1,
        "chunk_index": 0,
    }
    fake_doc.page_content = extracted.pages[0].text

    index_result = IndexResult(
        document_id="abc123",
        collection_name="doc_abc123",
        chunk_count=1,
        persist_directory=tmp_path / "chroma",
        replaced_existing=False,
        indexed_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )

    progress_messages: list[str] = []

    monkeypatch.setattr("services.rag_service.load_pdf", lambda _path: extracted)
    monkeypatch.setattr(
        "services.rag_service.chunk_document",
        lambda _extracted, document_id=None: [fake_doc],
    )
    monkeypatch.setattr(
        "services.rag_service.index_documents",
        lambda docs, document_id=None, **kwargs: index_result,
    )
    monkeypatch.setattr(
        "services.rag_service.build_document_id",
        lambda _path: "abc123",
    )

    service = RAGService()
    result = service.index_pdf(
        pdf_path,
        on_progress=progress_messages.append,
    )

    assert result.document_id == "abc123"
    assert result.chunk_count == 1
    assert any("Loading PDF" in message for message in progress_messages)
    assert any("Indexing completed" in message for message in progress_messages)


def test_rag_service_index_pdf_rejects_low_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    source_path = pdf_path.resolve()

    extracted = ExtractedDocument(
        source_path=source_path,
        file_name=pdf_path.name,
        page_count=1,
        pages=(
            ExtractedPage(
                page_number=1,
                text="",
                source_path=source_path,
                char_count=0,
                is_empty=True,
                is_short=False,
            ),
        ),
        total_char_count=0,
    )

    monkeypatch.setattr("services.rag_service.load_pdf", lambda _path: extracted)
    monkeypatch.setattr(
        "services.rag_service.build_document_id",
        lambda _path: "abc123",
    )

    service = RAGService()
    with pytest.raises(Exception, match="Could not extract enough text"):
        service.index_pdf(pdf_path)


def test_query_worker_emits_finished_with_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = RAGAnswer(
        question="مهلت چیست؟",
        answer="مهلت تا پایان ماه است.",
        document_id="abc123",
        sources=(
            RAGSource(
                page=1,
                chunk_index=0,
                page_chunk_index=0,
                source="/tmp/tender.pdf",
                file_name="tender.pdf",
                text_preview="مهلت ارسال پیشنهاد",
            ),
        ),
        retrieved_count=1,
        model_name="fake-model",
    )

    service = RAGService()
    monkeypatch.setattr(service, "ask", lambda question, document_id: answer)

    worker = QueryWorker("مهلت چیست؟", "abc123", service)
    results: list[tuple[str, list]] = []
    worker.finished.connect(lambda ans, src: results.append((ans, src)))
    worker.run()

    assert len(results) == 1
    assert results[0][0] == answer.answer
    assert results[0][1] == [asdict(source) for source in answer.sources]


def test_ingest_worker_emits_error_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    service = RAGService()

    def _raise(_path: str, on_progress=None):
        raise PDFNotFoundError("missing")

    monkeypatch.setattr(service, "index_pdf", _raise)

    worker = IngestWorker("/tmp/missing.pdf", service)
    errors: list[str] = []
    worker.error.connect(errors.append)
    worker.run()

    assert errors
    assert "PDF error" in errors[0]
