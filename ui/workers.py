"""QThread workers for ingest/query (Phase 6)."""

from __future__ import annotations

from dataclasses import asdict

from PyQt6.QtCore import QObject, pyqtSignal

from core.exceptions import (
    BidaiError,
    ChatAPIError,
    GraphInvocationError,
    IndexingError,
    MissingOpenAIAPIKeyError,
    PDFError,
    RAGError,
    RetrievalError,
)
from services.rag_service import RAGService


def format_user_error(exc: Exception) -> str:
    """Convert application exceptions into user-friendly messages."""
    if isinstance(exc, MissingOpenAIAPIKeyError):
        return (
            "OpenAI API key is missing. "
            "Set OPENAI_API_KEY in your .env file and restart the application."
        )
    if isinstance(exc, PDFError):
        return f"PDF error: {exc}"
    if isinstance(exc, IndexingError):
        return f"Indexing failed: {exc}"
    if isinstance(exc, RetrievalError):
        return f"Retrieval failed: {exc}"
    if isinstance(exc, (RAGError, ChatAPIError, GraphInvocationError)):
        return f"Answer generation failed: {exc}"
    if isinstance(exc, BidaiError):
        return str(exc)
    return f"Unexpected error: {exc}"


class IngestWorker(QObject):
    """Background worker that indexes a PDF without blocking the UI."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, pdf_path: str, rag_service: RAGService) -> None:
        super().__init__()
        self._pdf_path = pdf_path
        self._rag_service = rag_service

    def run(self) -> None:
        try:
            result = self._rag_service.index_pdf(
                self._pdf_path,
                on_progress=self.progress.emit,
            )
        except Exception as exc:
            self.error.emit(format_user_error(exc))
            return

        self.finished.emit(result.document_id)


class QueryWorker(QObject):
    """Background worker that runs a RAG query without blocking the UI."""

    finished = pyqtSignal(str, list)
    error = pyqtSignal(str)

    def __init__(
        self,
        question: str,
        document_id: str,
        rag_service: RAGService,
    ) -> None:
        super().__init__()
        self._question = question
        self._document_id = document_id
        self._rag_service = rag_service

    def run(self) -> None:
        try:
            result = self._rag_service.ask(self._question, self._document_id)
        except Exception as exc:
            self.error.emit(format_user_error(exc))
            return

        sources = [asdict(source) for source in result.sources]
        self.finished.emit(result.answer, sources)
