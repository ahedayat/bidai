"""PyQt main window (Phase 6)."""

from __future__ import annotations

from PyQt6.QtCore import QThread, Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from services.rag_service import RAGService
from ui.widgets.chat_panel import ChatPanel
from ui.widgets.file_picker import FilePicker
from ui.widgets.sources_panel import SourcesPanel
from ui.workers import IngestWorker, QueryWorker


class MainWindow(QMainWindow):
    """Main desktop window for the Persian tender-document QA MVP."""

    def __init__(self, rag_service: RAGService | None = None) -> None:
        super().__init__()
        self._rag_service = rag_service or RAGService()
        self._document_id: str | None = None
        self._ingest_thread: QThread | None = None
        self._query_thread: QThread | None = None

        self.setWindowTitle("Bidai — Persian Tender Document QA")
        self.resize(1000, 700)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        root_layout.addLayout(self._build_top_bar())
        root_layout.addWidget(self._build_main_area(), stretch=1)
        root_layout.addLayout(self._build_bottom_bar())

        self._set_status("No PDF selected")
        self._update_controls()

    def _build_top_bar(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        self._file_picker = FilePicker()
        self._file_picker.file_selected.connect(self._on_file_selected)

        self._index_button = QPushButton("Process/Index")
        self._index_button.clicked.connect(self._on_index_clicked)

        self._status_label = QLabel()
        self._status_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        layout.addWidget(self._file_picker)
        layout.addWidget(self._index_button)
        layout.addWidget(self._status_label, stretch=1)
        return layout

    def _build_main_area(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._chat_panel = ChatPanel()
        self._sources_panel = SourcesPanel()

        splitter.addWidget(self._chat_panel)
        splitter.addWidget(self._sources_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        return splitter

    def _build_bottom_bar(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        self._question_input = QLineEdit()
        self._question_input.setPlaceholderText("سؤال خود را به فارسی بنویسید...")
        self._question_input.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self._question_input.returnPressed.connect(self._on_send_clicked)

        self._send_button = QPushButton("Send")
        self._send_button.clicked.connect(self._on_send_clicked)

        layout.addWidget(self._question_input, stretch=1)
        layout.addWidget(self._send_button)
        return layout

    def _set_status(self, message: str) -> None:
        self._status_label.setText(f"Status: {message}")

    def _update_controls(self) -> None:
        has_pdf = self._file_picker.selected_path is not None
        is_indexed = self._document_id is not None
        is_indexing = self._ingest_thread is not None and self._ingest_thread.isRunning()
        is_querying = self._query_thread is not None and self._query_thread.isRunning()

        self._index_button.setEnabled(has_pdf and not is_indexing and not is_querying)
        self._file_picker.setEnabled(not is_indexing and not is_querying)
        self._question_input.setEnabled(is_indexed and not is_indexing and not is_querying)
        self._send_button.setEnabled(is_indexed and not is_indexing and not is_querying)

    def _on_file_selected(self, path: str) -> None:
        self._document_id = None
        self._chat_panel.clear()
        self._sources_panel.clear()
        self._set_status("PDF selected")
        self._update_controls()

    def _on_index_clicked(self) -> None:
        pdf_path = self._file_picker.selected_path
        if not pdf_path:
            QMessageBox.warning(self, "Missing PDF", "Please select a PDF file first.")
            return

        self._document_id = None
        self._chat_panel.clear()
        self._sources_panel.clear()
        self._set_status("Indexing...")
        self._update_controls()

        worker = IngestWorker(pdf_path, self._rag_service)
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._set_status)
        worker.finished.connect(self._on_ingest_finished)
        worker.error.connect(self._on_ingest_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_worker_thread_finished)

        self._ingest_thread = thread
        thread.start()

    def _on_ingest_finished(self, document_id: str) -> None:
        self._document_id = document_id
        self._set_status("Indexing completed")
        self._chat_panel.add_system_message(
            "سند با موفقیت پردازش شد. اکنون می‌توانید سؤال بپرسید."
        )
        self._update_controls()

    def _on_ingest_error(self, message: str) -> None:
        self._document_id = None
        self._set_status("Indexing failed")
        QMessageBox.critical(self, "Indexing Error", message)
        self._update_controls()

    def _on_send_clicked(self) -> None:
        question = self._question_input.text().strip()
        if not question:
            QMessageBox.warning(self, "Empty Question", "Please enter a question.")
            return

        if not self._document_id:
            QMessageBox.warning(
                self,
                "Document Not Indexed",
                "Please process and index a PDF before asking questions.",
            )
            return

        self._chat_panel.add_user_message(question)
        self._question_input.clear()
        self._sources_panel.clear()
        self._set_status("Answering...")
        self._update_controls()

        worker = QueryWorker(question, self._document_id, self._rag_service)
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_query_finished)
        worker.error.connect(self._on_query_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_worker_thread_finished)

        self._query_thread = thread
        thread.start()

    def _on_query_finished(self, answer: str, sources: list) -> None:
        self._chat_panel.add_assistant_message(answer)
        self._sources_panel.set_sources(sources)
        self._set_status("Ready")
        self._update_controls()

    def _on_query_error(self, message: str) -> None:
        self._set_status("Query failed")
        QMessageBox.critical(self, "Query Error", message)
        self._update_controls()

    def _on_worker_thread_finished(self) -> None:
        sender = self.sender()
        if sender is self._ingest_thread:
            self._ingest_thread = None
        elif sender is self._query_thread:
            self._query_thread = None
        self._update_controls()
