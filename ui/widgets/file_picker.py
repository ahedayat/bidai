"""File picker widget (Phase 6)."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPushButton, QWidget


class FilePicker(QWidget):
    """Widget for selecting a PDF file with basic validation."""

    file_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected_path: str | None = None

        self._browse_button = QPushButton("Select PDF")
        self._browse_button.clicked.connect(self._on_browse_clicked)

        self._path_label = QLabel("No PDF selected")
        self._path_label.setWordWrap(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._browse_button)
        layout.addWidget(self._path_label, stretch=1)

    @property
    def selected_path(self) -> str | None:
        return self._selected_path

    def set_path(self, path: str | None) -> None:
        """Set the displayed path programmatically."""
        if path is None:
            self._selected_path = None
            self._path_label.setText("No PDF selected")
            return

        resolved = str(Path(path).expanduser().resolve())
        if not self._is_valid_pdf_path(resolved):
            raise ValueError(f"Not a valid PDF file: {path}")

        self._selected_path = resolved
        self._path_label.setText(Path(resolved).name)
        self.file_selected.emit(resolved)

    def clear(self) -> None:
        self._selected_path = None
        self._path_label.setText("No PDF selected")

    def _on_browse_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PDF",
            "",
            "PDF Files (*.pdf)",
        )
        if not file_path:
            return

        if not self._is_valid_pdf_path(file_path):
            return

        self.set_path(file_path)

    @staticmethod
    def _is_valid_pdf_path(path: str) -> bool:
        return Path(path).suffix.lower() == ".pdf"
