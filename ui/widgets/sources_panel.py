"""Retrieved sources panel widget (Phase 6)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QTextBrowser, QVBoxLayout, QWidget


class SourcesPanel(QWidget):
    """Display retrieved source chunks with page and preview metadata."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        title = QLabel("Retrieved Sources")
        title.setStyleSheet("font-weight: bold;")

        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(False)
        self._browser.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self._browser.setPlaceholderText("منابع بازیابی‌شده اینجا نمایش داده می‌شوند.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(title)
        layout.addWidget(self._browser)

    def set_sources(self, sources: list[dict]) -> None:
        if not sources:
            self._browser.setPlainText("هیچ منبعی بازیابی نشد.")
            return

        parts: list[str] = []
        for index, source in enumerate(sources, start=1):
            page = source.get("page", "?")
            chunk_index = source.get("chunk_index", "?")
            file_name = source.get("file_name") or source.get("source", "unknown")
            preview = source.get("text_preview", "")

            header = f"{index}. صفحه {page} | بخش {chunk_index} | {file_name}"
            parts.append(f"{header}\n{preview}")

        self._browser.setPlainText("\n\n---\n\n".join(parts))

    def clear(self) -> None:
        self._browser.clear()
