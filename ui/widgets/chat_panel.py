"""Chat panel widget (Phase 6)."""

from __future__ import annotations

import html

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget


class ChatPanel(QWidget):
    """Scrollable chat history with Persian/RTL-friendly display."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        title = QLabel("Chat")
        title.setStyleSheet("font-weight: bold;")

        self._history = QTextEdit()
        self._history.setReadOnly(True)
        self._history.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self._history.setPlaceholderText("پرسش و پاسخ اینجا نمایش داده می‌شود.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(title)
        layout.addWidget(self._history)

    def add_user_message(self, text: str) -> None:
        self._append_message("شما", text, role="user")

    def add_assistant_message(self, text: str) -> None:
        self._append_message("دستیار", text, role="assistant")

    def add_system_message(self, text: str) -> None:
        self._append_message("سیستم", text, role="system")

    def clear(self) -> None:
        self._history.clear()

    def _append_message(self, label: str, text: str, *, role: str) -> None:
        escaped_label = html.escape(label)
        escaped_text = html.escape(text).replace("\n", "<br>")

        if role == "user":
            color = "#1a5276"
        elif role == "assistant":
            color = "#145a32"
        else:
            color = "#7d6608"

        block = (
            f'<div dir="rtl" style="margin-bottom: 12px;">'
            f'<div style="color: {color}; font-weight: bold;">{escaped_label}</div>'
            f'<div style="margin-top: 4px;">{escaped_text}</div>'
            f"</div>"
        )

        self._history.moveCursor(QTextCursor.MoveOperation.End)
        self._history.insertHtml(block)
        self._history.moveCursor(QTextCursor.MoveOperation.End)
