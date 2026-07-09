"""Entry point for the Persian tender-document QA MVP desktop GUI."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from config.logging_config import get_logger, setup_logging
from ui.main_window import MainWindow

logger = get_logger(__name__)


def main() -> int:
    """Launch the PyQt desktop application."""
    setup_logging()
    logger.info("Starting Bidai desktop application")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
