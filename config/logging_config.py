"""Application logging configuration (Phase 7)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from config.settings import settings

_CONFIGURED = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _resolve_log_level(level_name: str) -> int:
    normalized = level_name.strip().upper()
    level = getattr(logging, normalized, None)
    if isinstance(level, int):
        return level
    return logging.INFO


def setup_logging(*, force: bool = False) -> None:
    """Configure root logging for console and optional file output.

  * Console logging is always enabled.
  * File logging is enabled when ``LOG_FILE`` is set in the environment.
  * Log level is read from ``LOG_LEVEL`` (default: ``INFO``).

  Safe to call multiple times; configuration runs once unless ``force=True``.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    level = _resolve_log_level(settings.log_level)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root_logger.addHandler(console_handler)

    if settings.log_file:
        log_path = settings.log_file
        if not log_path.is_absolute():
            log_path = settings.project_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root_logger.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured first."""
    setup_logging()
    return logging.getLogger(name)
