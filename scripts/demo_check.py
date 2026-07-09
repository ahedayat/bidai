#!/usr/bin/env python3
"""Lightweight developer/demo environment check (Phase 7)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "OK" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


def main() -> int:
    print("Bidai demo / environment check\n")

    all_ok = True

    # Settings import
    try:
        from config.settings import settings

        all_ok &= _check("Settings import", True, f"chat_model={settings.chat_model}")
    except Exception as exc:
        all_ok &= _check("Settings import", False, str(exc))

    # Logging config
    try:
        from config.logging_config import setup_logging

        setup_logging()
        all_ok &= _check("Logging config", True)
    except Exception as exc:
        all_ok &= _check("Logging config", False, str(exc))

    # Core module imports
    modules = [
        "ingestion.pdf_loader",
        "ingestion.chunker",
        "ingestion.indexer",
        "retrieval.vector_store",
        "retrieval.retriever",
        "services.rag_service",
        "agent.graph",
        "ui.main_window",
    ]
    for module_name in modules:
        try:
            __import__(module_name)
            all_ok &= _check(f"Import {module_name}", True)
        except Exception as exc:
            all_ok &= _check(f"Import {module_name}", False, str(exc))

    # Chroma directory
    try:
        from config.settings import settings
        from retrieval.vector_store import ensure_persist_directory

        chroma_dir = ensure_persist_directory(settings.chroma_persist_dir)
        all_ok &= _check("Chroma persist directory", chroma_dir.is_dir(), str(chroma_dir))
    except Exception as exc:
        all_ok &= _check("Chroma persist directory", False, str(exc))

    # .env presence (warning only)
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        _check(".env file", True, "found")
    else:
        print("  [WARN] .env file not found — copy .env.example to .env and set OPENAI_API_KEY")

    # .env.example keys
    env_example = PROJECT_ROOT / ".env.example"
    required_keys = {
        "OPENAI_API_KEY",
        "EMBEDDING_MODEL",
        "CHAT_MODEL",
        "CHROMA_PERSIST_DIR",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
        "LOG_LEVEL",
    }
    if env_example.exists():
        content = env_example.read_text(encoding="utf-8")
        missing = [key for key in required_keys if key not in content]
        all_ok &= _check(
            ".env.example required keys",
            not missing,
            f"missing: {missing}" if missing else "all present",
        )
    else:
        all_ok &= _check(".env.example", False, "file not found")

    # API key warning (not a failure)
    try:
        from config.settings import settings

        if not settings.openai_api_key.strip():
            print("  [WARN] OPENAI_API_KEY is not set — indexing and querying will fail")
        else:
            _check("OPENAI_API_KEY", True, "set (value not shown)")
    except Exception:
        pass

    print()
    if all_ok:
        print("All checks passed. Run: python main.py")
        return 0

    print("Some checks failed. See messages above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
