"""Phase 7 polish and hardening tests."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from config.logging_config import setup_logging
from config.settings import load_settings
from services.rag_service import RAGService
from tests.fake_embeddings import FakeEmbeddings


def test_logging_config_setup() -> None:
    setup_logging(force=True)
    logger = logging.getLogger("bidai.test")
    assert logger.isEnabledFor(logging.INFO)


def test_settings_include_log_fields() -> None:
    settings = load_settings()
    assert isinstance(settings.log_level, str)
    assert settings.log_level
    assert settings.log_file is None or isinstance(settings.log_file, Path)


def test_env_example_contains_required_keys() -> None:
    project_root = Path(__file__).resolve().parent.parent
    content = (project_root / ".env.example").read_text(encoding="utf-8")
    required = {
        "OPENAI_API_KEY",
        "EMBEDDING_MODEL",
        "CHAT_MODEL",
        "CHROMA_PERSIST_DIR",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
        "LOG_LEVEL",
    }
    for key in required:
        assert key in content, f"Missing {key} in .env.example"


def test_readme_exists_and_has_key_sections() -> None:
    project_root = Path(__file__).resolve().parent.parent
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    assert "Bidai" in readme
    assert "python main.py" in readme
    assert "OPENAI_API_KEY" in readme
    assert "Troubleshooting" in readme


def test_rag_service_clear_document_index_deletes_collection(tmp_path: Path) -> None:
    from ingestion.indexer import index_documents
    from langchain_core.documents import Document
    from retrieval.vector_store import collection_exists, sanitize_collection_name

    document_id = "clear-test-doc"
    docs = [
        Document(
            page_content="متن آزمایشی برای پاک‌سازی شاخص.",
            metadata={
                "document_id": document_id,
                "source": "/tmp/clear-test.pdf",
                "page": 1,
                "chunk_index": 0,
            },
        ),
    ]
    index_documents(
        docs,
        document_id=document_id,
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
    )
    collection_name = sanitize_collection_name(document_id)
    assert collection_exists(tmp_path, collection_name)

    service = RAGService(embedding_function=FakeEmbeddings(), persist_directory=tmp_path)
    deleted = service.clear_document_index(document_id)
    assert deleted is True
    assert not collection_exists(tmp_path, collection_name)

    deleted_again = service.clear_document_index(document_id)
    assert deleted_again is False


def test_rag_service_clear_document_index_empty_id() -> None:
    service = RAGService()
    assert service.clear_document_index("") is False
    assert service.clear_document_index("   ") is False


def test_demo_check_script_importable() -> None:
    project_root = Path(__file__).resolve().parent.parent
    script = project_root / "scripts" / "demo_check.py"
    assert script.exists()


def test_main_window_has_new_document_button() -> None:
    import inspect

    from ui.main_window import MainWindow

    source = inspect.getsource(MainWindow)
    assert "_new_document_button" in source
    assert "_on_new_document_clicked" in source
