"""Phase 0 smoke tests."""

from config.settings import load_settings, settings


def test_settings_import() -> None:
    assert settings is not None


def test_settings_defaults() -> None:
    s = load_settings()
    assert s.embedding_model == "text-embedding-3-small"
    assert s.chat_model == "gpt-4o-mini"
    assert s.chunk_size == 1000
    assert s.chunk_overlap == 150
    assert s.top_k == 4
    assert s.chroma_persist_dir.name == "chroma"


def test_project_imports() -> None:
    import agent  # noqa: F401
    import core  # noqa: F401
    import ingestion  # noqa: F401
    import retrieval  # noqa: F401
    import services  # noqa: F401
    import ui  # noqa: F401
