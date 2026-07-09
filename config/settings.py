"""Application settings loaded from environment variables and .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    embedding_model: str
    chat_model: str
    chroma_persist_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    project_root: Path


def load_settings() -> Settings:
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
    chroma_path = Path(chroma_dir)
    if not chroma_path.is_absolute():
        chroma_path = _PROJECT_ROOT / chroma_path

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        chroma_persist_dir=chroma_path,
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
        top_k=_env_int("TOP_K", 4),
        project_root=_PROJECT_ROOT,
    )


settings = load_settings()
