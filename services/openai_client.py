"""OpenAI client wrappers for embeddings (Phase 3)."""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from config.settings import settings
from core.exceptions import MissingOpenAIAPIKeyError


def create_embeddings(
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> OpenAIEmbeddings:
    """Create an OpenAI embeddings client from settings or explicit overrides.

    The API key is read from the ``api_key`` argument, then ``settings.openai_api_key``,
    which is populated from environment variables or ``.env``.

    Raises:
        MissingOpenAIAPIKeyError: When no API key is available.
    """
    resolved_key = api_key if api_key is not None else settings.openai_api_key
    if not resolved_key or not resolved_key.strip():
        raise MissingOpenAIAPIKeyError(
            "OPENAI_API_KEY is not set. Provide it via environment variable or .env file."
        )

    return OpenAIEmbeddings(
        model=model or settings.embedding_model,
        api_key=resolved_key,
    )
