"""Shared test helpers for vector store and indexer tests."""

from __future__ import annotations

import hashlib
import struct

from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic, fixed-size embeddings for tests (no OpenAI calls)."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def _vectorize(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        for index in range(self.dimension):
            start = (index * 4) % len(digest)
            chunk = digest[start : start + 4]
            if len(chunk) < 4:
                chunk = (chunk + digest)[:4]
            values.append(struct.unpack("!i", chunk)[0] / 2_147_483_647)
        return values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)
