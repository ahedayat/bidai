"""LangGraph state schema (Phase 5)."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


def _replace_list(left: list, right: list) -> list:
    """Reducer that replaces a list field with the latest node output."""
    return right if right is not None else left


class QAState(TypedDict, total=False):
    """State passed through the linear RAG LangGraph workflow."""

    question: str
    document_id: str
    retrieved_docs: Annotated[list[Document], _replace_list]
    answer: str
    sources: Annotated[list[dict], _replace_list]
    messages: Annotated[list[BaseMessage], _replace_list]
    top_k: int | None
    error: str | None
