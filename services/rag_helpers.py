"""Shared RAG formatting and generation helpers (Phase 4–5)."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from agent.prompts import RAG_PROMPT
from config.settings import settings
from core.exceptions import ChatAPIError, RAGError
from core.models import RAGSource
from services.openai_client import create_chat_model

_DEFAULT_PREVIEW_LEN = 200
_CONTEXT_PREVIEW_LEN = 4000


def _truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def format_source(doc: Document, *, preview_len: int = _DEFAULT_PREVIEW_LEN) -> RAGSource:
    """Format a retrieved LangChain document as a UI-friendly source entry."""
    metadata = doc.metadata
    return RAGSource(
        page=int(metadata.get("page", 0)),
        chunk_index=int(metadata.get("chunk_index", -1)),
        page_chunk_index=(
            int(metadata["page_chunk_index"])
            if metadata.get("page_chunk_index") is not None
            else None
        ),
        source=str(metadata.get("source", "")),
        file_name=(
            str(metadata["file_name"]) if metadata.get("file_name") is not None else None
        ),
        text_preview=_truncate_text(doc.page_content, preview_len),
    )


def format_sources(
    docs: list[Document],
    *,
    preview_len: int = _DEFAULT_PREVIEW_LEN,
) -> tuple[RAGSource, ...]:
    """Format retrieved documents into source entries."""
    return tuple(format_source(doc, preview_len=preview_len) for doc in docs)


def format_context_block(
    docs: list[Document],
    *,
    max_chunk_len: int = _CONTEXT_PREVIEW_LEN,
) -> str:
    """Combine retrieved chunks into a context block with page/chunk metadata."""
    blocks: list[str] = []
    for doc in docs:
        metadata = doc.metadata
        page = metadata.get("page", "?")
        chunk_index = metadata.get("chunk_index", "?")
        page_chunk_index = metadata.get("page_chunk_index")
        file_name = metadata.get("file_name")

        header_parts = [f"[صفحه {page}", f"بخش {chunk_index}"]
        if page_chunk_index is not None:
            header_parts.append(f"زیربخش صفحه {page_chunk_index}")
        if file_name:
            header_parts.append(f"فایل: {file_name}")
        header = " | ".join(header_parts) + "]"

        content = _truncate_text(doc.page_content, max_chunk_len)
        blocks.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(blocks)


def generate_answer(
    question: str,
    retrieved_docs: list[Document],
    *,
    chat_model: BaseChatModel | None = None,
) -> tuple[str, str]:
    """Generate a Persian answer from retrieved documents.

    Returns:
        Tuple of ``(answer_text, model_name)``.
    """
    if chat_model is None:
        chat_model = create_chat_model()

    context = format_context_block(retrieved_docs)
    model_name = getattr(chat_model, "model_name", None) or settings.chat_model

    try:
        messages = RAG_PROMPT.format_messages(
            context=context,
            question=question,
        )
        response = chat_model.invoke(messages)
    except RAGError:
        raise
    except Exception as exc:
        raise ChatAPIError(f"Chat API call failed: {exc}") from exc

    if isinstance(response, AIMessage):
        answer_text = str(response.content)
    else:
        answer_text = str(response)

    return answer_text, model_name
