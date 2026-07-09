"""RAG orchestration service (Phase 4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from agent.prompts import RAG_PROMPT
from config.settings import settings
from core.exceptions import (
    ChatAPIError,
    EmptyQuestionError,
    NoRetrievedDocumentsError,
    RAGError,
)
from core.models import RAGAnswer, RAGSource
from retrieval.retriever import retrieve_documents
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


class RAGService:
    """Synchronous retrieve-and-generate RAG service (no LangGraph)."""

    def __init__(
        self,
        *,
        chat_model: BaseChatModel | None = None,
        embedding_function: Embeddings | None = None,
        persist_directory: Path | None = None,
    ) -> None:
        self._chat_model = chat_model
        self._embedding_function = embedding_function
        self._persist_directory = persist_directory

    def _get_chat_model(self) -> BaseChatModel:
        if self._chat_model is not None:
            return self._chat_model
        return create_chat_model()

    def ask(
        self,
        question: str,
        document_id: str,
        *,
        top_k: int | None = None,
    ) -> RAGAnswer:
        """Retrieve relevant chunks and generate a Persian answer.

        Args:
            question: User question.
            document_id: Indexed document identifier.
            top_k: Optional retrieval count override.

        Returns:
            ``RAGAnswer`` with generated text and formatted sources.

        Raises:
            EmptyQuestionError: When the question is empty.
            CollectionNotFoundError: When the document is not indexed.
            NoRetrievedDocumentsError: When retrieval returns no chunks.
            ChatAPIError: When the chat model invocation fails.
        """
        if not question or not question.strip():
            raise EmptyQuestionError("question must be a non-empty string")

        stripped_question = question.strip()
        retrieved = retrieve_documents(
            stripped_question,
            document_id,
            top_k=top_k,
            embedding_function=self._embedding_function,
            persist_directory=self._persist_directory,
        )

        if not retrieved:
            raise NoRetrievedDocumentsError(
                f"No relevant chunks retrieved for document_id={document_id!r}"
            )

        context = format_context_block(retrieved)
        sources = format_sources(retrieved)
        chat_model = self._get_chat_model()
        model_name = getattr(chat_model, "model_name", None) or settings.chat_model

        try:
            messages = RAG_PROMPT.format_messages(
                context=context,
                question=stripped_question,
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

        return RAGAnswer(
            question=stripped_question,
            answer=answer_text,
            document_id=document_id.strip(),
            sources=sources,
            retrieved_count=len(retrieved),
            model_name=model_name,
        )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ask a Persian question against an indexed tender document. "
            "Requires OPENAI_API_KEY and a previously indexed document_id."
        ),
    )
    parser.add_argument(
        "--document-id",
        required=True,
        help="Document ID used during indexing (e.g. from index_documents).",
    )
    parser.add_argument(
        "--question",
        required=True,
        help='Question in Persian, e.g. "مهلت ارسال پیشنهاد چیست؟"',
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Number of chunks to retrieve (default: {settings.top_k}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Optional CLI entry point for manual RAG debugging."""
    args = _build_cli_parser().parse_args(argv)
    service = RAGService()
    try:
        result = service.ask(args.question, args.document_id, top_k=args.top_k)
    except RAGError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(result.answer)
    print("\n--- Sources ---")
    for index, source in enumerate(result.sources, start=1):
        page_info = f"page {source.page}"
        if source.page_chunk_index is not None:
            page_info += f", chunk {source.chunk_index} (page chunk {source.page_chunk_index})"
        else:
            page_info += f", chunk {source.chunk_index}"
        print(f"{index}. [{page_info}] {source.text_preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
