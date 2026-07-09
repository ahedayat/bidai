"""RAG orchestration service (Phase 4–5)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from agent.graph import get_compiled_rag_graph
from config.settings import settings
from core.exceptions import (
    BidaiError,
    EmptyQuestionError,
    GraphInvocationError,
    RAGError,
    RetrievalError,
)
from core.models import RAGAnswer, RAGSource
from services.openai_client import create_chat_model
from services.rag_helpers import (
    format_context_block,
    format_source,
    format_sources,
    generate_answer,
)

__all__ = [
    "RAGService",
    "format_context_block",
    "format_source",
    "format_sources",
    "generate_answer",
]


class RAGService:
    """Synchronous RAG service backed by a compiled LangGraph workflow."""

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
        self._compiled_graph = None

    def _get_chat_model(self) -> BaseChatModel:
        if self._chat_model is not None:
            return self._chat_model
        return create_chat_model()

    def _get_compiled_graph(self):
        if self._compiled_graph is None:
            self._compiled_graph = get_compiled_rag_graph(
                chat_model=self._chat_model,
                embedding_function=self._embedding_function,
                persist_directory=self._persist_directory,
            )
        return self._compiled_graph

    def ask(
        self,
        question: str,
        document_id: str,
        *,
        top_k: int | None = None,
    ) -> RAGAnswer:
        """Retrieve relevant chunks and generate a Persian answer via LangGraph.

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
            GraphInvocationError: When graph invocation fails unexpectedly.
        """
        if not question or not question.strip():
            raise EmptyQuestionError("question must be a non-empty string")

        stripped_question = question.strip()
        stripped_document_id = document_id.strip()
        initial_state: dict = {
            "question": stripped_question,
            "document_id": stripped_document_id,
        }
        if top_k is not None:
            initial_state["top_k"] = top_k

        try:
            result = self._get_compiled_graph().invoke(initial_state)
        except (RAGError, RetrievalError):
            raise
        except BidaiError:
            raise
        except Exception as exc:
            raise GraphInvocationError(
                f"RAG graph invocation failed for document_id={stripped_document_id!r}: {exc}"
            ) from exc

        retrieved_docs = result.get("retrieved_docs", [])
        answer_text = result.get("answer", "")
        source_dicts = result.get("sources", [])
        sources = tuple(RAGSource(**source_dict) for source_dict in source_dicts)

        chat_model = self._get_chat_model()
        model_name = getattr(chat_model, "model_name", None) or settings.chat_model

        return RAGAnswer(
            question=stripped_question,
            answer=answer_text,
            document_id=stripped_document_id,
            sources=sources,
            retrieved_count=len(retrieved_docs),
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
