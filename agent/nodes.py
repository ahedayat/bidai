"""LangGraph nodes for the RAG workflow (Phase 5)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from agent.state import QAState
from core.exceptions import NoRetrievedDocumentsError
from retrieval.retriever import retrieve_documents
from services.rag_helpers import format_sources, generate_answer


def retrieve_node(
    state: QAState,
    *,
    embedding_function: Embeddings | None = None,
    persist_directory: Path | None = None,
) -> dict:
    """Retrieve top-k chunks for the question and store them in state."""
    question = state["question"]
    document_id = state["document_id"]
    top_k = state.get("top_k")

    retrieved = retrieve_documents(
        question,
        document_id,
        top_k=top_k,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )

    if not retrieved:
        raise NoRetrievedDocumentsError(
            f"No relevant chunks retrieved for document_id={document_id!r}"
        )

    return {"retrieved_docs": retrieved}


def generate_node(
    state: QAState,
    *,
    chat_model: BaseChatModel | None = None,
) -> dict:
    """Generate a Persian answer from retrieved context."""
    retrieved_docs = state.get("retrieved_docs", [])
    answer, _model_name = generate_answer(
        state["question"],
        retrieved_docs,
        chat_model=chat_model,
    )
    return {"answer": answer}


def format_sources_node(state: QAState) -> dict:
    """Format retrieved documents into UI-ready source dictionaries."""
    retrieved_docs = state.get("retrieved_docs", [])
    sources = format_sources(retrieved_docs)
    return {"sources": [asdict(source) for source in sources]}
