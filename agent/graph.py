"""LangGraph workflow definition (Phase 5)."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes import format_sources_node, generate_node, retrieve_node
from agent.state import QAState


def build_rag_graph(
    *,
    chat_model: BaseChatModel | None = None,
    embedding_function: Embeddings | None = None,
    persist_directory: Path | None = None,
) -> StateGraph:
    """Build the linear retrieve → generate → format_sources RAG graph."""
    graph = StateGraph(QAState)

    graph.add_node(
        "retrieve",
        partial(
            retrieve_node,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
        ),
    )
    graph.add_node(
        "generate",
        partial(generate_node, chat_model=chat_model),
    )
    graph.add_node("format_sources", format_sources_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "format_sources")
    graph.add_edge("format_sources", END)

    return graph


def get_compiled_rag_graph(
    *,
    chat_model: BaseChatModel | None = None,
    embedding_function: Embeddings | None = None,
    persist_directory: Path | None = None,
) -> CompiledStateGraph:
    """Return a compiled RAG graph ready for ``invoke()``."""
    return build_rag_graph(
        chat_model=chat_model,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    ).compile()
