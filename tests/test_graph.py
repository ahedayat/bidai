"""LangGraph RAG workflow tests (Phase 5)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from agent.graph import build_rag_graph, get_compiled_rag_graph
from agent.nodes import format_sources_node, generate_node, retrieve_node
from agent.state import QAState
from core.exceptions import (
    ChatAPIError,
    CollectionNotFoundError,
    EmptyQuestionError,
    GraphInvocationError,
    NoRetrievedDocumentsError,
)
from core.models import RAGSource
from ingestion.indexer import index_documents
from services.rag_helpers import format_sources
from services.rag_service import RAGService
from tests.fake_chat import FakeChatModel
from tests.fake_embeddings import FakeEmbeddings

SOURCE_PATH = Path("/tmp/graph-test-tender.pdf").resolve()
DOCUMENT_ID = "test-graph-doc"


def _make_doc(
    text: str,
    *,
    chunk_index: int,
    page: int = 1,
    page_chunk_index: int = 0,
    document_id: str = DOCUMENT_ID,
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "document_id": document_id,
            "source": str(SOURCE_PATH),
            "file_name": "graph-test-tender.pdf",
            "page": page,
            "chunk_index": chunk_index,
            "page_chunk_index": page_chunk_index,
        },
    )


def _index_sample_docs(tmp_path: Path) -> None:
    docs = [
        _make_doc(
            "مهلت ارسال پیشنهاد تا پایان ماه است.",
            chunk_index=0,
            page=1,
        ),
        _make_doc(
            "شرایط مناقصه در این بخش توضیح داده شده است.",
            chunk_index=1,
            page=2,
        ),
    ]
    index_documents(
        docs,
        DOCUMENT_ID,
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
        replace_existing=True,
    )


def test_build_rag_graph_succeeds() -> None:
    graph = build_rag_graph(
        chat_model=FakeChatModel(),
        embedding_function=FakeEmbeddings(),
    )
    assert graph is not None


def test_get_compiled_rag_graph_succeeds() -> None:
    compiled = get_compiled_rag_graph(
        chat_model=FakeChatModel(),
        embedding_function=FakeEmbeddings(),
    )
    assert compiled is not None
    assert hasattr(compiled, "invoke")


def test_retrieve_node_stores_retrieved_docs(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    state: QAState = {
        "question": "مهلت ارسال پیشنهاد تا پایان ماه است.",
        "document_id": DOCUMENT_ID,
        "top_k": 1,
    }
    result = retrieve_node(
        state,
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
    )
    assert "retrieved_docs" in result
    assert len(result["retrieved_docs"]) == 1
    assert isinstance(result["retrieved_docs"][0], Document)
    assert "مهلت" in result["retrieved_docs"][0].page_content


def test_retrieve_node_raises_on_empty_question(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    state: QAState = {
        "question": "  ",
        "document_id": DOCUMENT_ID,
    }
    with pytest.raises(EmptyQuestionError):
        retrieve_node(
            state,
            embedding_function=FakeEmbeddings(),
            persist_directory=tmp_path,
        )


def test_retrieve_node_raises_when_no_docs_returned(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    state: QAState = {
        "question": "سوال تست",
        "document_id": DOCUMENT_ID,
        "top_k": 1,
    }
    with patch(
        "agent.nodes.retrieve_documents",
        return_value=[],
    ):
        with pytest.raises(NoRetrievedDocumentsError):
            retrieve_node(
                state,
                embedding_function=FakeEmbeddings(),
                persist_directory=tmp_path,
            )


def test_generate_node_stores_answer() -> None:
    docs = [_make_doc("مهلت ارسال پیشنهاد تا پایان ماه است.", chunk_index=0)]
    state: QAState = {
        "question": "مهلت ارسال پیشنهاد چیست؟",
        "document_id": DOCUMENT_ID,
        "retrieved_docs": docs,
    }
    result = generate_node(state, chat_model=FakeChatModel())
    assert "answer" in result
    assert result["answer"]
    assert "مهلت" in result["answer"] or "پایان ماه" in result["answer"]


def test_generate_node_raises_chat_api_error() -> None:
    class FailingChatModel(FakeChatModel):
        def invoke(self, input, config=None, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated API failure")

    docs = [_make_doc("متن تست.", chunk_index=0)]
    state: QAState = {
        "question": "سوال؟",
        "document_id": DOCUMENT_ID,
        "retrieved_docs": docs,
    }
    with pytest.raises(ChatAPIError, match="simulated API failure"):
        generate_node(state, chat_model=FailingChatModel())


def test_format_sources_node_creates_ui_ready_sources() -> None:
    docs = [
        _make_doc("متن صفحه اول.", chunk_index=0, page=1, page_chunk_index=0),
        _make_doc("متن صفحه دوم.", chunk_index=1, page=2, page_chunk_index=0),
    ]
    state: QAState = {
        "question": "سوال؟",
        "document_id": DOCUMENT_ID,
        "retrieved_docs": docs,
    }
    result = format_sources_node(state)
    assert "sources" in result
    assert len(result["sources"]) == 2

    first = result["sources"][0]
    assert first["page"] == 1
    assert first["chunk_index"] == 0
    assert first["page_chunk_index"] == 0
    assert first["source"] == str(SOURCE_PATH)
    assert first["file_name"] == "graph-test-tender.pdf"
    assert "متن صفحه اول" in first["text_preview"]


def test_format_sources_node_matches_rag_source_asdict() -> None:
    docs = [_make_doc("متن نمونه.", chunk_index=3, page=5)]
    state: QAState = {
        "question": "سوال؟",
        "document_id": DOCUMENT_ID,
        "retrieved_docs": docs,
    }
    from dataclasses import asdict

    expected = [asdict(source) for source in format_sources(docs)]
    result = format_sources_node(state)
    assert result["sources"] == expected


def test_compiled_graph_invoke_returns_expected_fields(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    compiled = get_compiled_rag_graph(
        chat_model=FakeChatModel(),
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
    )
    result = compiled.invoke(
        {
            "question": "مهلت ارسال پیشنهاد تا پایان ماه است.",
            "document_id": DOCUMENT_ID,
            "top_k": 1,
        }
    )

    assert result["question"] == "مهلت ارسال پیشنهاد تا پایان ماه است."
    assert result["document_id"] == DOCUMENT_ID
    assert len(result["retrieved_docs"]) == 1
    assert result["answer"]
    assert len(result["sources"]) == 1
    assert isinstance(result["sources"][0], dict)
    assert "page" in result["sources"][0]
    assert "text_preview" in result["sources"][0]


def test_compiled_graph_flow_order(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    call_order: list[str] = []

    original_retrieve = retrieve_node
    original_generate = generate_node
    original_format = format_sources_node

    def tracked_retrieve(state, **kwargs):  # type: ignore[no-untyped-def]
        call_order.append("retrieve")
        return original_retrieve(state, **kwargs)

    def tracked_generate(state, **kwargs):  # type: ignore[no-untyped-def]
        call_order.append("generate")
        return original_generate(state, **kwargs)

    def tracked_format(state):  # type: ignore[no-untyped-def]
        call_order.append("format_sources")
        return original_format(state)

    from functools import partial

    from langgraph.graph import END, START, StateGraph

    from agent.state import QAState as _QAState

    graph = StateGraph(_QAState)
    graph.add_node(
        "retrieve",
        partial(
            tracked_retrieve,
            embedding_function=FakeEmbeddings(),
            persist_directory=tmp_path,
        ),
    )
    graph.add_node(
        "generate",
        partial(tracked_generate, chat_model=FakeChatModel()),
    )
    graph.add_node("format_sources", tracked_format)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "format_sources")
    graph.add_edge("format_sources", END)

    compiled = graph.compile()
    compiled.invoke(
        {
            "question": "مهلت ارسال پیشنهاد تا پایان ماه است.",
            "document_id": DOCUMENT_ID,
            "top_k": 1,
        }
    )
    assert call_order == ["retrieve", "generate", "format_sources"]


def test_compiled_graph_unknown_document_raises(tmp_path: Path) -> None:
    compiled = get_compiled_rag_graph(
        chat_model=FakeChatModel(),
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
    )
    with pytest.raises(CollectionNotFoundError):
        compiled.invoke(
            {
                "question": "سوال تست",
                "document_id": "missing-doc",
            }
        )


def test_rag_service_ask_still_returns_rag_answer(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    service = RAGService(
        chat_model=FakeChatModel(),
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
    )
    result = service.ask("مهلت ارسال پیشنهاد چیست؟", DOCUMENT_ID, top_k=1)

    assert result.question == "مهلت ارسال پیشنهاد چیست؟"
    assert result.document_id == DOCUMENT_ID
    assert result.retrieved_count == 1
    assert result.model_name == "fake-chat-model"
    assert isinstance(result.sources[0], RAGSource)
    assert len(result.sources) == 1
    assert "مهلت" in result.answer or "پایان ماه" in result.answer


def test_rag_service_graph_invocation_error(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    service = RAGService(
        chat_model=FakeChatModel(),
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
    )

    class BrokenGraph:
        def invoke(self, state):  # type: ignore[no-untyped-def]
            raise ValueError("broken graph")

    service._compiled_graph = BrokenGraph()
    with pytest.raises(GraphInvocationError, match="broken graph"):
        service.ask("مهلت ارسال پیشنهاد چیست؟", DOCUMENT_ID, top_k=1)
