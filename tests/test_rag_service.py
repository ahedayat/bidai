"""RAG service tests (Phase 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from core.exceptions import (
    ChatAPIError,
    EmptyQuestionError,
    NoRetrievedDocumentsError,
)
from core.models import RAGSource
from ingestion.indexer import index_documents
from services.rag_service import (
    RAGService,
    format_context_block,
    format_source,
    format_sources,
)
from tests.fake_chat import FakeChatModel
from tests.fake_embeddings import FakeEmbeddings

SOURCE_PATH = Path("/tmp/rag-test-tender.pdf").resolve()
DOCUMENT_ID = "test-rag-doc"


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
            "file_name": "rag-test-tender.pdf",
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


def test_format_source_includes_required_fields() -> None:
    doc = _make_doc("متن نمونه برای آزمایش.", chunk_index=3, page=5, page_chunk_index=1)
    source = format_source(doc)
    assert isinstance(source, RAGSource)
    assert source.page == 5
    assert source.chunk_index == 3
    assert source.page_chunk_index == 1
    assert source.source == str(SOURCE_PATH)
    assert source.file_name == "rag-test-tender.pdf"
    assert "متن نمونه" in source.text_preview


def test_format_source_truncates_long_preview() -> None:
    long_text = "الف" * 300
    doc = _make_doc(long_text, chunk_index=0)
    source = format_source(doc, preview_len=50)
    assert len(source.text_preview) <= 50
    assert source.text_preview.endswith("…")


def test_format_sources_returns_tuple() -> None:
    docs = [
        _make_doc("اول", chunk_index=0, page=1),
        _make_doc("دوم", chunk_index=1, page=2),
    ]
    sources = format_sources(docs)
    assert len(sources) == 2
    assert sources[0].chunk_index == 0
    assert sources[1].page == 2


def test_format_context_block_includes_page_and_chunk_metadata() -> None:
    docs = [
        _make_doc("متن صفحه اول.", chunk_index=0, page=1, page_chunk_index=0),
        _make_doc("متن صفحه دوم.", chunk_index=1, page=2, page_chunk_index=0),
    ]
    context = format_context_block(docs)
    assert "[صفحه 1" in context
    assert "بخش 0" in context
    assert "[صفحه 2" in context
    assert "بخش 1" in context
    assert "متن صفحه اول" in context
    assert "متن صفحه دوم" in context
    assert "---" in context


def test_rag_service_returns_answer_and_sources(tmp_path: Path) -> None:
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
    assert "مهلت" in result.answer or "پایان ماه" in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].page == 1


def test_rag_service_rejects_empty_question(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    service = RAGService(
        chat_model=FakeChatModel(),
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
    )
    with pytest.raises(EmptyQuestionError):
        service.ask("  ", DOCUMENT_ID)


def test_rag_service_unknown_document_raises(tmp_path: Path) -> None:
    service = RAGService(
        chat_model=FakeChatModel(),
        persist_directory=tmp_path,
    )
    with pytest.raises(Exception):  # CollectionNotFoundError
        service.ask("سوال تست", "nonexistent-doc")


def test_rag_service_no_context_handled_by_fake_model(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)
    service = RAGService(
        chat_model=FakeChatModel(
            response_text="پاسخ این پرسش در سند مناقصه ارائه‌شده یافت نشد.",
        ),
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
    )
    result = service.ask("موضوعی که در سند نیست چیست؟", DOCUMENT_ID, top_k=1)
    assert "یافت نشد" in result.answer


def test_rag_service_chat_api_failure(tmp_path: Path) -> None:
    _index_sample_docs(tmp_path)

    class FailingChatModel(FakeChatModel):
        def invoke(self, input, config=None, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated API failure")

    service = RAGService(
        chat_model=FailingChatModel(),
        embedding_function=FakeEmbeddings(),
        persist_directory=tmp_path,
    )
    with pytest.raises(ChatAPIError, match="simulated API failure"):
        service.ask("مهلت ارسال پیشنهاد چیست؟", DOCUMENT_ID, top_k=1)


def test_create_chat_model_requires_api_key() -> None:
    from core.exceptions import MissingOpenAIAPIKeyError
    from services.openai_client import create_chat_model

    with pytest.raises(MissingOpenAIAPIKeyError):
        create_chat_model(api_key="")
    with pytest.raises(MissingOpenAIAPIKeyError):
        create_chat_model(api_key="   ")
