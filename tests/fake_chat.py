"""Fake chat model for RAG service tests (no OpenAI calls)."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig


class FakeChatModel(BaseChatModel):
    """Deterministic chat model that returns a canned or context-aware response."""

    model_name: str = "fake-chat-model"
    response_text: str = "پاسخ آزمایشی بر اساس متن بازیابی‌شده."

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = self._build_response(messages)
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _build_response(self, messages: list[BaseMessage]) -> str:
        combined = "\n".join(str(message.content) for message in messages)
        if "مهلت ارسال پیشنهاد" in combined and "پرسش کاربر" in combined:
            return "طبق صفحه ۱، مهلت ارسال پیشنهاد تا پایان ماه است."
        if "موضوعی که در سند نیست" in combined:
            return "پاسخ این پرسش در سند مناقصه ارائه‌شده یافت نشد."
        return self.response_text

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def invoke(
        self,
        input: list[BaseMessage],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        result = self._generate(input, **kwargs)
        return result.generations[0].message
