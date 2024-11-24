from abc import ABC, abstractmethod
from typing import Sequence

from rllm.llm.types import (
    ChatMessage,
    LLMMetadata,
    ChatResponse,
    CompletionResponse,
)


class BaseLLM(ABC):
    """BaseLLM interface."""

    @property
    @abstractmethod
    def metadata(self) -> LLMMetadata:
        """LLM metadata.

        Returns:
            LLMMetadata:
                LLM metadata containing various information about the LLM.
        """
        raise NotImplementedError

    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        """Chat endpoint for LLM.

        Args:
            messages (Sequence[ChatMessage]):
                Sequence of chat messages.
            kwargs (Any):
                Additional keyword arguments to pass to the LLM.

        Returns:
            ChatResponse: Chat response from the LLM.

        Examples:
            ```python
            from rllm.llm.types import ChatMessage

            response = llm.chat([ChatMessage(role="user", content="Hello")])
            print(response.content)
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponse:
        """Completion endpoint for LLM.

        If the LLM is a chat model, the prompt is transformed into
        a single `user` message.

        Args:
            prompt (str):
                Prompt to send to the LLM.
            formatted (bool, optional):
                Whether the prompt is already formatted for the LLM,
                by default False.
            kwargs (Any):
                Additional keyword arguments to pass to the LLM.

        Returns:
            CompletionResponse: Completion response from the LLM.

        Examples:
            ```python
            response = llm.complete("your prompt")
            print(response.text)
            ```
        """
        raise NotImplementedError
