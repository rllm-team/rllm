from enum import Enum
from typing import Any, Dict, Optional, Union

from rllm.llm.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"
    CHATBOT = "chatbot"
    MODEL = "model"


# ===== Generic Model Input - Chat =====
class ChatMessage:
    """Chat message."""

    def __init__(
        self,
        role: Union[MessageRole, str] = MessageRole.USER,
        content: Optional[Any] = "",
        additional_kwargs: Optional[Dict] = None,
    ):
        if isinstance(role, str):
            role = MessageRole(role)
        self.role = role
        self.content = content
        self.additional_kwargs = additional_kwargs or {}

    def __str__(self) -> str:
        return f"{self.role.value}: {self.content}"

    @classmethod
    def from_str(
        cls,
        content: str,
        role: Union[MessageRole, str] = MessageRole.USER,
        **kwargs,
    ) -> "ChatMessage":
        if isinstance(role, str):
            role = MessageRole(role)
        return cls(role=role, content=content, **kwargs)

    def _recursive_serialization(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._recursive_serialization(value)
                for key, value in value.items()
            }
        if isinstance(value, list):
            return [self._recursive_serialization(item) for item in value]
        return value

    def dict(self, **kwargs) -> dict:
        # ensure all additional_kwargs are serializable
        msg = super().dict(**kwargs)

        for key, value in msg.get("additional_kwargs", {}).items():
            value = self._recursive_serialization(value)
            if not isinstance(value, (str, int, float, bool, dict, list, type(None))):
                raise ValueError(
                    f"Failed to serialize additional_kwargs value: {value}"
                )
            msg["additional_kwargs"][key] = value

        return msg


# ===== Generic Model Output - Chat =====
class ChatResponse:
    """Chat response."""

    def __init__(
        self,
        message: ChatMessage,
        raw: Optional[dict] = None,
        delta: Optional[str] = None,
        additional_kwargs: Dict = None,
    ):
        self.message = message
        self.raw = raw
        self.delta = delta
        self.additional_kwargs = (
            additional_kwargs if additional_kwargs is not None else {}
        )

    def __str__(self) -> str:
        return str(self.message)


# ===== Generic Model Output - Completion =====
class CompletionResponse:
    """
    Completion response.

    Args:
        text: Text content of the response if not streaming, or if streaming,
            the current extent of streamed text.
        additional_kwargs: Additional information on the response (i.e. token
            counts, function calling information).
        raw: Optional raw JSON that was parsed to populate text, if relevant.
        delta: New text that just streamed in (only relevant when streaming).
    """

    def __init__(
        self,
        text: str,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        raw: Optional[Dict[str, Any]] = None,
        delta: Optional[str] = None,
    ):
        self.text = text
        self.additional_kwargs = (
            additional_kwargs if additional_kwargs is not None else {}
        )
        self.raw = raw
        self.delta = delta

    def __str__(self) -> str:
        return self.text


class LLMMetadata:
    r"""The metadata for a certain LLM.
    Args:
        context_window (int):
            Total number of tokens the model can be input and output
            for one response.
        num_output (int):
            Number of tokens the model can output when generating a response.
        is_chat_model (bool):
            Set True if the model exposes a chat interface
            (i.e. can be passed a sequence of messages, rather than text),
            like OpenAI's" /v1/chat/completions endpoint.
        is_function_calling_model (bool):
            Set True if the model supports function calling messages,
            similar to OpenAI's function calling API. For example, converting
            'Email Anya to see if she wants to get coffee next Friday' to a
            function call like `send_email(to: string, body: string)`.
        model_name (str):
            The model's name used for logging, testing, and sanity checking.
            For some models this can be automatically discerned.
            For other models, like locally loaded models,
            this must be manually specified.
        system_role (MessageRole): expects for system prompt.
            E.g. 'SYSTEM' for OpenAI, 'CHATBOT' for Cohere.
    """

    def __init__(
        self,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        num_output: int = DEFAULT_NUM_OUTPUTS,
        is_chat_model: bool = False,
        is_function_calling_model: bool = False,
        model_name: str = "unknown",
        system_role: MessageRole = MessageRole.SYSTEM,
    ):
        self.context_window = context_window
        self.num_output = num_output
        self.is_chat_model = is_chat_model
        self.is_function_calling_model = is_function_calling_model
        self.model_name = model_name
        self.system_role = system_role
