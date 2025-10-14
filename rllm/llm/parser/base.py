from abc import ABC, abstractmethod
from typing import Any, List
from rllm.llm.types import MessageRole, ChatMessage


# ===== Generic Model Output - Parser =====
class BaseOutputParser(ABC):
    """Output parser class."""

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        raise NotImplementedError

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        return query

    def format_messages(
        self,
        messages: List[ChatMessage]
    ) -> List[ChatMessage]:
        """Format a list of messages with structured
        output formatting instructions."""
        # NOTE: apply output parser to either the first message
        # if it's a system message or the last message
        if messages:
            if messages[0].role == MessageRole.SYSTEM:
                messages[0].content = self.format(messages[0].content or "")
            else:
                messages[-1].content = self.format(messages[-1].content or "")

        return messages
