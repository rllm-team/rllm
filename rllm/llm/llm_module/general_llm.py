from typing import (
    Any,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable
)

from rllm.llm.types import (
    ChatMessage,
    MessageRole,
)
from rllm.llm.prompt.base import BasePromptTemplate
from rllm.llm.prompt.utils import messages_to_prompt
from rllm.llm.llm_module.base import BaseLLM
from rllm.llm.parser.base import BaseOutputParser


default_messages_to_prompt = messages_to_prompt


def default_completion_to_prompt(prompt: str) -> str:
    return prompt


# NOTE: These two protocols are only to appease mypy
@runtime_checkable
class MessagesToPromptType(Protocol):
    def __call__(self, messages: Sequence[ChatMessage]) -> str:
        pass


@runtime_checkable
class CompletionToPromptType(Protocol):
    def __call__(self, prompt: str) -> str:
        pass


class LLM(BaseLLM):
    """
    The LLM class is the main class for interacting with language models.

    Args:
        system_prompt (Optional[str]):
            System prompt for LLM calls.
        messages_to_prompt (Callable):
            Function to convert a list of messages to an LLM prompt.
        completion_to_prompt (Callable):
            Function to convert a completion to an LLM prompt.
        output_parser (Optional[BaseOutputParser]):
            Output parser to parse, validate,
            and correct errors programmatically.
    """
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[MessagesToPromptType] = None,
        completion_to_prompt: Optional[CompletionToPromptType] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ):
        super().__init__()
        self.system_prompt = system_prompt
        self.messages_to_prompt = \
            messages_to_prompt or default_messages_to_prompt
        self.completion_to_prompt = \
            completion_to_prompt or default_completion_to_prompt
        self.output_parser = output_parser

    def _get_prompt(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any
    ) -> str:
        formatted_prompt = prompt.format(
            llm=self,
            messages_to_prompt=self.messages_to_prompt,
            completion_to_prompt=self.completion_to_prompt,
            **prompt_args,
        )
        if self.output_parser is not None:
            formatted_prompt = self.output_parser.format(formatted_prompt)
        # print(self._extend_prompt(formatted_prompt))
        # exit(0)
        return self._extend_prompt(formatted_prompt)

    def _get_messages(
        self, prompt: BasePromptTemplate, **prompt_args: Any
    ) -> List[ChatMessage]:
        messages = prompt.format_messages(llm=self, **prompt_args)
        if self.output_parser is not None:
            messages = self.output_parser.format_messages(messages)
        return self._extend_messages(messages)

    def _parse_output(self, output: str) -> str:
        if self.output_parser is not None:
            return str(self.output_parser.parse(output))

        return output

    def _extend_prompt(
        self,
        formatted_prompt: str,
    ) -> str:
        """Add system and query wrapper prompts to base prompt."""
        extended_prompt = formatted_prompt

        if self.system_prompt:
            extended_prompt = self.system_prompt + "\n\n" + extended_prompt

        return extended_prompt

    def _extend_messages(
        self,
        messages: List[ChatMessage]
    ) -> List[ChatMessage]:
        """Add system prompt to chat message list."""
        if self.system_prompt:
            messages = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=self.system_prompt
                ),
                *messages,
            ]
        return messages

    # -- Prompt Predict --
    def predict(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> str:
        """Predict for a given prompt.

        Args:
            prompt (BasePromptTemplate):
                The prompt to use for prediction.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Returns:
            str: The prediction output.

        Examples:
            ```python
            from rllm.llm.prompts import PromptTemplate

            prompt = PromptTemplate(
                f"Please write a random name related to {topic}."
            )
            output = llm.predict(prompt, topic="cats")
            print(output)
            ```
        """
        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = self.chat(messages)
            output = chat_response.message.content or ""
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            response = self.complete(formatted_prompt, formatted=True)
            output = response.text
        parsed_output = self._parse_output(output)
        return parsed_output
