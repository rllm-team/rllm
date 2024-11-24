from typing import Callable, Optional, Sequence

from rllm.llm.types import (
    LLMMetadata,
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from rllm.llm.llm_module.general_llm import LLM
from rllm.llm.parser.base import BaseOutputParser
from rllm.llm.prompt.utils import completion_response_to_chat_response

from langchain.base_language import BaseLanguageModel


class LangChainLLM(LLM):
    """Adapter for a LangChain LLM.

    Examples:
        `pip install llama-index-llms-langchain`

        ```python
        from langchain_openai import ChatOpenAI

        from rllm.llm.llm_module.langchain import LangChainLLM

        llm = LangChainLLM(llm=ChatOpenAI(...))

        response_gen = llm.complete("What is the meaning of life?")
        ```
    """

    def __init__(
        self,
        llm: "BaseLanguageModel",
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        self._llm = llm
        super().__init__(
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "LangChainLLM"

    @property
    def llm(self) -> "BaseLanguageModel":
        return self._llm

    @property
    def metadata(self) -> LLMMetadata:
        from rllm.llm.llm_module.langchain_utils import get_llm_metadata

        return get_llm_metadata(self._llm)

    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        from rllm.llm.llm_module.langchain_utils import (
            from_lc_messages,
            to_lc_messages,
        )

        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            completion_response = self.complete(prompt, formatted=True, **kwargs)
            return completion_response_to_chat_response(completion_response)

        lc_messages = to_lc_messages(messages)
        lc_message = self._llm.predict_messages(messages=lc_messages, **kwargs)
        message = from_lc_messages([lc_message])[0]
        return ChatResponse(message=message)

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponse:
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        output_str = self._llm.predict(prompt, **kwargs)
        return CompletionResponse(text=output_str)

    def embedding(self, inputs):
        assert hasattr(self._llm, "embed_documents"), (
            "An embedding model should be provided!"
            "See https://python.langchain.com/v0.1/docs/integrations/text_embedding/"
        )  # noqa
        if isinstance(inputs, str):
            inputs = [inputs]
        return self._llm.embed_documents(inputs)
