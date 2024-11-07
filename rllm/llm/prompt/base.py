from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, List, Dict, Union, Tuple, Sequence, Callable, Optional

from rllm.llm.types import ChatMessage
from rllm.llm.parser.base import BaseOutputParser
from rllm.llm.prompt.utils import (
    get_template_vars,
    prompt_to_messages,
    messages_to_prompt,
)


from rllm.llm.llm_module.base import BaseLLM

default_messages_to_prompt = messages_to_prompt


class BasePromptTemplate(ABC):
    def __init__(
        self,
        metadata: Dict[str, Any],
        template_vars: List[str],
        kwargs: Dict[str, str],
        output_parser: Optional[BaseOutputParser] = None,
        template_var_mappings: Optional[Dict[str, Any]] = None,
        function_mappings: Optional[Dict[str, Callable]] = None,
    ):
        self.metadata = metadata
        self.template_vars = template_vars
        self.kwargs = kwargs
        self.output_parser = output_parser
        self.template_var_mappings = template_var_mappings or {}
        self.function_mappings = function_mappings or {}

    def _map_template_vars(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """For keys in template_var_mappings, swap in the right keys."""
        template_var_mappings = self.template_var_mappings or {}
        return {template_var_mappings.get(k, k): v for k, v in kwargs.items()}

    def _map_function_vars(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """For keys in function_mappings,
        compute values and combine with kwargs.

        User can pass in functions instead of fixed values as format variables.
        For each function, we call the function with the current kwargs,
        get back the value, and then use that value in the template
        for the corresponding format variable.
        """
        function_mappings = self.function_mappings or {}
        # First generate the values for the functions
        new_kwargs = {}
        for k, v in function_mappings.items():
            # TODO: figure out what variables to pass into each function
            # Assuming we decide to use all kwargs
            new_kwargs[k] = v(**kwargs)

        # Then, add the fixed variables only if not in new_kwargs already
        # (implying that function mapping will override fixed variables)
        for k, v in kwargs.items():
            if k not in new_kwargs:
                new_kwargs[k] = v

        return new_kwargs

    def _map_all_vars(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Map both template and function variables.

        We (1) first call function mappings to compute functions,
        and then (2) call the template_var_mappings.
        """
        # Map function
        new_kwargs = self._map_function_vars(kwargs)
        # Map template vars (to point to
        # existing format vars in string template)
        return self._map_template_vars(new_kwargs)

    @abstractmethod
    def partial_format(self, **kwargs) -> "BasePromptTemplate":
        raise NotImplementedError

    @abstractmethod
    def format(self, llm: Optional[BaseLLM] = None, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def format_messages(
        self, llm: Optional[BaseLLM] = None, **kwargs
    ) -> List[ChatMessage]:
        raise NotImplementedError

    @abstractmethod
    def get_template(self, llm: Optional[BaseLLM] = None) -> str:
        raise NotImplementedError


class PromptTemplate(BasePromptTemplate):
    r"""Template used for completion."""

    def __init__(
        self,
        template: str,
        output_parser: Optional[BaseOutputParser] = None,
        metadata: Optional[Dict[str, Any]] = None,
        template_var_mappings: Optional[Dict[str, Any]] = None,
        function_mappings: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ) -> None:
        if metadata is None:
            metadata = {}
        self.template = template
        template_vars = get_template_vars(template)

        super().__init__(
            template_vars=template_vars,
            kwargs=kwargs,
            metadata=metadata,
            output_parser=output_parser,
            template_var_mappings=template_var_mappings,
            function_mappings=function_mappings,
        )

    def partial_format(self, **kwargs) -> "PromptTemplate":
        """Partially format the prompt."""
        # NOTE: this is a hack to get around deepcopy failing on output parser
        output_parser = self.output_parser
        self.output_parser = None

        # get function and fixed kwargs, and add that to a copy
        # of the current prompt object
        prompt = deepcopy(self)
        prompt.kwargs.update(kwargs)

        # NOTE: put the output parser back
        prompt.output_parser = output_parser
        self.output_parser = output_parser
        return prompt

    def format(
        self,
        llm: Optional[BaseLLM] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        **kwargs,
    ) -> str:
        """Format the prompt into a string."""
        del llm  # unused
        all_kwargs = {
            **self.kwargs,
            **kwargs,
        }

        mapped_all_kwargs = self._map_all_vars(all_kwargs)
        prompt = self.template.format(**mapped_all_kwargs)

        if self.output_parser is not None:
            prompt = self.output_parser.format(prompt)

        if completion_to_prompt is not None:
            prompt = completion_to_prompt(prompt)

        return prompt

    def format_messages(
        self, llm: Optional[BaseLLM] = None, **kwargs
    ) -> List[ChatMessage]:
        """Format the prompt into a list of chat messages."""
        del llm  # unused
        prompt = self.format(**kwargs)
        return prompt_to_messages(prompt)

    def get_template(self, llm: Optional[BaseLLM] = None) -> str:
        return self.template


class ChatPromptTemplate(BasePromptTemplate):
    r"""Template used for chat."""

    def __init__(
        self,
        message_templates: List[ChatMessage],
        output_parser: Optional[BaseOutputParser] = None,
        metadata: Optional[Dict[str, Any]] = None,
        template_var_mappings: Optional[Dict[str, Any]] = None,
        function_mappings: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ):
        if metadata is None:
            metadata = {}

        template_vars = []
        for message_template in message_templates:
            template_vars.extend(get_template_vars(message_template.content or ""))

        super().__init__(
            message_templates=message_templates,
            kwargs=kwargs,
            metadata=metadata,
            output_parser=output_parser,
            template_vars=template_vars,
            template_var_mappings=template_var_mappings,
            function_mappings=function_mappings,
        )

    @classmethod
    def from_messages(
        cls,
        message_templates: Union[List[Tuple[str, str]], List[ChatMessage]],
        **kwargs,
    ) -> "ChatPromptTemplate":
        """From messages."""
        if isinstance(message_templates[0], tuple):
            message_templates = [
                ChatMessage.from_str(role=role, content=content)
                for role, content in message_templates
            ]
        return cls(message_templates=message_templates, **kwargs)

    def partial_format(self, **kwargs) -> "ChatPromptTemplate":
        prompt = deepcopy(self)
        prompt.kwargs.update(kwargs)
        return prompt

    def format(
        self,
        llm: Optional[BaseLLM] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        **kwargs,
    ) -> str:
        del llm  # unused
        messages = self.format_messages(**kwargs)

        if messages_to_prompt is not None:
            return messages_to_prompt(messages)

        return default_messages_to_prompt(messages)

    def format_messages(
        self, llm: Optional[BaseLLM] = None, **kwargs
    ) -> List[ChatMessage]:
        del llm  # unused
        """Format the prompt into a list of chat messages."""
        all_kwargs = {
            **self.kwargs,
            **kwargs,
        }
        mapped_all_kwargs = self._map_all_vars(all_kwargs)

        messages: List[ChatMessage] = []
        for message_template in self.message_templates:
            template_vars = get_template_vars(message_template.content or "")
            relevant_kwargs = {
                k: v for k, v in mapped_all_kwargs.items() if k in template_vars
            }
            content_template = message_template.content or ""

            # if there's mappings specified, make sure those are used
            content = content_template.format(**relevant_kwargs)

            message: ChatMessage = message_template.copy()
            message.content = content
            messages.append(message)

        if self.output_parser is not None:
            messages = self.output_parser.format_messages(messages)

        return messages

    def get_template(self, llm: Optional[BaseLLM] = None) -> str:
        return default_messages_to_prompt(self.message_templates)
