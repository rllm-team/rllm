from typing import Sequence, List

from rllm.llm.types import ChatMessage, LLMMetadata, MessageRole


class LC:
    from langchain.base_language import BaseLanguageModel
    from langchain.chat_models.base import BaseChatModel
    from langchain_community.llms import OpenAI
    from langchain_community.chat_models import ChatOpenAI
    from langchain.schema import (
        AIMessage,
        BaseMessage,
        ChatMessage,
        FunctionMessage,
        HumanMessage,
        SystemMessage,
    )


def is_chat_model(llm: LC.BaseLanguageModel) -> bool:
    return isinstance(llm, LC.BaseChatModel)


def get_llm_metadata(llm: LC.BaseLanguageModel) -> LLMMetadata:
    """Get LLM metadata from llm."""
    if not isinstance(llm, LC.BaseLanguageModel):
        raise ValueError("llm must be instance of LangChain BaseLanguageModel")

    is_chat_model_ = is_chat_model(llm)

    if isinstance(llm, LC.OpenAI):
        return LLMMetadata(
            context_window=llm.modelname_to_contextsize(llm.model_name),
            num_output=llm.max_tokens,
            is_chat_model=is_chat_model_,
            model_name=llm.model_name,
        )
    elif isinstance(llm, LC.ChatOpenAI):
        return LLMMetadata(
            context_window=llm.modelname_to_contextsize(llm.model_name),
            num_output=llm.max_tokens or -1,
            is_chat_model=is_chat_model_,
            model_name=llm.model_name,
        )
    else:
        return LLMMetadata(is_chat_model=is_chat_model_)


def to_lc_messages(messages: Sequence[ChatMessage]) -> List[LC.BaseMessage]:
    lc_messages: List[LC.BaseMessage] = []
    for message in messages:
        LC_MessageClass = LC.BaseMessage
        lc_kw = {
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
        }
        if message.role == "user":
            LC_MessageClass = LC.HumanMessage
        elif message.role == "assistant":
            LC_MessageClass = LC.AIMessage
        elif message.role == "function":
            LC_MessageClass = LC.FunctionMessage
        elif message.role == "system":
            LC_MessageClass = LC.SystemMessage
        elif message.role == "chatbot":
            LC_MessageClass = LC.ChatMessage
        else:
            raise ValueError(f"Invalid role: {message.role}")

        for req_key in LC_MessageClass.schema().get("required"):
            if req_key not in lc_kw:
                more_kw = lc_kw.get("additional_kwargs")
                if not isinstance(more_kw, dict):
                    raise ValueError(
                        f"additional_kwargs must be a dict, "
                        f"got {type(more_kw)}"
                    )
                if req_key not in more_kw:
                    raise ValueError(f"{req_key} needed for {LC_MessageClass}")
                lc_kw[req_key] = more_kw.pop(req_key)

        lc_messages.append(LC_MessageClass(**lc_kw))

    return lc_messages


def from_lc_messages(
    lc_messages: Sequence[LC.BaseMessage]
) -> List[ChatMessage]:
    messages: List[ChatMessage] = []
    for lc_message in lc_messages:
        li_kw = {
            "content": lc_message.content,
            "additional_kwargs": lc_message.additional_kwargs,
        }
        if isinstance(lc_message, LC.HumanMessage):
            li_kw["role"] = MessageRole.USER
        elif isinstance(lc_message, LC.AIMessage):
            li_kw["role"] = MessageRole.ASSISTANT
        elif isinstance(lc_message, LC.FunctionMessage):
            li_kw["role"] = MessageRole.FUNCTION
        elif isinstance(lc_message, LC.SystemMessage):
            li_kw["role"] = MessageRole.SYSTEM
        elif isinstance(lc_message, LC.ChatMessage):
            li_kw["role"] = MessageRole.CHATBOT
        else:
            raise ValueError(f"Invalid message type: {type(lc_message)}")
        messages.append(ChatMessage(**li_kw))

    return messages
