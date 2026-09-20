from __future__ import annotations

import json

from rllm.llm.types import ChatMessage, MessageRole


def test_chat_message_dict_exposes_public_fields_and_is_json_serializable():
    message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="done",
        additional_kwargs={"nested": {"count": 2}, "items": [1, "two"]},
    )

    serialized = message.dict()

    assert serialized == {
        "role": MessageRole.ASSISTANT,
        "content": "done",
        "additional_kwargs": {"nested": {"count": 2}, "items": [1, "two"]},
    }
    assert json.loads(json.dumps(serialized))["role"] == "assistant"


def test_chat_message_string_representation_includes_role_and_content():
    message = ChatMessage.from_str("hello", role="user")

    assert str(message) == "user: hello"
