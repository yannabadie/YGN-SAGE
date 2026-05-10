import json
import logging
import sys
import types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

from sage.llm.base import Message, Role, ToolCall
from sage.providers.openai_compat import (
    OpenAICompatProvider,
    supports_chat_completions_model,
)


def test_openai_preserves_tool_role_and_assistant_tool_calls():
    """OpenAI chat-completions must receive assistant tool_calls plus tool messages."""
    provider = OpenAICompatProvider(api_key="test", provider_name="openai")
    tool_call = ToolCall(id="call_123", name="execute_bash", arguments={"command": "pwd"})
    messages = [
        Message(role=Role.USER, content="Hello"),
        Message(role=Role.ASSISTANT, content="", tool_calls=[tool_call]),
        Message(
            role=Role.TOOL,
            content="C:/Code/YGN-SAGE",
            tool_call_id="call_123",
            name="execute_bash",
        ),
    ]

    converted = provider._convert_messages(messages)

    assert converted[1]["role"] == "assistant"
    assert converted[1]["tool_calls"][0]["id"] == "call_123"
    assert converted[1]["tool_calls"][0]["function"]["name"] == "execute_bash"
    assert json.loads(converted[1]["tool_calls"][0]["function"]["arguments"]) == {"command": "pwd"}
    assert converted[2]["role"] == "tool"
    assert converted[2]["tool_call_id"] == "call_123"
    assert converted[2]["content"] == "C:/Code/YGN-SAGE"


def test_non_openai_tool_role_rewrite_logs_warning(caplog):
    """Non-OpenAI compat providers still degrade tool messages conservatively."""
    provider = OpenAICompatProvider(api_key="test", provider_name="deepseek")
    messages = [
        Message(role=Role.SYSTEM, content="You are helpful"),
        Message(role=Role.USER, content="Hello"),
        Message(role=Role.TOOL, content="Tool result here", tool_call_id="call_1"),
    ]
    with caplog.at_level(logging.WARNING):
        converted = provider._convert_messages(messages)

    assert converted[2]["role"] == "user"
    assert "tool_call_id" not in converted[2]
    assert any("tool" in r.message.lower() and "user" in r.message.lower() for r in caplog.records)


def test_non_tool_messages_unchanged():
    """Verify system and user roles are not modified."""
    provider = OpenAICompatProvider(api_key="test")
    messages = [
        Message(role=Role.SYSTEM, content="System"),
        Message(role=Role.USER, content="User"),
    ]
    converted = provider._convert_messages(messages)
    assert converted[0]["role"] == "system"
    assert converted[1]["role"] == "user"


def test_openai_compat_rejects_gpt_55_pro_chat_path():
    """Deprecated chat-completions fallback shares the active routing policy."""
    assert supports_chat_completions_model("openai", "gpt-5.5") is True
    assert supports_chat_completions_model("openai", "gpt-5.5-pro") is False
    assert supports_chat_completions_model("openai", "gpt-5-pro") is False
    assert supports_chat_completions_model("deepseek", "gpt-5.5-pro") is True
