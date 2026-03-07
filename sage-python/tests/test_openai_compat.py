import sys
import types
import logging

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

from sage.providers.openai_compat import OpenAICompatProvider
from sage.llm.base import Message, Role


def test_tool_role_rewrite_logs_warning(caplog):
    """Verify tool->user rewrite emits a warning."""
    provider = OpenAICompatProvider(api_key="test")
    messages = [
        Message(role=Role.SYSTEM, content="You are helpful"),
        Message(role=Role.USER, content="Hello"),
        Message(role=Role.TOOL, content="Tool result here"),
    ]
    with caplog.at_level(logging.WARNING):
        converted = provider._convert_messages(messages)

    assert converted[2]["role"] == "user"
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
