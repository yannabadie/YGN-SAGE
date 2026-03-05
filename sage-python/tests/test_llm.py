"""Tests for the LLM abstraction layer."""
import pytest
from sage.llm.base import LLMConfig, Message, ToolDef, LLMResponse, Role


def test_message_creation():
    msg = Message(role=Role.USER, content="Hello")
    assert msg.role == Role.USER
    assert msg.content == "Hello"


def test_llm_config():
    config = LLMConfig(
        provider="anthropic",
        model="claude-opus-4-6",
        max_tokens=4096,
        temperature=0.7,
    )
    assert config.provider == "anthropic"
    assert config.model == "claude-opus-4-6"


def test_tool_def():
    tool = ToolDef(
        name="bash",
        description="Execute a bash command",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run"}
            },
            "required": ["command"],
        },
    )
    assert tool.name == "bash"


def test_llm_config_defaults():
    config = LLMConfig(provider="openai", model="gpt-5")
    assert config.max_tokens == 8192
    assert config.temperature == 0.0
