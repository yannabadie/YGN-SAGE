"""Tests for LiteLLMProvider — unified LLM adapter backed by LiteLLM."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from sage.llm.base import LLMConfig, Message, Role, ToolCall, ToolDef
from sage.providers.litellm_provider import LiteLLMProvider


# ---------------------------------------------------------------------------
# Helpers to build mock LiteLLM responses
# ---------------------------------------------------------------------------

def _make_response(
    content: str = "Hello",
    tool_calls: list | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
    response_cost: float | None = None,
) -> SimpleNamespace:
    """Build a fake ``litellm.acompletion`` response."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    resp = SimpleNamespace(
        choices=[choice],
        usage=usage,
        _hidden_params={"response_cost": response_cost} if response_cost is not None else {},
    )
    return resp


# ---------------------------------------------------------------------------
# Factory / model-string tests (sync, no mocking needed)
# ---------------------------------------------------------------------------

class TestFactory:
    def test_deepseek(self) -> None:
        p = LiteLLMProvider.for_sage_provider("deepseek", "deepseek-chat", "sk-abc")
        assert p.model_string == "deepseek/deepseek-chat"
        assert p.api_base is None

    def test_google(self) -> None:
        p = LiteLLMProvider.for_sage_provider("google", "gemini-3.1-pro-preview")
        assert p.model_string == "gemini/gemini-3.1-pro-preview"

    def test_xai(self) -> None:
        p = LiteLLMProvider.for_sage_provider("xai", "grok-4-1-fast-reasoning")
        assert p.model_string == "xai/grok-4-1-fast-reasoning"

    def test_openrouter(self) -> None:
        p = LiteLLMProvider.for_sage_provider("openrouter", "qwen/qwen3.5-plus-02-15")
        assert p.model_string == "openrouter/qwen/qwen3.5-plus-02-15"

    def test_kimi_custom_base(self) -> None:
        p = LiteLLMProvider.for_sage_provider("kimi", "moonshot-v1-8k", "sk-kimi")
        assert p.model_string == "openai/moonshot-v1-8k"
        assert p.api_base == "https://api.moonshot.ai/v1"

    def test_minimax_custom_base(self) -> None:
        p = LiteLLMProvider.for_sage_provider("minimax", "minimax-m2.7", "sk-mm")
        assert p.model_string == "openai/minimax-m2.7"
        assert p.api_base == "https://api.minimax.io/v1"

    def test_openai_responses_prefix(self) -> None:
        p = LiteLLMProvider.for_sage_provider("openai", "gpt-5.4-pro")
        assert p.model_string == "openai/responses/gpt-5.4-pro"

    def test_openai_non_pro(self) -> None:
        p = LiteLLMProvider.for_sage_provider("openai", "gpt-5.4-mini")
        assert p.model_string == "openai/gpt-5.4-mini"


# ---------------------------------------------------------------------------
# Async generate tests (mock litellm.acompletion)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_basic() -> None:
    """Simple text response — check content, usage, and cost extraction."""
    mock_resp = _make_response(
        content="The answer is 42.",
        prompt_tokens=12,
        completion_tokens=8,
        response_cost=0.0003,
    )
    provider = LiteLLMProvider(model_string="deepseek/deepseek-chat", api_key="sk-test")

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_call:
        result = await provider.generate(
            messages=[Message(role=Role.USER, content="What is 6*7?")],
            config=LLMConfig(provider="deepseek", model="deepseek-chat", max_tokens=256),
        )

    assert result.content == "The answer is 42."
    assert result.usage is not None
    assert result.usage["input_tokens"] == 12
    assert result.usage["output_tokens"] == 8
    assert result.usage["total_tokens"] == 20
    assert result.usage["cost_usd"] == 0.0003
    assert result.model == "deepseek/deepseek-chat"
    assert result.stop_reason == "stop"

    # Verify acompletion was called with the right model string
    call_kwargs = mock_call.call_args.kwargs
    assert call_kwargs["model"] == "deepseek/deepseek-chat"
    assert call_kwargs["api_key"] == "sk-test"


@pytest.mark.asyncio
async def test_generate_with_tools() -> None:
    """Response containing tool_calls — verify ToolCall parsing."""
    raw_tool_calls = [
        SimpleNamespace(
            id="call_abc123",
            function=SimpleNamespace(
                name="get_weather",
                arguments=json.dumps({"city": "Paris", "units": "metric"}),
            ),
        ),
        SimpleNamespace(
            id="call_def456",
            function=SimpleNamespace(
                name="get_time",
                arguments=json.dumps({"timezone": "CET"}),
            ),
        ),
    ]
    mock_resp = _make_response(content="", tool_calls=raw_tool_calls, finish_reason="tool_calls")
    provider = LiteLLMProvider(model_string="openai/gpt-5.4-mini", api_key="sk-oai")

    tools = [
        ToolDef(
            name="get_weather",
            description="Get weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        ),
    ]

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
        result = await provider.generate(
            messages=[Message(role=Role.USER, content="Weather in Paris?")],
            tools=tools,
        )

    assert len(result.tool_calls) == 2
    tc0 = result.tool_calls[0]
    assert isinstance(tc0, ToolCall)
    assert tc0.id == "call_abc123"
    assert tc0.name == "get_weather"
    assert tc0.arguments == {"city": "Paris", "units": "metric"}

    tc1 = result.tool_calls[1]
    assert tc1.id == "call_def456"
    assert tc1.name == "get_time"
    assert tc1.arguments == {"timezone": "CET"}

    assert result.stop_reason == "tool_calls"


@pytest.mark.asyncio
async def test_generate_maps_messages() -> None:
    """Ensure SAGE messages (including tool results) are converted correctly."""
    mock_resp = _make_response(content="Done.")
    provider = LiteLLMProvider(model_string="gemini/gemini-3.1-pro-preview")

    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="What's the weather?"),
        Message(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[
                ToolCall(id="call_001", name="get_weather", arguments={"city": "Paris"}),
            ],
        ),
        Message(
            role=Role.TOOL,
            content='{"temp": 18, "condition": "sunny"}',
            tool_call_id="call_001",
            name="get_weather",
        ),
    ]

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_call:
        await provider.generate(messages=messages)

    sent_messages = mock_call.call_args.kwargs["messages"]

    # System message
    assert sent_messages[0] == {"role": "system", "content": "You are a helpful assistant."}

    # User message
    assert sent_messages[1] == {"role": "user", "content": "What's the weather?"}

    # Assistant with tool_calls
    assistant_msg = sent_messages[2]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] is None
    assert len(assistant_msg["tool_calls"]) == 1
    tc = assistant_msg["tool_calls"][0]
    assert tc["id"] == "call_001"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}

    # Tool result
    tool_msg = sent_messages[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_001"
    assert tool_msg["name"] == "get_weather"
    assert tool_msg["content"] == '{"temp": 18, "condition": "sunny"}'
