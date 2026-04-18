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

    def test_minimax_native_prefix(self) -> None:
        p = LiteLLMProvider.for_sage_provider("minimax", "minimax-m2.7", "sk-mm")
        assert p.model_string == "minimax/minimax-m2.7"
        assert p.api_base is None  # LiteLLM handles minimax natively

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
async def test_gemini3_temperature_forced_to_one() -> None:
    """Gemini 3.x needs temperature=1.0 or it enters degenerate regimes.

    LiteLLM logs a loud warning on every call when temperature<1.0 on
    Gemini 3 models, and the 2026-04-17 SWE-bench smoke observed those
    same models falling into "infinite loops / degraded reasoning /
    failure on complex tasks". The provider now auto-overrides the
    temperature for gemini-3 model strings regardless of config.
    """
    mock_resp = _make_response(content="ok", prompt_tokens=1, completion_tokens=1, response_cost=0.0)
    provider = LiteLLMProvider(model_string="gemini/gemini-3.1-flash-lite-preview")

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_call:
        await provider.generate(
            messages=[Message(role=Role.USER, content="hi")],
            config=LLMConfig(provider="google", model="gemini-3.1-flash-lite-preview"),
        )

    assert mock_call.call_args.kwargs["temperature"] == 1.0


@pytest.mark.asyncio
async def test_non_gemini_preserves_config_temperature() -> None:
    """Other providers must keep the caller's temperature (determinism)."""
    mock_resp = _make_response(content="ok", prompt_tokens=1, completion_tokens=1, response_cost=0.0)
    provider = LiteLLMProvider(model_string="deepseek/deepseek-chat")

    class _Cfg:
        temperature = 0.25
        max_tokens = 100

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_call:
        await provider.generate(
            messages=[Message(role=Role.USER, content="hi")],
            config=_Cfg(),
        )

    assert mock_call.call_args.kwargs["temperature"] == 0.25


@pytest.mark.asyncio
async def test_gemini2_unaffected_by_override() -> None:
    """Gemini 2.x models keep the caller's temperature — override is strictly
    for Gemini 3.x (the family that LiteLLM and Google flag as fragile)."""
    mock_resp = _make_response(content="ok", prompt_tokens=1, completion_tokens=1, response_cost=0.0)
    provider = LiteLLMProvider(model_string="gemini/gemini-2.5-flash")

    class _Cfg:
        temperature = 0.3
        max_tokens = 100

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_call:
        await provider.generate(
            messages=[Message(role=Role.USER, content="hi")],
            config=_Cfg(),
        )

    assert mock_call.call_args.kwargs["temperature"] == 0.3


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


# --- Per-model routing tests (Codex review 2026-04-18) ---


@pytest.mark.asyncio
async def test_per_model_routing_honors_config_model():
    """When config.model names a different model than adapter default,
    LiteLLM should be called with the config model, not the adapter's.

    Regression test for: ModelAssigner decisions were silently discarded
    because LiteLLMProvider.generate() always sent self.model_string.
    """
    from sage.providers.litellm_provider import LiteLLMProvider
    from sage.llm.base import Message, Role, LLMConfig
    from unittest.mock import patch, AsyncMock, MagicMock

    provider = LiteLLMProvider.for_sage_provider("deepseek", "deepseek-chat", "sk-test")
    assert provider.model_string == "deepseek/deepseek-chat"

    # Request a different model via config
    cfg = LLMConfig(provider="gemini", model="gemini-3.1-flash-lite-preview", max_tokens=100)

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="hi", tool_calls=None), finish_reason="stop")]
    mock_resp.usage = MagicMock(prompt_tokens=5, completion_tokens=1, total_tokens=6)

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_call:
        await provider.generate(messages=[Message(role=Role.USER, content="ok")], config=cfg)

    # The adapter default is deepseek but config asked for gemini.
    called_with = mock_call.call_args.kwargs["model"]
    assert called_with == "gemini/gemini-3.1-flash-lite-preview", \
        f"Expected per-model routing to honor config.model; got {called_with!r}"


@pytest.mark.asyncio
async def test_per_model_routing_falls_back_to_adapter_default():
    """When config.model is None or empty, use adapter default."""
    from sage.providers.litellm_provider import LiteLLMProvider
    from sage.llm.base import Message, Role, LLMConfig
    from unittest.mock import patch, AsyncMock, MagicMock

    provider = LiteLLMProvider.for_sage_provider("deepseek", "deepseek-chat", "sk-test")
    cfg = LLMConfig(provider="", model="", max_tokens=100)

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="hi", tool_calls=None), finish_reason="stop")]
    mock_resp.usage = MagicMock(prompt_tokens=5, completion_tokens=1, total_tokens=6)

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_call:
        await provider.generate(messages=[Message(role=Role.USER, content="ok")], config=cfg)

    called_with = mock_call.call_args.kwargs["model"]
    assert called_with == "deepseek/deepseek-chat"


@pytest.mark.asyncio
async def test_per_model_routing_accepts_prefixed_model():
    """If config.model already has a / in it, treat as pre-formatted and pass through."""
    from sage.providers.litellm_provider import LiteLLMProvider
    from sage.llm.base import Message, Role, LLMConfig
    from unittest.mock import patch, AsyncMock, MagicMock

    provider = LiteLLMProvider.for_sage_provider("deepseek", "deepseek-chat", "sk-test")
    cfg = LLMConfig(provider="gemini", model="gemini/gemini-2.0-flash", max_tokens=50)

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="hi", tool_calls=None), finish_reason="stop")]
    mock_resp.usage = MagicMock(prompt_tokens=5, completion_tokens=1, total_tokens=6)

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_call:
        await provider.generate(messages=[Message(role=Role.USER, content="ok")], config=cfg)

    called_with = mock_call.call_args.kwargs["model"]
    assert called_with == "gemini/gemini-2.0-flash"
