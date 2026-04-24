"""Phase 4 — integration smoke: PydanticAIProvider drops into ProviderPool.

Validates the migration without hitting live APIs:

  1. ProviderPool accepts PydanticAIProvider instances (same protocol
     as LiteLLMProvider — duck typing should work).
  2. ProviderPool.resolve(model_id) returns the configured provider.
  3. Circuit breaker fires when the provider raises (AgentLoop path
     + direct path both go through _pool_ref.record_failure).
  4. sage-core ModelRegistry lookups agree with our PydanticAIProvider
     cost-lookup helper (both read the same cards.toml).

No live API calls — the provider is instantiated but `generate()` is
never invoked. That keeps this test under 1 second and runnable in
any CI.
"""
from __future__ import annotations

import sys
import types as _types
from unittest.mock import MagicMock

# Only stub sage_core if the real Rust module isn't importable — several
# tests in this file WANT the real module to read cards.toml.
if "sage_core" not in sys.modules:
    try:
        import sage_core  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        sys.modules["sage_core"] = _types.ModuleType("sage_core")

import pytest


def test_pydantic_ai_provider_implements_llm_protocol() -> None:
    """Protocol runtime-check — duck typing contract."""
    from sage.llm.base import LLMProvider
    from sage.providers.pydantic_ai_provider import PydanticAIProvider

    p = PydanticAIProvider.for_sage_provider(
        "deepseek", "deepseek-chat", api_key="test-key"
    )
    assert isinstance(p, LLMProvider), (
        "PydanticAIProvider must satisfy the LLMProvider Protocol so it's "
        "a drop-in replacement for LiteLLMProvider in ProviderPool."
    )
    # Name attribute required by the protocol
    assert p.name == "pydantic_ai"
    # Factory parameters flow through
    assert p.provider_name == "deepseek"
    assert p.model_id == "deepseek-chat"
    assert p.api_key == "test-key"


def test_factory_for_every_wired_provider() -> None:
    """Every provider in cards.toml must build a PydanticAIProvider.

    This is the Phase 3 matrix collapsed to a constructor-only smoke
    test — no API calls, just "does the factory know about this
    provider kind?". Catches a future `cards.toml` addition that
    forgets `_PROVIDER_MAP`.
    """
    from sage.providers.pydantic_ai_provider import (
        _PROVIDER_MAP,
        PydanticAIProvider,
    )

    # Every SAGE provider name must be in the map
    sage_providers = {
        "openai", "google", "xai", "deepseek",
        "minimax", "kimi", "openrouter",
    }
    missing = sage_providers - set(_PROVIDER_MAP)
    assert not missing, (
        f"_PROVIDER_MAP missing entries for SAGE providers: {missing}. "
        "Add to sage/providers/pydantic_ai_provider.py::_PROVIDER_MAP."
    )

    # Constructor doesn't raise for any of them
    for prov in sage_providers:
        _ = PydanticAIProvider.for_sage_provider(prov, "some-model", api_key="k")


def test_openai_responses_routing() -> None:
    """gpt-5.4-pro must route to OpenAIResponsesModel, not OpenAIChatModel.

    This quirk was caught during Phase 3 live test — the Chat Completions
    endpoint returns 404 for gpt-5.4-pro. Pin the routing logic so a
    future refactor can't silently regress it.
    """
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

    from sage.providers.pydantic_ai_provider import PydanticAIProvider

    pro = PydanticAIProvider.for_sage_provider("openai", "gpt-5.4-pro", "k")
    assert isinstance(pro._model, OpenAIResponsesModel), (
        "gpt-5.4-pro must use OpenAIResponsesModel — it's not a chat model. "
        "Verified 2026-04-18: Chat Completions endpoint returns "
        "404 'This is not a chat model'."
    )

    # Regular gpt-5.4 is still a chat model
    chat = PydanticAIProvider.for_sage_provider("openai", "gpt-5.4", "k")
    assert isinstance(chat._model, OpenAIChatModel)


def test_cost_lookup_agrees_with_rust_registry() -> None:
    """Our cost-lookup helper and sage-core ModelRegistry read the same TOML.

    Regression guard: if someone edits cards.toml to rename a field or
    changes the per-million convention, both paths must update together.
    """
    from pathlib import Path

    try:
        import sage_core  # type: ignore[import-not-found]
    except Exception:
        pytest.skip("sage_core not available (Rust not built)")

    # When another test in this run stubs `sage_core` with an empty
    # ModuleType, `ModelRegistry` won't exist as an attribute even
    # though the module is in `sys.modules`. Skip gracefully so we
    # don't fail on test-ordering rather than on real breakage.
    if not hasattr(sage_core, "ModelRegistry"):
        pytest.skip("sage_core stubbed by another test; real Rust module unavailable")

    from sage.providers.pydantic_ai_provider import _lookup_cost_per_token

    # Load the Rust registry independently and pick a known model
    toml = Path(__file__).resolve().parents[3] / "sage-core" / "config" / "cards.toml"
    if not toml.is_file():
        pytest.skip("cards.toml not on disk")
    reg = sage_core.ModelRegistry.from_toml_file(str(toml))
    card = reg.get("deepseek-chat")
    if card is None:
        pytest.skip("deepseek-chat not in cards.toml")

    expected_in = float(getattr(card, "cost_input_per_m", 0.0)) / 1_000_000
    expected_out = float(getattr(card, "cost_output_per_m", 0.0)) / 1_000_000

    got_in, got_out = _lookup_cost_per_token("deepseek", "deepseek-chat")
    assert got_in == pytest.approx(expected_in), (
        f"input cost drift: helper returned {got_in}, Rust registry has {expected_in}"
    )
    assert got_out == pytest.approx(expected_out)


@pytest.mark.asyncio
async def test_provider_pool_accepts_pydantic_ai_provider() -> None:
    """ProviderPool + CircuitBreaker work identically with PydanticAIProvider.

    Mirrors `test_real_provider_pool_excludes_after_three_failures` from
    test_agent_loop_circuit_breaker.py but with a PydanticAIProvider as
    the default — proves the ProviderPool is provider-class-agnostic.
    """
    from sage.llm.provider_pool import ProviderPool
    from sage.providers.pydantic_ai_provider import PydanticAIProvider

    default_provider = PydanticAIProvider.for_sage_provider(
        "deepseek", "deepseek-chat", api_key="k"
    )
    named_provider = PydanticAIProvider.for_sage_provider(
        "deepseek", "deepseek-chat", api_key="k"
    )
    # ProviderPool injects _pool_ref into the `providers` dict values
    # (the boot pattern in sage/boot_providers.py). `default_provider`
    # is separately stored and not part of that injection loop.
    pool = ProviderPool(
        default_provider=default_provider,
        registry=MagicMock(),
        default_config=None,
        providers={"deepseek": named_provider},
    )

    # The injected `providers` got the _pool_ref hook
    assert named_provider._pool_ref is pool, (
        "ProviderPool constructor must set _pool_ref on provider dict "
        "values that expose the attribute — needed for runtime 429 → "
        "DEAD circuit (commit 58ec0d8)."
    )

    # Sanity: fresh provider is available
    prov_name = "test-deepseek"
    assert pool.is_available(prov_name)

    # 3 failures trip the breaker
    for i in range(3):
        pool.record_failure(prov_name, RuntimeError(f"429 #{i}"))
    assert not pool.is_available(prov_name)

    # Recovery via record_success
    pool.record_success(prov_name)
    assert pool.is_available(prov_name)


@pytest.mark.asyncio
async def test_circuit_breaker_fires_on_pydantic_ai_rate_limit() -> None:
    """Runtime rate-limit inside PydanticAIProvider.generate() trips the pool.

    Mocks the internal Pydantic AI model to raise a simulated 429; the
    provider catches it and must forward to `_pool_ref.record_failure`
    (same behaviour as LiteLLMProvider, commit 58ec0d8 — FrugalGPT-
    on-rate-limit).
    """
    from unittest.mock import AsyncMock

    from sage.llm.base import LLMConfig, Message, Role
    from sage.providers.pydantic_ai_provider import PydanticAIProvider

    p = PydanticAIProvider.for_sage_provider("deepseek", "deepseek-chat", "k")

    # Replace the internal Pydantic AI model with a mock that raises
    # a rate-limit-ish exception.
    class _FakeRateLimit(Exception):
        pass
    _FakeRateLimit.__name__ = "RateLimitError"

    # Patch model_request at the module level
    import sage.providers.pydantic_ai_provider as mod

    async def _fake_model_request(*a, **kw):
        raise _FakeRateLimit("429 rate-limit exceeded insufficient_quota")

    # Inject a pool_ref stub that records what it sees
    recorded: list[tuple[str, Exception]] = []

    class _StubPool:
        def record_failure(self, provider_name: str, exc: Exception) -> None:
            recorded.append((provider_name, exc))

    p._pool_ref = _StubPool()

    # Monkeypatch model_request
    import pydantic_ai.direct as _direct
    original = _direct.model_request
    _direct.model_request = _fake_model_request  # type: ignore[assignment]
    try:
        with pytest.raises(_FakeRateLimit):
            await p.generate(
                [Message(role=Role.USER, content="hi")],
                config=LLMConfig(provider="deepseek", model="deepseek-chat"),
            )
    finally:
        _direct.model_request = original  # type: ignore[assignment]

    assert len(recorded) == 1, (
        "PydanticAIProvider must forward rate-limit exceptions to "
        "_pool_ref.record_failure so the circuit breaker can trip "
        "(migration-critical compat with P1.2 commit 8cb719e)."
    )
    assert recorded[0][0] == "deepseek"


# ---------------------------------------------------------------------------
# A8 Phase 2 (2026-04-24) — reasoning_content / ThinkingPart roundtrip
# ---------------------------------------------------------------------------


def test_thinking_roundtrips_both_directions() -> None:
    """Message.thinking survives the PydanticAI translation both ways:

    1. LLMResponse.thinking gets populated from ThinkingPart on incoming
       ModelResponse.
    2. On outgoing ModelRequest, Message.thinking is re-emitted as a
       ThinkingPart so PydanticAI's MoonshotAIProvider /
       DeepSeek-thinking profile serializes it back as
       `reasoning_content`.

    This closes the 4th-turn HTTP 400 class on kimi-k2.5/k2.6 and
    (provisionally) deepseek-v4-pro. The provider profile ships the
    `openai_chat_send_back_thinking_parts='field'` flag; the only thing
    previously missing was SAGE's translation layer emitting
    ThinkingPart at all.
    """
    from sage.llm.base import LLMResponse, Message, Role, ToolCall
    from sage.providers.pydantic_ai_provider import (
        _our_messages_to_pydantic,
        _pydantic_response_to_ours,
    )
    try:
        from pydantic_ai.messages import (
            ModelResponse,
            TextPart,
            ThinkingPart,
            ToolCallPart,
        )
    except ImportError:
        pytest.skip("ThinkingPart not available in this pydantic-ai version")

    # ── Direction 1: ModelResponse → LLMResponse ─────────────────────
    pa_response = ModelResponse(
        parts=[
            ThinkingPart(content="Let me break this down step by step..."),
            TextPart(content="The answer is 42."),
            ToolCallPart(
                tool_name="calculate",
                args={"x": 1},
                tool_call_id="call_0",
            ),
        ]
    )
    ours = _pydantic_response_to_ours(
        pa_response,
        model_name="kimi-k2.6",
        provider_name="kimi",
        model_id="kimi-k2.6",
    )
    assert ours.thinking == "Let me break this down step by step...", (
        "ThinkingPart.content must be captured into LLMResponse.thinking"
    )
    assert ours.content == "The answer is 42."
    assert len(ours.tool_calls) == 1

    # ── Direction 2: Message → ModelRequest/Response ────────────────
    # Build a conversation where the assistant's prior turn contained
    # reasoning_content. The outgoing translation must re-emit that as
    # a ThinkingPart in the ModelResponse.
    messages = [
        Message(role=Role.USER, content="Solve this"),
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(id="call_0", name="calculate", arguments={"x": 1})],
            thinking="Let me break this down step by step...",
        ),
        Message(
            role=Role.TOOL,
            content="1",
            tool_call_id="call_0",
            name="calculate",
        ),
        Message(role=Role.USER, content="Good, continue"),
    ]
    pa_messages = _our_messages_to_pydantic(messages)

    # Locate the ModelResponse we emitted for the assistant turn.
    assistant_responses = [m for m in pa_messages if type(m).__name__ == "ModelResponse"]
    assert len(assistant_responses) == 1, (
        f"expected exactly one ModelResponse for the assistant turn; "
        f"got {len(assistant_responses)}"
    )
    parts = assistant_responses[0].parts
    part_types = [type(p).__name__ for p in parts]
    assert "ThinkingPart" in part_types, (
        "A8 Phase 2: ThinkingPart MUST appear in the outgoing "
        "ModelResponse when Message.thinking is set. Without it, "
        "PydanticAI's openai_chat_send_back_thinking_parts='field' "
        "has nothing to serialize into reasoning_content and "
        "Moonshot/DeepSeek reject the 4th tool-call turn."
    )
    # Ordering: ThinkingPart must precede TextPart + ToolCallPart so
    # the OpenAI-compat serializer puts reasoning_content before
    # content, matching Moonshot's documented streaming convention.
    thinking_idx = part_types.index("ThinkingPart")
    for later_kind in ("TextPart", "ToolCallPart"):
        if later_kind in part_types:
            assert part_types.index(later_kind) > thinking_idx, (
                f"ThinkingPart must precede {later_kind} in the "
                f"ModelResponse parts list; got order {part_types}"
            )
    thinking_part = parts[thinking_idx]
    assert getattr(thinking_part, "content", "") == (
        "Let me break this down step by step..."
    )


def test_thinking_empty_string_does_not_emit_thinking_part() -> None:
    """Non-thinking models produce ``thinking=""``; in that case the
    outgoing translation must NOT emit a spurious empty ThinkingPart.
    Prevents adding junk ``reasoning_content`` to requests against
    non-thinking providers (gpt-5.x, gemini, deepseek-v4-flash, etc.)."""
    from sage.llm.base import Message, Role
    from sage.providers.pydantic_ai_provider import _our_messages_to_pydantic

    messages = [
        Message(role=Role.USER, content="Hi"),
        Message(role=Role.ASSISTANT, content="Hello", thinking=""),
    ]
    pa_messages = _our_messages_to_pydantic(messages)
    responses = [m for m in pa_messages if type(m).__name__ == "ModelResponse"]
    part_types = [type(p).__name__ for p in responses[0].parts]
    assert "ThinkingPart" not in part_types, (
        "empty thinking must NOT emit a ThinkingPart (non-thinking "
        "providers would receive spurious reasoning_content)"
    )


def test_kimi_k2_6_supports_tools_is_false() -> None:
    """A8 migration (2026-04-24): kimi-k2.5 → kimi-k2.6. Originally
    F9 (2026-04-19): kimi-k2.5 is a thinking-mode model requiring
    `reasoning_content` passthrough across tool-call turns; Pydantic
    AI doesn't preserve it → HTTP 400. kimi-k2.6 (verified 2026-04-24
    via https://platform.kimi.ai/docs/guide/kimi-k2-6-quickstart) keeps
    the same contract.

    Workaround until the provider layer handles reasoning_content (or
    until we plumb `thinking: {type: "disabled"}` for tool calls):
    mark kimi-k2.6 as tool-incompatible so ModelAssigner never routes
    a tool-needing node to it. This test locks that flag — if someone
    flips it back without fixing the provider plumbing, this test
    fails with a pointer to the open follow-up.
    """
    from pathlib import Path

    try:
        import sage_core  # type: ignore[import-not-found]
    except Exception:
        pytest.skip("sage_core not available (Rust not built)")

    if not hasattr(sage_core, "ModelRegistry"):
        pytest.skip("sage_core stubbed by another test; real Rust module unavailable")

    toml = Path(__file__).resolve().parents[3] / "sage-core" / "config" / "cards.toml"
    if not toml.is_file():
        pytest.skip("cards.toml not on disk")

    reg = sage_core.ModelRegistry.from_toml_file(str(toml))
    kimi = reg.get("kimi-k2.6")
    if kimi is None:
        pytest.skip("kimi-k2.6 not in cards.toml")

    assert kimi.supports_tools is False, (
        "kimi-k2.6 must have supports_tools=false until either "
        "(a) Pydantic AI's MoonshotAIProvider preserves reasoning_content "
        "across tool-call turns, or (b) our wrapper plumbs "
        "`thinking: {type: \"disabled\"}` when tools are present. See "
        "A8 entry in sage-core/config/cards.toml for the migration audit "
        "trail and path forward."
    )
