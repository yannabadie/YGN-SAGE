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
