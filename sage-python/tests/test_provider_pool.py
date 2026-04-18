"""Tests for ProviderPool — model_id → (LLMProvider, LLMConfig) resolution."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sage.llm.base import LLMConfig
from sage.llm.provider_pool import ProviderPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(model_id: str, provider_name: str) -> MagicMock:
    profile = MagicMock()
    profile.id = model_id
    profile.provider = provider_name
    return profile


def _make_registry(profile=None) -> MagicMock:
    """Registry that returns *profile* for any get() call."""
    registry = MagicMock()
    registry.get.return_value = profile
    return registry


def _make_provider(name: str = "mock") -> MagicMock:
    provider = MagicMock()
    provider.name = name
    return provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProviderPool:
    def test_resolve_known_model(self):
        """Model found in registry with a matching live provider → profile-based config."""
        profile = _make_profile("gemini-2.5-flash", "google")
        registry = _make_registry(profile)

        google_provider = _make_provider("google")
        default_provider = _make_provider("default")

        pool = ProviderPool(
            default_provider=default_provider,
            registry=registry,
            providers={"google": google_provider},
        )

        provider, config = pool.resolve("gemini-2.5-flash")

        assert provider is google_provider
        assert config.model == "gemini-2.5-flash"
        assert config.provider == "google"
        registry.get.assert_called_once_with("gemini-2.5-flash")

    def test_resolve_unknown_falls_back(self):
        """Model not in registry → default provider returned, model_id preserved in config."""
        registry = _make_registry(profile=None)  # get() returns None
        default_provider = _make_provider("default")

        pool = ProviderPool(
            default_provider=default_provider,
            registry=registry,
        )

        provider, config = pool.resolve("unknown-model-x")

        assert provider is default_provider
        assert config.model == "unknown-model-x"
        assert config.provider == "default"

    def test_resolve_caches(self):
        """Second call for the same model_id must not hit the registry again."""
        profile = _make_profile("gemini-2.5-flash", "google")
        registry = _make_registry(profile)

        google_provider = _make_provider("google")
        default_provider = _make_provider("default")

        pool = ProviderPool(
            default_provider=default_provider,
            registry=registry,
            providers={"google": google_provider},
        )

        result_first = pool.resolve("gemini-2.5-flash")
        result_second = pool.resolve("gemini-2.5-flash")

        # Same objects returned
        assert result_first[0] is result_second[0]
        assert result_first[1].model == result_second[1].model

        # Registry was only queried once
        registry.get.assert_called_once_with("gemini-2.5-flash")

    def test_resolve_empty_model_id_returns_default(self):
        """Empty string model_id → default provider + default config, registry not called."""
        registry = _make_registry()
        default_provider = _make_provider("default")
        default_config = LLMConfig(provider="my-default", model="my-model")

        pool = ProviderPool(
            default_provider=default_provider,
            registry=registry,
            default_config=default_config,
        )

        provider, config = pool.resolve("")

        assert provider is default_provider
        assert config is default_config
        registry.get.assert_not_called()

    def test_openai_pro_model_is_available_with_litellm(self):
        """GPT-5 Pro models are now available — LiteLLM handles Responses API natively."""
        profile = _make_profile("gpt-5.4-pro", "openai")
        registry = _make_registry(profile)
        openai_provider = _make_provider("openai")
        default_provider = _make_provider("default")

        pool = ProviderPool(
            default_provider=default_provider,
            registry=registry,
            providers={"openai": openai_provider},
        )

        assert pool.is_model_available("gpt-5.4-pro") is True

    def test_resolve_openai_pro_model_uses_openai_provider(self):
        """GPT-5 Pro models are now resolved to their OpenAI provider — LiteLLM handles the Responses API."""
        profile = _make_profile("gpt-5.4-pro", "openai")
        registry = _make_registry(profile)
        openai_provider = _make_provider("openai")
        default_provider = _make_provider("default")
        default_config = LLMConfig(provider="google", model="gemini-2.5-flash")

        pool = ProviderPool(
            default_provider=default_provider,
            registry=registry,
            default_config=default_config,
            providers={"openai": openai_provider},
        )

        provider, config = pool.resolve("gpt-5.4-pro")

        assert provider is openai_provider
        assert config.provider == "openai"
        assert config.model == "gpt-5.4-pro"


# --- Quota-aware health_check tests (Codex item F, 2026-04-18) ---


class TestHealthCheckQuotaAwareness:
    """health_check() must mark OpenAI-style quota exhaustion as DEAD,
    not ALIVE, so ModelAssigner routes elsewhere instead of 429-looping."""

    @pytest.mark.asyncio
    async def test_insufficient_quota_marks_dead(self):
        from unittest.mock import AsyncMock
        default_provider = _make_provider("deepseek")
        dead_provider = _make_provider("openai")
        dead_provider.generate = AsyncMock(
            side_effect=Exception(
                "RateLimitError Error code: 429 - insufficient_quota. "
                "You exceeded your current quota. Check your billing."
            )
        )

        pool = ProviderPool(
            default_provider=default_provider,
            registry=_make_registry(),
            default_config=LLMConfig(provider="deepseek", model="x"),
            providers={"openai": dead_provider, "deepseek": default_provider},
        )
        # Force deepseek probe to succeed
        default_provider.generate = AsyncMock(return_value=MagicMock(content="hi"))

        results = await pool.health_check(timeout=1.0)
        assert results == {"openai": False, "deepseek": True}

    @pytest.mark.asyncio
    async def test_transient_rate_limit_stays_alive(self):
        """A plain 429 without quota wording is probe noise, not a dead provider."""
        from unittest.mock import AsyncMock
        p = _make_provider("gemini")
        p.generate = AsyncMock(
            side_effect=Exception("429 Too Many Requests — please retry later")
        )
        pool = ProviderPool(
            default_provider=p, registry=_make_registry(),
            default_config=LLMConfig(provider="gemini", model="x"),
            providers={"gemini": p},
        )
        results = await pool.health_check(timeout=1.0)
        assert results == {"gemini": True}

    @pytest.mark.asyncio
    async def test_connection_error_still_marks_dead(self):
        """Regression guard: connection errors (old behavior) still mark dead."""
        from unittest.mock import AsyncMock
        p = _make_provider("openrouter")
        p.generate = AsyncMock(side_effect=Exception("DNS resolution failed"))
        pool = ProviderPool(
            default_provider=p, registry=_make_registry(),
            default_config=LLMConfig(provider="openrouter", model="x"),
            providers={"openrouter": p},
        )
        results = await pool.health_check(timeout=1.0)
        assert results == {"openrouter": False}

    @pytest.mark.asyncio
    async def test_401_stays_alive(self):
        """Regression: 401 auth errors remain ALIVE (reachable, probe misconfig)."""
        from unittest.mock import AsyncMock
        p = _make_provider("xai")
        p.generate = AsyncMock(side_effect=Exception("401 Unauthorized"))
        pool = ProviderPool(
            default_provider=p, registry=_make_registry(),
            default_config=LLMConfig(provider="xai", model="x"),
            providers={"xai": p},
        )
        results = await pool.health_check(timeout=1.0)
        assert results == {"xai": True}
