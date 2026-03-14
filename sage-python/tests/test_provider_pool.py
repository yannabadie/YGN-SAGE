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
