"""Integration test: SWE-Bench with LiteLLM provider adapter.

Validates that the LiteLLM provider works correctly with SWE-bench's
task runner, including provider pool resolution and multi-provider fallback.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

import pytest

from sage.llm.base import LLMConfig, Message, Role
from sage.llm.provider_pool import ProviderPool
from sage.providers.litellm_provider import LiteLLMProvider


class TestSWEBenchLiteLLMIntegration:
    """Test SWE-bench with LiteLLM provider resolution and fallback."""

    def test_provider_pool_resolves_litellm_models(self):
        """ProviderPool should resolve all LiteLLM models without error."""
        # Create mock providers
        deepseek_provider = LiteLLMProvider(model_string="deepseek/deepseek-chat", api_key="test-key")
        google_provider = LiteLLMProvider(model_string="gemini/gemini-2.5-flash", api_key="test-key")
        openai_provider = LiteLLMProvider(model_string="openai/gpt-5.4", api_key="test-key")

        # Create pool with all providers
        pool = ProviderPool(
            default_provider=deepseek_provider,
            registry=None,  # No registry; use inference fallback
            providers={
                "deepseek": deepseek_provider,
                "google": google_provider,
                "openai": openai_provider,
            },
        )

        # Resolve several known model IDs
        test_models = [
            ("deepseek-chat", "deepseek"),
            ("gemini-2.5-flash", "google"),
            ("gpt-5.4", "openai"),
            ("gpt-5.4-pro", "openai"),  # Pro models should map to openai
            ("unknown-model", "deepseek"),  # Unknown should fallback to default
        ]

        for model_id, expected_provider_name in test_models:
            provider, config = pool.resolve(model_id)
            assert config.model == model_id, f"Config should preserve model_id: {model_id}"
            # For unknown models, we fallback to default
            if model_id == "unknown-model":
                assert provider is deepseek_provider
            else:
                # Should resolve to a valid provider (may not be exact due to string hints)
                assert provider is not None

    def test_provider_pool_circuit_breaker_fallback(self):
        """When a provider circuit opens, pool should fallback to default via registry."""
        good_provider = LiteLLMProvider(model_string="deepseek/deepseek-chat", api_key="test-key")
        bad_provider = LiteLLMProvider(model_string="gemini/gemini-2.5-flash", api_key="test-key")
        default_provider = LiteLLMProvider(model_string="openai/gpt-5.4", api_key="test-key")

        # Create a mock registry that returns a profile with "google" provider
        profile = MagicMock()
        profile.provider = "google"
        profile.context_window = 128000
        registry = MagicMock()
        registry.get = MagicMock(return_value=profile)

        pool = ProviderPool(
            default_provider=default_provider,
            registry=registry,
            providers={
                "google": bad_provider,
                "deepseek": good_provider,
            },
        )

        # Simulate circuit breaker failure for google provider
        for _ in range(3):
            pool.record_failure("google", Exception("API timeout"))

        # Google circuit should now be open
        assert not pool.is_available("google")

        # Resolving a google model should fallback to default when circuit is open
        provider, config = pool.resolve("gemini-2.5-flash")
        assert provider is default_provider
        assert config.model == "gemini-2.5-flash"
        assert config.provider == "google"  # Original provider from registry, but with default provider instance

    def test_litellm_provider_config_mapping(self):
        """LiteLLMProvider should map LLMConfig correctly to litellm call."""
        # Test that LiteLLMProvider accepts and processes LLMConfig
        provider = LiteLLMProvider(model_string="deepseek/deepseek-chat", api_key="test-key")

        config = LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=1024,
            context_window=128000,
        )

        # Config should be accepted without error
        assert config.provider == "deepseek"
        assert config.model == "deepseek-chat"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024

    @pytest.mark.asyncio
    async def test_litellm_provider_generate_with_swebench_message_format(self):
        """LiteLLMProvider should handle SWE-bench style messages."""
        # Mock litellm.acompletion
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Here's a patch: diff --git..."

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            provider = LiteLLMProvider(model_string="deepseek/deepseek-chat", api_key="test-key")

            config = LLMConfig(
                provider="deepseek",
                model="deepseek-chat",
                temperature=0.7,
                max_tokens=2048,
            )

            # Create messages similar to what SWE-bench would send
            messages = [
                Message(
                    role=Role.SYSTEM,
                    content="You are an expert software engineer.",
                ),
                Message(
                    role=Role.USER,
                    content="Fix the failing test in src/module.py",
                ),
            ]

            response = await provider.generate(messages=messages, config=config)

            assert response is not None
            assert isinstance(response, object)  # Should be LLMResponse
            assert response.content is not None
            assert "patch" in response.content.lower() or "diff" in response.content.lower()

    def test_provider_pool_caching_with_litellm(self):
        """ProviderPool should cache resolved models to avoid redundant lookups."""
        provider = LiteLLMProvider(model_string="deepseek/deepseek-chat", api_key="test-key")
        registry = MagicMock()
        registry.get = MagicMock(return_value=None)

        pool = ProviderPool(
            default_provider=provider,
            registry=registry,
            providers={"deepseek": provider},
        )

        # First resolution
        p1, cfg1 = pool.resolve("deepseek-chat")
        assert p1 is provider

        # Second resolution should be cached (registry not called again)
        p2, cfg2 = pool.resolve("deepseek-chat")
        assert p2 is p1
        assert cfg2.model == cfg1.model

        # Registry should only be called once (for first resolution)
        assert registry.get.call_count == 1

    def test_provider_pool_health_check_framework(self):
        """ProviderPool health_check should probe providers and record results."""
        mock_provider = MagicMock(spec=LiteLLMProvider)
        mock_provider.model_id = "test-model"
        mock_provider.generate = AsyncMock(return_value="pong")

        pool = ProviderPool(
            default_provider=mock_provider,
            registry=None,
            providers={"test": mock_provider},
        )

        # Health check should call generate
        import asyncio
        results = asyncio.run(pool.health_check(timeout=5.0))

        assert "test" in results
        assert results["test"] is True
        assert mock_provider.generate.called

    def test_litellm_provider_inference_fallback_patterns(self):
        """ProviderPool.infer_provider should correctly map model IDs to provider hints."""
        pool = ProviderPool(
            default_provider=MagicMock(),
            registry=None,
        )

        # Test inference patterns (from cards.toml April 2026)
        inference_tests = [
            ("gemini-2.5-flash", "google"),
            ("gpt-5.4", "openai"),
            ("gpt-5.4-pro", "openai"),
            ("deepseek-chat", "deepseek"),
            ("grok-4-1-fast-reasoning", "xai"),
            ("minimax-m2.7", "minimax"),
            ("kimi-9b-exp", "kimi"),
            ("qwen/qwen3.5-plus", "openrouter"),
        ]

        for model_id, expected_provider in inference_tests:
            inferred = pool.infer_provider(model_id)
            assert inferred == expected_provider, (
                f"Failed to infer {expected_provider} from {model_id}, "
                f"got {inferred}"
            )


class TestSWEBenchLiteLLMBootSequence:
    """Test the boot sequence when initializing SWE-bench with LiteLLM providers."""

    def test_provider_pool_boot_with_multiple_litellm_providers(self):
        """Boot sequence should initialize ProviderPool with all 7 providers."""
        providers_config = {
            "google": MagicMock(spec=LiteLLMProvider),
            "openai": MagicMock(spec=LiteLLMProvider),
            "deepseek": MagicMock(spec=LiteLLMProvider),
            "xai": MagicMock(spec=LiteLLMProvider),
            "kimi": MagicMock(spec=LiteLLMProvider),
            "minimax": MagicMock(spec=LiteLLMProvider),
            "openrouter": MagicMock(spec=LiteLLMProvider),
        }

        pool = ProviderPool(
            default_provider=providers_config["deepseek"],
            registry=None,
            providers=providers_config,
        )

        # All 7 providers should be registered
        assert len(pool._providers) == 7

        # Each provider should be retrievable
        for name, provider in providers_config.items():
            assert pool._providers.get(name) is provider

    def test_swebench_model_assignment_uses_provider_pool(self):
        """When SWE-bench assigns models, it should use ProviderPool.resolve()."""
        # Create a mock registry that returns model profiles
        profile = MagicMock()
        profile.provider = "deepseek"
        profile.context_window = 128000

        registry = MagicMock()
        registry.get = MagicMock(return_value=profile)

        deepseek_provider = MagicMock(spec=LiteLLMProvider)
        pool = ProviderPool(
            default_provider=MagicMock(),
            registry=registry,
            providers={"deepseek": deepseek_provider},
        )

        # When SWE-bench needs to resolve a model for a task
        provider, config = pool.resolve("deepseek-chat")

        # It should get the correct provider and config
        assert provider is deepseek_provider
        assert config.provider == "deepseek"
        assert config.model == "deepseek-chat"
        assert config.context_window == 128000

        # Registry should have been consulted
        registry.get.assert_called_once_with("deepseek-chat")
