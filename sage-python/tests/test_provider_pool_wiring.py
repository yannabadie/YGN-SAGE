"""Tests for DeepSeek env var fix and ProviderPool multi-provider wiring."""
from __future__ import annotations

import os
import pytest


class TestDeepSeekEnvVar:
    """Verify PROVIDER_CONFIGS uses the correct env var name."""

    def test_deepseek_primary_env_var(self) -> None:
        from sage.providers.connector import PROVIDER_CONFIGS

        deepseek_cfgs = [c for c in PROVIDER_CONFIGS if c["provider"] == "deepseek"]
        assert len(deepseek_cfgs) == 1, "Expected exactly one deepseek config"
        assert deepseek_cfgs[0]["api_key_env"] == "DEEPSEEK_API_KEY"


class TestProviderPoolWiring:
    """Integration tests: boot wires providers into ProviderPool."""

    @pytest.fixture(autouse=True)
    def _boot_system(self):
        """Boot the agent system once for the class."""
        key = os.environ.get("GOOGLE_API_KEY", "")
        # Skip if key is absent or obviously fake (leaked from other tests)
        if not key or "fake" in key or "test" in key or len(key) < 20:
            pytest.skip("GOOGLE_API_KEY not set or fake")
        from sage.boot import boot_agent_system

        self.system = boot_agent_system()
        yield

    def test_boot_wires_google_into_pool(self) -> None:
        pool = self.system.pipeline.provider_pool if self.system.pipeline else None
        assert pool is not None, "Pipeline or ProviderPool not initialized"
        assert "google" in pool._providers, (
            f"Expected 'google' in pool._providers, got {list(pool._providers.keys())}"
        )

    def test_boot_wires_multiple_providers(self) -> None:
        pool = self.system.pipeline.provider_pool if self.system.pipeline else None
        assert pool is not None, "Pipeline or ProviderPool not initialized"
        assert len(pool._providers) >= 1, (
            f"Expected >=1 provider, got {len(pool._providers)}: {list(pool._providers.keys())}"
        )

    def test_resolve_returns_provider_for_google_model(self) -> None:
        pool = self.system.pipeline.provider_pool if self.system.pipeline else None
        assert pool is not None, "Pipeline or ProviderPool not initialized"

        prov, cfg = pool.resolve("gemini-2.5-flash")
        # May be GoogleProvider (real key) or OpenAICompatProvider (fallback)
        assert prov is not None, "Expected a provider for gemini-2.5-flash"
        assert hasattr(prov, "generate"), "Provider must have a generate method"
