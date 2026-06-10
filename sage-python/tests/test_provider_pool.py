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

    def test_resolve_runtime_alias_before_cache(self):
        """Compatibility aliases are rewritten to executable catalog ids."""
        legacy = _make_profile("deepseek-chat", "deepseek")
        legacy.runtime_selectable = False
        legacy.runtime_replacement = "deepseek-v4-flash"
        legacy.runtime_replacement_settings = {"thinking": "disabled"}
        current = _make_profile("deepseek-v4-flash", "deepseek")
        current.runtime_settings = {"thinking": "disabled"}
        current.context_window = 1_000_000

        registry = MagicMock()
        registry.get.side_effect = lambda mid: {
            "deepseek-chat": legacy,
            "deepseek-v4-flash": current,
        }.get(mid)
        deepseek_provider = _make_provider("deepseek")
        pool = ProviderPool(
            default_provider=_make_provider("default"),
            registry=registry,
            providers={"deepseek": deepseek_provider},
        )

        provider, config = pool.resolve("deepseek-chat")

        assert provider is deepseek_provider
        assert config.provider == "deepseek"
        assert config.model == "deepseek-v4-flash"
        assert config.extra["thinking"] == "disabled"
        assert config.extra["alias_from"] == "deepseek-chat"
        assert "deepseek-chat" in pool._cache
        assert "deepseek-v4-flash" not in pool._cache

    def test_resolve_active_card_applies_runtime_settings(self):
        """Active V4 cards can declare mode settings without using aliases."""
        profile = _make_profile("deepseek-v4-flash", "deepseek")
        profile.runtime_settings = {"thinking": "disabled"}
        registry = _make_registry(profile)
        deepseek_provider = _make_provider("deepseek")
        pool = ProviderPool(
            default_provider=_make_provider("default"),
            registry=registry,
            providers={"deepseek": deepseek_provider},
        )

        provider, config = pool.resolve("deepseek-v4-flash")

        assert provider is deepseek_provider
        assert config.model == "deepseek-v4-flash"
        assert config.extra["thinking"] == "disabled"

    def test_resolve_minimax_lowercase_alias(self):
        """Old lowercase MiniMax configs keep working through a non-selectable alias."""
        legacy = _make_profile("minimax-m2.7", "minimax")
        legacy.runtime_selectable = False
        legacy.runtime_replacement = "MiniMax-M2.7"
        legacy.runtime_replacement_settings = {}
        current = _make_profile("MiniMax-M2.7", "minimax")
        current.runtime_settings = {}

        registry = MagicMock()
        registry.get.side_effect = lambda mid: {
            "minimax-m2.7": legacy,
            "MiniMax-M2.7": current,
        }.get(mid)
        minimax_provider = _make_provider("minimax")
        pool = ProviderPool(
            default_provider=_make_provider("default"),
            registry=registry,
            providers={"minimax": minimax_provider},
        )

        provider, config = pool.resolve("minimax-m2.7")

        assert provider is minimax_provider
        assert config.provider == "minimax"
        assert config.model == "MiniMax-M2.7"
        assert config.extra["alias_from"] == "minimax-m2.7"


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


# --- Re-probe / TTL-based exclusion (2026-04-18) ---


class TestExclusionTTLAndReprobe:
    """Exclusion must be time-bounded and re-verified, not permanent."""

    @pytest.mark.asyncio
    async def test_dead_provider_recovers_after_reprobe(self):
        """After TTL, a provider that now responds should be removed from the dead list."""
        from unittest.mock import AsyncMock
        p = _make_provider("flaky")
        # First call fails (connection), second call succeeds.
        p.generate = AsyncMock(
            side_effect=[Exception("connection refused"), MagicMock(content="ok")]
        )
        pool = ProviderPool(
            default_provider=p, registry=_make_registry(),
            default_config=LLMConfig(provider="flaky", model="x"),
            providers={"flaky": p},
        )
        # First probe → dead
        r1 = await pool.health_check(timeout=1.0)
        assert r1 == {"flaky": False}
        assert "flaky" in pool._dead_at

        # Force TTL expiry so reprobe actually runs.
        pool._dead_at["flaky"] = 0.0  # very old
        r2 = await pool.reprobe_excluded_providers(timeout=1.0, ttl_sec=60.0)
        assert r2 == {"flaky": True}
        assert "flaky" not in pool._dead_at

    @pytest.mark.asyncio
    async def test_dead_provider_within_ttl_not_reprobed(self):
        """A recently-marked-dead provider should NOT be reprobed within TTL."""
        from unittest.mock import AsyncMock
        p = _make_provider("recent")
        p.generate = AsyncMock(side_effect=Exception("SSL handshake failed"))
        pool = ProviderPool(
            default_provider=p, registry=_make_registry(),
            default_config=LLMConfig(provider="recent", model="x"),
            providers={"recent": p},
        )
        await pool.health_check(timeout=1.0)
        # Dead timestamp just written; TTL=300s → reprobe should be a no-op.
        p.generate.reset_mock()
        r = await pool.reprobe_excluded_providers(timeout=1.0, ttl_sec=300.0)
        assert r == {}  # nothing reprobed
        assert p.generate.call_count == 0
        assert "recent" in pool._dead_at  # still dead

    @pytest.mark.asyncio
    async def test_still_dead_after_reprobe_resets_ttl(self):
        """A provider that stays dead on reprobe must keep the exclusion
        (but with a fresh TTL so we don't re-probe every call)."""
        from unittest.mock import AsyncMock
        import time as _time
        p = _make_provider("down")
        p.generate = AsyncMock(side_effect=Exception("DNS resolution failed"))
        pool = ProviderPool(
            default_provider=p, registry=_make_registry(),
            default_config=LLMConfig(provider="down", model="x"),
            providers={"down": p},
        )
        await pool.health_check(timeout=1.0)
        pool._dead_at["down"] = 0.0  # force expiry
        before = _time.time()
        r = await pool.reprobe_excluded_providers(timeout=1.0, ttl_sec=60.0)
        after = _time.time()
        assert r == {"down": False}
        # Timestamp refreshed
        assert before <= pool._dead_at["down"] <= after + 1

    @pytest.mark.asyncio
    async def test_refresh_exclusion_list_syncs_assigner(self):
        """refresh_exclusion_list must push the current dead list to the
        Rust ModelAssigner via exclude_providers()."""
        from unittest.mock import AsyncMock
        p = _make_provider("test")
        p.generate = AsyncMock(side_effect=Exception("SSL error"))
        pool = ProviderPool(
            default_provider=p, registry=_make_registry(),
            default_config=LLMConfig(provider="test", model="x"),
            providers={"test": p},
        )
        await pool.health_check(timeout=1.0)
        pool._dead_at["test"] = 0.0  # force TTL expiry

        # Simulate Rust assigner
        class _FakeAssigner:
            def __init__(self):
                self.last_exclusion: list[str] | None = None
            def exclude_providers(self, providers):
                self.last_exclusion = list(providers)
        assigner = _FakeAssigner()

        # Provider is still dead (same side_effect); refresh should call exclude_providers with it
        dead = await pool.refresh_exclusion_list(model_assigner=assigner, ttl_sec=60.0, timeout=1.0)
        assert dead == ["test"]
        assert assigner.last_exclusion == ["test"]

    def test_get_dead_providers_respects_ttl(self):
        import time as _time
        pool = ProviderPool(
            default_provider=_make_provider("a"), registry=_make_registry(),
            default_config=LLMConfig(provider="a", model="x"),
            providers={"a": _make_provider("a")},
        )
        now = _time.time()
        pool._dead_at["fresh"] = now                # within TTL
        pool._dead_at["ancient"] = now - 10_000.0   # way past TTL
        current = pool.get_dead_providers(ttl_sec=300.0)
        assert current == ["fresh"]  # expired entry not reported


# ---------------------------------------------------------------------------
# B2_RERUN_UNBLOCKERS bug 1 — provider attribution (2026-05-12 canary task #3)
# ---------------------------------------------------------------------------

class TestInferProvider:
    def test_registry_unknown_sentinel_falls_through_to_prefix(self):
        """B2 bug 1: model_profiles.toml entries without a `provider` key yield
        ModelProfile(provider="unknown") via registry._profile_from_toml. The
        truthy "unknown" sentinel used to win over the correct prefix fallback,
        so node_started carried provider_id="unknown" and the canary
        provider_gate went NO_GO (execution_outside_allowlist=["unknown"])."""
        profile = _make_profile("gemini-3.1-pro-preview", "unknown")
        registry = _make_registry(profile)
        pool = ProviderPool(
            default_provider=_make_provider(),
            registry=registry,
            providers={},
        )
        assert pool.infer_provider("gemini-3.1-pro-preview") == "google"

    def test_registry_empty_provider_falls_through_to_prefix(self):
        profile = _make_profile("deepseek-v4-flash", "")
        registry = _make_registry(profile)
        pool = ProviderPool(
            default_provider=_make_provider(),
            registry=registry,
            providers={},
        )
        assert pool.infer_provider("deepseek-v4-flash") == "deepseek"

    def test_registry_real_provider_still_wins_over_prefix(self):
        """A genuine registry provider must keep priority over the string
        fallback (the guard only rejects the "unknown" sentinel)."""
        profile = _make_profile("qwen/qwen3.5-plus-02-15", "openrouter")
        registry = _make_registry(profile)
        pool = ProviderPool(
            default_provider=_make_provider(),
            registry=registry,
            providers={},
        )
        assert pool.infer_provider("qwen/qwen3.5-plus-02-15") == "openrouter"


class TestModelProfilesTomlProviderTripwire:
    def test_every_curated_entry_declares_provider_matching_cards(self):
        """Tripwire for B2 bug 1: every entry of the curated knowledge base
        (sage-python/config/model_profiles.toml) MUST declare an explicit
        `provider`, and when the id also exists in cards.toml (the provider
        source of truth, directive #7) the two MUST agree. Without this, any
        model that is not live-discovered at boot silently becomes
        provider="unknown" at runtime."""
        import tomllib
        from pathlib import Path

        here = Path(__file__).resolve()
        sage_python_root = here.parents[1]
        profiles_path = sage_python_root / "config" / "model_profiles.toml"
        cards_path = sage_python_root.parent / "sage-core" / "config" / "cards.toml"
        assert profiles_path.exists(), profiles_path

        with open(profiles_path, "rb") as f:
            profiles = tomllib.load(f).get("models", {})
        assert profiles, "model_profiles.toml has no [models] entries"

        cards_by_id: dict[str, str] = {}
        if cards_path.exists():
            with open(cards_path, "rb") as f:
                cards_data = tomllib.load(f)
            for value in cards_data.values():
                if isinstance(value, list):
                    for card in value:
                        if isinstance(card, dict) and card.get("id"):
                            cards_by_id[str(card["id"])] = str(card.get("provider", ""))

        missing = [mid for mid, entry in profiles.items() if not entry.get("provider")]
        assert not missing, f"model_profiles.toml entries missing provider: {missing}"

        mismatched = {
            mid: (entry["provider"], cards_by_id[mid])
            for mid, entry in profiles.items()
            if mid in cards_by_id and cards_by_id[mid] and entry["provider"] != cards_by_id[mid]
        }
        assert not mismatched, f"provider disagrees with cards.toml: {mismatched}"
