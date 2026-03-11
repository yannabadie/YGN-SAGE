"""Tests for the ProviderConnector + ModelRegistry system.

All tests are offline -- no real API calls are made.
"""
import sys
import types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sage.providers.connector import ProviderConnector, DiscoveredModel, PROVIDER_CONFIGS
from sage.providers.registry import ModelRegistry, ModelProfile

try:
    import openai  # noqa: F401
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

_skip_no_openai = pytest.mark.skipif(
    not HAS_OPENAI, reason="openai not installed"
)


# ── ModelProfile tests ────────────────────────────────────────────────────────

class TestModelProfile:
    def test_default_scores(self):
        profile = ModelProfile(id="test-model", provider="google", family="test")
        assert profile.code_score == 0.5
        assert profile.reasoning_score == 0.5
        assert profile.tool_use_score == 0.5

    def test_custom_scores(self):
        profile = ModelProfile(
            id="test-model",
            provider="openai",
            family="gpt-5",
            code_score=0.9,
            reasoning_score=0.95,
            cost_input=2.0,
            cost_output=16.0,
        )
        assert profile.code_score == 0.9
        assert profile.reasoning_score == 0.95
        assert profile.cost_input == 2.0

    def test_default_availability(self):
        profile = ModelProfile(id="x", provider="y")
        assert profile.available is False

    def test_default_economics(self):
        profile = ModelProfile(id="x", provider="y")
        assert profile.cost_input == 0.0
        assert profile.cost_output == 0.0


# ── TOML loading tests ───────────────────────────────────────────────────────

class TestTomlLoading:
    def test_load_toml_from_package(self):
        """The bundled config/model_profiles.toml must be found and parsed."""
        registry = ModelRegistry()
        knowledge = registry._load_toml()
        assert "gemini-3.1-pro-preview" in knowledge
        assert "gpt-5.4" in knowledge
        assert "deepseek-chat" in knowledge

    def test_toml_entry_has_expected_fields(self):
        registry = ModelRegistry()
        knowledge = registry._load_toml()
        gemini = knowledge["gemini-3.1-pro-preview"]
        assert gemini["provider"] == "google"
        assert gemini["family"] == "gemini-3.1"
        assert 0.0 <= gemini["code_score"] <= 1.0
        assert gemini["cost_input"] > 0

    def test_load_toml_returns_empty_on_missing(self, tmp_path, monkeypatch):
        """If no TOML file is found, return empty dict."""
        # Point search paths to a temp dir with no TOML
        monkeypatch.setattr(
            "sage.providers.registry._toml_search_paths",
            lambda: [tmp_path / "nonexistent.toml"],
        )
        registry = ModelRegistry()
        knowledge = registry._load_toml()
        assert knowledge == {}

    def test_load_toml_custom_file(self, tmp_path, monkeypatch):
        """Custom TOML file is loaded correctly."""
        toml_content = b"""
[models."custom-model"]
provider = "test"
family = "custom"
code_score = 0.99
cost_input = 0.01
cost_output = 0.02
"""
        toml_file = tmp_path / "model_profiles.toml"
        toml_file.write_bytes(toml_content)

        monkeypatch.setattr(
            "sage.providers.registry._toml_search_paths",
            lambda: [toml_file],
        )
        registry = ModelRegistry()
        knowledge = registry._load_toml()
        assert "custom-model" in knowledge
        assert knowledge["custom-model"]["code_score"] == 0.99


# ── ProviderConnector tests ──────────────────────────────────────────────────

class TestProviderConnector:
    @pytest.mark.asyncio
    async def test_discover_skips_missing_api_key(self, monkeypatch):
        """Providers without API keys are silently skipped."""
        for cfg in PROVIDER_CONFIGS:
            monkeypatch.delenv(cfg["api_key_env"], raising=False)
        monkeypatch.setattr("shutil.which", lambda _: None)

        connector = ProviderConnector()
        models = await connector.discover_all()
        assert models == []

    @pytest.mark.asyncio
    async def test_discover_hardcoded_models(self, monkeypatch):
        """MiniMax-style hardcoded models are returned when API key is set."""
        monkeypatch.setenv("TEST_KEY", "fake-key-123")
        connector = ProviderConnector(configs=[{
            "provider": "test-provider",
            "api_key_env": "TEST_KEY",
            "base_url": "https://example.com",
            "sdk": "openai",
            "hardcoded_models": ["model-a", "model-b"],
        }])
        # Also mock codex CLI away
        monkeypatch.setattr("shutil.which", lambda _: None)

        models = await connector.discover_all()
        assert len(models) == 2
        assert models[0].id == "model-a"
        assert models[0].provider == "test-provider"
        assert models[1].id == "model-b"

    @pytest.mark.asyncio
    async def test_discover_codex_cli(self, monkeypatch):
        """Codex CLI is detected when shutil.which finds it."""
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/codex" if name == "codex" else None)
        # No API keys set
        for cfg in PROVIDER_CONFIGS:
            monkeypatch.delenv(cfg["api_key_env"], raising=False)

        connector = ProviderConnector()
        models = await connector.discover_all()
        assert len(models) == 1
        assert models[0].id == "codex-cli"
        assert models[0].provider == "codex"

    @_skip_no_openai
    @pytest.mark.asyncio
    async def test_discover_openai_compat(self, monkeypatch):
        """OpenAI-compatible discovery returns model IDs."""
        monkeypatch.setenv("TEST_KEY", "fake-key")
        monkeypatch.setattr("shutil.which", lambda _: None)

        mock_model_1 = MagicMock()
        mock_model_1.id = "gpt-5.4"
        mock_model_2 = MagicMock()
        mock_model_2.id = "gpt-5-mini"

        mock_client = MagicMock()
        mock_client.models.list.return_value = [mock_model_1, mock_model_2]

        with patch("openai.OpenAI", return_value=mock_client):
            connector = ProviderConnector(configs=[{
                "provider": "openai",
                "api_key_env": "TEST_KEY",
                "base_url": "https://api.openai.com/v1",
                "sdk": "openai",
            }])
            models = await connector.discover_all()

        assert len(models) == 2
        assert models[0].id == "gpt-5.4"
        assert models[1].id == "gpt-5-mini"
        assert all(m.provider == "openai" for m in models)

    @pytest.mark.asyncio
    async def test_discover_google(self, monkeypatch):
        """Google discovery strips 'models/' prefix and extracts metadata."""
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        monkeypatch.setattr("shutil.which", lambda _: None)

        mock_model = MagicMock()
        mock_model.name = "models/gemini-3.1-pro-preview"
        mock_model.input_token_limit = 1_000_000
        mock_model.output_token_limit = 65_536
        mock_model.supported_actions = None

        mock_client = MagicMock()
        mock_client.models.list.return_value = [mock_model]

        with patch("google.genai.Client", return_value=mock_client):
            connector = ProviderConnector(configs=[{
                "provider": "google",
                "api_key_env": "GOOGLE_API_KEY",
                "base_url": None,
                "sdk": "google-genai",
            }])
            models = await connector.discover_all()

        assert len(models) == 1
        assert models[0].id == "gemini-3.1-pro-preview"
        assert models[0].context_window == 1_000_000
        assert models[0].max_output_tokens == 65_536

    @_skip_no_openai
    @pytest.mark.asyncio
    async def test_provider_failure_doesnt_crash(self, monkeypatch):
        """A failing provider is skipped without crashing."""
        monkeypatch.setenv("BAD_KEY", "fake")
        monkeypatch.setattr("shutil.which", lambda _: None)

        connector = ProviderConnector(configs=[{
            "provider": "bad-provider",
            "api_key_env": "BAD_KEY",
            "base_url": "https://broken.example.com",
            "sdk": "openai",
        }])
        # openai.OpenAI will raise because the URL is unreachable,
        # but we mock it to raise explicitly
        with patch("openai.OpenAI", side_effect=Exception("Connection refused")):
            models = await connector.discover_all()
        # Should return empty (only codex attempt, which also returns empty)
        assert models == []


# ── ModelRegistry tests ──────────────────────────────────────────────────────

class TestModelRegistry:
    def test_select_for_code_task(self):
        registry = ModelRegistry()
        registry._profiles["model-a"] = ModelProfile(
            id="model-a", provider="test", family="test",
            available=True, code_score=0.9,
            cost_input=10.0, cost_output=50.0,
        )
        registry._profiles["model-b"] = ModelProfile(
            id="model-b", provider="test", family="test",
            available=True, code_score=0.8,
            cost_input=0.30, cost_output=1.20,
        )

        result = registry.select({"code": 1.0, "max_cost_per_1m": 5.0})
        assert result is not None
        assert result.id == "model-b"  # model-a filtered out by cost

    def test_select_best_quality_per_cost(self):
        registry = ModelRegistry()
        registry._profiles["expensive"] = ModelProfile(
            id="expensive", provider="test", family="test",
            available=True, code_score=0.9,
            cost_input=10.0, cost_output=50.0,
        )
        registry._profiles["cheap"] = ModelProfile(
            id="cheap", provider="test", family="test",
            available=True, code_score=0.7,
            cost_input=0.1, cost_output=0.5,
        )

        # No cost filter -- cheap should win on quality/cost ratio
        result = registry.select({"code": 1.0})
        assert result is not None
        assert result.id == "cheap"  # 0.7/0.1 = 7.0 vs 0.9/10.0 = 0.09

    def test_select_ignores_unavailable(self):
        registry = ModelRegistry()
        registry._profiles["gone"] = ModelProfile(
            id="gone", provider="test", family="test",
            available=False, code_score=1.0,
            cost_input=0.01, cost_output=0.01,
        )
        result = registry.select({"code": 1.0})
        assert result is None

    def test_select_empty_registry(self):
        registry = ModelRegistry()
        result = registry.select({"code": 1.0})
        assert result is None

    def test_select_with_min_context(self):
        registry = ModelRegistry()
        registry._profiles["small"] = ModelProfile(
            id="small", provider="test", family="test",
            available=True, code_score=0.9,
            context_window=8_000, cost_input=0.1,
        )
        registry._profiles["big"] = ModelProfile(
            id="big", provider="test", family="test",
            available=True, code_score=0.8,
            context_window=1_000_000, cost_input=0.5,
        )

        result = registry.select({"code": 1.0, "min_context": 100_000})
        assert result is not None
        assert result.id == "big"

    def test_select_with_require_tools(self):
        registry = ModelRegistry()
        registry._profiles["no-tools"] = ModelProfile(
            id="no-tools", provider="test", family="test",
            available=True, code_score=0.9,
            supports_tools=False, cost_input=0.01,
        )
        registry._profiles["has-tools"] = ModelProfile(
            id="has-tools", provider="test", family="test",
            available=True, code_score=0.7,
            supports_tools=True, cost_input=0.5,
        )

        result = registry.select({"code": 1.0, "require_tools": True})
        assert result is not None
        assert result.id == "has-tools"

    def test_list_available(self):
        registry = ModelRegistry()
        registry._profiles["a"] = ModelProfile(
            id="a", provider="test", family="test",
            available=True, cost_input=1.0, cost_output=5.0,
        )
        registry._profiles["b"] = ModelProfile(
            id="b", provider="test", family="test",
            available=False,
        )
        registry._profiles["c"] = ModelProfile(
            id="c", provider="test", family="test",
            available=True, cost_input=0.5, cost_output=2.0,
        )
        available = registry.list_available()
        assert len(available) == 2
        # Sorted by cost_input ascending
        assert available[0].id == "c"
        assert available[1].id == "a"

    def test_get_existing(self):
        registry = ModelRegistry()
        registry._profiles["x"] = ModelProfile(id="x", provider="test")
        assert registry.get("x") is not None
        assert registry.get("x").id == "x"

    def test_get_missing(self):
        registry = ModelRegistry()
        assert registry.get("nonexistent") is None

    def test_select_for_tier_reasoner(self):
        registry = ModelRegistry()
        registry._profiles["smart"] = ModelProfile(
            id="smart", provider="test", family="test",
            available=True, reasoning_score=0.95,
            cost_input=5.0, cost_output=20.0,
        )
        result = registry.select_for_tier("reasoner")
        assert result is not None
        assert result.id == "smart"

    @pytest.mark.asyncio
    async def test_refresh_with_mock_connector(self, monkeypatch):
        """Refresh discovers models and merges them with TOML data."""
        fake_discovered = [
            DiscoveredModel(
                id="gemini-3.1-pro-preview",
                provider="google",
                context_window=1_000_000,
                max_output_tokens=65_536,
            ),
            DiscoveredModel(
                id="brand-new-model",
                provider="openai",
            ),
        ]

        connector_mock = AsyncMock()
        connector_mock.discover_all = AsyncMock(return_value=fake_discovered)

        registry = ModelRegistry()
        registry._connector = connector_mock

        await registry.refresh()

        # gemini-3.1-pro-preview should be available with TOML scores merged
        gemini = registry.get("gemini-3.1-pro-preview")
        assert gemini is not None
        assert gemini.available is True
        assert gemini.code_score == 0.81  # from TOML
        assert gemini.context_window == 1_000_000  # from discovery

        # brand-new-model should be available with default scores
        new_model = registry.get("brand-new-model")
        assert new_model is not None
        assert new_model.available is True
        assert new_model.code_score == 0.5  # default

        # TOML-only models should be present but unavailable
        gpt54 = registry.get("gpt-5.4")
        assert gpt54 is not None
        assert gpt54.available is False
        assert gpt54.code_score == 0.86  # from TOML

    @pytest.mark.asyncio
    async def test_refresh_empty_discovery(self, monkeypatch):
        """When discovery returns nothing, TOML models are still loaded."""
        connector_mock = AsyncMock()
        connector_mock.discover_all = AsyncMock(return_value=[])

        registry = ModelRegistry()
        registry._connector = connector_mock

        await registry.refresh()

        # All TOML models should be present but unavailable
        assert len(registry._profiles) > 0
        for profile in registry._profiles.values():
            assert profile.available is False

    @pytest.mark.asyncio
    async def test_refresh_is_idempotent(self, monkeypatch):
        """Multiple refresh calls don't duplicate entries."""
        fake_models = [
            DiscoveredModel(id="model-x", provider="test"),
        ]
        connector_mock = AsyncMock()
        connector_mock.discover_all = AsyncMock(return_value=fake_models)

        registry = ModelRegistry()
        registry._connector = connector_mock

        await registry.refresh()
        count_after_first = len(registry._profiles)

        await registry.refresh()
        count_after_second = len(registry._profiles)

        assert count_after_first == count_after_second

    def test_select_multi_dimension(self):
        """Select with multiple dimension weights."""
        registry = ModelRegistry()
        registry._profiles["coder"] = ModelProfile(
            id="coder", provider="test", family="test",
            available=True, code_score=0.95, reasoning_score=0.5,
            cost_input=1.0,
        )
        registry._profiles["thinker"] = ModelProfile(
            id="thinker", provider="test", family="test",
            available=True, code_score=0.5, reasoning_score=0.95,
            cost_input=1.0,
        )

        # Code-heavy task
        result = registry.select({"code": 1.0, "reasoning": 0.1})
        assert result.id == "coder"

        # Reasoning-heavy task
        result = registry.select({"code": 0.1, "reasoning": 1.0})
        assert result.id == "thinker"
