"""Integration test: multi-provider routing end-to-end (all mocked).

Validates the complete flow: boot -> discover -> select -> cascade
works across providers with mocked API responses. No real API calls.
"""
import sys
import types as _types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = _types.ModuleType("sage_core")
    _mock = sys.modules["sage_core"]

    class _WM:
        def __init__(self, **kw):
            self._events = []
            self._counter = 0

        def add_event(self, t, c):
            self._counter += 1
            self._events.append({"type": t, "content": c})
            return f"e-{self._counter}"

        def event_count(self):
            return len(self._events)

        def recent_events(self, n):
            return self._events[-n:]

        def compact_to_arrow(self):
            return 0

        def compact_to_arrow_with_meta(self, *a):
            return 0

        def retrieve_relevant_chunks(self, *a):
            return []

        def get_page_out_candidates(self, *a):
            return []

        def smmu_chunk_count(self):
            return 0

        def get_latest_arrow_chunk(self):
            return None

    _mock.WorkingMemory = _WM

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sage.providers.registry import ModelProfile, ModelRegistry
from sage.providers.connector import DiscoveredModel
from sage.orchestrator import CognitiveOrchestrator, ModelAgent
from sage.strategy.metacognition import ComplexityRouter


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_profile(
    id: str,
    provider: str = "test",
    code: float = 0.5,
    reasoning: float = 0.5,
    cost_in: float = 1.0,
    cost_out: float = 5.0,
    **kwargs,
) -> ModelProfile:
    """Create a minimal available ModelProfile for testing."""
    return ModelProfile(
        id=id,
        provider=provider,
        family="test",
        available=True,
        code_score=code,
        reasoning_score=reasoning,
        cost_input=cost_in,
        cost_output=cost_out,
        **kwargs,
    )


def _mock_registry(*profiles: ModelProfile) -> ModelRegistry:
    """Create a ModelRegistry pre-populated with the given profiles."""
    reg = ModelRegistry.__new__(ModelRegistry)
    reg._profiles = {}
    reg._connector = MagicMock()
    for p in profiles:
        reg._profiles[p.id] = p
    return reg


# ── Test 1: S1 routing selects cheapest viable model ────────────────────────


class TestS1CheapestModelSelection:
    @pytest.mark.asyncio
    async def test_orchestrator_selects_cheapest_for_simple_task(self):
        """S1 routing should select the cheapest viable model for simple tasks.

        The task 'What is 2+2?' is simple (low complexity, low uncertainty),
        so ComplexityRouter routes to S1. S1 applies max_cost_per_1m=5.0 and
        cost_sensitivity=0.8, which should prefer the cheap model.
        """
        cheap = _make_profile(
            "cheap-flash", provider="google",
            code=0.6, reasoning=0.7, cost_in=0.025, cost_out=1.5,
        )
        expensive = _make_profile(
            "expensive-pro", provider="openai",
            code=0.9, reasoning=0.95, cost_in=15.0, cost_out=120.0,
        )

        reg = _mock_registry(cheap, expensive)
        orch = CognitiveOrchestrator(registry=reg)

        # Track which model was actually selected by capturing the agent
        selected_model_ids = []

        original_init = ModelAgent.__init__

        def tracking_init(self_agent, name, model, system_prompt="", registry=None):
            selected_model_ids.append(model.id)
            original_init(self_agent, name, model, system_prompt, registry)

        with patch.object(ModelAgent, "__init__", tracking_init), \
             patch.object(ModelAgent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "4"
            result = await orch.run("What is 2+2?")

        assert result == "4"
        # S1 should pick cheap-flash (expensive-pro is above max_cost_per_1m=5.0)
        assert "cheap-flash" in selected_model_ids


# ── Test 2: S2 routing prefers high code_score for code tasks ────────────────


class TestS2CodeModelSelection:
    @pytest.mark.asyncio
    async def test_orchestrator_routes_s2_for_implement_task(self):
        """S2 routing for code tasks: 'implement' keyword triggers higher complexity.

        'Implement a Python function that computes fibonacci with memoization'
        has complexity=0.55 (base 0.2 + implement 0.35), which routes to S2.
        S2 code path selects with code=1.0 and cost_sensitivity=0.2.
        With quality^1.8/cost^0.2, a model with much higher code_score at
        similar cost should win.
        """
        coder = _make_profile(
            "gpt-codex", provider="openai",
            code=0.95, reasoning=0.93, cost_in=0.50, cost_out=2.0,
        )
        generalist = _make_profile(
            "deepseek-chat", provider="deepseek",
            code=0.50, reasoning=0.50, cost_in=0.28, cost_out=0.42,
        )

        reg = _mock_registry(coder, generalist)
        orch = CognitiveOrchestrator(registry=reg)

        selected_model_ids = []
        original_init = ModelAgent.__init__

        def tracking_init(self_agent, name, model, system_prompt="", registry=None):
            selected_model_ids.append(model.id)
            original_init(self_agent, name, model, system_prompt, registry)

        with patch.object(ModelAgent, "__init__", tracking_init), \
             patch.object(ModelAgent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "def fibonacci(n): ..."
            result = await orch.run(
                "Implement a Python function that computes fibonacci with memoization"
            )

        assert result == "def fibonacci(n): ..."
        # S2 code path selects with code=1.0, cost_sensitivity=0.2
        # At similar cost, quality dominates -> gpt-codex (0.95) beats deepseek-chat (0.50)
        assert "gpt-codex" in selected_model_ids


# ── Test 3: Registry select respects cost cap ────────────────────────────────


class TestRegistryCostCap:
    def test_registry_select_respects_cost_cap(self):
        """Models above cost cap should be excluded from selection."""
        cheap = _make_profile(
            "budget", provider="google",
            code=0.5, reasoning=0.5, cost_in=0.025, cost_out=0.5,
        )
        expensive = _make_profile(
            "premium", provider="openai",
            code=0.9, reasoning=0.95, cost_in=15.0, cost_out=120.0,
        )

        reg = _mock_registry(cheap, expensive)

        # With max_cost_per_1m=5.0, premium (cost_in=15.0) should be excluded
        result = reg.select({"code": 0.5, "max_cost_per_1m": 5.0})
        assert result is not None
        assert result.id == "budget"

    def test_all_models_above_cap_returns_none(self):
        """If all models exceed cost cap, select returns None."""
        expensive = _make_profile("pricey", cost_in=10.0, cost_out=50.0)
        reg = _mock_registry(expensive)

        result = reg.select({"code": 0.5, "max_cost_per_1m": 1.0})
        assert result is None


# ── Test 4: Cascade across providers ─────────────────────────────────────────


class TestCascadeAcrossProviders:
    @pytest.mark.asyncio
    async def test_cascade_on_primary_failure(self):
        """When primary provider fails, cascade should try fallback from different provider."""
        primary = _make_profile(
            "primary-gpt", provider="openai",
            code=0.85, reasoning=0.9, cost_in=2.0, cost_out=10.0,
        )
        fallback = _make_profile(
            "fallback-gemini", provider="google",
            code=0.7, reasoning=0.8, cost_in=0.15, cost_out=3.5,
        )

        reg = _mock_registry(primary, fallback)
        agent = ModelAgent(name="test", model=primary, registry=reg)

        call_log: list[str] = []

        async def failing_then_ok(model, messages):
            call_log.append(model.id)
            if model.id == "primary-gpt":
                raise RuntimeError("OpenAI rate limited")
            return "Fallback response"

        agent._call_provider = failing_then_ok
        result = await agent.run("Test cascade")

        assert result == "Fallback response"
        assert call_log == ["primary-gpt", "fallback-gemini"]

    @pytest.mark.asyncio
    async def test_cascade_exhausted_returns_error_message(self):
        """When all models fail, the error message includes the last error."""
        model_a = _make_profile("model-a", provider="openai", cost_in=2.0)
        model_b = _make_profile("model-b", provider="google", cost_in=0.5)
        reg = _mock_registry(model_a, model_b)
        agent = ModelAgent(name="test", model=model_a, registry=reg)

        async def always_fail(model, messages):
            raise RuntimeError(f"{model.id} is down")

        agent._call_provider = always_fail
        result = await agent.run("Doomed task")

        assert "error" in result.lower()
        # Should mention both models were tried
        assert "model" in result.lower()


# ── Test 5: Full boot -> discover -> select -> execute flow ──────────────────


class TestFullBootToExecuteFlow:
    @pytest.mark.asyncio
    async def test_boot_discover_select_execute(self):
        """End-to-end: boot registry, discover models, select, execute with cascade.

        Simulates the real boot flow:
        1. ProviderConnector discovers models from Google + OpenAI
        2. ModelRegistry merges with TOML knowledge base
        3. CognitiveOrchestrator routes task
        4. ModelAgent calls provider (mocked)
        """
        # Simulate discovered models from two providers
        discovered = [
            DiscoveredModel(id="gemini-flash", provider="google",
                            context_window=1_000_000, max_output_tokens=8192),
            DiscoveredModel(id="gpt-5", provider="openai",
                            context_window=128_000, max_output_tokens=16384),
        ]

        # Custom TOML with scores for these models
        toml_knowledge = {
            "gemini-flash": {
                "provider": "google", "family": "gemini-3",
                "code_score": 0.7, "reasoning_score": 0.65,
                "cost_input": 0.075, "cost_output": 0.3,
            },
            "gpt-5": {
                "provider": "openai", "family": "gpt-5",
                "code_score": 0.92, "reasoning_score": 0.95,
                "cost_input": 2.5, "cost_output": 10.0,
            },
        }

        # Build registry with mocked connector and TOML
        reg = ModelRegistry()
        reg._connector = AsyncMock()
        reg._connector.discover_all = AsyncMock(return_value=discovered)

        with patch.object(reg, "_load_toml", return_value=toml_knowledge):
            await reg.refresh()

        # Verify discovery + merge worked
        assert reg.get("gemini-flash") is not None
        assert reg.get("gemini-flash").available is True
        assert reg.get("gemini-flash").code_score == 0.7
        assert reg.get("gpt-5") is not None
        assert reg.get("gpt-5").available is True
        assert reg.get("gpt-5").code_score == 0.92

        # Now run through orchestrator
        orch = CognitiveOrchestrator(registry=reg)

        with patch.object(ModelAgent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Hello from the selected model"
            result = await orch.run("Hello world")

        assert result == "Hello from the selected model"
        mock_run.assert_awaited_once()


# ── Test 6: Multi-provider selection with cost_sensitivity ───────────────────


class TestCostSensitivityRouting:
    def test_high_cost_sensitivity_prefers_cheap(self):
        """cost_sensitivity=1.0 should maximize quality/cost ratio (cheapest decent)."""
        premium = _make_profile("premium", code=0.95, cost_in=10.0, cost_out=50.0)
        budget = _make_profile("budget", code=0.7, cost_in=0.1, cost_out=0.5)
        reg = _mock_registry(premium, budget)

        result = reg.select({"code": 1.0, "cost_sensitivity": 1.0})
        assert result is not None
        assert result.id == "budget"  # Much better quality/cost ratio

    def test_zero_cost_sensitivity_prefers_best_quality(self):
        """cost_sensitivity=0.0 should pick the best quality regardless of cost."""
        premium = _make_profile("premium", code=0.95, cost_in=10.0, cost_out=50.0)
        budget = _make_profile("budget", code=0.7, cost_in=0.1, cost_out=0.5)
        reg = _mock_registry(premium, budget)

        result = reg.select({"code": 1.0, "cost_sensitivity": 0.0})
        assert result is not None
        assert result.id == "premium"  # Pure quality, cost irrelevant


# ── Test 7: Cross-provider fallback order ────────────────────────────────────


class TestCrossProviderFallbackOrder:
    @pytest.mark.asyncio
    async def test_fallback_order_by_cost(self):
        """Fallback models are picked from list_available() which sorts by cost."""
        primary = _make_profile("primary", provider="openai", cost_in=5.0, cost_out=20.0)
        cheap_fallback = _make_profile("cheap", provider="google", cost_in=0.1, cost_out=0.5)
        mid_fallback = _make_profile("mid", provider="deepseek", cost_in=1.0, cost_out=4.0)

        reg = _mock_registry(primary, cheap_fallback, mid_fallback)
        agent = ModelAgent(name="test", model=primary, registry=reg)

        call_log: list[str] = []

        async def fail_all_but_last(model, messages):
            call_log.append(model.id)
            if model.id != "mid":
                raise RuntimeError(f"{model.id} down")
            return "mid model answered"

        agent._call_provider = fail_all_but_last
        result = await agent.run("Test fallback order")

        # Primary fails -> picks from list_available (sorted by cost)
        # cheap (0.1) is first fallback, then mid (1.0) if cheap fails
        assert result == "mid model answered"
        assert call_log[0] == "primary"
        # The fallback order follows list_available() which sorts by cost ascending,
        # excluding already-tried IDs
        assert "cheap" in call_log
        assert "mid" in call_log


# ── Test 8: Registry handles mixed provider availability ─────────────────────


class TestMixedProviderAvailability:
    @pytest.mark.asyncio
    async def test_discovered_available_toml_only_unavailable(self):
        """Discovered models are available; TOML-only models are unavailable."""
        discovered = [
            DiscoveredModel(id="live-model", provider="google"),
        ]
        toml_knowledge = {
            "live-model": {
                "provider": "google", "code_score": 0.8,
                "cost_input": 0.1, "cost_output": 0.5,
            },
            "offline-model": {
                "provider": "openai", "code_score": 0.95,
                "cost_input": 5.0, "cost_output": 20.0,
            },
        }

        reg = ModelRegistry()
        reg._connector = AsyncMock()
        reg._connector.discover_all = AsyncMock(return_value=discovered)

        with patch.object(reg, "_load_toml", return_value=toml_knowledge):
            await reg.refresh()

        # Live model is available and selectable
        live = reg.get("live-model")
        assert live is not None
        assert live.available is True

        # Offline model exists but not available
        offline = reg.get("offline-model")
        assert offline is not None
        assert offline.available is False

        # Select should only return the live model
        result = reg.select({"code": 1.0})
        assert result is not None
        assert result.id == "live-model"


# ── Test 9: Routing system level matches task complexity ─────────────────────


class TestRoutingSystemLevelIntegration:
    @pytest.mark.asyncio
    async def test_simple_task_routes_s1(self):
        """Simple factual question routes to S1."""
        router = ComplexityRouter()
        profile = router.assess_complexity("What is 2+2?")
        decision = router.route(profile)
        assert decision.system == 1

    @pytest.mark.asyncio
    async def test_code_task_routes_s2(self):
        """Implementation task routes to S2 (complexity=0.55 from 'implement' keyword)."""
        router = ComplexityRouter()
        profile = router.assess_complexity(
            "Implement a Python function that computes fibonacci with memoization"
        )
        decision = router.route(profile)
        assert decision.system == 2

    @pytest.mark.asyncio
    async def test_complex_task_routes_s3(self):
        """Complex debug + design task routes to S3."""
        router = ComplexityRouter()
        profile = router.assess_complexity(
            "Debug the distributed race condition in the optimizer, "
            "then design a new architecture to prevent deadlocks"
        )
        decision = router.route(profile)
        assert decision.system == 3


# ── Test 10: Event bus receives orchestrator events ──────────────────────────


class TestEventBusIntegration:
    @pytest.mark.asyncio
    async def test_event_emitted_with_model_info(self):
        """Orchestrator should emit event with selected model ID and system level."""
        model = _make_profile("test-model", cost_in=0.1, cost_out=0.5)
        reg = _mock_registry(model)
        bus = MagicMock()
        orch = CognitiveOrchestrator(registry=reg, event_bus=bus)

        with patch.object(ModelAgent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "OK"
            await orch.run("Hello")

        bus.emit.assert_called_once()
        event = bus.emit.call_args[0][0]
        assert event.type == "ORCHESTRATOR"
        assert event.system == 1  # Simple task -> S1
        assert event.model == "test-model"
        assert event.latency_ms >= 0
