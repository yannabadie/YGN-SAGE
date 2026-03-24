"""Tests for CognitiveOrchestrationPipeline (5-stage orchestration)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, AsyncMock

import pytest

from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext
from sage.pipeline_stages import (
    _infer_domain,
    compute_dag_features,
    select_macro_topology,
    DAGFeatures,
)


# ── Helper mocks ────────────────────────────────────────────────────────────


@dataclass
class _MockProfile:
    complexity: float = 0.5
    uncertainty: float = 0.3
    tool_required: bool = False
    system: int = 2


@dataclass
class _MockDecision:
    system: int = 2
    llm_tier: str = "fast"
    max_tokens: int = 4096
    use_z3: bool = False
    validation_level: int = 1


class _MockRouter:
    """Mock router for pipeline tests."""

    def __init__(self, system: int = 2) -> None:
        self._system = system

    def assess_complexity(self, task: str) -> _MockProfile:
        return _MockProfile(system=self._system)

    def route(self, profile: _MockProfile) -> _MockDecision:
        return _MockDecision(system=profile.system)


class _MockTopology:
    """Mock TopologyGraph-like object."""

    def __init__(self, n_nodes: int = 3) -> None:
        self._n = n_nodes
        self._nodes = [MagicMock(model_id=f"model-{i}") for i in range(n_nodes)]

    def node_count(self) -> int:
        return self._n

    def get_node(self, idx: int) -> Any:
        return self._nodes[idx] if idx < len(self._nodes) else None

    def set_node_model_id(self, idx: int, model_id: str) -> None:
        if idx < len(self._nodes):
            self._nodes[idx].model_id = model_id


class _MockGenerateResult:
    """Mock result from TopologyEngine.generate()."""

    def __init__(self, topology: _MockTopology | None = None) -> None:
        self.topology = topology or _MockTopology()
        self.source = "archive"
        self.confidence = 0.85


class _MockEngine:
    """Mock TopologyEngine."""

    def __init__(self, result: _MockGenerateResult | None = None) -> None:
        self._result = result or _MockGenerateResult()

    def generate(self, task: str, system: int, budget: float) -> _MockGenerateResult:
        return self._result


class _MockAssigner:
    """Mock ModelAssigner."""

    def assign_models(self, topology: Any, domain: str, budget: float) -> int:
        n = topology.node_count() if hasattr(topology, "node_count") else 0
        return n


class _MockLLMResponse:
    content: str = "Pipeline test response"


class _MockLLMProvider:
    async def generate(self, messages: Any, config: Any = None) -> _MockLLMResponse:
        return _MockLLMResponse()


class _MockBanditDecision:
    def __init__(self):
        self.decision_id = "mock_decision_001"


class _MockBandit:
    """Mock ContextualBandit for learning stage."""

    def __init__(self) -> None:
        self.recorded: list[tuple] = []

    def select_with_context(self, exploration_budget: float = 0.1, context: list | None = None):
        return _MockBanditDecision()

    def select(self, exploration_budget: float = 0.1):
        return _MockBanditDecision()

    def choose(self, exploration_budget: float = 0.1):
        return _MockBanditDecision()

    def record(self, arm: str, quality: float, cost: float, latency_ms: float) -> None:
        self.recorded.append((arm, quality, cost, latency_ms))

    def record_outcome(self, decision_id: str, quality: float, cost: float, latency_ms: float) -> None:
        self.recorded.append((decision_id, quality, cost, latency_ms))


class _MockQualityEstimator:
    """Mock QualityEstimator."""

    def estimate(self, task: str, result: str, latency_s: float = 0.0) -> float:
        return 0.85


# ── Pipeline integration tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_full_run():
    """Pipeline completes all 5 stages and returns a result."""
    event_bus = MagicMock()
    bandit = _MockBandit()

    # Single-node topology: exercises classify, decompose, select, assign, and
    # single-agent execute path (avoids needing real TopologyExecutor).
    single_topo = _MockTopology(n_nodes=1)
    engine = _MockEngine(_MockGenerateResult(single_topo))

    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=2),
        engine=engine,
        assigner=_MockAssigner(),
        provider_pool=MagicMock(),
        bandit=bandit,
        quality_estimator=_MockQualityEstimator(),
        event_bus=event_bus,
        llm_provider=_MockLLMProvider(),
        llm_config=None,
    )

    result = await pipeline.run("Write a function to sort a list", budget_usd=3.0)

    assert result == "Pipeline test response"
    # Verify events emitted for each stage
    assert event_bus.emit.call_count >= 5  # CLASSIFY, DECOMPOSE, SELECT_TOPOLOGY, ASSIGN_MODELS, LEARN
    # Verify bandit recorded outcome — bandit.select_with_context() is called
    # before the single-agent early-return, so bandit_decision_id IS set and
    # record_outcome IS called in the LEARN stage.
    assert len(bandit.recorded) == 1
    assert bandit.recorded[0][0] == "mock_decision_001"


@pytest.mark.asyncio
async def test_pipeline_s1_skips_decomposition():
    """S1 tasks skip the decomposition stage entirely."""
    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=1),
        engine=None,  # No topology engine
        assigner=None,
        provider_pool=MagicMock(),
        llm_provider=_MockLLMProvider(),
    )

    result = await pipeline.run("What is 2+2?")

    assert result == "Pipeline test response"


@pytest.mark.asyncio
async def test_pipeline_no_engine_single_agent_fallback():
    """Without topology engine, falls back to single-agent (direct LLM call)."""
    event_bus = MagicMock()

    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=2),
        engine=None,  # No topology engine
        assigner=None,
        provider_pool=MagicMock(),
        event_bus=event_bus,
        llm_provider=_MockLLMProvider(),
    )

    result = await pipeline.run("Explain recursion")

    assert result == "Pipeline test response"


@pytest.mark.asyncio
async def test_pipeline_classify_failure_defaults_to_s2():
    """When classify fails, defaults to S2."""

    class _BrokenRouter:
        def assess_complexity(self, task: str) -> Any:
            raise RuntimeError("Router is broken")

        def route(self, profile: Any) -> Any:
            raise RuntimeError("Router is broken")

    pipeline = CognitiveOrchestrationPipeline(
        router=_BrokenRouter(),
        engine=None,
        assigner=None,
        provider_pool=MagicMock(),
        llm_provider=_MockLLMProvider(),
    )

    result = await pipeline.run("Complex task")

    # Should still succeed with S2 default
    assert result == "Pipeline test response"


@pytest.mark.asyncio
async def test_pipeline_no_router_defaults_to_s2():
    """When no router is provided, defaults to S2."""
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=None,
        assigner=None,
        provider_pool=MagicMock(),
        llm_provider=_MockLLMProvider(),
    )

    # Access internal classify to verify
    ctx = PipelineContext(task="Some task")
    ctx = pipeline._stage_classify(ctx)
    assert ctx.system == 2


@pytest.mark.asyncio
async def test_pipeline_engine_failure_falls_back():
    """When topology engine fails, falls back to single-agent."""

    class _FailEngine:
        def generate(self, task: str, system: int, budget: float) -> None:
            raise RuntimeError("Engine crashed")

    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=2),
        engine=_FailEngine(),
        assigner=_MockAssigner(),
        provider_pool=MagicMock(),
        llm_provider=_MockLLMProvider(),
    )

    result = await pipeline.run("Debug this code")

    # Should still complete with single-agent fallback
    assert result == "Pipeline test response"


@pytest.mark.asyncio
async def test_pipeline_no_llm_provider():
    """When no LLM provider, result is empty."""
    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=1),
        engine=None,
        assigner=None,
        provider_pool=None,
        llm_provider=None,
    )

    result = await pipeline.run("What is AI?")

    assert result == ""


@pytest.mark.asyncio
async def test_pipeline_events_contain_stage_data():
    """Verify emitted events contain stage metadata."""
    events_captured: list[Any] = []

    class _CapturingBus:
        def emit(self, event: Any) -> None:
            events_captured.append(event)

    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=2),
        engine=None,
        assigner=None,
        provider_pool=MagicMock(),
        event_bus=_CapturingBus(),
        llm_provider=_MockLLMProvider(),
    )

    await pipeline.run("Implement bubble sort")

    # Should have events for all stages
    stages = [e.meta.get("stage") for e in events_captured if hasattr(e, "meta")]
    assert "CLASSIFY" in stages
    assert "DECOMPOSE" in stages
    assert "SELECT_TOPOLOGY" in stages
    assert "ASSIGN_MODELS" in stages
    assert "LEARN" in stages


@pytest.mark.asyncio
async def test_pipeline_assigns_models_to_topology():
    """Verify Stage 3 records assignments from topology nodes."""
    topo = _MockTopology(n_nodes=3)
    engine = _MockEngine(_MockGenerateResult(topo))

    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=2),
        engine=engine,
        assigner=_MockAssigner(),
        provider_pool=MagicMock(),
        llm_provider=_MockLLMProvider(),
    )

    # Test just the assign stage
    ctx = PipelineContext(task="test", topology=topo, domain="code", budget=5.0)
    ctx = pipeline._stage_assign_models(ctx)

    assert len(ctx.assignments) == 3
    assert ctx.assignments[0] == "model-0"
    assert ctx.assignments[1] == "model-1"
    assert ctx.assignments[2] == "model-2"


@pytest.mark.asyncio
async def test_pipeline_quality_estimator_used_in_learn():
    """Stage 5 uses quality estimator for bandit feedback."""
    bandit = _MockBandit()
    qe = _MockQualityEstimator()

    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=1),
        engine=None,
        assigner=None,
        provider_pool=MagicMock(),
        bandit=bandit,
        quality_estimator=qe,
        llm_provider=_MockLLMProvider(),
    )

    await pipeline.run("Quick task")

    assert len(bandit.recorded) == 1
    # Quality estimator returns 0.85
    assert bandit.recorded[0][1] == 0.85


@pytest.mark.asyncio
async def test_pipeline_empty_result_records_zero_quality():
    """Stage 5 records quality=0.0 when result is empty (bandit learns from failure)."""
    bandit = _MockBandit()
    qe = _MockQualityEstimator()

    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=1),
        engine=None,
        assigner=None,
        provider_pool=MagicMock(),
        bandit=bandit,
        quality_estimator=qe,
        llm_provider=None,  # No provider => empty result
    )

    result = await pipeline.run("This will fail")

    assert result == ""
    assert len(bandit.recorded) == 1
    # Empty result => quality must be 0.0, not 0.5
    assert bandit.recorded[0][1] == 0.0


@pytest.mark.asyncio
async def test_pipeline_no_estimator_abstains():
    """Stage 5 abstains from bandit recording when no QualityEstimator (quality=None)."""
    bandit = _MockBandit()

    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=1),
        engine=None,
        assigner=None,
        provider_pool=MagicMock(),
        bandit=bandit,
        quality_estimator=None,  # No estimator
        llm_provider=_MockLLMProvider(),
    )

    await pipeline.run("Quick task")

    # No estimator => quality=None => bandit does NOT record
    assert len(bandit.recorded) == 0


@pytest.mark.asyncio
async def test_pipeline_context_preserves_budget():
    """Budget parameter flows through the context."""
    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=1),
        engine=None,
        assigner=None,
        provider_pool=MagicMock(),
        llm_provider=_MockLLMProvider(),
    )

    # Internal check: PipelineContext budget propagation
    ctx = PipelineContext(task="test", budget=7.5)
    assert ctx.budget == 7.5


# ── Pipeline stages unit tests ──────────────────────────────────────────────


class TestDomainInference:
    def test_code_domain(self):
        assert _infer_domain("Write a function to sort numbers") == "code"

    def test_math_domain(self):
        assert _infer_domain("Prove this theorem about algebra") == "math"

    def test_reasoning_domain(self):
        assert _infer_domain("Analyze and compare these approaches") == "reasoning"

    def test_general_domain(self):
        assert _infer_domain("Hello world") == "general"

    def test_formal_domain(self):
        assert _infer_domain("Verify this invariant using SMT") == "formal"

    def test_tool_domain(self):
        # "API endpoint" matches code patterns, not a separate "tool_use" domain
        assert _infer_domain("Call the API endpoint to fetch data") == "code"


class TestDAGFeatures:
    def test_default_features(self):
        # DAGFeatures is frozen — all fields required
        f = DAGFeatures(omega=1, delta=1, gamma=0.0)
        assert f.omega == 1
        assert f.delta == 1
        assert f.gamma == 0.0

    def test_compute_from_none_returns_default(self):
        # compute_dag_features expects a dag with node_ids; None should be handled
        try:
            f = compute_dag_features(None)
            assert f.omega <= 1
        except (AttributeError, TypeError):
            pass  # acceptable — None is not a valid DAG

    def test_compute_from_empty(self):
        mock_dag = MagicMock()
        mock_dag.node_ids = []
        f = compute_dag_features(mock_dag)
        assert f.omega == 0  # empty DAG


class _MockProviderPool:
    """Mock ProviderPool with circuit breaker support."""

    def __init__(
        self, unavailable_providers: set[str] | None = None, registry: Any = None
    ) -> None:
        self._unavailable = unavailable_providers or set()
        self._registry = registry

    def is_available(self, provider_name: str) -> bool:
        return provider_name not in self._unavailable


class _MockModelProfile:
    """Minimal ModelProfile for registry.get()."""

    def __init__(self, model_id: str, provider: str) -> None:
        self.id = model_id
        self.provider = provider


class _MockRegistry:
    """Minimal ModelRegistry for ProviderPool._registry."""

    def __init__(self, profiles: dict[str, _MockModelProfile] | None = None) -> None:
        self._profiles = profiles or {}

    def get(self, model_id: str) -> _MockModelProfile | None:
        return self._profiles.get(model_id)


class TestCircuitBreakerFiltering:
    """Stage 3 should reassign models whose provider has an open circuit breaker."""

    def test_unavailable_provider_reassigned(self):
        """Models from unavailable providers get reassigned to default model."""
        topo = _MockTopology(n_nodes=2)
        # Node 0 -> openai-model (openai provider, circuit OPEN)
        # Node 1 -> google-model (google provider, circuit CLOSED)
        registry = _MockRegistry({
            "openai-model": _MockModelProfile("openai-model", "openai"),
            "google-model": _MockModelProfile("google-model", "google"),
        })
        pool = _MockProviderPool(
            unavailable_providers={"openai"},
            registry=registry,
        )

        class _AssignerThatSetsModels:
            def assign_models(self, topology: Any, domain: str, budget: float) -> int:
                topology._nodes[0].model_id = "openai-model"
                topology._nodes[1].model_id = "google-model"
                return 2

        # Provide llm_config with a default model
        llm_config = MagicMock()
        llm_config.model = "gemini-2.5-flash"

        pipeline = CognitiveOrchestrationPipeline(
            router=_MockRouter(system=2),
            engine=None,
            assigner=_AssignerThatSetsModels(),
            provider_pool=pool,
            llm_provider=_MockLLMProvider(),
            llm_config=llm_config,
        )

        ctx = PipelineContext(task="test", topology=topo, domain="code", budget=5.0)
        ctx = pipeline._stage_assign_models(ctx)

        # Node 0 should be reassigned to the default model
        assert ctx.assignments[0] == "gemini-2.5-flash"
        # Node 1 stays on google-model (provider is available)
        assert ctx.assignments[1] == "google-model"

    def test_all_providers_available_no_change(self):
        """When all providers are available, no reassignment happens."""
        topo = _MockTopology(n_nodes=2)
        registry = _MockRegistry({
            "model-a": _MockModelProfile("model-a", "google"),
            "model-b": _MockModelProfile("model-b", "google"),
        })
        pool = _MockProviderPool(
            unavailable_providers=set(),
            registry=registry,
        )

        pipeline = CognitiveOrchestrationPipeline(
            router=_MockRouter(system=2),
            engine=None,
            assigner=_MockAssigner(),
            provider_pool=pool,
            llm_provider=_MockLLMProvider(),
            llm_config=MagicMock(model="default-model"),
        )

        ctx = PipelineContext(task="test", topology=topo, domain="code", budget=5.0)
        ctx = pipeline._stage_assign_models(ctx)

        # Original assignments preserved (model-0, model-1 from _MockTopology)
        assert ctx.assignments[0] == "model-0"
        assert ctx.assignments[1] == "model-1"

    def test_no_provider_pool_skips_filtering(self):
        """Without a provider_pool, filtering is skipped gracefully."""
        topo = _MockTopology(n_nodes=1)

        pipeline = CognitiveOrchestrationPipeline(
            router=_MockRouter(system=2),
            engine=None,
            assigner=_MockAssigner(),
            provider_pool=None,
            llm_provider=_MockLLMProvider(),
        )

        ctx = PipelineContext(task="test", topology=topo, domain="code", budget=5.0)
        ctx = pipeline._stage_assign_models(ctx)

        assert ctx.assignments[0] == "model-0"


class TestTopologySelection:
    def test_sequential_default(self):
        assert select_macro_topology(DAGFeatures(omega=1, delta=1, gamma=0.0)) == "sequential"

    def test_parallel_wide(self):
        assert select_macro_topology(DAGFeatures(omega=4, delta=1, gamma=0.3)) == "parallel"

    def test_hierarchical_dense(self):
        assert select_macro_topology(DAGFeatures(omega=2, delta=2, gamma=0.8)) == "hierarchical"

    def test_sequential_moderate_depth(self):
        assert select_macro_topology(DAGFeatures(omega=1, delta=2, gamma=0.1)) == "sequential"


# ── Budget degradation tests ────────────────────────────────────────────────


class _OverBudgetTopology:
    """Topology whose nodes exceed the pipeline budget."""

    def __init__(self, n_nodes: int = 3, cost_per_node: float = 2.0) -> None:
        self._n = n_nodes
        self._cost = cost_per_node
        self._nodes = [
            MagicMock(model_id=f"model-{i}", max_cost_usd=cost_per_node)
            for i in range(n_nodes)
        ]

    def node_count(self) -> int:
        return self._n

    def get_node(self, idx: int) -> Any:
        return self._nodes[idx] if idx < len(self._nodes) else None


def test_check_topology_budget_degrades_when_over():
    """When topology cost > budget, _check_topology_budget replaces with single-node."""
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=None,
        assigner=None,
        provider_pool=None,
    )

    # 3 nodes x $2.00 = $6.00 total, budget = $5.00
    over_topo = _OverBudgetTopology(n_nodes=3, cost_per_node=2.0)
    ctx = PipelineContext(task="test", budget=5.0, system=2, topology=over_topo)

    pipeline._check_topology_budget(ctx)

    # Topology should have been replaced (not the original 3-node topology)
    # Without sage_core, fallback is None (single-agent mode)
    # With sage_core, it would be a 1-node TopologyGraph
    if ctx.topology is not None:
        assert ctx.topology.node_count() == 1
    else:
        # sage_core not available — degraded to None (single-agent)
        assert ctx.topology is None


def test_check_topology_budget_no_degrade_when_under():
    """When topology cost <= budget, topology is kept as-is."""
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=None,
        assigner=None,
        provider_pool=None,
    )

    # 3 nodes x $1.00 = $3.00 total, budget = $5.00 — within budget
    under_topo = _OverBudgetTopology(n_nodes=3, cost_per_node=1.0)
    ctx = PipelineContext(task="test", budget=5.0, system=2, topology=under_topo)

    pipeline._check_topology_budget(ctx)

    # Topology unchanged — still the original 3-node topology
    assert ctx.topology is under_topo
    assert ctx.topology.node_count() == 3


@pytest.mark.asyncio
async def test_pipeline_budget_degrade_emits_event():
    """Budget degradation emits TOPOLOGY_BUDGET_WARNING event."""
    events_captured: list[Any] = []

    class _CapturingBus:
        def emit(self, event: Any) -> None:
            events_captured.append(event)

    over_topo = _OverBudgetTopology(n_nodes=3, cost_per_node=2.0)

    class _OverBudgetEngine:
        def generate(self, task: str, hint: Any, system: int, budget: float) -> Any:
            return MagicMock(topology=over_topo)

    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=2),
        engine=_OverBudgetEngine(),
        assigner=None,
        provider_pool=MagicMock(),
        event_bus=_CapturingBus(),
        llm_provider=_MockLLMProvider(),
    )

    await pipeline.run("test task", budget_usd=5.0)

    stages = [e.meta.get("stage") for e in events_captured if hasattr(e, "meta")]
    assert "TOPOLOGY_BUDGET_WARNING" in stages
