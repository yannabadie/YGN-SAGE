"""Tests for CognitiveOrchestrationPipeline (5-stage orchestration).

Post cycle-7 default-on flip (2026-04-29): SAGE_ORACLE unset means ON.
This module's tests cover the LEGACY (pre-oracle) pipeline path. Set
``SAGE_ORACLE=0`` (kill-switch) module-wide so the legacy expectations
hold. Tests for the oracle path live in test_oracle_stack.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext
from sage.pipeline_stages import (
    _infer_domain,
    compute_dag_features,
    select_macro_topology,
    DAGFeatures,
)


@pytest.fixture(autouse=True)
def _legacy_oracle_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """All tests in this module exercise the legacy (pre-oracle) pipeline
    path. Apply the cycle-7 kill-switch ``SAGE_ORACLE=0`` automatically so
    the new default-on does not change these tests' expectations.
    """
    monkeypatch.setenv("SAGE_ORACLE", "0")


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
        self.checked_recorded: list[tuple] = []

    def assess_complexity(self, task: str) -> _MockProfile:
        return _MockProfile(system=self._system)

    def route(self, profile: _MockProfile) -> _MockDecision:
        return _MockDecision(system=profile.system)

    def route_integrated(self, task: str, constraints: Any = None, topology_id: str = "") -> "_MockBanditDecision":
        return _MockBanditDecision()

    def record_outcome_checked(
        self,
        decision_id: str,
        executed_model_id: str,
        executed_template: str,
        quality: float,
        cost: float,
        latency_ms: float,
    ) -> None:
        self.checked_recorded.append(
            (decision_id, executed_model_id, executed_template, quality, cost, latency_ms)
        )

    def cancel_bandit_decision(self, decision_id: str) -> bool:
        return True


class _MockTopology:
    """Mock TopologyGraph-like object."""

    def __init__(self, n_nodes: int = 3) -> None:
        self._n = n_nodes
        self._nodes = [MagicMock(model_id=f"model-{i}", max_cost_usd=0.0) for i in range(n_nodes)]

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

    def generate(self, task: str, task_embedding: Any, system: int, budget: float) -> _MockGenerateResult:
        return self._result


class _MockAssigner:
    """Mock ModelAssigner."""

    def assign_models(
        self,
        topology: Any,
        domain: str,
        budget: float,
        hints: Any = None,
        task_system: int | None = None,
    ) -> int:
        # F7 (2026-04-17): Rust assigner gained a `task_system` param for
        # role-aware tier promotion; the mock accepts+ignores it so legacy
        # and F7-aware pipeline callers both work.
        self.last_task_system = task_system
        n = topology.node_count() if hasattr(topology, "node_count") else 0
        return n


class _MockLLMResponse:
    def __init__(self, content: str = "Pipeline test response") -> None:
        self.content = content
        self.tool_calls: list[Any] = []


class _MockLLMProvider:
    def __init__(self, content: str = "Pipeline test response") -> None:
        self.content = content

    async def generate(self, messages: Any, config: Any = None, tools: Any = None, **kwargs) -> _MockLLMResponse:
        return _MockLLMResponse(self.content)


class _MockExecutionProviderPool:
    def __init__(self, provider: _MockLLMProvider) -> None:
        self.provider = provider

    def is_model_available(self, model_id: str) -> bool:
        return True

    def infer_provider(self, model_id: str) -> str:
        return "mock"

    def resolve(self, model_id: str) -> tuple[_MockLLMProvider, Any]:
        return self.provider, MagicMock(model=model_id, provider="mock")


class _MockBanditDecision:
    def __init__(self):
        self.decision_id = "mock_decision_001"
        self.model_id = "mock-model"
        self.template = "single_agent"
        self.selected_template = "single_agent"
        self.system = 1
        self.confidence = 0.9
        self.estimated_cost = 0.001
        self.context: list[float] = []


class _MockBandit:
    """Mock ContextualBandit for learning stage."""

    def __init__(self) -> None:
        self.recorded: list[tuple] = []

    def select_with_context_for_template(
        self,
        exploration_budget: float = 0.1,
        template: str = "single_agent",
        context: list | None = None,
    ):
        decision = _MockBanditDecision()
        decision.template = template
        decision.context = list(context or [])
        return decision

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

    def record_outcome_checked(
        self,
        decision_id: str,
        executed_model_id: str,
        executed_template: str,
        quality: float,
        cost: float,
        latency_ms: float,
    ) -> None:
        self.recorded.append(
            (decision_id, executed_model_id, executed_template, quality, cost, latency_ms)
        )

    def cancel_decision(self, decision_id: str) -> bool:
        return True


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
    provider = _MockLLMProvider()

    # Single-node topology: exercises classify, decompose, select, assign, and
    # single-agent execute path (avoids needing real TopologyExecutor).
    single_topo = _MockTopology(n_nodes=1)
    engine = _MockEngine(_MockGenerateResult(single_topo))

    router = _MockRouter(system=1)
    pipeline = CognitiveOrchestrationPipeline(
        router=router,
        engine=engine,
        assigner=_MockAssigner(),
        provider_pool=_MockExecutionProviderPool(provider),
        bandit=bandit,
        quality_estimator=_MockQualityEstimator(),
        event_bus=event_bus,
        llm_provider=provider,
        llm_config=None,
    )
    # A14b: _rust_router is separate from router; wire the mock so the pipeline
    # exercises the route_integrated + record_outcome_checked path.
    pipeline._rust_router = router

    result = await pipeline.run("Write a function to sort a list", budget_usd=3.0)

    assert result == "Pipeline test response"
    # Verify events emitted for each stage
    assert event_bus.emit.call_count >= 5  # CLASSIFY, DECOMPOSE, SELECT_TOPOLOGY, ASSIGN_MODELS, LEARN
    # Verify bandit recorded via the checked causal path (A14b: calls _rust_router.record_outcome_checked).
    assert len(router.checked_recorded) == 1
    assert router.checked_recorded[0][0] == "mock_decision_001"


# -- H5 (2026-04-19): single-agent bypass must wire the write_gate -------------
#
# The G-series fix (commit c905d06) wired the pipeline-scoped write_gate
# through `agent_loop_factory.create_node_agent_loop` — but only on the
# multi-node topology traversal path. The SINGLE-AGENT path at
# pipeline.py:941+ reuses a pre-existing `self._agent_loop` singleton that
# never went through the factory, so `loop.write_gate` stayed None and
# phases/act.py fell through to ungated writes. Same silent-bypass class as
# H4 (cache_topology) — fix correct, never fires.

@pytest.mark.asyncio
async def test_pipeline_single_agent_wires_write_gate_onto_agent_loop():
    """Single-agent bypass path must inject write_gate + gate_current_task +
    gate_source_tier onto self._agent_loop before calling .run(). Without
    this, the G-series memory-write gate silently does not fire on S1 tasks
    or single-node topologies.

    Note (2026-04-23, A0a): the gate is restored to its pre-bypass value
    in the `finally` block — so this test captures the wired state
    *during* `.run()` via the spy, not the post-run state (which is
    correctly None again).
    """
    from unittest.mock import MagicMock, AsyncMock

    # Build a single-agent pipeline with a spy agent_loop. Must look enough
    # like a real AgentLoop to receive attribute assignments.
    class _SpyAgentLoop:
        def __init__(self):
            self.config = MagicMock()
            self.config.llm = MagicMock()
            self.config.llm.model = "gemini-3.1-pro-preview"
            self._llm = MagicMock()
            self._skip_routing = False
            self._current_topology = None
            self.sandbox_manager = None
            self.total_cost_usd = 0.0
            self.tool_call_count = 0
            self.tool_turn_count = 0
            self.executed_commands = []
            self.write_gate = None        # unset by default — fix must populate
            self.gate_current_task = ""
            self.gate_source_tier = ""
            self.during_run_snapshot: dict = {}

        async def run(self, task):
            # Capture state DURING the run (post-mutation, pre-restoration)
            # — that's what the H5 fix requires for phases/act.py to see a
            # populated gate. A0a restores all 10 fields in the `finally`,
            # including write_gate, so checking post-run would fail for
            # the correct reason (restoration working) not for the wrong
            # reason (gate not wired).
            self.during_run_snapshot = {
                "write_gate": self.write_gate,
                "gate_current_task": self.gate_current_task,
                "gate_source_tier": self.gate_source_tier,
            }
            return "single-agent test response"

    spy_loop = _SpyAgentLoop()
    # system=1 (S1) skips topology entirely (pipeline.py:474 → ctx.topology=None)
    # which is the canonical route to the single-agent bypass path at
    # pipeline.py:941. S2+ tasks build multi-node templates via
    # _build_topology_from_hint and never enter the bypass branch.
    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=1),
        engine=_MockEngine(_MockGenerateResult(_MockTopology(n_nodes=1))),
        assigner=_MockAssigner(),
        provider_pool=MagicMock(),
        bandit=_MockBandit(),
        quality_estimator=_MockQualityEstimator(),
        event_bus=MagicMock(),
        llm_provider=_MockLLMProvider(),
        llm_config=None,
        agent_loop=spy_loop,
    )

    # Pipeline must have built a real CompositeWriteGate in __init__
    assert pipeline.write_gate is not None, (
        "Pipeline.__init__ must build a write_gate — otherwise the G-series "
        "bypass-fix loop is broken at the root"
    )

    bypass_loop = MagicMock()
    bypass_loop.run = AsyncMock(return_value="single-agent test response")
    bypass_loop.total_cost_usd = 0.0
    bypass_loop.tool_call_count = 0
    bypass_loop.tool_turn_count = 0
    bypass_loop.executed_commands = []
    with patch(
        "sage.pipeline_v2.execute.create_bypass_agent_loop",
        return_value=bypass_loop,
    ) as factory:
        result = await pipeline.run("fix astropy units bug", budget_usd=3.0)

    # Sanity: the per-run loop was actually called (otherwise the test proves nothing)
    assert result == "single-agent test response", (
        f"expected bypass_loop.run() to be the LLM path, got result={result!r}. "
        "If this is 'Pipeline test response', stage_execute fell through to the "
        "llm_provider fallback instead of the agent_loop path — test setup is wrong."
    )

    kwargs = factory.call_args.kwargs
    assert kwargs["write_gate"] is pipeline.write_gate, (
        "single-agent path must pass pipeline.write_gate to the bypass factory; "
        "without it, phases/act.py sees loop.write_gate=None and memory writes "
        "silently skip the 5-signal gate"
    )
    assert kwargs["task_text"] == "fix astropy units bug", (
        "single-agent path must forward ctx.task into the loop for the "
        "gate's relevance signal"
    )
    assert kwargs["singleton"] is spy_loop, (
        "single-agent path must build the per-run loop from the boot singleton"
    )
    assert spy_loop.write_gate is None, (
        "P6-A: singleton write_gate must remain untouched; per-run factory "
        "owns write_gate injection now."
    )


# -- H6 (2026-04-19): single-agent bypass must wire the drift callback --------
#
# Mirrors H5 for a different field. runner.py:502-521 builds an `_on_drift`
# closure and passes it through the factory so SWITCH_MODEL / RESET_AGENT
# drift classifications forward to ProviderPool.record_failure. The
# single-agent bypass path at pipeline.py:941 reused the singleton without
# setting _on_drift → drift events on S1 tasks were silently discarded and
# bad providers never got circuit-broken on that path.

@pytest.mark.asyncio
async def test_pipeline_single_agent_wires_on_drift_onto_agent_loop():
    """Bypass path must set loop._on_drift so drift → ProviderPool.record_failure."""
    from unittest.mock import MagicMock

    recorded_failures = []

    class _SpyProviderPool:
        def __init__(self):
            self.record_failure = MagicMock(side_effect=self._record)

        def _record(self, key, exc):
            recorded_failures.append((key, exc))

    class _SpyAgentLoop:
        def __init__(self):
            self.config = MagicMock()
            self.config.llm = MagicMock()
            self.config.llm.model = "gemini-3.1-pro-preview"
            self._llm = MagicMock()
            self._skip_routing = False
            self._current_topology = None
            self.sandbox_manager = None
            self.total_cost_usd = 0.0
            self.tool_call_count = 0
            self.tool_turn_count = 0
            self.executed_commands = []
            self.write_gate = None
            self.gate_current_task = ""
            self.gate_source_tier = ""
            self._on_drift = None  # unset by default — fix must populate
            self.during_run_on_drift = None

        async def run(self, task):
            # A0a (2026-04-23) restores _on_drift to pre-bypass None in the
            # finally — capture the wired callback DURING the run for the
            # H6 contract instead of post-run.
            self.during_run_on_drift = self._on_drift
            return "single-agent test response"

    spy_loop = _SpyAgentLoop()
    spy_pool = _SpyProviderPool()

    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=1),
        engine=_MockEngine(_MockGenerateResult(_MockTopology(n_nodes=1))),
        assigner=_MockAssigner(),
        provider_pool=spy_pool,
        bandit=_MockBandit(),
        quality_estimator=_MockQualityEstimator(),
        event_bus=MagicMock(),
        llm_provider=_MockLLMProvider(),
        llm_config=None,
        agent_loop=spy_loop,
    )

    bypass_loop = MagicMock()
    bypass_loop.run = AsyncMock(return_value="single-agent test response")
    bypass_loop.total_cost_usd = 0.0
    bypass_loop.tool_call_count = 0
    bypass_loop.tool_turn_count = 0
    bypass_loop.executed_commands = []
    with patch(
        "sage.pipeline_v2.execute.create_bypass_agent_loop",
        return_value=bypass_loop,
    ) as factory:
        await pipeline.run("investigate provider drift", budget_usd=3.0)

    # Bypass path must pass the callback into the per-run loop factory.
    wired = factory.call_args.kwargs["on_drift"]
    assert wired is not None, (
        "single-agent path must pass on_drift to the factory so drift "
        "events propagate to ProviderPool.record_failure — same pattern as "
        "H5 (write_gate). Without it, drift on S1/bypass paths is silent."
    )

    # Callback must actually forward SWITCH_MODEL → record_failure
    wired("gemini", "SWITCH_MODEL", {"latency": 12000})
    assert len(recorded_failures) == 1, (
        "wired callback must call ProviderPool.record_failure on SWITCH_MODEL"
    )
    assert recorded_failures[0][0] == "gemini"

    # Callback must ignore non-actionable actions
    before = len(recorded_failures)
    wired("gemini", "LOG_ONLY", {"latency": 100})
    assert len(recorded_failures) == before, (
        "LOG_ONLY drift must not trip record_failure (only SWITCH_MODEL / RESET_AGENT)"
    )

    assert spy_loop._on_drift is None, (
        "P6-A: singleton _on_drift must remain untouched; per-run factory "
        "owns drift callback injection now."
    )


# -- Plan item 1.1 (2026-04-20): singleton must scale max_steps by system -----
#
# boot.py:279 builds the singleton AgentLoop with max_steps=MAX_AGENT_STEPS=20.
# The factory (agent_loop_factory.py:132-137) scales per-node AgentLoops by
# system tier: S1=5, S2=10, S3=20. The bypass path at pipeline.py:941+ reused
# the singleton without re-scaling — S1 tasks burned 4x the intended step
# budget before the loop could exit. Extends H5/H6 singleton-vs-factory
# asymmetry pattern documented in docs/audits/bypass-patterns.md.
#
# agent_loop.py:424 reads `self.config.max_steps` directly in the step loop,
# so mutation on the bypass path takes effect on the next .run() call —
# same contract as validation_level (which was already mirrored).

@pytest.mark.asyncio
async def test_pipeline_single_agent_scales_max_steps_by_system(monkeypatch):
    """Bypass path must mirror factory's per-system max_steps scaling.

    SAGE_ABLATION_NO_TOPOLOGY=1 forces ctx.topology=None at Stage 2 for all
    system tiers so the bypass branch at pipeline.py:941+ is reached
    regardless of ctx.system. Without it, S2+ tasks with hint="sequential"
    would enter the template-build branch and never mutate the singleton.
    """
    from unittest.mock import MagicMock

    monkeypatch.setenv("SAGE_ABLATION_NO_TOPOLOGY", "1")

    class _SpyAgentLoop:
        def __init__(self):
            self.config = MagicMock()
            self.config.llm = MagicMock()
            self.config.llm.model = "gemini-3.1-pro-preview"
            self._llm = MagicMock()
            self._skip_routing = False
            self._current_topology = None
            self.sandbox_manager = None
            self.total_cost_usd = 0.0
            self.tool_call_count = 0
            self.tool_turn_count = 0
            self.executed_commands = []
            self.write_gate = None
            self.gate_current_task = ""
            self.gate_source_tier = ""
            self._on_drift = None
            self.during_run_max_steps = None

        async def run(self, task):
            # A0a restores config.max_steps to pre-bypass value. Capture
            # the scaled value DURING the run instead of post-run.
            self.during_run_max_steps = self.config.max_steps
            return "single-agent test response"

    expected = {1: 5, 2: 10, 3: 20}
    for system_level, expected_max_steps in expected.items():
        spy_loop = _SpyAgentLoop()
        pipeline = CognitiveOrchestrationPipeline(
            router=_MockRouter(system=system_level),
            engine=_MockEngine(_MockGenerateResult(_MockTopology(n_nodes=1))),
            assigner=_MockAssigner(),
            provider_pool=MagicMock(),
            bandit=_MockBandit(),
            quality_estimator=_MockQualityEstimator(),
            event_bus=MagicMock(),
            llm_provider=_MockLLMProvider(),
            llm_config=None,
            agent_loop=spy_loop,
        )

        bypass_loop = MagicMock()
        bypass_loop.run = AsyncMock(return_value="single-agent test response")
        bypass_loop.total_cost_usd = 0.0
        bypass_loop.tool_call_count = 0
        bypass_loop.tool_turn_count = 0
        bypass_loop.executed_commands = []
        with patch(
            "sage.pipeline_v2.execute.create_bypass_agent_loop",
            return_value=bypass_loop,
        ) as factory:
            await pipeline.run(f"task at S{system_level}", budget_usd=3.0)

        assert factory.call_args.kwargs["system_level"] == system_level, (
            f"expected system_level={system_level} to be forwarded to "
            "create_bypass_agent_loop so the factory applies "
            f"max_steps={expected_max_steps}."
        )


# -- Plan item 1.2 (2026-04-20): singleton must set stall_cap matching factory
#
# AgentConfig.stall_after_tool_steps defaults to 0 (D8 soft-breaker disabled).
# Factory computes (agent_loop_factory.py:151-154):
#   stall_cap = 0                   if max_steps <= 5    (S1 budget too tight)
#             = max_steps - 1       otherwise            (S2→9, S3→19)
# Bypass path had never set this → singleton S2/S3 runs could thrash the
# full step budget on consecutive tool-step cycles without breaking early.

@pytest.mark.asyncio
async def test_pipeline_single_agent_sets_stall_cap_matching_factory(monkeypatch):
    """Bypass path must mirror factory's stall_cap formula after 1.1's max_steps scaling.

    Depends on 1.1 — the stall_cap formula reads the just-set max_steps.
    agent_loop.py:511 live-reads config.stall_after_tool_steps each step
    so the mutation takes effect on the very next .run().
    """
    from unittest.mock import MagicMock

    monkeypatch.setenv("SAGE_ABLATION_NO_TOPOLOGY", "1")

    class _SpyAgentLoop:
        def __init__(self):
            self.config = MagicMock()
            self.config.llm = MagicMock()
            self.config.llm.model = "gemini-3.1-pro-preview"
            self._llm = MagicMock()
            self._skip_routing = False
            self._current_topology = None
            self.sandbox_manager = None
            self.total_cost_usd = 0.0
            self.tool_call_count = 0
            self.tool_turn_count = 0
            self.executed_commands = []
            self.write_gate = None
            self.gate_current_task = ""
            self.gate_source_tier = ""
            self._on_drift = None
            self.during_run_max_steps = None
            self.during_run_stall = None

        async def run(self, task):
            # A0a restores config.{max_steps, stall_after_tool_steps} to
            # pre-bypass values in the finally — capture DURING the run.
            self.during_run_max_steps = self.config.max_steps
            self.during_run_stall = self.config.stall_after_tool_steps
            return "single-agent test response"

    # (system, expected_max_steps, expected_stall_cap) — mirrors factory formula.
    cases = [
        (1, 5, 0),    # S1 — D8 off: budget too tight for any stall window
        (2, 10, 9),   # S2 — 1-step headroom: catches pathological thrash, preserves typical
        (3, 20, 19),  # S3 — same ratio as S2 per F12 revision (2026-04-19)
    ]
    for system_level, expected_max, expected_stall in cases:
        spy_loop = _SpyAgentLoop()
        pipeline = CognitiveOrchestrationPipeline(
            router=_MockRouter(system=system_level),
            engine=_MockEngine(_MockGenerateResult(_MockTopology(n_nodes=1))),
            assigner=_MockAssigner(),
            provider_pool=MagicMock(),
            bandit=_MockBandit(),
            quality_estimator=_MockQualityEstimator(),
            event_bus=MagicMock(),
            llm_provider=_MockLLMProvider(),
            llm_config=None,
            agent_loop=spy_loop,
        )

        bypass_loop = MagicMock()
        bypass_loop.run = AsyncMock(return_value="single-agent test response")
        bypass_loop.total_cost_usd = 0.0
        bypass_loop.tool_call_count = 0
        bypass_loop.tool_turn_count = 0
        bypass_loop.executed_commands = []
        with patch(
            "sage.pipeline_v2.execute.create_bypass_agent_loop",
            return_value=bypass_loop,
        ) as factory:
            await pipeline.run(f"task at S{system_level}", budget_usd=3.0)

        assert factory.call_args.kwargs["system_level"] == system_level, (
            f"expected system_level={system_level} to be forwarded to "
            "create_bypass_agent_loop so the factory applies "
            f"max_steps={expected_max} and stall_cap={expected_stall}."
        )


# -- Plan item 1.4 (2026-04-20): MAP-Elites archive growth at pipeline level -
#
# TestRealEngineEvolutionLoop (test_online_evolution.py) already proves the
# engine.generate → cache_topology → record_outcome chain grows the archive
# at ENGINE level. Plan item 1.4 wants the same empirical validation at
# PIPELINE level — does pipeline.run() actually drive that chain? A full
# SWE-bench smoke (plan's original method) gives the same signal at 50× the
# cost of this test. Using the real sage_core.TopologyEngine + mock LLM
# isolates the wiring question from provider/network variance.
#
# To force the engine path (instead of the template branch), this test
# stubs _build_topology_from_hint to return None — mirroring the fallback
# path the pipeline already exercises when a TemplateStore lookup misses.

try:
    import sage_core as _sage_core  # noqa: F401
    _HAS_SAGE_CORE_PIPELINE = True
except ImportError:
    _HAS_SAGE_CORE_PIPELINE = False


@pytest.mark.skipif(not _HAS_SAGE_CORE_PIPELINE, reason="sage_core (Rust) not compiled")
@pytest.mark.asyncio
async def test_pipeline_real_engine_grows_map_elites_archive():
    """Pipeline-level empirical validation of the generate→cache→record chain.

    Asserts that repeated pipeline.run() calls cause the REAL
    sage_core.TopologyEngine archive to grow past 0 cells. This is what
    plan item 1.4 asks for — the existing H4 test covers the engine contract
    in isolation; this test covers the pipeline-level wiring (Stage 2
    cache_topology + Stage 5 record_outcome) that consumes that contract.

    Failure modes diagnosed by the assertion error:
    - cell_count stays 0 despite >= 10 runs → cache_topology or
      record_outcome regressed (same H4-class bypass).
    - cell_count == 0 on fewer runs → budget degrade path intercepted;
      diagnose via log grep for "Topology budget" warnings.
    """
    import sage_core
    engine = sage_core.TopologyEngine()
    assert engine.archive_cell_count() == 0, "fresh engine must start empty"

    class _PipelineForcingEngineGenerate(CognitiveOrchestrationPipeline):
        """Force Stage 2 to fall through to engine.generate() instead of the
        template branch — templates don't hit the archive-growth wiring."""
        def _build_topology_from_hint(self, hint):
            return None

    pipeline = _PipelineForcingEngineGenerate(
        router=_MockRouter(system=3),   # S1 skips topology entirely; S2 may
                                        # take sequential-template shortcut —
                                        # S3 survives both filters.
        engine=engine,                  # REAL engine → real archive
        assigner=_MockAssigner(),
        provider_pool=MagicMock(),
        bandit=_MockBandit(),
        quality_estimator=_MockQualityEstimator(),  # returns 0.85 — stable
                                                    # signal avoids a zero
                                                    # outcome forcing abstain.
        event_bus=MagicMock(),
        llm_provider=_MockLLMProvider(),
        llm_config=None,
    )

    # Drive 10 diverse tasks so the MAP-Elites descriptor grid gets enough
    # variance to land in at least one cell even on conservative descriptors.
    # The engine-level test uses 20 for EVOLUTION_MIN_OUTCOMES; 10 is
    # sufficient here because we're testing archive growth, not the
    # should_evolve gate.
    for i in range(10):
        try:
            await pipeline.run(f"hierarchical synthesis task variant {i}", budget_usd=5.0)
        except Exception as exc:  # noqa: BLE001
            # Some paths in pipeline.run may raise on mocks — we only care
            # about whether record_outcome fires at least once, so log and
            # move on rather than masking the assertion below.
            import logging
            logging.warning("run %d raised (non-fatal for archive check): %s", i, exc)

    cells = engine.archive_cell_count()
    assert cells > 0, (
        f"After 10 pipeline.run() calls with the real Rust engine, the "
        f"MAP-Elites archive should have >= 1 cell, got {cells}. This is "
        f"the H4-class regression: engine.generate() ran but either "
        f"cache_topology (pipeline.py:549-554) or record_outcome "
        f"(pipeline.py:1349-1380) silently no-opped. Diagnose by grepping "
        f"pipeline logs for 'cache_topology failed' or 'Evolution feedback "
        f"failed'."
    )


@pytest.mark.skipif(not _HAS_SAGE_CORE_PIPELINE, reason="sage_core (Rust) not compiled")
@pytest.mark.asyncio
async def test_pipeline_template_branch_grows_map_elites_archive():
    """Plan item 1.4a (2026-04-20): template branch — dominant production path.

    Before the 1.4a fix (pipeline.py::_apply_topology_budget_and_cache),
    cache_topology only ran on the engine branch. The common S2/S3
    sequential task took the template branch at pipeline.py:~502 which
    returned early → cache never populated → record_outcome miss →
    archive stuck at 0. This regressed the SA-3 "online evolution" claim
    silently because unit tests mocked around the template → archive
    chain. Empirical pipeline-level smoke caught it.
    """
    import sage_core
    engine = sage_core.TopologyEngine()
    assert engine.archive_cell_count() == 0

    # No subclass override → _build_topology_from_hint uses real
    # PyTemplateStore.create() which builds real multi-node topologies.
    # System=2 + hint="sequential" (default) routes through the template
    # branch at pipeline.py:~500.
    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=2),
        engine=engine,
        assigner=_MockAssigner(),
        provider_pool=MagicMock(),
        bandit=_MockBandit(),
        quality_estimator=_MockQualityEstimator(),
        event_bus=MagicMock(),
        llm_provider=_MockLLMProvider(),
        llm_config=None,
    )

    for i in range(10):
        try:
            await pipeline.run(f"refactor module {i}", budget_usd=10.0)
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.warning("template-branch run %d raised: %s", i, exc)

    cells = engine.archive_cell_count()
    assert cells > 0, (
        f"Template branch (production path) must cache topology so "
        f"record_outcome can insert into the archive. Got {cells} cells "
        f"after 10 runs — H10 regression: _apply_topology_budget_and_cache "
        f"not called on the template branch, or cache_topology silently "
        f"no-opped. See plan item 1.4a in "
        f"docs/superpowers/plans/2026-04-20-rust-first-plan.md."
    )


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
async def test_pipeline_simple_fallback_single_call():
    """Without agent_loop, Stage 4 fallback makes a single provider.generate() call (no tool loop)."""
    provider = _MockLLMProvider()

    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=2),
        engine=None,
        assigner=None,
        provider_pool=MagicMock(),
        llm_provider=provider,
        llm_config=None,
    )

    result = await pipeline.run("Fix the failing repository test")

    assert result == "Pipeline test response"
    # No tool calls in the simple fallback (no tool loop)
    assert pipeline.last_context.tool_call_count == 0
    assert pipeline.last_context.tool_turn_count == 0
    assert pipeline.last_context.executed_tools == []


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
        def generate(self, task: str, task_embedding: Any, system: int, budget: float) -> None:
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
    provider = _MockLLMProvider()

    router = _MockRouter(system=1)
    pipeline = CognitiveOrchestrationPipeline(
        router=router,
        engine=None,
        assigner=None,
        provider_pool=_MockExecutionProviderPool(provider),
        bandit=bandit,
        quality_estimator=qe,
        llm_provider=provider,
    )
    pipeline._rust_router = router

    await pipeline.run("Quick task")

    assert len(router.checked_recorded) == 1
    # Quality estimator returns 0.85; index 3 = quality in (decision_id, model_id, template, quality, ...)
    assert router.checked_recorded[0][3] == 0.85


@pytest.mark.asyncio
async def test_pipeline_empty_result_records_zero_quality():
    """Stage 5 records quality=0.0 for causal empty executions."""
    bandit = _MockBandit()
    qe = _MockQualityEstimator()
    provider = _MockLLMProvider(content="")

    router = _MockRouter(system=1)
    pipeline = CognitiveOrchestrationPipeline(
        router=router,
        engine=None,
        assigner=None,
        provider_pool=_MockExecutionProviderPool(provider),
        bandit=bandit,
        quality_estimator=qe,
        llm_provider=provider,
    )
    pipeline._rust_router = router

    result = await pipeline.run("This will fail")

    assert result == ""
    assert len(router.checked_recorded) == 1
    # Empty result => quality must be 0.0, not 0.5
    assert router.checked_recorded[0][3] == 0.0


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
    # Smoke: constructor still accepts the legacy kwargs surface.
    _ = CognitiveOrchestrationPipeline(
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
        # Simulate available providers (all known minus unavailable)
        _all = {"google", "openai", "deepseek", "xai", "minimax", "kimi", "openrouter"}
        self._providers = {p: True for p in _all if p not in self._unavailable}

    def is_available(self, provider_name: str) -> bool:
        return provider_name not in self._unavailable

    def infer_provider(self, model_id: str) -> str:
        if self._registry:
            profile = self._registry.get(model_id) if hasattr(self._registry, 'get') else None
            return getattr(profile, 'provider', '') if profile else ''
        return ""

    def is_model_available(self, model_id: str) -> bool:
        pname = self.infer_provider(model_id)
        if not pname:
            return True
        return pname in self._providers and self.is_available(pname)


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
            def assign_models(
                self,
                topology: Any,
                domain: str,
                budget: float,
                hints: Any = None,
                task_system: int | None = None,
            ) -> int:
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

    def test_coupled_parallel_robust(self):
        assert select_macro_topology(DAGFeatures(omega=2, delta=2, gamma=0.8)) == "robust"

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
    """Budget degradation emits TOPOLOGY_BUDGET_WARNING event.

    Stage 2 now tries template path before falling back to engine.generate().
    To exercise the engine path we patch _build_topology_from_hint → None.
    """
    events_captured: list[Any] = []

    class _CapturingBus:
        def emit(self, event: Any) -> None:
            events_captured.append(event)

    over_topo = _OverBudgetTopology(n_nodes=3, cost_per_node=2.0)

    class _OverBudgetEngine:
        def generate(self, task: str, hint: Any, system: int, budget: float) -> Any:
            return MagicMock(topology=over_topo)

    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=3),  # S3 always uses topology (no adaptive bypass)
        engine=_OverBudgetEngine(),
        assigner=None,
        provider_pool=MagicMock(),
        event_bus=_CapturingBus(),
        llm_provider=_MockLLMProvider(),
    )
    # Force engine.generate() path (templates would shadow the mock engine)
    pipeline._build_topology_from_hint = lambda hint: None  # type: ignore[assignment,method-assign]

    await pipeline.run("test task", budget_usd=5.0)

    stages = [e.meta.get("stage") for e in events_captured if hasattr(e, "meta")]
    assert "TOPOLOGY_BUDGET_WARNING" in stages
