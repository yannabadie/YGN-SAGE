"""In-run logging for Memory and Evolution pillars.

The Memory and Evolution pillars emit only boot-time signal today. Nothing is
captured during a benchmark run, so gains/regressions cannot be attributed to
pillar behavior. This module pins six structured log lines so future smokes
show per-task pillar activity:

  Memory pillar:
    memory.write_gate.fired             -- RustCompositeWriteGate decision
    memory.smmu.tier_transition         -- STM chunk compacted to Arrow
    memory.archive.grow                 -- MAP-Elites archive cell count delta
    memory.consolidation.fired          -- episodic -> semantic consolidation

  Evolution pillar:
    evolution.should_evolve.decision    -- every Rust should_evolve() call
    evolution.evolve.called             -- when evolve() actually fires
    evolution.mutator.update            -- AdaptiveMutator.record() posterior update

Format: key=value pairs with %-formatting, one line per event, INFO level.
All log lines must be parseable by a simple regex.
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from sage.pipeline import CognitiveOrchestrationPipeline


# ── Minimal stage mocks (subset of test_pipeline.py — kept self-contained) ───


class _MockProfile:
    def __init__(self, system: int = 2) -> None:
        self.complexity = 0.5
        self.uncertainty = 0.3
        self.tool_required = False
        self.system = system


class _MockDecision:
    def __init__(self, system: int = 2) -> None:
        self.system = system
        self.llm_tier = "fast"
        self.max_tokens = 4096
        self.use_z3 = False
        self.validation_level = 1


class _MockRouter:
    def __init__(self, system: int = 1) -> None:
        self._system = system

    def assess_complexity(self, task: str) -> _MockProfile:
        return _MockProfile(system=self._system)

    def route(self, profile: _MockProfile) -> _MockDecision:
        return _MockDecision(system=profile.system)


class _MockTopology:
    """Minimal TopologyGraph-like object with a stable id."""

    def __init__(self, n_nodes: int = 1, topo_id: str = "mock-topo") -> None:
        self._n = n_nodes
        self.id = topo_id
        self._nodes = [
            MagicMock(model_id=f"model-{i}", max_cost_usd=0.0)
            for i in range(n_nodes)
        ]

    def node_count(self) -> int:
        return self._n

    def get_node(self, idx: int) -> Any:
        return self._nodes[idx] if idx < len(self._nodes) else None

    def set_node_model_id(self, idx: int, model_id: str) -> None:
        if idx < len(self._nodes):
            self._nodes[idx].model_id = model_id


class _MockGenerateResult:
    def __init__(self, topology: _MockTopology) -> None:
        self.topology = topology
        self.source = "archive"
        self.confidence = 0.85


class _MockEngine:
    """Mock TopologyEngine with archive + should_evolve accessors.

    - `cache_topology` is a no-op (so Stage-2 _apply_topology_budget_and_cache
      succeeds without a real Rust engine).
    - `record_outcome` grows the internal archive counter by 1 on each call,
      so the `memory.archive.grow` log fires deterministically.
    - `should_evolve` returns a programmable answer (default False) so the
      `evolution.should_evolve.decision` log fires on every call, regardless
      of the decision.
    """

    def __init__(
        self,
        result: _MockGenerateResult | None = None,
        should_evolve_ret: bool = False,
    ) -> None:
        self._result = result or _MockGenerateResult(_MockTopology(n_nodes=1))
        self._cell_count = 0
        self._coverage = 0.0
        self._should_evolve_ret = should_evolve_ret
        self.cache_topology_calls: list[Any] = []
        self.record_outcome_calls = 0
        self.should_evolve_calls = 0
        self.evolve_calls = 0

    def generate(
        self, task: str, task_embedding: Any, system: int, budget: float,
    ) -> _MockGenerateResult:
        return self._result

    def cache_topology(self, topology: Any) -> None:
        self.cache_topology_calls.append(topology)

    def record_outcome(
        self,
        topology_id: str,
        task: str,
        keywords: list[str],
        embedding: Any,
        quality: float,
        cost: float,
        latency_ms: float,
    ) -> None:
        self.record_outcome_calls += 1
        # Simulate archive growth on first outcome only — mirrors
        # "first outcome for a cell grows cells, subsequent refines fitness"
        if self.record_outcome_calls == 1:
            self._cell_count += 1
            self._coverage = 0.1

    def archive_cell_count(self) -> int:
        return self._cell_count

    def archive_coverage(self) -> float:
        return self._coverage

    def should_evolve(self, min_outcomes: int, cooldown: int) -> bool:
        self.should_evolve_calls += 1
        return self._should_evolve_ret

    def evolve(self, pop_size: int = 5, generations: int = 2) -> None:
        self.evolve_calls += 1
        self._cell_count += generations  # simulate archive growth via evolve


class _MockAssigner:
    def assign_models(
        self,
        topology: Any,
        domain: str,
        budget: float,
        hints: Any = None,
        task_system: int | None = None,
    ) -> int:
        return topology.node_count() if hasattr(topology, "node_count") else 0


class _MockLLMResponse:
    content = "Mock LLM response payload for pillar logging test."
    tool_calls: list[Any] = []


class _MockLLMProvider:
    async def generate(
        self, messages: Any, config: Any = None, tools: Any = None, **kwargs: Any,
    ) -> _MockLLMResponse:
        return _MockLLMResponse()


class _MockBanditDecision:
    decision_id = "mock_decision_pillar_log"


class _MockBandit:
    def __init__(self) -> None:
        self.recorded: list[tuple[Any, ...]] = []

    def select_with_context(
        self, exploration_budget: float = 0.1, context: list | None = None,
    ) -> _MockBanditDecision:
        return _MockBanditDecision()

    def select(self, exploration_budget: float = 0.1) -> _MockBanditDecision:
        return _MockBanditDecision()

    def choose(self, exploration_budget: float = 0.1) -> _MockBanditDecision:
        return _MockBanditDecision()

    def record(
        self, arm: str, quality: float, cost: float, latency_ms: float,
    ) -> None:
        self.recorded.append((arm, quality, cost, latency_ms))

    def record_outcome(
        self, decision_id: str, quality: float, cost: float, latency_ms: float,
    ) -> None:
        self.recorded.append((decision_id, quality, cost, latency_ms))


class _MockQualityEstimator:
    def estimate(self, task: str, result: str, latency_s: float = 0.0) -> float:
        return 0.85


class _SpyAgentLoop:
    """Spy for the single-agent bypass path (S1 → topology=None).

    The pipeline's bypass branch (pipeline.py: H5/H6) wires the pipeline-level
    write_gate onto this loop and calls loop.run(task). Since our _MockLLMProvider
    returns a static string, we return it from .run() so the pipeline LEARN
    stage sees a populated ctx.result.
    """

    def __init__(self) -> None:
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
        self.executed_commands: list[Any] = []
        self.write_gate: Any = None
        self.gate_current_task: str = ""
        self.gate_source_tier: str = ""

    async def run(self, task: str) -> str:
        # Exercise the write gate once per task, mirroring what phases/act.py
        # does: evaluate() then log via the shared helper. The pipeline's
        # single-agent bypass wires the gate onto this loop before calling
        # run() -- so gate is guaranteed populated here.
        if self.write_gate is not None:
            try:
                decision = self.write_gate.evaluate(
                    f"response to task: {task}",
                    1.0,
                    task=self.gate_current_task or task,
                    source_tier=self.gate_source_tier or "unknown",
                    embedding=None,
                )
                from sage.memory.write_gate import log_write_gate_decision
                log_write_gate_decision(decision, source_tier=self.gate_source_tier)
            except Exception:
                pass
        return "spy-agent-loop-response"


def _build_pipeline(
    should_evolve_ret: bool = False,
    system: int = 1,
) -> tuple[CognitiveOrchestrationPipeline, _MockEngine, _SpyAgentLoop]:
    """Construct a pipeline wired end-to-end with minimal mocks.

    `system=1` routes via the single-agent bypass path (pipeline.py H5), which
    wires the write_gate onto our spy loop -- this exercises the write_gate
    log. S1 sets ctx.topology=None, so the LEARN-stage `record_outcome` and
    `archive.grow` logs won't fire on that path; a separate test uses
    `system=2` + a pre-injected 1-node topology to exercise those.
    """
    topology = _MockTopology(n_nodes=1, topo_id="mock-topo-pillar")
    engine = _MockEngine(_MockGenerateResult(topology), should_evolve_ret=should_evolve_ret)
    spy_loop = _SpyAgentLoop()
    pipeline = CognitiveOrchestrationPipeline(
        router=_MockRouter(system=system),
        engine=engine,
        assigner=_MockAssigner(),
        provider_pool=MagicMock(),
        bandit=_MockBandit(),
        quality_estimator=_MockQualityEstimator(),
        event_bus=MagicMock(),
        llm_provider=_MockLLMProvider(),
        llm_config=None,
        agent_loop=spy_loop,
    )
    return pipeline, engine, spy_loop


async def _run_with_injected_topology(
    pipeline: CognitiveOrchestrationPipeline,
    engine: _MockEngine,
    task: str,
) -> None:
    """Call pipeline.run() but patch Stage 2 to use our mock 1-node topology.

    We monkey-patch `_stage_select_topology` to always set ctx.topology to
    our mock. This ensures:
      - Stage 4 takes the single-agent bypass (node_count=1)
      - LEARN stage sees ctx.topology is not None -> record_outcome fires
      - LEARN stage calls should_evolve() after the outcome is recorded

    Without this, S2+sequential builds a real 3-node Rust topology (multi-node
    path) and our spy loop is bypassed entirely.
    """
    mock_topo = engine._result.topology
    original_stage = pipeline._stage_select_topology

    def _patched(ctx):  # type: ignore[no-untyped-def]
        original_stage(ctx)
        ctx.topology = mock_topo
        ctx.topology_id = getattr(mock_topo, "id", "")
        return ctx

    pipeline._stage_select_topology = _patched  # type: ignore[method-assign]
    try:
        await pipeline.run(task, budget_usd=3.0)
    finally:
        pipeline._stage_select_topology = original_stage  # type: ignore[method-assign]


# ── Memory pillar logs ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_memory_write_gate_fires_logs_decision(caplog: pytest.LogCaptureFixture) -> None:
    """After a pipeline.run(), write_gate.evaluate() must emit an INFO log with
    decision=(persist|abstain) plus the 5-signal breakdown. Without this log
    the Memory pillar is invisible during benchmark runs."""
    caplog.set_level(logging.INFO, logger="sage.memory.write_gate")
    caplog.set_level(logging.INFO, logger="sage.pipeline")
    caplog.set_level(logging.INFO, logger="sage.phases.act")

    # S1 path: ctx.topology=None -> single-agent bypass -> wire gate onto spy loop.
    pipeline, _engine, _spy = _build_pipeline(system=1)
    await pipeline.run("Write a small helper function", budget_usd=3.0)

    gate_fired = [
        r for r in caplog.records
        if "memory.write_gate.fired" in r.getMessage()
    ]
    assert gate_fired, (
        "No memory.write_gate.fired log line emitted during pipeline.run(). "
        "Expected RustCompositeWriteGate.evaluate() to log its decision on "
        f"every call. caplog messages: {[r.getMessage() for r in caplog.records]}"
    )
    # Structured format: decision=persist|abstain plus key=value pairs
    msg = gate_fired[0].getMessage()
    assert "decision=" in msg, (
        f"write_gate log missing decision=persist|abstain: {msg!r}"
    )
    # Salience breakdown fields should be present (from WriteDecision.signal_breakdown)
    assert "salience=" in msg or "score=" in msg, (
        f"write_gate log missing salience score: {msg!r}"
    )


@pytest.mark.asyncio
async def test_memory_smmu_tier_transition_logs_on_compact(caplog: pytest.LogCaptureFixture) -> None:
    """S-MMU STM -> Arrow compaction must log memory.smmu.tier_transition.
    This is the only structured signal that the S-MMU paging is even running
    on a given task."""
    caplog.set_level(logging.INFO, logger="sage.pipeline")

    # Wire a minimal working_memory spy that tracks events + compaction
    class _WM:
        def __init__(self) -> None:
            self._events = 0
            self.compacted = 0

        def add_event(self, event_type: str, content: str) -> None:
            self._events += 1

        def event_count(self) -> int:
            return self._events

        def compact_to_arrow(self) -> None:
            self.compacted += 1
            self._events = 0

    wm = _WM()
    pipeline, engine, _spy = _build_pipeline(system=1)
    pipeline.working_memory = wm
    await pipeline.run("Compute a simple sum", budget_usd=3.0)

    assert wm.compacted >= 1, (
        f"compact_to_arrow not triggered (events={wm.event_count()}), "
        "test can't prove logging"
    )
    smmu_logs = [
        r for r in caplog.records
        if "memory.smmu.tier_transition" in r.getMessage()
    ]
    assert smmu_logs, (
        "No memory.smmu.tier_transition log emitted despite compact_to_arrow "
        "firing. Pipeline._record_to_memory must log the STM -> Arrow "
        f"transition. caplog: {[r.getMessage() for r in caplog.records]}"
    )
    msg = smmu_logs[0].getMessage()
    assert "from=stm" in msg, f"smmu log missing from=stm: {msg!r}"
    assert "to=arrow" in msg, f"smmu log missing to=arrow: {msg!r}"


@pytest.mark.asyncio
async def test_memory_consolidation_fired_logs_at_interval(caplog: pytest.LogCaptureFixture) -> None:
    """MemoryConsolidator.consolidate() must emit memory.consolidation.fired
    with processed/entities/edges counts, even when the pass was a no-op
    (zero processed). Without the log, we cannot verify consolidation ran
    at the scheduled interval."""
    caplog.set_level(logging.INFO, logger="sage.pipeline")

    class _Consolidator:
        async def consolidate(self) -> Any:
            from sage.memory.consolidator import ConsolidationResult
            return ConsolidationResult(processed=3, entities_added=5, causal_edges_added=2)

    pipeline, engine, _spy = _build_pipeline(system=1)
    pipeline.consolidator = _Consolidator()

    # Consolidation fires on every CONSOLIDATION_INTERVAL_STEPS-th task. Rather
    # than running 10 tasks, advance the counter so the next run triggers.
    from sage.constants import CONSOLIDATION_INTERVAL_STEPS
    pipeline._task_count = CONSOLIDATION_INTERVAL_STEPS - 1

    await pipeline.run("Another simple task", budget_usd=3.0)

    cons_logs = [
        r for r in caplog.records
        if "memory.consolidation.fired" in r.getMessage()
    ]
    assert cons_logs, (
        "No memory.consolidation.fired log emitted at scheduled interval. "
        f"caplog: {[r.getMessage() for r in caplog.records]}"
    )
    msg = cons_logs[0].getMessage()
    assert "processed=3" in msg, f"consolidation log missing processed=3: {msg!r}"
    assert "entities=5" in msg, f"consolidation log missing entities=5: {msg!r}"


@pytest.mark.asyncio
async def test_memory_archive_grow_logs_on_cell_delta(caplog: pytest.LogCaptureFixture) -> None:
    """When record_outcome() grows the MAP-Elites archive, pipeline must log
    memory.archive.grow with cell count + delta + topology id. This is the
    observability the H4 (dc51976) bypass would have caught."""
    caplog.set_level(logging.INFO, logger="sage.pipeline")

    # Inject a 1-node mock topology into Stage 2 so LEARN sees ctx.topology
    # and fires record_outcome against our mock engine.
    pipeline, engine, _spy = _build_pipeline(system=2)
    await _run_with_injected_topology(pipeline, engine, "Write a quicksort implementation")

    # First call grew cells 0 → 1
    assert engine.archive_cell_count() == 1

    archive_grow = [
        r for r in caplog.records
        if "memory.archive.grow" in r.getMessage()
    ]
    assert archive_grow, (
        "No memory.archive.grow log emitted despite archive_cell_count "
        "growing 0 → 1. Pipeline must read archive_cell_count before + after "
        "record_outcome and log on positive delta. caplog: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    msg = archive_grow[0].getMessage()
    assert "cells=" in msg, f"archive_grow log missing cells=: {msg!r}"
    assert "delta=" in msg, f"archive_grow log missing delta=: {msg!r}"


# ── Evolution pillar logs ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evolution_should_evolve_logs_false_decision(caplog: pytest.LogCaptureFixture) -> None:
    """Every call to engine.should_evolve() — even when it returns False —
    must emit an INFO log. Today the pipeline only logs the True branch
    ('Online evolution fired'); False branches are invisible, so we cannot
    tell if should_evolve() was even called on a given task."""
    caplog.set_level(logging.INFO, logger="sage.pipeline")

    pipeline, engine, _spy = _build_pipeline(should_evolve_ret=False, system=2)
    await _run_with_injected_topology(pipeline, engine, "Solve a trivial addition task")

    # Sanity: should_evolve() was actually invoked
    assert engine.should_evolve_calls >= 1, (
        "engine.should_evolve() not called — pipeline wiring broken, "
        "test can't prove logging"
    )
    assert engine.evolve_calls == 0, "evolve() should NOT fire when should_evolve=False"

    should_evo_logs = [
        r for r in caplog.records
        if "evolution.should_evolve.decision" in r.getMessage()
    ]
    assert should_evo_logs, (
        "No evolution.should_evolve.decision log emitted. Every should_evolve() "
        "call must be logged, regardless of the bool returned. caplog: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    msg = should_evo_logs[0].getMessage()
    assert "decision=false" in msg.lower() or "decision=False" in msg, (
        f"should_evolve log must include decision=false for False branch: {msg!r}"
    )
    # Archive context should travel with the log so ops can correlate
    assert "cells=" in msg, f"should_evolve log missing cells=: {msg!r}"


@pytest.mark.asyncio
async def test_evolution_evolve_called_logs_when_triggered(caplog: pytest.LogCaptureFixture) -> None:
    """When should_evolve() returns True and evolve() fires, the log line must
    use the evolution.evolve.called taxonomy (not a free-form message), so
    post-run log analysis can count occurrences with a stable regex."""
    caplog.set_level(logging.INFO, logger="sage.pipeline")

    pipeline, engine, _spy = _build_pipeline(should_evolve_ret=True, system=2)
    await _run_with_injected_topology(pipeline, engine, "Optimize a sorting routine")

    assert engine.evolve_calls == 1, (
        f"evolve() should fire once when should_evolve=True, got "
        f"{engine.evolve_calls}"
    )

    evolve_called = [
        r for r in caplog.records
        if "evolution.evolve.called" in r.getMessage()
    ]
    assert evolve_called, (
        "No evolution.evolve.called log emitted despite evolve() firing. "
        f"caplog: {[r.getMessage() for r in caplog.records]}"
    )
    msg = evolve_called[0].getMessage()
    assert "pop_size=" in msg, f"evolve.called log missing pop_size=: {msg!r}"
    assert "generations=" in msg, f"evolve.called log missing generations=: {msg!r}"


# ── AdaptiveMutator log (unit test — not pipeline-exercised) ───────────────


def test_evolution_mutator_update_logs_posterior_update(caplog: pytest.LogCaptureFixture) -> None:
    """AdaptiveMutator.record() must log Thompson-sampling posterior updates.

    Note: AdaptiveMutator is NOT invoked on the pipeline runtime path in
    production today (no call sites outside llm_mutator.py itself). This test
    exercises the log hook directly so the observability is wired for when
    the offline evolution training path is re-activated. See commit message
    for details.
    """
    caplog.set_level(logging.INFO, logger="sage.evolution.llm_mutator")

    from sage.evolution.llm_mutator import AdaptiveMutator
    mutator = AdaptiveMutator(tiers=["budget", "reasoner"])
    mutator.record("budget", improved=True)
    mutator.record("reasoner", improved=False)

    updates = [
        r for r in caplog.records
        if "evolution.mutator.update" in r.getMessage()
    ]
    assert len(updates) >= 2, (
        "Expected 2 evolution.mutator.update log lines (one per record()); "
        f"got {len(updates)}. caplog: {[r.getMessage() for r in caplog.records]}"
    )
    combined = " ".join(u.getMessage() for u in updates)
    assert "tier=budget" in combined, f"missing tier=budget in {combined!r}"
    assert "improved=true" in combined.lower(), (
        f"missing improved=true flag in {combined!r}"
    )
    assert "success_rate=" in combined or "alpha=" in combined, (
        f"missing posterior stats in {combined!r}"
    )
