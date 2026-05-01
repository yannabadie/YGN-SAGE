"""In-run logging for Memory, Evolution, and Topology pillars.

The Memory / Evolution / Topology pillars emitted only boot-time signal
before the 2026-04-21 logging pass. Nothing was captured during a benchmark
run so gains/regressions could not be attributed to pillar behavior. This
module pins the structured log lines so future smokes show per-task pillar
activity:

  Memory pillar:
    memory.write_gate.fired             -- RustCompositeWriteGate decision
    memory.write_gate.skipped           -- gate bypassed (short content, etc.)
    memory.smmu.tier_transition         -- STM chunk compacted to Arrow
    memory.archive.grow                 -- MAP-Elites archive cell count delta
    memory.consolidation.fired          -- episodic -> semantic consolidation
    memory.semantic.query               -- SemanticMemory.get_context_for() read
    memory.episodic.query               -- EpisodicMemory.search() read
    memory.causal.query                 -- CausalMemory.get_context_for() / chain

  Evolution pillar:
    evolution.should_evolve.decision    -- every Rust should_evolve() call
    evolution.evolve.called             -- when evolve() actually fires
    evolution.mutator.update            -- AdaptiveMutator.record() posterior update

  Topology pillar:
    topology.edges                      -- DAG adjacency list (truncated >20)
    topology.source                     -- 6-path attribution + confidence

Format: key=value pairs with %-formatting, one line per event, INFO level.
All log lines must be parseable by a simple regex.
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from sage.pipeline import CognitiveOrchestrationPipeline


@pytest.fixture(autouse=True)
def _legacy_oracle_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests exercise the legacy pipeline path. Apply SAGE_ORACLE=0 so the
    cycle-7 default-on flip does not change expectations here."""
    monkeypatch.setenv("SAGE_ORACLE", "0")


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
        # Simulate the Rust-side op-name buffer being populated by evolve().
        # The real Rust impl (topology/engine.rs) pushes one entry per inner
        # per-child mutation attempt into last_applied_ops; we mirror that
        # with fixed ops so the Python log-drain hook can be unit-tested.
        self._last_ops_buffer = ["add_node", "swap_model", "mutate_prompt"]

    def drain_last_applied_ops(self) -> list[str]:
        buf = getattr(self, "_last_ops_buffer", [])
        self._last_ops_buffer = []
        return buf


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
async def test_evolution_mutation_applied_logs_per_child(caplog: pytest.LogCaptureFixture) -> None:
    """After engine.evolve() fires, pipeline must drain Rust's per-child
    op-name buffer and emit one evolution.mutation.applied log per attempt.

    The Rust `TopologyEngine.drain_last_applied_ops()` (pyo3_wrappers.rs)
    surfaces the operator Thompson-sampled for every mutation inside
    evolve()'s inner loop. The mock engine above returns three hard-coded
    ops (add_node, swap_model, mutate_prompt) so the pipeline must emit
    three structured log lines with real op names (not UNKNOWN).
    """
    caplog.set_level(logging.INFO, logger="sage.pipeline")

    pipeline, engine, _spy = _build_pipeline(should_evolve_ret=True, system=2)
    await _run_with_injected_topology(pipeline, engine, "Evolve a topology")

    assert engine.evolve_calls == 1, "evolve() should fire once"

    mutation_logs = [
        r for r in caplog.records
        if "evolution.mutation.applied" in r.getMessage()
    ]
    assert len(mutation_logs) == 3, (
        "Expected 3 evolution.mutation.applied log lines (one per child "
        f"mutation from the mock ops buffer); got {len(mutation_logs)}. "
        f"caplog: {[r.getMessage() for r in caplog.records]}"
    )
    combined = " ".join(r.getMessage() for r in mutation_logs)
    # Each of the 3 hard-coded op names must appear verbatim — proves we
    # emit the real Rust OPERATOR_NAMES strings, not a stub.
    for expected_op in ("add_node", "swap_model", "mutate_prompt"):
        assert f"op={expected_op}" in combined, (
            f"expected op={expected_op} somewhere in mutation logs; got: "
            f"{combined!r}"
        )
    # Structured fields: parent_cell, child_hash, tier must all appear.
    first_msg = mutation_logs[0].getMessage()
    assert "parent_cell=" in first_msg, (
        f"mutation log missing parent_cell=: {first_msg!r}"
    )
    assert "child_hash=" in first_msg, (
        f"mutation log missing child_hash=: {first_msg!r}"
    )
    assert "tier=" in first_msg, f"mutation log missing tier=: {first_msg!r}"


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


# ── Topology pillar logs ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_topology_edges_logs_adjacency_list(caplog: pytest.LogCaptureFixture) -> None:
    """Stage 2 must emit a topology.edges log line with an adjacency list.

    A full topology produces an edge list that reveals the DAG structure
    (not just node count). Post-run analysis can then attribute pass-rate
    differences to topology shape, not just template name.

    We run pipeline.run() without injected topology so the real sequential
    template (3 nodes, 2 edges: 0->1, 1->2) fires through Stage 2 and the
    structure log captures it. Stage 4 will later fail because the
    _MockLLMProvider isn't a real provider, but that happens AFTER the
    Stage 2 log — which is all this test cares about.
    """
    caplog.set_level(logging.INFO, logger="sage.pipeline")

    pipeline, engine, _spy = _build_pipeline(system=2)
    # Safe to let pipeline.run raise from Stage 4; we only assert on Stage 2 logs.
    try:
        await pipeline.run("Multi-node task", budget_usd=3.0)
    except Exception:
        pass

    edge_logs = [
        r for r in caplog.records if "topology.edges" in r.getMessage()
    ]
    assert edge_logs, (
        "No topology.edges log emitted. Stage 2 must log an adjacency list "
        f"alongside the template line. caplog: {[r.getMessage() for r in caplog.records]}"
    )
    msg = edge_logs[0].getMessage()
    assert "nodes=3" in msg, f"topology.edges missing nodes=3: {msg!r}"
    assert "edges=" in msg, f"topology.edges missing edges=: {msg!r}"
    # Adjacency tuple pair present in the rendered edges list
    assert "(0, 1)" in msg or "(0,1)" in msg, (
        f"topology.edges missing (0, 1) tuple: {msg!r}"
    )


def test_topology_edges_truncates_huge_adjacency(caplog: pytest.LogCaptureFixture) -> None:
    """_log_topology_structure() must cap adjacency at 20 entries and emit total=N.

    Unit-level test: call the helper directly with a 50-edge mock so we
    don't need to exercise the full pipeline (which would fail at Stage 4
    when the TopologyExecutor rejects non-Rust graph objects).
    """
    caplog.set_level(logging.INFO, logger="sage.pipeline")

    class _BigTopo:
        def __init__(self) -> None:
            self.id = "big-topo-ulid-12345"
            self.template_type = "selfmoa"
            self._edges = [(i, i + 1, "control") for i in range(50)]

        def node_count(self) -> int:
            return 51

        def edge_count(self) -> int:
            return 50

        def get_edges(self) -> list[tuple[int, int, str]]:
            return list(self._edges)

    pipeline, _engine, _spy = _build_pipeline(system=2)
    pipeline._log_topology_structure(_BigTopo(), source="archive_hit", confidence=0.90)

    edge_logs = [
        r for r in caplog.records if "topology.edges" in r.getMessage()
    ]
    assert edge_logs, "No topology.edges log emitted for big topology"
    msg = edge_logs[0].getMessage()
    assert "nodes=51" in msg, f"missing nodes=51: {msg!r}"
    # Truncation marker
    assert "total=50" in msg, (
        f"topology.edges must emit total=50 when truncated: {msg!r}"
    )
    # Log line bounded — single line, not huge
    assert len(msg) <= 500, (
        f"topology.edges log must be <= 500 chars for large topos, got {len(msg)}"
    )


@pytest.mark.asyncio
async def test_topology_source_logs_attribution_template_branch(caplog: pytest.LogCaptureFixture) -> None:
    """Template branch (no engine call) must log topology.source=dag_template.

    The dominant production path (per April 20 plan 1.4 comment) must be
    visible to source-attribution analysis.
    """
    caplog.set_level(logging.INFO, logger="sage.pipeline")

    pipeline, engine, _spy = _build_pipeline(system=2)

    # _MockRouter omega=0 → select_macro_topology returns "sequential" which
    # IS in the template shortcut list (pipeline.py:510-522). Stage 4 may
    # raise later (mock LLM provider), which is after Stage-2 logs fired.
    try:
        await pipeline.run("Simple sequential task", budget_usd=3.0)
    except Exception:
        pass

    source_logs = [
        r for r in caplog.records if "topology.source" in r.getMessage()
    ]
    assert source_logs, (
        "No topology.source log emitted from template branch. "
        f"caplog: {[r.getMessage() for r in caplog.records]}"
    )
    msg = source_logs[0].getMessage()
    assert "source=dag_template" in msg, (
        f"Template branch topology.source must attribute to dag_template: {msg!r}"
    )
    assert "archive_hit=false" in msg, (
        f"template branch source log must emit archive_hit=false: {msg!r}"
    )


def test_topology_source_logs_attribution_engine_branch(caplog: pytest.LogCaptureFixture) -> None:
    """Engine branch must log topology.source from PyGenerateResult.source().

    Unit-level: call the helper with a mock result-like topology. The
    engine-branch code path in _stage_select_topology extracts source/
    confidence from result (PyGenerateResult.source() returns str,
    .confidence() returns float) and forwards to _log_topology_structure.
    """
    caplog.set_level(logging.INFO, logger="sage.pipeline")

    class _EngineTopo:
        def __init__(self) -> None:
            self.id = "engine-topo-ulid"
            self.template_type = "selfmoa"

        def node_count(self) -> int:
            return 4

        def edge_count(self) -> int:
            return 3

        def get_edges(self) -> list[tuple[int, int, str]]:
            return [(0, 1, "control"), (0, 2, "control"), (1, 3, "message")]

    pipeline, _engine, _spy = _build_pipeline(system=2)
    pipeline._log_topology_structure(
        _EngineTopo(), source="archive_hit", confidence=0.85,
    )

    source_logs = [
        r for r in caplog.records if "topology.source" in r.getMessage()
    ]
    assert source_logs, (
        "No topology.source log emitted from engine branch. "
        f"caplog: {[r.getMessage() for r in caplog.records]}"
    )
    msg = source_logs[0].getMessage()
    assert "source=archive_hit" in msg, (
        f"topology.source missing source=archive_hit: {msg!r}"
    )
    assert "confidence=0.850" in msg, (
        f"topology.source missing confidence=0.850: {msg!r}"
    )
    assert "archive_hit=true" in msg, (
        f"engine archive_hit path must emit archive_hit=true flag: {msg!r}"
    )


# ── Memory read logs ────────────────────────────────────────────────────────


def test_memory_semantic_query_logs_hits(caplog: pytest.LogCaptureFixture) -> None:
    """SemanticMemory.get_context_for() must log memory.semantic.query.

    Captures read activity when an agent's perceive phase injects semantic
    context — currently invisible.
    """
    caplog.set_level(logging.INFO, logger="sage.memory.semantic")

    from sage.memory.memory_agent import ExtractionResult
    from sage.memory.semantic import SemanticMemory

    sm = SemanticMemory(max_relations=100, max_context_lines=10)
    sm.add_extraction(ExtractionResult(
        entities=["quicksort", "array"],
        relationships=[("quicksort", "sorts", "array")],
    ))
    sm.get_context_for("Implement quicksort on an array")

    query_logs = [
        r for r in caplog.records if "memory.semantic.query" in r.getMessage()
    ]
    assert query_logs, (
        "No memory.semantic.query log emitted. "
        f"caplog: {[r.getMessage() for r in caplog.records]}"
    )
    msg = query_logs[0].getMessage()
    assert "hits=" in msg, f"missing hits= in {msg!r}"


def test_memory_episodic_query_logs_hits(caplog: pytest.LogCaptureFixture) -> None:
    """EpisodicMemory.search() must log memory.episodic.query."""
    import asyncio

    caplog.set_level(logging.INFO, logger="sage.memory.episodic")

    from sage.memory.episodic import EpisodicMemory

    em = EpisodicMemory()  # in-memory
    asyncio.run(em.store("step-1", "quicksort implementation note"))
    asyncio.run(em.store("step-2", "another memory item"))
    results = asyncio.run(em.search("quicksort"))
    assert results, "Episodic search should find at least one match"

    query_logs = [
        r for r in caplog.records if "memory.episodic.query" in r.getMessage()
    ]
    assert query_logs, (
        "No memory.episodic.query log emitted. "
        f"caplog: {[r.getMessage() for r in caplog.records]}"
    )
    msg = query_logs[-1].getMessage()  # last search
    assert "hits=" in msg, f"missing hits= in {msg!r}"


def test_memory_causal_query_logs_hits(caplog: pytest.LogCaptureFixture) -> None:
    """CausalMemory.get_context_for() must log memory.causal.query."""
    caplog.set_level(logging.INFO, logger="sage.memory.causal")

    from sage.memory.causal import CausalMemory

    cm = CausalMemory()
    cm.add_entity("file_a")
    cm.add_entity("file_b")
    cm.add_causal_edge("file_a", "file_b", cause_type="modifies")
    cm.get_context_for("refactor file_a")

    query_logs = [
        r for r in caplog.records if "memory.causal.query" in r.getMessage()
    ]
    assert query_logs, (
        "No memory.causal.query log emitted. "
        f"caplog: {[r.getMessage() for r in caplog.records]}"
    )
    msg = query_logs[0].getMessage()
    assert "hits=" in msg, f"missing hits= in {msg!r}"


def test_memory_causal_chain_logs_hits(caplog: pytest.LogCaptureFixture) -> None:
    """CausalMemory.get_causal_chain() must log memory.causal.query scope=chain."""
    caplog.set_level(logging.INFO, logger="sage.memory.causal")

    from sage.memory.causal import CausalMemory

    cm = CausalMemory()
    cm.add_entity("A")
    cm.add_entity("B")
    cm.add_entity("C")
    cm.add_causal_edge("A", "B")
    cm.add_causal_edge("B", "C")

    chain = cm.get_causal_chain("A")
    assert chain == ["A", "B", "C"]

    query_logs = [
        r for r in caplog.records if "memory.causal.query" in r.getMessage()
    ]
    assert query_logs, "No memory.causal.query log emitted for chain traversal"
    assert any("scope=chain" in r.getMessage() for r in query_logs), (
        f"missing scope=chain tag; logs: {[r.getMessage() for r in query_logs]}"
    )


# ── Write-gate skip observability (Gap 5 investigation) ─────────────────────


def test_tool_heavy_turn_opens_episodic_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A turn with short prose (< 100 chars) but tool_calls MUST now pass
    the episodic length check (previously skipped silently).

    This is the fix for the write_gate=0 finding (0bcb92b): SWE-bench
    turns are mostly tool_calls with minimal prose, and the prior
    `len(content) > 100` gate on episodic persistence meant exploration
    knowledge was never written to memory. Fix: OR-in `bool(tool_names)`.
    """
    from sage.phases import act as _act  # noqa: F401 — import smoke
    import re
    src = (
        __import__("pathlib").Path(__file__)
        .resolve().parents[1]
        / "src/sage/bench/../phases/act.py"
    ).resolve()
    text = src.read_text(encoding="utf-8")
    # Fingerprint check: the episodic guard must accept either prose
    # length OR tool activity. Catches accidental revert.
    assert re.search(
        r"len\(content\)\s*>\s*100\s+or\s+bool\(tool_names\)",
        text,
    ), (
        "act.py episodic guard regressed to prose-only length check — "
        "tool-heavy turns will skip episodic memory again"
    )
    # Stored payload must switch to turn_signal when tool_names exist.
    assert "turn_signal[:500] if tool_names else content[:500]" in text, (
        "act.py episodic_payload composition regressed — lost the "
        "tool_signal marker in stored content"
    )


def test_memory_write_gate_skipped_short_content(caplog: pytest.LogCaptureFixture) -> None:
    """When act phase has short content and tool calls, the gate path is never
    entered (content<100 for episodic, <50 for semantic). We MUST log the
    skip so post-run analysis can distinguish "gate never fired" from
    "gate evaluated and abstained". Without this, 0 gate fires on a
    tool-heavy SWE-bench run looks identical to gate-not-wired.
    """
    caplog.set_level(logging.INFO, logger="sage.memory.write_gate")

    from sage.memory.write_gate import log_write_gate_skipped
    log_write_gate_skipped(
        reason="content_too_short",
        content_len=42,
        has_tool_calls=True,
        source_tier="fast",
    )

    skip_logs = [
        r for r in caplog.records if "memory.write_gate.skipped" in r.getMessage()
    ]
    assert skip_logs, (
        "log_write_gate_skipped() must emit memory.write_gate.skipped. "
        f"caplog: {[r.getMessage() for r in caplog.records]}"
    )
    msg = skip_logs[0].getMessage()
    assert "reason=content_too_short" in msg, f"missing reason=: {msg!r}"
    assert "content_len=42" in msg, f"missing content_len=: {msg!r}"
    assert "has_tool_calls=true" in msg.lower(), f"missing has_tool_calls=: {msg!r}"
