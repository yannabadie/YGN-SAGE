"""P9 cycle-11 test #2 (ADR-015 acceptance gate): oracle trainable gates side-effects.

Locks runtime-integrity-ledger.md invariant 2 ("Oracle evidence", cycle-7
default-on flip ``128e1b89``) and ADR-015 §"Contracts that MUST be
preserved" item #2 / characterization test #2.

The OracleStack ``verdict.trainable`` flag is the single authoritative
gate on **all four learning side-effects** in the pipeline. Cycle-7
crystallized this around the SAGE_ORACLE flip; cycle-8 R6.1c added
schema/redaction binding so the verdict's evidence is verifiable. The
contract this test locks is one half of the broader directive:

    "Any label that authorizes a side-effect or learning decision MUST
     be bound to verified content, schema, provenance, or executable
     proof." (CLAUDE.md directive #9)

Specifically, when ``OracleVerdict.trainable=False``, none of the
following must fire — *because the oracle has not certified that this
run produced trainable evidence*:

    1. SystemRouter.record_outcome_checked  (bandit posterior update)
    2. engine.record_outcome                (MAP-Elites archive insert)
    3. engine.should_evolve / engine.evolve (online evolution)
    4. consolidator.consolidate             (training-memory inter-tier)

Conversely, when ``trainable=True`` with valid evidence, all four MUST
fire (subject to their own per-call gate conditions: bandit needs a
pending decision_id, MAP-Elites needs a topology_id, evolution needs
``should_evolve()`` to return True, consolidation runs every
``CONSOLIDATION_INTERVAL_STEPS`` tasks).

Why this test is invariant-framed
=================================
The existing ``test_pipeline_bandit_causality.py`` covers per-mechanism
behavior (off-policy mismatches, multi-node ambiguity, abstain
cancellation). This test is **invariant-framed**: across all four
side-effect surfaces, ``verdict.trainable`` is the discriminator. If
cycle-12 phase 2 splits ``_stage_learn`` into ``pipeline_v2/learn.py``,
the gate must remain the single decision point — moving any of the
four side-effects out from under it is an ADR-015 contract change
that this test will catch.

Cycle-7 historical note: the test does not assert that
``SAGE_ORACLE`` env-var changes the gate semantics. Cycle-7's default-
on flip means oracle path is active by default; the kill-switch
explicitly bypasses the gate (legacy quality_estimator path takes
over). The test runs with ``SAGE_ORACLE=1`` explicitly to be
defensive against future default flips.

Scope of "training-memory" (cgpro VERIFY follow-up 2026-05-05)
==============================================================
ADR-015 #2 says the gate blocks "training_memory" updates. There is
no explicit ``training_memory.store`` symbol in pipeline.py today —
the closest match is the inter-tier consolidation step that this
test covers via ``consolidator.consolidate()``. cgpro VERIFY noted
this terminology should be made explicit:

  - ``consolidator.consolidate()`` (inter-tier semantic/causal
    promotion) IS gated by ``allow_training_updates`` and IS what
    this test checks.

  - Working/episodic trace memory writes (e.g. via
    ``_record_to_memory(...)`` BEFORE ``_stage_learn``) are tagged
    with ``is_training_evidence=verdict.trainable`` metadata but
    are NOT blocked under ``trainable=False`` — they're audit /
    provenance storage, not training updates. This test does NOT
    assert anything about those calls; doing so would be incorrect
    per the ledger's distinction between "trace memory writes" and
    "training-memory updates".

If a future commit introduces a real ``training_memory.store``
component (separate from consolidation), this test must be extended
to cover it as a 5th side-effect.
"""
from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sage.pipeline import (
    CognitiveOrchestrationPipeline as Pipeline,
    PipelineContext,
)
from sage.pipeline_v2.learn import learn
from sage.runtime.oracle.verdict import EvidenceRef, OracleVerdict


def _make_untrainable_verdict() -> OracleVerdict:
    """Construct a valid trainable=False verdict.

    The dataclass __post_init__ enforces verdict_source='abstain' +
    quality_label='unknown' for trainable=False. score=None is
    required (no concrete signal). Reason codes can be empty in the
    untrainable branch; we add one for realism.
    """
    return OracleVerdict(
        trainable=False,
        verdict_source="abstain",
        quality_label="unknown",
        score=None,
        confidence=0.0,
        reason_codes=("oracle_abstain_test",),
        evidence=(),
    )


def _make_trainable_verdict() -> OracleVerdict:
    """Construct a valid trainable=True verdict with concrete score + evidence.

    score=0.85, quality_label='pass' chosen to fall above any
    plausible cascade threshold (so the bandit/MAP-Elites positive
    paths fire without being intercepted by a low-quality cascade).
    """
    return OracleVerdict(
        trainable=True,
        verdict_source="exact",
        quality_label="pass",
        score=0.85,
        confidence=1.0,
        reason_codes=("exact_test_pass",),
        evidence=(EvidenceRef(run_id="01ORACLETESTRUN0000000001"),),
    )


class _SideEffectCapture:
    """Records calls to the four learning side-effects.

    Each callable on this object is an instance method or property that
    increments a counter when invoked. The tests below assert
    ``call_count == 0`` for the negative case and ``> 0`` for the
    positive control.
    """

    def __init__(self) -> None:
        self.bandit_record_calls: list[tuple[Any, ...]] = []
        self.map_elites_calls: list[tuple[Any, ...]] = []
        self.should_evolve_calls = 0
        self.evolve_calls = 0
        self.consolidate_calls = 0
        self._archive_cells = 0  # archive_cell_count return value

    def install_on(self, pipeline: Pipeline) -> None:
        """Wire all four capture points to a Pipeline instance."""
        # 1. SystemRouter (bandit posterior). Pipeline reads via
        # ``self._rust_router.record_outcome_checked``. We give it a
        # MagicMock with hasattr returning True so the
        # ``_record_bandit_outcome_checked`` recorder branch fires
        # when (and only when) the gate allows.
        capture = self

        class _StubRustRouter:
            def record_outcome_checked(
                self,
                decision_id: str,
                executed_model_id: str,
                executed_template: str,
                quality: float,
                cost: float,
                latency_ms: float,
            ) -> None:
                capture.bandit_record_calls.append(
                    (decision_id, executed_model_id, executed_template,
                     quality, cost, latency_ms)
                )

            def cancel_bandit_decision(self, decision_id: str) -> bool:
                return True

        pipeline._rust_router = _StubRustRouter()

        # 2 + 3. TopologyEngine: capture record_outcome (MAP-Elites)
        # and should_evolve / evolve (online evolution).
        class _StubEngine:
            def __init__(self) -> None:
                # Track via the outer capture so both engine instances
                # share counters (though we install one per test).
                pass

            def record_outcome(
                self,
                topology_id: str,
                task: str,
                keywords: list[str],
                task_embedding: Any,
                quality: float,
                cost: float,
                latency_ms: float,
            ) -> None:
                capture.map_elites_calls.append(
                    (topology_id, quality, cost, latency_ms)
                )

            def archive_cell_count(self) -> int:
                # Stable cell count → cells_after == cells_before so
                # the "archive grew" log branch is a no-op (we don't
                # care; we only care about whether record_outcome was
                # called).
                return capture._archive_cells

            def archive_coverage(self) -> float:
                return 0.0

            def should_evolve(self, min_outcomes: int, cooldown: int) -> bool:
                capture.should_evolve_calls += 1
                # Return True so a permitted call also fires evolve(),
                # which is the strongest positive-case assertion we
                # can make end-to-end.
                return True

            def evolve(self, pop_size: int, generations: int) -> None:
                capture.evolve_calls += 1

            def drain_last_applied_ops(self) -> list[str]:
                return []  # no per-mutation logging

        pipeline.engine = _StubEngine()

        # 4. Inter-tier consolidation. Pipeline awaits
        # ``self.consolidator.consolidate()`` on the
        # ``_task_count % CONSOLIDATION_INTERVAL_STEPS == 0`` schedule.
        consolidate_mock = AsyncMock()

        async def _consolidate_wrapper() -> Any:
            capture.consolidate_calls += 1
            return MagicMock(processed=0, entities_added=0, causal_edges_added=0)

        consolidate_mock.side_effect = _consolidate_wrapper
        pipeline.consolidator = MagicMock(consolidate=consolidate_mock)


def _build_learn_stage_pipeline(monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    """Surgical Pipeline stub for direct ``_stage_learn`` calls.

    SAGE_ORACLE=1 explicit (defensive against future default flips).
    quality_estimator=None and prm=None so the oracle path is the only
    quality source — matches the cycle-7+ default-on production state.

    Embedder load is short-circuited via a stub class returning
    is_semantic=False — prevents the ~60s HF model download/load
    inside the ``engine.record_outcome`` evolution-feedback branch
    (only reached on the trainable=True path; negative tests never
    hit it).
    """
    monkeypatch.setenv("SAGE_ORACLE", "1")

    # Embedder short-circuit — `_stage_learn` does
    # ``from sage.memory.embedder import Embedder; _embedder = Embedder()``
    # in two branches (evolution feedback at line ~2866 and topology
    # selection elsewhere). Both branches gate downstream work on
    # ``_embedder.is_semantic``. A stub that returns False keeps the
    # branch dormant without attempting to import / load any HF model.
    import sage.memory.embedder as embedder_mod

    class _StubEmbedder:
        is_semantic = False

        def embed(self, text: str) -> list[float]:
            return []  # never reached when is_semantic=False

    monkeypatch.setattr(embedder_mod, "Embedder", _StubEmbedder)

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._llm_tier = ""
    pipeline.controller = None
    pipeline.llm_provider = None
    pipeline.llm_config = None
    pipeline.provider_pool = None
    pipeline.assigner = None
    pipeline.event_bus = None
    pipeline.tool_registry = None
    pipeline._agent_loop = None
    pipeline.write_gate = None
    pipeline.episodic_memory = None
    pipeline.semantic_memory = None
    pipeline.memory_agent = None
    pipeline.causal_memory = None
    pipeline._emit = MagicMock()
    monkeypatch.setattr("sage.pipeline_v2.memory_gate.emit_budget_exceeded", MagicMock())
    monkeypatch.setattr(
        "sage.pipeline_v2.runtime_events.emit_bandit_attribution_mismatch", MagicMock()
    )
    pipeline._on_topology_evolve = None
    pipeline.harness_config = None
    pipeline._harness_patcher = None
    pipeline._agent_loop_bypass_lock = None
    pipeline._agent_loop_bypass_lock_loop = None
    pipeline.quality_estimator = None
    pipeline.prm = None
    pipeline._estimate_topology_cost = MagicMock(return_value=0.0)
    pipeline.router = None
    pipeline._topology_cache = {}
    pipeline._apply_topology_budget_and_cache = MagicMock()
    pipeline._log_topology_structure = MagicMock()
    pipeline._last_routing_decision = None
    pipeline._last_runtime_routing_source = ""
    pipeline._last_runtime_routing_confidence = None
    pipeline._last_runtime_routing_model_id = ""
    pipeline.bandit = None
    pipeline._task_count = 0  # incremented inside _stage_learn

    return pipeline


def _build_ctx_for_learn(*, oracle_verdict: OracleVerdict) -> PipelineContext:
    """Build a PipelineContext that exercises all four learn-side gates.

    Topology + topology_id present → MAP-Elites would fire on
    trainable. executed_template='single_agent' + non-empty
    executed_model_id + bandit_decision_id → bandit would fire.
    """
    topology = MagicMock()
    topology.id = "test-topology-id-01"
    topology.node_count = MagicMock(return_value=1)

    ctx = PipelineContext(task="def add(a, b):\n    return a + b")
    ctx.system = 2
    ctx.domain = "code"
    ctx.result = "def add(a, b):\n    return a + b"
    ctx.cost = 0.001
    ctx.latency_ms = 100.0
    ctx.topology = topology
    ctx.topology_id = "test-topology-id-01"
    ctx.bandit_decision_id = "d-test-oracle-gate"
    ctx.bandit_model_id = "test-model"
    ctx.bandit_template = "single_agent"
    ctx.executed_model_id = "test-model"
    ctx.executed_template = "single_agent"
    ctx.executed_model_ids = ["test-model"]
    ctx.oracle_verdict = oracle_verdict
    return ctx


# ─────────────────────────────────────────────────────────────────
# Negative cases: trainable=False blocks all four side-effects
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_oracle_trainable_false_blocks_bandit_record_outcome_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """trainable=False → SystemRouter.record_outcome_checked is NOT called.

    Invariant 2 (oracle evidence). When oracle is on AND verdict is
    untrainable, the bandit posterior must not update. Otherwise the
    contextual bandit learns from off-policy / unverified data, which
    is exactly the class of bug A14 was designed to close.
    """
    pipeline = _build_learn_stage_pipeline(monkeypatch)
    capture = _SideEffectCapture()
    capture.install_on(pipeline)
    ctx = _build_ctx_for_learn(oracle_verdict=_make_untrainable_verdict())

    await learn(pipeline, ctx)

    assert capture.bandit_record_calls == [], (
        f"Untrainable verdict allowed bandit posterior update: "
        f"{capture.bandit_record_calls!r}. Invariant 2 violated — "
        f"the bandit must not learn from oracle-abstained runs."
    )


@pytest.mark.asyncio
async def test_oracle_trainable_false_blocks_map_elites_record_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """trainable=False → engine.record_outcome (MAP-Elites) is NOT called.

    The MAP-Elites archive growth shapes future topology selection
    (engine.generate path). Letting untrainable runs into the
    archive would bias topology selection on noise.
    """
    pipeline = _build_learn_stage_pipeline(monkeypatch)
    capture = _SideEffectCapture()
    capture.install_on(pipeline)
    ctx = _build_ctx_for_learn(oracle_verdict=_make_untrainable_verdict())

    await learn(pipeline, ctx)

    assert capture.map_elites_calls == [], (
        f"Untrainable verdict allowed MAP-Elites archive insert: "
        f"{capture.map_elites_calls!r}. Topology evolution archive "
        f"would be biased by oracle-abstained runs."
    )


def test_map_elites_guard_names_allow_training_updates_explicitly() -> None:
    """MAP-Elites must not rely on `quality is not None` as an implicit oracle gate."""
    source = inspect.getsource(learn)

    assert (
        "allow_training_updates\n"
        "            and self.engine\n"
        "            and quality is not None\n"
        "            and ctx.topology is not None"
    ) in source


@pytest.mark.asyncio
async def test_oracle_trainable_false_blocks_online_evolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """trainable=False → engine.should_evolve / engine.evolve NOT called.

    Online evolution mutates topologies by sampling Thompson
    operators on the archive. Firing it under an untrainable verdict
    means the mutation is conditioned on a run we don't trust —
    same noise concern as MAP-Elites.
    """
    pipeline = _build_learn_stage_pipeline(monkeypatch)
    capture = _SideEffectCapture()
    capture.install_on(pipeline)
    ctx = _build_ctx_for_learn(oracle_verdict=_make_untrainable_verdict())

    await learn(pipeline, ctx)

    assert capture.should_evolve_calls == 0, (
        f"Untrainable verdict allowed should_evolve() probing: "
        f"{capture.should_evolve_calls} call(s). The gate must "
        f"short-circuit before should_evolve is consulted."
    )
    assert capture.evolve_calls == 0, (
        f"Untrainable verdict allowed evolve() call: "
        f"{capture.evolve_calls} call(s). Defense in depth: even "
        f"if should_evolve fires, evolve must not."
    )


@pytest.mark.asyncio
async def test_oracle_trainable_false_blocks_inter_tier_consolidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """trainable=False → consolidator.consolidate (training-memory) NOT called.

    Inter-tier consolidation moves episodic → semantic → causal. It
    runs every CONSOLIDATION_INTERVAL_STEPS tasks. We pre-set
    _task_count = N-1 so this run lands on the consolidate-fire
    boundary; if the gate didn't block, consolidate would call.
    """
    from sage.constants import CONSOLIDATION_INTERVAL_STEPS

    pipeline = _build_learn_stage_pipeline(monkeypatch)
    capture = _SideEffectCapture()
    capture.install_on(pipeline)
    # Land on the firing boundary: _task_count gets incremented to N
    # inside _stage_learn before the consolidate check.
    pipeline._task_count = CONSOLIDATION_INTERVAL_STEPS - 1
    ctx = _build_ctx_for_learn(oracle_verdict=_make_untrainable_verdict())

    await learn(pipeline, ctx)

    assert capture.consolidate_calls == 0, (
        f"Untrainable verdict allowed inter-tier consolidation: "
        f"{capture.consolidate_calls} call(s). Training-memory "
        f"consolidation must be gated by oracle trainable per "
        f"invariant 2."
    )


# ─────────────────────────────────────────────────────────────────
# Positive control: trainable=True allows all four side-effects
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_oracle_trainable_true_allows_all_four_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: trainable=True with valid evidence → ALL four fire.

    Without this counter-test, the four negative tests above could
    pass vacuously if the gate hard-coded ``False`` (e.g. a refactor
    that accidentally made nothing trainable). This test proves the
    gate is read correctly and the four call sites are reachable.

    Per-side-effect gate conditions (must hold for the call to fire):
      - bandit:        bandit_decision_id non-empty, single-node
                       executed_template, _rust_router has
                       record_outcome_checked. (Set in
                       _build_ctx_for_learn.)
      - MAP-Elites:    ctx.topology non-None, topology_id non-empty,
                       engine.record_outcome attr present. (Set.)
      - evolution:     engine.should_evolve returns True, allow_training_updates
                       is True. (Stub returns True.)
      - consolidation: _task_count % CONSOLIDATION_INTERVAL_STEPS == 0
                       after increment, consolidator non-None. (Set.)
    """
    from sage.constants import CONSOLIDATION_INTERVAL_STEPS

    pipeline = _build_learn_stage_pipeline(monkeypatch)
    capture = _SideEffectCapture()
    capture.install_on(pipeline)
    pipeline._task_count = CONSOLIDATION_INTERVAL_STEPS - 1
    ctx = _build_ctx_for_learn(oracle_verdict=_make_trainable_verdict())

    await learn(pipeline, ctx)

    # 1. Bandit posterior recorded with the trainable score.
    assert len(capture.bandit_record_calls) == 1, (
        f"Trainable verdict did not record bandit outcome: "
        f"{capture.bandit_record_calls!r}. The four gates downstream "
        f"of `quality is not None` should all fire."
    )
    decision_id, model_id, template, quality, cost, latency = (
        capture.bandit_record_calls[0]
    )
    assert decision_id == "d-test-oracle-gate"
    assert quality == pytest.approx(0.85)

    # 2. MAP-Elites archive insert.
    assert len(capture.map_elites_calls) == 1, (
        f"Trainable verdict did not insert into MAP-Elites archive: "
        f"{capture.map_elites_calls!r}."
    )

    # 3. Online evolution probed and (because should_evolve=True) fired.
    assert capture.should_evolve_calls >= 1, (
        f"Trainable verdict did not consult should_evolve(): "
        f"{capture.should_evolve_calls} call(s)."
    )
    assert capture.evolve_calls >= 1, (
        f"should_evolve returned True but evolve() was not called: "
        f"{capture.evolve_calls} call(s)."
    )

    # 4. Inter-tier consolidation fired on the schedule boundary.
    assert capture.consolidate_calls == 1, (
        f"Trainable verdict + schedule boundary did not fire "
        f"consolidation: {capture.consolidate_calls} call(s). The "
        f"task_count modulo arithmetic may be off — pre-call value "
        f"was {CONSOLIDATION_INTERVAL_STEPS - 1}, post-increment "
        f"should land on a fire."
    )
