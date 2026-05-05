"""P9 cycle-11 test #3 (ADR-015 acceptance gate): bandit attribution singleton settle.

Locks runtime-integrity-ledger.md invariant 6 ("Bandit attribution",
A14b cycle-8 closure ``6f23eea4``) and ADR-015 §"Contracts that MUST be
preserved" item #3 / characterization test #3.

The contract: every ``bandit_decision_id`` issued in Stage 0
(``SystemRouter.route_integrated`` at ``pipeline.py:1116``) must be
**settled exactly once** by end-of-run via either
``record_outcome_checked`` OR ``cancel_bandit_decision``, regardless
of the execution path taken in Stage 4. The four execution paths per
ADR-015 #3:

  1. Single-agent bypass (``_is_single_agent_execution(ctx) == True``)
  2. Topology runner (multi-agent) with single executed_model
  3. Topology runner (multi-agent) with multi-node attribution template
     OR multiple executed_model_ids (multi_node_ambiguous)
  4. Oracle abstain (verdict.trainable=False) — gate blocks before
     attribution check fires

These four paths produce different ``ctx`` states at Stage 5 entry,
and the bandit gatekeeping in ``_record_bandit_outcome_checked``
(line 2067-2115) routes each to either a record or a cancel — never
both, never neither.

Why this is invariant-framed (not redundant with bandit_causality)
==================================================================
``test_pipeline_bandit_causality.py`` covers per-mechanism behavior:
controller-upgrade mismatch, off-policy refusal, route_integrated
failure attribution. This file is **invariant-framed**: across all
four execution paths, every issued ``decision_id`` reaches exactly
one settle action.

The cycle-12 phase 2 decomposition will move ``_stage_learn`` into
``pipeline_v2/learn.py`` and the bandit attribution lifecycle into
``pipeline_v2/bandit_attribution.py`` (per ADR-015 module table).
The split is allowed only if this invariant survives byte-identically:
the new module owns Stage-0 → Stage-5 lifecycle, and the singleton
settle is the contract.

Rust SystemRouter side
======================
The Rust ``record_outcome_checked`` raises on a (model_id, template)
mismatch against the pending decision; the Python pipeline catches
and cancels. So in the mismatch path, two SystemRouter methods are
invoked: ``record_outcome_checked`` (which raised + did NOT mutate
the posterior) and ``cancel_bandit_decision`` (which removed the
pending). The end state — bandit posterior side — is settled once.
The tests track BOTH "settle attempts" (record_outcome_checked
calls regardless of outcome) AND "successful settles" (record OR
cancel that mutated state). The invariant is on the latter.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from sage.pipeline import (
    CognitiveOrchestrationPipeline as Pipeline,
    PipelineContext,
)
from sage.runtime.oracle.verdict import OracleVerdict


class _RustRouterCapture:
    """Tracks SystemRouter side-effects per decision_id.

    ``successful_settles[decision_id]`` is a list of ``"record"`` /
    ``"cancel"`` strings — the operations that actually mutated the
    posterior state. The invariant test asserts ``len(...) == 1`` for
    every issued decision_id.

    ``record_attempts`` counts every call to ``record_outcome_checked``
    including the ones that raised. The invariant cares about
    successful_settles, not raw call counts.
    """

    def __init__(self, *, raise_on_record: Exception | None = None) -> None:
        self.pending_decisions: set[str] = set()
        self.successful_settles: dict[str, list[str]] = {}
        self.record_attempts: list[tuple[str, ...]] = []
        self.cancel_calls: list[str] = []
        self._raise_on_record = raise_on_record

    def issue(self, decision_id: str) -> None:
        """Mark a decision_id as pending (issued by Stage 0)."""
        self.pending_decisions.add(decision_id)
        self.successful_settles.setdefault(decision_id, [])

    def record_outcome_checked(
        self,
        decision_id: str,
        executed_model_id: str,
        executed_template: str,
        quality: float,
        cost: float,
        latency_ms: float,
    ) -> Any:
        self.record_attempts.append(
            (decision_id, executed_model_id, executed_template,
             str(quality), str(cost), str(latency_ms))
        )
        if self._raise_on_record is not None:
            # Mismatch path: the call doesn't mutate the posterior;
            # the pipeline must follow up with cancel. Invariant
            # holds because record_attempts ≠ successful_settles.
            raise self._raise_on_record
        # Success: settle by record.
        self.pending_decisions.discard(decision_id)
        self.successful_settles.setdefault(decision_id, []).append("record")
        return MagicMock(status="recorded")

    def cancel_bandit_decision(self, decision_id: str) -> bool:
        self.cancel_calls.append(decision_id)
        was_pending = decision_id in self.pending_decisions
        if was_pending:
            self.pending_decisions.discard(decision_id)
            self.successful_settles.setdefault(decision_id, []).append("cancel")
        return was_pending


def _build_pipeline_for_learn(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raise_on_record: Exception | None = None,
) -> tuple[Pipeline, _RustRouterCapture]:
    """Build a minimal pipeline wired to capture the bandit settle ops.

    ``SAGE_ORACLE`` defaults to "0" so the gate doesn't pre-empt the
    attribution check; oracle-abstain test sets it back to "1".
    Embedder stub matches test #2 to avoid HF model load.
    """
    monkeypatch.setenv("SAGE_ORACLE", "0")

    import sage.memory.embedder as embedder_mod

    class _StubEmbedder:
        is_semantic = False

        def embed(self, text: str) -> list[float]:
            return []

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
    pipeline._emit_budget_exceeded = MagicMock()
    pipeline._emit_bandit_attribution_mismatch = MagicMock()
    pipeline._on_topology_evolve = None
    pipeline.engine = None
    pipeline.harness_config = None
    pipeline._harness_patcher = None
    pipeline._agent_loop_bypass_lock = None
    pipeline._agent_loop_bypass_lock_loop = None
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
    pipeline.consolidator = None
    pipeline._task_count = 0

    # quality_estimator returns a non-None float so the bandit-record
    # branch in _stage_learn fires. The exact value doesn't matter
    # for the settle invariant; only whether record/cancel was called.
    pipeline.quality_estimator = MagicMock()
    pipeline.quality_estimator.estimate = MagicMock(return_value=0.85)

    capture = _RustRouterCapture(raise_on_record=raise_on_record)
    pipeline._rust_router = capture
    return pipeline, capture


def _make_ctx_single_agent_bypass(decision_id: str) -> PipelineContext:
    """Path 1: single-agent bypass.

    ``_is_single_agent_execution`` returns True when ctx.topology is
    None. executed_template='single_agent', single executed_model_id
    matching the bandit-issued model_id → record-success path.
    """
    ctx = PipelineContext(task="single agent task")
    ctx.system = 1
    ctx.domain = "code"
    ctx.result = "single agent output"
    ctx.cost = 0.001
    ctx.latency_ms = 50.0
    ctx.topology = None
    ctx.bandit_decision_id = decision_id
    ctx.bandit_model_id = "bandit-model"
    ctx.bandit_template = "single_agent"
    ctx.executed_model_id = "bandit-model"
    ctx.executed_template = "single_agent"
    ctx.executed_model_ids = ["bandit-model"]
    return ctx


def _make_ctx_topology_runner_single_model(decision_id: str) -> PipelineContext:
    """Path 2: topology runner, executed_template='sequential' + single model.

    A sequential template with one node degenerates to a single
    executed_model_id. ``_record_bandit_outcome_checked`` allows
    this — neither the multi-template nor multi-model trip-wires
    fire — so it routes to the record path.
    """
    topology = MagicMock()
    topology.id = "topology-seq-1"
    topology.node_count = MagicMock(return_value=1)

    ctx = PipelineContext(task="seq task")
    ctx.system = 2
    ctx.domain = "code"
    ctx.result = "seq output"
    ctx.cost = 0.005
    ctx.latency_ms = 200.0
    ctx.topology = topology
    ctx.topology_id = "topology-seq-1"
    ctx.bandit_decision_id = decision_id
    ctx.bandit_model_id = "seq-model"
    ctx.bandit_template = "sequential"
    ctx.executed_model_id = "seq-model"
    ctx.executed_template = "sequential"
    ctx.executed_model_ids = ["seq-model"]
    return ctx


def _make_ctx_topology_runner_multi_node_template(
    decision_id: str,
) -> PipelineContext:
    """Path 3a: multi-agent with parallel template (multi_node_ambiguous).

    ``_MULTI_NODE_ATTRIBUTION_TEMPLATES = {"parallel", "parallel_fanout",
    "debate"}`` — these templates parallelise quality across nodes,
    so the SystemRouter can't attribute the outcome to a single
    model. The recorder branch in _record_bandit_outcome_checked
    explicitly cancels with reason 'multi_node_ambiguous'.
    """
    topology = MagicMock()
    topology.id = "topology-parallel-1"
    topology.node_count = MagicMock(return_value=3)

    ctx = PipelineContext(task="parallel task")
    ctx.system = 2
    ctx.domain = "code"
    ctx.result = "parallel output"
    ctx.cost = 0.01
    ctx.latency_ms = 300.0
    ctx.topology = topology
    ctx.topology_id = "topology-parallel-1"
    ctx.bandit_decision_id = decision_id
    ctx.bandit_model_id = "model-a"
    ctx.bandit_template = "parallel"
    ctx.executed_model_id = ""  # no single model
    ctx.executed_template = "parallel"
    ctx.executed_model_ids = ["model-a", "model-b", "model-c"]
    return ctx


def _make_ctx_topology_runner_multi_executed_models(
    decision_id: str,
) -> PipelineContext:
    """Path 3b: multi-agent with non-multi-node template + multiple model_ids.

    Template is 'sequential' (NOT in _MULTI_NODE_ATTRIBUTION_TEMPLATES)
    but executed_model_ids has 2 distinct entries — the
    ``len(set(executed_model_ids)) > 1`` trip-wire fires and routes
    to multi_node_ambiguous cancel. Demonstrates the invariant
    holds even on the executed-models trip-wire (not just the
    template trip-wire).
    """
    topology = MagicMock()
    topology.id = "topology-seq-multi"
    topology.node_count = MagicMock(return_value=2)

    ctx = PipelineContext(task="seq multi-model task")
    ctx.system = 2
    ctx.domain = "code"
    ctx.result = "seq multi output"
    ctx.cost = 0.008
    ctx.latency_ms = 250.0
    ctx.topology = topology
    ctx.topology_id = "topology-seq-multi"
    ctx.bandit_decision_id = decision_id
    ctx.bandit_model_id = "model-x"
    ctx.bandit_template = "sequential"
    ctx.executed_model_id = ""
    ctx.executed_template = "sequential"
    ctx.executed_model_ids = ["model-x", "model-y"]  # 2 distinct
    return ctx


def _make_ctx_oracle_abstain(decision_id: str) -> PipelineContext:
    """Path 4: oracle abstains, verdict.trainable=False.

    Oracle on + trainable=False → quality stays None → Stage 5 calls
    _cancel_bandit_decision + _clear_bandit_decision in the else
    branch (line 2853-2854). Invariant 6 fires before the recorder's
    multi-node check.
    """
    ctx = _make_ctx_single_agent_bypass(decision_id)
    ctx.oracle_verdict = OracleVerdict(
        trainable=False,
        verdict_source="abstain",
        quality_label="unknown",
        score=None,
        confidence=0.0,
        reason_codes=("abstain",),
        evidence=(),
    )
    return ctx


# Helpers for asserting the singleton-settle invariant.


def _assert_settled_exactly_once(
    capture: _RustRouterCapture,
    decision_id: str,
    *,
    expected: str,  # "record" or "cancel"
) -> None:
    """Assert decision_id was settled exactly once with the expected op."""
    settles = capture.successful_settles.get(decision_id, [])
    assert settles == [expected], (
        f"Singleton-settle invariant violated for decision_id="
        f"{decision_id!r}: got {settles!r}, expected exactly "
        f"[{expected!r}]. Per ledger invariant 6, every issued "
        f"decision_id must reach exactly one of "
        f"{{record_outcome_checked, cancel_bandit_decision}} "
        f"by end-of-run."
    )
    assert decision_id not in capture.pending_decisions, (
        f"decision_id={decision_id!r} is still pending after "
        f"_stage_learn — it was issued by Stage 0 but the "
        f"settle path did not remove it from the pending set."
    )


# ─────────────────────────────────────────────────────────────────
# Path-specific tests (one per execution path)
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_path_single_agent_bypass_settles_with_record_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path 1: single-agent bypass settles via record (no multi-node trip)."""
    pipeline, capture = _build_pipeline_for_learn(monkeypatch)
    capture.issue("d-bypass")
    ctx = _make_ctx_single_agent_bypass("d-bypass")

    await pipeline._stage_learn(ctx)

    _assert_settled_exactly_once(capture, "d-bypass", expected="record")


@pytest.mark.asyncio
async def test_path_topology_runner_single_model_settles_with_record_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path 2: topology runner, single executed_model, settles via record."""
    pipeline, capture = _build_pipeline_for_learn(monkeypatch)
    capture.issue("d-seq-1")
    ctx = _make_ctx_topology_runner_single_model("d-seq-1")

    await pipeline._stage_learn(ctx)

    _assert_settled_exactly_once(capture, "d-seq-1", expected="record")


@pytest.mark.asyncio
async def test_path_topology_runner_multi_node_template_cancels_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path 3a: multi-node template (parallel) cancels via multi_node_ambiguous."""
    pipeline, capture = _build_pipeline_for_learn(monkeypatch)
    capture.issue("d-parallel")
    ctx = _make_ctx_topology_runner_multi_node_template("d-parallel")

    await pipeline._stage_learn(ctx)

    _assert_settled_exactly_once(capture, "d-parallel", expected="cancel")
    pipeline._emit_bandit_attribution_mismatch.assert_called_once()
    args = pipeline._emit_bandit_attribution_mismatch.call_args
    assert args.args[1] == "multi_node_ambiguous", (
        f"Expected reason_code='multi_node_ambiguous', got {args.args[1]!r}."
    )


@pytest.mark.asyncio
async def test_path_topology_runner_multi_executed_models_cancels_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path 3b: ``len(set(executed_model_ids)) > 1`` → cancel."""
    pipeline, capture = _build_pipeline_for_learn(monkeypatch)
    capture.issue("d-seq-multi")
    ctx = _make_ctx_topology_runner_multi_executed_models("d-seq-multi")

    await pipeline._stage_learn(ctx)

    _assert_settled_exactly_once(capture, "d-seq-multi", expected="cancel")


@pytest.mark.asyncio
async def test_path_oracle_abstain_cancels_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path 4: oracle on + trainable=False → cancel before recorder runs."""
    monkeypatch.setenv("SAGE_ORACLE", "1")  # override the helper's "0"
    pipeline, capture = _build_pipeline_for_learn(monkeypatch)
    monkeypatch.setenv("SAGE_ORACLE", "1")  # re-apply after helper reset
    capture.issue("d-abstain")
    ctx = _make_ctx_oracle_abstain("d-abstain")

    await pipeline._stage_learn(ctx)

    _assert_settled_exactly_once(capture, "d-abstain", expected="cancel")
    # No record_outcome_checked attempt — the gate short-circuits
    # before the recorder is reached.
    assert capture.record_attempts == [], (
        f"Oracle abstain triggered a record attempt: "
        f"{capture.record_attempts!r}. The gate must block before "
        f"the recorder is consulted."
    )


# ─────────────────────────────────────────────────────────────────
# Path 5: record raises (mismatch from Rust side) → cancel follows
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_outcome_mismatch_raises_then_cancels_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path 5: SystemRouter.record_outcome_checked raises → pipeline cancels.

    Mismatch case (e.g. controller upgrade changed the model). The
    record attempt does NOT mutate the posterior; the cancel does.
    Invariant: successful_settles has exactly one entry — 'cancel'.
    record_attempts has one entry too (the failed attempt is observed
    on the call surface), but it didn't mutate state.
    """
    pipeline, capture = _build_pipeline_for_learn(
        monkeypatch,
        raise_on_record=RuntimeError("model_mismatch"),
    )
    capture.issue("d-mismatch")
    ctx = _make_ctx_single_agent_bypass("d-mismatch")
    ctx.executed_model_id = "different-model"  # forces mismatch

    await pipeline._stage_learn(ctx)

    _assert_settled_exactly_once(capture, "d-mismatch", expected="cancel")
    assert len(capture.record_attempts) == 1, (
        f"Expected exactly 1 record_outcome_checked attempt before "
        f"the cancel fallback, got {len(capture.record_attempts)}."
    )


# ─────────────────────────────────────────────────────────────────
# Meta-invariant: across ALL paths, settle count is exactly 1
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invariant_singleton_settle_per_decision_id_across_all_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Meta-loop: across all 5 paths above, no decision_id is ever
    settled 0 times or 2+ times.

    Without this meta-test, a future regression that double-settled
    in only one path could slip through if the per-path test only
    checked its own row of the truth table. This loop runs every
    path with a distinct decision_id and asserts the singleton
    contract globally.
    """
    paths = [
        ("d-meta-bypass",      _make_ctx_single_agent_bypass,             "record"),
        ("d-meta-seq",         _make_ctx_topology_runner_single_model,    "record"),
        ("d-meta-parallel",    _make_ctx_topology_runner_multi_node_template, "cancel"),
        ("d-meta-seq-multi",   _make_ctx_topology_runner_multi_executed_models, "cancel"),
        ("d-meta-abstain",     _make_ctx_oracle_abstain,                  "cancel"),
    ]

    pipeline, capture = _build_pipeline_for_learn(monkeypatch)
    # The abstain path requires SAGE_ORACLE=1; running it last under
    # a setenv applied here doesn't affect the earlier paths because
    # _stage_learn is awaited sequentially with the env state at
    # await time. The non-abstain paths run with SAGE_ORACLE=0 (set
    # by _build_pipeline_for_learn).
    for decision_id, ctx_factory, expected in paths:
        if decision_id == "d-meta-abstain":
            monkeypatch.setenv("SAGE_ORACLE", "1")
        else:
            monkeypatch.setenv("SAGE_ORACLE", "0")
        capture.issue(decision_id)
        ctx = ctx_factory(decision_id)
        await pipeline._stage_learn(ctx)
        _assert_settled_exactly_once(capture, decision_id, expected=expected)

    # Final invariant check: every issued decision_id has exactly
    # one settle entry. No 0s (unsettled), no 2+s (double-settled).
    all_settle_counts = {
        d: len(settles) for d, settles in capture.successful_settles.items()
    }
    assert all_settle_counts == {
        "d-meta-bypass": 1,
        "d-meta-seq": 1,
        "d-meta-parallel": 1,
        "d-meta-seq-multi": 1,
        "d-meta-abstain": 1,
    }, (
        f"Singleton-settle invariant violated globally: "
        f"{all_settle_counts!r}. Every issued decision_id must "
        f"appear exactly once across the union of "
        f"{{record_outcome_checked, cancel_bandit_decision}} calls."
    )
    # And nothing left pending.
    assert not capture.pending_decisions, (
        f"Pending decisions after all paths ran: "
        f"{capture.pending_decisions!r}. Every path must leave its "
        f"decision_id settled."
    )
