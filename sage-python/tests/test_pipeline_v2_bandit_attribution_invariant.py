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

from types import SimpleNamespace
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


# ─────────────────────────────────────────────────────────────────
# Cycle-11 cgpro VERIFY follow-up: explicit FrugalGPT cascade +
# multi-agent error-fallback path coverage.
#
# Original argument: cascade and error-fallback produce the same
# Stage-5 ctx state as plain topology-runner (executed_template +
# multi-node executed_model_ids stay set), so the multi_node_ambiguous
# trip-wire fires and cancels. cgpro VERIFY 2026-05-05 disagreed:
# "FrugalGPT can reassign and rerun models after the initial multi-
# agent runner, and the multi-agent error fallback generates a single-
# agent result after the topology path has already populated multi-
# agent attribution fields. Those are not safely equivalent to the
# plain topology-runner state."
#
# These two tests drive _stage_execute through those branches and
# then _stage_learn, asserting the singleton-settle invariant holds.
# They prove the contract empirically rather than by theoretical
# equivalence to the plain topology-runner ctx.
# ─────────────────────────────────────────────────────────────────


def _stub_topology_runner_for_execute(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_outcome: str | type[BaseException] = "stub multi-agent output",
    runner_instances: list[Any] | None = None,
) -> None:
    """Patch ``TopologyRunner`` so multi-agent _stage_execute returns or raises.

    ``run_outcome`` is either a string returned from .run() or an
    exception class to raise (used for the error-fallback test).
    """
    captured = runner_instances if runner_instances is not None else []

    class _StubRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(self)
            self.tool_call_count = 0
            self.tool_turn_count = 0
            self.executed_commands: list[str] = []
            self.total_cost_usd = 0.0
            self._kwargs = kwargs

        async def run(self, task: str) -> str:
            if isinstance(run_outcome, type) and issubclass(run_outcome, BaseException):
                raise run_outcome("stub topology runner failure")
            return str(run_outcome)

    import sage.topology.runner as runner_mod

    monkeypatch.setattr(runner_mod, "TopologyRunner", _StubRunner)

    import sys

    class _StubExecutor:
        def __init__(self, graph: Any) -> None:
            self.graph = graph

    existing = sys.modules.get("sage_core")
    sage_core_attrs: dict[str, Any] = {"TopologyExecutor": _StubExecutor}
    # Keep RoutingConstraints if a previous test put it there; needed
    # by Stage 0 if the test reaches it (these don't, but be safe).
    if existing is not None and hasattr(existing, "RoutingConstraints"):
        sage_core_attrs["RoutingConstraints"] = existing.RoutingConstraints
    if existing is not None:
        for attr, val in sage_core_attrs.items():
            monkeypatch.setattr(existing, attr, val, raising=False)
    else:
        from types import SimpleNamespace as SN
        monkeypatch.setitem(sys.modules, "sage_core", SN(**sage_core_attrs))


def _make_multi_agent_ctx(decision_id: str) -> PipelineContext:
    """Multi-agent ctx that triggers _stage_execute multi-agent branch."""
    topology = MagicMock()
    topology.id = "topology-cascade-test"
    topology.template_type = "sequential"
    topology.node_count = MagicMock(return_value=2)
    topology.get_node = MagicMock(return_value=MagicMock(model_id="model-a", max_cost_usd=0.0))

    ctx = PipelineContext(task="multi-agent path test")
    ctx.system = 2
    ctx.domain = "code"
    ctx.topology = topology  # type: ignore[assignment]
    ctx.assignments = {0: "model-a", 1: "model-b"}
    ctx.bandit_decision_id = decision_id
    ctx.bandit_model_id = "model-a"
    ctx.bandit_template = "sequential"
    ctx.verification_passed = True
    return ctx


@pytest.mark.asyncio
async def test_path_frugalgpt_cascade_settles_decision_id_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FrugalGPT cascade fires (quality < 0.3) → decision_id settled exactly once.

    Drives ``_stage_execute`` through the cascade branch
    (pipeline.py:2647-2729): runner1 runs → quality_estimator returns
    < 0.3 → assigner.assign_single_node reassigns → runner3 retries.
    After the cascade, ``ctx.executed_template`` and
    ``ctx.executed_model_ids`` remain populated from the original
    multi-agent assignments (per cgpro VERIFY observation: cascade
    does NOT refresh attribution fields). Stage 5 sees a multi-node
    state and cancels via ``multi_node_ambiguous``.

    Asserts:
      1. Two TopologyRunner instances were constructed (initial + retry).
      2. Pipeline result is from the retry, not the initial low-quality
         run.
      3. Decision_id settled exactly once via cancel.
    """
    pipeline, capture = _build_pipeline_for_learn(monkeypatch)

    # Override quality_estimator to force cascade (< 0.3 threshold).
    pipeline.quality_estimator = MagicMock()
    pipeline.quality_estimator.estimate = MagicMock(return_value=0.1)

    # Provide an assigner stub for the cascade re-assignment path.
    pipeline.assigner = MagicMock()
    pipeline.assigner.assign_single_node = MagicMock()

    # Provide llm_provider/config and provider_pool for runner config.
    pipeline.llm_provider = MagicMock()
    pipeline.llm_config = MagicMock()
    pipeline.provider_pool = MagicMock()
    pipeline.provider_pool.is_model_available = MagicMock(return_value=True)

    runner_instances: list[Any] = []
    _stub_topology_runner_for_execute(
        monkeypatch,
        run_outcome="cascade retry output",
        runner_instances=runner_instances,
    )

    capture.issue("d-cascade")
    ctx = _make_multi_agent_ctx("d-cascade")

    ctx = await pipeline._stage_execute(ctx)
    await pipeline._stage_learn(ctx)

    # Cascade fired: 2 runner instances (initial + cascade retry).
    assert len(runner_instances) >= 2, (
        f"FrugalGPT cascade did not fire — only {len(runner_instances)} "
        f"TopologyRunner instance(s) constructed. Expected 2 (initial "
        f"+ cascade retry). Check that quality_estimator stub returns "
        f"< 0.3 and assigner.assign_single_node is wired."
    )

    # Cascade retry result took precedence over initial low-quality run.
    assert ctx.result == "cascade retry output", (
        f"ctx.result is not the cascade retry output: {ctx.result!r}. "
        f"Expected 'cascade retry output' from runner3 retry."
    )

    # Singleton-settle invariant holds: cancel exactly once via
    # multi_node_ambiguous trip-wire (executed_model_ids has 2 distinct
    # entries from original assignments — not refreshed by cascade).
    _assert_settled_exactly_once(capture, "d-cascade", expected="cancel")


@pytest.mark.asyncio
async def test_path_multi_agent_error_fallback_settles_decision_id_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-agent runner raises → fallback provider runs → decision_id cancelled.

    Drives ``_stage_execute`` through the error-fallback branch
    (pipeline.py:2768-2805): TopologyRunner.run() raises RuntimeError
    → except handler picks fallback_provider via
    ``_pick_fallback_provider`` → fallback.generate() produces result.

    cgpro VERIFY observation: the fallback path does NOT clear
    ``ctx.executed_template`` ("sequential" from line 2519) or
    ``ctx.executed_model_ids`` (multi-model from line 2516). So the
    bench attribution sees "multi-agent + 2 models" even though the
    actual run was a single fallback provider call. Stage 5 cancels
    via the multi_node_ambiguous trip-wire — this preserves the
    singleton-settle invariant but the SEMANTIC mismatch is itself a
    finding worth surfacing.

    Asserts:
      1. Fallback provider was actually called (not topology runner).
      2. Pipeline result is from the fallback provider.
      3. Decision_id settled exactly once via cancel.
    """
    pipeline, capture = _build_pipeline_for_learn(monkeypatch)
    pipeline.llm_provider = MagicMock()
    pipeline.llm_config = MagicMock()
    pipeline.provider_pool = MagicMock()

    # Stub TopologyRunner.run() to raise; this triggers the fallback path.
    runner_instances: list[Any] = []
    _stub_topology_runner_for_execute(
        monkeypatch,
        run_outcome=RuntimeError,
        runner_instances=runner_instances,
    )

    # Stub _pick_fallback_provider to return a deterministic fake.
    fallback_calls: list[Any] = []

    async def _fallback_generate(messages: Any, config: Any = None, **kwargs: Any) -> Any:
        fallback_calls.append((messages, config))
        return SimpleNamespace(content="fallback provider output")

    fallback_provider = SimpleNamespace(
        generate=_fallback_generate,
        name="stub-fallback-provider",
    )
    pipeline._pick_fallback_provider = MagicMock(
        return_value=(fallback_provider, MagicMock()),
    )

    capture.issue("d-fallback")
    ctx = _make_multi_agent_ctx("d-fallback")

    ctx = await pipeline._stage_execute(ctx)
    await pipeline._stage_learn(ctx)

    # Multi-agent runner attempted exactly once before raising.
    assert len(runner_instances) == 1, (
        f"Expected exactly 1 multi-agent runner attempt before "
        f"fallback, got {len(runner_instances)}."
    )

    # Fallback provider was invoked.
    assert len(fallback_calls) == 1, (
        f"Fallback provider was not invoked — got "
        f"{len(fallback_calls)} call(s). Expected 1."
    )

    # Pipeline result is from the fallback provider.
    assert ctx.result == "fallback provider output", (
        f"ctx.result is not the fallback output: {ctx.result!r}. "
        f"Expected 'fallback provider output'."
    )

    # ctx.executed_template + ctx.executed_model_ids stay populated
    # from line 2516/2519 (NOT cleared by fallback handler) — this is
    # the cgpro VERIFY semantic mismatch finding. The trip-wire still
    # catches it.
    assert ctx.executed_template == "sequential", (
        f"ctx.executed_template should remain 'sequential' (set "
        f"before fallback) — got {ctx.executed_template!r}. The "
        f"fallback handler does NOT clear it; cgpro VERIFY flagged "
        f"this as a semantic mismatch worth tracking."
    )
    assert len(set(ctx.executed_model_ids)) >= 2, (
        f"ctx.executed_model_ids should have 2 distinct entries "
        f"(set before fallback): {ctx.executed_model_ids!r}."
    )

    # Singleton-settle invariant holds: cancel via multi_node_ambiguous.
    _assert_settled_exactly_once(capture, "d-fallback", expected="cancel")
