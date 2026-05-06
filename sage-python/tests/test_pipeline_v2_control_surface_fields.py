"""P9 cycle-11 test #5 (ADR-015 acceptance gate): control-surface fields populated.

Locks the runtime-integrity-ledger.md invariant 8 ("Control-surface
completeness", crystallized 2026-05-04 cycle-9 cgpro round-2 from the
α telemetry self-deception incident) and ADR-015 §"Contracts that MUST
be preserved" item #4 / characterization test #5.

The bench layer's ``_capture_control_surface``
(``sage/bench/bigcodebench_bench.py``) reads four fields off
``pipeline.last_context`` after every BCB task, and downstream
analysis ("topology X → topology Y") trusts them as the mechanism
ground truth. The cycle-9 α paired-diagnostic post-mortem reported
"5 → 3 nodes, robust → sequential" coupling that turned out to be
unsupported by the ledger because ``executed_template`` was actually
the ULID ``trace.topology_id`` (telemetry bug, fixed at ``c136463e``).
This test locks the producer side: the four fields are populated by
the right stages with non-empty values when topology is multi-node.

What this test proves
=====================
Per stage in the pipeline, the field that stage produces must be
populated post-execution:

1. ``_stage_decompose`` → ``ctx.dag_features`` is non-None and has
   ``omega`` / ``delta`` / ``gamma`` attributes (load-bearing for
   downstream ``select_macro_topology``).

2. ``_stage_classify`` (Rust SystemRouter branch) → ``ctx.bandit_template``
   is set from ``decision.selected_template`` when the router returns
   a decision with a non-empty template. Bench reads this as
   ``selected_template``.

3. ``_stage_select_topology`` (DAG-template branch, the common case
   for budget-tier S2 tasks) → ``ctx.topology`` is non-None and has an
   ``id`` attribute. Note: only the engine.generate() branch sets
   ``ctx.topology_id`` *explicitly* (line 1365). The DAG-template
   branch leaves ``ctx.topology_id`` empty; the bench reader at
   ``pipeline.py:648`` falls back to ``ctx.topology.id``. The test
   matches that disjunction so cycle-12 phase 2 can choose either
   approach.

4. ``_stage_execute`` (multi-agent branch) → ``ctx.executed_template``
   is set from ``ctx.topology.template_type``. This is the field
   whose blank value caused the cycle-9 α self-deception.

5. ``_stage_execute`` (single-agent bypass branch) → ``ctx.executed_template``
   = ``"single_agent"``. The bench layer must distinguish bypass from
   multi-agent runs, and this is the discriminator.

Why these stages and not pipeline.run() end-to-end
==================================================
End-to-end is test #1 (run byte-identical). This test focuses on
**producer-side** invariants — each stage individually populates its
field. If any one stage stops doing so, the bench reader can't
reconstruct the mechanism even if the rest of the pipeline is
correct. Per-stage assertions also survive cycle-12 phase 2 byte-
identically because the field-set / producer mapping is the contract
ADR-015 forbids changing.

Cycle-9 lesson preservation
===========================
The cycle-9 α self-deception was: the test passed by reading
``trace.topology_id`` (a ULID) into ``cs["executed_template"]`` and
the assertion ``assert cs.get("executed_template")`` was trivially
true (any ULID is non-empty). Test #5 prevents the inverse mistake
on the producer side by also asserting **type / shape**:
``executed_template`` must equal one of the known template strings
(``"sequential"``, ``"parallel"``, ``"avr"``, ``"robust"``,
``"horizon_pipeline"``, ``"parallel_fanout"``, ``"single_agent"``,
``"multi_agent"``), not just "non-empty".
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
from sage.pipeline_stages import DAGFeatures
from sage.pipeline_v2.classify import classify
from sage.pipeline_v2.decompose import decompose
from sage.pipeline_v2.select_topology import select_topology


# Known template strings that ``executed_template`` may equal.
# Source: TemplateStore template names + the bypass / multi-agent
# fallback strings in pipeline.py. If a new template is added, this
# allowlist must grow alongside it (intentional — surfaces the change
# in code review).
_KNOWN_TEMPLATES = frozenset({
    "sequential",
    "parallel",
    "parallel_fanout",
    "avr",
    "selfmoa",
    "hierarchical",
    "hub",
    "debate",
    "brainstorming",
    "robust",
    "horizon_pipeline",
    "formal_solver",
    "single_agent",
    "multi_agent",
})


class _MultiNodeTopology:
    """Minimal multi-node topology stub. Mirrors test #4."""

    template_type = "sequential"
    id = "topology-control-surface-test-id-01"

    def node_count(self) -> int:
        return 2

    def get_node(self, idx: int) -> Any:
        return MagicMock(model_id=f"model-{idx}", max_cost_usd=0.0)


def _build_minimal_pipeline() -> Pipeline:
    """Surgical Pipeline stub for direct stage calls.

    Mirrors the P4 pattern. Per-test customizations (set
    ``_rust_router``, ``engine``, ``llm_provider``) are layered on
    top by the individual tests below.
    """
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
    pipeline._on_topology_evolve = None
    pipeline.engine = None
    pipeline.harness_config = None
    pipeline._harness_patcher = None
    pipeline._agent_loop_bypass_lock = None
    pipeline._agent_loop_bypass_lock_loop = None
    pipeline.quality_estimator = None
    pipeline._estimate_topology_cost = MagicMock(return_value=0.0)
    pipeline.router = None
    pipeline._rust_router = None
    pipeline._topology_cache = {}
    pipeline._apply_topology_budget_and_cache = MagicMock()
    pipeline._log_topology_structure = MagicMock()
    pipeline._last_routing_decision = None
    pipeline._last_runtime_routing_source = ""
    pipeline._last_runtime_routing_confidence = None
    pipeline._last_runtime_routing_model_id = ""
    pipeline.bandit = None
    return pipeline


# ─────────────────────────────────────────────────────────────────
# Test 1: ctx.dag_features populated by _stage_decompose
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stage_decompose_populates_dag_features() -> None:
    """``ctx.dag_features`` is non-None with ``omega``/``delta``/``gamma``.

    The S1 short-circuit (``ctx.system == 1``) sets the trivial
    DAGFeatures(1, 1, 0.0). The fallback branch (LLM unavailable)
    also sets the trivial DAGFeatures. Both produce a populated
    field — the contract is "non-None after the stage runs", not
    "specific values".
    """
    pipeline = _build_minimal_pipeline()
    ctx = PipelineContext(task="def add(a, b):\n    return a + b")
    ctx.system = 1  # forces the trivial-DAG short-circuit path

    assert ctx.dag_features is None, (
        "PipelineContext default for dag_features must be None "
        "before the stage runs — sanity check, otherwise the "
        "post-condition asserts a value that was already set "
        "elsewhere."
    )

    # Cycle-11 cgpro VERIFY follow-up: use returned-ctx pattern,
    # NOT inspection of the original ctx. ADR-015 says
    # PipelineContext becomes an immutable per-stage clone in cycle-12;
    # the test must assert on what the stage returns so an immutable
    # refactor doesn't false-fail this test.
    ctx = await decompose(pipeline, ctx)

    assert ctx.dag_features is not None, (
        "_stage_decompose did not populate ctx.dag_features. "
        "Invariant 8 (runtime-integrity-ledger.md) requires this "
        "field present so downstream select_macro_topology can fire."
    )
    assert hasattr(ctx.dag_features, "omega")
    assert hasattr(ctx.dag_features, "delta")
    assert hasattr(ctx.dag_features, "gamma")
    # Type contract: omega/delta are ints, gamma is float.
    assert isinstance(ctx.dag_features.omega, int)
    assert isinstance(ctx.dag_features.delta, int)
    assert isinstance(ctx.dag_features.gamma, float)


# ─────────────────────────────────────────────────────────────────
# Test 2: ctx.bandit_template populated by _stage_classify
# ─────────────────────────────────────────────────────────────────


class _StubRustRouter:
    """Minimal Rust SystemRouter stub for _stage_classify.

    ``route_integrated`` returns a decision with a fixed
    ``selected_template`` so the bandit-template field is
    deterministic.
    """

    def __init__(self, *, selected_template: str) -> None:
        self._selected_template = selected_template

    def route_integrated(self, task: str, constraints: Any, topology_id: str) -> Any:
        return SimpleNamespace(
            decision_id="d-classify-test",
            system=2,
            model_id="stub-model",
            selected_template=self._selected_template,
            template=self._selected_template,
            confidence=0.9,
            estimated_cost=0.001,
            topology_id=topology_id,
        )


def _install_fake_sage_core_for_classify(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``sage_core.RoutingConstraints`` so _stage_classify imports succeed."""
    import sys
    from types import SimpleNamespace as SN

    class _FakeRoutingConstraints:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    existing = sys.modules.get("sage_core")
    if existing is not None:
        monkeypatch.setattr(
            existing, "RoutingConstraints", _FakeRoutingConstraints, raising=False
        )
    else:
        monkeypatch.setitem(
            sys.modules,
            "sage_core",
            SN(RoutingConstraints=_FakeRoutingConstraints),
        )


def test_stage_classify_populates_bandit_template_from_router_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ctx.bandit_template`` is set from ``decision.selected_template``.

    Bench layer reads this field as ``selected_template`` per the
    ledger module-cross-reference for invariant 8 and per
    ``bench/bigcodebench_bench.py:363``. cycle-9 c136463e fixed the
    upstream bug where bench was reading ``trace.topology_id`` (a
    ULID) instead of ``ctx.bandit_template``. This test locks the
    producer-side contract that the field IS populated when the
    Rust router returns a non-empty template.
    """
    _install_fake_sage_core_for_classify(monkeypatch)
    pipeline = _build_minimal_pipeline()
    pipeline._rust_router = _StubRustRouter(selected_template="sequential")
    ctx = PipelineContext(task="def add(a, b):\n    return a + b")

    assert ctx.bandit_template == "", (
        "PipelineContext default for bandit_template must be \"\" "
        "before the stage runs."
    )

    # Cycle-11 cgpro VERIFY follow-up: returned-ctx pattern.
    ctx = classify(pipeline, ctx)

    assert ctx.bandit_template == "sequential", (
        f"_stage_classify did not set ctx.bandit_template from the "
        f"router decision: got {ctx.bandit_template!r}, expected "
        f"'sequential'. Per invariant 8 the bench layer reads this "
        f"field as 'selected_template' downstream."
    )
    assert ctx.bandit_decision_id == "d-classify-test"


# ─────────────────────────────────────────────────────────────────
# Test 3: ctx.topology built by _stage_select_topology has .id
# ─────────────────────────────────────────────────────────────────


def test_stage_select_topology_dag_template_branch_sets_ctx_topology_id() -> None:
    """The DAG-template branch sets ``ctx.topology_id`` directly.

    Cycle-11 cgpro VERIFY follow-up (2026-05-05): the original test
    asserted only the disjunction ``ctx.topology_id or ctx.topology.id``
    because the DAG-template branch (the common case for budget-tier
    S2 tasks) was leaving ``ctx.topology_id`` blank. cgpro pointed out
    that the bench layer's ``_capture_control_surface`` at
    ``bench/bigcodebench_bench.py`` reads ``ctx.topology_id`` DIRECTLY
    (not via the disjunction at ``pipeline.py:648``, which is a
    different consumer — the runtime event log).

    Production fix: the DAG-template branch (and the engine.generate
    "elif result" fallback branch) now both set
    ``ctx.topology_id = getattr(topo, "id", "") or ""`` immediately
    after ``ctx.topology = topo`` — symmetric with the original
    engine.generate top-level branch. This test now asserts the
    direct field, not the disjunction, so a future regression that
    drops the explicit assignment will fail this test instead of
    silently breaking BCB control-surface telemetry.
    """
    pipeline = _build_minimal_pipeline()
    ctx = PipelineContext(task="def add(a, b):\n    return a + b")
    ctx.system = 2
    ctx.domain = "code"
    ctx.dag_features = DAGFeatures(omega=2, delta=2, gamma=0.4)

    # Cycle-11 cgpro VERIFY follow-up: returned-ctx pattern.
    ctx = select_topology(pipeline, ctx)

    assert ctx.topology is not None, (
        "_stage_select_topology did not produce a topology — check "
        "that DAG features land on a multi-node template branch "
        "(omega/delta/gamma chosen above force select_macro_topology "
        "to a non-bypass template)."
    )
    # Tightened assertion: ctx.topology_id must be set DIRECTLY. The
    # bench reader does not use the runtime-event-log disjunction.
    assert ctx.topology_id, (
        f"ctx.topology_id is empty after Stage 2 DAG-template branch: "
        f"{ctx.topology_id!r}. The bench layer "
        f"`_capture_control_surface` reads this field directly; an "
        f"empty value means BCB control-surface telemetry sees blank "
        f"topology IDs and downstream replay analysis cannot attribute "
        f"runs by topology. Production fix landed cycle-11 (2026-05-05)."
    )
    # Sanity: the field was set from ctx.topology.id (the canonical
    # ULID per ADR-015 plan item 1.4a, not the descriptor-keyed
    # semantic id from result.topology_id() that broke engine cache
    # lookups in the H4 incident).
    assert ctx.topology_id == getattr(ctx.topology, "id", ""), (
        f"ctx.topology_id ({ctx.topology_id!r}) does not match "
        f"ctx.topology.id ({getattr(ctx.topology, 'id', None)!r}). "
        f"They must be the same ULID — see plan item 1.4a (2026-04-20)."
    )


# ─────────────────────────────────────────────────────────────────
# Test 4: ctx.executed_template set by _stage_execute (multi-agent)
# ─────────────────────────────────────────────────────────────────


def _stub_topology_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``TopologyRunner`` so multi-agent _stage_execute returns quickly."""
    class _StubRunner:
        def __init__(self, **kwargs: Any) -> None:
            self.tool_call_count = 0
            self.tool_turn_count = 0
            self.executed_commands: list[str] = []
            self.total_cost_usd = 0.0

        async def run(self, task: str) -> str:
            return "stub multi-agent output"

    import sage.topology.runner as runner_mod

    monkeypatch.setattr(runner_mod, "TopologyRunner", _StubRunner)

    import sys
    from types import SimpleNamespace

    class _StubExecutor:
        def __init__(self, graph: Any) -> None:
            self.graph = graph

    existing = sys.modules.get("sage_core")
    if existing is not None:
        monkeypatch.setattr(
            existing, "TopologyExecutor", _StubExecutor, raising=False
        )
    else:
        monkeypatch.setitem(
            sys.modules,
            "sage_core",
            SimpleNamespace(TopologyExecutor=_StubExecutor),
        )


@pytest.mark.asyncio
async def test_stage_execute_multi_agent_populates_executed_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-agent branch sets ``ctx.executed_template`` from topology.template_type.

    Cycle-9 α self-deception precedent (commit ``c136463e``): bench
    was sourcing ``cs["executed_template"]`` from a ULID instead of
    this field, so the assertion was trivially true on any value.
    Test asserts both **non-empty** AND **a known template string**
    to prevent the inverse mistake on the producer side.
    """
    pipeline = _build_minimal_pipeline()
    pipeline.llm_provider = MagicMock()  # multi-agent branch needs a provider
    pipeline.llm_config = MagicMock()
    pipeline.provider_pool = MagicMock()
    pipeline.assigner = MagicMock()

    _stub_topology_runner(monkeypatch)

    ctx = PipelineContext(task="multi-node task")
    ctx.topology = _MultiNodeTopology()  # type: ignore[assignment]
    ctx.assignments = {0: "model-0", 1: "model-1"}
    ctx.system = 2
    ctx.domain = "code"
    ctx.verification_passed = True

    assert ctx.executed_template == "", (
        "PipelineContext default for executed_template must be \"\" "
        "before the stage runs."
    )

    # Cycle-11 cgpro VERIFY follow-up: returned-ctx pattern.
    ctx = await pipeline._stage_execute(ctx)

    assert ctx.executed_template, (
        "_stage_execute multi-agent branch did not populate "
        "ctx.executed_template. Invariant 8: 'when node_count > 0, "
        "executed_template MUST be non-empty'."
    )
    assert ctx.executed_template in _KNOWN_TEMPLATES, (
        f"ctx.executed_template={ctx.executed_template!r} is not a "
        f"known template name. Either a new template was added "
        f"(update _KNOWN_TEMPLATES allowlist) OR a non-template "
        f"value (ULID, decision_id, model_id) leaked into the "
        f"field — see cycle-9 α self-deception incident "
        f"(commit c136463e)."
    )
    # Specific assertion: the multi-agent branch reads template_type
    # off ctx.topology, so the stub's "sequential" must propagate.
    assert ctx.executed_template == "sequential"


# ─────────────────────────────────────────────────────────────────
# Test 5: ctx.executed_template = "single_agent" on bypass branch
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stage_execute_single_agent_bypass_marks_executed_template_single_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-agent bypass sets ``ctx.executed_template = "single_agent"``.

    The bench layer must distinguish bypass runs (one model, no
    topology) from multi-agent runs (topology executed). This field
    is the discriminator. Without it, the cycle-9 ablation
    "node_count: 5 → 3" narrative would have nothing to compare
    against on the bypass side.

    Tests the path through ``_is_single_agent_execution(ctx) == True``
    when ``ctx.topology is None``. The pipeline falls through to the
    fallback ``provider.generate()`` block (no agent_loop), but sets
    ``ctx.executed_template = "single_agent"`` first regardless.
    """
    pipeline = _build_minimal_pipeline()

    # Provide a stub provider so the fallback block at line 2467 fires.
    # Otherwise _stage_execute returns early without any side-effect
    # on executed_template.
    async def _stub_generate(messages: Any, config: Any = None, **kwargs: Any) -> Any:
        return SimpleNamespace(content="stub single-agent output")

    pipeline.llm_provider = SimpleNamespace(generate=_stub_generate)
    pipeline.llm_config = MagicMock()

    ctx = PipelineContext(task="simple task")
    ctx.topology = None  # forces _is_single_agent_execution → True
    ctx.system = 1
    ctx.domain = "code"
    ctx.verification_passed = True

    # Cycle-11 cgpro VERIFY follow-up: returned-ctx pattern.
    ctx = await pipeline._stage_execute(ctx)

    assert ctx.executed_template == "single_agent", (
        f"Single-agent bypass did not set ctx.executed_template to "
        f"'single_agent': got {ctx.executed_template!r}. The bench "
        f"layer needs this discriminator to attribute runs."
    )
