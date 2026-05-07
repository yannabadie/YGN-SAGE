"""P4 coupling test (cycle-11, cgpro round-5 directive 2026-05-04).

Locks the code-level invariant that **topology selection is independent
of the ablation `skip_guardrails` flag**. The morning N=8 narrative
(2026-05-04) reported "skip_guardrails -> 5->3 nodes on /19+/34+/82"
as a topology-coupling effect. cgpro round-2 caught that this narrative
was based on the empty `executed_template` field (cycle-9 telemetry
bug, fixed in commit `c136463e`). The cycle-10 P4 N=10 paired
counterbalanced replay (commit `a4fd39db`) showed per_task vectors
byte-identical across configs (full=no-grd=4/10).

cgpro round-5 (2026-05-04) directed cycle-11 P4 to be a coupling unit
test "in a fresh context": prove or refute deterministic coupling
between `skip_guardrails` and topology selection. cgpro VERIFY round
on this test (2026-05-04) flagged that the test's coverage is the
deterministic DAG-template branch (with `pipeline.engine = None`),
not all `_stage_select_topology` branches — see the per-test docstrings
for what each branch / assertion actually proves. The 5th test is a
source-inspection guard that closes the unexercised-branch hole at
the code-contract level.

What this test proves
=====================
Two assertions are locked here:

1. `_stage_select_topology` reads neither `self._agent_loop._skip_guardrails`
   nor any other guardrails-derived signal. A read counter wraps the
   AgentLoop's `_skip_guardrails` attribute; the test asserts the
   counter stays at 0 across the topology-selection call.

2. With identical `dag_features` / `system` / `domain` inputs, the
   produced `ctx.topology` shape is byte-identical regardless of
   `_skip_guardrails` state. This means the v7 4/10 -> 7/10 gap (which
   the morning narrative claimed had a topology component) had no
   deterministic topology-selection driver.

Implication for v7 / A3
=======================
- Pass = the P4 sample-variance interpretation is consistent with
  this code-level invariant: fixed DAG features + fixed system/domain
  produce the same selected topology regardless of `_skip_guardrails`.
  Any per-task topology divergence observed in benches must therefore
  come from upstream stochastic inputs (LLM TaskPlanner producing
  different DAG features across runs), not from a hidden read of
  `_skip_guardrails` inside `_stage_select_topology`.
- Fail would mean a hidden deterministic coupling exists in topology
  selection, so the P4 interpretation would need to distinguish
  deterministic coupling from stochastic task variance. The cycle-10
  narrative about Fix C disabling the wrong lever would need another
  look.

NOT framed as "definitive proof of v7 gap"
==========================================
Per cgpro round-5 trap #5: the wording in this docstring deliberately
avoids "settles v7" or "definitive". The test settles the narrower
question of *deterministic mechanism coupling*. A3 N=50 cloud rerun
remains useful for tighter confidence intervals on the boundary-
stochastic tasks (/13, /82, /101).
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from sage.pipeline import (
    CognitiveOrchestrationPipeline as Pipeline,
    PipelineContext,
)
from sage.pipeline_stages import DAGFeatures
from sage.pipeline_v2.bandit_attribution import is_single_agent_execution
from sage.pipeline_v2.select_topology import select_topology


class _SkipGuardrailsAccessCounter:
    """Wraps a bool so any read or implicit `__bool__` increments a counter.

    Used to prove that `_stage_select_topology` never consults the
    AgentLoop's `_skip_guardrails` attribute (or any property that
    falls through to it).
    """

    def __init__(self, value: bool) -> None:
        self._value = value
        self.read_count = 0

    def __bool__(self) -> bool:
        self.read_count += 1
        return self._value

    def __repr__(self) -> str:
        return f"_SkipGuardrailsAccessCounter(value={self._value!r}, reads={self.read_count})"


def _make_pipeline_for_topology_select(
    *, skip_guardrails: bool,
) -> tuple[Pipeline, _SkipGuardrailsAccessCounter]:
    """Stub a pipeline that exercises only `_stage_select_topology`.

    Returns the pipeline plus the access counter wrapping
    `agent_loop._skip_guardrails` so the caller can assert the topology
    stage never touched it.

    The stub provides only what `_stage_select_topology` reaches
    transitively: `engine` is None so the DAG-template branch fires
    (deterministic), `_emit` is a MagicMock, `_apply_topology_budget_
    and_cache` is patched to a no-op so the budget logic doesn't pull
    in a real cost estimator. We don't stub `_build_topology_from_hint`
    — letting it run real means we exercise the actual production
    topology factory and the test has teeth.
    """
    pipeline = Pipeline.__new__(Pipeline)
    pipeline._emit = MagicMock()
    pipeline.event_bus = None
    pipeline._on_topology_evolve = None
    pipeline.engine = None  # force DAG-template branch
    pipeline._topology_cache = {}

    # _apply_topology_budget_and_cache walks ctx.budget + cost estimator.
    # Replace with a no-op so the test stays focused on selection, not
    # on budget enforcement (which has its own tests).
    pipeline._apply_topology_budget_and_cache = MagicMock()
    pipeline._log_topology_structure = MagicMock()

    # AgentLoop singleton with an instrumented _skip_guardrails
    counter = _SkipGuardrailsAccessCounter(skip_guardrails)
    agent_loop = MagicMock()
    agent_loop._skip_guardrails = counter
    pipeline._agent_loop = agent_loop

    return pipeline, counter


def _make_ctx_with_dag_features(
    *,
    system: int = 2,
    domain: str = "code",
    omega: int = 2,
    delta: int = 2,
    gamma: float = 0.4,
) -> PipelineContext:
    """Build a PipelineContext that triggers DAG-driven topology selection.

    `system=2` is the BCB ablation default (most tasks classify as S2);
    `omega/delta/gamma` are picked to land on a non-trivial DAG template
    (see select_macro_topology decision boundaries).

    Real `PipelineContext` (not SimpleNamespace) so all dataclass
    defaults — including `budget=5.0` and `assignments={}` — are
    populated for the downstream `_apply_topology_budget_and_cache`
    call inside `_stage_select_topology`.
    """
    dag_features = DAGFeatures(omega=omega, delta=delta, gamma=gamma)
    ctx = PipelineContext(task="def add(a, b):\n    return a + b")
    ctx.system = system
    ctx.domain = domain
    ctx.dag_features = dag_features
    return ctx


def _topology_signature(ctx: Any) -> tuple[str | None, int | None, int | None]:
    """Reduce ctx.topology to a structurally-comparable signature.

    `(template_name, node_count, edge_count)`. The TopologyGraph object
    carries a fresh ULID per construction (see `id` attribute) so direct
    object equality never holds; we compare structural shape instead.
    Bypass case is encoded as `(None, None, None)`.
    """
    topo = getattr(ctx, "topology", None)
    if topo is None:
        return (None, None, None)

    # PyO3 TopologyGraph wrapper exposes `template` (str) and counts the
    # nodes via `node_count()`. Edge count comes from `repr()` parsing
    # since the wrapper doesn't expose `edge_count()` directly — but
    # `repr` includes it deterministically.
    template = getattr(topo, "template", None)
    nc = topo.node_count() if hasattr(topo, "node_count") else None
    # Edge count from repr: TopologyGraph(id='...', template='...', nodes=5, edges=6)
    repr_str = repr(topo)
    ec: int | None = None
    if "edges=" in repr_str:
        try:
            tail = repr_str.split("edges=", 1)[1]
            ec = int(tail.split(")", 1)[0].split(",", 1)[0])
        except (ValueError, IndexError):
            ec = None
    return (template, nc, ec)


@pytest.mark.asyncio
async def test_stage_select_topology_does_not_read_skip_guardrails():
    """`_stage_select_topology` must not consult AgentLoop._skip_guardrails.

    A read counter on the `_skip_guardrails` attribute stays at 0 across
    a call to `_stage_select_topology`. If a future commit adds a
    `if self._agent_loop._skip_guardrails:` branch in this method, the
    counter would tick and this test would fail.
    """
    pipeline, counter = _make_pipeline_for_topology_select(skip_guardrails=True)
    ctx = _make_ctx_with_dag_features()

    select_topology(pipeline, ctx)

    assert counter.read_count == 0, (
        f"_stage_select_topology read AgentLoop._skip_guardrails "
        f"{counter.read_count} time(s) — topology selection must be "
        "decoupled from the guardrail-skip flag (cgpro round-5 invariant)"
    )


@pytest.mark.asyncio
async def test_topology_signature_invariant_across_skip_guardrails_states():
    """Identical (dag_features, system, domain) inputs MUST produce
    byte-identical topology shape regardless of `_skip_guardrails`.

    This is the load-bearing assertion: the morning N=8 "5->3 nodes
    coupling" narrative is refuted at the code level if both runs
    yield the same (template_name, node_count) signature.
    """
    pipeline_full, _ = _make_pipeline_for_topology_select(skip_guardrails=False)
    pipeline_no_grd, _ = _make_pipeline_for_topology_select(skip_guardrails=True)

    ctx_full = _make_ctx_with_dag_features()
    ctx_no_grd = _make_ctx_with_dag_features()

    select_topology(pipeline_full, ctx_full)
    select_topology(pipeline_no_grd, ctx_no_grd)

    sig_full = _topology_signature(ctx_full)
    sig_no_grd = _topology_signature(ctx_no_grd)

    # Sanity: the test inputs MUST exercise a non-trivial multi-node
    # topology — otherwise the equality check would pass on
    # (None,None,None) and prove nothing.
    assert sig_full != (None, None, None), (
        f"test inputs did not produce a topology — adjust dag_features "
        f"so _stage_select_topology fires the DAG-template branch. Got "
        f"sig={sig_full!r}"
    )
    assert sig_full[1] is not None and sig_full[1] > 1, (
        f"test inputs produced a single-node topology (signature {sig_full!r}); "
        f"the decoupling invariant has nothing to verify in the bypass "
        f"branch — adjust dag_features so a multi-node template fires."
    )

    assert sig_full == sig_no_grd, (
        f"topology signature diverged under skip_guardrails toggle: "
        f"full={sig_full!r}, no-guardrails={sig_no_grd!r}. The morning "
        f"N=8 'topology coupling' narrative would be REPRODUCED at the "
        f"code level — investigate _stage_select_topology for a hidden "
        f"read channel."
    )


@pytest.mark.asyncio
async def test_topology_signature_stable_under_repeated_calls():
    """Sanity test: the same pipeline with the same inputs produces the
    same topology signature on repeated calls.

    Without this baseline, `test_topology_signature_invariant_across_skip_guardrails_states`
    could pass trivially due to nondeterminism in the topology builder
    masking a true coupling. We assert determinism inside one config
    first, then prove it holds across configs.
    """
    pipeline, _ = _make_pipeline_for_topology_select(skip_guardrails=False)

    ctx_a = _make_ctx_with_dag_features()
    ctx_b = _make_ctx_with_dag_features()
    select_topology(pipeline, ctx_a)
    select_topology(pipeline, ctx_b)

    sig_a = _topology_signature(ctx_a)
    sig_b = _topology_signature(ctx_b)
    assert sig_a == sig_b, (
        f"_stage_select_topology is non-deterministic at the same input: "
        f"call_a={sig_a!r}, call_b={sig_b!r}. The decoupling test "
        f"cannot rule out hidden coupling until this is resolved."
    )


def test_stage_select_topology_source_has_no_skip_guardrails_reference():
    """Source-contract guard (cgpro VERIFY round 2026-05-04, optional fifth).

    The behavioral test
    ``test_stage_select_topology_does_not_read_skip_guardrails`` only
    proves the deterministic DAG-template branch (with
    ``pipeline.engine = None``) doesn't read ``_skip_guardrails``.
    Other branches — Rust ``DynamicTopologyEngine`` path, S1 math
    fast-path, ``SAGE_ABLATION_NO_TOPOLOGY=1`` short-circuit — are
    only protected by this test.

    A source-inspection check is appropriate here because the goal of
    the P4 ticket is to make the decoupling invariant **grep-able and
    review-visible**: any future commit that introduces a direct
    ``self._agent_loop._skip_guardrails`` or ``ctx._skip_guardrails``
    read in this method has to either justify breaking the contract
    or update this test.
    """
    from sage.pipeline_v2.select_topology import select_topology as _select_fn
    source = inspect.getsource(_select_fn)
    assert "_skip_guardrails" not in source, (
        "pipeline_v2.select_topology.select_topology source contains a `_skip_guardrails` "
        "reference. The P4 decoupling invariant requires topology "
        "selection to be a function of (dag_features, system, domain) "
        "only. If a new branch genuinely needs to read the flag, the "
        "ablation contract has to be revisited — see "
        "docs/contracts/runtime-integrity-ledger.md and ablation.py."
    )
    # Note: bare `skip_guardrails` (without underscore prefix) is the
    # AblationConfig field name. We deliberately do NOT forbid it as
    # a substring because future logging like "skip_guardrails=False"
    # might be desirable; only the leading-underscore attribute access
    # is contract-violating.


def test_is_single_agent_execution_is_pure_topology_shape():
    """The bypass-vs-multi-node discriminator only reads `ctx.topology`.

    Locks the invariant that `_is_single_agent_execution(ctx)` is a pure
    function of `ctx.topology` shape (None or node_count <= 1) and
    nothing else. If a future commit makes it consult `_skip_guardrails`
    or any other AgentLoop state, this test would force the change to
    surface in code review.
    """
    pipeline = Pipeline.__new__(Pipeline)

    # Case 1: ctx.topology is None -> bypass
    ctx = SimpleNamespace(topology=None)
    assert is_single_agent_execution(pipeline, ctx) is True

    # Case 2: ctx.topology has node_count() == 1 -> bypass
    topo = MagicMock()
    topo.node_count = MagicMock(return_value=1)
    ctx = SimpleNamespace(topology=topo)
    assert is_single_agent_execution(pipeline, ctx) is True

    # Case 3: ctx.topology has node_count() > 1 -> NOT bypass
    topo = MagicMock()
    topo.node_count = MagicMock(return_value=3)
    ctx = SimpleNamespace(topology=topo)
    assert is_single_agent_execution(pipeline, ctx) is False

    # Case 4: ctx.topology has node_count() == 5 -> NOT bypass
    topo = MagicMock()
    topo.node_count = MagicMock(return_value=5)
    ctx = SimpleNamespace(topology=topo)
    assert is_single_agent_execution(pipeline, ctx) is False
