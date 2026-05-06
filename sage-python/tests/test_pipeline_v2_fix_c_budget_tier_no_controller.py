"""P9 cycle-11 test #4 (ADR-015 acceptance gate): Fix C budget tier disables controller.

Locks the runtime invariant that `_stage_execute` passes
``controller=None`` to ``TopologyRunner`` when ``_llm_tier == "budget"``,
regardless of whether ``self.controller`` was provided at construction.

Per ADR-015 §"Contracts that MUST be preserved" item #4: the cycle-11/12
decomposition of ``pipeline.py`` into ``pipeline_v2/`` must preserve this
behavior byte-identically. Fix C (``a23e196b``, 2026-05-03) is the
load-bearing knob behind the cycle-10 P4 v7 N=10 counterbalanced
analysis — disabling the adaptive ``TopologyController`` shaved
~30-50 s/task on budget-tier multi-agent runs that were otherwise
pushed past the 120 s cap by model upgrades + reroutes.

What this test proves
=====================
Four behavioral assertions on the multi-agent branch of
``_stage_execute`` (the only branch where ``controller`` flows into
``TopologyRunner``):

1. ``tier="budget"`` + ``controller=sentinel`` → ``TopologyRunner`` is
   constructed with ``controller=None``. **Load-bearing assertion.**
2. ``tier="reasoner"`` + ``controller=sentinel`` → ``TopologyRunner``
   is constructed with ``controller=sentinel``. Sanity baseline so
   case 1's ``None`` is not vacuously true.
3. ``tier=""`` (default unset) + ``controller=sentinel`` → controller
   is preserved. Only the literal string ``"budget"`` triggers the
   disable; empty/unset does not.
4. ``tier="budget"`` + ``controller=None`` → ``controller=None``
   regardless. Defensive case (asserts no exception when controller
   was never set in the first place).

Plus a source-inspection guard locking the literal ``_llm_tier == "budget"``
check, parallel to the P4 source-inspection test
(``test_pipeline_topology_skip_guardrails_decoupling.py::test_stage_select_topology_source_has_no_skip_guardrails_reference``).

Why behavioral, not just source-inspection
==========================================
The cycle-11/12 decomposition will move ``_stage_execute`` into
``pipeline_v2/execute.py``. A pure source-inspection test on
``Pipeline._stage_execute`` would no longer see the moved code. The
behavioral capture against ``TopologyRunner.__init__`` works against
either the current monolith OR the future ``pipeline_v2/execute.py``
because the contract is "TopologyRunner sees controller=None when
tier=budget", not "the literal source string lives in pipeline.py".

The 5th source-inspection test is the cycle-11 backstop. Once
cycle-12 phase 2 moves the logic, that test will need to be updated
to point at the new module.
"""
from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest

from sage.pipeline import (
    CognitiveOrchestrationPipeline as Pipeline,
    PipelineContext,
)
from sage.pipeline_v2.execute import execute


class _MultiNodeTopology:
    """Minimal multi-node topology that exits the bypass branch.

    ``_is_single_agent_execution`` returns ``False`` for ``node_count > 1``,
    so this stub forces ``_stage_execute`` into the multi-agent branch
    where ``_effective_controller`` is computed and forwarded to
    ``TopologyRunner``.
    """

    template_type = "sequential"
    id = "topology-fix-c-test"

    def node_count(self) -> int:
        return 2

    def get_node(self, idx: int) -> Any:
        return MagicMock(model_id=f"model-{idx}", max_cost_usd=0.0)


def _build_pipeline(*, llm_tier: str, controller: Any) -> Pipeline:
    """Surgical stub of ``Pipeline`` exposing only what ``_stage_execute``'s
    multi-agent branch reaches.

    Mirrors the P4 ``Pipeline.__new__(Pipeline)`` pattern — bypasses
    ``__init__`` so we don't pull in unrelated boot subsystems (oracle
    config, harness patcher, agent-loop bypass lock plumbing).
    """
    pipeline = Pipeline.__new__(Pipeline)
    pipeline._llm_tier = llm_tier
    pipeline.controller = controller

    # Multi-agent branch attribute reads.
    pipeline.llm_provider = MagicMock()
    pipeline.llm_config = MagicMock()
    pipeline.provider_pool = MagicMock()
    pipeline.assigner = MagicMock()
    pipeline.event_bus = None
    pipeline.tool_registry = None
    pipeline._agent_loop = None  # skips the agent_loop_factory branch
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

    # Skip the FrugalGPT cascade (line ~2647): the cascade fires when
    # quality_estimator is truthy AND quality < 0.3. Setting None
    # short-circuits at the first conjunct, keeping the test focused
    # on the controller-forwarding contract rather than cascade
    # mechanics (which has its own coverage in test_pipeline.py).
    pipeline.quality_estimator = None

    # Skip the post-execution cost estimator. The stub runner reports
    # total_cost_usd=0.0 so ``ctx.cost`` would otherwise trigger
    # ``self._estimate_topology_cost(ctx)`` (line ~2745). Patching to a
    # no-op MagicMock returning 0.0 keeps the test surface minimal —
    # cost estimation has dedicated tests elsewhere.
    pipeline._estimate_topology_cost = MagicMock(return_value=0.0)

    # P6-B bypass lock plumbing — multi-agent branch never enters the
    # bypass mutation block, but ensure the attrs exist for safety.
    pipeline._agent_loop_bypass_lock = None
    pipeline._agent_loop_bypass_lock_loop = None

    return pipeline


def _build_ctx() -> PipelineContext:
    """Build a ``PipelineContext`` that triggers the multi-agent branch.

    ``ctx.topology.node_count() == 2`` exits the bypass guard at the
    top of ``_stage_execute``; ``ctx.assignments`` populates the
    ``executed_model_ids`` rollup at the start of the multi-agent
    block.
    """
    ctx = PipelineContext(task="def add(a, b):\n    return a + b")
    ctx.topology = _MultiNodeTopology()  # type: ignore[assignment]
    ctx.assignments = {0: "model-0", 1: "model-1"}
    ctx.system = 2
    ctx.domain = "code"
    ctx.verification_passed = True
    ctx.cost_tracker = None  # skips the budget-exceeded short-circuit
    return ctx


class _RunnerCapture:
    """Captures the kwargs ``TopologyRunner.__init__`` was called with.

    Single-shot per test instance: the multi-agent branch constructs
    exactly one ``TopologyRunner`` per ``_stage_execute`` call.
    """

    def __init__(self) -> None:
        self.captured_kwargs: dict[str, Any] | None = None
        self.run_called: bool = False

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch ``sage.topology.runner.TopologyRunner`` to capture init kwargs.

        Also stubs ``sage_core.TopologyExecutor`` so the multi-agent
        branch's ``from sage_core import TopologyExecutor`` line
        succeeds without requiring the Rust extension.
        """
        capture = self

        class _StubRunner:
            def __init__(self, **kwargs: Any) -> None:
                capture.captured_kwargs = kwargs
                # Roll-up reads on the runner instance after construction.
                self.tool_call_count = 0
                self.tool_turn_count = 0
                self.executed_commands: list[str] = []
                self.total_cost_usd = 0.0

            async def run(self, task: str) -> str:
                capture.run_called = True
                return "stub multi-agent output"

        import sage.topology.runner as runner_mod

        monkeypatch.setattr(runner_mod, "TopologyRunner", _StubRunner)

        # Stub TopologyExecutor — the multi-agent branch imports it
        # locally as `from sage_core import TopologyExecutor`. We
        # inject a fake module so the import succeeds without the
        # Rust extension.
        import sys
        from types import SimpleNamespace

        class _StubExecutor:
            def __init__(self, graph: Any) -> None:
                self.graph = graph

        # Preserve any existing sage_core attrs (PyO3 wrappers etc.) by
        # patching only TopologyExecutor when sage_core is loaded; if
        # it's not loaded, install a minimal stub module.
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
async def test_fix_c_budget_tier_passes_none_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load-bearing: tier='budget' + controller=sentinel → TopologyRunner sees None.

    This is the assertion that breaks if Fix C is reverted in the
    decomposition. cgpro 2026-05-04 round-2 reassessment: Fix C is
    the only knob that disables the ``TopologyController`` for budget
    tier; rule-based ``guardrail_pipeline`` is a separate orthogonal
    surface controlled by ``AblationConfig.guardrails``.
    """
    sentinel_controller = MagicMock(name="TopologyController-sentinel")
    pipeline = _build_pipeline(
        llm_tier="budget", controller=sentinel_controller
    )
    ctx = _build_ctx()
    capture = _RunnerCapture()
    capture.install(monkeypatch)

    await execute(pipeline, ctx)

    assert capture.captured_kwargs is not None, (
        "TopologyRunner was never constructed — _stage_execute did not "
        "reach the multi-agent branch. Check that ctx.topology has "
        "node_count > 1 and ctx.assignments is populated."
    )
    assert capture.captured_kwargs["controller"] is None, (
        f"Fix C invariant violated: tier='budget' + "
        f"controller={sentinel_controller!r} produced "
        f"TopologyRunner(controller={capture.captured_kwargs['controller']!r}). "
        f"Expected None — see ADR-015 §'Contracts that MUST be preserved' "
        f"item #4 and pipeline.py:_stage_execute Fix C block."
    )


@pytest.mark.asyncio
async def test_non_budget_tier_preserves_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tier='reasoner' + controller=sentinel → TopologyRunner sees sentinel.

    Sanity baseline: without this, the budget-tier=None assertion
    could pass vacuously (e.g. if the code unconditionally passed
    None). Confirms the ``"budget"`` check is real, not a no-op.
    """
    sentinel_controller = MagicMock(name="TopologyController-sentinel")
    pipeline = _build_pipeline(
        llm_tier="reasoner", controller=sentinel_controller
    )
    ctx = _build_ctx()
    capture = _RunnerCapture()
    capture.install(monkeypatch)

    await execute(pipeline, ctx)

    assert capture.captured_kwargs is not None
    assert capture.captured_kwargs["controller"] is sentinel_controller, (
        f"Non-budget tier dropped the controller: tier='reasoner' "
        f"produced controller="
        f"{capture.captured_kwargs['controller']!r}, expected the "
        f"sentinel passed at construction."
    )


@pytest.mark.asyncio
async def test_unset_tier_preserves_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tier='' (default) + controller=sentinel → TopologyRunner sees sentinel.

    Locks that the Fix C check is **literal-string equality** to
    ``"budget"``, not a truthy check. An unset/empty tier is the
    historical default; pre-Fix-C runs all had ``llm_tier=""`` and
    fed the controller to the runner. The decomposition must not
    regress this.
    """
    sentinel_controller = MagicMock(name="TopologyController-sentinel")
    pipeline = _build_pipeline(
        llm_tier="", controller=sentinel_controller
    )
    ctx = _build_ctx()
    capture = _RunnerCapture()
    capture.install(monkeypatch)

    await execute(pipeline, ctx)

    assert capture.captured_kwargs is not None
    assert capture.captured_kwargs["controller"] is sentinel_controller, (
        f"Unset tier was treated as budget: tier='' produced "
        f"controller={capture.captured_kwargs['controller']!r}, "
        f"expected the sentinel."
    )


@pytest.mark.asyncio
async def test_budget_tier_with_no_controller_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tier='budget' + controller=None → TopologyRunner sees None, no exception.

    Defensive case: when no controller was passed at construction
    (e.g. minimal test pipeline), Fix C must not blow up trying to
    short-circuit something that's already None. Asserts the branch
    is well-formed regardless of input combination.
    """
    pipeline = _build_pipeline(llm_tier="budget", controller=None)
    ctx = _build_ctx()
    capture = _RunnerCapture()
    capture.install(monkeypatch)

    await execute(pipeline, ctx)

    assert capture.captured_kwargs is not None
    assert capture.captured_kwargs["controller"] is None


def test_stage_execute_source_contains_fix_c_budget_check() -> None:
    """Source-contract guard: `_stage_execute` references the literal "budget" check.

    Cycle-11 backstop. The behavioral tests above exercise the
    invariant against the current monolith. When cycle-12 phase 2
    moves ``_stage_execute`` into ``pipeline_v2/execute.py``, this
    test will need to be updated to inspect the new module's source
    instead. Until then, it locks the literal pattern that the cgpro
    2026-05-04 round-2 review confirmed is the correct knob:

        _effective_controller = (
            None if self._llm_tier == "budget" else self.controller
        )

    Any commit that drops the ``"budget"`` literal or moves the check
    to a helper without preserving the conditional must either justify
    the contract change in a follow-up ADR or update this test.
    """
    source = inspect.getsource(Pipeline._stage_execute)

    # The two load-bearing tokens. Either reordering would still match,
    # but rewriting `"budget"` to a constant or moving the check to a
    # helper would break this test (intentional — that's an ADR-015
    # contract change, not a refactor).
    assert "_llm_tier" in source, (
        "_stage_execute source no longer references self._llm_tier — "
        "Fix C (ADR-015 contract #4) requires this check. If the "
        "logic moved to a helper, update this test to inspect the "
        "new module per the cycle-12 phase 2 plan."
    )
    assert '"budget"' in source, (
        '_stage_execute source no longer contains the literal "budget" '
        "string — Fix C (ADR-015 contract #4) checks `_llm_tier == "
        '"budget"` to gate the controller. Replacing the literal with '
        "a named constant is fine, but this test must then be updated "
        "to reference the constant explicitly."
    )
