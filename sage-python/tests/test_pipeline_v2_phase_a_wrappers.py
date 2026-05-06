"""`pipeline_v2` stage ownership + legacy seam compatibility tests.

Cycle-13 K Phase 2.1 (cgpro `cgpro_phase21_facade_rewrite_20260506`,
2026-05-06): this file was originally written for cycle-12 Phase A
("additive wrappers, NO body movement"); Phase B then moved the
bodies and Phase 2.1 finished the façade extraction. The filename
is preserved to avoid test-rename churn, but the conceptual scope
has migrated to **stage-module ownership + transitional seam
compatibility**:

What this file proves NOW
=========================
1. The 6 stage modules exist and expose a callable named after the
   stage (`classify`, `decompose`, `select_topology`,
   `assign_models`, `execute`, `learn`). The bodies live there
   post-Phase-B; `pipeline._stage_<X>` is a 1-line LOCAL-import
   delegator.
2. The legacy `pipeline._stage_<X>` method seams remain on
   `CognitiveOrchestrationPipeline` post-Phase-2.1 as the
   monkey-patch-able runtime test seam used by 27 test files
   (`pipeline._stage_<X> = <fake>`). cgpro round-4 OPTION_3
   verdict: this contract stays intact in Phase 2.1; Phase 2.2
   DESIGN_LOCK rewrites the 27 tests + retires the seams.
3. `PipelineContext` is the SAME class object whether imported as
   `from sage.pipeline import PipelineContext`,
   `from sage.pipeline_v2 import PipelineContext`, or
   `from sage.pipeline_v2.context import PipelineContext`. Since
   Phase 2.1 Step E1 the dataclass body lives in
   `pipeline_v2/context.py`, with explicit
   `PipelineContext.__module__ = "sage.pipeline"` so existing
   tests / bench / dashboards / observability assertions on the
   legacy module path keep passing byte-identical.
4. `pipeline_v2/__init__.py` uses PEP 562 module-level
   `__getattr__` (Step E0) to defer `from sage.pipeline import …`
   to attribute-access time — breaks the otherwise-circular
   dependency once `pipeline.py` itself imports
   `pipeline_v2.context`.
5. The async stage callables (`decompose`, `execute`, `learn`) are
   true coroutine functions, not regular functions returning
   coroutines. Important because the orchestrator (Phase 2.1
   Step D) must `await` them ; signature drift would compile
   silently and break event ordering at runtime.
6. The top-level pipeline_v2 import allowlist on `sage.pipeline`:
   only `from sage.pipeline_v2.context import PipelineContext` is
   permitted at module scope; every other `pipeline_v2` symbol
   must be a LOCAL import inside a delegator method body
   (cgpro DESIGN trap #4 — partial-init avoidance).
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


# ────────────────────────────────────────────────────────────────────
# 1. Package structure: 6 stage modules + 3 helpers + __init__
# ────────────────────────────────────────────────────────────────────


def test_pipeline_v2_package_exposes_expected_modules() -> None:
    """All 9 modules + __init__ exist and import without error."""
    # Each import is the assertion; F401 noqa makes ruff happy.
    import sage.pipeline_v2  # noqa: F401
    import sage.pipeline_v2.assign_models  # noqa: F401
    import sage.pipeline_v2.bandit_attribution  # noqa: F401  (placeholder)
    import sage.pipeline_v2.classify  # noqa: F401
    import sage.pipeline_v2.context  # noqa: F401
    import sage.pipeline_v2.decompose  # noqa: F401
    import sage.pipeline_v2.execute  # noqa: F401
    import sage.pipeline_v2.learn  # noqa: F401
    import sage.pipeline_v2.runtime_events  # noqa: F401  (placeholder)
    import sage.pipeline_v2.select_topology  # noqa: F401


def test_pipeline_v2_init_reexports_pipeline_class_and_context() -> None:
    """`from sage.pipeline_v2 import Pipeline, CognitiveOrchestrationPipeline,
    PipelineContext` works and identifies the same objects as `sage.pipeline`.
    """
    from sage.pipeline import (
        CognitiveOrchestrationPipeline as LegacyPipeline,
        PipelineContext as LegacyCtx,
    )
    from sage.pipeline_v2 import (
        CognitiveOrchestrationPipeline,
        Pipeline,
        PipelineContext,
    )

    assert Pipeline is LegacyPipeline
    assert CognitiveOrchestrationPipeline is LegacyPipeline
    assert PipelineContext is LegacyCtx


def test_pipeline_v2_context_module_reexports_only() -> None:
    """`sage.pipeline_v2.context.PipelineContext` is the SAME class object
    as `sage.pipeline.PipelineContext`.

    cgpro 2026-05-05 DESIGN lock: do NOT move the dataclass. Re-export
    only. Bench / dashboards compare against `__module__ ==
    "sage.pipeline"`; a true move would silently break those
    consumers.
    """
    from sage.pipeline import PipelineContext as LegacyCtx
    from sage.pipeline_v2.context import PipelineContext as V2Ctx

    assert V2Ctx is LegacyCtx
    # Sanity: __module__ is the legacy path. If a future Phase C
    # session moves the dataclass, this assertion intentionally
    # breaks loudly to force an ADR + bench-consumer audit.
    assert LegacyCtx.__module__ == "sage.pipeline"


# ────────────────────────────────────────────────────────────────────
# 2. Wrapper contract: each function delegates to `pipeline._stage_<X>`
# ────────────────────────────────────────────────────────────────────


def test_classify_wrapper_runs_body_and_legacy_method_delegates_to_it() -> None:
    """Phase B inverted the direction (same as decompose): the wrapper
    now CONTAINS the body and `pipeline._stage_classify` delegates to
    it. Tests the AdaptiveRouter Priority-3 path (`_rust_router=None`,
    `router._knn=None`) — the simplest deterministic path through
    the body.

    Both direct-wrapper and legacy-delegator calls must produce the
    same `ctx.system` (driven by router.route(profile)) and the same
    `_last_runtime_routing_*` accessors.
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline, PipelineContext
    from sage.pipeline_v2 import classify as classify_mod

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._rust_router = None  # skip Priority 1
    # Priority 2 (kNN): router has no _knn attribute → falls through
    # to Priority 3 (AdaptiveRouter heuristic).
    pipeline.router = MagicMock()
    pipeline.router._knn = None
    pipeline.router.assess_complexity = MagicMock(return_value=SimpleNamespace(system=2))
    pipeline.router.route = MagicMock(
        return_value=SimpleNamespace(system=2, confidence=0.5, model_id="adaptive-test"),
    )
    pipeline._last_routing_decision = None
    pipeline._last_runtime_routing_source = ""
    pipeline._last_runtime_routing_confidence = None
    pipeline._last_runtime_routing_model_id = ""

    # Direct call to the wrapper:
    ctx_direct = PipelineContext(task="classify direct task")
    out_direct = classify_mod.classify(pipeline, ctx_direct)
    assert out_direct is ctx_direct
    assert out_direct.system == 2
    assert pipeline._last_runtime_routing_source == "adaptive_router"

    # Round-trip through the legacy delegator:
    pipeline._last_runtime_routing_source = ""  # reset
    ctx_round = PipelineContext(task="classify roundtrip task")
    out_round = pipeline._stage_classify(ctx_round)
    assert out_round is ctx_round
    assert out_round.system == 2
    assert pipeline._last_runtime_routing_source == "adaptive_router"


@pytest.mark.asyncio
async def test_decompose_wrapper_runs_body_and_legacy_method_delegates_to_it() -> None:
    """Phase B inverted the direction: the wrapper now CONTAINS the body, and
    `pipeline._stage_decompose` is a 1-line delegator that awaits this
    function.

    The Phase A version of this test asserted the wrapper delegated TO the
    legacy method. After the body move, the legacy method delegates TO the
    wrapper. This test proves both:
      1. Calling `decompose(pipeline, ctx)` directly executes the body
         (S1 short-circuit).
      2. Calling `pipeline._stage_decompose(ctx)` reaches this wrapper
         (round-trip through the delegator).

    Uses a real `Pipeline.__new__(Pipeline)` instance with `llm_provider=None`
    to force the trivial-DAG fallback path (no LLM call).
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline, PipelineContext
    from sage.pipeline_v2 import decompose as decompose_mod

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.llm_provider = None  # forces the trivial-DAG short-circuit

    # Direct call to the wrapper:
    ctx_direct = PipelineContext(task="x")
    ctx_direct.system = 1
    out_direct = await decompose_mod.decompose(pipeline, ctx_direct)
    assert out_direct is ctx_direct  # mutates in place
    assert out_direct.dag_features is not None
    assert out_direct.dag_features.omega == 1

    # Round-trip through the legacy delegator:
    ctx_round = PipelineContext(task="x")
    ctx_round.system = 1
    out_round = await pipeline._stage_decompose(ctx_round)
    assert out_round is ctx_round
    assert out_round.dag_features is not None
    assert out_round.dag_features.omega == 1


def test_select_topology_wrapper_runs_body_and_legacy_method_delegates_to_it() -> None:
    """Phase B inverted the direction (same as decompose / classify).

    The wrapper now CONTAINS the body. Tests the S1 fast-path early
    exit: `system=1`, `domain="general"`, and no DAG features set
    `ctx.topology=None` and return ctx. Both direct call and
    round-trip through `pipeline._stage_select_topology` produce the
    same identity result.
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline, PipelineContext
    from sage.pipeline_v2 import select_topology as st_mod

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.engine = None  # skip DynamicTopologyEngine path

    # Direct call:
    ctx_direct = PipelineContext(task="select topology direct task")
    ctx_direct.system = 1
    ctx_direct.domain = "general"
    ctx_direct.dag_features = None
    out_direct = st_mod.select_topology(pipeline, ctx_direct)
    assert out_direct is ctx_direct
    assert out_direct.topology is None

    # Round-trip through legacy delegator:
    ctx_round = PipelineContext(task="select topology roundtrip task")
    ctx_round.system = 1
    ctx_round.domain = "general"
    ctx_round.dag_features = None
    out_round = pipeline._stage_select_topology(ctx_round)
    assert out_round is ctx_round
    assert out_round.topology is None


def test_assign_models_wrapper_runs_body_and_legacy_method_delegates_to_it() -> None:
    """Phase B inverted the direction (same as decompose / classify).

    The wrapper now CONTAINS the body. Tests the early-exit path:
    `ctx.topology is None` short-circuits to `return ctx` without
    touching the assigner. Both direct call and round-trip through
    `pipeline._stage_assign_models` produce the same identity result.
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline, PipelineContext
    from sage.pipeline_v2 import assign_models as am_mod

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.assigner = None  # forces early exit at line 2 of the body
    pipeline.provider_pool = None

    # Direct call:
    ctx_direct = PipelineContext(task="assign direct task")
    ctx_direct.topology = None  # forces early exit
    out_direct = am_mod.assign_models(pipeline, ctx_direct)
    assert out_direct is ctx_direct

    # Round-trip through legacy delegator:
    ctx_round = PipelineContext(task="assign roundtrip task")
    ctx_round.topology = None
    out_round = pipeline._stage_assign_models(ctx_round)
    assert out_round is ctx_round


@pytest.mark.asyncio
async def test_execute_wrapper_runs_body_and_legacy_method_delegates_to_it() -> None:
    """Phase B inverted the direction (same as assign_models / learn).

    The wrapper now CONTAINS the body. Tests the deterministic no-provider
    single-agent path. Both direct call and round-trip through
    `pipeline._stage_execute` mark the same template.
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline, PipelineContext
    from sage.pipeline_v2 import execute as exec_mod

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._agent_loop = None
    pipeline.llm_provider = None
    pipeline.bandit = None
    pipeline.provider_pool = None

    async def _legacy_sentinel(
        _ctx: PipelineContext,
        event_log: Any | None = None,
        run_frame_builder: Any | None = None,
    ) -> PipelineContext:
        raise AssertionError("pipeline_v2.execute must not delegate to legacy stage")

    # Direct call to the wrapper: instance-level sabotage proves the wrapper
    # owns the body instead of delegating back to `pipeline._stage_execute`.
    pipeline._stage_execute = _legacy_sentinel
    ctx_direct = PipelineContext(task="execute direct task")
    ctx_direct.topology = None
    ctx_direct.cost_tracker = None
    ctx_direct.verification_passed = True
    out_direct = await exec_mod.execute(pipeline, ctx_direct)
    assert out_direct is ctx_direct
    assert out_direct.executed_template == "single_agent"
    assert out_direct.result == ""

    # Round-trip through the legacy delegator:
    delattr(pipeline, "_stage_execute")
    ctx_round = PipelineContext(task="execute roundtrip task")
    ctx_round.topology = None
    ctx_round.cost_tracker = None
    ctx_round.verification_passed = True
    out_round = await pipeline._stage_execute(ctx_round)
    assert out_round is ctx_round
    assert out_round.executed_template == "single_agent"
    assert out_round.result == ""


@pytest.mark.asyncio
async def test_learn_wrapper_runs_body_and_legacy_method_delegates_to_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase B inverted the direction (same as decompose / classify).

    The wrapper now CONTAINS the body. Tests the empty-result quality=0.0
    path with oracle disabled. Both direct call and round-trip through
    `pipeline._stage_learn` return None.
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline, PipelineContext
    from sage.pipeline_v2 import learn as learn_mod

    monkeypatch.setenv("SAGE_ORACLE", "0")

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.quality_estimator = None
    pipeline.engine = None
    pipeline.consolidator = None
    pipeline._rust_router = None
    pipeline.bandit = None
    pipeline.prm = None
    pipeline._task_count = 0
    pipeline._record_bandit_outcome_checked = MagicMock()
    pipeline._cancel_bandit_decision = MagicMock()
    pipeline._clear_bandit_decision = MagicMock()

    async def _legacy_sentinel(_ctx: PipelineContext) -> None:
        raise AssertionError("pipeline_v2.learn must not delegate to legacy stage")

    # Direct call to the wrapper: instance-level sabotage proves the wrapper
    # owns the body instead of delegating back to `pipeline._stage_learn`.
    pipeline._stage_learn = _legacy_sentinel
    ctx_direct = PipelineContext(task="learn direct task")
    ctx_direct.result = ""
    out_direct = await learn_mod.learn(pipeline, ctx_direct)
    assert out_direct is None
    pipeline._record_bandit_outcome_checked.assert_called_once_with(ctx_direct, 0.0)

    # Round-trip through the legacy delegator:
    delattr(pipeline, "_stage_learn")
    pipeline._record_bandit_outcome_checked.reset_mock()
    ctx_round = PipelineContext(task="learn roundtrip task")
    ctx_round.result = ""
    out_round = await pipeline._stage_learn(ctx_round)
    assert out_round is None
    pipeline._record_bandit_outcome_checked.assert_called_once_with(ctx_round, 0.0)


# ────────────────────────────────────────────────────────────────────
# 3. Async-vs-sync contract: wrappers preserve the legacy method's
#    sync/async nature. cgpro DESIGN trap #1: a Phase B delegator
#    that forgets `await` returns a coroutine silently.
# ────────────────────────────────────────────────────────────────────


def test_async_wrappers_are_coroutine_functions() -> None:
    """`decompose`, `execute`, `learn` MUST be coroutine functions."""
    from sage.pipeline_v2 import decompose as decompose_mod
    from sage.pipeline_v2 import execute as execute_mod
    from sage.pipeline_v2 import learn as learn_mod

    assert inspect.iscoroutinefunction(decompose_mod.decompose), (
        "pipeline_v2.decompose.decompose must be `async def` to match "
        "legacy `pipeline._stage_decompose`."
    )
    assert inspect.iscoroutinefunction(execute_mod.execute), (
        "pipeline_v2.execute.execute must be `async def`."
    )
    assert inspect.iscoroutinefunction(learn_mod.learn), (
        "pipeline_v2.learn.learn must be `async def`."
    )


def test_sync_wrappers_are_not_coroutine_functions() -> None:
    """`classify`, `select_topology`, `assign_models` MUST be sync."""
    from sage.pipeline_v2 import assign_models as am_mod
    from sage.pipeline_v2 import classify as classify_mod
    from sage.pipeline_v2 import select_topology as st_mod

    assert not inspect.iscoroutinefunction(classify_mod.classify), (
        "pipeline_v2.classify.classify must be sync to match legacy."
    )
    assert not inspect.iscoroutinefunction(st_mod.select_topology), (
        "pipeline_v2.select_topology.select_topology must be sync."
    )
    assert not inspect.iscoroutinefunction(am_mod.assign_models), (
        "pipeline_v2.assign_models.assign_models must be sync."
    )


# ────────────────────────────────────────────────────────────────────
# 4. Circular-import guard: pipeline_v2 imports must not cause
#    pipeline.py to import pipeline_v2 at top level.
# ────────────────────────────────────────────────────────────────────


def test_pipeline_top_level_pipeline_v2_imports_are_allowlisted() -> None:
    """`sage.pipeline` may only top-level import the `pipeline_v2.context` module.

    Cycle-13 K Phase 2.1 Step E1 (cgpro `cgpro_phase21_facade_rewrite_20260506`
    round-3 Q5 + round-4 OPTION_3, 2026-05-06): Phase 2.1 deliberately
    introduces a single top-level re-export
    ``from sage.pipeline_v2.context import PipelineContext`` so the
    canonical dataclass body lives in `sage.pipeline_v2.context` and the
    legacy ``from sage.pipeline import PipelineContext`` still resolves
    (cgpro round-3 Q4 backward-compat lock).

    All OTHER `pipeline_v2` symbols (the 6 stage modules, the
    orchestrator, the helper modules) MUST remain LOCAL imports inside
    delegator method bodies. The PEP 562 lazy re-export in
    `pipeline_v2/__init__.py` (Step E0) keeps the dependency graph
    acyclic at module-load time.

    cgpro DESIGN trap #4 (Phase A/B): top-level circular import =
    partial-initialization risk; Phase B delegators MUST use local
    imports (``from sage.pipeline_v2.<x> import <fn>`` inside the
    method body), not top-level.
    """
    import pathlib

    pipeline_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src" / "sage" / "pipeline.py"
    )
    assert pipeline_path.exists()
    text = pipeline_path.read_text(encoding="utf-8")

    # Allowlisted top-level pipeline_v2 imports (Phase 2.1 Step E1):
    allowlist_top_level = (
        "\nfrom sage.pipeline_v2.context import PipelineContext  # noqa: E402",
    )

    # Forbidden top-level patterns. Walk each occurrence and assert
    # it matches the allowlist exactly. Any non-allowlisted top-level
    # `\nfrom sage.pipeline_v2` is a partial-init risk regression.
    forbidden_prefixes = ("\nimport sage.pipeline_v2", "\nfrom sage.pipeline_v2")
    pos = 0
    while True:
        next_idx = -1
        matched_prefix: str | None = None
        for prefix in forbidden_prefixes:
            idx = text.find(prefix, pos)
            if idx != -1 and (next_idx == -1 or idx < next_idx):
                next_idx = idx
                matched_prefix = prefix
        if next_idx == -1:
            break
        # Extract the full line starting at `next_idx + 1` (skip the
        # leading newline) up to the next newline.
        line_end = text.find("\n", next_idx + 1)
        if line_end == -1:
            line_end = len(text)
        offending_line = text[next_idx:line_end]  # includes leading '\n'
        if not any(offending_line.startswith(allowed) for allowed in allowlist_top_level):
            raise AssertionError(
                f"Disallowed top-level pipeline_v2 import in pipeline.py:\n"
                f"  {offending_line.lstrip()!r}\n"
                f"Phase 2.1 Step E1 allowlist (cgpro round-3 Q5 + round-4 "
                f"OPTION_3) permits only `from sage.pipeline_v2.context "
                f"import PipelineContext # noqa: E402` at top level. All "
                f"other pipeline_v2 symbols must use LOCAL imports inside "
                f"delegator method bodies (cgpro DESIGN trap #4)."
            )
        assert matched_prefix is not None
        pos = next_idx + len(matched_prefix)


# ────────────────────────────────────────────────────────────────────
# 5. Smoke: the wrappers don't break sage.pipeline.run() boot.
#    Trivial — a MagicMock-based delegation test would pass even if
#    the package had an ImportError. This test imports for real.
# ────────────────────────────────────────────────────────────────────


def test_pipeline_v2_imports_clean_after_pipeline_loaded() -> None:
    """Loading sage.pipeline first, then sage.pipeline_v2, works."""
    import sage.pipeline  # noqa: F401
    import sage.pipeline_v2

    assert hasattr(sage.pipeline_v2, "Pipeline")
    assert hasattr(sage.pipeline_v2, "PipelineContext")


def test_pipeline_v2_imports_clean_when_loaded_first() -> None:
    """Loading sage.pipeline_v2 first transitively loads sage.pipeline.
    Both end up healthy."""
    import sage.pipeline_v2
    import sage.pipeline

    assert sage.pipeline_v2.Pipeline is sage.pipeline.CognitiveOrchestrationPipeline


# ────────────────────────────────────────────────────────────────────
# 6. End-to-end smoke: wrappers + real pipeline = same result as
#    direct legacy method call.
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decompose_wrapper_matches_legacy_method_on_real_pipeline() -> None:
    """Real `Pipeline.__new__` instance: decompose wrapper produces same
    output as direct `_stage_decompose` call.

    This is the smallest end-to-end smoke that proves Phase A
    delegation didn't introduce a logic bug. Stage 1 is chosen
    because it has the fewest dependencies (just dag_features
    population on S1 short-circuit).
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline, PipelineContext
    from sage.pipeline_v2.decompose import decompose

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.llm_provider = None  # forces the trivial-DAG path

    ctx_legacy = PipelineContext(task="def add(a, b):\n    return a + b")
    ctx_legacy.system = 1
    ctx_wrapper = PipelineContext(task="def add(a, b):\n    return a + b")
    ctx_wrapper.system = 1

    legacy_result = await pipeline._stage_decompose(ctx_legacy)
    wrapper_result = await decompose(pipeline, ctx_wrapper)

    # Both must populate dag_features identically.
    assert legacy_result.dag_features is not None
    assert wrapper_result.dag_features is not None
    assert legacy_result.dag_features.omega == wrapper_result.dag_features.omega
    assert legacy_result.dag_features.delta == wrapper_result.dag_features.delta
    assert legacy_result.dag_features.gamma == wrapper_result.dag_features.gamma
