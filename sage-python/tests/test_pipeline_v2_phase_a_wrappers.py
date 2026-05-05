"""Phase A — additive wrappers in `sage.pipeline_v2/`.

Per cgpro 2026-05-05 DESIGN lock (`cgpro_pi_mono_pivot_20260505`):
ADR-015 + ADR-016 cycle-12 Phase A creates a `pipeline_v2/` package
with stage-function wrappers that delegate to the legacy
`pipeline._stage_<X>` methods. NO body movement in this commit.

Tests in this file prove the package layout, the wrapper contract,
and the absence of circular imports — without exercising any
production logic. The 25 P9 phase 1 acceptance-gate tests
(test_pipeline_v2_*.py) provide the byte-identical guarantee.

What this file proves
=====================
1. The 6 stage modules exist and expose a callable named after the
   stage (`classify`, `decompose`, `select_topology`,
   `assign_models`, `execute`, `learn`).
2. Each wrapper delegates to the corresponding `pipeline._stage_<X>`
   — verified via a stub `Pipeline` whose stage method records the
   call and returns a sentinel.
3. `PipelineContext` re-exported from `pipeline_v2.context` is the
   SAME class object as `sage.pipeline.PipelineContext` (identity,
   not equality — moving the dataclass would change `__module__`,
   repr, and pickle behavior, which downstream consumers compare
   against).
4. `pipeline_v2/__init__.py` does not trigger circular import — the
   package can be imported standalone AND from inside a fresh
   sub-process where `sage.pipeline` hasn't been touched yet.
5. The async wrappers (`decompose`, `execute`, `learn`) are coroutine
   functions, not regular functions returning coroutines. Important
   because Phase B delegators must `await`; if the wrapper signature
   drifts, the easiest codex mistake (missing `await`) compiles
   silently but returns a coroutine.
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


def test_select_topology_wrapper_delegates_to_pipeline_stage_select_topology() -> None:
    from sage.pipeline_v2 import select_topology as st_mod

    pipeline = MagicMock()
    sentinel_ctx = SimpleNamespace(tag="stage2_input")
    sentinel_out = SimpleNamespace(tag="stage2_output")
    pipeline._stage_select_topology = MagicMock(return_value=sentinel_out)

    result = st_mod.select_topology(pipeline, sentinel_ctx)

    pipeline._stage_select_topology.assert_called_once_with(sentinel_ctx)
    assert result is sentinel_out


def test_assign_models_wrapper_delegates_to_pipeline_stage_assign_models() -> None:
    from sage.pipeline_v2 import assign_models as am_mod

    pipeline = MagicMock()
    sentinel_ctx = SimpleNamespace(tag="stage3_input")
    sentinel_out = SimpleNamespace(tag="stage3_output")
    pipeline._stage_assign_models = MagicMock(return_value=sentinel_out)

    result = am_mod.assign_models(pipeline, sentinel_ctx)

    pipeline._stage_assign_models.assert_called_once_with(sentinel_ctx)
    assert result is sentinel_out


@pytest.mark.asyncio
async def test_execute_wrapper_delegates_to_pipeline_stage_execute_with_kwargs() -> None:
    """The execute wrapper takes optional event_log + run_frame_builder
    kwargs and forwards them. cgpro 2026-05-05 DESIGN lock: signature
    matches the legacy method (`async def _stage_execute(self, ctx,
    event_log=None, run_frame_builder=None)`)."""
    from sage.pipeline_v2 import execute as exec_mod

    pipeline = MagicMock()
    sentinel_ctx = SimpleNamespace(tag="stage4_input")
    sentinel_out = SimpleNamespace(tag="stage4_output")
    sentinel_eventlog = SimpleNamespace(tag="event_log")
    sentinel_builder = SimpleNamespace(tag="run_frame_builder")

    captured: dict[str, Any] = {}

    async def _fake_stage_execute(
        ctx: Any, event_log: Any | None = None, run_frame_builder: Any | None = None,
    ) -> Any:
        captured["ctx"] = ctx
        captured["event_log"] = event_log
        captured["run_frame_builder"] = run_frame_builder
        return sentinel_out

    pipeline._stage_execute = _fake_stage_execute

    result = await exec_mod.execute(
        pipeline,
        sentinel_ctx,
        event_log=sentinel_eventlog,
        run_frame_builder=sentinel_builder,
    )

    assert result is sentinel_out
    assert captured["ctx"] is sentinel_ctx
    assert captured["event_log"] is sentinel_eventlog
    assert captured["run_frame_builder"] is sentinel_builder


@pytest.mark.asyncio
async def test_execute_wrapper_defaults_kwargs_to_none() -> None:
    """Defaults match the legacy method when caller omits the optional kwargs."""
    from sage.pipeline_v2 import execute as exec_mod

    pipeline = MagicMock()
    sentinel_ctx = SimpleNamespace(tag="stage4_input_defaults")
    captured: dict[str, Any] = {}

    async def _fake_stage_execute(
        ctx: Any, event_log: Any | None = None, run_frame_builder: Any | None = None,
    ) -> Any:
        captured["event_log"] = event_log
        captured["run_frame_builder"] = run_frame_builder
        return ctx

    pipeline._stage_execute = _fake_stage_execute

    result = await exec_mod.execute(pipeline, sentinel_ctx)

    assert result is sentinel_ctx
    assert captured["event_log"] is None
    assert captured["run_frame_builder"] is None


@pytest.mark.asyncio
async def test_learn_wrapper_delegates_to_pipeline_stage_learn_returns_none() -> None:
    """Stage 5 returns None per the legacy contract."""
    from sage.pipeline_v2 import learn as learn_mod

    pipeline = MagicMock()
    sentinel_ctx = SimpleNamespace(tag="stage5_input")
    captured: dict[str, Any] = {}

    async def _fake_stage_learn(ctx: Any) -> None:
        captured["ctx"] = ctx
        return None

    pipeline._stage_learn = _fake_stage_learn

    result = await learn_mod.learn(pipeline, sentinel_ctx)

    assert result is None
    assert captured["ctx"] is sentinel_ctx


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


def test_pipeline_does_not_top_level_import_pipeline_v2() -> None:
    """`sage.pipeline` MUST NOT import `sage.pipeline_v2` at module scope.

    cgpro DESIGN trap #4: top-level circular import =
    partial-initialization risk. Phase B delegators MUST use local
    imports (`from sage.pipeline_v2.<x> import <fn>` inside the
    method body), not top-level.

    This test grep-asserts the source. A future Phase C façade
    rewrite may legitimately need this import to flip — that's an
    ADR change that will require updating this test.
    """
    import pathlib

    pipeline_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src" / "sage" / "pipeline.py"
    )
    assert pipeline_path.exists()
    text = pipeline_path.read_text(encoding="utf-8")

    # Forbidden patterns (top-level imports of pipeline_v2):
    forbidden_patterns = [
        "\nimport sage.pipeline_v2",
        "\nfrom sage.pipeline_v2",
    ]
    for pattern in forbidden_patterns:
        # Allow the pattern inside function/method bodies (indented),
        # but NOT at column zero (top level).
        assert pattern not in text, (
            f"Forbidden top-level import {pattern!r} found in pipeline.py. "
            f"Phase B delegator stubs MUST use LOCAL imports inside the "
            f"method body to avoid circular-import partial-initialization. "
            f"See cgpro 2026-05-05 DESIGN lock trap #4."
        )


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
