"""`pipeline_v2` stage-module ownership + import-graph invariants.

Cycle-13 K Phase 2.2 (cgpro `cgpro_phase22_test_rewrite_20260506` B2.c,
2026-05-06): this file owns the lasting structural invariants for
the `sage.pipeline_v2` package. Earlier transitional tests that
asserted equivalence across the migration boundary were retired in
B2.c — they captured a cycle-12/Phase-2.1 migration contract that no
longer applies once the legacy `_stage_<X>` methods are removed
(Stage C). The filename is kept to avoid noise but the scope is now
narrower.

What this file proves
=====================
1. The 6 stage modules exist and expose a callable named after the
   stage (`classify`, `decompose`, `select_topology`,
   `assign_models`, `execute`, `learn`). Calling each with
   `(pipeline, ctx)` runs the production body.
2. `PipelineContext` is the SAME class object whether imported as
   `from sage.pipeline import PipelineContext`,
   `from sage.pipeline_v2 import PipelineContext`, or
   `from sage.pipeline_v2.context import PipelineContext`. The
   dataclass body lives in `pipeline_v2/context.py` with explicit
   `PipelineContext.__module__ = "sage.pipeline"` so existing
   tests / bench / dashboards / observability assertions on the
   legacy module path keep passing byte-identical.
3. `pipeline_v2/__init__.py` uses PEP 562 module-level
   `__getattr__` to defer `from sage.pipeline import …` to
   attribute-access time — keeps the dependency graph acyclic
   once `pipeline.py` itself imports `pipeline_v2.context`.
4. The async stage callables (`decompose`, `execute`, `learn`)
   are true coroutine functions, not regular functions returning
   coroutines. Important because `orchestrator.run_internal`
   `await`s them; signature drift would compile silently and
   break event ordering at runtime.
5. The top-level pipeline_v2 import allowlist on `sage.pipeline`:
   only `from sage.pipeline_v2.context import PipelineContext` is
   permitted at module scope; every other `pipeline_v2` symbol
   must be locally imported inside a method body to avoid
   partial-initialisation cycles.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


# ────────────────────────────────────────────────────────────────────
# 1. Package structure: stage modules + orchestrator + helper modules
# ────────────────────────────────────────────────────────────────────


def test_pipeline_v2_package_exposes_expected_modules() -> None:
    """The 14 representative pipeline_v2 modules import without error.

    The package hosts:
      - 6 stage bodies (classify / decompose / select_topology /
        assign_models / execute / learn)
      - the orchestrator (run_internal body)
      - the PipelineContext dataclass body (context)
      - 6 helper modules (bandit_attribution / runtime_events /
        memory_gate / topology_helpers / costing) plus the
        package `__init__`.

    Every module owns real production code that `pipeline.py`
    locally imports inside method bodies.
    """
    # Each import is the assertion; F401 noqa makes ruff happy.
    import sage.pipeline_v2  # noqa: F401
    import sage.pipeline_v2.assign_models  # noqa: F401
    import sage.pipeline_v2.bandit_attribution  # noqa: F401
    import sage.pipeline_v2.classify  # noqa: F401
    import sage.pipeline_v2.context  # noqa: F401
    import sage.pipeline_v2.costing  # noqa: F401
    import sage.pipeline_v2.decompose  # noqa: F401
    import sage.pipeline_v2.execute  # noqa: F401
    import sage.pipeline_v2.learn  # noqa: F401
    import sage.pipeline_v2.memory_gate  # noqa: F401
    import sage.pipeline_v2.orchestrator  # noqa: F401
    import sage.pipeline_v2.runtime_events  # noqa: F401
    import sage.pipeline_v2.select_topology  # noqa: F401
    import sage.pipeline_v2.topology_helpers  # noqa: F401


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


def test_pipeline_v2_context_module_preserves_legacy_identity() -> None:
    """`pipeline_v2.context` owns the dataclass body while preserving legacy identity.

    Cycle-13 K Phase 2.1 Step E1 (cgpro
    `cgpro_phase21_facade_rewrite_20260506` round-3 Q4 + round-4
    OPTION_3, 2026-05-06) moved the canonical `PipelineContext`
    dataclass body to `sage.pipeline_v2.context`. Backward
    compatibility is preserved on three fronts:

      - `from sage.pipeline import PipelineContext` (legacy) and
        `from sage.pipeline_v2.context import PipelineContext`
        (canonical) reference the SAME class object (identity, not
        equality).
      - `PipelineContext.__module__ == "sage.pipeline"` is preserved
        via an explicit `setattr` in `pipeline_v2/context.py` so
        `repr(ctx)` still renders the legacy module path and bench /
        dashboard / observability consumers comparing the literal
        path keep matching.
      - Pickle support against the legacy module path is unchanged —
        consumers that pickled `PipelineContext` instances pre-Phase-2.1
        keep deserialising correctly.

    A future regression of any of those three contracts is
    deliberately a loud failure here: bench/dashboards depend on the
    legacy module-path string, and the cgpro round-3 Q4 backward-
    compat lock was the explicit precondition for this move.
    """
    from sage.pipeline import PipelineContext as LegacyCtx
    from sage.pipeline_v2.context import PipelineContext as V2Ctx

    assert V2Ctx is LegacyCtx
    assert LegacyCtx.__module__ == "sage.pipeline"


# ────────────────────────────────────────────────────────────────────
# 2. Stage-module bodies execute correctly when called as
#    `pipeline_v2.<stage>.<fn>(pipeline, ctx)`. Each test exercises
#    the smallest deterministic path through the body.
# ────────────────────────────────────────────────────────────────────


def test_classify_module_function_runs_body() -> None:
    """`classify(pipeline, ctx)` exercises the AdaptiveRouter Priority-3 path.

    `_rust_router=None` skips Priority 1; the router has no `_knn`
    attribute so Priority 2 falls through to Priority 3 (AdaptiveRouter
    heuristic). The function must populate `ctx.system` from
    `router.route(profile)` and stamp the runtime-routing accessors.
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline, PipelineContext
    from sage.pipeline_v2 import classify as classify_mod

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._rust_router = None  # skip Priority 1
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

    ctx = PipelineContext(task="classify direct task")
    out = classify_mod.classify(pipeline, ctx)
    assert out is ctx
    assert out.system == 2
    assert pipeline._last_runtime_routing_source == "adaptive_router"


@pytest.mark.asyncio
async def test_decompose_module_function_runs_body() -> None:
    """`decompose(pipeline, ctx)` exercises the trivial-DAG short-circuit.

    `llm_provider=None` forces the no-LLM path. The function mutates
    `ctx.dag_features` in place and returns the same context.
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline, PipelineContext
    from sage.pipeline_v2 import decompose as decompose_mod

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.llm_provider = None  # forces the trivial-DAG short-circuit

    ctx = PipelineContext(task="x")
    ctx.system = 1
    out = await decompose_mod.decompose(pipeline, ctx)
    assert out is ctx  # mutates in place
    assert out.dag_features is not None
    assert out.dag_features.omega == 1


def test_select_topology_module_function_runs_body() -> None:
    """`select_topology(pipeline, ctx)` exercises the S1 fast-path early exit.

    `system=1`, `domain="general"`, and no DAG features set
    `ctx.topology=None` and return ctx unchanged.
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline, PipelineContext
    from sage.pipeline_v2 import select_topology as st_mod

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.engine = None  # skip DynamicTopologyEngine path

    ctx = PipelineContext(task="select topology direct task")
    ctx.system = 1
    ctx.domain = "general"
    ctx.dag_features = None
    out = st_mod.select_topology(pipeline, ctx)
    assert out is ctx
    assert out.topology is None


def test_assign_models_module_function_runs_body() -> None:
    """`assign_models(pipeline, ctx)` exercises the early-exit path.

    `ctx.topology is None` short-circuits to `return ctx` without
    touching the assigner.
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline, PipelineContext
    from sage.pipeline_v2 import assign_models as am_mod

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.assigner = None  # forces early exit
    pipeline.provider_pool = None

    ctx = PipelineContext(task="assign direct task")
    ctx.topology = None  # forces early exit
    out = am_mod.assign_models(pipeline, ctx)
    assert out is ctx


@pytest.mark.asyncio
async def test_execute_module_function_runs_body() -> None:
    """`execute(pipeline, ctx)` exercises the no-provider single-agent path.

    Returns the same context with `executed_template == "single_agent"`
    and an empty result string.
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline, PipelineContext
    from sage.pipeline_v2 import execute as exec_mod

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._agent_loop = None
    pipeline.llm_provider = None
    pipeline.bandit = None
    pipeline.provider_pool = None

    ctx = PipelineContext(task="execute direct task")
    ctx.topology = None
    ctx.cost_tracker = None
    ctx.verification_passed = True
    out = await exec_mod.execute(pipeline, ctx)
    assert out is ctx
    assert out.executed_template == "single_agent"
    assert out.result == ""


@pytest.mark.asyncio
async def test_learn_module_function_runs_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`learn(pipeline, ctx)` exercises the empty-result quality=0.0 path.

    Oracle gating disabled via `SAGE_ORACLE=0`. Returns None and records
    the outcome with quality 0.0 against the configured bandit hooks.
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

    ctx = PipelineContext(task="learn direct task")
    ctx.result = ""
    out = await learn_mod.learn(pipeline, ctx)
    assert out is None
    pipeline._record_bandit_outcome_checked.assert_called_once_with(ctx, 0.0)


# ────────────────────────────────────────────────────────────────────
# 3. Async-vs-sync contract: signature drift on a stage module would
#    let `await fn(...)` silently return a coroutine instead of
#    executing the body. The orchestrator awaits each async stage; a
#    sync regression would break event ordering at runtime.
# ────────────────────────────────────────────────────────────────────


def test_async_stage_modules_are_coroutine_functions() -> None:
    """`decompose`, `execute`, `learn` MUST be coroutine functions."""
    from sage.pipeline_v2 import decompose as decompose_mod
    from sage.pipeline_v2 import execute as execute_mod
    from sage.pipeline_v2 import learn as learn_mod

    assert inspect.iscoroutinefunction(decompose_mod.decompose), (
        "pipeline_v2.decompose.decompose must be `async def`."
    )
    assert inspect.iscoroutinefunction(execute_mod.execute), (
        "pipeline_v2.execute.execute must be `async def`."
    )
    assert inspect.iscoroutinefunction(learn_mod.learn), (
        "pipeline_v2.learn.learn must be `async def`."
    )


def test_sync_stage_modules_are_not_coroutine_functions() -> None:
    """`classify`, `select_topology`, `assign_models` MUST be sync."""
    from sage.pipeline_v2 import assign_models as am_mod
    from sage.pipeline_v2 import classify as classify_mod
    from sage.pipeline_v2 import select_topology as st_mod

    assert not inspect.iscoroutinefunction(classify_mod.classify), (
        "pipeline_v2.classify.classify must be sync."
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

    A single top-level re-export
    ``from sage.pipeline_v2.context import PipelineContext`` is
    permitted so the canonical dataclass body lives in
    `sage.pipeline_v2.context` and the legacy
    ``from sage.pipeline import PipelineContext`` still resolves.

    All OTHER `pipeline_v2` symbols (the 6 stage modules, the
    orchestrator, the helper modules) MUST remain locally imported
    inside method bodies. The PEP 562 lazy re-export in
    `pipeline_v2/__init__.py` keeps the dependency graph acyclic at
    module-load time. Top-level circular imports here would create
    a partial-initialisation risk.
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
# 6. End-to-end smoke: stage-module body produces a real DAG on a
#    real `Pipeline.__new__` instance.
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decompose_module_function_on_real_pipeline() -> None:
    """`decompose(pipeline, ctx)` populates dag_features on a real instance.

    Smallest end-to-end smoke that proves the body wires up correctly.
    Stage 1 is chosen because it has the fewest dependencies (just
    dag_features population on the S1 short-circuit).
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline, PipelineContext
    from sage.pipeline_v2.decompose import decompose

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.llm_provider = None  # forces the trivial-DAG path

    ctx = PipelineContext(task="def add(a, b):\n    return a + b")
    ctx.system = 1

    out = await decompose(pipeline, ctx)
    assert out.dag_features is not None
    assert out.dag_features.omega is not None
    assert out.dag_features.delta is not None
    assert out.dag_features.gamma is not None
