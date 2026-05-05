"""Stage 1 — DECOMPOSE.

Per ADR-015 + ADR-016 + cgpro 2026-05-05 DESIGN lock
(`cgpro_pi_mono_pivot_20260505`): cycle-12 Phase B moves the body of
`CognitiveOrchestrationPipeline._stage_decompose` here as a module-
level function `decompose(pipeline, ctx)`. The legacy method becomes a
1-line delegator with a LOCAL import (no top-level circular import).

Phase B contract preserved:
  - Same return value (the same `PipelineContext` instance, mutated).
  - Same exception class hierarchy (only `RuntimeError` and
    `TimeoutError` from the planner are caught).
  - Same logger name (`sage.pipeline`'s `log`, NOT a new one — the
    cycle-9 trace ledger consumers grep on the source).
  - Same `dag_features` semantics: trivial `DAGFeatures(1, 1, 0.0)`
    for the S1 short-circuit and the LLM-unavailable / planner-error
    fallback paths.
  - Same `sage_span` instrumentation under `op="sage.decompose"`.

The 25 P9 phase 1 acceptance-gate tests are the byte-identical
verification — they call `pipeline._stage_decompose` (now the
delegator), which awaits this module's `decompose()`, which produces
the same context the legacy method produced.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Module globals that the moved body uses unqualified. cgpro DESIGN
# trap #2: when a body moves, unqualified names resolve in THIS
# module's namespace, not pipeline.py's. The originals live in
# `sage.pipeline_stages`.
from sage.pipeline_stages import DAGFeatures, compute_dag_features

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext

# Per-module logger name MUST stay "sage.pipeline" for trace-grep
# compatibility with the cycle-9 ledger consumers. The legacy method
# emitted under that name; the module function emits under the same
# name to keep telemetry byte-identical.
log = logging.getLogger("sage.pipeline")


async def decompose(
    pipeline: "CognitiveOrchestrationPipeline", ctx: "PipelineContext",
) -> "PipelineContext":
    """Stage 1: Decompose task into sub-tasks (S2/S3 only).

    Body moved from `sage.pipeline.CognitiveOrchestrationPipeline._stage_decompose`
    in cycle-12 Phase B (commit chain post `3a851db3`). Behavior
    preserved byte-identically.
    """
    self = pipeline
    from sage.observability.spans import sage_span
    with sage_span("sage.decompose", op="sage.decompose"):
        if ctx.system == 1:
            ctx.dag_features = DAGFeatures(omega=1, delta=1, gamma=0.0)
            return ctx

        # Try LLM decomposition via TaskPlanner if available
        try:
            from sage.contracts.planner import TaskPlanner

            planner = TaskPlanner()
            if self.llm_provider and hasattr(planner, "plan_auto"):
                result = await planner.plan_auto(ctx.task, self.llm_provider)
                ctx.task_dag = result.dag
                ctx.dag_features = compute_dag_features(result.dag)
            else:
                ctx.dag_features = DAGFeatures(omega=1, delta=1, gamma=0.0)
        except (RuntimeError, TimeoutError) as exc:
            log.warning("Stage 1 decompose failed: %s, using single-node DAG", exc)
            ctx.dag_features = DAGFeatures(omega=1, delta=1, gamma=0.0)

        return ctx


__all__ = ["decompose"]
