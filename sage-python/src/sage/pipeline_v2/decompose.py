"""Stage 1 — DECOMPOSE.

Module function `decompose(pipeline, ctx)` is the canonical Stage 1
entry point; the orchestrator awaits it directly with the pipeline
instance as first argument.

Stage 1 contract:
  - Returns the same `PipelineContext` instance, mutated in place.
  - Exception strategy: only `RuntimeError` and `TimeoutError` from
    the planner are caught.
  - Logger name `sage.pipeline` (cycle-9 trace ledger consumers grep
    on the source).
  - `dag_features` semantics: trivial `DAGFeatures(1, 1, 0.0)` for
    the S1 short-circuit and the LLM-unavailable / planner-error
    fallback paths.
  - `sage_span` instrumentation under `op="sage.decompose"`.
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
    """Stage 1: Decompose task into sub-tasks (S2/S3 only)."""
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
        except RuntimeError as exc:
            from sage.pipeline_v2.provider_policy import ProviderPolicyViolation

            if isinstance(exc, ProviderPolicyViolation):
                raise
            log.warning("Stage 1 decompose failed: %s, using single-node DAG", exc)
            ctx.dag_features = DAGFeatures(omega=1, delta=1, gamma=0.0)
        except TimeoutError as exc:
            log.warning("Stage 1 decompose failed: %s, using single-node DAG", exc)
            ctx.dag_features = DAGFeatures(omega=1, delta=1, gamma=0.0)

        return ctx


__all__ = ["decompose"]
