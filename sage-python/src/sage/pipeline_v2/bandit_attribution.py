"""Cycle-13 K Phase 2.1 Step A2 — bandit-attribution lifecycle module.

cgpro DESIGN_LOCKED 2026-05-06 (`cgpro_phase21_facade_rewrite_20260506`)
fully promoted this module from the Phase A placeholder ("file
intentionally empty") to its real home. Per ADR-015 §"Module
boundaries" + invariant 6 in `docs/contracts/runtime-integrity-ledger.md`
("Bandit attribution"), this module now owns the bandit decision_id
lifecycle:

  - context vector construction (`bandit_task_context`)
  - single-agent vs multi-agent execution detection
    (`is_single_agent_execution`)
  - decision-id state mutation (`clear_bandit_decision`)
  - explicit cancel of in-flight Rust bandit decisions
    (`cancel_bandit_decision`)
  - oracle-gated outcome recording with attribution-mismatch
    cancellation (`record_bandit_outcome_checked`)

Each module function takes the host `pipeline` instance as the first
positional argument so the corresponding `CognitiveOrchestrationPipeline`
methods can collapse to 1-line delegators while preserving:

  - `pipeline._<method>` mockability for ~5+ test files (test_oracle_stack
    monkeypatches `pipeline._record_bandit_outcome_checked`,
    test_pipeline_topology_skip_guardrails_decoupling calls
    `pipeline._is_single_agent_execution(ctx)` directly, etc.)
  - byte-identical cross-call traces inside
    `record_bandit_outcome_checked` — internal calls to
    `_cancel_bandit_decision` / `_clear_bandit_decision` /
    `_emit_bandit_attribution_mismatch` go through the pipeline
    instance methods so existing mocks fire identically

Logger uses ``sage.pipeline`` per cgpro Q7 trap "logger name drift" —
modules carved out of `pipeline.py` keep the legacy logger name so
trace-grep continuity is preserved across the refactor.

`_MULTI_NODE_ATTRIBUTION_TEMPLATES` is bandit-specific and is owned
here. Tests reference the constant by literal value
(`{"parallel", "parallel_fanout", "debate"}`); no test imports the
symbol directly, so no re-export is required.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping

from sage.runtime.oracle import oracle_enabled

if TYPE_CHECKING:
    from sage.pipeline import (  # noqa: F401 - TYPE_CHECKING import only
        BanditAttributionReasonCode,
        CognitiveOrchestrationPipeline,
        PipelineContext,
    )


log = logging.getLogger("sage.pipeline")


_MULTI_NODE_ATTRIBUTION_TEMPLATES: frozenset[str] = frozenset(
    {"parallel", "parallel_fanout", "debate"}
)


def bandit_task_context(
    pipeline: "CognitiveOrchestrationPipeline",  # noqa: ARG001 - pipeline arg preserved for symmetry with other helpers and future signature stability
    ctx: "PipelineContext",
) -> list[float]:
    """Return the [system, len(task), node_count] context vector for bandit features.

    Pure function over `ctx`; the `pipeline` argument is unused but
    preserved for signature symmetry with the other lifecycle helpers
    (some of which need `pipeline._rust_router`).
    """
    return [
        float(ctx.system),
        float(len(ctx.task)),
        float(
            ctx.topology.node_count()
            if ctx.topology and hasattr(ctx.topology, "node_count")
            else 0
        ),
    ]


def is_single_agent_execution(
    pipeline: "CognitiveOrchestrationPipeline",  # noqa: ARG001 - signature symmetry
    ctx: "PipelineContext",
) -> bool:
    """Return True iff Stage 4 will take the single-agent bypass path."""
    return ctx.topology is None or (
        hasattr(ctx.topology, "node_count") and ctx.topology.node_count() <= 1
    )


def clear_bandit_decision(
    pipeline: "CognitiveOrchestrationPipeline",  # noqa: ARG001 - signature symmetry
    ctx: "PipelineContext",
) -> None:
    """Reset the per-run bandit decision metadata on `ctx` to the post-cancel default state."""
    ctx.bandit_decision_id = ""
    ctx.bandit_model_id = ""
    ctx.bandit_template = ""
    ctx.bandit_context = []
    ctx.bandit_attribution_state = "skipped"


def cancel_bandit_decision(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
    *,
    force: bool = False,
) -> bool:
    """Cancel the in-flight Rust bandit decision for `ctx`.

    Returns True iff the Rust SystemRouter reports the cancel succeeded.
    Returns False when there is no pending decision (and `force` is
    not set), when the SystemRouter does not expose
    `cancel_bandit_decision`, or when the underlying call raises.
    """
    decision_id = str(getattr(ctx, "bandit_decision_id", "") or "")
    if not decision_id and not force:
        return False
    rust_router = pipeline._rust_router
    if rust_router and hasattr(rust_router, "cancel_bandit_decision"):
        try:
            return bool(rust_router.cancel_bandit_decision(decision_id))
        except (ImportError, RuntimeError, ValueError) as exc:
            log.warning("Bandit decision cancel failed: %s", exc)
            return False
    log.warning("Bandit decision cancel unavailable: SystemRouter wrapper missing")
    return False


def record_bandit_outcome_checked(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
    quality: float,
) -> None:
    """Record a bandit outcome only when attribution is causal and oracle-trainable.

    The cross-calls to `_cancel_bandit_decision`, `_clear_bandit_decision`,
    and `_emit_bandit_attribution_mismatch` go through the pipeline
    instance methods (delegators) so existing mocks continue to fire.
    """
    if oracle_enabled():
        verdict = getattr(ctx, "oracle_verdict", None)
        if verdict is None or not verdict.trainable:
            pipeline._cancel_bandit_decision(ctx)
            pipeline._clear_bandit_decision(ctx)
            return

    if not getattr(ctx, "bandit_decision_id", ""):
        return

    raw_executed_model_ids = getattr(ctx, "executed_model_ids", [])
    if isinstance(raw_executed_model_ids, Mapping):
        raw_executed_model_ids = raw_executed_model_ids.values()
    executed_model_ids = [model_id for model_id in raw_executed_model_ids if model_id]
    if (
        len(set(executed_model_ids)) > 1
        or getattr(ctx, "executed_template", "") in _MULTI_NODE_ATTRIBUTION_TEMPLATES
    ):
        ctx.bandit_attribution_state = "skipped"
        pipeline._cancel_bandit_decision(ctx)
        pipeline._emit_bandit_attribution_mismatch(ctx, "multi_node_ambiguous")
        pipeline._clear_bandit_decision(ctx)
        return

    rust_router = pipeline._rust_router
    if not rust_router or not hasattr(rust_router, "record_outcome_checked"):
        log.warning(
            "Bandit outcome skipped: SystemRouter record_outcome_checked unavailable"
        )
        ctx.bandit_attribution_state = "mismatch"
        pipeline._cancel_bandit_decision(ctx)
        pipeline._emit_bandit_attribution_mismatch(ctx, "recorder_instance_mismatch")
        return

    try:
        rust_router.record_outcome_checked(
            ctx.bandit_decision_id,
            ctx.executed_model_id,
            ctx.executed_template,
            quality,
            ctx.cost,
            ctx.latency_ms,
        )
        ctx.bandit_attribution_state = "verified"
    except (ImportError, RuntimeError, ValueError) as exc:
        reason_code = pipeline._bandit_reason_from_exception(exc)
        ctx.bandit_attribution_state = "mismatch"
        pipeline._cancel_bandit_decision(ctx)
        pipeline._emit_bandit_attribution_mismatch(ctx, reason_code)


__all__ = [
    "bandit_task_context",
    "cancel_bandit_decision",
    "clear_bandit_decision",
    "is_single_agent_execution",
    "record_bandit_outcome_checked",
]
