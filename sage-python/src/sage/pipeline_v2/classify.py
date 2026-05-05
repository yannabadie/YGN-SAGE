"""Stage 0 — CLASSIFY.

Per ADR-015 + cgpro 2026-05-05 DESIGN lock
(`cgpro_pi_mono_pivot_20260505`): cycle-12 Phase B moves the body of
`CognitiveOrchestrationPipeline._stage_classify` here as a module-
level function `classify(pipeline, ctx)`. Legacy method becomes a
1-line delegator with a LOCAL import.

Stage 0 contract preserved:
  - Priority 1: Rust SystemRouter `route_integrated` (full
    integrated routing — sets `ctx.system`, `ctx.bandit_decision_id`,
    `ctx.bandit_model_id`, `ctx.bandit_template`,
    `ctx.bandit_attribution_state`, plus the
    `_last_routing_decision` / `_last_runtime_routing_*` accessors
    that downstream stages + telemetry read).
  - Priority 2: Python kNN fallback (93.3% accuracy, Rust-accelerated
    embedding).
  - Priority 3: AdaptiveRouter heuristic (legacy — DEAD CODE per
    CLAUDE.md directive #4 but still wired as the emergency
    fallback).
  - Same `sage.observability.spans.sage_span` instrumentation under
    `op="sage.classify"`.
  - Same exception strategy: bare `except Exception` for Rust router
    (Rust extension types raise unconventional exceptions); narrow
    `(ImportError, RuntimeError)` for the Python paths.
  - Bandit-attribution lifecycle helpers (`_clear_bandit_decision`,
    `_cancel_bandit_decision`, `_emit_bandit_attribution_mismatch`)
    stay as methods on `CognitiveOrchestrationPipeline` per cgpro
    DESIGN: helper ownership migration is Phase C territory.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Module globals that the moved body uses unqualified. cgpro DESIGN
# trap #2: when a body moves, unqualified names resolve in THIS
# module's namespace.
from sage.pipeline_stages import _infer_domain

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext

# Per-module logger MUST stay "sage.pipeline" for trace-grep
# compatibility with the cycle-9 ledger consumers.
log = logging.getLogger("sage.pipeline")


def classify(
    pipeline: "CognitiveOrchestrationPipeline", ctx: "PipelineContext",
) -> "PipelineContext":
    """Stage 0: Classify task complexity, domain, and select an integrated routing model.

    Priority: Rust SystemRouter route_integrated > Python kNN fallback > heuristic.
    A14b makes the SystemRouter-owned bandit decision here and records it
    only after Stage 4 proves which model/template actually executed.

    Returns: system (S1/S2/S3), domain, and stores RoutingDecision for
    model selection in Stage 3 (ModelAssigner) and Stage 5 telemetry.

    Body moved from `sage.pipeline.CognitiveOrchestrationPipeline._stage_classify`
    in cycle-12 Phase B. Behavior preserved byte-identically.
    """
    self = pipeline
    from sage.observability.spans import sage_span
    with sage_span("sage.classify", op="sage.classify"):
        # Priority 1: Rust SystemRouter (full integrated routing)
        if self._rust_router:
            try:
                ctx.domain = _infer_domain(ctx.task)
                import importlib

                RoutingConstraints = getattr(
                    importlib.import_module("sage_core"),
                    "RoutingConstraints",
                )

                constraints = RoutingConstraints(
                    max_cost_usd=float(ctx.budget or 0.0),
                    max_latency_ms=0.0,
                    min_quality=0.0,
                    required_capabilities=[],
                    security_label="",
                    exploration_budget=0.1,
                    domain_hint=ctx.domain,
                )
                decision = self._rust_router.route_integrated(ctx.task, constraints, "")
                ctx.system = int(decision.system)
                decision_id = str(getattr(decision, "decision_id", "") or "")
                selected_template = str(
                    getattr(decision, "selected_template", "")
                    or getattr(decision, "template", "")
                    or ""
                )
                if selected_template:
                    ctx.bandit_decision_id = decision_id
                    ctx.bandit_model_id = str(getattr(decision, "model_id", "") or "")
                    ctx.bandit_template = selected_template
                    ctx.bandit_attribution_state = "pending" if decision_id else "skipped"
                else:
                    self._clear_bandit_decision(ctx)
                # Store decision for model selection + telemetry
                self._last_routing_decision = decision
                self._last_runtime_routing_source = "rust_system_router"
                self._last_runtime_routing_confidence = getattr(
                    decision,
                    "confidence",
                    None,
                )
                self._last_runtime_routing_model_id = getattr(decision, "model_id", "") or ""
                log.info(
                    "Stage 0: Rust routing → S%d model=%s (conf=%.2f, cost=%.4f)",
                    ctx.system, decision.model_id, decision.confidence, decision.estimated_cost,
                )
                return ctx
            except Exception as exc:
                log.warning("Stage 0: Rust SystemRouter failed (%s), falling back to Python", exc)
                self._cancel_bandit_decision(ctx, force=True)
                ctx.bandit_attribution_state = "skipped"
                self._emit_bandit_attribution_mismatch(ctx, "router_fallback_degraded")
                self._clear_bandit_decision(ctx)

        # Priority 2: Python kNN (93.3% accuracy, Rust-accelerated embedding)
        if self.router and hasattr(self.router, '_knn') and self.router._knn is not None:
            try:
                knn_result = self.router._knn.route(ctx.task)
                if knn_result is not None:
                    ctx.system = knn_result.system
                    log.info("Stage 0: kNN routing → S%d (conf=%.2f, %s)",
                             knn_result.system, knn_result.confidence, knn_result.method)
                    ctx.domain = _infer_domain(ctx.task)
                    self._last_runtime_routing_source = "knn"
                    self._last_runtime_routing_confidence = getattr(
                        knn_result,
                        "confidence",
                        None,
                    )
                    self._last_runtime_routing_model_id = ""
                    return ctx
            except (ImportError, RuntimeError) as exc:
                log.debug("Stage 0: kNN failed (%s), falling back", exc)

        # Priority 3: AdaptiveRouter heuristic
        if self.router:
            try:
                profile = self.router.assess_complexity(ctx.task)
                decision = self.router.route(profile)
                ctx.system = getattr(decision, "system", 2)
                self._last_runtime_routing_source = "adaptive_router"
                self._last_runtime_routing_confidence = getattr(decision, "confidence", None)
                self._last_runtime_routing_model_id = getattr(decision, "model_id", "") or ""
            except (ImportError, RuntimeError) as exc:
                log.warning("Stage 0 classify failed: %s, defaulting to S2", exc)
                ctx.system = 2
                self._last_runtime_routing_source = "default"
                self._last_runtime_routing_confidence = None
                self._last_runtime_routing_model_id = ""
        else:
            ctx.system = 2
            self._last_runtime_routing_source = "default"
            self._last_runtime_routing_confidence = None
            self._last_runtime_routing_model_id = ""

        ctx.domain = _infer_domain(ctx.task)
        return ctx


__all__ = ["classify"]
