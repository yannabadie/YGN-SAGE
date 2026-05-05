"""Stage 3 — ASSIGN_MODELS.

Per ADR-015 + cgpro 2026-05-05 DESIGN lock
(`cgpro_pi_mono_pivot_20260505`): cycle-12 Phase B moves the body of
`CognitiveOrchestrationPipeline._stage_assign_models` here as a
module-level function. Legacy method becomes a 1-line LOCAL-import
delegator.

Stage 3 contract preserved:
  - Same Rust `ModelAssigner.assign_models` call signature (topology,
    domain, budget, hints_list, task_system) per F7 wiring (2026-04-17).
  - Same node assignment recording loop into `ctx.assignments`.
  - Same provider-pool dead-model fallback (replaces unavailable
    models with the default LLM config model_id).
  - Same non-blocking formal verification call to
    `self._verify_assignment_formal(ctx)`.
  - Same diagnostic call to `self._log_model_assigner_chosen_fallback(ctx)`
    when the assigner has no `_score_candidates` (T5 diagnostic for
    opaque Rust assigners).

Helpers stay on `CognitiveOrchestrationPipeline` per cgpro DESIGN
trap #6 — `_log_model_assigner_chosen_fallback` and
`_verify_assignment_formal` remain methods of the pipeline class. The
body accesses them via `self.<helper>(ctx)` after the `self = pipeline`
shim. Helper ownership migration is Phase C territory.

cgpro DESIGN trap #3 (`__file__` drift): `_load_model_catalog` (which
uses `Path(__file__).parent.parent.parent / "config" / "cards.toml"`)
is a SEPARATE method on the class, NOT inside Stage 3 — so this move
does not touch it. It stays in `sage.pipeline`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext

# Per-module logger MUST stay "sage.pipeline" for trace-grep
# compatibility with the cycle-9 ledger consumers.
log = logging.getLogger("sage.pipeline")


def assign_models(
    pipeline: "CognitiveOrchestrationPipeline", ctx: "PipelineContext",
) -> "PipelineContext":
    """Stage 3: Assign model_id to each topology node.

    Body moved from `sage.pipeline.CognitiveOrchestrationPipeline._stage_assign_models`
    in cycle-12 Phase B. Behavior preserved byte-identically.
    """
    self = pipeline
    from sage.observability.spans import sage_span
    with sage_span("sage.assign_models", op="sage.assign_models"):
        if ctx.topology is None or self.assigner is None:
            return ctx

        try:
            # Pass provider hints from Path 6 policy output (multi-provider dimension).
            hints_list = (
                list(ctx.provider_hints.items()) if ctx.provider_hints else None
            )
            # F7: forward the OVERALL task tier so the Rust ModelAssigner can
            # promote producer-role nodes (planner, coder, worker, verifier)
            # that the template hardcoded at a low system tier. Without this,
            # an S3 SWE-bench task's planner (template system=1) was matched
            # against S1 affinity and picked a flash-lite model — see
            # docs/benchmarks/2026-04-17-swebench-smoke-debug.md.
            task_system = ctx.system if ctx.system in (1, 2, 3) else None
            n_assigned = self.assigner.assign_models(
                ctx.topology,
                ctx.domain,
                ctx.budget,
                hints_list,
                task_system,
            )
            log.info(
                "Assigned models to %d nodes (domain=%s, budget=%.2f, task_system=%s, provider_hints=%d)",
                n_assigned,
                ctx.domain,
                ctx.budget,
                task_system,
                len(ctx.provider_hints),
            )

            # Record assignments for observability
            node_count = (
                ctx.topology.node_count()
                if hasattr(ctx.topology, "node_count")
                else 0
            )
            for i in range(node_count):
                node = (
                    ctx.topology.get_node(i)
                    if hasattr(ctx.topology, "get_node")
                    else None
                )
                if node:
                    ctx.assignments[i] = getattr(node, "model_id", "")
            self._log_model_assigner_chosen_fallback(ctx)
        except (ImportError, RuntimeError) as exc:
            log.warning("Stage 3 assign failed: %s", exc)

        # Bandit feedback: Thompson sampling already handles exploration/exploitation
        # via the Beta posterior. No pre-execution budget reduction — it creates a
        # self-degrading loop (Audit2 + Audit3 confirmed). The bandit learns post-
        # execution in Stage 5 (LEARN) and naturally deprioritizes bad arms.

        # Filter out models whose provider is dead (health check or circuit breaker)
        if self.provider_pool and hasattr(self.provider_pool, 'is_model_available'):
            for node_idx, model_id in list(ctx.assignments.items()):
                if not model_id:
                    continue
                if not self.provider_pool.is_model_available(model_id):
                    provider_name = self.provider_pool.infer_provider(model_id)
                    default_model = getattr(self.llm_config, 'model', '') if self.llm_config else ''
                    if default_model:
                        log.info(
                            "Stage 3: %s provider unavailable, "
                            "node %d reassigned %s -> %s",
                            provider_name, node_idx, model_id, default_model,
                        )
                        ctx.assignments[node_idx] = default_model
                        if hasattr(ctx.topology, 'set_node_model_id'):
                            ctx.topology.set_node_model_id(node_idx, default_model)

        # Formal verification (non-blocking): prove every node has a valid provider
        try:
            self._verify_assignment_formal(ctx)
        except (ImportError, RuntimeError) as exc:
            log.warning("Stage 3 formal verification error (non-blocking): %s", exc)

        return ctx


__all__ = ["assign_models"]
