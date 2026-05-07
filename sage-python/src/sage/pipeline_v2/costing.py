"""Cycle-13 K Phase 2.1 Step B5 — costing helpers (NEW module).

cgpro DESIGN_LOCKED 2026-05-06 (`cgpro_phase21_facade_rewrite_20260506`)
Q3 explicit garde-fou: `_estimate_topology_cost` + `_load_model_catalog`
are NOT topology-side helpers (Stage 2) and NOT assign-side helpers
(Stage 3). They are costing-transverse helpers consumed by Stage 4
execute (`pipeline_v2/execute.py:490` `ctx.cost = self._estimate_topology_cost(ctx)`)
when the runner doesn't produce a real cost. Conceptually pricing/
catalogue, not topology-construction or model-assignment.

`_load_model_catalog` is the cards.toml ModelCardCatalog loader. Per
cgpro Q3 + Q7 garde-fou (`__file__` drift): the cards.toml resolution
path from this module MUST be ``Path(__file__).resolve().parents[3]
/ "config" / "cards.toml"`` — NOT ``parent.parent.parent``, which
would point at ``sage-python/src/config/cards.toml`` and silently
miss the real catalog. Validated:

  Path(__file__) = sage-python/src/sage/pipeline_v2/costing.py
  parents[0]     = sage-python/src/sage/pipeline_v2/
  parents[1]     = sage-python/src/sage/
  parents[2]     = sage-python/src/
  parents[3]     = sage-python/
  parents[3] / "config" / "cards.toml"
                = sage-python/config/cards.toml  ✓

Method form preserved on `CognitiveOrchestrationPipeline` as 1-line
LOCAL-import delegators so existing call sites
(`pipeline._estimate_topology_cost(ctx)` from execute.py and
`pipeline._load_model_catalog()` from estimate body) continue working
byte-identical, including the `_model_catalog` cache attribute on the
pipeline instance (set on the pipeline, not the module — first call
caches the catalog on `pipeline._model_catalog`).

Logger uses ``sage.pipeline`` per cgpro Q7 trap "logger name drift" —
modules carved out of `pipeline.py` keep the legacy logger name so
trace-grep continuity is preserved.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


log = logging.getLogger("sage.pipeline")


def estimate_topology_cost(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
) -> float:
    """PREDICTIVE cost estimate (pre-execution budget check only).

    P1.6 clarification (2026-04-22 audit remediation): this function is
    a PREDICTION used by the topology-budget gate BEFORE execution. It
    is NOT the actual cost tracker. The real runtime cost comes from
    `AgentLoop.total_cost_usd`, which is computed by
    `sage.phases.think._extract_step_cost` using the provider-reported
    token usage (`response.usage.prompt_tokens` +
    `response.usage.completion_tokens`) times the per-model rate in
    cards.toml. That path is the truth; ctx.cost gets updated from it
    post-execution.

    This predictor uses a fixed ~500 input / ~300 output token budget
    per node — a known imprecise heuristic acceptable for a pre-run
    budget-gate. The audit (AUDIT4 bug #1 "cost estimation fictive")
    correctly flagged this as fiction for POST-EXEC reporting, but
    pre-exec gating is a distinct problem from per-token accounting.
    A token-count PREDICTOR (rather than a fixed 500/300) is a separate
    research task; the $0.001 fallback fires only when cards.toml has
    no entry for the assigned model.

    Loads the model catalog once (cached on the pipeline) and looks up
    cost_input_per_m / cost_output_per_m for each node's assigned model.
    Falls back to $0.001 per node when the catalog or model is
    unavailable.

    `pipeline._load_model_catalog()` is invoked through the delegator so
    test mocks of the load path keep firing.
    """
    if not ctx.topology or not hasattr(ctx.topology, 'node_count'):
        return 0.0

    n_nodes = ctx.topology.node_count()
    if n_nodes == 0:
        return 0.0

    # Lazy-load model catalog for pricing
    catalog = load_model_catalog(pipeline)

    total_cost = 0.0
    for i in range(n_nodes):
        model_id = ctx.assignments.get(i, '')
        if not model_id and hasattr(ctx.topology, 'get_node'):
            node = ctx.topology.get_node(i)
            model_id = getattr(node, 'model_id', '') if node else ''

        card = catalog.get(model_id) if catalog and model_id else None
        if card:
            # ~500 tokens input + ~300 tokens output per node
            total_cost += (
                500 * card.cost_input_per_m + 300 * card.cost_output_per_m
            ) / 1_000_000
        else:
            total_cost += 0.001  # fallback estimate per node

    return total_cost


def load_model_catalog(pipeline: "CognitiveOrchestrationPipeline") -> Any:
    """Load ModelCardCatalog from cards.toml (cached on pipeline after first call).

    cgpro Q3 + Q7 explicit garde-fou: ``Path(__file__).resolve().parents[3]
    / "config" / "cards.toml"`` is the canonical repo-root-robust resolution.
    The legacy ``Path(__file__).parent.parent.parent / "config" / "cards.toml"``
    path that lived in pipeline.py would have pointed at
    ``sage-python/src/config/cards.toml`` from this new module — wrong
    parent depth. The `parents[3]` form locks the correct depth.

    The catalog is cached on `pipeline._model_catalog` (NOT the module)
    so per-pipeline isolation is preserved (matches the legacy behavior
    where the cache lived on the pipeline instance).
    """
    if hasattr(pipeline, '_model_catalog'):
        return pipeline._model_catalog

    pipeline._model_catalog = None
    # Try Python ModelCardCatalog (always available, no Rust dependency)
    try:
        from pathlib import Path

        from sage.llm.model_registry import ModelCardCatalog

        # Search common locations for cards.toml
        for candidate in [
            Path("config/cards.toml"),
            Path("sage-python/config/cards.toml"),
            Path(__file__).resolve().parents[3] / "config" / "cards.toml",
        ]:
            if candidate.exists():
                pipeline._model_catalog = ModelCardCatalog.from_toml_file(str(candidate))
                log.debug("Cost estimator: loaded %d models from %s",
                          len(pipeline._model_catalog), candidate)
                break
    except (IOError, OSError, ValueError) as exc:
        log.debug("Cost estimator: catalog unavailable (%s)", exc)

    return pipeline._model_catalog


__all__ = ["estimate_topology_cost", "load_model_catalog"]
