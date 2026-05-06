"""Topology construction + structure-logging + budget helpers.

cgpro DESIGN_LOCKED 2026-05-06 (`cgpro_phase21_facade_rewrite_20260506`)
real home for topology-construction helpers carved out of
`CognitiveOrchestrationPipeline` so `pipeline.py` can shrink toward
< 300 LOC without losing the mockable surface that ~10 test files rely on.

Step A1 landed `build_topology_from_hint`. Step B5 (this commit) lands
the remaining 9 helpers per cgpro Q3 verdict ("ne pas faire monter
select_topology.py à ~950 LOC ; tri-split"):

  - candidate parsing : `topology_candidate_items`,
    `log_topology_candidates`, `candidate_text_attr`,
    `candidate_float_attr`, `candidate_node_count`
  - structure logging : `log_topology_structure` (gap 1+2 attribution)
  - budget gate       : `apply_topology_budget_and_cache`,
    `check_topology_budget`, `make_single_node_topology`

Stage 2 flow stays in `pipeline_v2/select_topology.py` (lisible).
The cost-side helpers (`_estimate_topology_cost`, `_load_model_catalog`)
live in `pipeline_v2/costing.py` per cgpro Q3 explicit garde-fou
(they are consumed by Stage 4 execute — costing-transverse, NOT
topology-construction).

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


def build_topology_from_hint(hint: str) -> Any | None:
    """Create a topology from a template hint using Rust TemplateStore.

    No hardcoded prompts — nodes use their role-based defaults.
    The runner builds system prompts from each node's role field.

    Returns ``None`` when the Rust TemplateStore is unavailable
    (sage_core not installed, e.g. CI subset, type-check pass)
    or when the hint does not resolve to a known template.
    """
    try:
        from sage_core import PyTemplateStore  # type: ignore[import-not-found]

        store = PyTemplateStore()
        return store.create(hint, "")
    except (ImportError, ValueError):
        return None


def topology_candidate_items(
    pipeline: "CognitiveOrchestrationPipeline",  # noqa: ARG001 - signature symmetry
    result: Any,
) -> list[Any]:
    """Normalise a Rust `PyGenerateResult` (or list/tuple) into a Python list."""
    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        return list(result)
    candidates_attr = getattr(result, "candidates", None)
    if callable(candidates_attr):
        try:
            candidates_attr = candidates_attr()
        except Exception:
            candidates_attr = None
    if candidates_attr is not None:
        try:
            return list(candidates_attr)
        except TypeError:
            pass
    return [result]


def log_topology_candidates(
    pipeline: "CognitiveOrchestrationPipeline",
    candidates: list[Any],
) -> None:
    """Emit one INFO line per candidate (path / source / archive_hit / score / template)."""
    for path, candidate in enumerate(candidates, start=1):
        source = candidate_text_attr(candidate, ("source",), "unknown")
        score = candidate_float_attr(
            candidate,
            ("score", "confidence", "quality"),
            0.0,
        )
        topology = getattr(candidate, "topology", None)
        template = candidate_text_attr(
            topology if topology is not None else candidate,
            ("template_type", "template"),
            "unknown",
        )
        nodes = candidate_node_count(topology if topology is not None else candidate)
        log.info(
            "topology.candidate path=%d source=%s archive_hit=%s "
            "score=%.3f template_type=%s nodes=%d",
            path,
            source,
            "true" if source in ("archive", "archive_hit") else "false",
            score,
            template,
            nodes,
        )


def candidate_text_attr(
    obj: Any,
    names: tuple[str, ...],
    default: str,
) -> str:
    """Pure helper — read a string attribute from a candidate, falling back to `default`."""
    if obj is None:
        return default
    for name in names:
        value = getattr(obj, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if value is not None:
            text = str(value)
            if text:
                return text
    return default


def candidate_float_attr(
    obj: Any,
    names: tuple[str, ...],
    default: float,
) -> float:
    """Pure helper — read a numeric attribute from a candidate, falling back to `default`."""
    if obj is None:
        return default
    for name in names:
        value = getattr(obj, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return default


def candidate_node_count(obj: Any) -> int:
    """Pure helper — return a candidate/topology's node count, 0 on any failure."""
    if obj is None:
        return 0
    try:
        node_count = getattr(obj, "node_count", None)
        if callable(node_count):
            return int(node_count())
        if node_count is not None:
            return int(node_count)
    except Exception:
        return 0
    return 0


def log_topology_structure(
    pipeline: "CognitiveOrchestrationPipeline",  # noqa: ARG001 - signature symmetry
    topology: Any,
    source: str,
    confidence: float | None,
) -> None:
    """Gap 1+2 (2026-04-21): emit two INFO lines describing the DAG shape.

    Called from each Stage-2 branch right before the final cache step so
    every selected topology gets attribution regardless of which of the
    6 paths (smmu_hit / archive_hit / llm_synthesis / mutation /
    mcts_search / template_fallback) or fallback branches produced it.

    topology.edges — adjacency list. Truncated at 20 tuples; when the
    graph has >20 edges, include `total=N` so readers can tell the
    truncation happened. When the graph exposes no `get_edges`, we
    emit count-only so post-run analysis still sees the structure.

    topology.source — 6-path attribution with confidence. "dag_template"
    and "template_fallback" are Python-side branches (no engine call);
    all Rust-side sources are the canonical string from
    PyGenerateResult.source() (sage-core/src/topology/pyo3_wrappers.rs).
    """
    if topology is None:
        return

    nodes = 0
    try:
        if hasattr(topology, "node_count"):
            nc = topology.node_count()
            nodes = nc() if callable(nc) else int(nc)
    except Exception:
        nodes = 0

    template = getattr(topology, "template_type", None) or "unknown"
    topo_id = getattr(topology, "id", "") or ""

    # --- edges line ---
    edges_render: str = "[]"
    total_edges = 0
    truncated = False
    try:
        if hasattr(topology, "get_edges"):
            raw_edges = topology.get_edges()
            edges_iter = list(raw_edges) if raw_edges is not None else []
            total_edges = len(edges_iter)
            # Keep only (from, to) tuples; flow_type (3rd field) omitted
            # to keep the line short. Flow type is dominated by "control"
            # for DAG templates and not load-bearing for grep.
            pairs = [(int(e[0]), int(e[1])) for e in edges_iter[:20]]
            edges_render = repr(pairs)
            if total_edges > 20:
                truncated = True
        elif hasattr(topology, "edge_count"):
            ec = topology.edge_count()
            total_edges = ec() if callable(ec) else int(ec)
            edges_render = "<count-only>"
    except Exception:
        edges_render = "<unreadable>"

    if truncated:
        log.info(
            "topology.edges nodes=%d template=%s id=%s edges=%s total=%d",
            nodes, template, (topo_id[:8] if topo_id else "none"),
            edges_render, total_edges,
        )
    else:
        log.info(
            "topology.edges nodes=%d template=%s id=%s edges=%s",
            nodes, template, (topo_id[:8] if topo_id else "none"),
            edges_render,
        )

    # --- source line ---
    conf_str = (
        f"{float(confidence):.3f}"
        if confidence is not None
        else "n/a"
    )
    # archive_hit flag (boolean) distinguishes the fast archive path from
    # every other 6-path source — useful for MAP-Elites growth attribution.
    archive_hit = (source == "archive_hit")
    log.info(
        "topology.source source=%s confidence=%s archive_hit=%s template=%s id=%s",
        source, conf_str, "true" if archive_hit else "false",
        template, (topo_id[:8] if topo_id else "none"),
    )


def apply_topology_budget_and_cache(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
) -> None:
    """Plan item 1.4a (2026-04-20): apply budget check + cache the final topology.

    `check_topology_budget` may replace ctx.topology with a degraded
    single-node fallback; we cache AFTER that replacement so the id
    stored in topology_cache matches whatever record_outcome will
    reference in Stage 5. Before this helper existed, cache_topology
    was only wired on the engine branch (H4, commit dc51976), leaving
    three production paths silently uncached:
      - template branch (line ~502, dominant production path)
      - engine-branch budget degrade (`make_single_node_topology`)
      - fallback TopologyGraph + TopologyNode path
    Empirically verified by plan-1.4 smoke: template branch → 0 cells
    after 10 pipeline.run() calls; with this helper → archive grows.

    The internal call to `pipeline._check_topology_budget(ctx)` goes
    through the delegator that lives in `pipeline.py` so existing test
    mocks on the method continue to fire byte-identical.
    """
    pipeline._check_topology_budget(ctx)
    if (
        ctx.topology is not None
        and pipeline.engine is not None
        and hasattr(pipeline.engine, "cache_topology")
    ):
        try:
            pipeline.engine.cache_topology(ctx.topology)
        except (RuntimeError, TypeError) as exc:
            log.debug("cache_topology failed: %s", exc)


def check_topology_budget(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
) -> None:
    """Pre-validate budget feasibility — degrade to single-node if over budget."""
    if ctx.budget <= 0:
        return
    if ctx.topology and hasattr(ctx.topology, 'node_count'):
        total_node_cost = 0.0
        nc = ctx.topology.node_count()
        for i in range(nc):
            node = ctx.topology.get_node(i) if hasattr(ctx.topology, 'get_node') else None
            if node:
                total_node_cost += getattr(node, 'max_cost_usd', 0.0)
        if total_node_cost > ctx.budget:
            log.warning(
                "Topology budget %.2f > pipeline budget %.2f — degrading to single-node",
                total_node_cost, ctx.budget,
            )
            pipeline._emit("TOPOLOGY_BUDGET_WARNING", {"total_cost": total_node_cost, "budget": ctx.budget})
            # Degrade: replace with single-node template topology
            ctx.topology = pipeline._make_single_node_topology(ctx)


def make_single_node_topology(
    pipeline: "CognitiveOrchestrationPipeline",  # noqa: ARG001 - signature symmetry
    ctx: "PipelineContext",
) -> Any:
    """Create a minimal single-node topology as budget-safe fallback."""
    try:
        from sage_core import TopologyGraph, TopologyNode  # type: ignore[import-not-found]

        topo = TopologyGraph("sequential")
        node = TopologyNode(role="agent", model_id="", system=ctx.system)
        topo.add_node(node)
        return topo
    except ImportError:
        log.debug("sage_core unavailable, topology=None (single-agent mode)")
        return None


__all__ = [
    "apply_topology_budget_and_cache",
    "build_topology_from_hint",
    "candidate_float_attr",
    "candidate_node_count",
    "candidate_text_attr",
    "check_topology_budget",
    "log_topology_candidates",
    "log_topology_structure",
    "make_single_node_topology",
    "topology_candidate_items",
]
