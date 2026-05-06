"""Stage 2 - SELECT_TOPOLOGY.

Per ADR-015 + cgpro 2026-05-05 DESIGN lock + cgpro Phase 2.1 round-2
GO_STEP_B 2026-05-06:

  - cycle-12 Phase B moved the body of
    `CognitiveOrchestrationPipeline._stage_select_topology` here as a
    module-level function. Legacy method became a 1-line LOCAL-import
    delegator.
  - cycle-13 K Phase 2.1 Step B5 (2026-05-06) moved the
    topology-construction helpers out to
    `pipeline_v2/topology_helpers.py` (`build_topology_from_hint`,
    `log_topology_structure`, `apply_topology_budget_and_cache`,
    `topology_candidate_items`, `log_topology_candidates`,
    `candidate_text_attr`, `candidate_float_attr`,
    `candidate_node_count`, `check_topology_budget`,
    `make_single_node_topology`). The pipeline class retains 1-line
    delegator methods (`_<helper>`) so the call sites below
    (`self._<helper>(...)` where self is the pipeline) continue working
    byte-identical and the ~10 test files that mock these methods keep
    firing. Stage 2 flow stays in this module per cgpro Q3 ("ne pas
    faire monter select_topology.py à ~950 LOC").

cgpro Q3 explicit garde-fou: `_estimate_topology_cost` and
`_load_model_catalog` are NOT topology-construction helpers — they
live in `pipeline_v2/costing.py` (consumed by Stage 4 execute when
the runner doesn't produce a real cost; conceptually pricing/
catalogue, not topology selection).
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

# Module globals that the moved body uses unqualified. cgpro DESIGN
# trap #2: when a body moves, unqualified names resolve in THIS
# module's namespace, not pipeline.py's.
from sage.pipeline_stages import select_macro_topology

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext

# Per-module logger MUST stay "sage.pipeline" for trace-grep
# compatibility with the cycle-9 ledger consumers.
log = logging.getLogger("sage.pipeline")


def select_topology(
    pipeline: "CognitiveOrchestrationPipeline", ctx: "PipelineContext",
) -> "PipelineContext":
    """Stage 2: Select optimal topology.

    S1 (simple) tasks skip topology entirely — direct single-agent call
    is faster AND equally effective (confirmed by MASBENCH: topology helps
    only when base accuracy < 60%, per AdaptOrch arXiv 2602.16873).
    """
    self = pipeline
    from sage.observability.spans import sage_span
    with sage_span("sage.topology_select", op="sage.topology_select"):
        skip_dag_template = (
            os.environ.get("SAGE_TOPOLOGY_SKIP_DAG_TEMPLATE") == "1"
            or os.environ.get("SAGE_TOPOLOGY_FORCE_ENGINE") == "1"
        )
        log_all_candidates = (
            os.environ.get("SAGE_TOPOLOGY_LOG_ALL_CANDIDATES") == "1"
        )

        # Sprint 5 ablation: force bypass to measure framework delta.
        if os.environ.get("SAGE_ABLATION_NO_TOPOLOGY") == "1":
            ctx.topology = None
            log.info("Stage 2: topology disabled by SAGE_ABLATION_NO_TOPOLOGY=1 (ablation)")
            return ctx

        # S1 fast path: skip topology for non-math tasks.
        # Math tasks use formal_solver (SatLM NeurIPS 2023): LLM formalizes,
        # Rust solves exactly. Falls back to single-agent if solver fails.
        if ctx.system == 1 and not skip_dag_template:
            if ctx.domain == "math":
                topo = self._build_topology_from_hint("formal_solver")
                if topo:
                    ctx.topology = topo
                    # Cycle-11 cgpro VERIFY follow-up (2026-05-05):
                    # bench `_capture_control_surface` reads
                    # `ctx.topology_id` directly, NOT the disjunction
                    # at pipeline.py:648. Setting it here keeps the
                    # bench-visible topology id consistent across all
                    # branches that build a topology.
                    ctx.topology_id = getattr(topo, "id", "") or ""
                    log.info("S1 math: formal_solver (formalizer → Rust solver, fallback to CoT)")
                    self._log_topology_structure(topo, source="dag_template", confidence=None)
                    self._apply_topology_budget_and_cache(ctx)
                    return ctx
            ctx.topology = None
            log.debug("S1 task: skipping topology (direct single-agent)")
            return ctx

        # Structure-driven template selection from DAG decomposition
        # Uses omega (parallelism), delta (depth), gamma (coupling) —
        # no regex heuristics, purely structural signals from Stage 1.
        hint = "sequential"
        if ctx.dag_features and not skip_dag_template:
            hint = select_macro_topology(ctx.dag_features, ctx.system, ctx.domain)

        # S2+sequential: use the sequential topology template instead of bypass.
        # Research (AdaptOrch 2602.16873, MASS 2502.02533) shows sequential
        # planner→coder→synthesizer pipeline beats single-agent by 12-23%.
        # The old bypass (SAGE_BYPASS_S2_SEQUENTIAL=1) is available for A/B testing.
        if (
            ctx.system == 2
            and hint == "sequential"
            and not skip_dag_template
        ):
            # `os` is imported at module level; a redundant local `import os`
            # here would shadow it and break the earlier SAGE_ABLATION_NO_TOPOLOGY
            # check (UnboundLocalError when Python treats `os` as local).
            if os.environ.get("SAGE_BYPASS_S2_SEQUENTIAL") == "1":
                ctx.topology = None
                log.info("Stage 2: BYPASS topology (SAGE_BYPASS_S2_SEQUENTIAL=1)")
                return ctx

        # Build topology from template hint. All DAG-selected templates go
        # through TemplateStore which creates multi-node topologies.
        if (
            not skip_dag_template
            and hint in (
                "sequential",
                "avr",
                "parallel",
                "robust",
                "horizon_pipeline",
                "parallel_fanout",
            )
        ):
            topo = self._build_topology_from_hint(hint)
            if topo:
                ctx.topology = topo
                # Cycle-11 cgpro VERIFY follow-up (2026-05-05):
                # bench `_capture_control_surface` reads
                # `ctx.topology_id` directly, NOT the disjunction
                # at pipeline.py:648. Setting it here keeps the
                # bench-visible topology id consistent across all
                # branches that build a topology — was previously
                # missing on the DAG-template branch (the common
                # case for budget-tier S2), causing blank topology
                # IDs in BCB control-surface telemetry.
                ctx.topology_id = getattr(topo, "id", "") or ""
                log.info(
                    "Stage 2: DAG-driven template=%s (%d nodes, omega=%s delta=%s gamma=%s)",
                    hint, topo.node_count(),
                    ctx.dag_features.omega if ctx.dag_features else "?",
                    ctx.dag_features.delta if ctx.dag_features else "?",
                    f"{ctx.dag_features.gamma:.2f}" if ctx.dag_features else "?",
                )
                # Gap 1+2 (2026-04-21): emit structure log alongside template
                # name so post-run analysis can attribute pass-rate by DAG
                # shape (edges) and 6-path source, not just template name.
                self._log_topology_structure(topo, source="dag_template", confidence=None)
                self._apply_topology_budget_and_cache(ctx)
                return ctx

        # Try DynamicTopologyEngine
        if self.engine:
            try:
                # Compute real embedding for S-MMU semantic retrieval
                task_embedding = None
                try:
                    from sage.memory.embedder import Embedder
                    _emb = Embedder()
                    if _emb.is_semantic:
                        task_embedding = _emb.embed(ctx.task[:500])
                except (ImportError, RuntimeError, OSError):
                    # OSError: model weights not on disk / HF_HUB_OFFLINE=1 without cache
                    pass
                raw_result = self.engine.generate(
                    ctx.task, task_embedding, ctx.system, ctx.budget
                )
                candidates = self._topology_candidate_items(raw_result)
                if log_all_candidates:
                    self._log_topology_candidates(candidates)
                result = (
                    candidates[0]
                    if isinstance(raw_result, (list, tuple)) and candidates
                    else raw_result
                )
                if result and hasattr(result, "topology"):
                    ctx.topology = result.topology
                    # Plan item 1.4a (2026-04-20): always use ctx.topology.id
                    # (full ULID) — the engine's topology_cache / archive lookup
                    # is keyed by this ULID. result.topology_id() returns a
                    # descriptor-keyed semantic ID (e.g. "avr:n3:01KPN3XZ")
                    # which is NOT cache-compatible; using it caused record_outcome
                    # cache misses → archive never grew on the engine branch.
                    ctx.topology_id = getattr(ctx.topology, "id", "")
                elif result:
                    ctx.topology = result
                    # Cycle-11 cgpro VERIFY follow-up (2026-05-05):
                    # symmetric with the result.topology branch above —
                    # set ctx.topology_id so bench control-surface
                    # telemetry sees the right value.
                    ctx.topology_id = getattr(result, "id", "") or ""

                # Gap 1+2 (2026-04-21): log DAG edges + 6-path source
                # (smmu_hit / archive_hit / llm_synthesis / mutation /
                # mcts_search / template_fallback) with confidence. The
                # source is exposed by PyGenerateResult.source() per
                # sage-core/src/topology/pyo3_wrappers.rs.
                _src = None
                _conf = None
                if result is not None:
                    _src_attr = getattr(result, "source", None)
                    if callable(_src_attr):
                        try:
                            _src = _src_attr()
                        except Exception:
                            _src = None
                    else:
                        _src = _src_attr
                    _conf_attr = getattr(result, "confidence", None)
                    if callable(_conf_attr):
                        try:
                            _conf = _conf_attr()
                        except Exception:
                            _conf = None
                    else:
                        _conf = _conf_attr
                self._log_topology_structure(
                    ctx.topology, source=_src or "engine_unknown", confidence=_conf,
                )
                self._apply_topology_budget_and_cache(ctx)
                return ctx
            except (ImportError, RuntimeError) as exc:
                log.warning(
                    "Stage 2 topology engine failed: %s, using template", exc
                )

        # Fallback: create topology from template
        try:
            from sage_core import TopologyGraph, TopologyNode  # type: ignore[import-not-found]

            topo = TopologyGraph(hint)
            node = TopologyNode(role="agent", model_id="", system=ctx.system)
            topo.add_node(node)
            ctx.topology = topo
            log.debug("Stage 2 fallback to template: %s", hint)
        except ImportError:
            log.debug("sage_core unavailable, topology=None (single-agent mode)")
            ctx.topology = None

        # Gap 1+2 (2026-04-21): log structure for the fallback path too.
        if ctx.topology is not None:
            self._log_topology_structure(
                ctx.topology, source="template_fallback", confidence=None,
            )
        self._apply_topology_budget_and_cache(ctx)
        return ctx



__all__ = ["select_topology"]
