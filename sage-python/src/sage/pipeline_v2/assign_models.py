"""Stage 3 — ASSIGN_MODELS + assign-side helpers.

Module function `assign_models(pipeline, ctx)` is the canonical
Stage 3 entry point; the orchestrator calls it directly with the
pipeline instance as first argument. Two assign-side helpers also
live here:

  - `log_model_assigner_chosen_fallback` — T5 diagnostic for opaque
    Rust assigners.
  - `verify_assignment_formal` — non-blocking OxiZ / Z3 SAT check.

Stage 3 contract:
  - Rust `ModelAssigner.assign_models` call signature (topology,
    domain, budget, hints_list, task_system) per F7 wiring (2026-04-17).
  - Node assignment recording loop into `ctx.assignments`.
  - Provider-pool dead-model fallback (replaces unavailable models
    with the default LLM config model_id).
  - Non-blocking formal verification.
  - T5 diagnostic on assigner output.

`load_model_catalog` (which uses
`Path(__file__).parent.parent.parent / "config" / "cards.toml"`) is
a costing-side helper and lives in `pipeline_v2/costing.py`, NOT
here.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext

# Per-module logger MUST stay "sage.pipeline" for trace-grep
# compatibility with the cycle-9 ledger consumers.
log = logging.getLogger("sage.pipeline")


def assign_models(
    pipeline: "CognitiveOrchestrationPipeline", ctx: "PipelineContext",
) -> "PipelineContext":
    """Stage 3: Assign model_id to each topology node."""
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
            # Option C (cgpro 2026-05-07): clamp task_system to a
            # tier-appropriate maximum so budget/fast tiers do not get
            # OpenAI gpt-5.4/gpt-5.5-pro assigned to complex code tasks.
            # The routing may still report system=3 for a SWE-bench task,
            # but the tier cap ensures the Rust ModelAssigner picks models
            # that the tier can actually afford.
            _tier = getattr(self, "_llm_tier", "") or ""
            _TIER_SYSTEM_CAP: dict[str, int] = {"budget": 1, "fast": 2}
            _max_system = _TIER_SYSTEM_CAP.get(_tier, 3)
            if task_system is not None and task_system > _max_system:
                log.info(
                    "Stage 3: clamping task_system %d→%d (tier=%s)",
                    task_system, _max_system, _tier or "default",
                )
                task_system = _max_system
            try:
                from sage.pipeline_v2.provider_policy import provider_policy_assigner_kwargs

                n_assigned = self.assigner.assign_models(
                    ctx.topology,
                    ctx.domain,
                    ctx.budget,
                    hints_list,
                    task_system,
                    **provider_policy_assigner_kwargs(self),
                )
            except TypeError:
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
            from sage.pipeline_v2.assign_models import (
                log_model_assigner_chosen_fallback,
                log_model_assigner_trace_if_available,
            )
            if not log_model_assigner_trace_if_available(self, ctx):
                log_model_assigner_chosen_fallback(self, ctx)
        except (ImportError, RuntimeError) as exc:
            log.warning("Stage 3 assign failed: %s", exc)

        # Bandit feedback: Thompson sampling already handles exploration/exploitation
        # via the Beta posterior. No pre-execution budget reduction — it creates a
        # self-degrading loop (Audit2 + Audit3 confirmed). The bandit learns post-
        # execution in Stage 5 (LEARN) and naturally deprioritizes bad arms.

        provider_policy_blocks = False
        policy_decision = None
        try:
            from sage.pipeline_v2.provider_policy import (
                ProviderPolicyViolation,
                evaluate_provider_policy,
            )

            policy_decision = evaluate_provider_policy(self, ctx)
            setattr(ctx, "_provider_policy_decision", policy_decision.to_dict())
            provider_policy_blocks = bool(policy_decision.violations)
        except ProviderPolicyViolation:
            # Cycle-13 A4 (cgpro DESIGN 2026-05-10): defensive guard.
            # ProviderPolicyViolation subclasses RuntimeError, so the broad
            # `except (RuntimeError, ...)` below would otherwise swallow it
            # and let Stage-4 enforcement be the only line of defense. Today
            # `evaluate_provider_policy` does not raise, but if a future
            # refactor decides to fail-closed at evaluation time, the
            # violation MUST escape this try block. Order matters: this
            # narrow `except` MUST precede the broad one.
            raise
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            log.warning("Stage 3 provider policy precheck skipped: %s", exc)

        # Cycle-13 A4 (cgpro DESIGN 2026-05-10): fail-closed preassignment
        # gate. When `evaluate_provider_policy` detected violations, the
        # topology already carries a model whose provider violates the active
        # provider policy. Raise BEFORE Stage-4 emits `model_assigned` and
        # BEFORE the topology executor attempts the disallowed provider call.
        # Stage-4 `enforce_provider_policy` remains active as defense in
        # depth in case the runtime re-routes between Stage-3 and Stage-4.
        if provider_policy_blocks and policy_decision is not None:
            from sage.pipeline_v2.provider_policy import (
                enforce_provider_policy_preassignment,
            )

            enforce_provider_policy_preassignment(self, ctx, policy_decision)

        # Filter out models whose provider is dead (health check or circuit breaker).
        # If the raw assignment already violates an active provider policy, do not
        # mask the assignment by silently falling back before the audit event fires.
        if (
            not provider_policy_blocks
            and self.provider_pool
            and hasattr(self.provider_pool, 'is_model_available')
        ):
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
            from sage.pipeline_v2.assign_models import verify_assignment_formal
            verify_assignment_formal(self, ctx)
        except (ImportError, RuntimeError) as exc:
            log.warning("Stage 3 formal verification error (non-blocking): %s", exc)

        return ctx


def log_model_assigner_chosen_fallback(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
) -> None:
    """T5 diagnostic fallback for opaque Rust assigners.

    The Python fallback logs true top-3 candidates from its scoring
    table. Rust's PyO3 assigner does not expose that table yet, so
    this branch emits only the chosen model as rank 1, with unknown
    score components. Activated by env-var `SAGE_ASSIGNER_LOG_TOP3=1`.
    """
    if os.environ.get("SAGE_ASSIGNER_LOG_TOP3") != "1":
        return
    if pipeline.assigner is not None and hasattr(pipeline.assigner, "_score_candidates"):
        return
    # P1-2B: don't run fallback when Rust trace is available
    if pipeline.assigner is not None and hasattr(pipeline.assigner, "last_assignment_trace"):
        return
    if ctx.topology is None:
        return

    node_count = (
        ctx.topology.node_count()
        if hasattr(ctx.topology, "node_count")
        else 0
    )
    for node_idx in range(node_count):
        model_id = ctx.assignments.get(node_idx, "")
        if not model_id and hasattr(ctx.topology, "get_node"):
            node = ctx.topology.get_node(node_idx)
            model_id = getattr(node, "model_id", "") if node else ""
        if not model_id:
            continue
        log.info(
            "model_assigner.selected node_id=%d model=%s "
            "trace_available=false score_source=unavailable "
            "reason_code=rust_trace_not_exposed",
            node_idx,
            model_id,
        )


def verify_assignment_formal(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
) -> None:
    """Formally verify provider assignment via OxiZ / Z3 (NON-BLOCKING).

    Builds a lightweight adapter that bridges TopologyGraph nodes
    into the interface expected by ``verify_provider_assignment``
    without requiring a full TaskDAG conversion.

    Skips silently when:
      - No SMT backend is available (ImportError from z3_verify)
      - topology is None
      - No nodes with capability requirements are present

    LOCAL imports for `_Z3_VERIFY_AVAILABLE`, `verify_provider_assignment`,
    and `ProviderSpec` from `sage.pipeline` to avoid a top-level
    pipeline_v2 → pipeline circular-import risk that becomes critical
    once Step E moves `PipelineContext`.
    """
    from sage.pipeline import (
        ProviderSpec,
        _Z3_VERIFY_AVAILABLE,
        verify_provider_assignment,
    )

    if not _Z3_VERIFY_AVAILABLE or ctx.topology is None:
        return

    node_count = (
        ctx.topology.node_count()
        if hasattr(ctx.topology, "node_count")
        else 0
    )
    if node_count == 0:
        return

    # ── Build minimal adapter objects ──────────────────────────────────

    # Collect (node_index, capabilities) from topology
    topo_nodes: list[tuple[str, list[str]]] = []
    for i in range(node_count):
        node = (
            ctx.topology.get_node(i)
            if hasattr(ctx.topology, "get_node")
            else None
        )
        if node is None:
            continue
        # Capabilities: TopologyNode may expose .capabilities or .capabilities_required
        caps: list[str] = []
        for attr in ("capabilities", "capabilities_required"):
            raw = getattr(node, attr, None)
            if raw:
                caps = list(raw)
                break
        topo_nodes.append((str(i), caps))

    # Only verify if at least one node has capability requirements
    if not any(caps for _, caps in topo_nodes):
        return

    # ── DAG adapter ────────────────────────────────────────────────────

    class _NodeAdapter:
        """Minimal shim that looks like TaskNode to z3_verify."""

        def __init__(self, nid: str, capabilities: list[str]) -> None:
            self._nid = nid
            self.capabilities_required = capabilities

    class _DagAdapter:
        """Minimal shim that looks like TaskDAG to z3_verify."""

        def __init__(self, nodes: list[tuple[str, list[str]]]) -> None:
            self._nodes = {nid: _NodeAdapter(nid, caps) for nid, caps in nodes}

        @property
        def node_ids(self) -> list[str]:
            return list(self._nodes.keys())

        def get_node(self, nid: str) -> "_NodeAdapter | None":
            return self._nodes.get(nid)

    dag_adapter = _DagAdapter(topo_nodes)

    # ── ProviderSpec list: one entry per distinct assigned model_id ────

    # Build providers from assigned model_ids.
    # Priority: ctx.assignments (set by assigner) > topology node model_id attribute.
    # Each model is treated as a provider that offers the capabilities
    # of the node it was assigned to (optimistic: if a model was chosen
    # for a node, it can serve that node's capabilities).
    model_caps: dict[str, set[str]] = {}

    # Try ctx.assignments first (populated by _stage_assign_models assigner)
    for i, model_id in ctx.assignments.items():
        if not model_id:
            continue
        nid = str(i)
        node = dag_adapter.get_node(nid)
        node_caps: set[str] = set(node.capabilities_required) if node else set()
        if model_id not in model_caps:
            model_caps[model_id] = set()
        model_caps[model_id].update(node_caps)

    # Fallback: read model_id directly from topology nodes
    if not model_caps:
        for nid, caps in topo_nodes:
            node_obj = (
                ctx.topology.get_node(int(nid))
                if hasattr(ctx.topology, "get_node")
                else None
            )
            model_id = getattr(node_obj, "model_id", "") if node_obj else ""
            if not model_id:
                continue
            if model_id not in model_caps:
                model_caps[model_id] = set()
            model_caps[model_id].update(caps)

    if not model_caps:
        log.debug(
            "Stage 3 formal verify: no model_ids found in topology, skipping SAT check"
        )
        return

    providers = [
        ProviderSpec(name=model_id, capabilities=caps)
        for model_id, caps in model_caps.items()
    ]

    # ── Run SAT check ──────────────────────────────────────────────────

    try:
        verdict = verify_provider_assignment(dag_adapter, providers)  # type: ignore[arg-type]
    except ImportError as exc:
        log.debug("Stage 3 formal verify skipped (no SMT backend): %s", exc)
        return
    except RuntimeError as exc:
        log.warning("Stage 3 formal verify raised unexpected error: %s", exc)
        return

    if not verdict.satisfied:
        ctx.verification_passed = False
        log.warning(
            "Stage 3 formal provider assignment verification FAILED "
            "(non-blocking): %s",
            verdict.counterexample,
        )
        pipeline._emit(
            "ASSIGN_MODELS_VERIFY_FAIL",
            {"counterexample": verdict.counterexample or "UNSAT"},
        )
    else:
        log.debug(
            "Stage 3 formal provider assignment verification PASSED"
        )


def log_model_assigner_trace_if_available(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
) -> bool:
    """Log real Rust ModelAssigner top-3 trace when PyO3 exposes it (P1-2B).

    Returns True when the assigner exposes a trace API, even if logging
    is disabled or empty.  Returns False only when this is an assigner
    without Rust trace support, so the P1-2A fallback can run.
    """
    assigner = getattr(pipeline, "assigner", None)
    if assigner is None or not hasattr(assigner, "last_assignment_trace"):
        return False

    if os.environ.get("SAGE_ASSIGNER_LOG_TOP3") != "1":
        return True

    try:
        trace = list(assigner.last_assignment_trace())
    except (AttributeError, TypeError, RuntimeError, ValueError) as exc:
        log.warning(
            "model_assigner.trace_error trace_available=false "
            "source=rust_pyo3_trace reason_code=rust_trace_error error=%s",
            exc,
        )
        return True

    for item in trace:
        log.info(
            "model_assigner.candidates node_id=%d rank=%d model=%s "
            "trace_available=true source=rust_pyo3_trace reason_code=%s "
            "score=%.6f affinity=%.6f domain=%.6f cost_norm=%.6f "
            "hint_bonus=%.6f diversity_penalty=%.6f",
            int(getattr(item, "node_idx", -1)),
            int(getattr(item, "rank", 0)),
            str(getattr(item, "model_id", "")),
            str(getattr(item, "filtered_reason", "ok")),
            float(getattr(item, "total_score", 0.0)),
            float(getattr(item, "affinity_score", 0.0)),
            float(getattr(item, "domain_score", 0.0)),
            float(getattr(item, "cost_norm", 0.0)),
            float(getattr(item, "hint_bonus", 0.0)),
            float(getattr(item, "diversity_penalty", 0.0)),
        )

    return True


__all__ = [
    "assign_models",
    "log_model_assigner_trace_if_available",
    "log_model_assigner_chosen_fallback",
    "verify_assignment_formal",
]
