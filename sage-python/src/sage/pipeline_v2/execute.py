"""Stage 4 — EXECUTE.

Module function `execute(pipeline, ctx, *, event_log=None,
run_frame_builder=None)` is the canonical Stage 4 entry point; the
orchestrator awaits it directly with the pipeline instance as first
argument. `pick_fallback_provider(pipeline)` lives here too — it is
consumed by the multi-agent error-fallback path inside `execute`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sage.agent_loop_factory import create_bypass_agent_loop

# Constants + module-level helpers consumed unqualified inside `execute`.
from sage.pipeline import (
    BUDGET_EXCEEDED_RESULT,
    EXECUTE_HALTED_UNVERIFIED,
    EXECUTE_UNVERIFIED,
    _is_strict_governance,
)
# Module-attribute imports (not aliased function references) so production
# calls resolve `<mod>_mod.<fn>` at call time and pick up
# `monkeypatch.setattr("sage.pipeline_v2.<mod>.<fn>", ...)` from tests.
from sage.pipeline_v2 import memory_gate as memory_gate_mod
from sage.pipeline_v2 import provider_policy as provider_policy_mod
from sage.pipeline_v2 import runtime_events as runtime_events_mod

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext

log = logging.getLogger("sage.pipeline")


def _select_bypass_model(
    pipeline: Any,
    decision: Any,
    bandit_provider: Any,
    bandit_config: Any,
) -> tuple[Any, Any, str]:
    """Select (provider, config, active_model_id) for the bypass path.

    Priority order MUST match the legacy mutation block:
      1. bandit_provider + decision (when both present)
      2. Rust routing decision via provider_pool.resolve()
      3. Fallback: singleton AgentLoop's _llm + config.llm

    Per cgpro DESIGN Q1: the fallback default comes from the SINGLETON,
    not from pipeline.llm_provider - preserves the legacy semantics
    where the bypass mutated `self._agent_loop._llm` and ran on it.
    """
    singleton = pipeline._agent_loop
    selected_provider = singleton._llm
    selected_config = singleton.config.llm
    active_model_id = ""

    if decision is not None and bandit_provider is not None:
        selected_provider = bandit_provider
        selected_config = bandit_config
        active_model_id = decision.model_id
        log.info(
            "Stage 4 bypass: agent_loop using bandit-selected %s",
            decision.model_id,
        )
    else:
        routing_decision = getattr(pipeline, "_last_routing_decision", None)
        if routing_decision and routing_decision.model_id and pipeline.provider_pool:
            try:
                if pipeline.provider_pool.is_model_available(routing_decision.model_id):
                    selected_provider, selected_config = pipeline.provider_pool.resolve(
                        routing_decision.model_id
                    )
                    active_model_id = routing_decision.model_id
                    log.info(
                        "Stage 4 bypass: agent_loop using Rust-selected %s",
                        routing_decision.model_id,
                    )
            except Exception:
                pass  # Keep default provider.

    if not active_model_id:
        active_model_id = (
            getattr(selected_config, "model", "")
            or getattr(selected_provider, "model_id", "")
            or getattr(selected_provider, "model_string", "")
            or getattr(selected_provider, "name", "")
        )
    return selected_provider, selected_config, active_model_id


def _make_bypass_drift_callback(provider_pool: Any, model_id: str) -> Any | None:
    """Build the H6 drift callback closure for the bypass path.

    Returns None when provider_pool is None or doesn't have
    record_failure (no callback is wired).

    Per cgpro DESIGN Q2: closure captures pool + model_id at build
    time (NOT lazy-resolved on every callback - same semantics as
    the legacy code).
    """
    if provider_pool is None or not hasattr(provider_pool, "record_failure"):
        return None

    pool_ref = provider_pool
    bypass_model_id = model_id or "default"

    def _on_drift_bypass(
        provider_hint: str,
        action: str,
        details: dict[str, Any],
        _pool: Any = pool_ref,
        _model: str = bypass_model_id,
    ) -> None:
        if action not in ("SWITCH_MODEL", "RESET_AGENT"):
            return
        key = provider_hint or _model or "unknown"
        try:
            _pool.record_failure(
                key,
                RuntimeError(
                    f"drift_{action.lower()} latency={details.get('latency', '?')}"
                ),
            )
        except Exception:  # noqa: BLE001
            pass

    return _on_drift_bypass


async def execute(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
    event_log: Any | None = None,
    run_frame_builder: Any | None = None,
) -> "PipelineContext":
    """Stage 4: Execute topology with per-node model resolution."""
    self = pipeline
    from sage.observability.spans import sage_span
    with sage_span("sage.execute", op="sage.execute"):
        cost_tracker = getattr(ctx, "cost_tracker", None)
        if cost_tracker is not None and cost_tracker.is_over_budget:
            memory_gate_mod.emit_budget_exceeded(self, ctx)
            ctx.result = BUDGET_EXCEEDED_RESULT
            return ctx

        if not ctx.verification_passed:
            # A0b (2026-04-23, ALIRE2 §6): strict mode aborts here instead
            # of falling through to EXECUTE_UNVERIFIED. The default keeps
            # the historical "log and continue" behaviour so dev smokes
            # don't break on a Z3 unsat that would normally be a soft
            # signal. Production / audit runs set SAGE_STRICT_GOVERNANCE=1.
            if _is_strict_governance():
                log.error(
                    "Stage 4: aborting under SAGE_STRICT_GOVERNANCE=1 — "
                    "verification failed on provider assignment (SAT check)."
                )
                self._emit(
                    EXECUTE_HALTED_UNVERIFIED,
                    {"reason": "SAT check failed in Stage 3"},
                )
                raise RuntimeError(
                    "SAGE_STRICT_GOVERNANCE: pipeline aborted — provider "
                    "assignment failed verification (SAT check)."
                )
            log.warning("Stage 4: executing with unverified provider assignment (SAT check failed)")
            self._emit(EXECUTE_UNVERIFIED, {"reason": "SAT check failed in Stage 3"})

        # Single-agent mode (no topology or single node)
        from sage.pipeline_v2 import bandit_attribution as _bandit_attr
        if _bandit_attr.is_single_agent_execution(self, ctx):
            ctx.executed_template = "single_agent"
            decision = None
            bandit_provider = None
            bandit_config = None

            if self._agent_loop:
                # Model selection: bandit > Rust routing > singleton default.
                # Helper preserves the EXACT priority order of the legacy
                # mutation block (cgpro DESIGN Q1).
                selected_provider, selected_config, active_model_id = _select_bypass_model(
                    pipeline=self,
                    decision=decision,
                    bandit_provider=bandit_provider,
                    bandit_config=bandit_config,
                )
                provider_policy_mod.enforce_model_provider_policy(
                    self,
                    model_id=active_model_id,
                    provider_id=getattr(selected_config, "provider", "") or "",
                    event_log=event_log,
                    node_id="single_agent",
                )

                # H6 drift callback - built AFTER active_model_id is finalized
                # (legacy code captured _bypass_model_id BEFORE the bandit/Rust
                # selection took effect; that was a latent bug).
                on_drift_bypass = _make_bypass_drift_callback(
                    provider_pool=self.provider_pool,
                    model_id=active_model_id,
                )

                # Per-run AgentLoop instance - no shared mutable state. The
                # factory propagates toolforge / evolution_memory /
                # dangerous_tools (commit 9f7783cc).
                from sage.patch_artifacts import artifact_profile_active

                bypass_loop = create_bypass_agent_loop(
                    singleton=self._agent_loop,
                    llm_provider=selected_provider,
                    llm_config=selected_config,
                    system_level=ctx.system,
                    write_gate=self.write_gate,
                    task_text=ctx.task,
                    on_drift=on_drift_bypass,
                    run_frame_builder=run_frame_builder,
                    runtime_node_run_id=None,
                    # F4: verified operator-set profile (env), never an
                    # LLM-inferred label (cgpro DESIGN amendment #6).
                    task_profile=(
                        "unified_diff" if artifact_profile_active() else None
                    ),
                )

                ctx.executed_model_id = active_model_id
                ctx.executed_template = "single_agent"
                # G1 GroundingEnvelope for the bypass emitter (cgpro
                # GROUNDING DESIGN_LOCKED 2026-06-11): verbatim localized
                # file bytes ahead of the task; compose_grounded_task's
                # marker guard prevents bypass/topology double-injection.
                bypass_task = ctx.task
                if artifact_profile_active():
                    import os as _os

                    from sage.grounding import (
                        build_grounding_block,
                        compose_grounded_task,
                    )

                    _repo_dir = _os.getcwd()
                    if _os.path.isdir(_os.path.join(_repo_dir, ".git")):
                        try:
                            _g_block, _g_tel = await build_grounding_block(
                                _repo_dir, ctx.task, selected_provider
                            )
                        except Exception:  # noqa: BLE001 - best-effort
                            _g_block, _g_tel = "", {}
                        if _g_block:
                            bypass_task = compose_grounded_task(
                                _g_block, ctx.task
                            )
                        ctx.grounding_telemetry = _g_tel
                ctx.result = await bypass_loop.run(bypass_task)
                ctx.cost = bypass_loop.total_cost_usd
                ctx.tool_call_count = getattr(bypass_loop, "tool_call_count", 0)
                ctx.tool_turn_count = getattr(bypass_loop, "tool_turn_count", 0)
                ctx.executed_commands = list(getattr(bypass_loop, "executed_commands", []))
            elif self.llm_provider or bandit_provider is not None:
                # Simple fallback: single provider.generate() call (no tool loop).
                # Used only when pipeline is created without agent_loop (e.g., tests).
                from sage.llm.base import Message, Role

                active_provider = self.llm_provider
                active_config = self.llm_config
                active_model_id = (
                    getattr(self.llm_config, "model", "")
                    if self.llm_config is not None
                    else ""
                )
                if decision is not None and bandit_provider is not None:
                    active_provider = bandit_provider
                    active_config = bandit_config
                    active_model_id = decision.model_id
                else:
                    routing_decision = getattr(self, '_last_routing_decision', None)
                    if routing_decision and routing_decision.model_id and self.provider_pool:
                        try:
                            if self.provider_pool.is_model_available(routing_decision.model_id):
                                active_provider, active_config = self.provider_pool.resolve(
                                    routing_decision.model_id
                                )
                                active_model_id = routing_decision.model_id
                        except Exception:
                            pass
                if active_provider is not None and not active_model_id:
                    active_model_id = (
                        getattr(active_config, "model", "")
                        or getattr(active_provider, "model_id", "")
                        or getattr(active_provider, "model_string", "")
                        or getattr(active_provider, "name", "")
                    )
                provider_policy_mod.enforce_model_provider_policy(
                    self,
                    model_id=active_model_id,
                    provider_id=getattr(active_config, "provider", "") or "",
                    event_log=event_log,
                    node_id="single_agent",
                )
                ctx.executed_model_id = active_model_id
                ctx.executed_template = "single_agent"

                messages = [Message(role=Role.USER, content=ctx.task)]
                try:
                    response = await active_provider.generate(
                        messages=messages, config=active_config,
                    )
                    ctx.result = response.content or ""
                except (RuntimeError, TimeoutError) as exc:
                    log.error("Stage 4 fallback failed: %s", exc)
                    ctx.result = f"Error: {exc}"
            return ctx

        # Multi-agent mode: use TopologyRunner with ProviderPool
        ctx.executed_model_ids = [
            model_id for _, model_id in sorted(ctx.assignments.items())
        ]
        ctx.executed_template = getattr(ctx.topology, "template_type", "") or "multi_agent"
        try:
            from sage.topology.runner import TopologyRunner

            # Get executor
            try:
                from sage_core import TopologyExecutor  # type: ignore[import-not-found]

                executor = TopologyExecutor(ctx.topology)
            except ImportError:
                log.warning("sage_core TopologyExecutor unavailable, falling back")
                ctx.result = "Error: TopologyExecutor unavailable"
                return ctx

            # Phase 2: create agent_loop factory for per-node execution
            _agent_loop_factory = None
            if self._agent_loop and self.tool_registry:
                from sage.agent_loop_factory import create_node_agent_loop
                from functools import partial

                _agent_loop_factory = partial(
                    create_node_agent_loop,
                    tool_registry=self.tool_registry,
                    system_level=ctx.system,
                    task_domain=ctx.domain or "",  # F7-symmetric domain gate for PRM
                    on_event=(
                        self.event_bus.emit
                        if self.event_bus and hasattr(self.event_bus, "emit")
                        else None
                    ),
                    # G-series: pipeline-scoped gate + task text for relevance
                    write_gate=self.write_gate,
                    task_text=ctx.task,
                    # T2 phase 0/1 (cgpro 2026-04-29): forward memory
                    # backends so per-node loops can write to real
                    # episodic/semantic/causal stores instead of
                    # always hitting memory_backend_unwired.
                    episodic_memory=self.episodic_memory,
                    semantic_memory=self.semantic_memory,
                    memory_agent=self.memory_agent,
                    causal_memory=self.causal_memory,
                )

            # Fix C (2026-05-03): adaptive controller adds ~30-50s overhead
            # per task on budget tier (model upgrades + reroutes push tasks
            # over 120s cap). v7 ablation: no-guardrails 7/10 vs full 4/10.
            _effective_controller = (
                None if self._llm_tier == "budget" else self.controller
            )
            runner = TopologyRunner(
                graph=ctx.topology,
                executor=executor,
                llm_provider=self.llm_provider,
                llm_config=self.llm_config,
                provider_pool=self.provider_pool,
                controller=_effective_controller,
                axis_hint=ctx.axis_hint,
                agent_loop_factory=_agent_loop_factory,
                cost_tracker=getattr(ctx, "cost_tracker", None),
                assigner=self.assigner,
                task_domain=getattr(ctx, "domain", "") or "",
                budget_usd=float(getattr(ctx, "budget", 0.0) or 0.0),
                event_log=event_log,
                run_frame_builder=run_frame_builder,
            )
            result = await runner.run(ctx.task)
            # Roll up tool-use telemetry from TopologyRunner → ctx. Without
            # this the bench manifest sees zero even on multi-agent paths
            # (Codex 2026-04-18 review flagged this gap at pipeline.py:963).
            ctx.tool_call_count = getattr(runner, "tool_call_count", 0)
            ctx.tool_turn_count = getattr(runner, "tool_turn_count", 0)
            ctx.executed_commands = list(getattr(runner, "executed_commands", []))
            # Same roll-up for cost. Before Apr 18 2026 ctx.cost came only
            # from the single-loop bypass path, so multi-agent topology runs
            # reported _cost_usd=0 even when each node had metered cost.
            ctx.cost = float(getattr(runner, "total_cost_usd", 0.0) or 0.0)
            if result == BUDGET_EXCEEDED_RESULT:
                memory_gate_mod.emit_budget_exceeded(self, ctx)
                ctx.result = result
                return ctx
            if result == "__REROUTE__" and self.engine:
                log.info("Topology reroute triggered — REBUILDING full topology (not in-place mutation)")
                self._emit("REROUTE_REBUILD", {"reason": "controller_triggered"})
                # Direct module-function calls (LOCAL imports for the
                # circular-import discipline).
                from sage.pipeline_v2.assign_models import assign_models
                from sage.pipeline_v2.select_topology import select_topology
                ctx = select_topology(self, ctx)  # new topology
                ctx = assign_models(self, ctx)    # re-assign models
                runtime_events_mod.runtime_emit_topology_selected(
                    self,
                    ctx,
                    event_log,
                    run_frame_builder,
                    reason="reroute",
                )
                runtime_events_mod.runtime_emit_model_assigned(self, ctx, event_log, run_frame_builder)
                # Slice 10D reroute wiring (cgpro VERIFY 2026-05-11
                # NEXT_BLOCK_ID=reroute follow-up). The reroute path
                # produces a fresh `model_assigned` set; the witness
                # MUST follow with `assignment_phase="reroute"` so the
                # downstream chain (routing → policy → assignments) is
                # reconstructible per reroute attempt.
                runtime_events_mod.runtime_emit_provider_execution_witness(
                    self,
                    ctx,
                    event_log,
                    routing_model_id=getattr(
                        self, "_last_runtime_routing_model_id", ""
                    ) or "",
                    assignment_phase="reroute",
                )
                # cgpro VERIFY 2026-05-12 NEXT_BLOCK_ID=
                # REROUTE_REBUILD_I11_INLINE_BINDING: the witness
                # advertises a reroute phase, but the runtime MUST
                # also enforce provider policy here — otherwise the
                # I-11 inline binding never fires for the reroute
                # attempt and a blocked reroute candidate could
                # reach node_started/provider dispatch. cgpro lock:
                # "bind the path, not weaken the ledger". This call
                # raises ProviderPolicyViolation (and/or
                # EventLogInvariantViolation under FAIL_CLOSED) if
                # the reroute candidate violates the policy. The
                # outer caller (system.run) propagates the exception.
                provider_policy_mod.enforce_provider_policy(
                    self, ctx, event_log,
                )
                ctx.executed_model_ids = [
                    model_id for _, model_id in sorted(ctx.assignments.items())
                ]
                ctx.executed_template = (
                    getattr(ctx.topology, "template_type", "") or "multi_agent"
                )
                # Fresh executor for the regenerated topology (old one is stale)
                from sage_core import TopologyExecutor as _TE  # type: ignore[import-not-found]
                executor_rerouted = _TE(ctx.topology)
                # Re-execute with new topology (no controller to avoid infinite loop)
                runner2 = TopologyRunner(
                    graph=ctx.topology, executor=executor_rerouted,
                    llm_provider=self.llm_provider, llm_config=self.llm_config,
                    provider_pool=self.provider_pool,
                    controller=None,  # no controller on retry to prevent loop
                    agent_loop_factory=_agent_loop_factory,
                    cost_tracker=getattr(ctx, "cost_tracker", None),
                    assigner=self.assigner,
                    task_domain=getattr(ctx, "domain", "") or "",
                    budget_usd=float(getattr(ctx, "budget", 0.0) or 0.0),
                    event_log=event_log,
                    run_frame_builder=run_frame_builder,
                )
                result = await runner2.run(ctx.task)
                # Prefer the post-reroute telemetry (it's the attempt that
                # actually produced the final output).
                ctx.tool_call_count = getattr(runner2, "tool_call_count", 0)
                ctx.tool_turn_count = getattr(runner2, "tool_turn_count", 0)
                ctx.executed_commands = list(getattr(runner2, "executed_commands", []))
                ctx.cost = float(getattr(runner2, "total_cost_usd", 0.0) or 0.0)
                if result == BUDGET_EXCEEDED_RESULT:
                    memory_gate_mod.emit_budget_exceeded(self, ctx)
                    ctx.result = result
                    return ctx

            # FrugalGPT quality-gated cascade: if result quality is low, retry with upgraded models
            if result and result != "__REROUTE__" and self.quality_estimator:
                quality = self.quality_estimator.estimate(ctx.task, result)
                if quality is not None and quality < 0.3 and self.assigner:
                    log.info("Stage 4: quality=%.2f < 0.3, triggering FrugalGPT cascade retry", quality)
                    # Reassign with upgraded models (exclude current + budget escalation)
                    try:
                        if hasattr(ctx.topology, 'node_count'):
                            for i in range(ctx.topology.node_count()):
                                if self.assigner and hasattr(self.assigner, 'assign_single_node'):
                                    current_model = ctx.assignments.get(i, "")
                                    # F7 wiring (2026-04-17): forward task_system so the
                                    # Rust ModelAssigner promotes producer nodes correctly
                                    # during the cascade upgrade (otherwise the upgrade picks
                                    # the next best per-node-tier model, ignoring the overall
                                    # task complexity).
                                    #
                                    # Interaction note (advisor 2026-04-17): the cascade
                                    # stays at the F7-effective tier (S2 floor for non-rigour
                                    # S3 tasks). It does NOT escalate beyond what F7 already
                                    # set — exhausting S2 candidates before touching S3.
                                    # That's intentional: cascade is "swap to a different
                                    # model in the same tier", not "tier-escalate". If a
                                    # task genuinely needs an S3 model on a node F7 floored
                                    # at S2, that's a separate routing decision (not yet
                                    # implemented; would need a TierEscalator).
                                    cascade_task_system = (
                                        ctx.system if isinstance(getattr(ctx, "system", None), int)
                                        and ctx.system in (1, 2, 3) else None
                                    )
                                    try:
                                        from sage.pipeline_v2.provider_policy import (
                                            provider_policy_assigner_kwargs,
                                        )

                                        policy_kwargs = provider_policy_assigner_kwargs(self)
                                        self.assigner.assign_single_node(
                                            ctx.topology, i, ctx.domain,
                                            ctx.budget * 1.5,
                                            exclude_model_ids=[current_model] if current_model else None,
                                            task_system=cascade_task_system,
                                            **policy_kwargs,
                                        )
                                    except TypeError:
                                        if policy_kwargs:
                                            try:
                                                self.assigner.assign_single_node(
                                                    ctx.topology, i, ctx.domain,
                                                    ctx.budget * 1.5,
                                                    exclude_model_ids=[current_model] if current_model else None,
                                                    task_system=cascade_task_system,
                                                )
                                            except TypeError:
                                                try:
                                                    self.assigner.assign_single_node(
                                                        ctx.topology, i, ctx.domain,
                                                        ctx.budget * 1.5,
                                                        exclude_model_ids=[current_model] if current_model else None,
                                                    )
                                                except (ValueError, RuntimeError):
                                                    pass
                                            except (ValueError, RuntimeError):
                                                pass
                                        else:
                                            try:
                                                self.assigner.assign_single_node(
                                                    ctx.topology, i, ctx.domain,
                                                    ctx.budget * 1.5,
                                                    exclude_model_ids=[current_model] if current_model else None,
                                                )
                                            except (ValueError, RuntimeError):
                                                pass
                                    except (ValueError, RuntimeError):
                                        pass
                                # Verify upgraded model has an available provider
                                node = ctx.topology.get_node(i) if hasattr(ctx.topology, 'get_node') else None
                                new_model = getattr(node, 'model_id', '') if node else ''
                                if new_model and self.provider_pool and hasattr(self.provider_pool, 'is_model_available'):
                                    if not self.provider_pool.is_model_available(new_model):
                                        default_model = getattr(self.llm_config, 'model', '') if self.llm_config else ''
                                        if default_model and hasattr(ctx.topology, 'set_node_model_id'):
                                            ctx.topology.set_node_model_id(i, default_model)
                                            log.debug("FrugalGPT: reverted node %d %s -> %s (provider dead)", i, new_model, default_model)
                        # Re-execute with upgraded models
                        runtime_events_mod.runtime_emit_model_assigned(
                            self,
                            ctx,
                            event_log,
                            run_frame_builder,
                        )
                        # Slice 10D upgrade wiring (cgpro VERIFY
                        # 2026-05-11): FrugalGPT cascade re-assigns
                        # individual nodes in place. Distinct from
                        # REROUTE_REBUILD (which rebuilds the whole
                        # topology), so we use `assignment_phase=
                        # "upgrade"` to disambiguate downstream.
                        runtime_events_mod.runtime_emit_provider_execution_witness(
                            self,
                            ctx,
                            event_log,
                            routing_model_id=getattr(
                                self, "_last_runtime_routing_model_id", ""
                            ) or "",
                            assignment_phase="upgrade",
                        )
                        provider_policy_mod.enforce_provider_policy(
                            self,
                            ctx,
                            event_log,
                        )
                        from sage_core import TopologyExecutor as _TE  # type: ignore[import-not-found]
                        executor2 = _TE(ctx.topology)
                        runner3 = TopologyRunner(
                            graph=ctx.topology, executor=executor2,
                            llm_provider=self.llm_provider, llm_config=self.llm_config,
                            provider_pool=self.provider_pool,
                            agent_loop_factory=_agent_loop_factory,
                            cost_tracker=getattr(ctx, "cost_tracker", None),
                            assigner=self.assigner,
                            task_domain=getattr(ctx, "domain", "") or "",
                            budget_usd=float(getattr(ctx, "budget", 0.0) or 0.0),
                            event_log=event_log,
                            run_frame_builder=run_frame_builder,
                        )
                        retry_result = await runner3.run(ctx.task)
                        if retry_result:
                            result = retry_result
                            log.info("Stage 4: FrugalGPT cascade succeeded on retry")
                    except provider_policy_mod.ProviderPolicyViolation:
                        raise
                    except (RuntimeError, TimeoutError) as exc:
                        log.debug("Stage 4: FrugalGPT cascade retry failed: %s", exc)

            if result == BUDGET_EXCEEDED_RESULT:
                memory_gate_mod.emit_budget_exceeded(self, ctx)
                ctx.result = result
                return ctx

            ctx.result = result
            # Prefer the runner's aggregated real cost (summed from per-node
            # AgentLoop.total_cost_usd, which now prefers provider-reported
            # cost_usd). Fall back to the 500-in/300-out per-node estimate
            # only when no node reported a real cost (e.g. fully-mocked
            # tests). Before Apr 18 2026 this was always the estimate, so
            # benches never saw real provider metering even when LiteLLM
            # populated it correctly.
            if not ctx.cost:
                from sage.pipeline_v2 import costing as _costing
                ctx.cost = _costing.estimate_topology_cost(self, ctx)
        except (
            provider_policy_mod.ProviderPolicyViolation,
            provider_policy_mod.EventLogInvariantViolation,
        ):
            # cgpro DESIGN_LOCKED 2026-05-12 GO_FIX (deterministic
            # fixture finding): the outer multi-agent fallback MUST
            # NOT swallow I-11 invariant exceptions. Without this
            # filter, the outer `except (RuntimeError, ...)` at the
            # next line catches `ProviderPolicyViolation` (a
            # `RuntimeError` subclass) and falls back to single-agent,
            # silently bypassing the policy denial. The nested
            # handlers at 574 + 651 already mirror this re-raise
            # discipline — the outer was asymmetric. Re-raising both
            # PPV and EventLogInvariantViolation here restores the
            # I-11 ledger contract: "evaluated denials still emit
            # failure(error_type=provider_policy_violation) AND raise
            # ProviderPolicyViolation regardless of FAIL_CLOSED gate."
            raise
        except (ImportError, RuntimeError, TimeoutError) as exc:
            log.error("Stage 4 multi-agent execution failed: %s — falling back to single-agent", exc)
            # Fallback: run task directly on a healthy provider.
            #
            # 2026-04-21 v17 fix: previously used self.llm_provider
            # unconditionally, which is typically the boot-default — often
            # the same provider the multi-agent stage just failed on (e.g.
            # minimax 529 storm). Result: fallback hits the same 529
            # immediately, or returns "" as "success" and SAGE emits an
            # EMPTY patch (5/10 tasks on 2026-04-21 v13 smoke). Now we
            # prefer a healthy provider from the pool — if the default is
            # dead we try the first alive one instead. If the provider
            # returns empty content, we RAISE (not silently emit "") so
            # the bench classifier records an honest error.
            from sage.pipeline_v2.execute import pick_fallback_provider
            fallback_provider, fallback_config = pick_fallback_provider(self)
            if fallback_provider is not None:
                try:
                    from sage.llm.base import Message, Role
                    fallback_model_id = (
                        getattr(fallback_config, "model", "")
                        if fallback_config is not None
                        else ""
                    )
                    fallback_provider_id = (
                        getattr(fallback_config, "provider", "")
                        if fallback_config is not None
                        else ""
                    ) or getattr(fallback_provider, "name", "") or getattr(
                        fallback_provider,
                        "provider_name",
                        "",
                    )
                    provider_policy_mod.enforce_model_provider_policy(
                        self,
                        model_id=fallback_model_id,
                        provider_id=fallback_provider_id,
                        event_log=event_log,
                        node_id="single_agent_fallback",
                    )
                    response = await fallback_provider.generate(
                        messages=[Message(role=Role.USER, content=ctx.task)],
                        config=fallback_config or self.llm_config,
                    )
                    content = (response.content or "").strip()
                    if not content:
                        raise RuntimeError(
                            "Stage 4 fallback returned empty content — "
                            "treating as failure rather than emitting empty patch"
                        )
                    ctx.result = response.content or ""
                    log.info(
                        "Stage 4 fallback single-agent succeeded (%d chars, provider=%s)",
                        len(ctx.result),
                        getattr(fallback_provider, "name", type(fallback_provider).__name__),
                    )
                except provider_policy_mod.ProviderPolicyViolation:
                    raise
                except (RuntimeError, TimeoutError) as fallback_exc:
                    log.error("Stage 4 fallback also failed: %s", fallback_exc)
                    ctx.result = ""
            else:
                log.error("Stage 4 fallback: no healthy provider available")
                ctx.result = ""

            # PATCH_FALLBACK_ATTRIBUTION (cgpro 2026-05-07): the multi-agent
            # runner never ran — it raised before producing results. Reset
            # `executed_template` to "single_agent" and clear the stale
            # multi-agent `executed_model_ids` so the bandit attribution
            # sees the actual execution mode (single-agent fallback), not
            # the failed multi-agent assignment.
            #
            # Also cancel the in-flight bandit decision: the fallback
            # provider/model is different from the bandit-selected one,
            # so the attribution would be off-policy. Cancelling ensures
            # the posterior is not contaminated by a model-id mismatch.
            ctx.executed_template = "single_agent"
            ctx.executed_model_ids = []
            fallback_model_id = (
                getattr(fallback_config, "model", "")
                if fallback_config is not None
                else ""
            )
            if fallback_model_id:
                ctx.executed_model_id = fallback_model_id
            from sage.pipeline_v2.bandit_attribution import (
                cancel_bandit_decision,
                clear_bandit_decision,
            )
            cancel_bandit_decision(pipeline, ctx, force=True)
            clear_bandit_decision(pipeline, ctx)

        return ctx


def pick_fallback_provider(
    pipeline: "CognitiveOrchestrationPipeline",
) -> "tuple[Any, Any]":
    """Return (provider, config) for a healthy fallback, or (default, default_config).

    Preference order (first match wins):

      1. ``pipeline.llm_provider`` if its provider name is alive in
         the pool (i.e. the boot default is not currently dead).
      2. Any provider in ``pipeline.provider_pool._providers`` whose
         circuit is closed and whose TTL'd exclusion hasn't fired.
      3. ``pipeline.llm_provider`` as a last resort.

    Used by Stage 4 single-agent fallback after multi-agent execution
    failed. Routing to an alternative healthy provider recovers tasks
    that would otherwise return empty content when the boot-default
    provider is degraded (e.g. minimax 529 storm 2026-04-21).

    `LLMConfig` is imported LAZILY — only when an alternative provider
    is selected from the pool.
    """
    pool = getattr(pipeline, "provider_pool", None)

    # Helper: is a provider alive?
    def _alive(pname: str) -> bool:
        if pool is None:
            return True  # No pool → assume alive
        # Dead if TTL'd exclusion or circuit-open.
        if pname in getattr(pool, "_dead_at", {}):
            return False
        if hasattr(pool, "is_available") and not pool.is_available(pname):
            return False
        return True

    def _policy_allows(pname: str, model_id: str = "") -> bool:
        try:
            provider_policy_mod.enforce_model_provider_policy(
                pipeline,
                model_id=model_id,
                provider_id=pname,
            )
        except provider_policy_mod.ProviderPolicyViolation:
            return False
        return True

    # 1. Try the default provider first if it's alive.
    default = pipeline.llm_provider
    default_name = ""
    if default is not None:
        default_name = getattr(default, "name", "") or getattr(default, "provider_name", "")
        default_model = (
            getattr(pipeline.llm_config, "model", "")
            if pipeline.llm_config is not None
            else ""
        )
        if default_name and _alive(default_name) and _policy_allows(default_name, default_model):
            return default, pipeline.llm_config

    # 2. Iterate the pool for any alive provider that's not the dead default.
    if pool is not None:
        providers = getattr(pool, "_providers", {}) or {}
        for pname, prov in providers.items():
            if pname == default_name:
                continue
            if not _alive(pname):
                continue
            model_id = getattr(prov, "model_id", "") or getattr(prov, "model_string", "")
            if not _policy_allows(pname, model_id):
                continue
            from sage.llm.base import LLMConfig
            cfg = LLMConfig(
                provider=pname,
                model=model_id,
                context_window=getattr(pipeline.llm_config, "context_window", 128000) if pipeline.llm_config else 128000,
            )
            log.info(
                "Stage 4 fallback: rerouting from dead default=%s to healthy %s",
                default_name or "(none)", pname,
            )
            return prov, cfg

    # 3. Last resort: default even if marked dead (better than nothing).
    return default, pipeline.llm_config


__all__ = ["execute", "pick_fallback_provider"]
