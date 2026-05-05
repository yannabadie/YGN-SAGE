"""Stage 4 (EXECUTE) implementation for pipeline_v2.

See pipeline_v2/__init__.py module docstring for the Phase A/B/C
plan. Phase B moves the legacy body here while the class method
becomes a local-import delegator.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sage.agent_loop_factory import create_bypass_agent_loop

# Module-level imports for the moved Stage 4 body. cgpro DESIGN trap #2:
# unqualified names in the moved body resolve in THIS module's namespace.
# The `BUDGET_EXCEEDED_RESULT` / `EXECUTE_*` constants and
# `_is_strict_governance()` helper are imported by reference.
from sage.pipeline import (
    BUDGET_EXCEEDED_RESULT,
    EXECUTE_HALTED_UNVERIFIED,
    EXECUTE_UNVERIFIED,
    _is_strict_governance,
)

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
            self._emit_budget_exceeded(ctx)
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
        if self._is_single_agent_execution(ctx):
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
                )

                ctx.executed_model_id = active_model_id
                ctx.executed_template = "single_agent"
                ctx.result = await bypass_loop.run(ctx.task)
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
                self._emit_budget_exceeded(ctx)
                ctx.result = result
                return ctx
            if result == "__REROUTE__" and self.engine:
                log.info("Topology reroute triggered — REBUILDING full topology (not in-place mutation)")
                self._emit("REROUTE_REBUILD", {"reason": "controller_triggered"})
                ctx = self._stage_select_topology(ctx)  # new topology
                ctx = self._stage_assign_models(ctx)    # re-assign models
                self._runtime_emit_topology_selected(
                    ctx,
                    event_log,
                    run_frame_builder,
                    reason="reroute",
                )
                self._runtime_emit_model_assigned(ctx, event_log, run_frame_builder)
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
                    self._emit_budget_exceeded(ctx)
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
                                        self.assigner.assign_single_node(
                                            ctx.topology, i, ctx.domain,
                                            ctx.budget * 1.5,
                                            exclude_model_ids=[current_model] if current_model else None,
                                            task_system=cascade_task_system,
                                        )
                                    except TypeError:
                                        # Older binding without task_system kwarg.
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
                        self._runtime_emit_model_assigned(
                            ctx,
                            event_log,
                            run_frame_builder,
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
                    except (RuntimeError, TimeoutError) as exc:
                        log.debug("Stage 4: FrugalGPT cascade retry failed: %s", exc)

            if result == BUDGET_EXCEEDED_RESULT:
                self._emit_budget_exceeded(ctx)
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
                ctx.cost = self._estimate_topology_cost(ctx)
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
            fallback_provider, fallback_config = self._pick_fallback_provider()
            if fallback_provider is not None:
                try:
                    from sage.llm.base import Message, Role
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
                except (RuntimeError, TimeoutError) as fallback_exc:
                    log.error("Stage 4 fallback also failed: %s", fallback_exc)
                    ctx.result = ""
            else:
                log.error("Stage 4 fallback: no healthy provider available")
                ctx.result = ""

        return ctx



__all__ = ["execute"]
