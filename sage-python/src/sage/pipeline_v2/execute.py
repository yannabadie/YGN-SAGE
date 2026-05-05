"""Stage 4 (EXECUTE) implementation for pipeline_v2.

See pipeline_v2/__init__.py module docstring for the Phase A/B/C
plan. Phase B moves the legacy body here while the class method
becomes a local-import delegator.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

# Module-level imports for the moved Stage 4 body. cgpro DESIGN trap #2:
# unqualified names in the moved body resolve in THIS module's namespace.
# `_BYPASS_AGENT_LOOP_ACTIVE` is the SAME ContextVar object as in
# `sage.pipeline` because Python imports return the same identity — so a
# writer here and a reader elsewhere see the same context state. The
# `BUDGET_EXCEEDED_RESULT` / `EXECUTE_*` constants and
# `_is_strict_governance()` helper are likewise imported by reference.
# The `asyncio` and `os` modules are NOT used at module scope here:
# the bypass lock is created via `self._get_agent_loop_bypass_lock()`
# which stays on the pipeline class, and `_is_strict_governance` does
# its own `os.environ.get()` internally.
from sage.pipeline import (
    BUDGET_EXCEEDED_RESULT,
    EXECUTE_HALTED_UNVERIFIED,
    EXECUTE_UNVERIFIED,
    _BYPASS_AGENT_LOOP_ACTIVE,
    _is_strict_governance,
)

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext

log = logging.getLogger("sage.pipeline")


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
                # P6-B (cycle-11, cgpro round-4 review 2026-05-04):
                # serialize concurrent bypass entries on the boot
                # singleton AgentLoop and fail fast on re-entry. The
                # snapshot/mutate/run/restore block below mutates 12
                # fields on the shared singleton; two same-event-loop
                # concurrent calls would interleave and clobber each
                # other's restoration. The reentry guard catches the
                # `sage_recurse` deadlock case (a tool registered at
                # boot can call back into pipeline.run from inside
                # this very block — re-acquiring the lock from the
                # same task hangs forever). Per-run AgentLoop factory
                # (P6-A) is the structural fix and is deferred to a
                # later cycle (ADR-015 characterization tests first).
                if _BYPASS_AGENT_LOOP_ACTIVE.get():
                    raise RuntimeError(
                        "Recursive AgentLoop bypass disabled: "
                        "pipeline.run() was re-entered from inside the "
                        "single-agent bypass mutation block (likely "
                        "via the sage_recurse tool). The shared "
                        "singleton AgentLoop cannot be safely re-used "
                        "while its config snapshot is held — use the "
                        "topology path or the per-run AgentLoop "
                        "factory (P6-A, deferred)."
                    )

                bypass_lock = self._get_agent_loop_bypass_lock()
                async with bypass_lock:
                    _bypass_token = _BYPASS_AGENT_LOOP_ACTIVE.set(True)
                    try:
                        # Phase 1: agent_loop.run() provides tools + S2/S3 validation +
                        # guardrails + memory. Replaces the raw provider.generate() loop.

                        # A0a (2026-04-23, ALIRE2 §4 "shared mutable state"):
                        # snapshot EVERY field we are about to mutate before touching
                        # any of them. The prior code snapshotted only `_llm` and
                        # `config.llm`, leaving 8 others (write_gate, gate_*, _on_drift,
                        # validation_level, max_steps, stall_after_tool_steps,
                        # _current_topology) dirty for the next caller after this
                        # bypass path returned. The `finally` block below restores
                        # every one of these; concurrency-safe restoration is now
                        # handled by P6-B (lock + ContextVar reentry guard above).
                        _orig_bypass_state = {
                            "_skip_routing": getattr(self._agent_loop, "_skip_routing", False),
                            "_current_topology": self._agent_loop._current_topology,
                            "write_gate": getattr(self._agent_loop, "write_gate", None),
                            "gate_current_task": getattr(self._agent_loop, "gate_current_task", None),
                            "gate_source_tier": getattr(self._agent_loop, "gate_source_tier", None),
                            "_on_drift": getattr(self._agent_loop, "_on_drift", None),
                            "_run_frame_builder": getattr(
                                self._agent_loop,
                                "_run_frame_builder",
                                None,
                            ),
                            "_runtime_node_run_id": getattr(
                                self._agent_loop,
                                "_runtime_node_run_id",
                                None,
                            ),
                            "validation_level": self._agent_loop.config.validation_level,
                            "max_steps": self._agent_loop.config.max_steps,
                            "stall_after_tool_steps": self._agent_loop.config.stall_after_tool_steps,
                        }

                        # H1: Skip routing in agent_loop (pipeline already routed in Stage 0)
                        self._agent_loop._skip_routing = True
                        # H4: Clear topology (pipeline owns topology, not agent_loop)
                        self._agent_loop._current_topology = None

                        # H5 audit fix (2026-04-19): wire the pipeline-scoped write gate
                        # onto the shared AgentLoop for the single-agent bypass path.
                        # The G-series fix (commit c905d06) only wired the gate through
                        # `agent_loop_factory.create_node_agent_loop` for multi-node
                        # topology traversal. This code path reuses a pre-existing
                        # `self._agent_loop` singleton built at boot — it never saw the
                        # factory wiring, so `loop.write_gate is None` and phases/act.py
                        # fell through to ungated writes. Same silent-bypass class as
                        # H4 (cache_topology) — fix perfectly wired, never fires.
                        self._agent_loop.write_gate = self.write_gate
                        self._agent_loop.gate_current_task = ctx.task
                        self._agent_loop._run_frame_builder = run_frame_builder
                        self._agent_loop._runtime_node_run_id = None
                        try:
                            from sage.memory.write_gate import infer_source_tier
                            model_id = getattr(
                                getattr(self._agent_loop.config, "llm", None),
                                "model", None,
                            )
                            self._agent_loop.gate_source_tier = infer_source_tier(model_id)
                        except (ImportError, AttributeError):
                            self._agent_loop.gate_source_tier = "unknown"

                        # H6 audit fix (2026-04-19): wire the drift callback on the
                        # bypass path. The multi-node path sets `_on_drift` via the
                        # factory (topology/runner.py:502-521) so SWITCH_MODEL /
                        # RESET_AGENT classifications forward to
                        # `ProviderPool.record_failure` — tripping the provider's
                        # circuit breaker so subsequent resolve() picks a different
                        # provider. On the bypass path this was never wired; drift
                        # events on S1 tasks logged but had zero effect on routing.
                        # Same silent-bypass class as H5 (write_gate).
                        if (self.provider_pool is not None
                                and hasattr(self.provider_pool, "record_failure")):
                            _pool_ref = self.provider_pool
                            _bypass_model_id = getattr(
                                getattr(self._agent_loop.config, "llm", None),
                                "model", "",
                            ) or "default"

                            def _on_drift_bypass(
                                provider_hint: str,
                                action: str,
                                details: dict[str, Any],
                                _pool: Any = _pool_ref,
                                _model: str = _bypass_model_id,
                            ) -> None:
                                if action not in ("SWITCH_MODEL", "RESET_AGENT"):
                                    return
                                _key = (provider_hint or _model or "unknown")
                                try:
                                    _pool.record_failure(
                                        _key,
                                        RuntimeError(
                                            f"drift_{action.lower()} "
                                            f"latency={details.get('latency', '?')}"
                                        ),
                                    )
                                except Exception:  # noqa: BLE001
                                    pass

                            self._agent_loop._on_drift = _on_drift_bypass

                        # Set validation level from system classification
                        if ctx.system >= 3:
                            self._agent_loop.config.validation_level = 3
                        elif ctx.system >= 2 and self._agent_loop.sandbox_manager:
                            self._agent_loop.config.validation_level = 2
                        else:
                            self._agent_loop.config.validation_level = 1

                        # Plan item 1.1 (2026-04-20): scale singleton max_steps by
                        # ctx.system — close the H5-class bypass extending the
                        # singleton-vs-factory asymmetry. boot.py:279 built the
                        # singleton with max_steps=MAX_AGENT_STEPS=20; the factory
                        # (agent_loop_factory.py:132-137) scales 5/10/20 per system
                        # tier for per-node AgentLoops. Without this line, S1 tasks
                        # on the bypass path run at 4x the factory-intended budget.
                        # agent_loop.py:424 reads self.config.max_steps directly in
                        # the step loop — mutation takes effect on the next .run().
                        self._agent_loop.config.max_steps = {1: 5, 2: 10, 3: 20}.get(ctx.system, 10)

                        # Plan item 1.2 (2026-04-20): scale singleton D8 stall cap
                        # to match the factory (agent_loop_factory.py:151-154).
                        # AgentConfig.stall_after_tool_steps defaults to 0 (D8
                        # disabled), so the singleton never broke out of a tool-step
                        # thrash on S2/S3 bypass. Factory formula:
                        #   stall_cap = (max_steps - 1) if max_steps > 5 else 0
                        # (S1 budget too tight for any window — D8 off; S2→9, S3→19.)
                        # agent_loop.py:511 live-reads config.stall_after_tool_steps
                        # each step → mutation takes effect on next .run().
                        _new_max = self._agent_loop.config.max_steps
                        self._agent_loop.config.stall_after_tool_steps = (
                            _new_max - 1 if _new_max > 5 else 0
                        )

                        _original_llm = self._agent_loop._llm
                        _original_config = self._agent_loop.config.llm
                        active_model_id = ""
                        if decision is not None and bandit_provider is not None:
                            self._agent_loop._llm = bandit_provider
                            self._agent_loop.config.llm = bandit_config
                            active_model_id = decision.model_id
                            log.info(
                                "Stage 4 bypass: agent_loop using bandit-selected %s (S%d)",
                                decision.model_id, ctx.system,
                            )
                        else:
                            # Resolve model from Rust routing decision (preserve legacy selection)
                            routing_decision = getattr(self, '_last_routing_decision', None)
                            if routing_decision and routing_decision.model_id and self.provider_pool:
                                try:
                                    if self.provider_pool.is_model_available(routing_decision.model_id):
                                        resolved_provider, resolved_config = self.provider_pool.resolve(
                                            routing_decision.model_id
                                        )
                                        self._agent_loop._llm = resolved_provider
                                        self._agent_loop.config.llm = resolved_config
                                        active_model_id = routing_decision.model_id
                                        log.info(
                                            "Stage 4 bypass: agent_loop using Rust-selected %s (S%d)",
                                            routing_decision.model_id, ctx.system,
                                        )
                                except Exception:
                                    pass  # Keep default provider
                        if not active_model_id:
                            active_model_id = (
                                getattr(self._agent_loop.config.llm, "model", "")
                                or getattr(self._agent_loop._llm, "model_id", "")
                                or getattr(self._agent_loop._llm, "model_string", "")
                                or getattr(self._agent_loop._llm, "name", "")
                            )
                        ctx.executed_model_id = active_model_id
                        ctx.executed_template = "single_agent"

                        try:
                            ctx.result = await self._agent_loop.run(ctx.task)
                            ctx.cost = self._agent_loop.total_cost_usd
                            # Forward tool-use telemetry from the agent loop so bench
                            # manifests reflect actual usage, not dead zeros.
                            ctx.tool_call_count = getattr(self._agent_loop, "tool_call_count", 0)
                            ctx.tool_turn_count = getattr(self._agent_loop, "tool_turn_count", 0)
                            ctx.executed_commands = list(getattr(self._agent_loop, "executed_commands", []))
                        finally:
                            # A0a restoration — complete (12 fields, matches the
                            # snapshot taken before the first mutation above).
                            # Prior to 2026-04-23 this restored only 3 of the 10
                            # mutated fields, leaving write_gate / _on_drift /
                            # validation_level / max_steps / stall_after_tool_steps
                            # / _current_topology dirty for the next caller.
                            # P6-B (2026-05-04) wraps this entire block in a
                            # serializing lock — restoration still runs on
                            # exception/cancellation paths because the outer
                            # try/finally below releases the lock and the
                            # ContextVar token regardless of how we exit.
                            self._agent_loop._skip_routing = _orig_bypass_state["_skip_routing"]
                            self._agent_loop._current_topology = _orig_bypass_state["_current_topology"]
                            self._agent_loop.write_gate = _orig_bypass_state["write_gate"]
                            self._agent_loop.gate_current_task = _orig_bypass_state["gate_current_task"]
                            self._agent_loop.gate_source_tier = _orig_bypass_state["gate_source_tier"]
                            self._agent_loop._on_drift = _orig_bypass_state["_on_drift"]
                            self._agent_loop._run_frame_builder = _orig_bypass_state[
                                "_run_frame_builder"
                            ]
                            self._agent_loop._runtime_node_run_id = _orig_bypass_state[
                                "_runtime_node_run_id"
                            ]
                            self._agent_loop.config.validation_level = _orig_bypass_state["validation_level"]
                            self._agent_loop.config.max_steps = _orig_bypass_state["max_steps"]
                            self._agent_loop.config.stall_after_tool_steps = _orig_bypass_state["stall_after_tool_steps"]
                            self._agent_loop._llm = _original_llm
                            self._agent_loop.config.llm = _original_config
                    finally:
                        _BYPASS_AGENT_LOOP_ACTIVE.reset(_bypass_token)

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
