"""Structured agent runtime: perceive -> think -> act -> learn."""
from __future__ import annotations

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from collections.abc import AsyncIterator
from typing import Any, Callable

from sage.agent import AgentConfig
from sage.llm.base import LLMProvider, Message, Role
from sage.tools.registry import ToolRegistry
from sage.memory.working import WorkingMemory
from sage.memory.compressor import MemoryCompressor
from sage.topology.kg_rlvr import ProcessRewardModel
from sage.resilience import CircuitBreaker
from sage.memory.relevance_gate import RelevanceGate
from sage.monitoring.drift import DriftMonitor
from sage.constants import (
    S2_AVR_MAX_ITERATIONS as _S2_AVR_MAX_ITERATIONS,
    S2_MAX_RETRIES_BEFORE_ESCALATION as _S2_MAX_RETRIES_BEFORE_ESCALATION,
    MAX_AGENT_MESSAGES,
    DRIFT_CHECK_INTERVAL,
    RELEVANCE_GATE_THRESHOLD,
    S3_MAX_RETRIES,
    DEFAULT_COST_PER_1K,
    CONSOLIDATION_INTERVAL_STEPS,
)

# Re-export utility functions for backward compatibility.
# Canonical implementations live in agent_loop_utils.py.
from sage.agent_loop_utils import (  # noqa: F401
    _COST_PER_1K,
    _load_cost_table,
    _estimate_tokens,
    _text_entropy,
    _extract_code_blocks,
    _strip_markdown_fences,
    _validate_code_syntax,
    _is_stagnating,
    _is_code_task,
    _shell_quote,
)

log = logging.getLogger(__name__)

S2_MAX_RETRIES_BEFORE_ESCALATION = _S2_MAX_RETRIES_BEFORE_ESCALATION
S2_AVR_MAX_ITERATIONS = _S2_AVR_MAX_ITERATIONS  # Max Act-Verify-Refine iterations per code block
MAX_MESSAGES = MAX_AGENT_MESSAGES  # Keep system + user + last N exchanges


class LoopPhase(str, Enum):
    PERCEIVE = "perceive"
    THINK = "think"
    ACT = "act"
    LEARN = "learn"


@dataclass
class LoopEvent:
    phase: LoopPhase
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    step: int = 0


@dataclass
class AgentLoopExhaustion:
    """Structured metadata when the loop exits without final content.

    D4 fix (2026-04-18 audit docs/audits/2026-04-18-astropy-14995-*):
    before this struct existed, the loop returned a sentinel STRING
    (``phases/learn.py:EMPTY_STEP_SENTINEL``) that callers had to detect
    via substring match. Controllers had no way to learn WHY the agent
    stalled — just that "something" emitted 51 chars.

    This dataclass carries the *why* forward. Callers (TopologyRunner,
    TopologyController) read ``loop.last_exhaustion`` after ``run()`` to
    decide whether to upgrade_model, spawn_subagent, or fail the node.

    Attributes
    ----------
    reason:
        "budget_exhausted" — ran to max_steps with no final content.
        "stalled" — D8 soft cap: N consecutive tool turns, no content.
    step_count:
        Step the loop exited at. Equals ``loop.config.max_steps`` for
        budget_exhausted; less for stalled.
    consecutive_tool_steps:
        How many steps in a row called tools without producing final
        content. Useful for distinguishing true thrash (20/20) from
        near-miss (18/20 tools, 2 with reasoning).
    last_tool_name:
        Name of the last tool invoked, if any. Often the repeated tool
        that wasted the budget.
    last_assistant_snippet:
        First 200 chars of the last ASSISTANT message content if any —
        lets controllers see what the agent was *trying* to say.
    """
    reason: str = "budget_exhausted"
    step_count: int = 0
    consecutive_tool_steps: int = 0
    last_tool_name: str | None = None
    last_assistant_snippet: str | None = None


class AgentLoopBudgetExhausted(RuntimeError):
    """Raised when the agent loop hits max_steps with no final content.

    D4 fix. Not raised by default to preserve backward compat with
    callers that depend on the sentinel-string return. Opt-in via
    ``AgentConfig.raise_on_exhaustion=True`` (added in the same patch).

    Carries the ``AgentLoopExhaustion`` metadata as ``.detail``.
    """
    def __init__(self, detail: AgentLoopExhaustion):
        self.detail = detail
        super().__init__(
            f"agent_loop budget exhausted at step {detail.step_count} "
            f"(reason={detail.reason}, "
            f"consecutive_tool_steps={detail.consecutive_tool_steps})"
        )


@dataclass
class AgentEvent:
    """Versioned structured event for observability (v1)."""
    type: str                           # PERCEIVE, THINK, ACT, LEARN
    step: int
    timestamp: float
    schema_version: int = 1
    latency_ms: float | None = None
    cost_usd: float | None = None
    tokens_est: int | None = None
    model: str | None = None
    system: int | None = None           # 1, 2, or 3
    routing_source: str | None = None   # "llm" or "heuristic"
    validation: str | None = None       # s2_avr_pass, s2_avr_fail, s3_prm_pass, etc.
    meta: dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    """Structured agent loop with event emission for dashboard."""

    def __init__(
        self,
        config: AgentConfig,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry | None = None,
        memory_compressor: MemoryCompressor | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
    ):
        self.config = config
        self._llm = llm_provider
        self._tools = tool_registry or ToolRegistry()
        self._on_event = on_event or self._default_event_handler
        self.working_memory = WorkingMemory(agent_id=config.name)
        self.memory_compressor = memory_compressor
        self.prm = ProcessRewardModel()
        self.agent_pool: Any = {}  # dict or AgentPool, injected by boot.py
        # Injected by boot.py
        self.metacognition: Any = None
        self.topology_population: Any = None
        self.episodic_memory: Any = None  # EpisodicMemory for cross-session storage
        self.sandbox_manager: Any = None  # SandboxManager for S2 validation
        self.exocortex: Any = None  # ExoCortex for File Search grounding
        self.guardrail_pipeline: Any = None  # GuardrailPipeline for input/output/runtime checks
        self.memory_agent: Any = None       # MemoryAgent for entity extraction
        self.semantic_memory: Any = None    # SemanticMemory entity graph
        self.causal_memory: Any = None      # CausalMemory for causal relations (injected by boot.py)
        self.consolidator: Any = None       # MemoryConsolidator for episodic->semantic consolidation
        self.tool_executor: Any = None  # Injected by boot.py (Rust tree-sitter + subprocess)
        self.topology_engine: Any = None  # Injected by boot.py (Rust TopologyEngine)
        self._current_topology: Any = None  # Set by boot.py before each run

        # Ablation skip flags (set by AblationConfig.apply)
        self._skip_memory: bool = False
        self._skip_avr: bool = False
        self._skip_routing: bool = False
        self._skip_guardrails: bool = False
        self._auto_evolve: bool = False  # Enabled by boot.py when topology_population is set

        # Stats
        self.step_count = 0
        self.total_inference_time = 0.0
        self.total_cost_usd = 0.0
        # Tool-call telemetry. Telemetry-only: the counters don't gate any
        # decision; they're surfaced to PipelineContext and bench manifests
        # so we can tell "agent never called tools" from "tools were called
        # but the task budget ran out" without trusting reporter inference.
        self.tool_call_count = 0   # total individual function_call invocations
        self.tool_turn_count = 0   # turns on which the LLM returned >=1 tool call
        self.executed_commands: list[str] = []  # bash commands executed (truncated)
        # D4 audit fix (2026-04-18): populated by phases/learn.py when the
        # loop exits without final content. Runner/Controller read this
        # to decide whether to upgrade_model / spawn_subagent / fail.
        # None means "run completed with final content".
        self.last_exhaustion: AgentLoopExhaustion | None = None
        # D8 audit fix (2026-04-18): tracks consecutive steps where the
        # LLM called tools but emitted no final content. Used for soft-cap
        # stall detection. Reset to 0 on any step that produces final
        # content.
        self._consecutive_tool_steps = 0
        # Last tool name invoked — carried into AgentLoopExhaustion for
        # controller diagnostics (often the repeated tool that thrashed).
        self._last_tool_name: str | None = None
        self.start_time = 0.0
        self._s3_retries = 0
        self._max_s3_retries = S3_MAX_RETRIES
        self._s2_avr_retries = 0
        self._max_s2_avr_retries = S2_MAX_RETRIES_BEFORE_ESCALATION
        self._avr_error_history: list[str] = []
        self._last_avr_iterations: int = 0
        self._last_error: Exception | None = None
        self._s3_degraded: bool = False

        # CRAG-style relevance gate for memory injection
        self._relevance_gate = RelevanceGate(threshold=RELEVANCE_GATE_THRESHOLD)

        # G-series audit fix (2026-04-19): shared-state write gate for memory
        # writes in phases/act.py. Injected by boot/factory, None if unavailable.
        # Shared across nodes of ONE task so cross-node duplicate writes hit
        # the exact-dedup hash (the failure mode we saw in astropy-14995
        # where planner/coder/synthesizer each emitted the same sentinel).
        self.write_gate: Any = None
        # Source tier string for the gate's reliability signal — mapped from
        # the LLM model id via cards.toml in agent_loop_factory.py.
        self.gate_source_tier: str = "unknown"
        # Current task text for the gate's relevance signal — set at the
        # start of each run() or forwarded from the outer pipeline context.
        self.gate_current_task: str = ""

        # Plateau detector (P1.1 of 2026-04-18 mega-plan).
        # Loops that repeat the same tool-call arguments or the same empty
        # reply step after step usually waste 90 % of their step budget,
        # then the bench-side wall-clock timeout fires and the prediction
        # is classified empty. Tracking the last K step signatures lets us
        # bail early with the current best output. Size 3 = we tolerate one
        # accidental repeat; three in a row is the "stuck" signal.
        self._recent_step_signatures: deque[str] = deque(maxlen=3)

        # Circuit breakers for best-effort subsystems
        self._cb_semantic = CircuitBreaker("semantic_memory")
        self._cb_smmu = CircuitBreaker("smmu_context")
        self._cb_runtime_guard = CircuitBreaker("runtime_guardrails")
        self._cb_episodic = CircuitBreaker("episodic_store")
        self._cb_entity = CircuitBreaker("entity_extraction")
        self._cb_evo = CircuitBreaker("evolution_stats")
        self._cb_causal = CircuitBreaker("causal_memory")

        # Drift detection — monitors latency/error/cost trends
        self._drift_monitor = DriftMonitor()
        self._drift_events: list[AgentEvent] = []
        self._drift_check_interval = DRIFT_CHECK_INTERVAL  # Analyze every N events
        self._consolidation_steps_total = 0  # Persist maintenance cadence across runs
        # D6 audit fix (2026-04-18): drift action was previously log-only.
        # Callers (TopologyRunner) inject a callback to forward SWITCH_MODEL
        # / RESET_AGENT events to ProviderPool.record_failure so repeated
        # drift against the same provider trips the circuit breaker and
        # subsequent resolve() calls pick a different provider. Signature:
        # on_drift(provider_hint: str, action: str, details: dict) -> None.
        self._on_drift: Callable[[str, str, dict[str, Any]], None] | None = None

    def _emit(self, phase: LoopPhase, **data: Any) -> None:
        evt = AgentEvent(
            type=phase.value.upper(),
            step=self.step_count,
            timestamp=time.time(),
            latency_ms=data.pop("latency_ms", None),
            cost_usd=data.pop("cost_usd", None),
            tokens_est=data.pop("tokens_est", None),
            model=data.pop("model", None),
            system=data.pop("system", None),
            routing_source=data.pop("routing_source", None),
            validation=data.pop("validation", None),
            meta=data,
        )
        self._on_event(evt)

        # Drift monitoring: accumulate events and check periodically
        self._drift_events.append(evt)
        if len(self._drift_events) >= self._drift_check_interval:
            report = self._drift_monitor.analyze(self._drift_events)
            if report.action != "CONTINUE":
                log.warning("Drift detected: score=%.3f action=%s details=%s",
                            report.drift_score, report.action, report.details)
                # D6 audit fix: forward actionable drift to caller callback
                # so ProviderPool can record a failure / flip the circuit.
                # Was log-only before — SWITCH_MODEL classifications in
                # monitoring/drift.py:99 had zero downstream effect.
                if self._on_drift is not None:
                    try:
                        _hint = str(getattr(evt, "model", "") or "")
                        self._on_drift(_hint, report.action, report.details or {})
                    except Exception as _drift_exc:  # noqa: BLE001 — telemetry
                        log.debug("on_drift callback failed: %s", _drift_exc)
                self._on_event(AgentEvent(
                    type="DRIFT",
                    step=self.step_count,
                    timestamp=time.time(),
                    meta={"drift_score": report.drift_score,
                          "drift_action": report.action,
                          "drift_details": report.details},
                ))
            # Sliding window: keep last half for overlap
            self._drift_events = self._drift_events[self._drift_check_interval // 2:]

    def _default_event_handler(self, event: AgentEvent) -> None:
        log.info(f"[{event.type}] step={event.step} model={event.model}")

    async def _maybe_run_consolidation(self) -> None:
        """Run bounded inter-tier consolidation on cumulative step intervals."""
        self._consolidation_steps_total += 1

        if (self.consolidator is None
                or self._consolidation_steps_total % CONSOLIDATION_INTERVAL_STEPS != 0):
            return

        try:
            consolidation_result = await self.consolidator.consolidate()
            if getattr(consolidation_result, "processed", 0) > 0:
                log.debug(
                    "AgentLoop consolidation: %d episodes -> %d entities",
                    consolidation_result.processed,
                    getattr(consolidation_result, "entities_added", 0),
                )
        except Exception as exc:
            log.debug("AgentLoop consolidation failed: %s", exc)

    def _schedule_from_topology(self) -> list[dict]:
        """Use Rust TopologyExecutor to get node execution order."""
        from sage.agent_loop_execution import schedule_from_topology
        return schedule_from_topology(self._current_topology)

    async def _run_topology(self, task: str) -> str | None:
        """Execute multi-node topology via TopologyRunner."""
        from sage.agent_loop_execution import run_topology
        return await run_topology(
            task, self._current_topology, self._llm, self.config.llm, self._emit,
        )

    async def _cegar_repair(
        self,
        content: str,
        prm_details: dict[str, Any] | str,
        invariant_feedback: list[str],
    ) -> str | None:
        """Attempt CEGAR repair of failed S3 verification."""
        from sage.agent_loop_execution import cegar_repair
        return await cegar_repair(
            content, prm_details, invariant_feedback,
            system_prompt=self.config.system_prompt,
            llm_provider=self._llm,
            llm_config=self.config.llm,
            prm=self.prm,
        )

    async def _execute_tool_call(self, tc) -> str:
        """Execute a tool call with argument validation.

        Forwards the loop's ToolForge (if any, wired in boot.py) so unknown
        tools trigger autonomous synthesis instead of a hard error.
        """
        from sage.agent_loop_execution import execute_tool_call
        return await execute_tool_call(
            tc,
            self._tools,
            self._emit,
            toolforge=getattr(self, "toolforge", None),
            task_context=getattr(self, "_current_task", "") or "",
        )

    async def run(self, task: str) -> str:
        """Execute the full perceive -> think -> act -> learn cycle.

        Delegates to phase modules in sage.phases for maintainability.

        Note: ExoCortex passive grounding removed per Sprint 3 evidence.
        Use active tool (search_exocortex) instead — agent invokes when needed.
        """
        from sage.phases.perceive import perceive
        from sage.phases.think import think
        from sage.phases.act import act
        from sage.phases.learn import learn_step, learn_final

        # Initialize run state
        self.start_time = time.perf_counter()
        self.total_cost_usd = 0.0
        self.tool_call_count = 0
        self.tool_turn_count = 0
        self.executed_commands = []
        self._recent_step_signatures.clear()
        self._s3_retries = 0
        self._s2_avr_retries = 0
        self._avr_error_history = []
        self._s3_degraded = False
        self._original_validation_level = self.config.validation_level
        self.step_count = 0
        # Exposed for _execute_tool_call so synthesized tools (ToolForge
        # CreationTicket) know what task they were asked to support.
        self._current_task = task

        # Reset ToolForge per-run counters so each run gets its own
        # MAX_CREATIONS budget (default: 2).
        if getattr(self, "toolforge", None) is not None:
            try:
                self.toolforge.reset_run()
            except Exception:
                pass

        # === PERCEIVE ===
        p_result = await perceive(task, self)
        if p_result.blocked_reason:
            return p_result.blocked_reason

        messages = p_result.messages
        system_prompt = p_result.system_prompt
        tool_defs = p_result.tool_defs
        result_text = ""

        # === Main loop: THINK -> ACT -> LEARN ===
        while self.step_count < self.config.max_steps:
            self.step_count += 1

            # Memory compression if needed
            if self.memory_compressor:
                compressed = await self.memory_compressor.step(self.working_memory)
                if compressed:
                    messages = self._rebuild_messages(system_prompt)

            # === THINK ===
            t_result = await think(task, messages, system_prompt, tool_defs, self)

            if t_result.loop_action == "break":
                # Topology result used — skip to final LEARN
                result_text = t_result.content
                await self._maybe_run_consolidation()
                break

            if t_result.loop_action == "continue":
                # S3 retry or S3->S2 degradation — re-enter loop
                await self._maybe_run_consolidation()
                continue

            # Plateau detector (P1.1): record a signature for this step
            # and bail early when the last 3 are identical — that means
            # the agent is repeating itself (same content, same tool calls)
            # and burning its step budget for nothing. Signature is the
            # content plus the sorted tool-call argument tuples. No tools
            # on this turn ⇒ only content is considered (a stuck "I need
            # to…" monologue still counts).
            _sig_parts = [str(t_result.content or "").strip()[:512]]
            _tc = getattr(t_result.response, "tool_calls", None) or []
            for _call in _tc:
                _name = getattr(_call, "name", "") or ""
                _args = getattr(_call, "arguments", None)
                _sig_parts.append(f"{_name}:{_args!r}")
            _sig = "|".join(_sig_parts)
            self._recent_step_signatures.append(_sig)
            if (
                len(self._recent_step_signatures) == self._recent_step_signatures.maxlen
                and len(set(self._recent_step_signatures)) == 1
                and _sig
            ):
                log.warning(
                    "[%s] plateau detected after step %d — breaking early with "
                    "current best output to avoid burning remaining step budget",
                    self.config.name, self.step_count,
                )
                result_text = t_result.content or result_text
                await self._maybe_run_consolidation()
                break

            # === ACT ===
            a_result = await act(
                task, t_result.content, t_result.response, t_result.brake,
                messages, self,
            )

            if a_result.loop_action == "break":
                result_text = a_result.result_text
                await self._maybe_run_consolidation()
                break

            if a_result.loop_action == "continue":
                # AVR retry or S2->S3 escalation — re-enter loop
                await self._maybe_run_consolidation()
                continue

            # === D8 soft-cap stall detection (audit 2026-04-18) ===
            # Track consecutive tool-calling steps that produce no final
            # content. After `stall_after_tool_steps` in a row, bail early
            # with structured AgentLoopExhaustion(reason="stalled") instead
            # of burning the full max_steps budget. Default 0 = disabled
            # (backward compat); TopologyRunner wires it to 10 for
            # sequential nodes so a thrashing coder doesn't eat 20 steps.
            _tool_step_this_turn = bool(_tc)
            _produced_content = bool(a_result.result_text)
            if _tool_step_this_turn and not _produced_content:
                self._consecutive_tool_steps += 1
                # Capture the most recent tool name for controller diagnostics
                try:
                    self._last_tool_name = getattr(_tc[0], "name", None)
                except (AttributeError, IndexError):
                    pass
            else:
                self._consecutive_tool_steps = 0

            _stall_cap = int(getattr(self.config, "stall_after_tool_steps", 0) or 0)
            if _stall_cap > 0 and self._consecutive_tool_steps >= _stall_cap:
                log.warning(
                    "[%s] stall detected: %d consecutive tool steps with no "
                    "final content — breaking early (D8 soft cap, step=%d/%d)",
                    self.config.name,
                    self._consecutive_tool_steps,
                    self.step_count,
                    self.config.max_steps,
                )
                from sage.agent_loop import AgentLoopExhaustion as _AgentLoopExhaustion
                self.last_exhaustion = _AgentLoopExhaustion(
                    reason="stalled",
                    step_count=self.step_count,
                    consecutive_tool_steps=self._consecutive_tool_steps,
                    last_tool_name=self._last_tool_name,
                    last_assistant_snippet=(t_result.content or "")[:200] if t_result.content else None,
                )
                # Let learn_final observe last_exhaustion and return the
                # sentinel (or raise if raise_on_exhaustion=True).
                await self._maybe_run_consolidation()
                break

            # === LEARN (in-loop) ===
            await learn_step(self)
            await self._maybe_run_consolidation()

        # === LEARN (final) ===
        return await learn_final(task, result_text, self)

    async def stream(self, task: str) -> AsyncIterator[str]:
        """Stream LLM response tokens for non-AVR tasks.

        For code tasks (AVR path) or providers without streaming support,
        falls back to ``run()`` and yields the full result as a single chunk.

        Yields chunks of text as they arrive from the LLM.  This is the
        Phase 1 implementation: streaming only applies to simple text
        generation (non-code, non-AVR).  The perceive/act/learn phases
        are not modified.
        """
        from sage.llm.base import StreamingLLMProvider

        # Code tasks need AVR -- fall back to full run()
        if _is_code_task(task):
            log.debug("stream(): code task detected, falling back to run()")
            result = await self.run(task)
            yield result
            return

        # Provider must support streaming
        if not isinstance(self._llm, StreamingLLMProvider):
            log.debug("stream(): provider %s has no streaming, falling back to run()",
                       getattr(self._llm, "name", type(self._llm).__name__))
            result = await self.run(task)
            yield result
            return

        # --- Lightweight perceive (build messages) ---
        from sage.phases.perceive import perceive

        self.start_time = time.perf_counter()
        self.total_cost_usd = 0.0
        self.tool_call_count = 0
        self.tool_turn_count = 0
        self.executed_commands = []
        self._s3_retries = 0
        self._s2_avr_retries = 0
        self._avr_error_history = []
        self._s3_degraded = False
        self._original_validation_level = self.config.validation_level
        self.step_count = 0

        p_result = await perceive(task, self)
        if p_result.blocked_reason:
            yield p_result.blocked_reason
            return

        messages = p_result.messages

        self.step_count = 1
        self._emit(LoopPhase.THINK, model=self.config.llm.model)

        # --- Stream tokens from the provider ---
        collected_chunks: list[str] = []
        try:
            async for chunk in self._llm.generate_stream(
                messages=messages,
                config=self.config.llm,
            ):
                collected_chunks.append(chunk)
                yield chunk
        except (RuntimeError, TimeoutError) as exc:
            log.warning("stream(): streaming failed (%s), falling back to run()", exc)
            # If we already yielded partial content, the caller has
            # inconsistent output -- but this is best-effort Phase 1.
            if not collected_chunks:
                result = await self.run(task)
                yield result
            return

        full_text = "".join(collected_chunks)

        # Emit final THINK event with aggregated content
        tokens = _estimate_tokens(full_text)
        _load_cost_table()
        cost_per_k = _COST_PER_1K.get(self.config.llm.model, DEFAULT_COST_PER_1K)
        step_cost = (tokens / 1000) * cost_per_k
        self.total_cost_usd += step_cost
        self._emit(
            LoopPhase.THINK,
            model=self.config.llm.model,
            content=full_text,
            cost_usd=round(self.total_cost_usd, 4),
        )

        # Record in working memory
        self.working_memory.add_event("ASSISTANT", full_text)

    def _compute_aio(self) -> float:
        wall = time.perf_counter() - self.start_time
        if wall <= 0:
            return 0.0
        return max(0.0, (wall - self.total_inference_time) / wall)

    def _rebuild_messages(self, system_prompt: str) -> list[Message]:
        msgs = [Message(role=Role.SYSTEM, content=system_prompt)]
        for event in self.working_memory._events:
            role_map = {
                "SYSTEM": Role.SYSTEM, "USER": Role.USER,
                "ASSISTANT": Role.ASSISTANT, "TOOL": Role.USER,
                "summary": Role.SYSTEM,
            }
            role = role_map.get(event["type"], Role.USER)
            msgs.append(Message(role=role, content=event["content"]))
        return msgs
