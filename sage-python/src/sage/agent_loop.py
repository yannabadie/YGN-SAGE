"""Structured agent runtime: perceive -> think -> act -> learn."""
from __future__ import annotations

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
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
        """Execute a tool call with argument validation."""
        from sage.agent_loop_execution import execute_tool_call
        return await execute_tool_call(tc, self._tools, self._emit)

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
        self._s3_retries = 0
        self._s2_avr_retries = 0
        self._avr_error_history = []
        self._s3_degraded = False
        self._original_validation_level = self.config.validation_level
        self.step_count = 0

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
                break

            if t_result.loop_action == "continue":
                # S3 retry or S3->S2 degradation — re-enter loop
                continue

            # === ACT ===
            a_result = await act(
                task, t_result.content, t_result.response, t_result.brake,
                messages, self,
            )

            if a_result.loop_action == "break":
                result_text = a_result.result_text
                break

            if a_result.loop_action == "continue":
                # AVR retry or S2->S3 escalation — re-enter loop
                continue

            # === LEARN (in-loop) ===
            await learn_step(self)

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
