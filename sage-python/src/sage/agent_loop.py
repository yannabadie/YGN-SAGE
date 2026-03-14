"""Structured agent runtime: perceive -> think -> act -> learn."""
from __future__ import annotations

import ast
import math
import os
import re
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
    STAGNATION_WINDOW,
    DRIFT_CHECK_INTERVAL,
    RELEVANCE_GATE_THRESHOLD,
    S3_MAX_RETRIES,
    DEFAULT_COST_PER_1K,
)

log = logging.getLogger(__name__)


# Approximate cost per 1K tokens (USD) for dashboard estimation
# Updated 2026-03-14 — real models verified via API discovery
_COST_PER_1K = {
    # OpenAI (gpt-4.1 family)
    "gpt-4.1": 0.005,
    "gpt-4.1-mini": 0.001,
    "gpt-4.1-nano": 0.00025,
    "gpt-4o": 0.00625,
    # xAI (Grok family)
    "grok-4.20-beta": 0.009,
    "grok-4-1-fast-reasoning": 0.00035,
    "grok-code-fast-1": 0.00035,
    "grok-3": 0.009,
    # MiniMax
    "MiniMax-Text-01": 0.00075,
    # Google (available when API key works)
    "gemini-2.5-flash": 0.0003,
    "gemini-2.5-flash-lite": 0.00015,
    "gemini-2.5-pro": 0.00565,
    # GLM/Zhipu
    "glm-4.7": 0.001,
    "glm-5": 0.002,
    # DeepSeek
    "deepseek-chat": 0.00035,
    "deepseek-reasoner": 0.00035,
    # Kimi
    "kimi-k2.5": 0.00155,
}


def _estimate_tokens(text: str, actual_count: int | None = None) -> int:
    """Return actual token count from API if available, else rough estimate."""
    if actual_count is not None and actual_count > 0:
        return actual_count
    return max(1, len(text) // 4)


def _text_entropy(text: str) -> float:
    """Shannon entropy of character distribution (normalised 0-1)."""
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    ent = -sum((c / n) * math.log2(c / n) for c in freq.values() if c > 0)
    max_ent = math.log2(max(len(freq), 2))
    return ent / max_ent if max_ent > 0 else 0.0


def _extract_code_blocks(text: str) -> list[str]:
    """Extract fenced code blocks from markdown-style LLM output."""
    pattern = r"```(?:\w+)?\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


def _strip_markdown_fences(code: str) -> str:
    """Strip leading/trailing markdown fences from a code string."""
    code = code.strip()
    # Remove leading ```python or ``` line
    if code.startswith("```"):
        first_newline = code.find("\n")
        if first_newline != -1:
            code = code[first_newline + 1:]
        else:
            code = code[3:]
    # Remove trailing ```
    if code.rstrip().endswith("```"):
        code = code.rstrip()[:-3]
    return code.strip()


def _validate_code_syntax(code: str) -> tuple[bool, str]:
    """Validate Python code syntax via ast.parse().

    Returns (is_valid, error_message). If valid, error_message is empty.
    The error_message uses SLF (Single-Line Feedback) format for concise LLM guidance.
    """
    cleaned = _strip_markdown_fences(code)
    if not cleaned:
        return False, "SyntaxError: empty code block after stripping markdown fences"
    try:
        ast.parse(cleaned, mode="exec")
        return True, ""
    except SyntaxError as e:
        line_info = f" (line {e.lineno})" if e.lineno else ""
        return False, f"SyntaxError{line_info}: {e.msg}"


def _is_stagnating(error_history: list[str], window: int = STAGNATION_WINDOW) -> bool:
    """Detect stagnation: True if the last `window` errors are identical.

    This means the LLM is producing the same broken code repeatedly
    and retrying will not help — escalation is needed.
    """
    if len(error_history) < window:
        return False
    recent = error_history[-window:]
    return all(e == recent[0] for e in recent)


S2_MAX_RETRIES_BEFORE_ESCALATION = _S2_MAX_RETRIES_BEFORE_ESCALATION
S2_AVR_MAX_ITERATIONS = _S2_AVR_MAX_ITERATIONS  # Max Act-Verify-Refine iterations per code block
MAX_MESSAGES = MAX_AGENT_MESSAGES  # Keep system + user + last N exchanges


def _is_code_task(task: str) -> bool:
    """Detect if task is primarily about code generation.

    Used to skip episodic/semantic memory injection for code tasks,
    which Sprint 3 evidence shows degrades accuracy (30% vs 50% no-memory).
    """
    lower = task.lower()
    return bool(re.search(
        r'\b(?:implement|code|function|class|method|algorithm|program|'
        r'write\s+(?:a\s+)?(?:function|method|class|code|script)|'
        r'python|javascript|rust|java|def\s|return\s)\b', lower
    ))


def _shell_quote(code: str) -> str:
    """Shell-quote a code string for subprocess execution."""
    import shlex
    return shlex.quote(code)


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
        """Use Rust TopologyExecutor to get node execution order.

        Returns list of node specs for topology-aware execution.
        Falls back to empty list if executor unavailable.
        """
        if not self._current_topology:
            return []
        try:
            from sage_core import TopologyExecutor as PyTopologyExecutor  # noqa: E402
            executor = PyTopologyExecutor(self._current_topology)
            schedule: list[dict] = []
            while not executor.is_done():
                ready = executor.next_ready(self._current_topology)
                if not ready:
                    break
                for idx in ready:
                    node = self._current_topology.get_node(idx)
                    schedule.append({
                        "index": idx,
                        "role": node.role,
                        "model_id": node.model_id,
                        "system": node.system,
                    })
                    executor.mark_completed(idx)
            return schedule
        except Exception:
            return []

    async def _run_topology(self, task: str) -> str | None:
        """Execute multi-node topology via TopologyRunner.

        Returns the result string if topology has >1 node, or None to fall
        through to standard single-LLM execution.
        """
        if not self._current_topology:
            return None

        schedule = self._schedule_from_topology()
        if len(schedule) <= 1:
            return None  # Single node or empty — use standard path

        try:
            from sage.topology.runner import TopologyRunner
            from sage_core import TopologyExecutor as PyTopologyExecutor  # noqa: E402
            executor = PyTopologyExecutor(self._current_topology)
            runner = TopologyRunner(
                graph=self._current_topology,
                executor=executor,
                llm_provider=self._llm,
                llm_config=self.config.llm,
            )
            result = await runner.run(task)
            self._emit(
                LoopPhase.THINK,
                topology_execution="multi_agent",
                node_count=len(schedule),
            )
            return result
        except Exception as e:
            log.warning("TopologyRunner failed (%s), falling back to single-LLM", e)
            return None

    async def _cegar_repair(
        self,
        content: str,
        prm_details: dict[str, Any] | str,
        invariant_feedback: list[str],
    ) -> str | None:
        """Attempt CEGAR repair of failed S3 verification.

        Extracts failed clauses from PRM details and invariant feedback,
        builds a targeted repair prompt, and makes a single LLM call.
        Returns repaired content if PRM passes, or None if repair fails.
        """
        repair_prompt = (
            "SYSTEM: Your formal verification FAILED. "
            "Do NOT regenerate from scratch — fix the specific failures below.\n\n"
            f"Verification error: {prm_details}\n"
        )
        if invariant_feedback:
            repair_prompt += (
                "\nFailed invariant clauses:\n"
                + "\n".join(f"- {f}" for f in invariant_feedback)
                + "\n"
            )
        repair_prompt += (
            "\nFix your reasoning by adding the missing formal assertions. "
            "Use <think> tags with Z3 assertions for each step."
        )

        messages = [
            Message(role=Role.SYSTEM, content=self.config.system_prompt),
            Message(role=Role.ASSISTANT, content=content),
            Message(role=Role.USER, content=repair_prompt),
        ]

        try:
            response = await self._llm.generate(
                messages=messages,
                config=self.config.llm,
            )
            repaired = response.content or ""
            if not repaired:
                return None

            # Verify the repair
            r_path, details = self.prm.calculate_r_path(repaired)
            if r_path >= 0.0 or "error" not in details:
                log.info("CEGAR repair succeeded: r_path=%.2f", r_path)
                return repaired
            else:
                log.warning("CEGAR repair failed: %s", details)
                return None
        except Exception as e:
            log.warning("CEGAR repair LLM call failed: %s", e)
            return None

    async def _execute_tool_call(self, tc) -> str:
        """Execute a tool call with argument validation."""
        tool = self._tools.get(tc.name)
        if tool is None:
            return f"Error: Unknown tool '{tc.name}'"
        kwargs = tc.arguments
        if not isinstance(kwargs, dict):
            log.warning("Tool '%s' received non-dict arguments: %s", tc.name, type(kwargs))
            return f"Error: Tool '{tc.name}' received invalid arguments (expected dict, got {type(kwargs).__name__})"
        try:
            result = await tool.execute(kwargs.copy())
            return result.output
        except Exception as e:
            log.error("Tool '%s' execution failed: %s", tc.name, e)
            return f"Error executing tool '{tc.name}': {type(e).__name__}: {e}"

    async def run(self, task: str) -> str:
        """Execute the full perceive -> think -> act -> learn cycle.

        Delegates to phase modules in sage.phases for maintainability.
        Set SAGE_AGENT_LOOP_LEGACY=1 to use the frozen legacy implementation.

        Note: ExoCortex passive grounding removed per Sprint 3 evidence.
        Use active tool (search_exocortex) instead — agent invokes when needed.
        """
        if os.environ.get("SAGE_AGENT_LOOP_LEGACY") == "1":
            return await self._run_legacy(task)

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
        except Exception as exc:
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

    async def _run_legacy(self, task: str) -> str:
        """Legacy run() — frozen copy for SAGE_AGENT_LOOP_LEGACY=1 fallback."""
        self.start_time = time.perf_counter()
        self.total_cost_usd = 0.0
        self._s3_retries = 0
        self._s2_avr_retries = 0
        self._avr_error_history = []
        self._s3_degraded = False
        self._original_validation_level = self.config.validation_level
        self.step_count = 0

        # === PERCEIVE: Gather context ===
        perceive_meta: dict[str, Any] = {"task": task, "agent": self.config.name}

        # Metacognitive routing (if wired)
        if self._skip_routing:
            perceive_meta["system"] = 2
            perceive_meta["routing_source"] = "ablation_forced_s2"
        elif self.metacognition:
            profile = await self.metacognition.assess_complexity_async(task)
            decision = self.metacognition.route(profile)
            perceive_meta["system"] = decision.system
            perceive_meta["routed_tier"] = decision.llm_tier
            perceive_meta["use_z3"] = decision.use_z3
            perceive_meta["validation_level"] = decision.validation_level
            perceive_meta["complexity"] = round(profile.complexity, 2)
            perceive_meta["uncertainty"] = round(profile.uncertainty, 2)
            if profile.reasoning:
                perceive_meta["routing_reason"] = profile.reasoning
            perceive_meta["routing_source"] = "llm" if profile.reasoning != "heuristic" else "heuristic"

        self._emit(LoopPhase.PERCEIVE, **perceive_meta)

        # Input guardrail check
        if self.guardrail_pipeline and not self._skip_guardrails:
            try:
                input_results = await self.guardrail_pipeline.check_all(
                    input=task, context={"step": 0, "agent": self.config.name}
                )
                for r in input_results:
                    self._emit(LoopPhase.PERCEIVE,
                               guardrail=r.__class__.__name__ if hasattr(r, '__class__') else "input",
                               guardrail_passed=r.passed,
                               guardrail_reason=r.reason)
                if self.guardrail_pipeline.any_blocked(input_results):
                    blocked = [r for r in input_results if not r.passed]
                    return f"Blocked by guardrail: {blocked[0].reason}"
            except Exception as e:
                log.warning("Input guardrail error: %s", e)

        system_prompt = self.config.system_prompt
        if self.config.validation_level >= 3:
            system_prompt += (
                "\n\nCRITICAL: Use <think>...</think> tags for formal reasoning. "
                "Include Z3-verifiable assertions in your reasoning steps:\n"
                "- assert bounds(address, limit) — prove memory safety\n"
                "- assert loop(variable) — prove loop termination\n"
                "- assert arithmetic(expression, expected) — prove arithmetic correctness\n"
                "- assert invariant(\"precondition\", \"postcondition\") — prove logical invariants\n"
                "Your reasoning is verified by Z3 SMT solver. "
                "Steps with proven assertions score 1.0. Steps without score 0.0."
            )
        elif self.config.validation_level >= 2:
            system_prompt += (
                "\n\nUse step-by-step reasoning to solve this task. "
                "Show your work clearly."
            )
            if _is_code_task(task):
                system_prompt += (
                    "\n\nIMPORTANT edge cases to handle: "
                    "empty inputs ([], \"\"), negative numbers, zero values, "
                    "single-element collections, very large numbers, "
                    "boolean-is-int in Python (type(True)==int), "
                    "floating-point precision (use abs(a-b)<epsilon). "
                    "Return ONLY the complete function in a ```python block."
                )

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=task),
        ]
        self.working_memory.add_event("USER", task)
        tool_defs = self._tools.get_tool_defs(
            self.config.tools if self.config.tools else None
        )

        result_text = ""

        # Semantic memory context injection (one-time, before loop)
        if self.semantic_memory and not self._cb_semantic.should_skip() and not self._skip_memory:
            try:
                sem_context = self.semantic_memory.get_context_for(task)
                if sem_context and self._relevance_gate.is_relevant(task, sem_context):
                    messages.insert(1, Message(
                        role=Role.SYSTEM,
                        content=f"Relevant knowledge from previous interactions:\n{sem_context}",
                    ))
                self._cb_semantic.record_success()
            except Exception as e:
                self._cb_semantic.record_failure(e)

        # S-MMU context injection (graph-based retrieval from compacted chunks)
        if not self._cb_smmu.should_skip() and not self._skip_memory:
            try:
                from sage.memory.smmu_context import retrieve_smmu_context
                smmu_context = retrieve_smmu_context(self.working_memory)
                if smmu_context and self._relevance_gate.is_relevant(task, smmu_context):
                    messages.insert(
                        min(2, len(messages)),
                        Message(role=Role.SYSTEM, content=smmu_context),
                    )
                self._cb_smmu.record_success()
            except Exception as e:
                self._cb_smmu.record_failure(e)

        while self.step_count < self.config.max_steps:
            self.step_count += 1

            # Memory compression if needed
            if self.memory_compressor:
                compressed = await self.memory_compressor.step(self.working_memory)
                if compressed:
                    messages = self._rebuild_messages(system_prompt)

            # === THINK: Call LLM ===
            if self.step_count == 1:
                topology_result = await self._run_topology(task)
                if topology_result is not None:
                    content = topology_result
                    result_text = content
                    self.working_memory.add_event("ASSISTANT", content)
                    break

            model_name = self.config.llm.model
            self._emit(LoopPhase.THINK, model=model_name)

            t0 = time.perf_counter()
            response = await self._llm.generate(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                config=self.config.llm,
            )
            inference_ms = (time.perf_counter() - t0) * 1000
            self.total_inference_time += inference_ms / 1000

            content = response.content or ""

            usage = getattr(response, "usage", None) or {}
            actual_total = usage.get("total_tokens") if isinstance(usage, dict) else None
            tokens = _estimate_tokens(content, actual_count=actual_total)
            cost_per_k = _COST_PER_1K.get(model_name, DEFAULT_COST_PER_1K)
            step_cost = (tokens / 1000) * cost_per_k
            self.total_cost_usd += step_cost

            entropy = _text_entropy(content)
            brake = False
            if self.metacognition:
                self.metacognition.record_output_entropy(entropy)
                brake = self.metacognition.should_brake()

            self._emit(
                LoopPhase.THINK,
                model=model_name,
                content=content,
                latency_ms=round(inference_ms, 1),
                cost_usd=round(self.total_cost_usd, 4),
                entropy=round(entropy, 3),
                brake=brake,
            )

            if self.memory_compressor and content:
                await self.memory_compressor.generate_internal_state(
                    f"[Step {self.step_count}] {content[:300]}"
                )

            # System 3 validation (Z3 PRM)
            if self.config.validation_level >= 3 and content:
                r_path, details = self.prm.calculate_r_path(content)
                self._emit(LoopPhase.THINK, r_path=r_path, details=details)

                if r_path < 0.0 and "error" in details:
                    self._s3_retries += 1
                    if self._s3_retries <= self._max_s3_retries:
                        messages.append(Message(
                            role=Role.USER,
                            content=(
                                "SYSTEM: Your reasoning lacks formal assertions. "
                                "Use <think> tags with Z3 assertions:\n"
                                "- assert bounds(addr, limit)\n"
                                "- assert loop(var)\n"
                                "- assert arithmetic(expr, expected)\n"
                                "- assert invariant(\"precondition\", \"postcondition\")\n"
                                "Include at least one formal assertion per reasoning step."
                            ),
                        ))
                        continue

                    inv_feedback = getattr(self.prm.kg, "_last_invariant_feedback", [])
                    repaired = await self._cegar_repair(content, details, inv_feedback)
                    if repaired is not None:
                        content = repaired
                        self._s3_retries = 0
                    else:
                        log.warning(
                            "S3 verification failed after CEGAR repair — "
                            "degrading to S2 AVR."
                        )
                        self._emit(
                            LoopPhase.THINK,
                            s3_degradation=True,
                            reason="CEGAR repair failed",
                        )
                        self.config.validation_level = 2
                        self._s3_degraded = True
                        self._s3_retries = 0
                        self._s2_avr_retries = 0
                        continue
                else:
                    self._s3_retries = 0

            # System 2 validation (Empirical — AVR)
            elif self.config.validation_level == 2 and content and not self._skip_avr:
                code_blocks = _extract_code_blocks(content)

                if code_blocks and self.sandbox_manager:
                    raw_code = code_blocks[-1]
                    cleaned_code = _strip_markdown_fences(raw_code)

                    _te_rejected = False
                    if self.tool_executor:
                        try:
                            te_result = self.tool_executor.validate(cleaned_code)
                            if not te_result.valid:
                                te_err = "; ".join(te_result.errors)
                                log.warning("ToolExecutor rejected code: %s", te_err)
                                _te_rejected = True
                                syntax_ok, syntax_err = False, te_err
                            else:
                                log.debug("ToolExecutor validated code successfully")
                        except Exception as e:
                            log.warning("ToolExecutor failed, falling back to Python: %s", e)

                    if not _te_rejected:
                        syntax_ok, syntax_err = _validate_code_syntax(cleaned_code)

                    if not syntax_ok:
                        self._s2_avr_retries += 1
                        self._avr_error_history.append(syntax_err)

                        if _is_stagnating(self._avr_error_history, window=3):
                            log.warning("S2 AVR stagnation detected (same error %d times), forcing escalation.",
                                        len(self._avr_error_history))
                            self._s2_avr_retries = self._max_s2_avr_retries + 1
                        else:
                            budget_left = self._max_s2_avr_retries - self._s2_avr_retries + 1
                            self._emit(LoopPhase.ACT,
                                       validation="s2_avr_fail",
                                       avr_iteration=self._s2_avr_retries,
                                       avr_budget_left=budget_left,
                                       error_type="syntax",
                                       error=syntax_err)
                            if self._s2_avr_retries <= self._max_s2_avr_retries:
                                log.info("S2 AVR syntax fail (iteration %d/%d): %s",
                                         self._s2_avr_retries, self._max_s2_avr_retries, syntax_err)
                                messages.append(Message(
                                    role=Role.USER,
                                    content=(
                                        f"SYSTEM [AVR {self._s2_avr_retries}/{self._max_s2_avr_retries}]: "
                                        f"Syntax error in your code:\n```\n{syntax_err}\n```\n"
                                        f"Fix the syntax error and return ONLY corrected Python code "
                                        f"in a ```python fenced block."
                                    ),
                                ))
                                continue
                    else:
                        if self.guardrail_pipeline and not self._cb_runtime_guard.should_skip() and not self._skip_guardrails:
                            try:
                                runtime_results = await self.guardrail_pipeline.check_all(
                                    input=cleaned_code,
                                    context={"step": self.step_count, "phase": "runtime"}
                                )
                                for r in runtime_results:
                                    self._emit(LoopPhase.ACT,
                                               guardrail="runtime",
                                               guardrail_passed=r.passed,
                                               guardrail_reason=r.reason)
                                self._cb_runtime_guard.record_success()
                            except Exception as e:
                                self._cb_runtime_guard.record_failure(e)

                        sandbox = await self.sandbox_manager.create()
                        try:
                            result = await sandbox.execute(
                                f"python3 -c {_shell_quote(cleaned_code)}"
                            )
                            if result.exit_code != 0:
                                stderr_full = (result.stderr or "").strip()
                                stderr_last = stderr_full.split("\n")[-1][:200]
                                stdout_snippet = (result.stdout or "").strip()[:200]
                                runtime_err = f"RuntimeError (exit {result.exit_code}): {stderr_last}"
                                self._s2_avr_retries += 1
                                self._avr_error_history.append(runtime_err)

                                if _is_stagnating(self._avr_error_history, window=3):
                                    log.warning("S2 AVR stagnation detected (same runtime error %d times), forcing escalation.",
                                                len(self._avr_error_history))
                                    self._s2_avr_retries = self._max_s2_avr_retries + 1
                                else:
                                    budget_left = self._max_s2_avr_retries - self._s2_avr_retries + 1
                                    self._emit(LoopPhase.ACT,
                                               validation="s2_avr_fail",
                                               avr_iteration=self._s2_avr_retries,
                                               avr_budget_left=budget_left,
                                               error_type="runtime",
                                               error=runtime_err)
                                    if self._s2_avr_retries <= self._max_s2_avr_retries:
                                        log.info("S2 AVR runtime fail (iteration %d/%d): %s",
                                                 self._s2_avr_retries, self._max_s2_avr_retries, runtime_err)
                                        feedback_parts = [
                                            f"SYSTEM [AVR {self._s2_avr_retries}/{self._max_s2_avr_retries}]: Code execution failed.",
                                        ]
                                        if stderr_full:
                                            feedback_parts.append(f"Traceback:\n```\n{stderr_full[:500]}\n```")
                                        if stdout_snippet:
                                            feedback_parts.append(f"Stdout: {stdout_snippet}")
                                        feedback_parts.append(
                                            "Analyze the error, fix the bug, and return ONLY corrected Python code "
                                            "in a ```python fenced block."
                                        )
                                        messages.append(Message(
                                            role=Role.USER,
                                            content="\n".join(feedback_parts),
                                        ))
                                        continue
                            else:
                                self._emit(LoopPhase.ACT,
                                           validation="s2_avr_pass",
                                           stdout=result.stdout[:200])
                                self._s2_avr_retries = 0
                                self._avr_error_history.clear()
                        finally:
                            await self.sandbox_manager.destroy(sandbox.id)

                elif not code_blocks and self.step_count == 1:
                    has_reasoning = "<think>" in content or "\n1." in content or "\n- " in content
                    if not has_reasoning:
                        self._s2_avr_retries += 1
                        if self._s2_avr_retries <= self._max_s2_avr_retries:
                            log.info("S2 validation: missing reasoning, requesting CoT.")
                            messages.append(Message(
                                role=Role.USER,
                                content="SYSTEM: Provide step-by-step reasoning for this task.",
                            ))
                            continue

                # S2 -> S3 escalation
                if self._s2_avr_retries > self._max_s2_avr_retries and self.config.validation_level == 2 and not self._s3_degraded:
                    log.info("S2 AVR exhausted — escalating to S3 (formal verification).")
                    self.config.validation_level = 3
                    self._s3_retries = 0
                    self._avr_error_history.clear()
                    self._emit(LoopPhase.THINK, escalation="s2_to_s3",
                               reason="AVR budget exhausted")
                    escalation_msg = (
                        "SYSTEM: Escalating to formal verification. Use <think> tags "
                        "with Z3 assertions (assert bounds, assert loop, assert arithmetic, "
                        "assert invariant) for rigorous step-by-step reasoning."
                    )
                    inv_feedback = getattr(self.prm.kg, "_last_invariant_feedback", [])
                    if inv_feedback:
                        escalation_msg += (
                            "\n\nPrevious invariant verification failures:\n"
                            + "\n".join(f"- {f}" for f in inv_feedback)
                        )
                    messages.append(Message(
                        role=Role.USER,
                        content=escalation_msg,
                    ))
                    continue

            # CGRS: stop if converged
            if brake:
                log.info("CGRS self-brake triggered — stopping reasoning loop.")
                result_text = content
                self.working_memory.add_event("ASSISTANT", content)
                messages.append(Message(role=Role.ASSISTANT, content=content))
                break

            self.working_memory.add_event("ASSISTANT", content)

            if self.episodic_memory and len(content) > 100 and not self._cb_episodic.should_skip() and not self._skip_memory:
                try:
                    await self.episodic_memory.store(
                        key=f"step-{self.step_count}",
                        content=content[:500],
                        metadata={"task": task, "step": self.step_count},
                    )
                    self._cb_episodic.record_success()
                except Exception as e:
                    self._cb_episodic.record_failure(e)

            if self.memory_agent and self.semantic_memory and content and len(content) > 50 and not self._cb_entity.should_skip() and not self._skip_memory:
                try:
                    extraction = await self.memory_agent.extract(content[:1000])
                    if extraction.entities:
                        self.semantic_memory.add_extraction(extraction)
                    self._cb_entity.record_success()
                except Exception as e:
                    self._cb_entity.record_failure(e)

            if not response.tool_calls:
                result_text = content
                messages.append(Message(role=Role.ASSISTANT, content=content))
                break

            # === ACT: Execute tools ===
            messages.append(Message(role=Role.ASSISTANT, content=content))

            for tc in response.tool_calls:
                self._emit(LoopPhase.ACT, tool=tc.name, args=tc.arguments)
                output = await self._execute_tool_call(tc)
                self.working_memory.add_event("TOOL", f"{tc.name} -> {output}")
                messages.append(Message(
                    role=Role.TOOL, content=output,
                    tool_call_id=tc.id, name=tc.name,
                ))

            if len(messages) > MAX_MESSAGES:
                messages = messages[:2] + messages[-(MAX_MESSAGES - 2):]

            # === LEARN: Update memory and stats ===
            aio = self._compute_aio()
            wall = time.perf_counter() - self.start_time
            learn_meta: dict[str, Any] = {
                "aio_ratio": aio,
                "events": self.working_memory.event_count(),
                "wall_time_s": round(wall, 1),
                "cost_usd": round(self.total_cost_usd, 4),
            }

            if self.agent_pool and hasattr(self.agent_pool, "list_agents"):
                learn_meta["sub_agents"] = self.agent_pool.list_agents()

            if self.semantic_memory:
                learn_meta["semantic_entities"] = self.semantic_memory.entity_count()

            if self._auto_evolve and self.topology_population and self.topology_population.size() > 0 and not self._cb_evo.should_skip():
                try:
                    cells = []
                    best_fitness = 0.0
                    for (x, y), (genome, score) in self.topology_population._grid.items():
                        cells.append({"x": x, "y": y, "fitness": round(score, 2)})
                        best_fitness = max(best_fitness, score)
                    learn_meta["evo_cells"] = cells
                    learn_meta["evo_best"] = round(best_fitness, 2)
                    learn_meta["evo_grid_size"] = len(cells)
                    self._cb_evo.record_success()
                except Exception as e:
                    self._cb_evo.record_failure(e)

            self._emit(LoopPhase.LEARN, **learn_meta)

        self._last_avr_iterations = self._s2_avr_retries

        if self.guardrail_pipeline and result_text and not self._skip_guardrails:
            try:
                output_results = await self.guardrail_pipeline.check_all(
                    output=result_text,
                    context={"cost_usd": self.total_cost_usd, "steps": self.step_count}
                )
                for r in output_results:
                    self._emit(LoopPhase.LEARN,
                               guardrail="output",
                               guardrail_passed=r.passed,
                               guardrail_reason=r.reason)
            except Exception as e:
                log.warning("Output guardrail error: %s", e)

        final_meta: dict[str, Any] = {
            "result": "complete",
            "steps": self.step_count,
            "cost_usd": round(self.total_cost_usd, 4),
            "response_text": result_text or "",
            "task": task,
        }
        if self.semantic_memory:
            final_meta["semantic_entities"] = self.semantic_memory.entity_count()
        self._emit(LoopPhase.LEARN, **final_meta)
        self.config.validation_level = self._original_validation_level
        return result_text or f"Agent finished at step {self.step_count}"

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
