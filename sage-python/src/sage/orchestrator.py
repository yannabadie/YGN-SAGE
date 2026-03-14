"""Cognitive Orchestrator: dynamic multi-provider task decomposition and routing.

Takes a task, uses :class:`ComplexityRouter` to assess complexity,
selects the best model per subtask from the :class:`ModelRegistry`, and
composes lightweight agent runners into a topology (Sequential or Parallel).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from sage.execution_decision import ExecutionDecision
from sage.providers.registry import ModelRegistry, ModelProfile
from sage.providers.connector import PROVIDER_CONFIGS
from sage.quality_estimator import QualityEstimator
from sage.strategy.metacognition import ComplexityRouter
from sage.agents.sequential import SequentialAgent
from sage.agents.parallel import ParallelAgent
from sage.llm.base import LLMConfig, Message, Role
from sage.constants import (
    ORCHESTRATOR_S1_QUALITY,
    ORCHESTRATOR_S2_QUALITY,
    ORCHESTRATOR_S3_QUALITY,
    MAX_CASCADE_ATTEMPTS as _MAX_CASCADE_ATTEMPTS,
    MAX_TOPOLOGY_AGENTS,
)

log = logging.getLogger(__name__)

# ── Provider config lookup ────────────────────────────────────────────────────

# Build a fast lookup: provider name -> config dict
_PROVIDER_CFG_MAP: dict[str, dict[str, Any]] = {
    cfg["provider"]: cfg for cfg in PROVIDER_CONFIGS
}


def _provider_cfg_for(provider_name: str) -> dict[str, Any]:
    """Return the PROVIDER_CONFIGS entry for a given provider name.

    Falls back to a sensible default (OpenAI-compatible, no base_url).
    """
    return _PROVIDER_CFG_MAP.get(provider_name, {
        "provider": provider_name,
        "api_key_env": "",
        "base_url": None,
        "sdk": "openai",
    })


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SubTask:
    """A decomposed subtask with its cognitive requirements."""

    description: str
    needs_code: bool = False
    needs_reasoning: bool = False
    needs_tools: bool = False
    depends_on: list[str] = field(default_factory=list)  # names of prerequisite subtasks


@dataclass
class ExecutionPlan:
    """Plan of subtasks with their assigned models."""

    subtasks: list[SubTask]
    is_decomposed: bool = False  # True if task was decomposed, False if single-task


# ── ModelAgent ───────────────────────────────────────────────────────────────

class ModelAgent:
    """Lightweight agent that calls a specific LLM model via the appropriate provider.

    Supports cascade fallback: if the primary model fails, tries next-best
    models from the registry (FrugalGPT pattern, max 3 attempts).
    """

    MAX_CASCADE_ATTEMPTS = _MAX_CASCADE_ATTEMPTS

    def __init__(self, name: str, model: ModelProfile, system_prompt: str = "",
                 registry: ModelRegistry | None = None,
                 quality_threshold: float | None = None):
        self.name = name
        self.model = model
        self._system_prompt = system_prompt or "You are a helpful AI assistant. Be precise and concise."
        self._registry = registry
        self._quality_threshold = quality_threshold

    async def run(self, task: str) -> str:
        """Call the LLM model with quality-gated cascade.

        Flow: try model → check quality → if quality < threshold, escalate.
        Exception fallback still works as defense in depth.
        """
        messages: list[Message] = []
        if self._system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=self._system_prompt))
        messages.append(Message(role=Role.USER, content=task))

        tried_ids: set[str] = set()
        current_model = self.model
        last_error: Exception | None = None

        for attempt in range(self.MAX_CASCADE_ATTEMPTS):
            tried_ids.add(current_model.id)
            try:
                result = await self._call_provider(current_model, messages)

                # Quality-gated cascade: check if response is good enough
                if self._quality_threshold is not None and self._registry:
                    quality = QualityEstimator.estimate(task, result)
                    if quality < self._quality_threshold:
                        log.info(
                            "ModelAgent %s: quality %.2f < %.2f on %s, escalating",
                            self.name, quality, self._quality_threshold, current_model.id,
                        )
                        fallback = self._pick_fallback(tried_ids)
                        if fallback:
                            current_model = fallback
                            continue
                        # No better model available — return what we have
                        return result

                return result

            except Exception as e:
                last_error = e
                log.warning(
                    "ModelAgent %s: attempt %d/%d failed on %s (%s): %s",
                    self.name, attempt + 1, self.MAX_CASCADE_ATTEMPTS,
                    current_model.id, current_model.provider, e,
                )
                if self._registry:
                    fallback = self._pick_fallback(tried_ids)
                    if fallback:
                        log.info("Cascading to fallback model: %s", fallback.id)
                        current_model = fallback
                        continue
                break

        error_msg = str(last_error) if last_error else "Unknown error"
        return f"[Agent {self.name} error: all {len(tried_ids)} models failed. Last: {error_msg}]"

    async def _call_provider(self, model: ModelProfile, messages: list[Message]) -> str:
        """Make a single LLM call to the given model."""
        provider = self._create_provider_for(model)
        config = LLMConfig(
            provider=model.provider,
            model=model.id,
            temperature=0.3,
        )
        response = await provider.generate(messages, config=config)
        return response.content or ""

    def _pick_fallback(self, exclude_ids: set[str]) -> ModelProfile | None:
        """Pick the next-best available model not yet tried.

        Uses registry.select() to respect the original scoring formula,
        then falls back to cheapest-available if select() returns an
        already-tried model.
        """
        if not self._registry:
            return None
        for candidate in self._registry.list_available():
            if candidate.id not in exclude_ids and candidate.cost_input > 0:
                return candidate
        return None

    def _create_provider_for(self, model: ModelProfile):
        """Create the appropriate LLM provider for a model.

        Uses the ``sdk`` field from PROVIDER_CONFIGS:
        - ``"google-genai"`` -> :class:`GoogleProvider`
        - anything else     -> :class:`OpenAICompatProvider`
        """
        cfg = _provider_cfg_for(model.provider)
        sdk = cfg.get("sdk", "openai")
        api_key_env = cfg.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""

        if sdk == "google-genai":
            from sage.llm.google import GoogleProvider
            return GoogleProvider(api_key=api_key)
        else:
            from sage.providers.openai_compat import OpenAICompatProvider
            return OpenAICompatProvider(
                api_key=api_key,
                base_url=cfg.get("base_url"),
                model_id=model.id,
                provider_name=model.provider,
            )

    def _create_provider(self):
        """Backward-compatible: create provider for self.model."""
        return self._create_provider_for(self.model)


# ── CognitiveOrchestrator ───────────────────────────────────────────────────

class CognitiveOrchestrator:
    """Creates and executes dynamic agent topologies based on task analysis.

    Workflow:
        1. Assess task complexity via :class:`ComplexityRouter`.
        2. Route to S1 (fast), S2 (algorithmic), or S3 (complex/decomposed).
        3. Select optimal model(s) from :class:`ModelRegistry`.
        4. Build topology (single agent, Sequential, or Parallel).
        5. Execute and return result.

    Parameters
    ----------
    registry:
        Model registry with available model profiles.
    metacognition:
        Complexity router for task assessment and S1/S2/S3 routing.
    event_bus:
        Optional :class:`EventBus` for emitting orchestrator events.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        metacognition: ComplexityRouter | None = None,
        event_bus: Any = None,
    ):
        self.registry = registry
        self.metacognition = metacognition or ComplexityRouter()
        self.event_bus = event_bus

    async def run(self, task: str, decision: ExecutionDecision | None = None) -> str:
        """Analyze task, select models, build topology, execute.

        Parameters
        ----------
        task:
            The task string to execute.
        decision:
            Optional authoritative :class:`ExecutionDecision` produced by
            :class:`AgentSystem`.  When provided, re-routing is skipped —
            this eliminates the split-brain problem where both AgentSystem
            and CognitiveOrchestrator independently route the same task.
            When absent, falls back to local assessment for backward compat.
        """
        t0 = time.perf_counter()

        if decision is not None:
            # Use authoritative decision from AgentSystem — no re-routing.
            profile = type("Profile", (), {
                "complexity": 0.5, "uncertainty": 0.3,
                "tool_required": False, "system": decision.system,
            })()
            log.info(
                "Orchestrator: using ExecutionDecision S%d model=%s",
                decision.system, decision.model_id,
            )
        else:
            # Fallback: local assessment (backward compatibility)
            profile = await self.metacognition.assess_complexity_async(task)
            decision_local = self.metacognition.route(profile)
            decision = ExecutionDecision(
                system=decision_local.system,
                model_id="unknown",
            )
            log.info(
                "Orchestrator: task routed locally to S%d (c=%.2f u=%.2f)",
                decision.system, profile.complexity, profile.uncertainty,
            )

        # 2. S1: fast model for simple tasks (cost-sensitive)
        if decision.system == 1:
            model = self.registry.select({
                "code": 0.3, "reasoning": 0.3,
                "cost_sensitivity": 0.8,  # S1: optimize for speed/cost
                "max_cost_per_1m": 5.0,
            })
            if not model:
                model = self._any_available_model()
            if not model:
                return "No models available."
            agent = ModelAgent(name="s1-fast", model=model, registry=self.registry, quality_threshold=ORCHESTRATOR_S1_QUALITY)
            result = await agent.run(task)
            self._emit_event("ORCHESTRATOR", decision.system, model.id, time.perf_counter() - t0)
            return result

        # 3. S2: best model for moderate tasks (quality-first)
        if decision.system == 2:
            code_keywords = ("code", "function", "debug", "fix", "write", "implement", "refactor")
            if profile.tool_required or any(w in task.lower() for w in code_keywords):
                model = self.registry.select({
                    "code": 1.0,
                    "cost_sensitivity": 0.2,  # S2: quality dominates
                })
            else:
                model = self.registry.select({
                    "reasoning": 0.7, "code": 0.3,
                    "cost_sensitivity": 0.2,  # S2: quality dominates
                })
            if not model:
                model = self._any_available_model()
            if not model:
                return "No models available."
            agent = ModelAgent(name="s2-worker", model=model, registry=self.registry, quality_threshold=ORCHESTRATOR_S2_QUALITY)
            result = await agent.run(task)
            self._emit_event("ORCHESTRATOR", decision.system, model.id, time.perf_counter() - t0)
            return result

        # 4. S3: try to decompose into subtasks
        plan = await self._decompose(task)

        if not plan.is_decomposed or len(plan.subtasks) <= 1:
            # Single complex task — use the absolute best reasoner (no cost limit)
            model = self.registry.select({
                "reasoning": 1.0,
                "cost_sensitivity": 0.0,  # S3: pure quality, cost irrelevant
            })
            if not model:
                model = self._any_available_model()
            if not model:
                return "No models available."
            agent = ModelAgent(name="s3-reasoner", model=model, registry=self.registry, quality_threshold=ORCHESTRATOR_S3_QUALITY)
            result = await agent.run(task)
            self._emit_event("ORCHESTRATOR", decision.system, model.id, time.perf_counter() - t0)
            return result

        # 5. Multi-subtask: assign best model per subtask (quality-first)
        agents: list[ModelAgent] = []
        for subtask in plan.subtasks:
            needs: dict[str, float] = {"cost_sensitivity": 0.1}  # S3 subtasks: quality-first
            if subtask.needs_code:
                needs["code"] = 1.0
            if subtask.needs_reasoning:
                needs["reasoning"] = 1.0
            if "code" not in needs and "reasoning" not in needs:
                needs["reasoning"] = 0.5
                needs["code"] = 0.5

            model = self.registry.select(needs)
            if not model:
                model = self._any_available_model()
            if not model:
                continue

            agent = ModelAgent(
                name=f"subtask-{len(agents)}",
                model=model,
                system_prompt=f"Focus on this specific subtask: {subtask.description}",
                registry=self.registry,
                quality_threshold=ORCHESTRATOR_S3_QUALITY,
            )
            agents.append(agent)
            log.info(
                "  Subtask '%s' -> %s (%s)",
                subtask.description[:50], model.id, model.provider,
            )

        if not agents:
            return "No models available for any subtask."

        # Check dependencies — if all independent, run parallel; otherwise sequential
        has_deps = any(st.depends_on for st in plan.subtasks)

        if has_deps or len(agents) == 1:
            topology = SequentialAgent(name="s3-sequential", agents=agents)
        else:

            def synthesize(results: dict[str, str]) -> str:
                parts = [f"## {name}\n{text}" for name, text in results.items()]
                return "\n\n".join(parts)

            topology = ParallelAgent(name="s3-parallel", agents=agents, aggregator=synthesize)

        result = await topology.run(task)
        self._emit_event("ORCHESTRATOR", decision.system, "multi-model", time.perf_counter() - t0)
        return result

    # ── Decomposition ────────────────────────────────────────────────────

    async def _decompose(self, task: str) -> ExecutionPlan:
        """Decompose a complex task into subtasks using a cheap/fast model.

        Best-effort: if decomposition fails, returns a single-subtask plan.
        """
        # Use the cheapest available model for decomposition
        model = self.registry.select({"code": 0.2, "reasoning": 0.2, "max_cost_per_1m": 1.0})
        if not model:
            return ExecutionPlan(subtasks=[SubTask(description=task)], is_decomposed=False)

        decompose_prompt = (
            "Analyze this task and break it into 2-4 independent subtasks. "
            "For each subtask, indicate if it needs: code generation, deep reasoning, or tool use. "
            "Format: one subtask per line, starting with [CODE], [REASON], or [GENERAL].\n\n"
            f"Task: {task}\n\n"
            "Subtasks:"
        )

        try:
            agent = ModelAgent(name="decomposer", model=model, registry=self.registry)
            response = await agent.run(decompose_prompt)
            subtasks = self._parse_subtasks(response)
            if subtasks:
                return ExecutionPlan(subtasks=subtasks, is_decomposed=True)
        except Exception as e:
            log.warning("Task decomposition failed: %s", e)

        return ExecutionPlan(subtasks=[SubTask(description=task)], is_decomposed=False)

    def _parse_subtasks(self, response: str) -> list[SubTask]:
        """Parse decomposition response into SubTask objects.

        Expects one subtask per line, optionally prefixed with [CODE], [REASON],
        or [GENERAL].  Returns at most 4 subtasks.  Lines shorter than 10 chars
        are discarded.
        """
        subtasks: list[SubTask] = []
        for line in response.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if not line or len(line) < 10:
                continue
            upper = line.upper()
            needs_code = "[CODE]" in upper or "code" in line.lower()
            needs_reasoning = (
                "[REASON]" in upper
                or "reason" in line.lower()
                or "prove" in line.lower()
                or "verify" in line.lower()
            )
            clean = (
                line.replace("[CODE]", "")
                .replace("[REASON]", "")
                .replace("[GENERAL]", "")
                .strip()
            )
            if clean:
                subtasks.append(SubTask(
                    description=clean,
                    needs_code=needs_code,
                    needs_reasoning=needs_reasoning,
                ))
        return subtasks[:MAX_TOPOLOGY_AGENTS]  # Max subtasks

    # ── Helpers ──────────────────────────────────────────────────────────

    def _any_available_model(self) -> ModelProfile | None:
        """Fallback: return any available model (cheapest first)."""
        available = self.registry.list_available()
        return available[0] if available else None

    def _emit_event(self, type_: str, system: int, model: str, elapsed: float) -> None:
        """Emit an orchestrator event on the EventBus."""
        if self.event_bus:
            try:
                from sage.agent_loop import AgentEvent
                self.event_bus.emit(AgentEvent(
                    type=type_,
                    step=0,
                    timestamp=time.time(),
                    system=system,
                    model=model,
                    latency_ms=round(elapsed * 1000, 1),
                ))
            except Exception as e:
                log.warning("Failed to emit orchestrator event: %s", e)
