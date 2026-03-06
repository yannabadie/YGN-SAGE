"""Cognitive Orchestrator: dynamic multi-provider task decomposition and routing.

Takes a task, uses :class:`MetacognitiveController` to assess complexity,
selects the best model per subtask from the :class:`ModelRegistry`, and
composes lightweight agent runners into a topology (Sequential or Parallel).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from sage.providers.registry import ModelRegistry, ModelProfile
from sage.providers.connector import PROVIDER_CONFIGS
from sage.strategy.metacognition import MetacognitiveController
from sage.agents.sequential import SequentialAgent
from sage.agents.parallel import ParallelAgent
from sage.llm.base import LLMConfig, Message, Role

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

    Resolves provider connection details (API key env var, base URL, SDK)
    from :data:`PROVIDER_CONFIGS` based on ``model.provider``.
    """

    def __init__(self, name: str, model: ModelProfile, system_prompt: str = ""):
        self.name = name
        self.model = model
        self._system_prompt = system_prompt or "You are a helpful AI assistant. Be precise and concise."

    async def run(self, task: str) -> str:
        """Call the LLM model and return the response text."""
        provider = self._create_provider()
        messages: list[Message] = []
        if self._system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=self._system_prompt))
        messages.append(Message(role=Role.USER, content=task))

        config = LLMConfig(
            provider=self.model.provider,
            model=self.model.id,
            temperature=0.3,
        )

        response = await provider.generate(messages, config=config)
        return response.content or ""

    def _create_provider(self):
        """Create the appropriate LLM provider for this model.

        Uses the ``sdk`` field from PROVIDER_CONFIGS:
        - ``"google-genai"`` -> :class:`GoogleProvider`
        - anything else     -> :class:`OpenAICompatProvider`
        """
        cfg = _provider_cfg_for(self.model.provider)
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
                model_id=self.model.id,
            )


# ── CognitiveOrchestrator ───────────────────────────────────────────────────

class CognitiveOrchestrator:
    """Creates and executes dynamic agent topologies based on task analysis.

    Workflow:
        1. Assess task complexity via :class:`MetacognitiveController`.
        2. Route to S1 (fast), S2 (algorithmic), or S3 (complex/decomposed).
        3. Select optimal model(s) from :class:`ModelRegistry`.
        4. Build topology (single agent, Sequential, or Parallel).
        5. Execute and return result.

    Parameters
    ----------
    registry:
        Model registry with available model profiles.
    metacognition:
        Metacognitive controller for complexity assessment and routing.
    event_bus:
        Optional :class:`EventBus` for emitting orchestrator events.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        metacognition: MetacognitiveController | None = None,
        event_bus: Any = None,
    ):
        self.registry = registry
        self.metacognition = metacognition or MetacognitiveController()
        self.event_bus = event_bus

    async def run(self, task: str) -> str:
        """Analyze task, select models, build topology, execute."""
        t0 = time.perf_counter()

        # 1. Assess complexity
        profile = self.metacognition.assess_complexity(task)
        decision = self.metacognition.route(profile)

        log.info(
            "Orchestrator: task routed to S%d (c=%.2f u=%.2f)",
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
            agent = ModelAgent(name="s1-fast", model=model)
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
            agent = ModelAgent(name="s2-worker", model=model)
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
            agent = ModelAgent(name="s3-reasoner", model=model)
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
            agent = ModelAgent(name="decomposer", model=model)
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
        return subtasks[:4]  # Max 4 subtasks

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
