"""Metacognitive Controller: SOFAI-style System 1/3 routing + self-braking."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class CognitiveProfile:
    """Assessment of a task's cognitive requirements."""
    complexity: float     # 0.0 = trivial, 1.0 = extremely complex
    uncertainty: float    # 0.0 = certain, 1.0 = highly uncertain
    tool_required: bool   # Whether tool use is expected


@dataclass
class RoutingDecision:
    """Which system and LLM tier to use."""
    system: int           # 1 = fast/intuitive, 3 = formal/deliberate
    llm_tier: str         # fast, mutator, reasoner, codex
    max_tokens: int
    use_z3: bool          # Whether to validate with Z3 PRM


class MetacognitiveController:
    """Routes tasks between System 1 (fast) and System 3 (formal reasoning).

    SOFAI pattern: evaluates task complexity and model confidence to decide
    whether to use fast heuristic LLM or full Z3-backed reasoning pipeline.

    Self-braking (CGRS): monitors output entropy to detect convergence
    and suppress unnecessary reasoning loops.
    """

    def __init__(
        self,
        complexity_threshold: float = 0.5,
        uncertainty_threshold: float = 0.4,
        brake_window: int = 3,
        brake_entropy_threshold: float = 0.15,
    ):
        self.complexity_threshold = complexity_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.brake_window = brake_window
        self.brake_entropy_threshold = brake_entropy_threshold
        self._entropy_history: deque[float] = deque(maxlen=10)

    def route(self, profile: CognitiveProfile) -> RoutingDecision:
        """Decide which cognitive system to engage."""
        needs_system3 = (
            profile.complexity > self.complexity_threshold
            or profile.uncertainty > self.uncertainty_threshold
            or profile.tool_required
        )

        if needs_system3:
            # High complexity: use formal reasoning
            tier = "reasoner"
            if profile.complexity > 0.8:
                tier = "codex"  # Hardest tasks get agentic Codex
            return RoutingDecision(
                system=3, llm_tier=tier,
                max_tokens=8192, use_z3=True,
            )
        else:
            # Simple task: fast heuristic
            return RoutingDecision(
                system=1, llm_tier="fast",
                max_tokens=2048, use_z3=False,
            )

    def record_output_entropy(self, entropy: float) -> None:
        """Record the entropy of the latest LLM output for self-braking."""
        self._entropy_history.append(entropy)

    def should_brake(self) -> bool:
        """Determine if the agent should stop reasoning (convergence detected).

        CGRS: If the last N outputs all have low entropy, the model has
        implicitly converged and further reasoning is wasteful.
        """
        if len(self._entropy_history) < self.brake_window:
            return False
        recent = list(self._entropy_history)[-self.brake_window:]
        return all(e < self.brake_entropy_threshold for e in recent)

    def assess_complexity(self, task: str) -> CognitiveProfile:
        """Quick heuristic assessment of task complexity."""
        lower = task.lower()

        complexity = 0.3  # Base
        if any(w in lower for w in ["debug", "fix", "error", "crash"]):
            complexity += 0.3
        if any(w in lower for w in ["optimize", "evolve", "design", "architect"]):
            complexity += 0.2
        if len(task) > 500:
            complexity += 0.1

        uncertainty = 0.2
        if "?" in task:
            uncertainty += 0.2
        if any(w in lower for w in ["maybe", "possibly", "explore", "investigate"]):
            uncertainty += 0.2

        tool_required = any(w in lower for w in [
            "file", "search", "run", "execute", "compile", "test", "deploy"
        ])

        return CognitiveProfile(
            complexity=min(1.0, complexity),
            uncertainty=min(1.0, uncertainty),
            tool_required=tool_required,
        )
