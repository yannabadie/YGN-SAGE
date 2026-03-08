"""Complexity-based S1/S2/S3 router with self-braking.

Routes tasks to three tiers based on heuristic or LLM-based complexity assessment:
  S1 (fast): Gemini Flash Lite, no validation (~$0.0005/call, <1s).
  S2 (algorithmic): Gemini Flash/Pro, empirical validation (~$0.0015/1K).
  S3 (formal): Codex/Reasoner, Z3 PRM formal verification (~$0.03/1K).

Uses Gemini 2.5 Flash Lite for LLM-based task assessment.
Falls back to heuristic if GOOGLE_API_KEY is not set.
"""
from __future__ import annotations

import logging
import os
import re
from collections import deque
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class CognitiveProfile:
    """Assessment of a task's cognitive requirements."""
    complexity: float     # 0.0 = trivial, 1.0 = extremely complex
    uncertainty: float    # 0.0 = certain, 1.0 = highly uncertain
    tool_required: bool   # Whether tool use is expected
    reasoning: str = ""   # LLM explanation of the assessment


@dataclass
class RoutingDecision:
    """Which system and LLM tier to use."""
    system: int           # 1 = fast/intuitive, 2 = algorithmic/deliberate, 3 = formal/verified
    llm_tier: str         # fast, mutator, reasoner, codex
    max_tokens: int
    use_z3: bool          # Whether to validate with Z3 PRM
    validation_level: int = 1  # 1=none, 2=empirical, 3=formal(Z3)


# Structured output schema for Gemini Flash Lite routing
_ROUTING_SCHEMA = {
    "type": "object",
    "properties": {
        "complexity": {
            "type": "number",
            "description": "Task complexity from 0.0 (trivial) to 1.0 (extremely complex)",
        },
        "uncertainty": {
            "type": "number",
            "description": "Epistemic uncertainty from 0.0 (certain) to 1.0 (highly uncertain)",
        },
        "tool_required": {
            "type": "boolean",
            "description": "Whether the task likely requires tool use (file I/O, search, execution)",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief (1-2 sentence) explanation of the assessment",
        },
    },
    "required": ["complexity", "uncertainty", "tool_required", "reasoning"],
}

_ROUTING_PROMPT = """You are a metacognitive router. Assess the following task and return a JSON object with:
- complexity (0.0-1.0): how hard is this task? 0=trivial factual, 0.3=simple, 0.5=moderate, 0.7=hard reasoning, 1.0=multi-step research
- uncertainty (0.0-1.0): how uncertain is the answer? 0=well-known fact, 0.5=requires analysis, 1.0=open-ended/speculative
- tool_required (bool): does this need file access, code execution, search, or external tools?
- reasoning: 1-2 sentence explanation

Task: {task}"""


class ComplexityRouter:
    """Complexity-based S1/S2/S3 router.

    Routes tasks to one of three tiers based on assessed complexity/uncertainty:
      S1: Fast/intuitive (Gemini Flash Lite, no validation)
      S2: Algorithmic/deliberate (Gemini Flash/Pro, empirical validation)
      S3: Formal/verified (Codex/Reasoner, Z3 PRM validation)

    Self-braking (CGRS): monitors output entropy to detect convergence.
    """

    def __init__(
        self,
        s1_complexity_ceil: float = 0.50,
        s1_uncertainty_ceil: float = 0.3,
        s3_complexity_floor: float = 0.65,
        s3_uncertainty_floor: float = 0.6,
        brake_window: int = 3,
        brake_entropy_threshold: float = 0.15,
    ):
        self.s1_complexity_ceil = s1_complexity_ceil
        self.s1_uncertainty_ceil = s1_uncertainty_ceil
        self.s3_complexity_floor = s3_complexity_floor
        self.s3_uncertainty_floor = s3_uncertainty_floor
        self.brake_window = brake_window
        self.brake_entropy_threshold = brake_entropy_threshold
        self._entropy_history: deque[float] = deque(maxlen=10)
        self._llm_available: bool | None = None

    def route(self, profile: CognitiveProfile) -> RoutingDecision:
        """Decide which cognitive system to engage (S1/S2/S3)."""
        c, u = profile.complexity, profile.uncertainty

        # System 3: high complexity OR high uncertainty
        if c > self.s3_complexity_floor or u > self.s3_uncertainty_floor:
            tier = "codex" if c > 0.8 else "reasoner"
            return RoutingDecision(
                system=3, llm_tier=tier,
                max_tokens=8192, use_z3=True, validation_level=3,
            )

        # System 1: low complexity AND low uncertainty AND no tools
        if (c <= self.s1_complexity_ceil
                and u <= self.s1_uncertainty_ceil
                and not profile.tool_required):
            return RoutingDecision(
                system=1, llm_tier="fast",
                max_tokens=2048, use_z3=False, validation_level=1,
            )

        # System 2: everything in between
        tier = "reasoner" if c > 0.55 else "mutator"
        return RoutingDecision(
            system=2, llm_tier=tier,
            max_tokens=4096, use_z3=False, validation_level=2,
        )

    def record_output_entropy(self, entropy: float) -> None:
        """Record the entropy of the latest LLM output for self-braking."""
        self._entropy_history.append(entropy)

    def should_brake(self) -> bool:
        """CGRS: stop if last N outputs all have low entropy (convergence)."""
        if len(self._entropy_history) < self.brake_window:
            return False
        recent = list(self._entropy_history)[-self.brake_window:]
        return all(e < self.brake_entropy_threshold for e in recent)

    async def assess_complexity_async(self, task: str) -> CognitiveProfile:
        """LLM-based task assessment via Gemini Flash Lite.

        Cost: ~$0.0005 per call. Latency: <1s.
        Falls back to heuristic if Google API key is missing.
        """
        if self._llm_available is None:
            self._llm_available = bool(os.environ.get("GOOGLE_API_KEY"))

        if self._llm_available:
            try:
                return await self._assess_via_llm(task)
            except Exception as e:
                log.warning(f"LLM routing failed ({e}), falling back to heuristic")

        return self._assess_heuristic(task)

    def assess_complexity(self, task: str) -> CognitiveProfile:
        """Synchronous assessment (heuristic only). Use assess_complexity_async for LLM."""
        # Sync callers get heuristic; async callers get LLM
        return self._assess_heuristic(task)

    async def _assess_via_llm(self, task: str) -> CognitiveProfile:
        """Call Gemini Flash Lite with structured JSON output."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        prompt = _ROUTING_PROMPT.format(task=task[:2000])  # Cap input length

        config = types.GenerateContentConfig(
            max_output_tokens=256,
            temperature=0.0,  # Deterministic routing
            response_mime_type="application/json",
            response_schema=_ROUTING_SCHEMA,
        )

        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=config,
        )

        import json
        data = json.loads(response.text)

        profile = CognitiveProfile(
            complexity=max(0.0, min(1.0, float(data.get("complexity", 0.5)))),
            uncertainty=max(0.0, min(1.0, float(data.get("uncertainty", 0.3)))),
            tool_required=bool(data.get("tool_required", False)),
            reasoning=str(data.get("reasoning", "")),
        )
        log.info(f"LLM routing: c={profile.complexity:.2f} u={profile.uncertainty:.2f} "
                 f"tool={profile.tool_required} — {profile.reasoning}")
        return profile

    def _assess_heuristic(self, task: str) -> CognitiveProfile:
        """Fast keyword-based fallback (no LLM call).

        Calibrated so that simple factual and simple code tasks route to S1,
        moderate code generation routes to S2, and complex debug/design tasks
        route to S3.
        """
        lower = task.lower()
        words = lower.split()
        word_count = len(words)

        # --- Complexity ---
        complexity = 0.2  # base: simple factual tasks

        # Algorithmic / implementation keywords (+0.35)
        if re.search(r'\b(?:implement|build|algorithm)\b', lower):
            complexity += 0.35
        # Simple code generation keywords (+0.15) — only if no algorithmic match
        elif re.search(r'\b(?:write|create|code|function|class|method)\b', lower):
            complexity += 0.15

        # Debug / error keywords (+0.3)
        if re.search(
            r'\b(?:debug|fix|error|crash|bug|race condition|deadlock)\b',
            lower,
        ):
            complexity += 0.3

        # Design / architecture keywords (+0.2)
        if re.search(
            r'\b(?:optimize|evolve|design|architect|refactor|distributed)\b',
            lower,
        ):
            complexity += 0.2

        # Multi-step indicators (+0.1)
        if re.search(r'\b(?:then|after|first|next|finally|step)\b', lower):
            complexity += 0.1

        # Task-length scaling (word count)
        if word_count > 100:
            complexity += 0.15
        elif word_count > 50:
            complexity += 0.1
        elif word_count > 20:
            complexity += 0.05

        # --- Uncertainty ---
        uncertainty = 0.2

        if "?" in task:
            uncertainty += 0.1

        if re.search(r'\b(?:maybe|possibly|explore|investigate)\b', lower):
            uncertainty += 0.2

        # Flakiness / nondeterminism keywords (+0.15)
        if re.search(r'\b(?:intermittent|sometimes|random|flaky)\b', lower):
            uncertainty += 0.15

        # --- Tool requirement ---
        # Note: "read/write" only match with file/disk/data context to avoid
        # false positives on code-gen tasks like "write a function".
        tool_required = bool(re.search(
            r'\b(?:file|search|run|execute|compile|test|deploy|download|upload)\b',
            lower,
        )) or bool(re.search(
            r'\b(?:read|write)\s+(?:file|disk|data|csv|json|log|output)\b',
            lower,
        ))

        return CognitiveProfile(
            complexity=min(1.0, round(complexity, 4)),
            uncertainty=min(1.0, round(uncertainty, 4)),
            tool_required=tool_required,
            reasoning="heuristic",
        )


# Backward compatibility alias
MetacognitiveController = ComplexityRouter
