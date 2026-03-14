"""Complexity-based S1/S2/S3 router with self-braking.

Routes tasks to three tiers based on heuristic or LLM-based complexity assessment:
  S1 (fast): Gemini Flash Lite, no validation (~$0.0005/call, <1s).
  S2 (algorithmic): Gemini Flash/Pro, empirical validation (~$0.0015/1K).
  S3 (formal): Codex/Reasoner, Z3 PRM formal verification (~$0.03/1K).

Supports injected LLM provider for vendor-agnostic assessment.
Falls back to Google Gemini if GOOGLE_API_KEY is set, then to heuristic.
"""
from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Any

from sage.constants import (
    S1_COMPLEXITY_CEIL,
    S1_UNCERTAINTY_CEIL,
    S3_COMPLEXITY_FLOOR,
    S3_UNCERTAINTY_FLOOR,
    S2_REASONER_COMPLEXITY_FLOOR,
    S3_CODEX_COMPLEXITY_FLOOR,
    BRAKE_WINDOW,
    BRAKE_ENTROPY_THRESHOLD,
    BRAKE_HISTORY_MAXLEN,
    HEURISTIC_COMPLEXITY_DENOM,
    HEURISTIC_QUESTION_UNCERTAINTY,
    HEURISTIC_DEFAULT_UNCERTAINTY,
    MAX_TOKENS_S1,
    MAX_TOKENS_S2,
    MAX_TOKENS_S3,
)

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
        s1_complexity_ceil: float = S1_COMPLEXITY_CEIL,
        s1_uncertainty_ceil: float = S1_UNCERTAINTY_CEIL,
        s3_complexity_floor: float = S3_COMPLEXITY_FLOOR,
        s3_uncertainty_floor: float = S3_UNCERTAINTY_FLOOR,
        brake_window: int = BRAKE_WINDOW,
        brake_entropy_threshold: float = BRAKE_ENTROPY_THRESHOLD,
        llm_provider: Any = None,
    ):
        self.s1_complexity_ceil = s1_complexity_ceil
        self.s1_uncertainty_ceil = s1_uncertainty_ceil
        self.s3_complexity_floor = s3_complexity_floor
        self.s3_uncertainty_floor = s3_uncertainty_floor
        self.brake_window = brake_window
        self.brake_entropy_threshold = brake_entropy_threshold
        self._entropy_history: deque[float] = deque(maxlen=BRAKE_HISTORY_MAXLEN)
        self._llm_available: bool | None = None
        self._llm_provider = llm_provider

    def route(self, profile: CognitiveProfile) -> RoutingDecision:
        """Decide which cognitive system to engage (S1/S2/S3)."""
        c, u = profile.complexity, profile.uncertainty

        # System 3: high complexity OR high uncertainty
        if c > self.s3_complexity_floor or u > self.s3_uncertainty_floor:
            tier = "codex" if c > S3_CODEX_COMPLEXITY_FLOOR else "reasoner"
            return RoutingDecision(
                system=3, llm_tier=tier,
                max_tokens=MAX_TOKENS_S3, use_z3=True, validation_level=3,
            )

        # System 1: low complexity AND low uncertainty AND no tools
        if (c <= self.s1_complexity_ceil
                and u <= self.s1_uncertainty_ceil
                and not profile.tool_required):
            return RoutingDecision(
                system=1, llm_tier="fast",
                max_tokens=MAX_TOKENS_S1, use_z3=False, validation_level=1,
            )

        # System 2: everything in between
        tier = "reasoner" if c > S2_REASONER_COMPLEXITY_FLOOR else "mutator"
        return RoutingDecision(
            system=2, llm_tier=tier,
            max_tokens=MAX_TOKENS_S2, use_z3=False, validation_level=2,
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
        """LLM-based task assessment. Uses injected provider if available,
        else tries Google Gemini, else falls back to heuristic."""
        if self._llm_provider is not None:
            try:
                return await self._assess_via_provider(task)
            except Exception as e:
                log.warning("LLM routing via provider failed (%s), falling back", e)

        if self._llm_available is None:
            self._llm_available = bool(os.environ.get("GOOGLE_API_KEY"))

        if self._llm_available:
            try:
                return await self._assess_via_llm(task)
            except Exception as e:
                log.warning("LLM routing failed (%s), falling back to heuristic", e)

        return self._assess_heuristic(task)

    def assess_complexity(self, task: str) -> CognitiveProfile:
        """Synchronous assessment (heuristic only). Use assess_complexity_async for LLM."""
        # Sync callers get heuristic; async callers get LLM
        return self._assess_heuristic(task)

    async def _assess_via_provider(self, task: str) -> CognitiveProfile:
        """Assess complexity using the injected LLM provider."""
        from sage.llm.base import Message, Role, LLMConfig
        import json

        prompt = _ROUTING_PROMPT.format(task=task[:2000])
        response = await self._llm_provider.generate(
            messages=[Message(role=Role.USER, content=prompt)],
            config=LLMConfig(provider="auto", model="auto", temperature=0.0, max_tokens=256),
        )
        data = json.loads(response.content)
        profile = CognitiveProfile(
            complexity=max(0.0, min(1.0, float(data.get("complexity", 0.5)))),
            uncertainty=max(0.0, min(1.0, float(data.get("uncertainty", 0.3)))),
            tool_required=bool(data.get("tool_required", False)),
            reasoning=str(data.get("reasoning", "")),
        )
        log.info("Provider routing: c=%.2f u=%.2f tool=%s — %s",
                 profile.complexity, profile.uncertainty,
                 profile.tool_required, profile.reasoning)
        return profile

    async def _assess_via_llm(self, task: str) -> CognitiveProfile:
        """Call Gemini Flash Lite with structured JSON output (legacy fallback)."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        from sage.llm._ssl import patch_genai_ssl
        patch_genai_ssl(client)
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
        """Degraded keyword-count fallback (no regex).

        Used only when ONNX model and kNN are both unavailable.
        Returns complexity estimate in [0.0, 1.0].
        """
        import warnings
        warnings.warn(
            "Using degraded keyword-count heuristic (52% accuracy). "
            "Install sage_core[onnx] or build kNN exemplars for accurate routing.",
            UserWarning,
            stacklevel=2,
        )
        words = task.lower().split()
        complex_kw = {"implement", "feature", "module", "component", "service",
                      "algorithm", "optimize", "distributed", "concurrent",
                      "debug", "fix", "race", "deadlock", "proof", "verify", "formal"}
        hits = sum(1 for w in words if w in complex_kw)
        return CognitiveProfile(
            complexity=min(hits / HEURISTIC_COMPLEXITY_DENOM, 1.0),
            uncertainty=HEURISTIC_QUESTION_UNCERTAINTY if "?" in task else HEURISTIC_DEFAULT_UNCERTAINTY,
            tool_required=False,
            reasoning="degraded_heuristic",
        )


# Backward compatibility alias
MetacognitiveController = ComplexityRouter
