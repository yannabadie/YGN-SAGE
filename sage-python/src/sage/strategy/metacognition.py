"""Cognitive routing types — CognitiveProfile and RoutingDecision.

These dataclasses are the shared vocabulary between all routing components
(AdaptiveRouter, kNN, SystemRouter). The heuristic ComplexityRouter that
previously lived here was removed — kNN (100% GT) is the primary router
and AdaptiveRouter is the unified interface.
"""
from __future__ import annotations

from dataclasses import dataclass


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
