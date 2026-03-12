"""ModelCard — structured capability descriptor for LLM models.

Python reimplementation of sage-core/src/routing/model_card.rs (469 LOC).
Field-for-field compatible with Rust PyO3 class. Uses dataclasses + tomllib.

NOTE: Field names match Rust EXACTLY (cost_input_per_m, not cost_per_1m_input;
context_window, not max_context; latency_ttft_ms, not latency_p50).
"""
from __future__ import annotations

import enum
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


class CognitiveSystem(enum.IntEnum):
    """Kahneman-inspired cognitive modes for task routing."""
    S1 = 1  # Fast / Intuitive
    S2 = 2  # Deliberate / Tools
    S3 = 3  # Formal / Reasoning


@dataclass
class ModelCard:
    """Structured capability card for an LLM model."""

    # Identity
    id: str
    provider: str
    family: str

    # Benchmark scores (0.0–1.0)
    code_score: float = 0.0
    reasoning_score: float = 0.0
    tool_use_score: float = 0.0
    math_score: float = 0.0
    formal_z3_strength: float = 0.0

    # Cost & latency
    cost_input_per_m: float = 0.0
    cost_output_per_m: float = 0.0
    latency_ttft_ms: float = 0.0
    tokens_per_sec: float = 0.0

    # Cognitive affinities (0.0–1.0)
    s1_affinity: float = 0.5
    s2_affinity: float = 0.5
    s3_affinity: float = 0.5

    # Topology & capabilities
    recommended_topologies: list[str] = field(default_factory=list)
    supports_tools: bool = False
    supports_json_mode: bool = False
    supports_vision: bool = False
    context_window: int = 128000

    # Domain scores & safety
    domain_scores: dict[str, float] = field(default_factory=dict)
    safety_rating: float = 0.5

    def best_system(self) -> CognitiveSystem:
        """Return cognitive system with highest affinity. Ties favor S1."""
        if self.s1_affinity >= self.s2_affinity and self.s1_affinity >= self.s3_affinity:
            return CognitiveSystem.S1
        elif self.s2_affinity >= self.s3_affinity:
            return CognitiveSystem.S2
        return CognitiveSystem.S3

    def affinity_for(self, system: CognitiveSystem | int) -> float:
        """Return affinity score for a cognitive system (matches Rust name)."""
        s = int(system)
        if s == 1:
            return self.s1_affinity
        elif s == 2:
            return self.s2_affinity
        elif s == 3:
            return self.s3_affinity
        return 0.0

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost (USD) for given token counts."""
        return (
            input_tokens * self.cost_input_per_m
            + output_tokens * self.cost_output_per_m
        ) / 1_000_000

    def domain_score(self, domain: str) -> float:
        """Get domain-specific score. Returns 0.5 (neutral) if unknown.

        NOTE: Default is 0.5, NOT 0.0 — matches Rust unwrap_or(0.5).
        """
        return self.domain_scores.get(domain, 0.5)

    @classmethod
    def parse_toml(cls, toml_str: str) -> list[ModelCard]:
        """Parse ModelCards from a TOML string with [[models]] array."""
        data = tomllib.loads(toml_str)
        models = data.get("models", [])
        return [cls._from_dict(m) for m in models]

    @classmethod
    def load_from_file(cls, path: str) -> list[ModelCard]:
        """Load ModelCards from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        models = data.get("models", [])
        return [cls._from_dict(m) for m in models]

    @classmethod
    def _from_dict(cls, d: dict) -> ModelCard:
        return cls(
            id=d.get("id", ""),
            provider=d.get("provider", ""),
            family=d.get("family", ""),
            code_score=d.get("code_score", 0.0),
            reasoning_score=d.get("reasoning_score", 0.0),
            tool_use_score=d.get("tool_use_score", 0.0),
            math_score=d.get("math_score", 0.0),
            formal_z3_strength=d.get("formal_z3_strength", 0.0),
            cost_input_per_m=d.get("cost_input_per_m", 0.0),
            cost_output_per_m=d.get("cost_output_per_m", 0.0),
            latency_ttft_ms=d.get("latency_ttft_ms", 0.0),
            tokens_per_sec=d.get("tokens_per_sec", 0.0),
            s1_affinity=d.get("s1_affinity", 0.5),
            s2_affinity=d.get("s2_affinity", 0.5),
            s3_affinity=d.get("s3_affinity", 0.5),
            recommended_topologies=d.get("recommended_topologies", []),
            supports_tools=d.get("supports_tools", False),
            supports_json_mode=d.get("supports_json_mode", False),
            supports_vision=d.get("supports_vision", False),
            context_window=d.get("context_window", 128000),
            domain_scores=d.get("domain_scores", {}),
            safety_rating=d.get("safety_rating", 0.5),
        )

    def __repr__(self) -> str:
        return (
            f"ModelCard(id='{self.id}', provider='{self.provider}', "
            f"s1={self.s1_affinity:.2f}, s2={self.s2_affinity:.2f}, "
            f"s3={self.s3_affinity:.2f})"
        )
