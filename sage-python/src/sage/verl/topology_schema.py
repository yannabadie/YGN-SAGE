"""Shared topology YAML schema — single source of truth.

This module defines the contract between:
  - Policy model output (training)
  - reward.py (scoring)
  - pipeline.py (Path 6 parsing)
  - topology_env.py (multi-step env)
  - edge_credit.py / rewardflow.py (credit assignment)
  - Runtime TopologyGraph construction

Every consumer of topology YAML should validate against this schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


# Valid model tiers — maps to S1/S2/S3 via ModelAssigner
VALID_MODEL_TIERS = frozenset({"budget", "fast", "balanced", "reasoner", "strong"})

# Valid flow types for edges
VALID_FLOW_TYPES = frozenset({"message", "control", "state"})

# Valid difficulty levels
VALID_DIFFICULTIES = frozenset({"simple", "moderate", "complex"})


@dataclass
class TopologyNodeSchema:
    """Schema for a single node in the topology YAML."""
    role: str
    model_tier: str = ""
    prompt: str = ""
    fallback_tier: str = ""
    provider_hint: str = ""  # optional: preferred provider (e.g., "google", "deepseek")

    def is_tier_valid(self) -> bool:
        return self.model_tier.lower() in VALID_MODEL_TIERS if self.model_tier else True

    def is_provider_hint_valid(self) -> bool:
        # Any non-empty string is accepted — runtime resolves via ProviderPool
        return True


@dataclass
class TopologyEdgeSchema:
    """Schema for a single edge in the topology YAML."""
    from_idx: int
    to_idx: int
    flow_type: str = "message"

    def is_valid(self, node_count: int) -> bool:
        return (
            0 <= self.from_idx < node_count
            and 0 <= self.to_idx < node_count
            and self.from_idx != self.to_idx
        )


@dataclass
class AdaptationSchema:
    """Schema for adaptation metadata (Phase C checkpoints)."""
    checkpoints: list[int] = field(default_factory=list)
    max_upgrades: int = 0
    max_reroutes: int = 1
    quality_threshold: float = 0.5


@dataclass
class TopologySchema:
    """Complete topology YAML schema — shared contract.

    Example YAML:
        difficulty: moderate
        reasoning: "Task requires coding and review"
        nodes:
          - role: coder
            model_tier: reasoner
            prompt: "Write the solution"
            provider_hint: deepseek    # optional
          - role: reviewer
            model_tier: fast
            prompt: "Review the code"
        edges:
          - from_idx: 0
            to_idx: 1
            flow_type: message
        adaptation:
          checkpoints: [0]
          max_upgrades: 1
          quality_threshold: 0.5
    """
    difficulty: str = "moderate"
    reasoning: str = ""
    nodes: list[TopologyNodeSchema] = field(default_factory=list)
    edges: list[TopologyEdgeSchema] = field(default_factory=list)
    adaptation: AdaptationSchema = field(default_factory=AdaptationSchema)

    @classmethod
    def from_yaml(cls, text: str) -> TopologySchema | None:
        """Parse YAML text into TopologySchema. Returns None on parse failure."""
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            return None
        if not isinstance(data, dict):
            return None
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> TopologySchema | None:
        """Build from a parsed dict. Returns None if 'nodes' key is missing."""
        if "nodes" not in data:
            return None

        raw_nodes = data.get("nodes", [])
        if not isinstance(raw_nodes, list):
            return None

        nodes = []
        for n in raw_nodes:
            if isinstance(n, dict):
                nodes.append(TopologyNodeSchema(
                    role=n.get("role", "agent"),
                    model_tier=n.get("model_tier", ""),
                    prompt=n.get("prompt", ""),
                    fallback_tier=n.get("fallback_tier", ""),
                    provider_hint=n.get("provider_hint", ""),
                ))

        edges = []
        for e in data.get("edges", []):
            if isinstance(e, dict):
                edges.append(TopologyEdgeSchema(
                    from_idx=e.get("from_idx", 0),
                    to_idx=e.get("to_idx", 0),
                    flow_type=e.get("flow_type", "message"),
                ))

        adapt_raw = data.get("adaptation", {})
        adaptation = AdaptationSchema()
        if isinstance(adapt_raw, dict):
            adaptation = AdaptationSchema(
                checkpoints=adapt_raw.get("checkpoints", []),
                max_upgrades=adapt_raw.get("max_upgrades", 0),
                max_reroutes=adapt_raw.get("max_reroutes", 1),
                quality_threshold=adapt_raw.get("quality_threshold", 0.5),
            )

        return cls(
            difficulty=str(data.get("difficulty", "moderate")),
            reasoning=str(data.get("reasoning", "")),
            nodes=nodes,
            edges=edges,
            adaptation=adaptation,
        )

    def validate(self) -> list[str]:
        """Return list of validation errors. Empty = valid."""
        errors = []
        if not self.nodes:
            errors.append("No nodes defined")

        for i, node in enumerate(self.nodes):
            if not node.role:
                errors.append(f"Node {i}: missing role")
            if node.model_tier and not node.is_tier_valid():
                errors.append(f"Node {i}: invalid model_tier '{node.model_tier}' "
                              f"(valid: {', '.join(sorted(VALID_MODEL_TIERS))})")

        for i, edge in enumerate(self.edges):
            if not edge.is_valid(len(self.nodes)):
                errors.append(f"Edge {i}: invalid indices ({edge.from_idx}->{edge.to_idx}) "
                              f"for {len(self.nodes)} nodes")

        if self.difficulty and self.difficulty.lower() not in VALID_DIFFICULTIES:
            errors.append(f"Invalid difficulty '{self.difficulty}' "
                          f"(valid: {', '.join(sorted(VALID_DIFFICULTIES))})")

        for cp in self.adaptation.checkpoints:
            if cp < 0 or cp >= len(self.nodes):
                errors.append(f"Checkpoint {cp} out of range for {len(self.nodes)} nodes")

        return errors

    @property
    def has_checkpoints(self) -> bool:
        return bool(self.adaptation.checkpoints)

    @property
    def has_provider_hints(self) -> bool:
        return any(n.provider_hint for n in self.nodes)

    @property
    def tier_ratio(self) -> float:
        """Fraction of nodes with valid model_tier."""
        if not self.nodes:
            return 0.0
        valid = sum(1 for n in self.nodes if n.is_tier_valid() and n.model_tier)
        return valid / len(self.nodes)
