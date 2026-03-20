"""Graph-GRPO edge-level credit assignment (arXiv 2603.02701).

Computes per-edge success rates across K topologies for the same prompt,
then normalizes to advantages. Provides finer-grained credit than
per-topology reward.

Usage in reward function:
    edges = parse_edges_from_yaml(yaml_text)
    edge_advs = compute_edge_advantages(group_topologies)
    credit = sum(edge_advs.get(e, 0.0) for e in edges) / max(len(edges), 1)
    final_reward = base_reward + edge_credit_weight * credit
"""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class EdgeStats:
    """Track per-edge success rates across a group of topologies."""

    _counts: dict[tuple[int, int], int] = field(default_factory=dict)
    _successes: dict[tuple[int, int], float] = field(default_factory=dict)

    def record(self, edge: tuple[int, int], reward: float) -> None:
        self._counts[edge] = self._counts.get(edge, 0) + 1
        self._successes[edge] = self._successes.get(edge, 0.0) + reward

    def success_rate(self, edge: tuple[int, int]) -> float:
        count = self._counts.get(edge, 0)
        if count == 0:
            return 0.0
        return self._successes.get(edge, 0.0) / count

    @classmethod
    def from_topologies(cls, topologies: list[dict]) -> EdgeStats:
        stats = cls()
        for topo in topologies:
            reward = topo.get("reward", 0.0)
            binary = 1.0 if reward > 0.5 else 0.0
            for edge in topo.get("edges", []):
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    stats.record((edge[0], edge[1]), binary)
        return stats

    @property
    def all_edges(self) -> list[tuple[int, int]]:
        return list(self._counts.keys())


def compute_edge_advantages(
    topologies: list[dict],
    eps: float = 1e-6,
) -> dict[tuple[int, int], float]:
    """Compute normalized edge advantages (Graph-GRPO Eq. 4-5).

    For a group of K topologies for the same prompt:
    1. S_ij = P(Success | edge(i,j) in G)  — per-edge success rate
    2. A_ij = (S_ij - mean(S)) / (std(S) + eps)  — normalized advantage

    Args:
        topologies: list of {"edges": [(i,j), ...], "reward": float}

    Returns:
        dict mapping (from_idx, to_idx) -> advantage float
    """
    stats = EdgeStats.from_topologies(topologies)
    edges = stats.all_edges
    if not edges:
        return {}

    rates = {e: stats.success_rate(e) for e in edges}
    values = list(rates.values())
    n = len(values)
    if n == 0:
        return {}

    mean_s = sum(values) / n
    var_s = sum((v - mean_s) ** 2 for v in values) / max(n, 1)
    std_s = var_s ** 0.5

    return {
        edge: (rate - mean_s) / (std_s + eps)
        for edge, rate in rates.items()
    }


def parse_edges_from_yaml(yaml_text: str) -> list[tuple[int, int]]:
    """Extract edges from a YAML topology string."""
    try:
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            return []
        edges = []
        for ed in data.get("edges", []):
            if isinstance(ed, dict):
                edges.append((ed.get("from_idx", 0), ed.get("to_idx", 0)))
        return edges
    except Exception:
        return []
