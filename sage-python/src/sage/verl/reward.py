"""Reward function for veRL topology training.

veRL signature: compute_score(data_source, solution_str, ground_truth, extra_info) -> float

Delegates Rust scoring to existing infrastructure (DRY).
Adds edge-level credit integration for Graph-GRPO (arXiv 2603.02701).

Register in veRL config:
    custom_reward_function.path=<path>/sage/verl/reward.py
    custom_reward_function.name=compute_score
"""
from __future__ import annotations

import logging
import math

import yaml

log = logging.getLogger("verl_reward")


# ── Format scoring ───────────────────────────────────────────

def _score_format(text: str) -> float:
    """YAML format validity. Range: [-2.0, +1.0]."""
    try:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            return -1.5
        if "nodes" not in data:
            return -0.5
        nodes = data["nodes"]
        if not isinstance(nodes, list) or len(nodes) == 0:
            return -0.25
        return 1.0
    except yaml.YAMLError:
        return -2.0
    except Exception:
        return -2.0


# ── Structure scoring ────────────────────────────────────────

def _score_structure(text: str) -> float:
    """Structural quality. Range: [0.0, 1.0]."""
    try:
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or "nodes" not in data:
            return 0.0
        nodes = data.get("nodes", [])
        if not isinstance(nodes, list):
            return 0.0
        score = 0.0
        if 1 <= len(nodes) <= 10:
            score += 0.3
        if data.get("edges"):
            score += 0.2
        if all(isinstance(n, dict) and "role" in n for n in nodes):
            score += 0.3
        if data.get("reasoning"):
            score += 0.2
        return score
    except Exception:
        return 0.0


# ── Rust density scoring ─────────────────────────────────────

def _score_rust_density(text: str, extra_info: dict) -> float:
    """Rust TopologyReward + TopologyDensity. Fallback: 0.5 for valid topology."""
    try:
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or "nodes" not in data:
            return 0.0

        try:
            from sage_core import (
                TopologyReward, TopologyDensity, TopologyGraph,
                TopologyNode, TopologyEdge, PyHybridVerifier,
            )
        except ImportError:
            return 0.5 if isinstance(data.get("nodes"), list) and len(data["nodes"]) > 0 else 0.0

        nodes = data.get("nodes", [])
        difficulty = data.get("difficulty", extra_info.get("difficulty", "moderate"))
        system = {"simple": 1, "moderate": 2, "complex": 3}.get(str(difficulty).lower(), 2)

        graph = TopologyGraph("sequential")
        for nd in nodes:
            if isinstance(nd, dict):
                graph.add_node(TopologyNode(
                    role=nd.get("role", "agent"),
                    model_id=nd.get("model_tier", ""),
                    system=system,
                    prompt=nd.get("prompt", ""),
                ))

        for ed in data.get("edges", []):
            if isinstance(ed, dict):
                fi, ti = ed.get("from_idx", 0), ed.get("to_idx", 0)
                if 0 <= fi < graph.node_count() and 0 <= ti < graph.node_count():
                    graph.add_edge(fi, ti, TopologyEdge(ed.get("flow_type", "message")))

        if graph.edge_count() == 0 and graph.node_count() > 1:
            for i in range(graph.node_count() - 1):
                graph.add_edge(i, i + 1, TopologyEdge("message"))

        density = TopologyDensity()
        verifier = PyHybridVerifier()
        scorer = TopologyReward()

        d = density.compute(graph, system)
        v = verifier.verify(graph)
        structural = 1.0 if v.valid else 0.5

        reward = scorer.compute(
            execution_passed=True,
            structural_score=structural,
            density_score=d.s_complex,
            temporal_score=None,
        )
        score = reward.total

        if d.over_budget:
            n_nodes = graph.node_count()
            penalty = math.tanh(float(d.n_max - n_nodes) / float(max(d.n_max, 1)))
            score = score * max(0.0, 1.0 + penalty)

        return float(score)
    except Exception:
        return 0.0


# ── Combined reward (veRL entry point) ───────────────────────

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """Combined topology reward for veRL.

    This is registered in veRL config as:
        custom_reward_function.path=sage-python/src/sage/verl/reward.py
        custom_reward_function.name=compute_score
    """
    if extra_info is None:
        extra_info = {}

    fmt = _score_format(solution_str)
    struct = _score_structure(solution_str)
    rust = _score_rust_density(solution_str, extra_info)

    fmt_norm = (fmt + 2.0) / 3.0  # [-2.0, 1.0] -> [0.0, 1.0]
    combined = (fmt_norm + struct + rust) / 3.0
    return float(combined)


# ── Batch-level edge credit (Graph-GRPO) ─────────────────────

def compute_score_with_edge_credit(
    topologies: list[dict],
    edge_weight: float = 0.2,
) -> list[float]:
    """Batch-level reward with Graph-GRPO edge credit (arXiv 2603.02701).

    Called after collecting K topologies for the same prompt.
    Adjusts per-topology rewards by edge-level advantage.

    Args:
        topologies: list of {"yaml": str, "base_reward": float}
        edge_weight: weight of edge credit bonus (default 0.2)

    Returns:
        list of adjusted rewards (same length as input)
    """
    from sage.verl.edge_credit import compute_edge_advantages, parse_edges_from_yaml

    edge_data = []
    for topo in topologies:
        edges = parse_edges_from_yaml(topo.get("yaml", ""))
        edge_data.append({
            "edges": edges,
            "reward": topo.get("base_reward", 0.0),
        })

    advantages = compute_edge_advantages(edge_data)

    adjusted = []
    for topo, ed in zip(topologies, edge_data):
        base = topo.get("base_reward", 0.0)
        edges = ed["edges"]
        if edges and advantages:
            edge_bonus = sum(advantages.get(tuple(e), 0.0) for e in edges) / len(edges)
        else:
            edge_bonus = 0.0
        adjusted.append(base + edge_weight * edge_bonus)

    return adjusted
