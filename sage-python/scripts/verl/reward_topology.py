"""Topology reward function for veRL training.

Combined reward: format + structure + execution (topology runner).
veRL expects: compute_score(data_source, solution_str, ground_truth, extra_info) -> float

This is the veRL equivalent of the 3 TRL reward functions:
  - format_reward (YAML validity)
  - structure_reward (node/edge quality)
  - execution_reward (TopologyRunner + sandbox)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re as _re
from typing import Any

import yaml

log = logging.getLogger("verl_reward")


# ── Format scoring (YAML validity) ──────────────────────────────

def _score_format(text: str) -> float:
    """Graduated YAML format score. Range: [-2.0, +1.0]."""
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


# ── Structure scoring ────────────────────────────────────────────

def _score_structure(text: str) -> float:
    """Structural quality score. Range: [0.0, 1.0]."""
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


# ── Execution scoring (lightweight, no API calls) ────────────────
# NOTE: Full TopologyRunner execution is expensive (~40s/eval with DeepSeek).
# For veRL training at scale on H100, we use a lightweight proxy:
#   - Rust structural verification (HybridVerifier)
#   - Rust density scoring (S_complex)
#   - No LLM execution during training (too slow for K=8 * batch_size * epochs)
# The full TopologyRunner execution is used for EVALUATION, not training.
# This matches AgentConductor: their Eq.9 reward = re(execution) + rg(density)
# where re comes from actual code execution in the environment.

def _score_execution_proxy(text: str, extra_info: dict) -> float:
    """Lightweight execution proxy using Rust infrastructure.

    Uses TopologyReward (structural + density) without LLM execution.
    For full execution scoring, use evaluation mode with TopologyRunner.
    """
    try:
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or "nodes" not in data:
            return 0.0

        # Try Rust scoring
        try:
            from sage_core import (
                TopologyReward, TopologyDensity, TopologyGraph,
                TopologyNode, TopologyEdge, PyHybridVerifier,
            )
        except ImportError:
            # No Rust available — return format-only score
            return 0.5 if isinstance(data.get("nodes"), list) and len(data["nodes"]) > 0 else 0.0

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        difficulty = data.get("difficulty", extra_info.get("difficulty", "moderate"))
        system = {"simple": 1, "moderate": 2, "complex": 3}.get(str(difficulty).lower(), 2)

        graph = TopologyGraph("sequential")
        for node_data in nodes:
            if not isinstance(node_data, dict):
                continue
            node = TopologyNode(
                role=node_data.get("role", "agent"),
                model_id=node_data.get("model_tier", ""),
                system=system,
                prompt=node_data.get("prompt", ""),
            )
            graph.add_node(node)

        for edge_data in edges:
            if not isinstance(edge_data, dict):
                continue
            from_idx = edge_data.get("from_idx", 0)
            to_idx = edge_data.get("to_idx", 0)
            flow_type = edge_data.get("flow_type", "message")
            if 0 <= from_idx < graph.node_count() and 0 <= to_idx < graph.node_count():
                graph.add_edge(from_idx, to_idx, TopologyEdge(flow_type))

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
            execution_passed=True,  # Assume pass for structural scoring
            structural_score=structural,
            density_score=d.s_complex,
            temporal_score=None,
        )

        import math
        score = reward.total
        if d.over_budget:
            n_nodes = graph.node_count()
            penalty = math.tanh((d.n_max - n_nodes) / max(d.n_max, 1))
            score = score * max(0.0, 1.0 + penalty)

        return float(score)

    except Exception:
        return 0.0


# ── Combined reward for veRL ─────────────────────────────────────

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """Combined topology reward for veRL training.

    Normalizes and sums format + structure + execution_proxy.
    This is the single entry point registered in veRL config.

    Args:
        data_source: "sage_topology" (identifies this reward function)
        solution_str: The model's generated YAML topology
        ground_truth: The reference topology YAML (from SFT data)
        extra_info: {"task_id", "difficulty", "node_count", ...}
    """
    if extra_info is None:
        extra_info = {}

    fmt = _score_format(solution_str)
    struct = _score_structure(solution_str)
    exec_proxy = _score_execution_proxy(solution_str, extra_info)

    # Normalize to [0, 1] range then combine
    fmt_norm = (fmt + 2.0) / 3.0  # [-2.0, 1.0] → [0.0, 1.0]
    struct_norm = struct             # already [0.0, 1.0]
    exec_norm = exec_proxy           # already [0.0, 1.0]

    # Equal weights (same as TRL normalize_then_sum with [1,1,1])
    combined = (fmt_norm + struct_norm + exec_norm) / 3.0

    return float(combined)
