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


# ── Resilience scoring ──────────────────────────────────────

def _score_resilience(trace: list[dict]) -> float:
    """Bonus for topologies that survived adaptation.

    0.0 — no adaptation triggered
    0.3 — adaptation triggered and succeeded
    0.5 — adaptation triggered, succeeded, and terminal PASSED
    """
    adaptation_triggered = any(t.get("was_upgraded", False) for t in trace)
    if not adaptation_triggered:
        return 0.0

    adaptation_succeeded = any(
        t.get("was_upgraded", False) and not str(t.get("output", "")).startswith("ERROR")
        for t in trace
    )
    final_passed = trace[-1].get("status") == "PASSED" if trace else False

    if adaptation_succeeded and final_passed:
        return 0.5
    elif adaptation_succeeded:
        return 0.3
    return 0.0


# ── Cost efficiency scoring (inspired by CARD 2603.01089) ───

BUDGET_REF = {"simple": 0.01, "moderate": 0.05, "complex": 0.20}


def _score_cost_efficiency(total_cost: float, difficulty: str) -> float:
    """CARD-style price penalty. Range: [0.0, 1.0].

    R_cost = 1.0 - tanh(cost / budget_ref[difficulty])
    """
    ref = BUDGET_REF.get(difficulty, 0.05)
    return 1.0 - math.tanh(total_cost / ref)


# ── Combined reward (veRL entry point) ───────────────────────

# Mode selection: structural (fast, $0) or execution (real multi-provider, ~$0.003/call)
# Set SAGE_VERL_EXEC=1 to enable execution mode with real topology execution.
# In execution mode, evaluate_topology() from execution_reward.py is called,
# which uses TopologyRunner + ProviderPool to execute each node with the
# provider assigned by ModelAssigner (multi-provider: Google, DeepSeek, OpenAI, etc.)
import os


def _is_exec_mode() -> bool:
    """Check execution mode dynamically (not frozen at import time).

    train_topology.sh sets SAGE_VERL_EXEC=0 for Phase A, then SAGE_VERL_EXEC=1
    for Phase B. Each phase is a SEPARATE python process, so module re-import
    happens. But if somehow the module is cached, this function still reads
    the current env var.
    """
    return os.environ.get("SAGE_VERL_EXEC", "0") == "1"


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """Combined topology reward for veRL.

    Two modes controlled by SAGE_VERL_EXEC env var:

    SAGE_VERL_EXEC=0 (default): Structural only — YAML format + structure + Rust density.
        Fast, no API calls. Use for early epochs to learn format.

    SAGE_VERL_EXEC=1: Full execution via TopologyRunner + ProviderPool.
        Each node is executed by its assigned provider (multi-provider).
        model_tier in YAML (reasoner/fast/budget) is resolved to real models
        via ModelAssigner → ProviderPool.resolve() → actual LLM API calls.
        This teaches the model that model assignment MATTERS — putting a
        reasoner on the planner and a fast on the reviewer produces different
        results than all-budget.

    Register in veRL config:
        custom_reward_function.path=sage-python/src/sage/verl/reward.py
        custom_reward_function.name=compute_score
    """
    if extra_info is None:
        extra_info = {}

    fmt = _score_format(solution_str)
    struct = _score_structure(solution_str)
    rust = _score_rust_density(solution_str, extra_info)

    fmt_norm = (fmt + 2.0) / 3.0  # [-2.0, 1.0] -> [0.0, 1.0]
    structural = (fmt_norm + struct + rust) / 3.0

    if not _is_exec_mode() or fmt < 0.0:
        # Structural only (invalid YAML can't be executed)
        return float(structural)

    # Check provider availability BEFORE attempting execution
    from sage.execution import _get_agent_provider
    provider, _ = _get_agent_provider()
    if provider is None:
        return float(structural)

    # Execution mode: run the real multi-provider topology
    try:
        return _compute_execution_reward(solution_str, extra_info, structural)
    except Exception as exc:
        log.warning("Execution reward failed, falling back to structural: %s", exc)
        return float(structural)


def _compute_execution_reward(
    solution_str: str, extra_info: dict, structural_score: float,
) -> float:
    """Execute topology via the real SAGE pipeline with multi-provider support.

    Uses evaluate_topology() from execution_reward.py which:
    1. Parses YAML → TopologyGraph (Rust)
    2. Executes via TopologyRunner with per-node ProviderPool resolution
    3. Extracts code from final node output
    4. Tests code in sandbox
    5. Returns graduated reward (PASSED=1.5, WRONG_ANSWER=1.0, etc.)
    6. Combines with Rust density scoring (AgentConductor Eq.9)
    """
    import asyncio
    from sage.execution import evaluate_topology

    try:
        topo = yaml.safe_load(solution_str)
    except Exception:
        return float(structural_score)

    if not isinstance(topo, dict) or "nodes" not in topo:
        return float(structural_score)

    # Inject task_id for test case matching
    topo["_task_id"] = extra_info.get("task_id", "")
    task_prompt = extra_info.get("prompt", "")
    if not task_prompt:
        # Try to reconstruct from the veRL data
        task_prompt = extra_info.get("question", str(topo.get("reasoning", "")))

    semaphore = asyncio.Semaphore(8)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                exec_score = pool.submit(
                    lambda: asyncio.run(evaluate_topology(task_prompt, topo, semaphore))
                ).result(timeout=120)
        else:
            exec_score = asyncio.run(evaluate_topology(task_prompt, topo, semaphore))
    except Exception:
        return float(structural_score)

    # Combine: 30% structural + 70% execution (execution dominates)
    combined = 0.3 * structural_score + 0.7 * exec_score
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
