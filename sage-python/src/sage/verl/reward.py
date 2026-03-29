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


# ── Utility ──────────────────────────────────────────────────

import re

def _strip_code_fence(text: str) -> str:
    """Strip markdown code fences (```yaml ... ```) from model output."""
    text = text.strip()
    # Remove opening fence
    text = re.sub(r'^```(?:yaml|yml)?\s*\n?', '', text)
    # Remove closing fence
    text = re.sub(r'\n?```\s*$', '', text)
    return text.strip()


# ── Partial credit for truncated YAML (V5 reward shaping) ────

def _partial_credit(text: str) -> float:
    """Give partial credit for YAML-like text that failed to parse.

    V5 fix: V4 had 97% reward=0 because truncated YAML scored -2.0.
    This function detects YAML structure even in truncated output and
    returns a gradient-friendly score instead of the cliff at -2.0.

    Returns: [-2.0, -0.3] — always less than valid YAML (-0.25 min)
    so the model still prefers complete YAML, but truncated attempts
    get signal to keep generating YAML-like text.
    """
    text = _strip_code_fence(text)
    score = -2.0

    # Check for YAML-like structural markers
    has_nodes_key = "nodes:" in text or "nodes :" in text
    has_role = "role:" in text or "role :" in text
    has_reasoning = "reasoning:" in text or "reasoning :" in text
    has_yaml_list = "- " in text and ("role" in text or "name" in text)

    if has_nodes_key:
        score += 1.0  # -2.0 → -1.0 (major: knows the top-level key)
    if has_role:
        score += 0.3  # knows node structure
    if has_yaml_list:
        score += 0.2  # uses YAML list syntax
    if has_reasoning:
        score += 0.2  # includes reasoning field

    return min(score, -0.3)  # cap below valid-but-empty-nodes (-0.25)


# ── Format scoring ───────────────────────────────────────────

def _score_format(text: str) -> float:
    """YAML/JSON format validity. Range: [-3.0, +1.0].

    Accepts both YAML and JSON (yaml.safe_load handles valid JSON natively).
    Falls back to json.loads for JSON with trailing commas that YAML rejects.

    V7 FIX: Strong -3.0 penalty for <think> output.
    The Qwen3/Nemotron weights strongly prefer generating <think> after
    the assistant turn. Without explicit penalty, 100% of GRPO completions
    start with <think>, all get reward=0, zero advantage variance, no learning.

    V5 FIX: Partial credit for truncated-but-YAML-like text.
    Now: if text looks like YAML (starts with 'nodes:' or contains YAML structure)
    but fails to parse, give partial credit instead of -2.0.
    """
    # V7: Strong negative for <think> output (Qwen3 weight residual)
    stripped = text.strip()
    if stripped.startswith('<think>') or stripped.startswith('</think>'):
        return -3.0  # Worse than any other failure (-2.0), creates strong gradient

    try:
        text = _strip_code_fence(text)
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            # Fallback: try json.loads for JSON with trailing commas
            import json
            data = json.loads(text)
        if not isinstance(data, dict):
            return -1.5
        if "nodes" not in data:
            return -0.5
        nodes = data["nodes"]
        if not isinstance(nodes, list) or len(nodes) == 0:
            return -0.25
        return 1.0
    except (yaml.YAMLError, Exception):
        # V5: Reward shaping for truncated YAML (reduces sparsity)
        return _partial_credit(text)


# ── Structure scoring ────────────────────────────────────────

# Use shared schema for validation constants (HyEvo hybrid: LLM + code nodes)
from sage.verl.topology_schema import VALID_MODEL_TIERS, TopologySchema

# Structural reward weights — subject to ablation (CLAUDE.md directive 2).
# These are calibrated initial values, NOT proven optimal. Each should be
# validated via ablation study once Phase A converges.
# Reference: The Conductor (2512.04388) uses binary reward (1.0/0.5/0.0)
# and efficiency emerges naturally. Consider simplifying if ablation shows
# diminishing returns from fine-grained bonuses.
_W_NODES_VALID = 0.3       # topology has 1-10 nodes
_W_EDGES_PRESENT = 0.2     # edges defined (DAG structure)
_W_ROLES_DEFINED = 0.3     # all nodes have role field
_W_REASONING = 0.2         # reasoning field present
_W_UNDERSIZED_PENALTY = 0.5  # multiplier for too-few-nodes
_W_TIER_VALID = 0.1        # valid model_tier names (multi-model dimension)
_W_CHECKPOINTS = 0.1       # adaptation metadata (Phase C)
_W_PROVIDER_HINT = 0.05    # provider hints (multi-provider dimension)
_W_HYBRID_BONUS = 0.1      # LLM + code mix (HyEvo 2603.19639: 13-19x cost reduction)
_EXPECTED_MIN_NODES = {"simple": 1, "moderate": 2, "complex": 3}


def _score_structure(text: str) -> float:
    """Structural quality of a topology YAML.

    All weights are named constants subject to ablation (see _W_* above).
    """
    try:
        text = _strip_code_fence(text)
        schema = TopologySchema.from_yaml(text)
        if schema is None or not schema.nodes:
            return 0.0

        score = 0.0
        n = len(schema.nodes)

        if 1 <= n <= 10:
            score += _W_NODES_VALID
        if schema.edges:
            score += _W_EDGES_PRESENT
        if all(node.role for node in schema.nodes):
            score += _W_ROLES_DEFINED
        if schema.reasoning:
            score += _W_REASONING

        expected_min = _EXPECTED_MIN_NODES.get(schema.difficulty.lower(), 2)
        if n < expected_min:
            score *= _W_UNDERSIZED_PENALTY

        if schema.tier_ratio > 0:
            score += _W_TIER_VALID * schema.tier_ratio
        if schema.has_checkpoints:
            score += _W_CHECKPOINTS
        if schema.has_provider_hints:
            score += _W_PROVIDER_HINT
        if schema.has_code_nodes and schema.llm_ratio > 0:
            score += _W_HYBRID_BONUS

        return min(score, 1.3)
    except Exception:
        return 0.0


# ── Rust density scoring ─────────────────────────────────────

def _score_rust_density(text: str, extra_info: dict) -> float:
    """Rust TopologyReward + TopologyDensity. Fallback: 0.5 for valid topology."""
    try:
        text = _strip_code_fence(text)
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
    """Check execution mode dynamically (not frozen at import time)."""
    return os.environ.get("SAGE_VERL_EXEC", "0") == "1"


# Reward mode tracing — records the mode used by the last compute_score call.
# Training scripts can read this to log which mode was actually used per batch.
# Thread-safe for single-threaded verl training (one call at a time).
_last_reward_mode: str = "unknown"
_last_reward_reason: str = ""


def get_last_reward_mode() -> tuple[str, str]:
    """Return (mode, reason) from the last compute_score call.

    Modes: "structural", "exec_real", "exec_fallback_no_provider",
           "exec_fallback_invalid_yaml", "exec_fallback_error"
    """
    return _last_reward_mode, _last_reward_reason


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
    global _last_reward_mode, _last_reward_reason

    if extra_info is None:
        extra_info = {}

    fmt = _score_format(solution_str)
    struct = _score_structure(solution_str)
    rust = _score_rust_density(solution_str, extra_info)

    fmt_norm = (fmt + 3.0) / 4.0  # [-3.0, 1.0] -> [0.0, 1.0]
    structural = (fmt_norm + struct + rust) / 3.0

    if not _is_exec_mode() or fmt < 0.0:
        if _is_exec_mode() and fmt < 0.0:
            _last_reward_mode = "exec_fallback_invalid_yaml"
            _last_reward_reason = f"fmt={fmt:.2f}"
        else:
            _last_reward_mode = "structural"
            _last_reward_reason = "SAGE_VERL_EXEC=0"
        return float(structural)

    # Check provider availability BEFORE attempting execution
    from sage.execution import _get_agent_provider
    provider, _ = _get_agent_provider()
    if provider is None:
        _last_reward_mode = "exec_fallback_no_provider"
        _last_reward_reason = "no API provider configured"
        log.warning("SAGE_VERL_EXEC=1 but no provider — structural fallback")
        return float(structural)

    # Execution mode: run the real multi-provider topology
    try:
        import time as _t
        _t0 = _t.time()
        result = _compute_execution_reward(solution_str, extra_info, structural)
        _elapsed = _t.time() - _t0
        _last_reward_mode = "exec_real"
        _last_reward_reason = f"provider available, execution completed in {_elapsed:.1f}s"
        log.info("EXEC_REAL: score=%.4f, time=%.1fs", result, _elapsed)
        return result
    except Exception as exc:
        _last_reward_mode = "exec_fallback_error"
        _last_reward_reason = str(exc)[:200]
        log.error("EXEC_FALLBACK_ERROR: %s: %s", type(exc).__name__, str(exc)[:200])
        import traceback
        log.error("EXEC_TRACEBACK: %s", traceback.format_exc()[-500:])
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
