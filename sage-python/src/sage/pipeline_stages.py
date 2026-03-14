"""Pipeline stages — pure functions for the CognitiveOrchestrationPipeline.

Each function transforms a PipelineContext. No side effects, no I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Stage 0 helper: domain inference
# ---------------------------------------------------------------------------

_CODE_PATTERNS = re.compile(
    r"\b(function|def |class |import |variable|algorithm|code|implement|program|script|bug|debug|refactor|api|endpoint|database|sql|html|css|python|java|rust|typescript)\b",
    re.IGNORECASE,
)
_MATH_PATTERNS = re.compile(
    r"\b(integral|derivative|equation|matrix|vector|theorem|proof|calculus|algebra|probability|statistics|sum|product|factorial|logarithm|exponential|polynomial)\b",
    re.IGNORECASE,
)
_REASONING_PATTERNS = re.compile(
    r"\b(analyze|evaluate|compare|contrast|pros and cons|trade-?off|argue|critique|assess|reason|logic|implications|consequences|strategy|decision)\b",
    re.IGNORECASE,
)
_FORMAL_PATTERNS = re.compile(
    r"\b(verify|prove|invariant|z3|smt|specification|correctness|type.?check|formal|assertion|precondition|postcondition|safety)\b",
    re.IGNORECASE,
)
_CREATIVE_PATTERNS = re.compile(
    r"\b(write|story|poem|creative|brainstorm|imagine|design|invent|compose|narrative)\b",
    re.IGNORECASE,
)


def _infer_domain(task: str, profile: Any = None) -> str:
    """Map task text to ModelCard domain name via keyword heuristics.

    Returns one of: "code", "math", "reasoning", "formal", "creative", "general"
    No LLM call — pure regex matching.
    """
    scores = {
        "code": len(_CODE_PATTERNS.findall(task)),
        "math": len(_MATH_PATTERNS.findall(task)),
        "reasoning": len(_REASONING_PATTERNS.findall(task)),
        "formal": len(_FORMAL_PATTERNS.findall(task)),
        "creative": len(_CREATIVE_PATTERNS.findall(task)),
    }
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] == 0:
        return "general"
    return best


# ---------------------------------------------------------------------------
# Stage 1 helper: DAG structural features (AdaptOrch)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DAGFeatures:
    """Three structural metrics from AdaptOrch (arXiv 2602.16873).

    omega: parallelism width (max antichain size)
    delta: critical path depth (longest path)
    gamma: coupling density (0.0-1.0)
    """

    omega: int
    delta: int
    gamma: float


def compute_dag_features(dag: Any) -> DAGFeatures:
    """Compute AdaptOrch DAG features from a TaskDAG or compatible object.

    dag must have: node_ids (list), successors(nid) -> list
    For small DAGs (<20 nodes), uses simplified greedy approach.
    """
    node_ids = getattr(dag, 'node_ids', [])
    n = len(node_ids)
    if n == 0:
        return DAGFeatures(omega=0, delta=0, gamma=0.0)
    if n == 1:
        return DAGFeatures(omega=1, delta=1, gamma=0.0)

    # Build adjacency for analysis
    successors: dict[str, list[str]] = {}
    predecessors: dict[str, list[str]] = {nid: [] for nid in node_ids}
    edge_count = 0
    for nid in node_ids:
        succs = list(dag.successors(nid)) if hasattr(dag, 'successors') else []
        successors[nid] = succs
        edge_count += len(succs)
        for s in succs:
            if s in predecessors:
                predecessors[s].append(nid)

    # delta: critical path depth (longest path via topological BFS)
    depth: dict[str, int] = {}
    # Find roots (no predecessors)
    queue = [nid for nid in node_ids if not predecessors[nid]]
    for nid in queue:
        depth[nid] = 1
    visited: set[str] = set()
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        for s in successors.get(nid, []):
            depth[s] = max(depth.get(s, 0), depth.get(nid, 1) + 1)
            if all(p in visited for p in predecessors[s]):
                queue.append(s)
    delta = max(depth.values()) if depth else 1

    # omega: max antichain (simplified — count nodes at depth with most nodes)
    depth_counts: dict[int, int] = {}
    for d in depth.values():
        depth_counts[d] = depth_counts.get(d, 0) + 1
    omega = max(depth_counts.values()) if depth_counts else 1

    # gamma: coupling density = edges / max_possible_edges
    max_edges = n * (n - 1) / 2 if n > 1 else 1
    gamma = edge_count / max_edges

    return DAGFeatures(omega=omega, delta=delta, gamma=min(gamma, 1.0))


# ---------------------------------------------------------------------------
# Stage 2 helper: macro topology selection (AdaptOrch heuristic)
# ---------------------------------------------------------------------------

# AdaptOrch thresholds (Table 2 in paper)
_THETA_OMEGA = 0.5  # parallelism threshold (relative to node count)
_THETA_GAMMA = 0.6  # coupling threshold
_THETA_DELTA = 5    # depth threshold


def select_macro_topology(features: DAGFeatures) -> str:
    """AdaptOrch-inspired heuristic: map DAG features to topology template hint.

    Returns: "sequential", "parallel", "hierarchical", or "hybrid"
    """
    if features.omega <= 1 and features.delta <= _THETA_DELTA:
        return "sequential"
    if features.gamma >= _THETA_GAMMA:
        return "hierarchical"
    if features.omega >= 2 and features.gamma < _THETA_GAMMA:
        return "parallel"
    return "hybrid"
