"""Tests for compute_dag_features — AdaptOrch DAG structural metrics."""
from __future__ import annotations

import pytest
from sage.pipeline_stages import DAGFeatures, compute_dag_features


# ---------------------------------------------------------------------------
# Mock DAG helpers
# ---------------------------------------------------------------------------


class MockDAG:
    """Minimal DAG stub with node_ids and successors()."""

    def __init__(self, nodes: list[str], edges: list[tuple[str, str]]) -> None:
        self._nodes = nodes
        self._succ: dict[str, list[str]] = {n: [] for n in nodes}
        for src, dst in edges:
            self._succ[src].append(dst)

    @property
    def node_ids(self) -> list[str]:
        return list(self._nodes)

    def successors(self, node_id: str) -> list[str]:
        return list(self._succ.get(node_id, []))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_dag():
    """Zero-node DAG → all metrics are 0."""
    dag = MockDAG([], [])
    features = compute_dag_features(dag)
    assert features.omega == 0
    assert features.delta == 0
    assert features.gamma == 0.0


def test_single_node_dag():
    """1-node DAG → omega=1, delta=1, gamma=0.0."""
    dag = MockDAG(["A"], [])
    features = compute_dag_features(dag)
    assert features.omega == 1
    assert features.delta == 1
    assert features.gamma == 0.0


# ---------------------------------------------------------------------------
# Linear chain: A → B → C
# ---------------------------------------------------------------------------


def test_linear_dag():
    """A→B→C: omega=1 (one per depth level), delta=3 (3 levels deep)."""
    dag = MockDAG(["A", "B", "C"], [("A", "B"), ("B", "C")])
    features = compute_dag_features(dag)
    assert features.omega == 1
    assert features.delta == 3
    # 2 edges, max_possible=3 → gamma=2/3 ≈ 0.667
    assert 0.0 < features.gamma <= 1.0


# ---------------------------------------------------------------------------
# Fully parallel: A, B, C with no edges
# ---------------------------------------------------------------------------


def test_parallel_dag():
    """A, B, C (no edges): all nodes at depth 1 → omega=3, delta=1."""
    dag = MockDAG(["A", "B", "C"], [])
    features = compute_dag_features(dag)
    assert features.omega == 3
    assert features.delta == 1
    assert features.gamma == 0.0


# ---------------------------------------------------------------------------
# Fan-out: A → B, A → C, A → D
# ---------------------------------------------------------------------------


def test_fanout_dag():
    """A→B, A→C, A→D: depth 1 has 1 node, depth 2 has 3 → omega=3."""
    dag = MockDAG(["A", "B", "C", "D"], [("A", "B"), ("A", "C"), ("A", "D")])
    features = compute_dag_features(dag)
    assert features.omega == 3   # B, C, D all at depth 2
    assert features.delta == 2   # max depth = 2
    assert features.gamma > 0.0


# ---------------------------------------------------------------------------
# Diamond: A → B, A → C, B → D, C → D
# ---------------------------------------------------------------------------


def test_diamond_dag():
    """Diamond topology: A at depth 1, B/C at depth 2, D at depth 3."""
    dag = MockDAG(
        ["A", "B", "C", "D"],
        [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
    )
    features = compute_dag_features(dag)
    assert features.delta == 3          # A→B→D or A→C→D
    assert features.omega == 2          # depth 2 has B and C
    assert 0.0 < features.gamma <= 1.0


# ---------------------------------------------------------------------------
# DAGFeatures dataclass is frozen (immutable)
# ---------------------------------------------------------------------------


def test_dag_features_frozen():
    """DAGFeatures is a frozen dataclass — assignment should raise."""
    features = DAGFeatures(omega=2, delta=4, gamma=0.5)
    with pytest.raises((AttributeError, TypeError)):
        features.omega = 99  # type: ignore[misc]


def test_dag_features_gamma_capped():
    """Gamma is capped at 1.0."""
    dag = MockDAG(["A", "B"], [("A", "B")])
    features = compute_dag_features(dag)
    assert features.gamma <= 1.0


def test_dag_features_fields():
    """DAGFeatures exposes omega, delta, gamma."""
    f = DAGFeatures(omega=3, delta=5, gamma=0.7)
    assert f.omega == 3
    assert f.delta == 5
    assert f.gamma == pytest.approx(0.7)
