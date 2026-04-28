"""R5.1 edge-binding contract test (cgpro 2026-04-28 cycle 2 reassess).

Pins the R6 prerequisite: Python code accessing topology edges MUST use
`TopologyGraph.get_edges()` (Python-visible, returns tuples of
`(src_idx, dst_idx, edge_type_str)`) — NOT `edges_of_type()` which is
Rust-only and unavailable to PyO3 callers.

R6 StateCore v0 will partition incoming edges into Control / Message /
State channels. This test prevents accidental drift toward the Rust-only
helper.
"""
from __future__ import annotations

import pytest

# Optional skip if sage_core isn't installed (e.g. when wasm wheel is missing).
sage_core = pytest.importorskip("sage_core")

from sage_core import TopologyEdge, TopologyGraph, TopologyNode  # noqa: E402


def _make_graph_with_three_edge_types() -> TopologyGraph:
    """Build a 2-node graph with one state, one message, one control edge."""
    graph = TopologyGraph("sequential")
    graph.add_node(TopologyNode("a", "", 1, [], 0, 5.0, 60.0))
    graph.add_node(TopologyNode("b", "", 1, [], 0, 5.0, 60.0))
    graph.add_edge(0, 1, TopologyEdge("state"))
    graph.add_edge(0, 1, TopologyEdge("message"))
    graph.add_edge(1, 0, TopologyEdge("control"))
    return graph


def test_get_edges_returns_python_visible_tuples_with_edge_type() -> None:
    """get_edges() must be the canonical Python-visible API for edge typing.

    Returns iterable of (src_idx, dst_idx, edge_type_str). edge_type_str is
    one of "control" / "message" / "state". Stable order = insertion order.
    """
    graph = _make_graph_with_three_edge_types()
    edges = list(graph.get_edges())

    # Three edges, each tuple has exactly 3 elements
    assert len(edges) == 3
    assert all(len(e) == 3 for e in edges)

    # Edge types preserved as strings
    types_in_order = [e[2] for e in edges]
    assert types_in_order == ["state", "message", "control"]

    # (src, dst) topology preserved
    assert (edges[0][0], edges[0][1]) == (0, 1)  # state
    assert (edges[1][0], edges[1][1]) == (0, 1)  # message
    assert (edges[2][0], edges[2][1]) == (1, 0)  # control


def test_edges_of_type_is_NOT_python_visible() -> None:
    """edges_of_type() is a pure-Rust method on TopologyGraph (not exposed
    via PyO3). R6 implementation MUST NOT call it from Python.

    This test pins the binding contract. If a future PyO3 wrapper is
    added, update R6 callers AND this test together — don't silently
    introduce a second canonical edge-typing API.
    """
    graph = _make_graph_with_three_edge_types()
    assert not hasattr(graph, "edges_of_type"), (
        "edges_of_type() became Python-visible. R6 callers using "
        "graph.get_edges() must be reviewed; pick ONE canonical API."
    )


def test_get_predecessors_returns_indices_only_no_edge_type() -> None:
    """get_predecessors(node_idx) returns predecessor indices but does NOT
    expose edge types. R6 must NOT rely on it for channel partitioning;
    it must use get_edges() and partition explicitly.
    """
    graph = _make_graph_with_three_edge_types()
    predecessors_of_b = graph.get_predecessors(1)
    # Node 0 has 2 outgoing edges to node 1 (state + message). The
    # predecessors list does NOT deduplicate; the same source appears
    # once per incoming edge — and crucially, edge_type info is
    # discarded entirely.
    assert sorted(predecessors_of_b) == [0, 0]

    # The contract: get_predecessors() does NOT carry edge_type info,
    # and may include duplicates per-edge. R6 channel partitioning must
    # come from get_edges(), not from this.


def test_edge_type_strings_are_lowercase_and_stable() -> None:
    """Edge type strings must be lowercase, stable across roundtrip via
    add_edge / get_edges. R6 channel-partition logic compares against the
    canonical lowercase form.
    """
    graph = _make_graph_with_three_edge_types()
    edges = list(graph.get_edges())
    types = {e[2] for e in edges}
    assert types == {"control", "message", "state"}
    for t in types:
        assert t == t.lower(), f"edge type {t!r} is not canonical lowercase"
