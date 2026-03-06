"""Tests for TaskDAG builder and topological scheduler."""
from __future__ import annotations

import pytest
from sage.contracts.task_node import TaskNode, IOSchema
from sage.contracts.dag import TaskDAG, CycleError


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_add_node():
    dag = TaskDAG()
    node = TaskNode(node_id="a", description="step A")
    dag.add_node(node)
    assert dag.get_node("a") is node


def test_add_duplicate_node_raises():
    dag = TaskDAG()
    node = TaskNode(node_id="a", description="step A")
    dag.add_node(node)
    with pytest.raises(ValueError, match="already exists"):
        dag.add_node(node)


def test_add_edge():
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A"))
    dag.add_node(TaskNode(node_id="b", description="B"))
    dag.add_edge("a", "b")
    assert "b" in dag.successors("a")
    assert "a" in dag.predecessors("b")


def test_add_edge_unknown_node_raises():
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A"))
    with pytest.raises(KeyError):
        dag.add_edge("a", "z")


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

def test_topo_sort_linear():
    """A -> B -> C should sort to [A, B, C]."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A"))
    dag.add_node(TaskNode(node_id="b", description="B"))
    dag.add_node(TaskNode(node_id="c", description="C"))
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")
    order = dag.topological_sort()
    assert order == ["a", "b", "c"]


def test_topo_sort_diamond():
    """Diamond: A -> B, A -> C, B -> D, C -> D.
    Valid orders: A before B,C; B,C before D."""
    dag = TaskDAG()
    for nid in ["a", "b", "c", "d"]:
        dag.add_node(TaskNode(node_id=nid, description=nid.upper()))
    dag.add_edge("a", "b")
    dag.add_edge("a", "c")
    dag.add_edge("b", "d")
    dag.add_edge("c", "d")
    order = dag.topological_sort()
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


def test_topo_sort_independent_nodes():
    """Independent nodes should all appear (order unspecified)."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="x", description="X"))
    dag.add_node(TaskNode(node_id="y", description="Y"))
    order = dag.topological_sort()
    assert set(order) == {"x", "y"}


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

def test_cycle_detection():
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A"))
    dag.add_node(TaskNode(node_id="b", description="B"))
    dag.add_edge("a", "b")
    dag.add_edge("b", "a")
    with pytest.raises(CycleError):
        dag.topological_sort()


def test_self_loop_detection():
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A"))
    dag.add_edge("a", "a")
    with pytest.raises(CycleError):
        dag.topological_sort()


# ---------------------------------------------------------------------------
# Dependency validation
# ---------------------------------------------------------------------------

def test_io_compatibility_valid():
    """B's input should be subset of A's output for edge A->B."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="A",
        output_schema=IOSchema(fields={"summary": "string", "score": "float"}),
    ))
    dag.add_node(TaskNode(
        node_id="b", description="B",
        input_schema=IOSchema(fields={"summary": "string"}),
    ))
    dag.add_edge("a", "b")
    issues = dag.validate_io_compatibility()
    assert issues == []


def test_io_compatibility_missing_field():
    """B needs 'summary' but A doesn't produce it."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="A",
        output_schema=IOSchema(fields={"score": "float"}),
    ))
    dag.add_node(TaskNode(
        node_id="b", description="B",
        input_schema=IOSchema(fields={"summary": "string"}),
    ))
    dag.add_edge("a", "b")
    issues = dag.validate_io_compatibility()
    assert len(issues) == 1
    assert "summary" in issues[0]


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------

def test_ready_nodes_returns_roots():
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A"))
    dag.add_node(TaskNode(node_id="b", description="B"))
    dag.add_edge("a", "b")
    ready = dag.ready_nodes(completed=set())
    assert ready == ["a"]


def test_ready_nodes_after_completion():
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A"))
    dag.add_node(TaskNode(node_id="b", description="B"))
    dag.add_edge("a", "b")
    ready = dag.ready_nodes(completed={"a"})
    assert ready == ["b"]


def test_ready_nodes_all_completed():
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A"))
    ready = dag.ready_nodes(completed={"a"})
    assert ready == []
