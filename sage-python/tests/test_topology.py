"""Tests for the topology engine."""
import pytest
from sage.topology.engine import TopologyEngine, TopologyType, AgentNode
from sage.topology import patterns


def test_create_topology():
    engine = TopologyEngine()
    topo_id = engine.create_topology(TopologyType.VERTICAL)
    assert topo_id in engine.list_topologies()


def test_add_node():
    engine = TopologyEngine()
    topo_id = engine.create_topology(TopologyType.VERTICAL)
    node_id = engine.add_node(topo_id, name="coordinator", role="coordinator")
    topo = engine.get_topology(topo_id)
    assert topo.node_count() == 1
    assert topo.get_node(node_id).name == "coordinator"


def test_vertical_delegation():
    engine = TopologyEngine()
    topo_id = engine.create_topology(TopologyType.VERTICAL)
    parent_id = engine.add_node(topo_id, name="boss", role="coordinator")
    child_id = engine.add_node(topo_id, name="worker", role="worker")
    engine.connect(topo_id, parent_id, child_id, "delegates_to")

    topo = engine.get_topology(topo_id)
    children = topo.get_children(parent_id)
    assert len(children) == 1
    assert children[0].name == "worker"
    assert topo.get_node(child_id).parent_id == parent_id


def test_horizontal_peers():
    engine = TopologyEngine()
    topo_id = engine.create_topology(TopologyType.HORIZONTAL)
    a_id = engine.add_node(topo_id, name="explorer-a", role="explorer")
    b_id = engine.add_node(topo_id, name="explorer-b", role="explorer")
    engine.connect(topo_id, a_id, b_id, "peers_with")

    topo = engine.get_topology(topo_id)
    peers_a = topo.get_peers(a_id)
    peers_b = topo.get_peers(b_id)
    assert len(peers_a) == 1
    assert len(peers_b) == 1
    assert peers_a[0].name == "explorer-b"


def test_vertical_pattern():
    engine = TopologyEngine()
    topo_id = patterns.vertical(engine, "boss", ["coder", "tester", "reviewer"])
    topo = engine.get_topology(topo_id)
    assert topo.type == TopologyType.VERTICAL
    assert topo.node_count() == 4
    children = topo.get_children(topo.root_id)
    assert len(children) == 3


def test_horizontal_pattern():
    engine = TopologyEngine()
    topo_id = patterns.horizontal(engine, ["alpha", "beta", "gamma"])
    topo = engine.get_topology(topo_id)
    assert topo.type == TopologyType.HORIZONTAL
    assert topo.node_count() == 3
    # Each should have 2 peers
    for node in topo.all_nodes():
        assert len(topo.get_peers(node.id)) == 2


def test_mesh_pattern():
    engine = TopologyEngine()
    topo_id = patterns.mesh(engine, "coordinator", ["agent-1", "agent-2", "agent-3"])
    topo = engine.get_topology(topo_id)
    assert topo.type == TopologyType.MESH
    assert topo.node_count() == 4  # coordinator + 3 agents
    coord = topo.get_node(topo.root_id)
    assert len(coord.children) == 3


def test_nonexistent_topology():
    engine = TopologyEngine()
    with pytest.raises(ValueError):
        engine.add_node("nonexistent", name="x", role="x")


def test_all_edges():
    engine = TopologyEngine()
    topo_id = patterns.vertical(engine, "boss", ["w1", "w2"])
    topo = engine.get_topology(topo_id)
    edges = topo.all_edges()
    assert len(edges) == 2
    assert all(e.edge_type == "delegates_to" for e in edges)
