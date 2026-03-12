"""Tests for SOTA Phase 3 DX parity features."""
import pytest
from sage.checkpoint import Checkpoint
from sage.topology.py_graph import PyTopologyGraph, PyTopologyNode


class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        cp = Checkpoint(db_path=tmp_path / "test.db", session_id="test1")
        cp.save("THINK", 5, {"key": "value"})
        result = cp.load("THINK")
        assert result is not None
        step, state = result
        assert step == 5
        assert state["key"] == "value"
        cp.close()

    def test_load_missing_returns_none(self, tmp_path):
        cp = Checkpoint(db_path=tmp_path / "test.db", session_id="test1")
        assert cp.load("NONEXISTENT") is None
        cp.close()

    def test_clear_phase(self, tmp_path):
        cp = Checkpoint(db_path=tmp_path / "test.db", session_id="test1")
        cp.save("THINK", 1, {})
        cp.save("ACT", 2, {})
        cp.clear("THINK")
        assert cp.load("THINK") is None
        assert cp.load("ACT") is not None
        cp.close()

    def test_clear_all(self, tmp_path):
        cp = Checkpoint(db_path=tmp_path / "test.db", session_id="test1")
        cp.save("THINK", 1, {})
        cp.save("ACT", 2, {})
        cp.clear()
        assert cp.load("THINK") is None
        assert cp.load("ACT") is None
        cp.close()


class TestPyTopologyGraph:
    def test_add_node(self):
        g = PyTopologyGraph()
        idx = g.add_node(role="analyzer", model_id="gpt-4")
        assert idx == 0
        assert g.node_count() == 1

    def test_add_edge(self):
        g = PyTopologyGraph()
        g.add_node(role="a")
        g.add_node(role="b")
        g.add_edge(0, 1)
        assert g.edge_count() == 1

    def test_get_node(self):
        g = PyTopologyGraph()
        g.add_node(role="synthesizer", system=2)
        node = g.get_node(0)
        assert node.role == "synthesizer"
        assert node.system == 2

    def test_topological_sort(self):
        g = PyTopologyGraph()
        g.add_node(role="a")
        g.add_node(role="b")
        g.add_node(role="c")
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        order = g.topological_sort()
        assert order == [0, 1, 2]

    def test_is_acyclic_true(self):
        g = PyTopologyGraph()
        g.add_node()
        g.add_node()
        g.add_edge(0, 1)
        assert g.is_acyclic() is True

    def test_is_acyclic_false(self):
        g = PyTopologyGraph()
        g.add_node()
        g.add_node()
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        assert g.is_acyclic() is False

    def test_node_ids(self):
        g = PyTopologyGraph()
        g.add_node()
        g.add_node()
        g.add_node()
        assert g.node_ids() == [0, 1, 2]


class TestEpisodicMemoryScoping:
    """Test that episodic memory supports agent_id scoping."""

    def test_episodic_accepts_agent_id(self):
        from sage.memory.episodic import EpisodicMemory
        # Should accept agent_id without error
        em = EpisodicMemory(agent_id="agent-1")
        assert em._agent_id == "agent-1"

    def test_episodic_default_no_agent_id(self):
        from sage.memory.episodic import EpisodicMemory
        em = EpisodicMemory()
        assert em._agent_id is None
