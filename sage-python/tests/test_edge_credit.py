"""Tests for Graph-GRPO edge-level credit assignment (arXiv 2603.02701)."""
import pytest
from sage.verl.edge_credit import compute_edge_advantages, EdgeStats, parse_edges_from_yaml


class TestEdgeStats:
    def test_success_rate_basic(self):
        stats = EdgeStats()
        stats.record((0, 1), 1.0)
        stats.record((0, 1), 1.0)
        stats.record((0, 1), 0.0)
        assert stats.success_rate((0, 1)) == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_success_rate_unknown_edge(self):
        stats = EdgeStats()
        assert stats.success_rate((9, 9)) == 0.0

    def test_from_topologies(self):
        topologies = [
            {"edges": [(0, 1), (1, 2)], "reward": 1.5},  # PASSED → binary 1.0
            {"edges": [(0, 1), (1, 2)], "reward": 1.5},  # PASSED → binary 1.0
            {"edges": [(0, 1), (0, 2)], "reward": 0.0},  # CRASH → binary 0.0
        ]
        stats = EdgeStats.from_topologies(topologies)
        # Edge (1,2) in 2 passing topos
        assert stats.success_rate((1, 2)) == pytest.approx(1.0, abs=0.01)
        # Edge (0,1) in 2 passing + 1 failing = 2/3
        assert stats.success_rate((0, 1)) == pytest.approx(2.0 / 3.0, abs=0.01)
        # Edge (0,2) in 1 failing = 0/1
        assert stats.success_rate((0, 2)) == pytest.approx(0.0, abs=0.01)

    def test_all_edges(self):
        stats = EdgeStats()
        stats.record((0, 1), 1.0)
        stats.record((1, 2), 0.0)
        assert set(stats.all_edges) == {(0, 1), (1, 2)}


class TestComputeEdgeAdvantages:
    def test_passing_edge_positive_advantage(self):
        topologies = [
            {"edges": [(0, 1), (1, 2)], "reward": 1.5},
            {"edges": [(0, 1)], "reward": 0.0},
            {"edges": [(0, 1), (1, 2), (2, 3)], "reward": 1.5},
        ]
        advantages = compute_edge_advantages(topologies)
        # Edge (1,2) in 2 passing, 0 failing → high success → positive advantage
        assert advantages[(1, 2)] > 0.0
        # Edge (0,1) in 2 passing + 1 failing → lower success rate
        assert advantages[(0, 1)] < advantages[(1, 2)]

    def test_normalization(self):
        """Advantages should be approximately centered."""
        topologies = [
            {"edges": [(0, 1), (1, 2)], "reward": 1.5},
            {"edges": [(0, 1)], "reward": 0.0},
            {"edges": [(0, 1), (1, 2), (2, 3)], "reward": 1.0},
        ]
        advantages = compute_edge_advantages(topologies)
        mean_adv = sum(advantages.values()) / len(advantages)
        assert abs(mean_adv) < 0.5

    def test_empty_edges(self):
        topologies = [{"edges": [], "reward": 1.0}]
        advantages = compute_edge_advantages(topologies)
        assert len(advantages) == 0

    def test_empty_list(self):
        assert compute_edge_advantages([]) == {}


class TestParseEdgesFromYaml:
    def test_parse_edges(self):
        yaml_text = "nodes:\n- role: a\n- role: b\nedges:\n- from_idx: 0\n  to_idx: 1\n  flow_type: message"
        edges = parse_edges_from_yaml(yaml_text)
        assert edges == [(0, 1)]

    def test_no_edges(self):
        assert parse_edges_from_yaml("nodes:\n- role: coder") == []

    def test_invalid_yaml(self):
        assert parse_edges_from_yaml("{{broken") == []

    def test_multiple_edges(self):
        yaml_text = (
            "nodes:\n- role: a\n- role: b\n- role: c\n"
            "edges:\n- from_idx: 0\n  to_idx: 1\n- from_idx: 1\n  to_idx: 2\n"
        )
        edges = parse_edges_from_yaml(yaml_text)
        assert edges == [(0, 1), (1, 2)]
