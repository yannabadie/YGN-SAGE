"""Tests for V2 adaptive topology components."""
import os
import tempfile
import numpy as np
import pytest


class TestTrainingMemory:
    def test_store_and_query(self):
        from sage.verl.training_memory import TrainingMemory
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            mem = TrainingMemory(db_path=db_path)
            emb = np.random.randn(768).astype(np.float32)
            mem.store_episode(
                task_id="test/1", prompt_hash="abc123", domain="algorithm",
                topology_yaml="nodes:\n- role: coder", n_nodes=1,
                difficulty="simple", outcome="PASSED", total_reward=0.8,
                per_node_results=[{"role": "coder", "reward": 0.8}],
                adaptations_triggered=0, embedding=emb,
            )
            results = mem.query_similar(emb, k=1)
            assert len(results) == 1
            assert results[0]["outcome"] == "PASSED"
        finally:
            mem.close()
            os.unlink(db_path)

    def test_format_context_empty(self):
        from sage.verl.training_memory import TrainingMemory
        mem = TrainingMemory(db_path=":memory:")
        ctx = mem.format_context([])
        assert ctx == ""

    def test_format_context_with_episodes(self):
        from sage.verl.training_memory import TrainingMemory
        mem = TrainingMemory(db_path=":memory:")
        episodes = [
            {"domain": "algo", "difficulty": "moderate", "outcome": "PASSED",
             "total_reward": 0.85, "n_nodes": 3, "adaptations_triggered": 1},
        ]
        ctx = mem.format_context(episodes)
        assert "PASSED" in ctx
        assert "moderate" in ctx


class TestRewardFlow:
    def _make_trace(self, nodes, terminal_reward):
        """Helper: build a mock EpisodeTrace-like list of dicts."""
        return {
            "node_traces": nodes,
            "terminal_reward": terminal_reward,
        }

    def test_single_rollout_propagation(self):
        from sage.verl.rewardflow import RewardFlowPropagator
        prop = RewardFlowPropagator(damping=0.85, max_iters=20)

        rollouts = [
            self._make_trace(
                [{"node_idx": 0, "role": "coder", "quality": 0.8},
                 {"node_idx": 1, "role": "reviewer", "quality": 0.6},
                 {"node_idx": 2, "role": "synthesizer", "quality": 0.9}],
                terminal_reward=1.0,
            ),
        ]
        result = prop.compute(rollouts)
        assert len(result) == 1
        # Each node should get a propagated reward > 0
        for node_idx, reward in result[0].items():
            assert reward > 0.0

    def test_multiple_rollouts_differentiation(self):
        from sage.verl.rewardflow import RewardFlowPropagator
        prop = RewardFlowPropagator()

        rollouts = [
            self._make_trace(
                [{"node_idx": 0, "role": "coder", "quality": 0.9},
                 {"node_idx": 1, "role": "synthesizer", "quality": 0.8}],
                terminal_reward=1.0,
            ),
            self._make_trace(
                [{"node_idx": 0, "role": "coder", "quality": 0.2},
                 {"node_idx": 1, "role": "synthesizer", "quality": 0.3}],
                terminal_reward=0.0,
            ),
        ]
        result = prop.compute(rollouts)
        assert len(result) == 2
        # First rollout (PASSED) should have higher node rewards
        assert sum(result[0].values()) > sum(result[1].values())

    def test_empty_rollouts(self):
        from sage.verl.rewardflow import RewardFlowPropagator
        prop = RewardFlowPropagator()
        assert prop.compute([]) == []


class TestRewardV2:
    def test_resilience_score_no_adaptation(self):
        from sage.verl.reward import _score_resilience
        trace = [{"role": "coder", "was_upgraded": False, "output": "code here"}]
        assert _score_resilience(trace) == 0.0

    def test_resilience_score_upgrade_succeeded_passed(self):
        from sage.verl.reward import _score_resilience
        trace = [
            {"role": "coder", "was_upgraded": True, "output": "good code", "status": ""},
            {"role": "synthesizer", "was_upgraded": False, "output": "final", "status": "PASSED"},
        ]
        assert _score_resilience(trace) == 0.5

    def test_resilience_score_upgrade_succeeded_no_pass(self):
        from sage.verl.reward import _score_resilience
        trace = [
            {"role": "coder", "was_upgraded": True, "output": "good code", "status": ""},
            {"role": "synthesizer", "was_upgraded": False, "output": "final", "status": ""},
        ]
        assert _score_resilience(trace) == 0.3

    def test_resilience_score_upgrade_failed(self):
        from sage.verl.reward import _score_resilience
        trace = [
            {"role": "coder", "was_upgraded": True, "output": "ERROR: timeout", "status": ""},
        ]
        assert _score_resilience(trace) == 0.0

    def test_cost_efficiency_budget_model(self):
        from sage.verl.reward import _score_cost_efficiency
        # Budget model, very low cost relative to simple ref (0.01)
        assert _score_cost_efficiency(0.001, "simple") > 0.8

    def test_cost_efficiency_expensive(self):
        from sage.verl.reward import _score_cost_efficiency
        # Very expensive execution
        assert _score_cost_efficiency(0.50, "simple") < 0.2

    def test_cost_efficiency_at_budget(self):
        from sage.verl.reward import _score_cost_efficiency
        # Exactly at budget ref: tanh(1.0) ~= 0.76, so 1 - 0.76 ~= 0.24
        score = _score_cost_efficiency(0.05, "moderate")
        assert 0.2 < score < 0.3

    def test_cost_efficiency_unknown_difficulty(self):
        from sage.verl.reward import _score_cost_efficiency
        # Unknown difficulty defaults to moderate ref (0.05)
        score = _score_cost_efficiency(0.005, "unknown")
        assert score > 0.8

    def test_budget_ref_values(self):
        from sage.verl.reward import BUDGET_REF
        assert BUDGET_REF == {"simple": 0.01, "moderate": 0.05, "complex": 0.20}
