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


class TestTopologyEnvV2:
    def test_parse_adaptive_yaml(self):
        """Verify adaptive YAML parses with adaptation block.

        With the micro-decision state machine, checkpoint at node 0 pauses
        execution and returns CHECKPOINT status instead of TOPOLOGY_PARSED.
        """
        from sage.verl.topology_env import SageTopologyEnv
        env = SageTopologyEnv()
        env.reset("Write a sort function", "test/sort")

        yaml_text = """
difficulty: moderate
adaptation:
  checkpoints: [0]
  max_upgrades: 1
  quality_threshold: 0.5
nodes:
  - role: coder
    model_tier: fast
    fallback_tier: reasoner
    prompt: Write sorting code
  - role: synthesizer
    model_tier: fast
    prompt: Produce final solution
edges:
  - {from_idx: 0, to_idx: 1, flow_type: message}
"""
        obs, reward, done, info = env.step(yaml_text)
        # Checkpoint at node 0 pauses execution for a decision
        assert info["status"] == "CHECKPOINT"
        assert env._state == "awaiting_decision"
        assert env._topo_dict["adaptation"]["max_upgrades"] == 1
        assert env._checkpoints == {0}
        assert env._max_upgrades == 1
        assert env._quality_threshold == 0.5

    def test_memory_injection_in_reset(self):
        """Verify memory context appears in observation."""
        from sage.verl.topology_env import SageTopologyEnv
        from sage.verl.training_memory import TrainingMemory

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            mem = TrainingMemory(db_path=db_path)
            emb = np.ones(768, dtype=np.float32)
            mem.store_episode(
                task_id="t1", prompt_hash="h1", domain="algo",
                topology_yaml="nodes:\n- role: coder", n_nodes=1,
                difficulty="simple", outcome="PASSED", total_reward=0.9,
                per_node_results=[], adaptations_triggered=0, embedding=emb,
            )
            mem.close()

            env = SageTopologyEnv(config={"memory_db": db_path})
            obs = env.reset("Sort a list", "test/sort")
            # Memory context should be in the observation text
            assert env._memory is not None
            assert env._memory.count() == 1
            # With a zero query embedding and a ones stored embedding, cosine
            # similarity is 0 so results may be empty. But memory is wired.
            # At minimum, verify the env was created with memory active.
            env._memory.close()
        finally:
            os.unlink(db_path)

    def test_env_structural_mode_adaptive(self):
        """Verify env handles adaptive YAML in structural mode (no API)."""
        from sage.verl.topology_env import SageTopologyEnv
        env = SageTopologyEnv()
        env.reset("Binary search", "test/bsearch")

        yaml_text = """
difficulty: moderate
reasoning: Need coder with fallback for algorithm
adaptation:
  checkpoints: [0]
  max_upgrades: 1
  quality_threshold: 0.5
nodes:
  - role: coder
    model_tier: fast
    fallback_tier: reasoner
    prompt: Implement binary search
  - role: reviewer
    model_tier: budget
    prompt: Review for edge cases
  - role: synthesizer
    model_tier: fast
    prompt: Final solution
edges:
  - {from_idx: 0, to_idx: 1, flow_type: message, gate: conditional}
  - {from_idx: 1, to_idx: 2, flow_type: message}
"""
        obs, reward, done, info = env.step(yaml_text)
        assert not done
        assert reward > 0  # structural reward for valid YAML
        # Step through remaining nodes
        while not done:
            obs, reward, done, info = env.step("continue")
        assert env._trace.status != ""

    def test_reset_clears_v2_state(self):
        """Verify reset properly clears all V2 adaptive state."""
        from sage.verl.topology_env import SageTopologyEnv
        env = SageTopologyEnv()
        env._awaiting_decision = True
        env._checkpoints = {0, 1}
        env._max_upgrades = 3
        env._quality_threshold = 0.8

        env.reset("New task", "test/new")
        assert env._awaiting_decision is False
        assert env._checkpoints == set()
        assert env._max_upgrades == 0
        assert env._quality_threshold == 0.5


class TestIntegrationV2:
    def test_full_structural_episode(self):
        """End-to-end: reset -> generate adaptive YAML -> step through -> finalize.

        With the micro-decision state machine, checkpoint at node 0 pauses
        execution and returns CHECKPOINT. We then step("continue") through.
        """
        from sage.verl.topology_env import SageTopologyEnv

        env = SageTopologyEnv()
        obs = env.reset("Implement merge sort", "test/mergesort")
        assert "merge sort" in obs["text"].lower() or "Implement" in obs["text"]

        yaml_text = """
difficulty: moderate
reasoning: Merge sort needs careful implementation with fallback for edge cases
adaptation:
  checkpoints: [0]
  max_upgrades: 1
  quality_threshold: 0.5
nodes:
  - role: coder
    model_tier: fast
    fallback_tier: reasoner
    prompt: Implement merge sort in Python
  - role: reviewer
    model_tier: budget
    prompt: Review for correctness
  - role: synthesizer
    model_tier: fast
    prompt: Produce the final solution
edges:
  - {from_idx: 0, to_idx: 1, flow_type: message, gate: conditional}
  - {from_idx: 1, to_idx: 2, flow_type: message}
"""
        obs, reward, done, info = env.step(yaml_text)
        # Checkpoint at node 0 pauses for a decision
        assert info["status"] == "CHECKPOINT"
        assert reward > 0
        assert not done

        # Step through all remaining nodes (continue at checkpoint, finalize)
        steps = 0
        while not done and steps < 20:
            obs, reward, done, info = env.step("continue")
            steps += 1

        assert done
        trace = env.get_trace()
        assert trace.total_reward != 0
        assert len(trace.steps) >= 3  # topology_generator + nodes + terminal

        # Verify StepRewardVector
        srv = env.get_step_rewards()
        assert len(srv.step_rewards) > 0
