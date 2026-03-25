"""Training smoke tests for CI — covers reward, env, memory.

Issue H audit fix: CI had zero coverage for the training pipeline.
These tests run without API keys, GPU, or verl-agent.
"""
import os

import pytest


class TestRewardStructural:
    def test_compute_score_structural_returns_float(self):
        from sage.verl.reward import compute_score

        old = os.environ.get("SAGE_VERL_EXEC")
        os.environ["SAGE_VERL_EXEC"] = "0"
        try:
            score = compute_score(
                "test", "nodes:\n  - role: coder\nreasoning: test\n", "", {}
            )
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
        finally:
            if old is not None:
                os.environ["SAGE_VERL_EXEC"] = old
            else:
                os.environ.pop("SAGE_VERL_EXEC", None)

    def test_score_format_valid_yaml(self):
        from sage.verl.reward import _score_format

        score = _score_format("nodes:\n  - role: coder\n    model_tier: budget\n")
        assert score == 1.0

    def test_score_format_invalid_yaml(self):
        from sage.verl.reward import _score_format

        score = _score_format("not yaml at all")
        assert score <= -0.3  # partial credit or -2.0

    def test_partial_credit_function(self):
        from sage.verl.reward import _partial_credit

        # Fully wrong text
        assert _partial_credit("hello world") == -2.0
        # Has nodes: key → partial credit
        score = _partial_credit("nodes:\n  - role: coder")
        assert score > -2.0


class TestRewardExecFallback:
    def test_exec_mode_no_provider_graceful(self):
        from sage.verl.reward import compute_score

        old = os.environ.get("SAGE_VERL_EXEC")
        os.environ["SAGE_VERL_EXEC"] = "1"
        try:
            score = compute_score(
                "test", "nodes:\n  - role: coder\nreasoning: test\n", "", {}
            )
            assert isinstance(score, float)
        finally:
            if old is not None:
                os.environ["SAGE_VERL_EXEC"] = old
            else:
                os.environ.pop("SAGE_VERL_EXEC", None)


class TestEdgeCreditBatch:
    def test_compute_score_with_edge_credit(self):
        from sage.verl.reward import compute_score_with_edge_credit

        topologies = [
            {
                "yaml": "nodes:\n  - role: coder\nedges:\n  - from_idx: 0\n    to_idx: 1\n",
                "base_reward": 0.5,
            },
            {
                "yaml": "nodes:\n  - role: reviewer\n",
                "base_reward": 0.3,
            },
        ]
        result = compute_score_with_edge_credit(topologies, edge_weight=0.2)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, float) for r in result)


class TestTopologyEnvImport:
    def test_sage_topology_env_exists(self):
        from sage.verl.topology_env import SageTopologyEnv
        assert SageTopologyEnv is not None

    def test_step_result_dataclass(self):
        from sage.verl.topology_env import StepResult
        sr = StepResult(step_idx=0, node_idx=0, role="test", output="ok",
                        reward=0.5, latency=1.0, anchor_key="a:b:c")
        assert sr.step_idx == 0
        assert sr.reward == 0.5


class TestTrainingMemoryRoundtrip:
    def test_store_and_query(self, tmp_path):
        from sage.verl.training_memory import TrainingMemory
        import numpy as np

        db_path = str(tmp_path / "test_memory.db")
        mem = TrainingMemory(db_path=db_path)

        emb = np.random.randn(768).astype(np.float32)

        # Store an episode (match actual signature)
        mem.store_episode(
            task_id="test/0",
            prompt_hash="abc123",
            domain="code",
            topology_yaml="nodes:\n  - role: coder\n",
            n_nodes=1,
            difficulty="moderate",
            outcome="PASSED",
            total_reward=0.8,
            per_node_results=[],
            adaptations_triggered=0,
            embedding=emb,
        )

        # Query similar
        results = mem.query_similar(
            query_embedding=emb,
            domain="code",
            k=5,
        )
        assert isinstance(results, list)
        assert len(results) >= 1
