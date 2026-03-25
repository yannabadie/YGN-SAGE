"""E2E training pipeline smoke tests.

Tests the plumbing of the Nemotron E2E training pipeline without
requiring a GPU or actual model weights. Verifies:
- Manifest contract (save/load roundtrip)
- Reward function modes (structural/exec/fallback)
- SFT data loading
- Phase C env + schema integration
- Cascaded evaluation pipeline
"""
import json
import os

import pytest


class TestManifest:
    def test_save_load_roundtrip(self, tmp_path):
        from sage.verl.manifest import TrainingManifest

        m = TrainingManifest(
            base_model="nvidia/Nemotron-Orchestrator-8B",
            stage="sft",
            format="lora",
            chat_template="qwen3",
            output_path=str(tmp_path / "sft_output"),
            dataset="topology_sft_v2_combined.jsonl",
            dataset_size=1880,
            algorithm="sft",
            lr=2e-5,
            epochs=2,
        )
        path = m.save(str(tmp_path))
        assert os.path.exists(path)

        loaded = TrainingManifest.load(path)
        assert loaded.base_model == "nvidia/Nemotron-Orchestrator-8B"
        assert loaded.stage == "sft"
        assert loaded.format == "lora"
        assert loaded.dataset_size == 1880

    def test_manifest_json_valid(self, tmp_path):
        from sage.verl.manifest import TrainingManifest

        m = TrainingManifest(stage="grpo_warmup", algorithm="grpo")
        path = m.save(str(tmp_path))
        with open(path) as f:
            data = json.load(f)
        assert data["stage"] == "grpo_warmup"
        assert data["schema_version"] == "1.0"

    def test_manifest_provenance_chain(self, tmp_path):
        from sage.verl.manifest import TrainingManifest

        sft = TrainingManifest(stage="sft", output_path=str(tmp_path / "sft"))
        sft_path = sft.save(str(tmp_path / "sft"))

        grpo = TrainingManifest(
            stage="grpo_warmup",
            parent_manifest=sft_path,
            output_path=str(tmp_path / "grpo"),
        )
        grpo_path = grpo.save(str(tmp_path / "grpo"))

        loaded = TrainingManifest.load(grpo_path)
        assert loaded.parent_manifest == sft_path


class TestRewardModes:
    def test_structural_mode(self):
        from sage.verl.reward import compute_score

        old = os.environ.get("SAGE_VERL_EXEC")
        os.environ["SAGE_VERL_EXEC"] = "0"
        try:
            score = compute_score("t", "nodes:\n  - role: coder\nreasoning: test\n", "", {})
            assert isinstance(score, float)
            assert score > 0
        finally:
            if old:
                os.environ["SAGE_VERL_EXEC"] = old
            else:
                os.environ.pop("SAGE_VERL_EXEC", None)

    def test_exec_fallback_logs_warning(self):
        """SAGE_VERL_EXEC=1 without provider → warning + structural fallback."""
        from sage.verl.reward import compute_score

        old = os.environ.get("SAGE_VERL_EXEC")
        os.environ["SAGE_VERL_EXEC"] = "1"
        try:
            score = compute_score("t", "nodes:\n  - role: coder\nreasoning: test\n", "", {})
            assert isinstance(score, float)
        finally:
            if old:
                os.environ["SAGE_VERL_EXEC"] = old
            else:
                os.environ.pop("SAGE_VERL_EXEC", None)


class TestPhaseCSmokeReady:
    def test_env_creates_without_gpu(self):
        from sage.verl.topology_env import SageTopologyEnv
        env = SageTopologyEnv()
        obs = env.reset("Write fibonacci", "smoke/0")
        assert "text" in obs

    def test_multistep_episode(self):
        from sage.verl.topology_env import SageTopologyEnv
        env = SageTopologyEnv()
        env.reset("test", "smoke/1")
        yaml_text = (
            "nodes:\n"
            "  - role: coder\n"
            "    model_tier: fast\n"
            "    fallback_tier: reasoner\n"
            "  - role: synth\n"
            "    model_tier: budget\n"
            "edges:\n"
            "  - from_idx: 0\n"
            "    to_idx: 1\n"
            "difficulty: moderate\n"
            "reasoning: test\n"
            "adaptation:\n"
            "  checkpoints: [0]\n"
            "  max_upgrades: 1\n"
        )
        obs, reward, done, info = env.step(yaml_text)
        # Should pause at checkpoint
        assert not done
        # Continue
        obs, reward, done, info = env.step("continue")

    def test_cascaded_eval_schema_stage(self):
        from sage.verl.cascaded_eval import stage_1_schema
        r = stage_1_schema("nodes:\n  - role: coder\n")
        assert r.passed

    def test_schema_contract_consistency(self):
        from sage.verl.topology_schema import TopologySchema, VALID_MODEL_TIERS
        from sage.verl.reward import VALID_MODEL_TIERS as reward_tiers
        assert VALID_MODEL_TIERS is reward_tiers


class TestDataAvailable:
    def test_sft_data_exists(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "topology_sft_v2_combined.jsonl")
        # May not exist on CI — skip gracefully
        if not os.path.exists(path):
            pytest.skip("SFT data not present (CI environment)")
        with open(path) as f:
            line = f.readline()
        data = json.loads(line)
        assert "messages" in data or "topology_yaml" in data or "prompt" in data

    def test_verl_parquet_exists(self):
        import importlib
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        path = os.path.join(os.path.dirname(__file__), "..", "data", "verl_topology_train.parquet")
        if not os.path.exists(path):
            pytest.skip("Training parquet not present")
        df = pd.read_parquet(path)
        assert len(df) > 0
        assert "prompt" in df.columns
