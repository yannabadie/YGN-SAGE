"""Tests for the verl-agent env_package wrapper around SageTopologyEnv.

Verifies that SageTopologyVerlEnv provides the correct interface for
verl-agent integration: observation format, step mechanics, anchor keys,
projection, and StepRewardVector retrieval.
"""
import pytest

from sage.verl.env_package import SageTopologyVerlEnv, sage_topology_projection, build_sage_topology_envs
from sage.verl.env_package.envs import SageTopologyVerlEnv as EnvsImport
from sage.verl.env_package.projection import sage_topology_projection as ProjImport
from sage.verl.step_reward import StepRewardVector


# ---------------------------------------------------------------------------
# Sample YAML topologies for tests
# ---------------------------------------------------------------------------

SIMPLE_YAML = """\
nodes:
  - role: coder
    prompt: Write code
  - role: reviewer
    prompt: Review code
edges:
  - from_idx: 0
    to_idx: 1
reasoning: Code then review
difficulty: simple"""

CHECKPOINT_YAML = """\
difficulty: moderate
adaptation:
  checkpoints: [0]
  max_upgrades: 1
  quality_threshold: 0.5
nodes:
  - role: coder
    model_tier: fast
    fallback_tier: reasoner
    prompt: Write sort
  - role: synthesizer
    model_tier: fast
    prompt: Produce final
edges:
  - from_idx: 0
    to_idx: 1
    flow_type: message
reasoning: Sort with checkpoint"""

INVALID_YAML = "{{not yaml at all}}"


# ---------------------------------------------------------------------------
# Module-level imports and factory
# ---------------------------------------------------------------------------

class TestModuleImports:
    def test_import_from_package(self):
        """env_package __init__ re-exports correctly."""
        assert SageTopologyVerlEnv is EnvsImport
        assert sage_topology_projection is ProjImport

    def test_build_factory(self):
        env = build_sage_topology_envs({"n_envs": 2})
        assert isinstance(env, SageTopologyVerlEnv)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_proper_observation_format(self):
        """Each observation must have text, image, and anchor keys."""
        env = SageTopologyVerlEnv()
        obs = env.reset(["Write a sort function", "Write hello world"])

        assert len(obs) == 2
        for o in obs:
            assert "text" in o, "Observation missing 'text' key"
            assert "image" in o, "Observation missing 'image' key"
            assert "anchor" in o, "Observation missing 'anchor' key"
            assert isinstance(o["text"], str)
            assert isinstance(o["anchor"], str)

    def test_reset_with_task_ids(self):
        env = SageTopologyVerlEnv()
        obs = env.reset(
            ["task 1", "task 2"],
            task_ids=["t/1", "t/2"],
        )
        assert len(obs) == 2
        # Anchor should start with topology_generator
        for o in obs:
            assert o["anchor"].startswith("topology_generator:")

    def test_reset_creates_correct_number_of_envs(self):
        env = SageTopologyVerlEnv()
        env.reset(["a", "b", "c"])
        assert env.n_envs == 3

    def test_reset_text_contains_prompt(self):
        env = SageTopologyVerlEnv()
        obs = env.reset(["Implement bubble sort"])
        assert "Implement bubble sort" in obs[0]["text"]

    def test_env_name_property(self):
        env = SageTopologyVerlEnv()
        assert env.env_name == "sage_topology"


# ---------------------------------------------------------------------------
# Step with YAML
# ---------------------------------------------------------------------------

class TestStepYAML:
    def test_valid_yaml_topology(self):
        """Valid YAML should execute nodes and reach terminal."""
        env = SageTopologyVerlEnv()
        env.reset(["Write a sort function"])

        obs, rewards, dones, infos = env.step([SIMPLE_YAML])

        assert len(obs) == 1
        assert len(rewards) == 1
        assert len(dones) == 1
        assert len(infos) == 1
        # Simple topology with no checkpoints -> should terminate
        assert dones[0] is True
        assert "anchor" in obs[0]

    def test_invalid_yaml_terminates(self):
        env = SageTopologyVerlEnv()
        env.reset(["test"])
        obs, rewards, dones, infos = env.step([INVALID_YAML])

        assert dones[0] is True
        assert rewards[0] < 0
        assert infos[0]["status"] == "YAML_ERROR"
        assert "anchor" in obs[0]

    def test_missing_nodes_key(self):
        env = SageTopologyVerlEnv()
        env.reset(["test"])
        obs, rewards, dones, infos = env.step(["reasoning: just text"])

        assert dones[0] is True
        assert infos[0]["status"] == "INVALID_YAML"

    def test_empty_nodes(self):
        env = SageTopologyVerlEnv()
        env.reset(["test"])
        obs, rewards, dones, infos = env.step(["nodes: []"])

        assert dones[0] is True
        assert infos[0]["status"] == "EMPTY_TOPOLOGY"


# ---------------------------------------------------------------------------
# Step with decisions (checkpoint flow)
# ---------------------------------------------------------------------------

class TestStepDecisions:
    def test_checkpoint_pauses_for_decision(self):
        """Checkpoint node should pause and return CHECKPOINT status."""
        env = SageTopologyVerlEnv()
        env.reset(["Sort a list"])

        obs, rewards, dones, infos = env.step([CHECKPOINT_YAML])

        # Node 0 is a checkpoint -> should ask for a decision, not be done
        assert dones[0] is False
        assert infos[0].get("status") == "CHECKPOINT"
        assert "anchor" in obs[0]
        assert "[CHECKPOINT]" in obs[0]["text"]

    def test_continue_decision(self):
        """After checkpoint, 'continue' should resume execution."""
        env = SageTopologyVerlEnv()
        env.reset(["Sort a list"])
        env.step([CHECKPOINT_YAML])  # pauses at checkpoint

        obs, rewards, dones, infos = env.step(["continue"])

        # After continuing, should finish (only 2 nodes, checkpoint was at 0)
        assert dones[0] is True

    def test_reroute_decision(self):
        """'reroute' should terminate with REROUTED status."""
        env = SageTopologyVerlEnv()
        env.reset(["Sort a list"])
        env.step([CHECKPOINT_YAML])  # pauses at checkpoint

        obs, rewards, dones, infos = env.step(["reroute"])

        assert dones[0] is True
        assert infos[0]["status"] == "REROUTED"
        # The terminal step's reward is exec_score (0.0), but the reroute
        # penalty (-0.3) is recorded in the StepRewardVector.
        vecs = env.get_step_rewards()
        # Reroute penalty step should be present in step_rewards
        assert any(r < 0 for r in vecs[0].step_rewards), \
            "Reroute penalty should appear as a negative step reward"
        assert any("reroute" in a for a in vecs[0].anchor_keys), \
            "Reroute anchor should appear in anchor keys"

    def test_upgrade_decision(self):
        """'upgrade' should re-execute the node with fallback tier."""
        env = SageTopologyVerlEnv()
        env.reset(["Sort a list"])
        env.step([CHECKPOINT_YAML])  # pauses at checkpoint

        obs, rewards, dones, infos = env.step(["upgrade"])

        # Should eventually terminate (upgrade + continue through remaining nodes)
        assert dones[0] is True


# ---------------------------------------------------------------------------
# Anchor keys
# ---------------------------------------------------------------------------

class TestAnchors:
    def test_anchor_present_in_all_observations(self):
        """Every observation from reset and step must have anchor."""
        env = SageTopologyVerlEnv()
        obs = env.reset(["test"])
        assert "anchor" in obs[0]

        obs, _, _, _ = env.step([SIMPLE_YAML])
        assert "anchor" in obs[0]

    def test_anchor_is_string(self):
        env = SageTopologyVerlEnv()
        obs = env.reset(["test"])
        assert isinstance(obs[0]["anchor"], str)
        assert len(obs[0]["anchor"]) > 0


# ---------------------------------------------------------------------------
# get_step_rewards
# ---------------------------------------------------------------------------

class TestGetStepRewards:
    def test_returns_step_reward_vectors(self):
        """get_step_rewards() should return one StepRewardVector per env."""
        env = SageTopologyVerlEnv()
        env.reset(["Write sort", "Write hello"])
        env.step([SIMPLE_YAML, SIMPLE_YAML])

        vecs = env.get_step_rewards()

        assert len(vecs) == 2
        for vec in vecs:
            assert isinstance(vec, StepRewardVector)
            assert vec.n_steps >= 2  # at least topology + terminal
            assert len(vec.anchor_keys) == vec.n_steps
            assert all(isinstance(a, str) for a in vec.anchor_keys)

    def test_step_reward_to_verl_format(self):
        """StepRewardVector.to_verl_format() should produce valid dict."""
        env = SageTopologyVerlEnv()
        env.reset(["test"])
        env.step([SIMPLE_YAML])

        vecs = env.get_step_rewards()
        verl_fmt = vecs[0].to_verl_format()

        assert "rewards" in verl_fmt
        assert "anchor_keys" in verl_fmt
        assert "total_return" in verl_fmt
        assert "n_steps" in verl_fmt
        assert verl_fmt["n_steps"] == len(verl_fmt["rewards"])


# ---------------------------------------------------------------------------
# build_text_obs
# ---------------------------------------------------------------------------

class TestBuildTextObs:
    def test_extracts_text_from_observations(self):
        env = SageTopologyVerlEnv()
        obs = env.reset(["Write sort", "Write hello"])

        texts = env.build_text_obs(obs)

        assert len(texts) == 2
        assert all(isinstance(t, str) for t in texts)
        assert "Write sort" in texts[0]
        assert "Write hello" in texts[1]

    def test_handles_missing_text_key(self):
        env = SageTopologyVerlEnv()
        texts = env.build_text_obs([{"image": None, "anchor": "x"}])
        assert texts == [""]


# ---------------------------------------------------------------------------
# success_evaluator
# ---------------------------------------------------------------------------

class TestSuccessEvaluator:
    def test_passed_is_success(self):
        env = SageTopologyVerlEnv()
        result = env.success_evaluator([
            {"infos": [{"status": "PASSED"}]},
        ])
        assert result == [True]

    def test_failed_is_not_success(self):
        env = SageTopologyVerlEnv()
        result = env.success_evaluator([
            {"infos": [{"status": "EXEC_ERROR"}]},
        ])
        assert result == [False]

    def test_empty_trajectory(self):
        env = SageTopologyVerlEnv()
        result = env.success_evaluator([{"infos": []}])
        assert result == [False]

    def test_no_infos_key(self):
        env = SageTopologyVerlEnv()
        result = env.success_evaluator([{}])
        assert result == [False]


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

class TestProjection:
    def test_identity_strips_whitespace(self):
        assert sage_topology_projection("  continue  ") == "continue"

    def test_yaml_passthrough(self):
        yaml_text = "nodes:\n  - role: coder\n"
        assert sage_topology_projection(yaml_text) == yaml_text.strip()

    def test_empty_string(self):
        assert sage_topology_projection("") == ""

    def test_multiline_preserves_content(self):
        text = "upgrade\nsome reasoning"
        assert sage_topology_projection(text) == text


# ---------------------------------------------------------------------------
# Multiple environments (vectorized)
# ---------------------------------------------------------------------------

class TestVectorized:
    def test_mixed_valid_invalid(self):
        """One valid, one invalid YAML in same batch."""
        env = SageTopologyVerlEnv()
        env.reset(["task 1", "task 2"])
        obs, rewards, dones, infos = env.step([SIMPLE_YAML, INVALID_YAML])

        # Valid topology terminates (no checkpoints, 2 nodes)
        assert dones[0] is True
        # Invalid YAML terminates immediately
        assert dones[1] is True
        assert infos[1]["status"] == "YAML_ERROR"

    def test_already_done_noop(self):
        """Stepping a done env should return no-op."""
        env = SageTopologyVerlEnv()
        env.reset(["test"])
        env.step([SIMPLE_YAML])  # terminates

        # Step again -- should be no-op
        obs, rewards, dones, infos = env.step(["anything"])
        assert dones[0] is True
        assert rewards[0] == 0.0
        assert infos[0]["status"] == "ALREADY_DONE"

    def test_close(self):
        env = SageTopologyVerlEnv()
        env.reset(["test"])
        env.close()
        assert env.n_envs == 0


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config(self):
        env = SageTopologyVerlEnv()
        assert env._config == {}

    def test_custom_config(self):
        env = SageTopologyVerlEnv({"n_envs": 4, "memory_db": "/tmp/test.db"})
        assert env._config["n_envs"] == 4
        assert env._config["memory_db"] == "/tmp/test.db"

    def test_none_config(self):
        env = SageTopologyVerlEnv(None)
        assert env._config == {}
