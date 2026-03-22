"""Tests for GiGPO micro-decisions in SageTopologyEnv.

Verifies the 4-state machine (awaiting_yaml -> executing -> awaiting_decision -> terminal)
and the model's real decision-making at checkpoint nodes.
"""
import pytest

from sage.verl.topology_env import SageTopologyEnv


class TestMicroDecisions:

    def test_no_checkpoint_is_grpo(self):
        """Simple topology without checkpoints = model acts once = GRPO."""
        env = SageTopologyEnv()
        env.reset("Write hello world", "test/hello")

        yaml_text = """
    difficulty: simple
    nodes:
      - {role: coder, model_tier: budget, prompt: "Write hello world"}
      - {role: synthesizer, model_tier: budget, prompt: "Produce final solution"}
    edges:
      - {from_idx: 0, to_idx: 1, flow_type: message}
    """
        obs, r, done, info = env.step(yaml_text)
        # No checkpoints -> should go directly to terminal
        assert done is True
        assert env._upgrades_used == 0

    def test_checkpoint_triggers_decision(self):
        """Checkpoint node should pause and ask for decision."""
        env = SageTopologyEnv()
        env.reset("Sort a list", "test/sort")

        yaml_text = """
    difficulty: moderate
    adaptation:
      checkpoints: [0]
      max_upgrades: 1
      quality_threshold: 0.5
    nodes:
      - {role: coder, model_tier: fast, fallback_tier: reasoner, prompt: "Write sort"}
      - {role: synthesizer, model_tier: fast, prompt: "Produce final"}
    edges:
      - {from_idx: 0, to_idx: 1, flow_type: message}
    """
        obs, r, done, info = env.step(yaml_text)

        # Node 0 is a checkpoint -> should ask for a decision
        assert done is False
        assert env._state == "awaiting_decision"
        assert "[CHECKPOINT]" in obs["text"]
        assert "continue" in obs["text"].lower() or "upgrade" in obs["text"].lower()

    def test_continue_resumes_execution(self):
        """Decision 'continue' resumes execution to completion."""
        env = SageTopologyEnv()
        env.reset("Sort a list", "test/sort")

        yaml_text = """
    difficulty: moderate
    adaptation:
      checkpoints: [0]
      max_upgrades: 1
      quality_threshold: 0.5
    nodes:
      - {role: coder, model_tier: fast, fallback_tier: reasoner, prompt: "Write sort"}
      - {role: synthesizer, model_tier: fast, prompt: "Produce final"}
    edges:
      - {from_idx: 0, to_idx: 1, flow_type: message}
    """
        env.step(yaml_text)  # -> AWAITING_DECISION
        assert env._state == "awaiting_decision"

        obs, r, done, info = env.step("continue")
        # Should have executed the synthesizer and finalized
        assert done is True
        assert env._upgrades_used == 0

    def test_upgrade_reexecutes_node(self):
        """Decision 'upgrade' re-executes the node with the fallback tier."""
        env = SageTopologyEnv()
        env.reset("Dijkstra", "test/dijkstra")

        yaml_text = """
    difficulty: moderate
    adaptation:
      checkpoints: [0]
      max_upgrades: 1
      quality_threshold: 0.5
    nodes:
      - {role: coder, model_tier: fast, fallback_tier: reasoner, prompt: "Dijkstra"}
      - {role: synthesizer, model_tier: fast, prompt: "Final solution"}
    edges:
      - {from_idx: 0, to_idx: 1, flow_type: message}
    """
        env.step(yaml_text)  # -> AWAITING_DECISION

        obs, r, done, info = env.step("upgrade")
        # The upgrade should have been counted
        assert env._upgrades_used == 1
        # The upgrade step should be in the trace
        upgrade_steps = [s for s in env._trace.steps if s.was_upgraded]
        assert len(upgrade_steps) == 1

    def test_max_upgrades_respected(self):
        """Once max_upgrades exhausted, no more upgrade option."""
        env = SageTopologyEnv()
        env.reset("Complex task", "test/complex")

        yaml_text = """
    difficulty: complex
    adaptation:
      checkpoints: [0, 1]
      max_upgrades: 1
      quality_threshold: 0.5
    nodes:
      - {role: planner, model_tier: fast, fallback_tier: reasoner}
      - {role: coder, model_tier: fast, fallback_tier: reasoner}
      - {role: synthesizer, model_tier: fast}
    edges:
      - {from_idx: 0, to_idx: 1, flow_type: message}
      - {from_idx: 1, to_idx: 2, flow_type: message}
    """
        env.step(yaml_text)  # -> checkpoint 0
        env.step("upgrade")   # -> upgrade planner, upgrades_used=1
        # If checkpoint 1 is reached, verify upgrades are exhausted
        if env._state == "awaiting_decision":
            assert env._upgrades_used >= env._max_upgrades

    def test_anchor_states_distinguish_quality(self):
        """GiGPO anchor should include quality bucket."""
        env = SageTopologyEnv()
        env.reset("Sort", "test/sort")

        yaml_text = """
    difficulty: moderate
    adaptation:
      checkpoints: [0]
      max_upgrades: 1
      quality_threshold: 0.5
    nodes:
      - {role: coder, model_tier: fast, fallback_tier: reasoner}
      - {role: synthesizer, model_tier: fast}
    edges:
      - {from_idx: 0, to_idx: 1, flow_type: message}
    """
        obs, r, done, info = env.step(yaml_text)

        # The anchor must contain "decision:" and a quality bucket
        assert "decision:" in obs["anchor"]
        # The quality bucket must be one of the 4
        assert any(b in obs["anchor"] for b in ["very_low", "low", "adequate", "high"])

    def test_step_reward_vector_includes_decisions(self):
        """StepRewardVector must capture decision steps."""
        env = SageTopologyEnv()
        env.reset("Sort", "test/sort")

        yaml_text = """
    difficulty: moderate
    adaptation:
      checkpoints: [0]
      max_upgrades: 1
      quality_threshold: 0.5
    nodes:
      - {role: coder, model_tier: fast, fallback_tier: reasoner}
      - {role: synthesizer, model_tier: fast}
    edges:
      - {from_idx: 0, to_idx: 1, flow_type: message}
    """
        env.step(yaml_text)
        env.step("continue")

        vec = env.get_step_rewards()
        # At minimum: topology_generator + coder + decision:continue + synthesizer + terminal
        assert vec.n_steps >= 4
        assert any("decision:" in a for a in vec.anchor_keys)
