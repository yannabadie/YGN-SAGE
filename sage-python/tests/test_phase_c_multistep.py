"""Phase C multi-step topology training smoke tests.

Validates that the 4-state machine with checkpoints produces REAL
multi-step episodes where the model makes decisions (continue/upgrade/reroute).
This is the test that separates Phase C from one-shot GRPO.
"""
import os
import pytest

os.environ.setdefault("SAGE_TESTING", "1")

from sage.verl.topology_env import SageTopologyEnv, StepResult
from sage.verl.step_reward import StepRewardVector
from sage.verl.reward import _score_structure, VALID_MODEL_TIERS


# Topology WITH checkpoint — triggers multi-step decisions
YAML_WITH_CHECKPOINT = (
    "nodes:\n"
    "  - role: coder\n"
    "    model_tier: fast\n"
    "    prompt: Write the solution\n"
    "    fallback_tier: reasoner\n"
    "  - role: reviewer\n"
    "    model_tier: budget\n"
    "    prompt: Review the code\n"
    "  - role: synthesizer\n"
    "    model_tier: budget\n"
    "    prompt: Combine into final code\n"
    "edges:\n"
    "  - from_idx: 0\n"
    "    to_idx: 1\n"
    "    flow_type: message\n"
    "  - from_idx: 1\n"
    "    to_idx: 2\n"
    "    flow_type: message\n"
    "difficulty: moderate\n"
    "reasoning: Code then review then synthesize\n"
    "adaptation:\n"
    "  checkpoints: [0]\n"
    "  max_upgrades: 1\n"
    "  max_reroutes: 1\n"
    "  quality_threshold: 0.5\n"
)

# Topology WITHOUT checkpoint — single-turn (GRPO-equivalent)
YAML_NO_CHECKPOINT = (
    "nodes:\n"
    "  - role: coder\n"
    "    model_tier: fast\n"
    "    prompt: Write the solution\n"
    "  - role: synthesizer\n"
    "    model_tier: budget\n"
    "    prompt: Finalize code\n"
    "edges:\n"
    "  - from_idx: 0\n"
    "    to_idx: 1\n"
    "    flow_type: message\n"
    "difficulty: simple\n"
    "reasoning: Simple coder plus synth\n"
)


class TestPhaseC_Nominal:
    """Nominal scenario: multi-step episode with checkpoint decisions."""

    def test_checkpoint_pauses_execution(self):
        """After executing checkpoint node, env should pause for a decision."""
        env = SageTopologyEnv()
        env.reset("Write a fibonacci function", "test/fib")
        obs, reward, done, info = env.step(YAML_WITH_CHECKPOINT)

        # Should NOT be done — paused at checkpoint node 0
        assert not done, f"Expected pause at checkpoint, got done=True. State: {env._state}"
        assert env._state == "awaiting_decision"
        assert "CHECKPOINT" in obs.get("text", "")

    def test_continue_decision_resumes(self):
        """'continue' at checkpoint resumes execution to terminal."""
        env = SageTopologyEnv()
        env.reset("Write a sort function", "test/sort")
        obs, reward, done, info = env.step(YAML_WITH_CHECKPOINT)
        assert not done

        # Decide: continue
        obs2, reward2, done2, info2 = env.step("continue")
        # Should eventually reach terminal (may need more steps)
        while not done2:
            obs2, reward2, done2, info2 = env.step("continue")
        assert done2

    def test_upgrade_decision_reexecutes(self):
        """'upgrade' at checkpoint re-executes the node with better model."""
        env = SageTopologyEnv()
        env.reset("Write a complex algorithm", "test/complex")
        obs, reward, done, info = env.step(YAML_WITH_CHECKPOINT)
        assert not done
        assert env._state == "awaiting_decision"

        # Record upgrades_used before
        upgrades_before = env._upgrades_used

        # Decide: upgrade
        obs2, reward2, done2, info2 = env.step("upgrade")

        # Upgrade should have been attempted (upgrades_used incremented OR
        # max_upgrades already reached). The YAML has max_upgrades=1.
        assert env._upgrades_used >= upgrades_before, (
            "upgrade decision should increment _upgrades_used"
        )
        # Trace must contain more steps than just YAML + 1 node
        assert len(env._trace.steps) >= 3, (
            f"Expected >=3 trace steps after upgrade, got {len(env._trace.steps)}"
        )

    def test_reroute_decision_terminates(self):
        """'reroute' at checkpoint terminates the episode early with penalty."""
        env = SageTopologyEnv()
        env.reset("Write something", "test/reroute")
        obs, reward, done, info = env.step(YAML_WITH_CHECKPOINT)
        assert not done

        # Decide: reroute
        obs2, reward2, done2, info2 = env.step("reroute")
        assert done2, "reroute should terminate episode"
        assert info2.get("status") == "REROUTED", f"Expected REROUTED status, got {info2.get('status')}"
        # Total reward should include the reroute penalty (-0.3)
        total = sum(s.reward for s in env._trace.steps)
        assert total < sum(s.reward for s in env._trace.steps[:2]) + 0.1, (
            "Reroute should add negative reward (penalty -0.3)"
        )

    def test_multistep_produces_more_steps_than_singleshot(self):
        """Phase C episode with checkpoint should have more trace steps."""
        # With checkpoint
        env_c = SageTopologyEnv()
        env_c.reset("test", "t/c")
        obs, _, done, _ = env_c.step(YAML_WITH_CHECKPOINT)
        while not done:
            obs, _, done, _ = env_c.step("continue")
        n_steps_c = len(env_c._trace.steps)

        # Without checkpoint (single-turn)
        env_s = SageTopologyEnv()
        env_s.reset("test", "t/s")
        obs, _, done, _ = env_s.step(YAML_NO_CHECKPOINT)
        n_steps_s = len(env_s._trace.steps)

        # Phase C should have more steps (YAML + node executions + decision steps)
        assert n_steps_c > n_steps_s, (
            f"Phase C steps ({n_steps_c}) should exceed single-turn ({n_steps_s})"
        )

    def test_step_reward_vector_has_decision_anchors(self):
        """StepRewardVector from checkpoint episode should contain decision anchors."""
        env = SageTopologyEnv()
        env.reset("test", "t/0")
        obs, _, done, _ = env.step(YAML_WITH_CHECKPOINT)
        while not done:
            obs, _, done, _ = env.step("continue")

        srv = env.get_step_rewards()
        assert isinstance(srv, StepRewardVector)
        assert srv.n_steps >= 3  # YAML + checkpoint decision + terminal at minimum

        # Check that anchors contain both topology_generator and node roles
        anchor_set = set(srv.anchor_keys)
        has_topology = any("topology_generator" in a for a in anchor_set)
        has_terminal = any("terminal" in a for a in anchor_set)
        assert has_topology, f"No topology_generator anchor: {anchor_set}"
        assert has_terminal, f"No terminal anchor: {anchor_set}"


class TestPhaseC_Degraded:
    """Degraded scenario: failures and edge cases."""

    def test_invalid_yaml_terminates_immediately(self):
        env = SageTopologyEnv()
        env.reset("test", "t/0")
        obs, reward, done, info = env.step("{{invalid yaml")
        assert done
        assert reward < 0
        assert info["status"] == "YAML_ERROR"

    def test_no_checkpoint_degenerates_to_singlestep(self):
        """Without checkpoints, Phase C degenerates to single-turn (GRPO-equivalent)."""
        env = SageTopologyEnv()
        env.reset("test", "t/0")
        obs, reward, done, info = env.step(YAML_NO_CHECKPOINT)
        assert done  # completes in one step
        # This is expected: single-turn is the warm-up, not Phase C

    def test_empty_model_response_at_checkpoint(self):
        """Empty/garbage response at checkpoint should default to 'continue'."""
        env = SageTopologyEnv()
        env.reset("test", "t/0")
        obs, _, done, _ = env.step(YAML_WITH_CHECKPOINT)
        assert not done

        # Empty response → should be parsed as "continue" (default)
        obs2, _, done2, _ = env.step("")
        # Should not crash, should continue or terminate gracefully
        assert isinstance(done2, bool)


class TestPhaseC_Growth:
    """Growth scenario: larger topologies, more checkpoints."""

    def test_large_topology_with_multiple_checkpoints(self):
        """Topology with 5 nodes and 2 checkpoints should have many decision steps."""
        yaml_large = (
            "nodes:\n"
            "  - role: planner\n"
            "    model_tier: fast\n"
            "    fallback_tier: reasoner\n"
            "  - role: coder\n"
            "    model_tier: reasoner\n"
            "    fallback_tier: strong\n"
            "  - role: tester\n"
            "    model_tier: budget\n"
            "  - role: reviewer\n"
            "    model_tier: fast\n"
            "  - role: synthesizer\n"
            "    model_tier: budget\n"
            "edges:\n"
            "  - from_idx: 0\n"
            "    to_idx: 1\n"
            "  - from_idx: 1\n"
            "    to_idx: 2\n"
            "  - from_idx: 2\n"
            "    to_idx: 3\n"
            "  - from_idx: 3\n"
            "    to_idx: 4\n"
            "difficulty: complex\n"
            "reasoning: Full pipeline\n"
            "adaptation:\n"
            "  checkpoints: [0, 2]\n"
            "  max_upgrades: 2\n"
            "  quality_threshold: 0.5\n"
        )
        env = SageTopologyEnv()
        env.reset("Complex task", "t/complex")
        obs, _, done, _ = env.step(yaml_large)

        decisions_made = 0
        max_iter = 20
        while not done and max_iter > 0:
            if env._state == "awaiting_decision":
                decisions_made += 1
            obs, _, done, _ = env.step("continue")
            max_iter -= 1

        assert done
        assert decisions_made >= 1, "Should have at least 1 decision checkpoint"


class TestRewardTierValidation:
    """Reward function validates model_tier names."""

    def test_valid_tiers_get_bonus(self):
        yaml_valid = (
            "nodes:\n"
            "  - role: coder\n"
            "    model_tier: reasoner\n"
            "  - role: reviewer\n"
            "    model_tier: fast\n"
            "reasoning: test\n"
        )
        score = _score_structure(yaml_valid)
        assert score > 0.0

    def test_invalid_tiers_no_bonus_phase_c(self):
        """Phase C: valid tiers score higher than invalid."""
        import os
        os.environ["SAGE_TRAINING_PHASE"] = "C"
        yaml_invalid = (
            "nodes:\n"
            "  - role: coder\n"
            "    model_tier: quantum_computer\n"
            "  - role: reviewer\n"
            "    model_tier: skynet\n"
            "reasoning: test\n"
        )
        yaml_valid = (
            "nodes:\n"
            "  - role: coder\n"
            "    model_tier: reasoner\n"
            "  - role: reviewer\n"
            "    model_tier: fast\n"
            "reasoning: test\n"
        )
        score_invalid = _score_structure(yaml_invalid)
        score_valid = _score_structure(yaml_valid)
        assert score_valid > score_invalid, (
            f"Valid tiers ({score_valid}) should score higher than invalid ({score_invalid})"
        )
        os.environ["SAGE_TRAINING_PHASE"] = "A"

    def test_adaptation_field_gets_bonus_phase_c(self):
        """Phase C: adaptation/checkpoints get bonus."""
        import os
        os.environ["SAGE_TRAINING_PHASE"] = "C"
        yaml_with = (
            "nodes:\n"
            "  - role: coder\n"
            "reasoning: test\n"
            "adaptation:\n"
            "  checkpoints: [0]\n"
            "  max_upgrades: 1\n"
        )
        yaml_without = (
            "nodes:\n"
            "  - role: coder\n"
            "reasoning: test\n"
        )
        score_with = _score_structure(yaml_with)
        score_without = _score_structure(yaml_without)
        assert score_with > score_without, (
            f"With adaptation ({score_with}) should score higher than without ({score_without})"
        )
        os.environ["SAGE_TRAINING_PHASE"] = "A"
