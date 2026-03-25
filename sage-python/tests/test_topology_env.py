"""Tests for SageTopologyEnv — GiGPO multi-step environment."""
import pytest
from sage.verl.topology_env import SageTopologyEnv, _make_anchor, SageTopologyEnvManager
from sage.verl.step_reward import StepRewardVector


class TestMakeAnchor:
    def test_basic(self):
        assert _make_anchor("coder", "moderate", "abc") == "coder:moderate:abc"

    def test_empty_context(self):
        assert _make_anchor("reviewer", "simple", "") == "reviewer:simple:"

    def test_same_inputs_same_output(self):
        a1 = _make_anchor("coder", "complex", "xyz")
        a2 = _make_anchor("coder", "complex", "xyz")
        assert a1 == a2

    def test_different_roles_different_anchors(self):
        a1 = _make_anchor("coder", "moderate", "abc")
        a2 = _make_anchor("reviewer", "moderate", "abc")
        assert a1 != a2


class TestSageTopologyEnvReset:
    def test_reset_returns_observation(self):
        env = SageTopologyEnv()
        obs = env.reset("Write sort function", "test/1")
        assert "text" in obs
        assert "anchor" in obs
        assert "image" in obs
        assert obs["text"] == "Write sort function"
        assert obs["anchor"].startswith("topology_generator:")

    def test_reset_clears_state(self):
        env = SageTopologyEnv()
        env.reset("task 1", "t1")
        env.step("nodes:\n- role: coder")  # run a topology
        env.reset("task 2", "t2")
        # V3 state machine: _state resets to "awaiting_yaml", _exec_cursor to 0
        assert env._state == "awaiting_yaml"
        assert env._exec_cursor == 0
        assert env._trace.prompt == "task 2"


class TestSageTopologyEnvStep0:
    def test_invalid_yaml(self):
        env = SageTopologyEnv()
        env.reset("test", "t/0")
        obs, reward, done, info = env.step("{{not yaml")
        assert done is True
        assert reward < 0
        assert info["status"] == "YAML_ERROR"

    def test_no_nodes_key(self):
        env = SageTopologyEnv()
        env.reset("test", "t/0")
        obs, reward, done, info = env.step("reasoning: just text")
        assert done is True
        assert info["status"] == "INVALID_YAML"

    def test_empty_nodes(self):
        env = SageTopologyEnv()
        env.reset("test", "t/0")
        obs, reward, done, info = env.step("nodes: []")
        assert done is True
        assert info["status"] == "EMPTY_TOPOLOGY"

    def test_valid_topology_executes_to_terminal(self):
        """V3 state machine: without checkpoints, YAML step executes all nodes
        and reaches terminal in a single step() call."""
        env = SageTopologyEnv()
        env.reset("test", "t/0")
        yaml_text = (
            "nodes:\n"
            "  - role: coder\n"
            "    prompt: Write code\n"
            "  - role: reviewer\n"
            "    prompt: Review code\n"
            "edges:\n"
            "  - from_idx: 0\n"
            "    to_idx: 1\n"
            "reasoning: Need coding and review\n"
            "difficulty: moderate"
        )
        obs, reward, done, info = env.step(yaml_text)
        # Without checkpoints, the env executes all nodes immediately
        assert done is True
        assert "anchor" in obs
        # Trace should contain step 0 (topology) + per-node steps
        assert len(env._trace.steps) >= 2

    def test_structural_score_in_trace(self):
        """Structural score for YAML: nodes(0.3) + edges(0.2) + roles(0.3) + reasoning(0.2) = 1.0."""
        env = SageTopologyEnv()
        env.reset("test", "t/0")
        yaml_text = (
            "nodes:\n"
            "  - role: coder\n"
            "  - role: reviewer\n"
            "edges:\n"
            "  - from_idx: 0\n"
            "    to_idx: 1\n"
            "reasoning: test\n"
            "difficulty: simple"
        )
        obs, reward, done, info = env.step(yaml_text)
        # Step 0 (topology_generator) structural reward is in the trace
        step0 = env._trace.steps[0]
        assert step0.role == "topology_generator"
        assert step0.reward == pytest.approx(1.0)


class TestSageTopologyEnvMultiStep:
    def test_full_episode_without_checkpoints(self):
        """V3: without checkpoints, YAML step runs all nodes → terminal in one call."""
        env = SageTopologyEnv()
        env.reset("Write a sort function", "test/sort")
        yaml_text = (
            "nodes:\n"
            "  - role: coder\n"
            "    prompt: Write sort\n"
            "  - role: reviewer\n"
            "    prompt: Review code\n"
            "edges:\n"
            "  - from_idx: 0\n"
            "    to_idx: 1\n"
            "difficulty: moderate\n"
            "reasoning: Code then review"
        )

        obs, reward, done, info = env.step(yaml_text)
        # Without checkpoints: all nodes executed in single step
        assert done is True
        # Trace: step 0 (topology) + 2 node steps + terminal
        assert len(env._trace.steps) >= 3

    def test_anchor_states_differ_by_role_in_trace(self):
        """V3: anchors are tracked in the trace (not via external step calls)."""
        env = SageTopologyEnv()
        env.reset("test", "t/0")
        yaml_text = (
            "nodes:\n"
            "  - role: planner\n"
            "  - role: coder\n"
            "  - role: synthesizer\n"
            "difficulty: moderate\n"
            "reasoning: Plan code synth"
        )
        env.step(yaml_text)  # executes all nodes in one call

        # Collect anchors from the trace (not from step calls)
        anchors = set()
        for step in env._trace.steps:
            anchors.add(step.anchor_key)

        # Step 0 has topology_generator anchor, each node has its role anchor
        assert len(anchors) >= 2, f"Expected >=2 unique anchors, got {anchors}"


class TestStepRewardVector:
    def test_from_episode_trace(self):
        env = SageTopologyEnv()
        env.reset("test", "t/0")
        yaml_text = "nodes:\n  - role: coder\n  - role: reviewer\ndifficulty: simple\nreasoning: test"
        env.step(yaml_text)
        while True:
            _, _, done, _ = env.step("continue")
            if done:
                break

        vec = env.get_step_rewards()
        assert isinstance(vec, StepRewardVector)
        assert vec.n_steps >= 2  # at least step 0 + terminal
        assert len(vec.anchor_keys) == vec.n_steps
        assert all(isinstance(a, str) for a in vec.anchor_keys)

        verl_fmt = vec.to_verl_format()
        assert "rewards" in verl_fmt
        assert "anchor_keys" in verl_fmt
        assert verl_fmt["n_steps"] == vec.n_steps


class TestSageTopologyEnvManager:
    def test_make_envs(self):
        mgr = SageTopologyEnvManager()
        envs = mgr.make(n_envs=3)
        assert len(envs) == 3

    def test_batch_reset(self):
        mgr = SageTopologyEnvManager()
        mgr.make(n_envs=2)
        obs = mgr.reset(["task 1", "task 2"], ["t/1", "t/2"])
        assert len(obs) == 2
        assert all("anchor" in o for o in obs)

    def test_batch_step(self):
        mgr = SageTopologyEnvManager()
        mgr.make(n_envs=2)
        mgr.reset(["task 1", "task 2"])
        obs, rewards, dones, infos = mgr.step(["nodes:\n- role: coder", "{{invalid"])
        assert len(obs) == 2
        assert len(rewards) == 2
        assert dones[1] is True  # invalid YAML terminates
