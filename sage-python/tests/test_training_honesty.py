"""Tests that training scripts and environments are honest about what they do.

Issue G audit fix: VeRLGIGPO branch was claiming GiGPO when training scripts
actually use GRPO via vanilla verl 0.7.1.
"""
import os
import re

import pytest


def test_v5_script_uses_grpo_not_gigpo():
    """V5 training script must declare GRPO, not GiGPO."""
    script_path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "verl", "train_topology_v5.sh"
    )
    with open(script_path) as f:
        content = f.read()

    # Must contain GRPO as the adv_estimator
    assert "adv_estimator=grpo" in content, "V5 script must use adv_estimator=grpo"

    # Must NOT claim GiGPO in the algorithm header
    header_match = re.search(r"# Algorithm:.*", content)
    assert header_match, "V5 script must have an Algorithm comment"
    header = header_match.group(0)
    assert "GiGPO-equivalent" not in header, (
        f"V5 script header still claims GiGPO-equivalent: {header}"
    )


def test_v5_experiment_name_says_grpo():
    """V5 experiment name must reflect the actual algorithm (GRPO)."""
    script_path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "verl", "train_topology_v5.sh"
    )
    with open(script_path) as f:
        content = f.read()

    match = re.search(r"experiment_name=(\S+)", content)
    assert match, "V5 script must have an experiment_name"
    name = match.group(1)
    assert "grpo" in name.lower(), f"experiment_name should contain 'grpo', got: {name}"
    assert "gigpo" not in name.lower(), f"experiment_name should NOT contain 'gigpo', got: {name}"


def test_topology_env_requires_verl_agent():
    """SageTopologyEnv must fail explicitly when verl-agent is missing."""
    # Clear cached check
    from sage.verl.topology_env import SageTopologyEnv

    SageTopologyEnv._VERL_AGENT_AVAILABLE = None

    # Remove SAGE_TESTING if set, to test the guard
    old = os.environ.pop("SAGE_TESTING", None)
    try:
        # verl-agent is not installed in test env
        try:
            import agent_system  # noqa: F401
            pytest.skip("verl-agent is installed, guard test not applicable")
        except ImportError:
            pass

        with pytest.raises(RuntimeError, match="requires verl-agent"):
            SageTopologyEnv()
    finally:
        if old is not None:
            os.environ["SAGE_TESTING"] = old
        # Reset cached state
        SageTopologyEnv._VERL_AGENT_AVAILABLE = None


def test_topology_env_works_in_test_mode():
    """SageTopologyEnv should work when SAGE_TESTING=1 (even without verl-agent)."""
    from sage.verl.topology_env import SageTopologyEnv

    SageTopologyEnv._VERL_AGENT_AVAILABLE = None
    old = os.environ.get("SAGE_TESTING")
    os.environ["SAGE_TESTING"] = "1"
    try:
        env = SageTopologyEnv()
        assert env is not None
    finally:
        if old is None:
            os.environ.pop("SAGE_TESTING", None)
        else:
            os.environ["SAGE_TESTING"] = old
        SageTopologyEnv._VERL_AGENT_AVAILABLE = None
