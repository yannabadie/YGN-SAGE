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


def test_topology_env_works_without_verl_agent():
    """SageTopologyEnv must work without verl-agent (direct use mode).

    train_phase_c_custom.py uses SageTopologyEnv directly, without verl-agent.
    The env must NOT block this use case — only env_register.py guards
    against verl-agent absence (for the verl-agent integration path).
    """
    from sage.verl.topology_env import SageTopologyEnv

    SageTopologyEnv._VERL_AGENT_AVAILABLE = None
    old = os.environ.pop("SAGE_TESTING", None)
    try:
        # Must NOT raise — direct use is always allowed
        env = SageTopologyEnv()
        assert env is not None
    finally:
        if old is not None:
            os.environ["SAGE_TESTING"] = old
        SageTopologyEnv._VERL_AGENT_AVAILABLE = None


def test_topology_env_logs_verl_agent_status(caplog):
    """SageTopologyEnv should log verl-agent availability status."""
    from sage.verl.topology_env import SageTopologyEnv
    import logging

    SageTopologyEnv._VERL_AGENT_AVAILABLE = None
    with caplog.at_level(logging.INFO, logger="topology_env"):
        env = SageTopologyEnv()
    SageTopologyEnv._VERL_AGENT_AVAILABLE = None

    # Should have logged something about verl-agent status
    try:
        import agent_system  # noqa: F401
        # verl-agent installed — no warning expected
    except ImportError:
        assert any("verl-agent" in r.message for r in caplog.records), (
            "Should log verl-agent availability when not installed"
        )
