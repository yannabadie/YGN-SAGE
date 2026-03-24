"""Register SageTopologyEnv in the verl-agent environment registry.

verl-agent discovers environments via its env_manager.py make_envs() factory.
This module provides a registration hook that must be imported before training.

Usage in train script (before verl.trainer.main_ppo):
    python3 -c "import sage.verl.env_register" && python3 -m verl.trainer.main_ppo ...

Or via PYTHONPATH injection in train_topology.sh.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("sage_env_register")


def _extract_env_config(verl_config) -> dict:
    """Extract SageTopologyEnv config from a verl-agent Hydra config object.

    verl-agent passes its full Hydra config to make_envs(). SageTopologyEnv
    expects a plain dict with optional keys like 'memory_db'. This function
    bridges the two.

    Also reads SAGE_TRAINING_MEMORY_DB env var for episodic memory path.
    """
    env_dict: dict = {}

    # Try to extract env sub-config from Hydra object
    try:
        env_sub = verl_config.env
        # max_steps used by verl-agent rollout, not by SageTopologyEnv directly
        if hasattr(env_sub, "max_steps"):
            env_dict["max_steps"] = int(env_sub.max_steps)
    except (AttributeError, TypeError):
        pass

    # Episodic memory DB from env var (set in train_topology_phase_c.sh)
    memory_db = os.environ.get("SAGE_TRAINING_MEMORY_DB", "")
    if memory_db:
        env_dict["memory_db"] = memory_db

    return env_dict


def register_sage_topology_env():
    """Register SageTopologyEnv in verl-agent's environment system.

    verl-agent uses env_manager.py with a make_envs() factory that dispatches
    by env_name string. We monkey-patch the registry to add our environment.

    The patched factory:
    1. Checks if env_name contains 'sage_topology'
    2. Extracts a plain dict config from the Hydra config object
    3. Returns a SageTopologyEnvManager initialized with that dict
    """
    try:
        from agent_system.environments.env_manager import make_envs as _original_make_envs
        from sage.verl.topology_env import SageTopologyEnvManager

        # Wrap the original make_envs to add sage_topology support
        def patched_make_envs(config):
            if "sage_topology" in config.env.env_name.lower():
                env_config = _extract_env_config(config)
                log.info("Creating SageTopologyEnvManager with config: %s", env_config)
                return SageTopologyEnvManager(env_config)
            return _original_make_envs(config)

        # Monkey-patch
        import agent_system.environments.env_manager as em
        em.make_envs = patched_make_envs
        log.info("Registered SageTopologyEnv in verl-agent registry")

    except ImportError:
        log.warning(
            "verl-agent not installed (agent_system not found). "
            "SageTopologyEnv registration skipped. "
            "Install verl-agent: pip install -e /workspace/verl-agent"
        )


# Auto-register on import
register_sage_topology_env()
