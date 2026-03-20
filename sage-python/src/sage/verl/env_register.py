"""Register SageTopologyEnv in the verl-agent environment registry.

verl-agent discovers environments via its env_manager.py make_envs() factory.
This module provides a registration hook that must be imported before training.

Usage in train script (before verl.trainer.main_ppo):
    python3 -c "import sage.verl.env_register" && python3 -m verl.trainer.main_ppo ...

Or via PYTHONPATH injection in train_topology.sh.
"""
from __future__ import annotations

import logging

log = logging.getLogger("sage_env_register")


def register_sage_topology_env():
    """Register SageTopologyEnv in verl-agent's environment system.

    verl-agent uses env_manager.py with a make_envs() factory that dispatches
    by env_name string. We monkey-patch the registry to add our environment.
    """
    try:
        from agent_system.environments.env_manager import make_envs as _original_make_envs
        from sage.verl.topology_env import SageTopologyEnvManager

        # Wrap the original make_envs to add sage_topology support
        def patched_make_envs(config):
            if "sage_topology" in config.env.env_name.lower():
                return SageTopologyEnvManager(config)
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
