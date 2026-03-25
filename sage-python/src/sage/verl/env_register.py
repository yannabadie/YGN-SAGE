"""Register SageTopologyEnv in the verl-agent environment registry.

verl-agent discovers environments via its env_manager.py make_envs() factory.
This module provides clean registration with graceful fallback.

Strategy (in order of preference):
1. Import SageTopologyVerlEnv from env_package (available on PYTHONPATH)
2. Patch make_envs() to dispatch 'sage_topology' to our env_package
3. Monkey-patch as last resort (with warning)

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


def register_sage_topology_env() -> str:
    """Register SageTopologyEnv in verl-agent's environment system.

    Returns:
        str: Registration method used ("env_package", "patch", "monkey_patch",
             or "skipped" if verl-agent is not installed).
    """
    # Step 1: Verify our env_package is importable
    try:
        from sage.verl.env_package import SageTopologyVerlEnv, build_sage_topology_envs
    except ImportError as exc:
        log.error("Failed to import sage.verl.env_package: %s", exc)
        return "error"

    # Step 2: Try to import verl-agent's make_envs
    try:
        import agent_system.environments.env_manager as em
    except ImportError:
        log.info(
            "verl-agent not installed (agent_system not found). "
            "SageTopologyEnv registration skipped — using GRPO via vanilla verl, NOT GiGPO. "
            "For GiGPO multi-step training: pip install -e /workspace/verl-agent"
        )
        return "skipped"

    # Step 3: Try clean registration -- add elif to make_envs dispatch
    original_make_envs = getattr(em, "make_envs", None)
    if original_make_envs is None:
        log.warning("verl-agent env_manager has no make_envs() -- cannot register")
        return "error"

    # Check if already registered (idempotent)
    if getattr(em, "_sage_topology_registered", False):
        log.debug("SageTopologyEnv already registered")
        return "already_registered"

    # Create the patched factory
    def patched_make_envs(config):
        """make_envs with sage_topology dispatch added."""
        try:
            env_name = config.env.env_name.lower()
        except (AttributeError, TypeError):
            # Config doesn't have env.env_name -- pass through
            return original_make_envs(config)

        if "sage_topology" in env_name:
            env_config = _extract_env_config(config)
            log.info(
                "Creating SageTopologyVerlEnv (env_package) with config: %s",
                env_config,
            )
            return build_sage_topology_envs(env_config)

        return original_make_envs(config)

    # Apply the patch
    em.make_envs = patched_make_envs
    em._sage_topology_registered = True

    log.info(
        "Registered SageTopologyEnv via env_package "
        "(patched make_envs -> SageTopologyVerlEnv)"
    )
    return "patch"


# Auto-register on import
_registration_method = register_sage_topology_env()
