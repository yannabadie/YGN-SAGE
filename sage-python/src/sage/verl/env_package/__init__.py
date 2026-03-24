"""Proper verl-agent environment package for SageTopologyEnv.

Instead of monkey-patching make_envs(), this module provides:
1. SageTopologyVerlEnv -- verl-agent compatible vectorized environment
2. sage_topology_projection() -- identity text projection
3. build_sage_topology_envs() -- factory function matching verl-agent convention

Usage in train script:
    export PYTHONPATH="$SAGE_ROOT/sage-python/src:$PYTHONPATH"
    # Then in verl-agent config: env.env_name=sage_topology

The env_package can be discovered by:
- verl-agent's make_envs() factory if patched via env_register.py
- Direct import: from sage.verl.env_package import SageTopologyVerlEnv
"""

from sage.verl.env_package.envs import SageTopologyVerlEnv
from sage.verl.env_package.projection import sage_topology_projection

__all__ = [
    "SageTopologyVerlEnv",
    "sage_topology_projection",
    "build_sage_topology_envs",
]


def build_sage_topology_envs(config: dict) -> SageTopologyVerlEnv:
    """Factory function matching verl-agent convention.

    Args:
        config: Environment configuration dict. Expected keys:
            - n_envs (int): Number of parallel environments (default 1)
            - memory_db (str): Path to episodic memory SQLite DB
            - max_steps (int): Max steps per episode

    Returns:
        SageTopologyVerlEnv instance ready for reset()/step() calls.
    """
    return SageTopologyVerlEnv(config)
