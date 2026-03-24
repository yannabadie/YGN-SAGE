"""Text-to-action projection for SageTopologyEnv.

verl-agent uses projection functions to map model text output to
environment actions. For SageTopologyEnv, this is an identity mapping:
the model output (YAML topology or decision text) IS the action.

In topology generation (awaiting_yaml state), the model outputs YAML.
In checkpoint decisions (awaiting_decision state), the model outputs
"continue", "upgrade", or "reroute".
"""
from __future__ import annotations


def sage_topology_projection(text: str) -> str:
    """Identity projection -- model output IS the action.

    The model generates either:
    - YAML topology (step 0, awaiting_yaml state)
    - Decision text: "continue" / "upgrade" / "reroute" (checkpoint state)

    No transformation needed; the SageTopologyEnv parser handles both formats.

    Args:
        text: Raw model output text.

    Returns:
        Stripped text ready for env.step().
    """
    return text.strip()
