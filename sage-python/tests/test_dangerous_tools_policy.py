"""P0-1 (REVIEW3): execute_bash must boot with capability declaration.

Before the fix, SAGE_DANGEROUS_TOOLS=1 crashed at boot because
execute_bash was built without capability=ToolCapability.DANGEROUS
and the ToolPolicy resolver rejects tools with undeclared capability.
"""

from __future__ import annotations

import os

import pytest


def test_dangerous_tools_boot_registers_bash_with_capability(monkeypatch):
    """SAGE_DANGEROUS_TOOLS=1 must register execute_bash without crashing."""
    monkeypatch.setenv("SAGE_DANGEROUS_TOOLS", "1")
    monkeypatch.setenv("SAGE_TOOL_GRANTS", "dangerous")
    from sage.boot import boot_agent_system

    system = boot_agent_system(use_mock_llm=True)
    tool = system.tool_registry.get("execute_bash")
    assert tool is not None, "execute_bash not found in registry"
    assert tool.capability is not None, "execute_bash has no capability declared"
    from sage.policy import ToolCapability
    assert tool.capability is ToolCapability.DANGEROUS, (
        f"expected ToolCapability.DANGEROUS, got {tool.capability!r}"
    )


def test_dangerous_tools_boot_without_env_var_does_not_register_bash(monkeypatch):
    """Without SAGE_DANGEROUS_TOOLS=1, execute_bash must NOT be registered."""
    monkeypatch.delenv("SAGE_DANGEROUS_TOOLS", raising=False)
    from sage.boot import boot_agent_system

    system = boot_agent_system(use_mock_llm=True)
    tool = system.tool_registry.get("execute_bash")
    assert tool is None, (
        "execute_bash should not be registered without SAGE_DANGEROUS_TOOLS=1"
    )
