"""Test that routing always goes through the same path (B8)."""
import inspect

import pytest

from sage.boot import boot_agent_system


def test_no_dual_routing_paths():
    """AgentSystem.run should not have environment-dependent routing branches."""
    system = boot_agent_system(use_mock_llm=True)
    source = inspect.getsource(type(system).run)
    # Should not contain orchestrator branching in run()
    assert "orchestrator.run" not in source, (
        "run() should not branch to orchestrator — single control plane only"
    )
    assert "registry.refresh" not in source, (
        "run() should not call registry.refresh — that creates an env-dependent branch"
    )
    assert "registry.list_available" not in source, (
        "run() should not check registry.list_available — single path only"
    )


def test_orchestrator_still_available():
    """Orchestrator and registry fields still exist for explicit use."""
    from sage.boot import AgentSystem
    import dataclasses

    field_names = [f.name for f in dataclasses.fields(AgentSystem)]
    assert "orchestrator" in field_names, "orchestrator field should be retained"
    assert "registry" in field_names, "registry field should be retained"
