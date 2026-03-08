"""Test that routing has orchestrator-primary with legacy fallback."""
import inspect

import pytest

from sage.boot import boot_agent_system


def test_orchestrator_primary_with_fallback():
    """AgentSystem.run should use orchestrator as primary, ModelRouter as fallback."""
    system = boot_agent_system(use_mock_llm=True)
    source = inspect.getsource(type(system).run)
    # Orchestrator is the primary path (wired in Task 2)
    assert "orchestrator.run" in source, (
        "run() should call orchestrator.run as primary routing path"
    )
    # Legacy ModelRouter fallback is retained
    assert "ModelRouter.get_config" in source, (
        "run() should retain ModelRouter.get_config as legacy fallback"
    )
    # Must not call registry.refresh in run() — that belongs in boot
    assert "registry.refresh" not in source, (
        "run() should not call registry.refresh — that belongs at boot time"
    )


def test_orchestrator_still_available():
    """Orchestrator and registry fields still exist for explicit use."""
    from sage.boot import AgentSystem
    import dataclasses

    field_names = [f.name for f in dataclasses.fields(AgentSystem)]
    assert "orchestrator" in field_names, "orchestrator field should be retained"
    assert "registry" in field_names, "registry field should be retained"
