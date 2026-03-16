"""Test that routing has orchestrator-primary with legacy fallback."""
import inspect

import pytest

from sage.boot import boot_agent_system


def test_pipeline_primary_with_fallback():
    """AgentSystem.run should use Pipeline as primary, ModelRouter as fallback."""
    system = boot_agent_system(use_mock_llm=True)
    source = inspect.getsource(type(system).run)
    # Pipeline is the primary path (orchestrator removed in cleanup)
    assert "pipeline" in source.lower(), (
        "run() should reference pipeline as primary routing path"
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
