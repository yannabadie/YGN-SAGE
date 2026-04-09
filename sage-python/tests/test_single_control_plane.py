"""Test that routing uses a single control plane: the pipeline.

After the system.run() simplification, the control plane is:
- Mock mode: direct agent_loop (H9 bypass)
- Non-mock: CognitiveOrchestrationPipeline (5-stage)
- Fallback: direct agent_loop if pipeline not initialized

No legacy ModelRouter or orchestrator — pipeline IS the control plane.
"""
import inspect

import pytest

from sage.boot import boot_agent_system


def test_pipeline_is_primary_control_plane():
    """AgentSystem.run should use Pipeline as the primary execution path."""
    system = boot_agent_system(use_mock_llm=True)
    source = inspect.getsource(type(system).run)
    # Pipeline is the primary path for non-mock mode
    assert "pipeline" in source.lower(), (
        "run() should reference pipeline as primary routing path"
    )
    # No legacy ModelRouter — pipeline handles routing via its _stage_classify
    assert "ModelRouter.get_config" not in source, (
        "run() should not reference legacy ModelRouter — pipeline handles routing"
    )
    # Must not call registry.refresh in run() — that belongs in boot
    assert "registry.refresh" not in source, (
        "run() should not call registry.refresh — that belongs at boot time"
    )


def test_registry_field_still_available():
    """Registry field still exists for pipeline and boot-time use."""
    from sage.boot import AgentSystem
    import dataclasses

    field_names = [f.name for f in dataclasses.fields(AgentSystem)]
    # registry is retained for pipeline's model assignment stage
    assert "registry" in field_names, "registry field should be retained"
    # pipeline field is the control plane
    assert "pipeline" in field_names, "pipeline field should be present"
