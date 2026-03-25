"""Tests for execution path tracking.

Issue A audit fix: AgentSystem must track and log which execution path
was taken (pipeline, pipeline_fallback_legacy, or legacy).
"""
import pytest
from sage.boot import AgentSystem


def test_last_execution_path_attribute_exists():
    """AgentSystem must have _last_execution_path attribute."""
    assert hasattr(AgentSystem, "_last_execution_path")


def test_last_execution_path_default_empty():
    """_last_execution_path should start empty."""
    assert AgentSystem._last_execution_path == ""


@pytest.mark.asyncio
async def test_mock_mode_sets_legacy_path():
    """Mock provider should use legacy path, not pipeline."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)
    result = await system.run("test task")
    assert system._last_execution_path in ("legacy", "pipeline", "pipeline_fallback_legacy"), (
        f"Expected a valid path, got '{system._last_execution_path}'"
    )
    # Mock mode should use legacy (pipeline skips mock provider)
    assert system._last_execution_path == "legacy", (
        f"Mock mode should use legacy path, got '{system._last_execution_path}'"
    )
