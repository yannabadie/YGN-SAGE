"""Tests for pipeline bypass path constructor + fallback paths.

Phase 1 of unified entry point: single-agent bypass calls agent_loop
instead of provider.generate(), gaining tools + validation + guardrails.

The 4 legacy "bypass mutates singleton X / restores X" tests were retired
in cycle-12 P6-A Phase B (commit 7e20372e). The singleton is no longer
mutated — each bypass run gets a fresh AgentLoop from
`create_bypass_agent_loop()`. The structural-isolation contract that
replaces them lives in `test_pipeline_bypass_structural_isolation.py`
(singleton fields UNCHANGED before/after, concurrent bypass independence,
recursive no-deadlock).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext
from sage.pipeline_v2.execute import execute


def _make_pipeline(agent_loop=None):
    """Create a minimal pipeline for testing."""
    return CognitiveOrchestrationPipeline(
        router=MagicMock(),
        engine=None,
        assigner=MagicMock(),
        provider_pool=MagicMock(),
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        agent_loop=agent_loop,
    )


def test_pipeline_accepts_agent_loop_param():
    """Pipeline constructor should accept and store agent_loop."""
    mock_loop = MagicMock()
    pipeline = _make_pipeline(agent_loop=mock_loop)
    assert pipeline._agent_loop is mock_loop


def test_pipeline_agent_loop_defaults_none():
    """Pipeline without agent_loop should default to None."""
    pipeline = _make_pipeline()
    assert pipeline._agent_loop is None


@pytest.mark.asyncio
async def test_bypass_without_agent_loop_uses_provider_loop():
    """When agent_loop is None, bypass should fall back to provider.generate() loop."""
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "provider result"
    mock_response.tool_calls = None
    mock_provider.generate = AsyncMock(return_value=mock_response)

    pipeline = _make_pipeline(agent_loop=None)
    pipeline.llm_provider = mock_provider

    ctx = PipelineContext(task="simple question", system=1)
    ctx.topology = None

    result_ctx = await execute(pipeline, ctx)

    mock_provider.generate.assert_called_once()
    assert result_ctx.result == "provider result"


@pytest.mark.asyncio
async def test_system_run_mock_bypass():
    """Mock mode should bypass pipeline and call agent_loop.run() directly."""
    from sage.boot import boot_agent_system

    system = boot_agent_system(use_mock_llm=True)
    result = await system.run("test task")
    assert system._last_execution_path == "mock"
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_system_run_pipeline_path():
    """Non-mock mode should use pipeline (when pipeline is available)."""
    from sage.boot import boot_agent_system

    system = boot_agent_system(use_mock_llm=True)
    # Simulate non-mock with pipeline available
    system.agent_loop.config.llm.provider = "test"
    system.pipeline = MagicMock()
    system.pipeline.run = AsyncMock(return_value="pipeline result")

    result = await system.run("test task")
    assert system._last_execution_path == "pipeline"
    assert result == "pipeline result"
