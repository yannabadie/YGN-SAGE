"""Tests for pipeline bypass path using agent_loop.run().

Phase 1 of unified entry point: single-agent bypass calls agent_loop
instead of provider.generate(), gaining tools + validation + guardrails.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


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
async def test_bypass_calls_agent_loop_run():
    """Stage 4 bypass should call agent_loop.run() instead of provider.generate()."""
    mock_loop = MagicMock()
    mock_loop.run = AsyncMock(return_value="agent_loop result")
    mock_loop.total_cost_usd = 0.001
    mock_loop._skip_routing = False
    mock_loop._current_topology = None
    mock_loop.config = MagicMock()
    mock_loop.config.validation_level = 1
    mock_loop.sandbox_manager = None

    pipeline = _make_pipeline(agent_loop=mock_loop)

    ctx = PipelineContext(task="Write hello world", system=2)
    ctx.topology = None  # bypass mode

    result_ctx = await pipeline._stage_execute(ctx)

    mock_loop.run.assert_called_once_with("Write hello world")
    assert result_ctx.result == "agent_loop result"
    assert result_ctx.cost == 0.001


@pytest.mark.asyncio
async def test_bypass_sets_skip_routing():
    """H1 fix: _skip_routing must be True during agent_loop.run(), restored after."""
    captured_skip = {}

    async def _capture_run(task):
        captured_skip["during"] = mock_loop._skip_routing
        return "result"

    mock_loop = MagicMock()
    mock_loop.run = _capture_run
    mock_loop.total_cost_usd = 0.0
    mock_loop._skip_routing = False
    mock_loop._current_topology = None
    mock_loop.config = MagicMock()
    mock_loop.config.validation_level = 1
    mock_loop.sandbox_manager = None

    pipeline = _make_pipeline(agent_loop=mock_loop)
    ctx = PipelineContext(task="test", system=1)
    ctx.topology = None

    await pipeline._stage_execute(ctx)

    assert captured_skip["during"] is True, "skip_routing must be True during run"
    assert mock_loop._skip_routing is False, "skip_routing must be restored after run"


@pytest.mark.asyncio
async def test_bypass_clears_topology():
    """H4 fix: _current_topology must be None during agent_loop.run()."""
    captured_topo = {}

    async def _capture_run(task):
        captured_topo["during"] = mock_loop._current_topology
        return "result"

    mock_loop = MagicMock()
    mock_loop.run = _capture_run
    mock_loop.total_cost_usd = 0.0
    mock_loop._skip_routing = False
    mock_loop._current_topology = "stale_topology"
    mock_loop.config = MagicMock()
    mock_loop.config.validation_level = 1
    mock_loop.sandbox_manager = None

    pipeline = _make_pipeline(agent_loop=mock_loop)
    ctx = PipelineContext(task="test", system=2)
    ctx.topology = None

    await pipeline._stage_execute(ctx)

    assert captured_topo["during"] is None, "topology must be cleared during run"


@pytest.mark.asyncio
async def test_bypass_sets_validation_level():
    """Validation level should match system classification from routing.

    A0a (2026-04-23) restores config.validation_level to its pre-bypass
    value in the `finally` — so this test captures the mutated value
    DURING the run via the same `_capture_run` idiom as the
    sibling tests in this file.
    """
    captured = {}

    async def _capture_run(task):
        captured["during"] = mock_loop.config.validation_level
        return "result"

    mock_loop = MagicMock()
    mock_loop.run = _capture_run
    mock_loop.total_cost_usd = 0.0
    mock_loop._skip_routing = False
    mock_loop._current_topology = None
    mock_loop.config = MagicMock()
    mock_loop.config.validation_level = 1
    mock_loop.sandbox_manager = MagicMock()  # sandbox available

    pipeline = _make_pipeline(agent_loop=mock_loop)
    ctx = PipelineContext(task="test", system=3)
    ctx.topology = None

    await pipeline._stage_execute(ctx)

    assert captured["during"] == 3, (
        "bypass path must set config.validation_level=3 for system=3 "
        f"during run; got {captured.get('during')!r}"
    )
    # A0a restoration: post-run the value returns to pre-bypass (1).
    assert mock_loop.config.validation_level == 1, (
        f"A0a: validation_level must be restored to 1 after run; "
        f"got {mock_loop.config.validation_level!r}"
    )


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

    result_ctx = await pipeline._stage_execute(ctx)

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
