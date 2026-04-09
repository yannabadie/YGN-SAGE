"""Tests for speculative S1+S2 zone detection.

Speculative zone detection was originally in the legacy system.run() path.
After the system.run() simplification, mock mode bypasses the pipeline
entirely (H9), so speculative zone detection now lives in the pipeline's
_stage_classify() (Rust SystemRouter).

These tests verify that mock mode completes without error (the speculative
log is no longer emitted from system.run() in mock mode).

TODO: Add pipeline-level tests that exercise _stage_classify() speculative
zone detection with non-mock providers.
"""
import logging

import pytest
from unittest.mock import AsyncMock

from sage.boot import boot_agent_system
from sage.strategy.metacognition import CognitiveProfile


@pytest.mark.asyncio
async def test_speculative_zone_mock_bypass(caplog):
    """Mock mode bypasses pipeline — no speculative zone log from system.run().

    Speculative zone detection is now in pipeline._stage_classify(), which
    mock mode does not invoke (H9: mock bypass).
    """
    system = boot_agent_system(use_mock_llm=True)
    profile = CognitiveProfile(complexity=0.45, uncertainty=0.5, tool_required=False)
    system.metacognition.assess_complexity_async = AsyncMock(return_value=profile)

    with caplog.at_level(logging.INFO, logger="sage.boot"):
        result = await system.run("Explain quantum computing")

    assert result is not None
    # Mock mode takes the direct agent_loop path — no speculative zone log
    assert system._last_execution_path == "mock"


@pytest.mark.asyncio
async def test_speculative_zone_boundary_low_mock_bypass(caplog):
    """Mock mode at low boundary (0.35) completes without pipeline routing."""
    system = boot_agent_system(use_mock_llm=True)
    profile = CognitiveProfile(complexity=0.35, uncertainty=0.2, tool_required=False)
    system.metacognition.assess_complexity_async = AsyncMock(return_value=profile)

    with caplog.at_level(logging.INFO, logger="sage.boot"):
        result = await system.run("Simple task")

    assert result is not None
    assert system._last_execution_path == "mock"


@pytest.mark.asyncio
async def test_speculative_zone_boundary_high_mock_bypass(caplog):
    """Mock mode at high boundary (0.55) completes without pipeline routing."""
    system = boot_agent_system(use_mock_llm=True)
    profile = CognitiveProfile(complexity=0.55, uncertainty=0.3, tool_required=False)
    system.metacognition.assess_complexity_async = AsyncMock(return_value=profile)

    with caplog.at_level(logging.INFO, logger="sage.boot"):
        result = await system.run("Moderate task")

    assert result is not None
    assert system._last_execution_path == "mock"


@pytest.mark.asyncio
async def test_non_speculative_low_complexity_no_log(caplog):
    """When complexity is clearly S1 (low), no speculative log."""
    system = boot_agent_system(use_mock_llm=True)
    profile = CognitiveProfile(complexity=0.1, uncertainty=0.1, tool_required=False)
    system.metacognition.assess_complexity_async = AsyncMock(return_value=profile)

    with caplog.at_level(logging.INFO, logger="sage.boot"):
        result = await system.run("Hello")

    assert result is not None
    assert not any("Speculative zone" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_non_speculative_high_complexity_no_log(caplog):
    """When complexity is clearly S3 (high), no speculative log."""
    system = boot_agent_system(use_mock_llm=True)
    profile = CognitiveProfile(complexity=0.9, uncertainty=0.8, tool_required=True)
    system.metacognition.assess_complexity_async = AsyncMock(return_value=profile)

    with caplog.at_level(logging.INFO, logger="sage.boot"):
        result = await system.run("Debug a complex distributed system crash")

    assert result is not None
    assert not any("Speculative zone" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_speculative_zone_not_triggered_for_s3(caplog):
    """When decision.system == 3, speculative zone should NOT trigger even if complexity is in range."""
    system = boot_agent_system(use_mock_llm=True)
    # complexity in range but uncertainty pushes to S3
    profile = CognitiveProfile(complexity=0.50, uncertainty=0.9, tool_required=False)
    system.metacognition.assess_complexity_async = AsyncMock(return_value=profile)

    with caplog.at_level(logging.INFO, logger="sage.boot"):
        result = await system.run("Highly uncertain task")

    assert result is not None
    assert not any("Speculative zone" in r.message for r in caplog.records)
