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
