"""Tests for dead code attribute fixes in AgentLoop."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from sage.agent_loop import AgentLoop


class FakeConfig:
    """Minimal AgentConfig stub."""
    def __init__(self):
        self.name = "test-agent"
        self.system_prompt = "You are a test agent."
        self.max_steps = 10
        self.validation_level = 2
        self.tools = []
        self.llm = MagicMock()
        self.llm.model = "test-model"


def test_dead_code_attributes_initialized():
    """Verify _last_avr_iterations, _last_error, _s3_degraded exist after init."""
    config = FakeConfig()
    mock_llm = AsyncMock()
    loop = AgentLoop(config=config, llm_provider=mock_llm)

    assert hasattr(loop, '_last_avr_iterations')
    assert loop._last_avr_iterations == 0
    assert hasattr(loop, '_last_error')
    assert loop._last_error is None
    assert hasattr(loop, '_s3_degraded')
    assert loop._s3_degraded is False
