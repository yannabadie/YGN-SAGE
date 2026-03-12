"""Tests for S3 hard gating with CEGAR repair and S2 degradation."""
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
        self.validation_level = 3  # S3
        self.tools = []
        self.llm = MagicMock()
        self.llm.model = "test-model"


class FakePRM:
    """ProcessRewardModel that fails N times then passes."""
    def __init__(self, fail_count: int = 999):
        self._calls = 0
        self._fail_count = fail_count
        self.kg = MagicMock()
        self.kg._last_invariant_feedback = ["clause X failed: x > 0 not satisfied"]

    def calculate_r_path(self, content: str) -> tuple[float, dict]:
        self._calls += 1
        if self._calls <= self._fail_count:
            return -1.0, {"error": "No <think> blocks found. System 3 reasoning required."}
        return 0.8, {"total_steps": 1, "step_scores": [0.8], "verifiable_ratio": 1.0}


@pytest.mark.asyncio
async def test_s3_cegar_repair_returns_none_when_prm_fails():
    """CEGAR repair with always-failing PRM returns None (triggers S2 degradation)."""
    config = FakeConfig()
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(
        return_value=MagicMock(
            content="repaired but still no valid assertions",
            usage={},
        )
    )

    loop = AgentLoop(config=config, llm_provider=mock_llm)
    loop.prm = FakePRM(fail_count=999)  # Always fails

    result = await loop._cegar_repair("bad content", "error: no assertions", ["clause X failed"])

    assert result is None  # Repair failed — should return None
    assert mock_llm.generate.call_count == 1  # Tried one repair call


@pytest.mark.asyncio
async def test_s3_cegar_repair_succeeds_when_prm_passes():
    """CEGAR repair with passing PRM returns repaired content."""
    config = FakeConfig()
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(
        return_value=MagicMock(
            content="<think>assert arithmetic(2+2, 4)</think>\nThe answer is 4.",
            usage={},
        )
    )

    loop = AgentLoop(config=config, llm_provider=mock_llm)
    loop.prm = FakePRM(fail_count=0)  # Passes immediately

    result = await loop._cegar_repair("bad content", "error: no assertions", [])

    assert result is not None
    assert "<think>" in result
    assert mock_llm.generate.call_count == 1


@pytest.mark.asyncio
async def test_s3_degraded_flag_prevents_re_escalation():
    """Once S3 degrades to S2, the _s3_degraded flag prevents ping-pong back to S3."""
    config = FakeConfig()
    config.validation_level = 3
    mock_llm = AsyncMock()

    loop = AgentLoop(config=config, llm_provider=mock_llm)

    # Verify the flag exists and defaults to False
    assert hasattr(loop, '_s3_degraded')
    assert loop._s3_degraded is False
