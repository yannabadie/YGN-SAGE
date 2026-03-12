"""Integration test: verify all Phase 0 audit fixes work together."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from sage.orchestrator import ModelAgent
from sage.quality_estimator import QualityEstimator


class FakeModel:
    def __init__(self, id: str, provider: str = "google", cost_input: float = 0.001):
        self.id = id
        self.provider = provider
        self.cost_input = cost_input
        self.cost_output = cost_input


class FakeRegistry:
    def __init__(self, models: list[FakeModel]):
        self._models = models

    def list_available(self) -> list[FakeModel]:
        return self._models


@pytest.mark.asyncio
async def test_quality_estimator_signal5_fires():
    """QualityEstimator Signal 5 (AVR convergence) now works with avr_iterations > 0."""
    score_with_avr = QualityEstimator.estimate(
        task="implement sort",
        result="def sort(arr): return sorted(arr)",
        avr_iterations=2,
    )
    score_without_avr = QualityEstimator.estimate(
        task="implement sort",
        result="def sort(arr): return sorted(arr)",
        avr_iterations=0,
    )
    # Signal 5 should add 0.15 for avr_iterations <= 2
    assert score_with_avr > score_without_avr


@pytest.mark.asyncio
async def test_quality_cascade_cost_savings():
    """Verify quality cascade saves money by NOT escalating when cheap model is good enough."""
    cheap = FakeModel("cheap", cost_input=0.0003)
    expensive = FakeModel("expensive", cost_input=0.01)
    registry = FakeRegistry([cheap, expensive])

    agent = ModelAgent(
        name="test",
        model=cheap,
        registry=registry,
        quality_threshold=0.5,
    )

    good_response = (
        "Here is the implementation:\n"
        "```python\ndef fibonacci(n):\n"
        "    if n <= 1: return n\n"
        "    return fibonacci(n-1) + fibonacci(n-2)\n```\n"
        "This correctly handles edge cases."
    )

    agent._call_provider = AsyncMock(return_value=good_response)
    result = await agent.run("implement fibonacci")

    # Cheap model was good enough — no escalation
    assert agent._call_provider.call_count == 1
    quality = QualityEstimator.estimate("implement fibonacci", result)
    assert quality >= 0.5  # Passes threshold


def test_topology_runner_imports():
    """TopologyRunner module exists and is importable."""
    from sage.topology.runner import TopologyRunner, EDGE_CONTROL, EDGE_MESSAGE, EDGE_STATE
    assert EDGE_CONTROL == 0
    assert EDGE_MESSAGE == 1
    assert EDGE_STATE == 2


def test_agent_loop_has_cegar_repair():
    """AgentLoop._cegar_repair method exists after Task 6."""
    from sage.agent_loop import AgentLoop
    assert hasattr(AgentLoop, '_cegar_repair')
    assert callable(AgentLoop._cegar_repair)


def test_agent_loop_has_run_topology():
    """AgentLoop._run_topology method exists after Task 3."""
    from sage.agent_loop import AgentLoop
    assert hasattr(AgentLoop, '_run_topology')
    assert callable(AgentLoop._run_topology)
