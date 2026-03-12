"""Tests for TopologyRunner — real multi-agent topology execution."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


class FakeNode:
    """Minimal node matching TopologyGraph.get_node() return type."""
    def __init__(self, role: str, model_id: str, system: int, required_capabilities: list[str] | None = None):
        self.role = role
        self.model_id = model_id
        self.system = system
        self.required_capabilities = required_capabilities or []


class FakeGraph:
    """Minimal TopologyGraph stub for testing without Rust."""
    def __init__(self, nodes: list[FakeNode]):
        self._nodes = nodes

    def node_count(self) -> int:
        return len(self._nodes)

    def get_node(self, idx: int) -> FakeNode:
        return self._nodes[idx]


class FakeExecutor:
    """Minimal TopologyExecutor stub — yields batches of ready node indices."""
    def __init__(self, order: list[list[int]]):
        self._batches = list(order)
        self._batch_idx = 0

    def next_ready(self, graph) -> list[int]:
        if self._batch_idx >= len(self._batches):
            return []
        batch = self._batches[self._batch_idx]
        self._batch_idx += 1
        return batch

    def mark_completed(self, idx: int) -> None:
        pass

    def is_done(self) -> bool:
        return self._batch_idx >= len(self._batches)


@pytest.mark.asyncio
async def test_sequential_two_node_topology():
    """Two-node sequential: node0 output feeds node1 as context."""
    from sage.topology.runner import TopologyRunner

    graph = FakeGraph(nodes=[
        FakeNode(role="researcher", model_id="gemini-2.5-flash", system=1),
        FakeNode(role="writer", model_id="gemini-2.5-flash", system=1),
    ])
    executor = FakeExecutor(order=[[0], [1]])

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(side_effect=[
        MagicMock(content="Research findings about X"),
        MagicMock(content="Final article about X based on research"),
    ])

    runner = TopologyRunner(graph, executor, llm_provider=mock_provider)
    result = await runner.run("Write about X")

    assert "Final article" in result
    assert mock_provider.generate.call_count == 2
    # Second call should include first node's output in messages
    second_call_msgs = mock_provider.generate.call_args_list[1].kwargs.get(
        "messages", mock_provider.generate.call_args_list[1].args[0] if mock_provider.generate.call_args_list[1].args else []
    )
    msg_texts = " ".join(m.content for m in second_call_msgs if hasattr(m, "content"))
    assert "Research findings" in msg_texts


@pytest.mark.asyncio
async def test_parallel_three_node_topology():
    """Two parallel workers + one aggregator: workers run concurrently, aggregator sees both outputs."""
    from sage.topology.runner import TopologyRunner

    graph = FakeGraph(nodes=[
        FakeNode(role="analyst-A", model_id="gemini-2.5-flash", system=1),
        FakeNode(role="analyst-B", model_id="gemini-2.5-flash", system=1),
        FakeNode(role="synthesizer", model_id="gemini-2.5-flash", system=2),
    ])
    # Batch [0, 1] runs in parallel; batch [2] runs after both complete
    executor = FakeExecutor(order=[[0, 1], [2]])

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(side_effect=[
        MagicMock(content="Analysis from A: market is bullish"),
        MagicMock(content="Analysis from B: risk is low"),
        MagicMock(content="Synthesis: combined view is positive"),
    ])

    runner = TopologyRunner(graph, executor, llm_provider=mock_provider)
    result = await runner.run("Analyse the market")

    # Final result is synthesizer's output
    assert "Synthesis" in result
    # All 3 LLM calls made
    assert mock_provider.generate.call_count == 3
    # Synthesizer (3rd call, index 2) must have received BOTH analysts' outputs in context
    third_call_msgs = mock_provider.generate.call_args_list[2].kwargs.get(
        "messages",
        mock_provider.generate.call_args_list[2].args[0] if mock_provider.generate.call_args_list[2].args else [],
    )
    msg_texts = " ".join(m.content for m in third_call_msgs if hasattr(m, "content"))
    assert "Analysis from A" in msg_texts
    assert "Analysis from B" in msg_texts


@pytest.mark.asyncio
async def test_single_node_returns_direct():
    """Single-node topology: result equals the direct LLM output, exactly 1 call made."""
    from sage.topology.runner import TopologyRunner

    graph = FakeGraph(nodes=[
        FakeNode(role="solo-agent", model_id="gemini-2.5-flash", system=1),
    ])
    executor = FakeExecutor(order=[[0]])

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(side_effect=[
        MagicMock(content="Direct answer from solo agent"),
    ])

    runner = TopologyRunner(graph, executor, llm_provider=mock_provider)
    result = await runner.run("What is 2+2?")

    assert result == "Direct answer from solo agent"
    assert mock_provider.generate.call_count == 1


@pytest.mark.asyncio
async def test_empty_topology_returns_empty():
    """Empty topology: result is empty string, 0 LLM calls made."""
    from sage.topology.runner import TopologyRunner

    graph = FakeGraph(nodes=[])
    executor = FakeExecutor(order=[])

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock()

    runner = TopologyRunner(graph, executor, llm_provider=mock_provider)
    result = await runner.run("Any task")

    assert result == ""
    assert mock_provider.generate.call_count == 0
