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
