from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class FakeNode:
    def __init__(
        self,
        role: str,
        model_id: str,
        system: int,
        required_capabilities: list[str] | None = None,
    ) -> None:
        self.role = role
        self.model_id = model_id
        self.system = system
        self.required_capabilities = required_capabilities or []


class FakeGraph:
    def __init__(self, nodes: list[FakeNode]) -> None:
        self._nodes = nodes

    def node_count(self) -> int:
        return len(self._nodes)

    def get_node(self, idx: int) -> FakeNode:
        return self._nodes[idx]


class FakeExecutor:
    def __init__(self, ready_sequence: list[list[int]]) -> None:
        self._batches = list(ready_sequence)
        self._batch_idx = 0

    def next_ready(self, graph: FakeGraph) -> list[int]:
        if self._batch_idx >= len(self._batches):
            return []
        batch = self._batches[self._batch_idx]
        self._batch_idx += 1
        return batch

    def mark_completed(self, idx: int) -> None:
        pass

    def is_done(self) -> bool:
        return self._batch_idx >= len(self._batches)


class CountingController:
    def __init__(self) -> None:
        self.calls: dict[int, int] = {}

    def evaluate_and_decide(
        self,
        node_idx,
        result,
        task,
        topology,
        ctx,
        parallel_outputs=None,
    ):
        self.calls[node_idx] = self.calls.get(node_idx, 0) + 1
        return SimpleNamespace(action="continue", target_node=node_idx)


class FakeAgentLoop:
    def __init__(self) -> None:
        self.tool_call_count = 0
        self.tool_turn_count = 0
        self.total_cost_usd = 0.0
        self.executed_commands: list[str] = []

    async def run(self, task: str) -> str:
        return "node output"


def _agent_loop_factory(**_kwargs: object) -> FakeAgentLoop:
    return FakeAgentLoop()


@pytest.mark.asyncio
async def test_controller_evaluated_once_per_node_agent_loop_path() -> None:
    from sage.topology.runner import TopologyRunner

    graph = FakeGraph(
        nodes=[
            FakeNode(role="actor", model_id="gemini-2.5-flash", system=1),
            FakeNode(role="verifier", model_id="gemini-2.5-flash", system=1),
        ]
    )
    executor = FakeExecutor(ready_sequence=[[0], [1]])
    controller = CountingController()

    runner = TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=MagicMock(),
        agent_loop_factory=_agent_loop_factory,
        controller=controller,
    )

    assert runner._agent_loop_factory is not None

    await runner.run("task")

    assert controller.calls == {0: 1, 1: 1}


@pytest.mark.asyncio
async def test_controller_evaluated_once_with_parallel_nodes() -> None:
    from sage.topology.runner import TopologyRunner

    graph = FakeGraph(
        nodes=[
            FakeNode(role="worker-a", model_id="gemini-2.5-flash", system=1),
            FakeNode(role="worker-b", model_id="gemini-2.5-flash", system=1),
            FakeNode(role="synthesizer", model_id="gemini-2.5-flash", system=1),
        ]
    )
    executor = FakeExecutor(ready_sequence=[[0, 1], [2]])
    controller = CountingController()

    runner = TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=MagicMock(),
        agent_loop_factory=_agent_loop_factory,
        controller=controller,
    )

    assert runner._agent_loop_factory is not None

    await runner.run("task")

    assert controller.calls == {0: 1, 1: 1, 2: 1}
