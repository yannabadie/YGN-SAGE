from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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

    def get_predecessors(self, idx: int) -> list[int]:
        return list(range(idx))


class FakeExecutor:
    def __init__(self, ready_sequence: list[list[int]]) -> None:
        self._batches = list(ready_sequence)
        self._batch_idx = 0
        self.skipped: list[int] = []
        self.opened: list[tuple[int, int]] = []
        self.reset: list[int] = []

    def next_ready(self, graph: FakeGraph) -> list[int]:
        if self._batch_idx >= len(self._batches):
            return []
        batch = self._batches[self._batch_idx]
        self._batch_idx += 1
        return batch

    def mark_completed(self, idx: int) -> None:
        pass

    def mark_skipped(self, idx: int) -> None:
        self.skipped.append(idx)

    def is_done(self) -> bool:
        return self._batch_idx >= len(self._batches)

    def open_gate(self, graph: FakeGraph, source: int, target: int) -> None:
        self.opened.append((source, target))

    def reset_node(self, idx: int) -> None:
        self.reset.append(idx)


class CountingController:
    """Default controller returning continue, optionally scripted per node."""

    def __init__(self, script: dict[int, Any] | None = None) -> None:
        self.calls: dict[int, int] = {}
        self.decisions: list[Any] = []
        self.script = script or {}

    def evaluate_and_decide(
        self,
        node_idx: int,
        result: str,
        task: str,
        topology: FakeGraph,
        ctx: dict[str, Any],
        parallel_outputs: list[str] | None = None,
    ) -> Any:
        self.calls[node_idx] = self.calls.get(node_idx, 0) + 1
        decision = self.script.get(
            node_idx,
            SimpleNamespace(action="continue", target_node=node_idx),
        )
        self.decisions.append(decision)
        return decision


class FakeAgentLoop:
    def __init__(self, output: str = "node output", delay: float = 0.0) -> None:
        self.tool_call_count = 0
        self.tool_turn_count = 0
        self.total_cost_usd = 0.0
        self.executed_commands: list[str] = []
        self._output = output
        self._delay = delay

    async def run(self, task: str) -> str:
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._output


def _factory(delay: float = 0.0):
    return lambda **_kwargs: FakeAgentLoop(delay=delay)


def _make_runner(
    graph: FakeGraph,
    ready_sequence: list[list[int]],
    controller: CountingController | None = None,
    factory_delay: float = 0.0,
):
    from sage.topology.runner import TopologyRunner

    return TopologyRunner(
        graph=graph,
        executor=FakeExecutor(ready_sequence),
        llm_provider=MagicMock(),
        agent_loop_factory=_factory(delay=factory_delay),
        controller=controller,
    )


@pytest.mark.asyncio
async def test_run_and_run_traced_same_final_output() -> None:
    graph = FakeGraph(
        nodes=[
            FakeNode(role="planner", model_id="gemini", system=1),
            FakeNode(role="worker", model_id="gemini", system=1),
        ]
    )
    ready = [[0], [1]]

    run_runner = _make_runner(graph, ready)
    final_run = await run_runner.run("task")

    traced_runner = _make_runner(graph, ready)
    traces = await traced_runner.run_traced("task")

    assert final_run == traces[-1]["output"]


@pytest.mark.asyncio
async def test_run_stream_emits_same_node_order_as_run_traced() -> None:
    graph = FakeGraph(
        nodes=[
            FakeNode(role="a", model_id="gemini", system=1),
            FakeNode(role="b", model_id="gemini", system=1),
            FakeNode(role="c", model_id="gemini", system=1),
        ]
    )
    ready = [[0, 1], [2]]

    traced_runner = _make_runner(graph, ready)
    traces = await traced_runner.run_traced("task")
    traced_order = [trace["node_idx"] for trace in traces]

    stream_runner = _make_runner(graph, ready)
    stream_order: list[int] = []
    async for event in stream_runner.run_stream("task"):
        if event.get("type") == "node_done":
            stream_order.append(event["node_idx"])

    assert stream_order == traced_order


@pytest.mark.asyncio
async def test_controller_decision_count_identical_across_modes() -> None:
    graph = FakeGraph(
        nodes=[
            FakeNode(role="a", model_id="gemini", system=1),
            FakeNode(role="b", model_id="gemini", system=1),
            FakeNode(role="c", model_id="gemini", system=1),
        ]
    )
    ready = [[0, 1], [2]]
    expected = {0: 1, 1: 1, 2: 1}

    run_controller = CountingController()
    run_runner = _make_runner(graph, ready, controller=run_controller)
    await run_runner.run("task")
    assert run_controller.calls == expected, f"run(): {run_controller.calls}"

    traced_controller = CountingController()
    traced_runner = _make_runner(graph, ready, controller=traced_controller)
    await traced_runner.run_traced("task")
    assert traced_controller.calls == expected, f"run_traced(): {traced_controller.calls}"

    stream_controller = CountingController()
    stream_runner = _make_runner(graph, ready, controller=stream_controller)
    async for _ in stream_runner.run_stream("task"):
        pass
    assert stream_controller.calls == expected, f"run_stream(): {stream_controller.calls}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    [
        "upgrade_model",
        "spawn_subagent",
        "reroute_topology",
        "prune_node",
        "open_gate",
    ],
)
async def test_run_traced_handles_all_5_controller_actions(action: str) -> None:
    graph = FakeGraph(
        nodes=[
            FakeNode(role="actor", model_id="gemini", system=1),
            FakeNode(role="verifier", model_id="gemini", system=1),
        ]
    )
    ready = [[0], [1]]

    if action == "upgrade_model":
        decision = SimpleNamespace(
            action="upgrade_model",
            target_node=0,
            reason="quality low",
            new_model_id="upgraded",
            invariant_feedback="",
        )
    elif action == "spawn_subagent":
        decision = SimpleNamespace(action="spawn_subagent", target_node=0, reason="spawn")
    elif action == "reroute_topology":
        decision = SimpleNamespace(action="reroute_topology", target_node=0, reason="bad")
    elif action == "prune_node":
        decision = SimpleNamespace(action="prune_node", target_node=1, reason="useless")
    else:
        decision = SimpleNamespace(
            action="open_gate",
            target_node=0,
            gate_source=0,
            gate_target=1,
            reason="iterate",
        )

    controller = CountingController(script={0: decision})

    if action == "reroute_topology":
        runner = _make_runner(graph, ready, controller=controller)
        events = []
        async for event in runner._run_core("task"):
            events.append(event)
            if type(event).__name__ == "_RerouteEvent":
                break
        type_names = [type(event).__name__ for event in events]
        assert "_RerouteEvent" in type_names, f"events: {type_names}"
    else:
        runner = _make_runner(graph, ready, controller=controller)
        runner._retry_with_upgrade = AsyncMock(return_value="upgraded output")
        runner._spawn_sub = AsyncMock(return_value=None)

        traces = await runner.run_traced("task")

        if action == "upgrade_model":
            runner._retry_with_upgrade.assert_called_once()
        elif action == "spawn_subagent":
            runner._spawn_sub.assert_called_once()
        elif action == "prune_node":
            assert 1 in runner.executor.skipped, f"executor.skipped: {runner.executor.skipped}"
        elif action == "open_gate":
            assert runner.executor.opened == [(0, 1)]
            assert runner.executor.reset == [1]
            assert len(traces) >= 1


@pytest.mark.asyncio
async def test_run_stream_handles_prune_and_spawn() -> None:
    graph = FakeGraph(
        nodes=[
            FakeNode(role="actor", model_id="gemini", system=1),
            FakeNode(role="victim", model_id="gemini", system=1),
        ]
    )
    ready = [[0], [1]]

    prune_decision = SimpleNamespace(action="prune_node", target_node=1, reason="prune")
    prune_controller = CountingController(script={0: prune_decision})
    prune_runner = _make_runner(graph, ready, controller=prune_controller)
    async for _ in prune_runner.run_stream("task"):
        pass
    assert 1 in prune_runner.executor.skipped, (
        f"executor.skipped: {prune_runner.executor.skipped}"
    )

    spawn_decision = SimpleNamespace(action="spawn_subagent", target_node=0, reason="spawn")
    spawn_controller = CountingController(script={0: spawn_decision})
    spawn_runner = _make_runner(graph, ready, controller=spawn_controller)
    spawn_runner._spawn_sub = AsyncMock(return_value=None)
    async for _ in spawn_runner.run_stream("task"):
        pass
    spawn_runner._spawn_sub.assert_called_once()


@pytest.mark.asyncio
async def test_run_traced_runs_parallel_batches_in_parallel() -> None:
    graph = FakeGraph(
        nodes=[
            FakeNode(role="a", model_id="gemini", system=1),
            FakeNode(role="b", model_id="gemini", system=1),
            FakeNode(role="c", model_id="gemini", system=1),
        ]
    )
    ready = [[0, 1, 2]]
    runner = _make_runner(graph, ready, factory_delay=0.1)

    import time

    t0 = time.monotonic()
    traces = await runner.run_traced("task")
    elapsed = time.monotonic() - t0

    assert len(traces) == 3, f"expected 3 traces, got {len(traces)}"
    assert elapsed < 0.25, f"elapsed {elapsed:.3f}s; run_traced should be parallel"
