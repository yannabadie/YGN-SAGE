"""R5 RuntimeEventLog v0 acceptance tests."""
from __future__ import annotations

import asyncio
import json
import pathlib
import shutil
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from sage.runtime.event_log import EventLogUnavailable, RuntimeEventLog, SCHEMA_VERSION


@pytest.fixture
def tmp_path() -> pathlib.Path:
    """Local temp fixture.

    The default pytest temp root is ACL-denied on this Windows host. Keep these
    tests isolated under the repo's existing untracked .tmp area.
    """
    path = pathlib.Path(".tmp") / "pytest-runtime-event-log" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


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
    def __init__(self, nodes: list[FakeNode], graph_id: str = "topo-test") -> None:
        self._nodes = nodes
        self.id = graph_id
        self.template_type = "test_template"

    def node_count(self) -> int:
        return len(self._nodes)

    def get_node(self, idx: int) -> FakeNode:
        return self._nodes[idx]

    def get_predecessors(self, idx: int) -> list[int]:
        return list(range(idx))

    def get_edges(self) -> list[tuple[int, int, int]]:
        return [(idx, idx + 1, 0) for idx in range(max(0, len(self._nodes) - 1))]


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
            SimpleNamespace(action="continue", target_node=node_idx, reason=""),
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
    event_log: RuntimeEventLog,
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
        event_log=event_log,
    )


def _read_events(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _event_path(tmp_path: pathlib.Path, run_id: str) -> pathlib.Path:
    return tmp_path / f"{run_id}.jsonl"


def test_one_task_started_one_final_result_per_run(tmp_path: pathlib.Path) -> None:
    run_id = "01TESTULID0000000000000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    log.set_task_text("hello")
    log.emit_task_started("hello")
    log.emit_final_result(
        status="success",
        output="bye",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=1,
    )
    log.close()

    events = _read_events(_event_path(tmp_path, run_id))
    assert SCHEMA_VERSION == "1.0"
    assert len([e for e in events if e["event_type"] == "task_started"]) == 1
    assert len([e for e in events if e["event_type"] == "final_result"]) == 1


def test_seq_strictly_monotonic_per_run(tmp_path: pathlib.Path) -> None:
    run_id = "01TEST00000000000000000002"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    log.emit_task_started("t")
    log.emit_routing_decision(
        routing_source="knn",
        system=1,
        domain="code",
        confidence=0.92,
    )
    log.emit_final_result(
        status="success",
        output="o",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=0,
    )
    log.close()

    events = _read_events(_event_path(tmp_path, run_id))
    assert [e["seq"] for e in events] == list(range(len(events)))


@pytest.mark.asyncio
async def test_every_node_started_has_completed_or_failure(tmp_path: pathlib.Path) -> None:
    run_id = "01TEST00000000000000000003"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)

    graph = FakeGraph(
        nodes=[
            FakeNode(role="first", model_id="gemini", system=1),
            FakeNode(role="second", model_id="gemini", system=1),
        ],
    )
    runner = _make_runner(graph, [[0], [1]], event_log=log)
    runner._execute_node = AsyncMock(
        side_effect=["first output", RuntimeError("provider down")]
    )

    with pytest.raises(RuntimeError, match="provider down"):
        await runner.run("task")
    log.close()

    events = _read_events(_event_path(tmp_path, run_id))
    started = [e["node_id"] for e in events if e["event_type"] == "node_started"]
    completed = {e["node_id"] for e in events if e["event_type"] == "node_completed"}
    failed = {e["node_id"] for e in events if e["event_type"] == "failure"}
    assert started == ["0", "1"]
    assert set(started) <= completed | failed


@pytest.mark.parametrize(
    "action",
    [
        "continue",
        "upgrade_model",
        "spawn_subagent",
        "reroute_topology",
        "prune_node",
        "open_gate",
    ],
)
@pytest.mark.asyncio
async def test_controller_decision_event_records_all_6_actions(
    tmp_path: pathlib.Path,
    action: str,
) -> None:
    run_id = f"01TESTACTION{action[:8].upper():0<12}"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    graph = FakeGraph(
        nodes=[
            FakeNode(role="actor", model_id="gemini", system=1),
            FakeNode(role="verifier", model_id="gemini", system=1),
        ],
    )

    if action == "upgrade_model":
        decision = SimpleNamespace(
            action="upgrade_model",
            target_node=0,
            reason="quality low",
            new_model_id="upgraded",
        )
    elif action == "spawn_subagent":
        decision = SimpleNamespace(action="spawn_subagent", target_node=0, reason="spawn")
    elif action == "reroute_topology":
        decision = SimpleNamespace(action="reroute_topology", target_node=0, reason="bad")
    elif action == "prune_node":
        decision = SimpleNamespace(action="prune_node", target_node=1, reason="useless")
    elif action == "open_gate":
        decision = SimpleNamespace(
            action="open_gate",
            target_node=0,
            gate_source=0,
            gate_target=1,
            reason="iterate",
        )
    else:
        decision = SimpleNamespace(action="continue", target_node=0, reason="")

    runner = _make_runner(
        graph,
        [[0], [1]],
        event_log=log,
        controller=CountingController(script={0: decision}),
    )
    runner._retry_with_upgrade = AsyncMock(return_value="upgraded output")
    runner._spawn_sub = AsyncMock(return_value=None)

    if action == "reroute_topology":
        async for event in runner._run_core("task"):
            if type(event).__name__ == "_RerouteEvent":
                break
    else:
        await runner.run_traced("task")
    log.close()

    events = _read_events(_event_path(tmp_path, run_id))
    actions = [
        e.get("action")
        for e in events
        if e["event_type"] == "controller_decision"
    ]
    assert action in actions


@pytest.mark.asyncio
async def test_run_run_traced_run_stream_emit_same_core_event_sequence(
    tmp_path: pathlib.Path,
) -> None:
    def _graph() -> FakeGraph:
        return FakeGraph(
            nodes=[
                FakeNode(role="a", model_id="gemini", system=1),
                FakeNode(role="b", model_id="gemini", system=1),
                FakeNode(role="c", model_id="gemini", system=1),
            ],
        )

    async def _run_mode(mode: str) -> list[tuple[Any, ...]]:
        run_id = f"01TESTMODE{mode.upper():0<16}"
        log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
        runner = _make_runner(
            _graph(),
            [[0, 1], [2]],
            event_log=log,
            controller=CountingController(),
        )
        if mode == "run":
            await runner.run("task")
        elif mode == "traced":
            await runner.run_traced("task")
        else:
            async for _ in runner.run_stream("task"):
                pass
        log.close()
        events = _read_events(_event_path(tmp_path, run_id))
        return [
            (
                event["event_type"],
                event["source_component"],
                event.get("node_id", ""),
                event.get("node_role", ""),
                event.get("action", ""),
                event.get("model_id", ""),
                event.get("provider_id", ""),
                event.get("attempt", ""),
                event.get("topology_id", ""),
            )
            for event in events
        ]

    assert await _run_mode("run") == await _run_mode("traced") == await _run_mode("stream")


def test_jsonl_sink_emits_one_valid_json_object_per_line(tmp_path: pathlib.Path) -> None:
    run_id = "01TEST00000000000000000006"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    log.emit_task_started("t")
    log.emit_node_started(
        topology_id="t1",
        node_id="n0",
        node_role="r",
        attempt=1,
        model_id="m",
        provider_id="p",
        predecessor_ids=(),
        edge_ids=(),
    )
    log.emit_node_completed(
        node_id="n0",
        node_role="r",
        output="ok",
        latency_ms=1.0,
        cost_usd=0.0,
        model_id="m",
        provider_id="p",
    )
    log.emit_final_result(
        status="success",
        output="ok",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=1,
    )
    log.close()

    raw = _event_path(tmp_path, run_id).read_text(encoding="utf-8")
    lines = [line for line in raw.splitlines() if line]
    parsed = [json.loads(line) for line in lines]
    assert len(parsed) == len(lines) > 0


def test_raw_prompts_hashed_redacted_by_default(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SAGE_TRACE_RAW", raising=False)
    run_id = "01TEST00000000000000000007"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    secret_task = "sk-1234567890abcdef1234567890abcdef ABC123"
    log.set_task_text(secret_task)
    log.emit_task_started(secret_task)
    log.emit_final_result(
        status="success",
        output="o",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=0,
    )
    log.close()

    raw = _event_path(tmp_path, run_id).read_text(encoding="utf-8")
    assert "sk-1234567890abcdef1234567890abcdef" not in raw
    events = [json.loads(line) for line in raw.splitlines()]
    assert all("payload" not in event for event in events)
    assert "task_hash" in raw


@pytest.mark.asyncio
async def test_concurrent_runs_dont_interleave(tmp_path: pathlib.Path) -> None:
    async def _run_one(run_id: str) -> None:
        log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
        for i in range(20):
            log.emit_task_started(f"task {i} for {run_id}")
        log.emit_final_result(
            status="success",
            output="o",
            total_cost_usd=0.0,
            total_latency_ms=1.0,
            node_count=0,
        )
        log.close()

    await asyncio.gather(
        _run_one("01TESTAAAAAAAAAAAAAAAAAAA1"),
        _run_one("01TESTBBBBBBBBBBBBBBBBBBB2"),
    )
    a = _read_events(_event_path(tmp_path, "01TESTAAAAAAAAAAAAAAAAAAA1"))
    b = _read_events(_event_path(tmp_path, "01TESTBBBBBBBBBBBBBBBBBBB2"))
    assert all(e["run_id"] == "01TESTAAAAAAAAAAAAAAAAAAA1" for e in a)
    assert all(e["run_id"] == "01TESTBBBBBBBBBBBBBBBBBBB2" for e in b)


def test_sink_failure_default_disables_writer(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    log = RuntimeEventLog(run_id="01TEST00000000000000000009", trace_dir=tmp_path)
    log._fh.write = MagicMock(side_effect=OSError("disk full"))

    log.emit_task_started("t")

    assert log.disabled is True
    log.emit_final_result(
        status="success",
        output="o",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=0,
    )


def test_sink_failure_fail_closed_raises(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_TRACE_FAIL_CLOSED", "1")
    log = RuntimeEventLog(run_id="01TEST00000000000000000010", trace_dir=tmp_path)
    log._fh.write = MagicMock(side_effect=OSError("disk full"))

    with pytest.raises(EventLogUnavailable):
        log.emit_task_started("t")


def test_run_id_is_canonical_ulid_even_when_ulid_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro 2026-04-28 R5 verify push-back: ensure the pipeline run_id
    fallback (when the ulid library is unavailable) still produces a
    canonical 26-char Crockford-Base32 ULID. The schema contract is
    'canonical 26-char uppercase ULID' — must hold even on import error.
    """
    import re
    import sys

    from sage.pipeline import _new_runtime_run_id

    # Force the soft import to fail by stubbing the ulid module to None.
    monkeypatch.setitem(sys.modules, "ulid", None)

    run_id = _new_runtime_run_id()

    # Crockford Base32 excludes I, L, O, U.
    assert re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", run_id), (
        f"run_id {run_id!r} is not a canonical 26-char Crockford-Base32 ULID"
    )
    # Must be uppercase
    assert run_id == run_id.upper()

    # And the writer must accept it as-is for the trace filename.
    log = RuntimeEventLog(run_id=run_id, trace_dir=None)
    assert log.run_id == run_id
    assert log.trace_id == run_id  # v0 contract: trace_id == run_id
