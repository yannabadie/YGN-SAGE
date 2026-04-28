"""R6 StateCore v0 acceptance tests (12 contract tests)."""
from __future__ import annotations

import json
import pathlib
import shutil
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from sage.runtime.event_log import RuntimeEventLog
from sage.runtime.state import (
    EvidenceRef,
    StateApplyResult,
    StateConflict,
    StateDelta,
    StateFrame,
    apply_delta,
    apply_deltas,
    normalize_assumption_id,
)


@pytest.fixture
def tmp_path() -> pathlib.Path:
    """Local temp fixture.

    The default pytest temp root is ACL-denied on this Windows host. Keep these
    tests isolated under the repo's existing untracked .tmp area.
    """
    path = pathlib.Path(".tmp") / "pytest-statecore" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class FakeNode:
    def __init__(
        self,
        role: str,
        model_id: str = "",
        system: int = 1,
        *,
        prompt: str = "",
    ) -> None:
        self.role = role
        self.model_id = model_id
        self.system = system
        self.prompt = prompt
        self.node_type = "llm"
        self.required_capabilities: list[str] = []


class FakeGraph:
    def __init__(self, nodes: list[FakeNode], edges: list[tuple[int, int, str]]) -> None:
        self._nodes = nodes
        self._edges = edges
        self.id = "statecore-test"
        self.template_type = "statecore"

    def node_count(self) -> int:
        return len(self._nodes)

    def get_edges(self) -> list[tuple[int, int, str]]:
        return list(self._edges)

    def get_node(self, idx: int) -> FakeNode:
        return self._nodes[idx]

    def get_predecessors(self, idx: int) -> list[int]:
        return [src for src, dst, _edge_type in self._edges if dst == idx]


class FakeExecutor:
    def __init__(self, ready_sequence: list[list[int]]) -> None:
        self._batches = list(ready_sequence)
        self._batch_idx = 0
        self.completed: list[int] = []
        self.skipped: list[int] = []

    def next_ready(self, graph: FakeGraph) -> list[int]:
        if self._batch_idx >= len(self._batches):
            return []
        batch = self._batches[self._batch_idx]
        self._batch_idx += 1
        return batch

    def mark_completed(self, idx: int) -> None:
        self.completed.append(idx)

    def mark_skipped(self, idx: int) -> None:
        self.skipped.append(idx)

    def is_done(self) -> bool:
        return self._batch_idx >= len(self._batches)


def _make_provider(outputs: list[str]) -> MagicMock:
    provider = MagicMock()
    provider.generate = AsyncMock(
        side_effect=[SimpleNamespace(content=output, usage={}) for output in outputs],
    )
    return provider


def _make_runner(
    graph: FakeGraph,
    ready_sequence: list[list[int]],
    *,
    outputs: list[str],
    event_log: RuntimeEventLog | None = None,
):
    from sage.topology.runner import TopologyRunner

    provider = _make_provider(outputs)
    runner = TopologyRunner(
        graph=graph,
        executor=FakeExecutor(ready_sequence),
        llm_provider=provider,
        event_log=event_log,
    )
    return runner, provider


def _message_texts(provider: MagicMock, call_index: int) -> str:
    call = provider.generate.call_args_list[call_index]
    messages = call.kwargs.get("messages", call.args[0] if call.args else [])
    return "\n".join(getattr(message, "content", "") for message in messages)


def _read_events(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _event_path(tmp_path: pathlib.Path, run_id: str) -> pathlib.Path:
    return tmp_path / f"{run_id}.jsonl"


@pytest.mark.asyncio
async def test_state_edge_does_not_appear_in_message_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_STATECORE", "1")
    graph = FakeGraph(
        [FakeNode("state-writer"), FakeNode("consumer")],
        [(0, 1, "state")],
    )
    runner, provider = _make_runner(
        graph,
        [[0], [1]],
        outputs=["STATE ONLY TEXT", "consumer done"],
    )

    await runner.run("task")

    second_messages = _message_texts(provider, 1)
    assert "STATE ONLY TEXT" not in second_messages
    assert "Context from previous agents" not in second_messages
    assert "StateCore frame" in second_messages


@pytest.mark.asyncio
async def test_message_edge_does_not_apply_state_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_STATECORE", "1")
    graph = FakeGraph(
        [FakeNode("producer"), FakeNode("consumer")],
        [(0, 1, "message")],
    )
    runner, provider = _make_runner(
        graph,
        [[0], [1]],
        outputs=["message output", "consumer done"],
    )
    runner._node_state_deltas[0] = StateDelta(add_constraints=("must_not_apply",))

    await runner.run("task")

    second_messages = _message_texts(provider, 1)
    assert "message output" in second_messages
    assert "must_not_apply" not in second_messages
    assert 1 not in runner._node_state_frames


@pytest.mark.asyncio
async def test_control_edge_does_not_feed_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_STATECORE", "1")
    graph = FakeGraph(
        [FakeNode("gate"), FakeNode("worker")],
        [(0, 1, "control")],
    )
    _runner, provider = _make_runner(
        graph,
        [[0], [1]],
        outputs=["CONTROL ONLY TEXT", "worker done"],
    )

    await _runner.run("task")

    second_messages = _message_texts(provider, 1)
    assert "CONTROL ONLY TEXT" not in second_messages
    assert "Context from previous agents" not in second_messages


def test_state_delta_version_increment() -> None:
    frame = StateFrame(task_id="t", version=0)
    delta = StateDelta(add_constraints=("must_be_python",))
    result = apply_delta(frame, delta, source_node_id="n0")
    assert isinstance(result, StateApplyResult)
    assert result.applied is True
    assert result.after.version == 1
    assert "must_be_python" in result.after.constraints
    decision_result = apply_delta(
        StateFrame(task_id="t", version=0),
        StateDelta(
            add_decisions=({"id": "d1", "status": "accepted"},),
            evidence=(EvidenceRef(kind="node", id="n0"),),
        ),
    )
    assert decision_result.applied is True


def test_invalidated_assumption_removed() -> None:
    assert normalize_assumption_id(" File-X Unchanged ") == "file_x_unchanged"
    frame = StateFrame(task_id="t", version=0, assumptions=("file_x_unchanged",))
    delta = StateDelta(invalidate_assumptions=("File-X Unchanged",))
    result = apply_delta(frame, delta, source_node_id="n0")
    assert result.applied is True
    assert "file_x_unchanged" not in result.after.assumptions
    assert "file_x_unchanged" in result.after.invalidated_assumptions


@pytest.mark.asyncio
async def test_state_conflict_is_not_silent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    frame = StateFrame(task_id="t", version=0)
    left = StateDelta(update_entities={"file.py": {"status": "old"}})
    right = StateDelta(update_entities={"file.py": {"status": "new"}})

    result = apply_deltas(frame, (("n0", left), ("n1", right)))

    assert result.applied is False
    assert result.conflicts
    assert result.after is frame or result.after == frame
    with pytest.raises(StateConflict):
        apply_deltas(frame, (("n0", left), ("n1", right)), raise_on_conflict=True)

    monkeypatch.setenv("SAGE_STATECORE", "1")
    run_id = "01STATECONFLICT000000000001"
    event_log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    graph = FakeGraph(
        [FakeNode("left"), FakeNode("right"), FakeNode("join")],
        [(0, 2, "state"), (1, 2, "state")],
    )
    runner, provider = _make_runner(
        graph,
        [[0, 1], [2]],
        outputs=["left output", "right output"],
        event_log=event_log,
    )
    runner._node_state_deltas[0] = left
    runner._node_state_deltas[1] = right

    await runner.run("task")
    event_log.close()

    assert provider.generate.call_count == 2
    events = _read_events(_event_path(tmp_path, run_id))
    state_events = [event for event in events if event["event_type"] == "state_applied"]
    assert len(state_events) == 1
    event = state_events[0]
    assert event["target_node_id"] == "2"
    assert event["conflict_count"] > 0
    assert event["applied"] is False
    assert event["before_version"] == event["after_version"]
    assert any(
        item["event_type"] == "failure" and item.get("node_id") == "2"
        for item in events
    )


@pytest.mark.asyncio
async def test_legacy_all_control_edges_preserve_message_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SAGE_STATECORE", raising=False)
    graph = FakeGraph(
        [FakeNode("legacy-source"), FakeNode("legacy-target")],
        [(0, 1, "control")],
    )
    _runner, provider = _make_runner(
        graph,
        [[0], [1]],
        outputs=["LEGACY CONTROL TEXT", "target done"],
    )

    await _runner.run("task")

    assert "LEGACY CONTROL TEXT" in _message_texts(provider, 1)


@pytest.mark.asyncio
async def test_typed_control_edge_is_not_message_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_STATECORE", "1")
    graph = FakeGraph(
        [FakeNode("typed-source"), FakeNode("typed-target")],
        [(0, 1, "control")],
    )
    runner, provider = _make_runner(
        graph,
        [[0], [1]],
        outputs=["STRICT CONTROL TEXT", "target done"],
    )

    await runner.run("task")

    assert "STRICT CONTROL TEXT" not in _message_texts(provider, 1)


@pytest.mark.asyncio
async def test_state_applied_event_emitted_with_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setenv("SAGE_STATECORE", "1")
    run_id = "01STATEAPPLIED000000000001"
    event_log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    graph = FakeGraph(
        [FakeNode("state-source"), FakeNode("state-target")],
        [(0, 1, "state")],
    )
    runner, _provider = _make_runner(
        graph,
        [[0], [1]],
        outputs=["source output", "target output"],
        event_log=event_log,
    )
    runner._node_state_deltas[0] = StateDelta(add_constraints=("must_be_python",))

    await runner.run("task")
    event_log.close()

    assert "must_be_python" in runner._node_state_frames[1].constraints
    events = _read_events(_event_path(tmp_path, run_id))
    state_event = next(event for event in events if event["event_type"] == "state_applied")
    assert state_event["target_node_id"] == "1"
    assert state_event["source_node_ids"] == ["0"]
    assert state_event["before_version"] == 0
    assert state_event["after_version"] == 1
    assert state_event["delta_count"] == 1
    assert state_event["conflict_count"] == 0
    assert state_event["applied"] is True


@pytest.mark.asyncio
async def test_statecore_off_preserves_legacy_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    async def _run_with_env(value: str | None) -> tuple[str, list[str]]:
        if value is None:
            monkeypatch.delenv("SAGE_STATECORE", raising=False)
            run_id = "01STATEOFFUNSET0000000001"
        else:
            monkeypatch.setenv("SAGE_STATECORE", value)
            run_id = "01STATEOFFZERO00000000001"
        event_log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
        graph = FakeGraph(
            [FakeNode("producer"), FakeNode("consumer")],
            [(0, 1, "control")],
        )
        _runner, provider = _make_runner(
            graph,
            [[0], [1]],
            outputs=["legacy output", "consumer done"],
            event_log=event_log,
        )
        await _runner.run("task")
        event_log.close()
        context_message = next(
            line
            for line in _message_texts(provider, 1).splitlines()
            if line.startswith("Context from previous agents")
        )
        all_text = _message_texts(provider, 1)
        events = _read_events(_event_path(tmp_path, run_id))
        return all_text[all_text.index(context_message) :], [
            event["event_type"] for event in events
        ]

    unset_text, unset_event_types = await _run_with_env(None)
    zero_text, zero_event_types = await _run_with_env("0")
    golden = "Context from previous agents:\n[producer]: legacy output\ntask"

    assert unset_text == zero_text
    assert golden in unset_text
    assert "state_applied" not in unset_event_types
    assert "state_applied" not in zero_event_types


def test_unknown_edge_type_raises_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setenv("SAGE_STATECORE", "1")
    run_id = "01STATEUNKNOWN00000000001"
    event_log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    graph = FakeGraph(
        [FakeNode("source"), FakeNode("target")],
        [(0, 1, "mystery")],
    )
    runner, _provider = _make_runner(
        graph,
        [[1]],
        outputs=["target output"],
        event_log=event_log,
    )

    with pytest.raises(ValueError, match="unknown edge type"):
        runner._partition_incoming_edges(1)
    event_log.close()

    events = _read_events(_event_path(tmp_path, run_id))
    assert any(
        event["event_type"] == "failure" and event.get("node_id") == "1"
        for event in events
    )


@pytest.mark.asyncio
async def test_planner_injection_respects_message_channel_in_statecore_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_STATECORE", "1")
    monkeypatch.setenv("SAGE_PLANNER_INJECTION", "1")

    def _runner_for(edge_type: str):
        graph = FakeGraph(
            [FakeNode("planner"), FakeNode("coder")],
            [(0, 1, edge_type)],
        )
        runner, _provider = _make_runner(
            graph,
            [[0], [1]],
            outputs=["planner output", "coder output"],
        )
        runner._node_outputs[0] = f"PLAN VIA {edge_type.upper()}"
        return runner

    assert _runner_for("control")._maybe_planner_injection(1, "You are coder.") == (
        "You are coder."
    )
    assert _runner_for("state")._maybe_planner_injection(1, "You are coder.") == (
        "You are coder."
    )

    message_prompt = _runner_for("message")._maybe_planner_injection(
        1,
        "You are coder.",
    )
    assert "Upstream plan (from planner)" in message_prompt
    assert "PLAN VIA MESSAGE" in message_prompt


@pytest.mark.asyncio
async def test_node_started_predecessors_by_channel_only_in_statecore_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """cgpro 2026-04-29 R6 verify push-back: NodeStarted.payload must carry
    `predecessors_by_channel` partition WHEN AND ONLY WHEN SAGE_STATECORE=1.
    OFF mode must NOT include the field at all (preserves byte-identical
    R5 schema). ON mode MUST include partitioned dict {control, message,
    state}.
    """
    # SAGE_TRACE_RAW=1 surfaces the payload field in the JSONL line so we can
    # introspect predecessors_by_channel directly.
    monkeypatch.setenv("SAGE_TRACE_RAW", "1")

    async def _capture_node_started_payload(
        statecore: str | None,
    ) -> dict[str, object]:
        if statecore is None:
            monkeypatch.delenv("SAGE_STATECORE", raising=False)
            run_id = "01NSCHANOFF000000000000001"
        else:
            monkeypatch.setenv("SAGE_STATECORE", statecore)
            run_id = "01NSCHANON0000000000000001"
        event_log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
        graph = FakeGraph(
            [
                FakeNode("controller"),
                FakeNode("messager"),
                FakeNode("stater"),
                FakeNode("consumer"),
            ],
            [
                (0, 3, "control"),
                (1, 3, "message"),
                (2, 3, "state"),
            ],
        )
        runner, _provider = _make_runner(
            graph,
            [[0, 1, 2], [3]],
            outputs=["c", "m", "s", "consumer done"],
            event_log=event_log,
        )
        await runner.run("task")
        event_log.close()
        events = _read_events(_event_path(tmp_path, run_id))
        # Pick the consumer's NodeStarted (node_id == "3")
        node_started = next(
            event
            for event in events
            if event["event_type"] == "node_started" and event["node_id"] == "3"
        )
        return node_started

    on_event = await _capture_node_started_payload("1")
    off_event = await _capture_node_started_payload(None)

    # ON: predecessors_by_channel present and partitioned
    assert "payload" in on_event, "ON event must surface payload (SAGE_TRACE_RAW=1)"
    on_payload = on_event["payload"]
    assert "predecessors_by_channel" in on_payload, (
        "ON mode NodeStarted.payload must carry predecessors_by_channel"
    )
    on_partition = on_payload["predecessors_by_channel"]
    assert set(on_partition.keys()) == {"control", "message", "state"}
    assert on_partition["control"] == ["0"]
    assert on_partition["message"] == ["1"]
    assert on_partition["state"] == ["2"]
    # Flat predecessor_ids still present for back-compat
    assert "predecessor_ids" in on_payload
    assert sorted(on_payload["predecessor_ids"]) == ["0", "1", "2"]

    # OFF: predecessors_by_channel absent (byte-identical R5 schema)
    assert "payload" in off_event, "OFF event also has payload (SAGE_TRACE_RAW=1)"
    off_payload = off_event["payload"]
    assert "predecessors_by_channel" not in off_payload, (
        "OFF mode NodeStarted.payload must NOT carry predecessors_by_channel "
        "(byte-identical R5 schema guarantee)"
    )
    # Flat predecessor_ids still present
    assert "predecessor_ids" in off_payload
