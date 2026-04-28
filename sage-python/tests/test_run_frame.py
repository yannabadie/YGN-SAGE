"""R7 RunFrame v0 acceptance tests (18 contract tests)."""
from __future__ import annotations

import json
import pathlib
import shutil
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from sage.pipeline import CognitiveOrchestrationPipeline
from sage.runtime.event_log import EventLogUnavailable, RuntimeEventLog
from sage.runtime.run_frame import NodeRunRecord, RUN_FRAME_SCHEMA_VERSION, RunFrame
from sage.runtime.run_frame.builder import _RunFrameBuilder
from sage.runtime.state import StateDelta


@pytest.fixture
def tmp_path() -> pathlib.Path:
    path = pathlib.Path(".tmp") / "pytest-run-frame" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class FakeNode:
    def __init__(
        self,
        role: str,
        model_id: str = "model-a",
        system: int = 1,
        *,
        provider_id: str = "provider-a",
    ) -> None:
        self.role = role
        self.model_id = model_id
        self.system = system
        self.provider_id = provider_id
        self.node_type = "llm"
        self.required_capabilities: list[str] = []


class FakeGraph:
    def __init__(
        self,
        nodes: list[FakeNode],
        edges: list[tuple[int, int, str | int]] | None = None,
        *,
        graph_id: str = "topo-a",
    ) -> None:
        self._nodes = nodes
        self._edges = edges if edges is not None else [
            (idx, idx + 1, "message") for idx in range(max(0, len(nodes) - 1))
        ]
        self.id = graph_id
        self.template_type = "run_frame_test"

    def node_count(self) -> int:
        return len(self._nodes)

    def get_node(self, idx: int) -> FakeNode:
        return self._nodes[idx]

    def get_predecessors(self, idx: int) -> list[int]:
        return [src for src, dst, _edge_type in self._edges if dst == idx]

    def get_edges(self) -> list[tuple[int, int, str | int]]:
        return list(self._edges)


class FakeExecutor:
    def __init__(self, ready_sequence: list[list[int]]) -> None:
        self._batches = list(ready_sequence)
        self._batch_idx = 0
        self.completed: list[int] = []
        self.skipped: list[int] = []
        self.opened: list[tuple[int, int]] = []

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

    def open_gate(self, graph: FakeGraph, source: int, target: int) -> None:
        self.opened.append((source, target))

    def is_done(self) -> bool:
        return self._batch_idx >= len(self._batches)


def _read_events(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _event_path(tmp_path: pathlib.Path, run_id: str) -> pathlib.Path:
    return tmp_path / f"{run_id}.jsonl"


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
    event_log: RuntimeEventLog | None,
    builder: _RunFrameBuilder | None,
    controller: Any | None = None,
):
    from sage.topology.runner import TopologyRunner

    return TopologyRunner(
        graph=graph,
        executor=FakeExecutor(ready_sequence),
        llm_provider=_make_provider(outputs),
        event_log=event_log,
        run_frame_builder=builder,
        controller=controller,
    )


def _builder(run_id: str = "01RUNFRAME00000000000001") -> _RunFrameBuilder:
    builder = _RunFrameBuilder(
        run_id=run_id,
        task_id=run_id,
        task_hash="task-hash",
    )
    builder.capture_feature_flags({})
    builder.record_topology_selected(
        seq=None,
        topology_id="topo-a",
        graph_digest="digest-a",
        reason="initial",
    )
    return builder


def _make_pipeline(output: str = "pipeline output") -> CognitiveOrchestrationPipeline:
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=None,
        assigner=None,
        provider_pool=None,
        llm_provider=MagicMock(),
    )

    def _classify(ctx):
        ctx.system = 1
        ctx.domain = "code"
        return ctx

    async def _decompose(ctx):
        return ctx

    def _select(ctx):
        ctx.topology = FakeGraph([FakeNode("solo")], graph_id="pipe-topo")
        ctx.topology_id = "pipe-topo"
        return ctx

    def _assign(ctx):
        ctx.assignments = {0: "model-a"}
        ctx.provider_hints = {0: "provider-a"}
        return ctx

    async def _execute(ctx, **_kwargs):
        ctx.result = output
        ctx.cost = 0.0
        return ctx

    async def _learn(ctx):
        return None

    pipeline._stage_classify = _classify
    pipeline._stage_decompose = _decompose
    pipeline._stage_select_topology = _select
    pipeline._stage_assign_models = _assign
    pipeline._stage_execute = _execute
    pipeline._stage_learn = _learn
    pipeline._record_to_memory = lambda _ctx: None
    return pipeline


async def _run_pipeline_trace(
    *,
    trace_dir: pathlib.Path,
    run_id: str,
    run_frame: str | None,
    monkeypatch: pytest.MonkeyPatch,
    with_frame: bool,
) -> tuple[str, RunFrame | None, str]:
    monkeypatch.setenv("SAGE_TRACE_JSONL_DIR", str(trace_dir))
    if run_frame is None:
        monkeypatch.delenv("SAGE_RUN_FRAME", raising=False)
    else:
        monkeypatch.setenv("SAGE_RUN_FRAME", run_frame)
    monkeypatch.setattr("sage.pipeline._new_runtime_run_id", lambda: run_id)

    stamps = iter(range(10_000, 10_100))
    monotonic = iter([1.0, 2.0])
    monkeypatch.setattr("sage.runtime.event_log.writer.time.time_ns", lambda: next(stamps, 10_099))
    monkeypatch.setattr("sage.pipeline.time.monotonic", lambda: next(monotonic, 2.0))

    pipeline = _make_pipeline()
    if with_frame:
        result, frame = await pipeline.run_with_frame("task")
    else:
        result = await pipeline.run("task")
        frame = None
    raw = _event_path(trace_dir, run_id).read_text(encoding="utf-8")
    return result, frame, raw


def test_run_frame_constructed_with_run_id_at_task_started() -> None:
    builder = _RunFrameBuilder(
        run_id="01TASKSTARTED00000000001",
        task_id="01TASKSTARTED00000000001",
        task_hash="hash-a",
    )
    builder.capture_feature_flags({"SAGE_RUN_FRAME": "1"})

    frame = builder.finalize()

    assert isinstance(frame, RunFrame)
    assert frame.schema_version == RUN_FRAME_SCHEMA_VERSION == "0"
    assert frame.run_id == "01TASKSTARTED00000000001"
    assert frame.task_id == "01TASKSTARTED00000000001"
    assert frame.task_hash == "hash-a"


@pytest.mark.asyncio
async def test_run_frame_node_records_populated_on_each_node_completion(
    tmp_path: pathlib.Path,
) -> None:
    run_id = "01RFNODERECORD0000000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    builder = _builder(run_id)
    graph = FakeGraph([FakeNode("first"), FakeNode("second")])
    runner = _make_runner(
        graph,
        [[0], [1]],
        outputs=["first output", "second output"],
        event_log=log,
        builder=builder,
    )

    await runner.run("task")
    log.close()
    frame = builder.finalize()

    assert len(frame.node_records) == 2
    assert all(isinstance(record, NodeRunRecord) for record in frame.node_records.values())
    assert {record.status for record in frame.node_records.values()} == {"success"}
    assert {record.output_length for record in frame.node_records.values()} == {
        len("first output"),
        len("second output"),
    }


@pytest.mark.asyncio
async def test_run_frame_state_frames_per_node_when_statecore_on(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_STATECORE", "1")
    run_id = "01RFSTATECOREON000000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    builder = _builder(run_id)
    graph = FakeGraph(
        [FakeNode("state-source"), FakeNode("consumer")],
        edges=[(0, 1, "state")],
    )
    runner = _make_runner(
        graph,
        [[0], [1]],
        outputs=["source", "consumer"],
        event_log=log,
        builder=builder,
    )
    runner._node_state_deltas[0] = StateDelta(add_constraints=("must_be_python",))

    await runner.run("task")
    log.close()
    frame = builder.finalize()

    assert "1" in frame.state_frames
    assert "must_be_python" in frame.state_frames["1"].constraints
    consumer = next(record for record in frame.node_records.values() if record.node_id == "1")
    assert consumer.state_before_version == 0
    assert consumer.state_after_version == 1
    assert consumer.state_delta is not None
    assert "must_be_python" in consumer.state_delta.add_constraints


@pytest.mark.asyncio
async def test_run_frame_state_frames_empty_when_statecore_off(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SAGE_STATECORE", raising=False)
    run_id = "01RFSTATECOREOFF00000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    builder = _builder(run_id)
    graph = FakeGraph(
        [FakeNode("source"), FakeNode("consumer")],
        edges=[(0, 1, "state")],
    )
    runner = _make_runner(
        graph,
        [[0], [1]],
        outputs=["source", "consumer"],
        event_log=log,
        builder=builder,
    )
    runner._node_state_deltas[0] = StateDelta(add_constraints=("must_not_apply",))

    await runner.run("task")
    log.close()
    frame = builder.finalize()

    assert frame.state_frames == {}
    assert all(record.state_before_version is None for record in frame.node_records.values())
    assert all(record.state_after_version is None for record in frame.node_records.values())
    assert all(record.state_delta is None for record in frame.node_records.values())
    assert all(record.predecessors_by_channel is None for record in frame.node_records.values())


def test_run_frame_feature_flags_snapshot_at_task_started() -> None:
    builder = _RunFrameBuilder(run_id="run", task_id="run", task_hash="hash")
    builder.capture_feature_flags(
        {
            "SAGE_RUN_FRAME": "1",
            "SAGE_STATECORE": "1",
            "SAGE_TRACE_JSONL_DIR": r"C:\secret\trace",
            "SAGE_TRACE_RAW": "0",
            "SAGE_TRACE_FAIL_CLOSED": "1",
            "SAGE_DIFF_VERIFIER_MODE": "observe",
            "SAGE_ENABLE_PATH6": "",
            "SAGE_DASHBOARD_TOKEN": "secret-token",
        },
    )

    frame = builder.finalize()

    assert frame.feature_flags["SAGE_TRACE_JSONL_DIR"] == "<path>"
    assert frame.feature_flags["SAGE_RUN_FRAME"] == "1"
    assert "SAGE_DASHBOARD_TOKEN" not in frame.feature_flags
    assert tuple(frame.feature_flags) == tuple(sorted(frame.feature_flags))


@pytest.mark.asyncio
async def test_run_frame_node_record_ids_join_to_event_log_seqs(
    tmp_path: pathlib.Path,
) -> None:
    run_id = "01RFJOINSEQS00000000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    builder = _builder(run_id)
    graph = FakeGraph([FakeNode("solo")])
    runner = _make_runner(
        graph,
        [[0]],
        outputs=["solo output"],
        event_log=log,
        builder=builder,
    )

    await runner.run("task")
    log.close()
    events = _read_events(_event_path(tmp_path, run_id))
    frame = builder.finalize()
    record = next(iter(frame.node_records.values()))

    started = next(event for event in events if event["event_type"] == "node_started")
    completed = next(event for event in events if event["event_type"] == "node_completed")
    assert record.node_run_id == "0:0:1"
    assert record.node_started_seq == started["seq"]
    assert record.node_completed_seq == completed["seq"]
    assert record.event_seqs == (started["seq"], completed["seq"])


@pytest.mark.asyncio
async def test_run_frame_summary_event_only_when_r7_enabled(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    off_dir = tmp_path / "off"
    on_dir = tmp_path / "on"
    off_dir.mkdir()
    on_dir.mkdir()

    await _run_pipeline_trace(
        trace_dir=off_dir,
        run_id="01RFSUMMARYOFF000000001",
        run_frame=None,
        monkeypatch=monkeypatch,
        with_frame=True,
    )
    _, frame, _ = await _run_pipeline_trace(
        trace_dir=on_dir,
        run_id="01RFSUMMARYON0000000001",
        run_frame="1",
        monkeypatch=monkeypatch,
        with_frame=True,
    )

    off_events = _read_events(_event_path(off_dir, "01RFSUMMARYOFF000000001"))
    on_events = _read_events(_event_path(on_dir, "01RFSUMMARYON0000000001"))
    assert "run_frame_summary" not in [event["event_type"] for event in off_events]
    assert on_events[-2]["event_type"] == "final_result"
    assert on_events[-1]["event_type"] == "run_frame_summary"
    assert on_events[-1]["parent_event_id"] == on_events[-2]["seq"]
    assert frame is not None
    assert on_events[-1]["payload"]["final_result_seq"] == frame.final_result_seq


@pytest.mark.asyncio
async def test_pipeline_run_with_frame_returns_typed_runframe(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_TRACE_JSONL_DIR", str(tmp_path))
    monkeypatch.delenv("SAGE_RUN_FRAME", raising=False)
    monkeypatch.setattr("sage.pipeline._new_runtime_run_id", lambda: "01RFTYPED00000000000001")

    result, frame = await _make_pipeline("typed output").run_with_frame("task")

    assert result == "typed output"
    assert isinstance(frame, RunFrame)
    assert frame.run_id == "01RFTYPED00000000000001"
    assert frame.status == "success"


@pytest.mark.asyncio
async def test_run_with_frame_signature_mirrors_run(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R7.0.2 (cgpro 2026-04-29 cycle 4 reassess): run_with_frame() must
    accept budget_usd + system_hint kwargs identically to run(), so bench
    and traced adapters can switch entry points without parameter loss.
    """
    import inspect

    from sage.pipeline import CognitiveOrchestrationPipeline

    run_sig = inspect.signature(CognitiveOrchestrationPipeline.run)
    frame_sig = inspect.signature(CognitiveOrchestrationPipeline.run_with_frame)

    # Same input parameters (excluding self) — only return type differs
    run_params = list(run_sig.parameters.keys())
    frame_params = list(frame_sig.parameters.keys())
    assert run_params == frame_params, (
        f"run_with_frame signature must mirror run(); "
        f"run params={run_params} vs frame params={frame_params}"
    )

    # Smoke-test: passing budget_usd + system_hint must not error
    monkeypatch.setenv("SAGE_TRACE_JSONL_DIR", str(tmp_path))
    monkeypatch.delenv("SAGE_RUN_FRAME", raising=False)
    monkeypatch.setattr(
        "sage.pipeline._new_runtime_run_id",
        lambda: "01RFPARITY0000000000000001",
    )
    result, frame = await _make_pipeline("parity output").run_with_frame(
        "task",
        budget_usd=2.0,
        system_hint=2,
    )
    assert result == "parity output"
    assert isinstance(frame, RunFrame)


@pytest.mark.asyncio
async def test_run_frame_off_exactly_matches_r6_jsonl(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_dir = tmp_path / "baseline"
    with_frame_dir = tmp_path / "with-frame"
    baseline_dir.mkdir()
    with_frame_dir.mkdir()

    _, _, baseline = await _run_pipeline_trace(
        trace_dir=baseline_dir,
        run_id="01RFBYTEIDENTICAL000001",
        run_frame=None,
        monkeypatch=monkeypatch,
        with_frame=False,
    )
    _, _, with_frame = await _run_pipeline_trace(
        trace_dir=with_frame_dir,
        run_id="01RFBYTEIDENTICAL000001",
        run_frame=None,
        monkeypatch=monkeypatch,
        with_frame=True,
    )

    assert with_frame == baseline


@pytest.mark.asyncio
async def test_run_frame_on_matches_r6_jsonl_after_stripping_trailing_summary(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_dir = tmp_path / "baseline"
    on_dir = tmp_path / "on"
    baseline_dir.mkdir()
    on_dir.mkdir()

    _, _, baseline = await _run_pipeline_trace(
        trace_dir=baseline_dir,
        run_id="01RFONSTRIP000000000001",
        run_frame=None,
        monkeypatch=monkeypatch,
        with_frame=False,
    )
    _, _, on_raw = await _run_pipeline_trace(
        trace_dir=on_dir,
        run_id="01RFONSTRIP000000000001",
        run_frame="1",
        monkeypatch=monkeypatch,
        with_frame=True,
    )

    stripped = "\n".join(on_raw.splitlines()[:-1]) + "\n"
    assert stripped == baseline


@pytest.mark.asyncio
async def test_run_frame_node_records_track_failures(
    tmp_path: pathlib.Path,
) -> None:
    run_id = "01RFFAILURE000000000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    builder = _builder(run_id)
    graph = FakeGraph([FakeNode("bad")])
    runner = _make_runner(
        graph,
        [[0]],
        outputs=["unused"],
        event_log=log,
        builder=builder,
    )
    runner._execute_node = AsyncMock(side_effect=RuntimeError("provider down"))

    with pytest.raises(RuntimeError, match="provider down"):
        await runner.run("task")
    log.close()
    frame = builder.finalize()
    record = next(iter(frame.node_records.values()))

    assert record.status == "failure"
    assert record.failure_seq is not None
    assert frame.failure_seqs == (record.failure_seq,)
    assert frame.terminal_failure_seq == record.failure_seq


def test_run_frame_handles_reroute() -> None:
    builder = _builder("01RFREROUTE000000000001")
    first = builder.record_node_started(
        seq=1,
        node_id="0",
        provider_id="p",
        model_id="m",
        predecessor_ids=(),
        predecessors_by_channel=None,
    )
    builder.record_controller_decision(seq=2, node_run_id=first, action="reroute_topology")
    builder.record_failure(seq=3, node_run_id=first, kind="controller_reroute")
    builder.record_topology_selected(
        seq=4,
        topology_id="topo-b",
        graph_digest="digest-b",
        reason="reroute",
    )
    second = builder.record_node_started(
        seq=5,
        node_id="0",
        provider_id="p",
        model_id="m",
        predecessor_ids=(),
        predecessors_by_channel=None,
    )
    builder.record_node_completed(seq=6, node_run_id=second, output="ok")

    frame = builder.finalize()

    assert [ref.reason for ref in frame.topology_history] == ["initial", "reroute"]
    assert [ref.topology_epoch for ref in frame.topology_history] == [0, 1]
    assert set(frame.node_records) == {"0:0:1", "1:0:1"}


def test_run_frame_handles_budget_exceeded() -> None:
    builder = _builder("01RFBUDGET0000000000001")
    builder.record_budget(
        seq=7,
        kind="exceeded",
        budget_limit_usd=1.0,
        budget_remaining_usd=0.0,
        cost_so_far_usd=1.2,
    )
    builder.record_final_result(seq=8, status="budget_exceeded")

    frame = builder.finalize()

    assert frame.status == "budget_exceeded"
    assert frame.budget_snapshot == {
        "kind": "exceeded",
        "budget_limit_usd": 1.0,
        "budget_remaining_usd": 0.0,
        "cost_so_far_usd": 1.2,
        "seq": 7,
    }


@pytest.mark.asyncio
async def test_run_frame_existing_events_schema_version_unchanged_when_r7_on(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    on_dir = tmp_path / "on"
    on_dir.mkdir()
    await _run_pipeline_trace(
        trace_dir=on_dir,
        run_id="01RFSCHEMAUNCHANGED0001",
        run_frame="1",
        monkeypatch=monkeypatch,
        with_frame=True,
    )

    events = _read_events(_event_path(on_dir, "01RFSCHEMAUNCHANGED0001"))
    existing = [event for event in events if event["event_type"] != "run_frame_summary"]
    assert existing
    assert all(event["schema_version"] == "1.0" for event in existing)


@pytest.mark.asyncio
async def test_run_frame_summary_write_failure_does_not_change_pipeline_result(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_TRACE_JSONL_DIR", str(tmp_path))
    monkeypatch.setenv("SAGE_RUN_FRAME", "1")
    monkeypatch.setattr("sage.pipeline._new_runtime_run_id", lambda: "01RFSUMFAIL000000000001")

    def _raise_summary(self, **_kwargs):
        raise EventLogUnavailable("summary sink failed")

    monkeypatch.setattr(RuntimeEventLog, "emit_run_frame_summary", _raise_summary)

    result, frame = await _make_pipeline("survived").run_with_frame("task")

    assert result == "survived"
    assert frame.status == "success"
    assert frame.final_result_seq is not None


@pytest.mark.asyncio
async def test_run_frame_event_log_disabled_still_returns_frame_with_none_seqs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SAGE_TRACE_JSONL_DIR", raising=False)
    monkeypatch.delenv("SAGE_RUN_FRAME", raising=False)
    monkeypatch.setattr("sage.pipeline._new_runtime_run_id", lambda: "01RFDISABLED00000000001")

    result, frame = await _make_pipeline("disabled").run_with_frame("task")

    assert result == "disabled"
    assert frame.final_result_seq is None
    assert frame.topology_history[0].topology_selected_seq is None


def test_run_frame_does_not_capture_secret_env_vars() -> None:
    builder = _RunFrameBuilder(run_id="run", task_id="run", task_hash="hash")
    builder.capture_feature_flags(
        {
            "SAGE_RUN_FRAME": "1",
            "SAGE_DASHBOARD_TOKEN": "must-not-leak",
            "SAGE_TRACE_JSONL_DIR": r"C:\trace",
        },
    )

    frame = builder.finalize()

    assert "SAGE_DASHBOARD_TOKEN" not in frame.feature_flags
    assert "must-not-leak" not in json.dumps(frame.to_summary_dict(redacted=True))


def test_run_frame_multi_attempt_records_do_not_overwrite() -> None:
    builder = _builder("01RFMULTIATTEMPT0000001")
    first = builder.record_node_started(
        seq=1,
        node_id="0",
        provider_id="p",
        model_id="m",
        predecessor_ids=(),
        predecessors_by_channel=None,
    )
    builder.record_failure(seq=2, node_run_id=first, kind="provider_error")
    second = builder.record_node_started(
        seq=3,
        node_id="0",
        provider_id="p",
        model_id="m2",
        predecessor_ids=(),
        predecessors_by_channel=None,
    )
    builder.record_node_completed(seq=4, node_run_id=second, output="retry ok")

    frame = builder.finalize()

    assert set(frame.node_records) == {"0:0:1", "0:0:2"}
    assert frame.node_records["0:0:1"].status == "failure"
    assert frame.node_records["0:0:2"].status == "success"
    assert frame.node_records["0:0:2"].model_id == "m2"


def test_run_frame_reroute_topology_epoch_prevents_node_id_collision() -> None:
    builder = _builder("01RFCOLLISION0000000001")
    first = builder.record_node_started(
        seq=1,
        node_id="shared",
        provider_id="p",
        model_id="m",
        predecessor_ids=(),
        predecessors_by_channel=None,
    )
    builder.record_node_completed(seq=2, node_run_id=first, output="before")
    builder.record_topology_selected(
        seq=3,
        topology_id="topo-rerouted",
        graph_digest="digest-rerouted",
        reason="reroute",
    )
    second = builder.record_node_started(
        seq=4,
        node_id="shared",
        provider_id="p",
        model_id="m",
        predecessor_ids=(),
        predecessors_by_channel=None,
    )
    builder.record_node_completed(seq=5, node_run_id=second, output="after")

    frame = builder.finalize()

    assert set(frame.node_records) == {"0:shared:1", "1:shared:1"}
    assert frame.node_records["0:shared:1"].output_length == len("before")
    assert frame.node_records["1:shared:1"].output_length == len("after")
