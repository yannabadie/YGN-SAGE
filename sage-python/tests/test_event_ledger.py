"""Tests for sage.bench.event_ledger.

Tests three contracts:

1. Append-only NDJSON: every emit writes one valid JSON line.
2. Crash-safe: a process death between emits leaves the file as a
   valid prefix (no partial line) — simulated by reopening and
   parsing.
3. Standard event emitters set the right discriminators and pass
   through caller-supplied fields unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sage.bench.event_ledger import BenchEventLedger, build_run_meta


def _read_lines(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_run_meta_includes_required_fields() -> None:
    meta = build_run_meta(
        bench_type="ablation",
        tier="budget",
        timeout_s=120.0,
        limit=10,
    )
    for key in (
        "bench_type", "tier", "timeout_s", "limit",
        "git_sha", "git_dirty_hash", "pid", "host", "os",
        "python_version",
    ):
        assert key in meta, f"missing {key} in run_meta"
    assert meta["bench_type"] == "ablation"
    assert meta["tier"] == "budget"
    assert meta["timeout_s"] == 120.0


def test_run_meta_extra_merges(tmp_path: Path) -> None:
    meta = build_run_meta(
        bench_type="ablation",
        tier="budget",
        timeout_s=60.0,
        extra={"subset": "hard", "split": "instruct"},
    )
    assert meta["subset"] == "hard"
    assert meta["split"] == "instruct"


def test_emit_appends_one_line_per_event(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    meta = build_run_meta(bench_type="t", tier="budget", timeout_s=10)
    ledger = BenchEventLedger(path, meta)
    try:
        ledger.emit_run_start()
        ledger.emit_config_start(config_label="full", config_dict={"memory": True})
        ledger.emit_task_start(
            config_label="full", idx=1, task_id="BCB/13", timeout_s=120.0,
        )
        ledger.emit_task_end(
            config_label="full", idx=1, task_id="BCB/13",
            status="PASS", elapsed_wall_ms=37000.0, passed=True,
        )
        ledger.emit_config_end(config_label="full", passed=1, total=1)
        ledger.emit_run_end()
    finally:
        ledger.close()

    lines = _read_lines(path)
    assert [r["event"] for r in lines] == [
        "RUN_START", "CONFIG_START", "TASK_START", "TASK_END",
        "CONFIG_END", "RUN_END",
    ]


def test_run_id_constant_across_emits(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    meta = build_run_meta(bench_type="t", tier="budget", timeout_s=10)
    ledger = BenchEventLedger(path, meta)
    try:
        ledger.emit_run_start()
        ledger.emit_run_end()
    finally:
        ledger.close()

    lines = _read_lines(path)
    run_ids = {r["run_id"] for r in lines}
    assert len(run_ids) == 1
    assert run_ids.pop() == ledger.run_id


def test_each_record_has_iso_ts_and_run_id(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    meta = build_run_meta(bench_type="t", tier="budget", timeout_s=10)
    with BenchEventLedger(path, meta) as ledger:
        ledger.emit_run_start()
        ledger.emit_run_end()

    lines = _read_lines(path)
    for r in lines:
        assert "ts" in r
        assert "run_id" in r
        # ISO-8601: starts with YYYY-MM-DD and contains T separator.
        assert r["ts"][4] == "-" and r["ts"][7] == "-" and "T" in r["ts"]


def test_task_end_passes_through_control_surface(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    meta = build_run_meta(bench_type="t", tier="budget", timeout_s=10)
    with BenchEventLedger(path, meta) as ledger:
        ledger.emit_run_start()
        ledger.emit_task_end(
            config_label="full",
            idx=5,
            task_id="BCB/82",
            status="FAIL",
            elapsed_wall_ms=93000.0,
            passed=False,
            control_surface={
                "executed_template": "robust",
                "node_count": 5,
                "controller_attached": False,
                "skip_guardrails": False,
                "skip_avr": False,
                "skip_memory": False,
                "skip_routing": False,
                "frugal_cascade_attempted": True,
                "controller_decision_count": 0,
            },
            host_suspend_or_event_loop_stall=False,
        )
        ledger.emit_run_end()

    lines = _read_lines(path)
    task_end = next(r for r in lines if r["event"] == "TASK_END")
    cs = task_end["control_surface"]
    assert cs["executed_template"] == "robust"
    assert cs["node_count"] == 5
    assert cs["controller_attached"] is False
    assert cs["frugal_cascade_attempted"] is True
    assert task_end["host_suspend_or_event_loop_stall"] is False


def test_task_abort_event_marks_excluded(tmp_path: Path) -> None:
    """TASK_ABORT is the canonical exclusion signal for sleep-poisoned tasks."""
    path = tmp_path / "events.jsonl"
    meta = build_run_meta(bench_type="t", tier="budget", timeout_s=10)
    with BenchEventLedger(path, meta) as ledger:
        ledger.emit_run_start()
        ledger.emit_task_abort(
            config_label="full",
            idx=34,
            task_id="BCB/273",
            reason="host_suspend_detected",
            elapsed_wall_ms=20278211.0,
        )
        ledger.emit_run_abort(reason="host_suspend_detected")

    lines = _read_lines(path)
    abort = next(r for r in lines if r["event"] == "TASK_ABORT")
    assert abort["reason"] == "host_suspend_detected"
    assert abort["elapsed_wall_ms"] == 20278211.0
    run_abort = next(r for r in lines if r["event"] == "RUN_ABORT")
    assert run_abort["reason"] == "host_suspend_detected"


def test_partial_run_is_a_valid_prefix(tmp_path: Path) -> None:
    """Crash before close() must leave a parseable file (every line a complete JSON object).

    We emulate ``kill -9`` by NOT calling ``close()``: each emit fsyncs,
    so the file on disk after the last successful emit must already be
    valid NDJSON. No half-written line allowed.
    """
    path = tmp_path / "events.jsonl"
    meta = build_run_meta(bench_type="t", tier="budget", timeout_s=10)
    ledger = BenchEventLedger(path, meta)
    ledger.emit_run_start()
    ledger.emit_config_start(config_label="full", config_dict={"memory": True})
    ledger.emit_task_start(
        config_label="full", idx=1, task_id="BCB/13", timeout_s=120.0,
    )
    # Simulate process death: no close(), no run_end. The file on disk
    # should still parse cleanly because every emit fsyncs.
    del ledger

    lines = _read_lines(path)
    assert [r["event"] for r in lines] == [
        "RUN_START", "CONFIG_START", "TASK_START",
    ]


def test_emit_after_close_raises(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    meta = build_run_meta(bench_type="t", tier="budget", timeout_s=10)
    ledger = BenchEventLedger(path, meta)
    ledger.emit_run_start()
    ledger.close()
    with pytest.raises(RuntimeError, match="after close"):
        ledger.emit("X")


def test_context_manager_emits_run_abort_on_exception(tmp_path: Path) -> None:
    """``with BenchEventLedger(...) as l:`` raising mid-run records RUN_ABORT."""
    path = tmp_path / "events.jsonl"
    meta = build_run_meta(bench_type="t", tier="budget", timeout_s=10)

    with pytest.raises(ValueError):
        with BenchEventLedger(path, meta) as ledger:
            ledger.emit_run_start()
            raise ValueError("synthetic crash")

    lines = _read_lines(path)
    assert lines[-1]["event"] == "RUN_ABORT"
    assert "ValueError" in lines[-1]["reason"]


def test_run_meta_embedded_in_run_start(tmp_path: Path) -> None:
    """The ledger should be self-describing: RUN_START must carry run_meta."""
    path = tmp_path / "events.jsonl"
    meta = build_run_meta(bench_type="ablation", tier="budget", timeout_s=120, limit=8)
    with BenchEventLedger(path, meta) as ledger:
        ledger.emit_run_start()
        ledger.emit_run_end()

    lines = _read_lines(path)
    rs = next(r for r in lines if r["event"] == "RUN_START")
    assert rs["run_meta"]["tier"] == "budget"
    assert rs["run_meta"]["timeout_s"] == 120
    assert rs["run_meta"]["limit"] == 8
