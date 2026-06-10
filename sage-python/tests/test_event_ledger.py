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

from sage.bench.event_ledger import (
    EMPTY_PATCH_REASON_CODES,
    TIMEOUT_REASON_CODES,
    BenchEventLedger,
    build_run_meta,
    categorize_timeout,
    classify_non_timeout_empty_patch,
)


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


# ---------------------------------------------------------------------------
# Block A5 (cgpro DESIGN 2026-05-10, conv
# `cgpro_ygn_sage_global_analysis_20260510`):
# `categorize_timeout` distinguishes scoring_boot_impossible vs
# reasoner_thinking_overflow vs stage_deadlock vs provider_call_timeout
# from the per-task event log content.
# ---------------------------------------------------------------------------


def _cli_progress(stage: str, elapsed_ms: int) -> dict:
    return {
        "event_type": "cli_progress",
        "payload": {"stage": stage, "elapsed_ms": elapsed_ms},
    }


def _model_assigned(model_id: str, provider_id: str) -> dict:
    return {
        "event_type": "model_assigned",
        "payload": {"model_id": model_id, "provider_id": provider_id},
    }


def _node_started(node_id: str) -> dict:
    return {"event_type": "node_started", "payload": {"node_id": node_id}}


def _routing_decision(model_id: str) -> dict:
    return {"event_type": "routing_decision", "payload": {"model_id": model_id}}


def test_categorize_timeout_scoring_boot_impossible() -> None:
    """No progress, no routing, no assignment → scoring_boot_impossible."""
    result = categorize_timeout(
        progress_events=[],
        model_assigned_events=[],
        node_started_events=[],
        routing_decision_events=[],
        elapsed_total_ms=60_000.0,
    )
    assert result["reason_code"] == "scoring_boot_impossible"
    assert result["last_stage"] is None
    assert result["elapsed_ms_by_stage"] == {}
    assert result["provider_attempted"] is False
    assert result["model_id_final"] is None
    assert result["provider_final"] is None


def test_categorize_timeout_reasoner_thinking_overflow_time_based() -> None:
    """Time-based heuristic per cgpro DESIGN: majority temporal of run in
    a reasoner stage with regular heartbeats → reasoner_thinking_overflow.
    NOT count-based — six events all at decompose with cumulative
    elapsed_ms covering 50s of a 60s run.
    """
    progress = [
        _cli_progress("decompose", 10_000),
        _cli_progress("decompose", 20_000),
        _cli_progress("decompose", 30_000),
        _cli_progress("decompose", 40_000),
        _cli_progress("decompose", 50_000),
        _cli_progress("decompose", 58_000),
    ]
    result = categorize_timeout(
        progress_events=progress,
        model_assigned_events=[_model_assigned("deepseek-v4-pro", "deepseek")],
        node_started_events=[],
        routing_decision_events=[_routing_decision("deepseek-v4-pro")],
        elapsed_total_ms=60_000.0,
    )
    assert result["reason_code"] == "reasoner_thinking_overflow"
    assert result["last_stage"] == "decompose"
    # Stage duration = last - first elapsed_ms = 58000 - 10000 = 48000 ms.
    assert result["elapsed_ms_by_stage"]["decompose"] == 48_000
    # provider_attempted MUST stay False — model_assigned alone is not
    # proof of a call attempt per cgpro DESIGN correction.
    assert result["provider_attempted"] is False
    assert result["model_id_final"] == "deepseek-v4-pro"
    assert result["provider_final"] == "deepseek"


def test_categorize_timeout_stage_deadlock_silent_heartbeat() -> None:
    """Progression then silence: last heartbeat far before timeout
    → stage_deadlock (NOT reasoner overflow even if last_stage is a
    reasoner stage).
    """
    progress = [
        _cli_progress("decompose", 5_000),
        _cli_progress("decompose", 10_000),
        # Then silence: timeout fires at 60_000 ms, last heartbeat at 10s.
        # Gap = 50_000 ms > heartbeat_max_gap_ms default 30_000.
    ]
    result = categorize_timeout(
        progress_events=progress,
        model_assigned_events=[],
        node_started_events=[],
        routing_decision_events=[_routing_decision("deepseek-v4-flash")],
        elapsed_total_ms=60_000.0,
    )
    assert result["reason_code"] == "stage_deadlock"
    assert result["last_stage"] == "decompose"


def test_categorize_timeout_provider_attempted_strict_requires_node_started() -> None:
    """provider_attempted must be False when only model_assigned is
    emitted — cgpro DESIGN: assignment is not a call attempt.
    """
    only_assignment_result = categorize_timeout(
        progress_events=[_cli_progress("execute", 30_000)],
        model_assigned_events=[_model_assigned("deepseek-v4-flash", "deepseek")],
        node_started_events=[],
        routing_decision_events=[_routing_decision("deepseek-v4-flash")],
        elapsed_total_ms=60_000.0,
    )
    assert only_assignment_result["provider_attempted"] is False

    # Sanity: with node_started, the flag flips True.
    with_node_started_result = categorize_timeout(
        progress_events=[_cli_progress("execute", 30_000)],
        model_assigned_events=[_model_assigned("deepseek-v4-flash", "deepseek")],
        node_started_events=[_node_started("node-0")],
        routing_decision_events=[_routing_decision("deepseek-v4-flash")],
        elapsed_total_ms=60_000.0,
    )
    assert with_node_started_result["provider_attempted"] is True


def test_categorize_timeout_provider_call_timeout() -> None:
    """provider_attempted=True AND last_stage='execute' →
    provider_call_timeout (positive evidence of provider RPC hang).
    """
    progress = [
        _cli_progress("decompose", 5_000),
        _cli_progress("execute", 15_000),
        _cli_progress("execute", 30_000),
        _cli_progress("execute", 55_000),
    ]
    result = categorize_timeout(
        progress_events=progress,
        model_assigned_events=[_model_assigned("deepseek-v4-pro", "deepseek")],
        node_started_events=[_node_started("node-0")],
        routing_decision_events=[_routing_decision("deepseek-v4-pro")],
        elapsed_total_ms=60_000.0,
    )
    assert result["reason_code"] == "provider_call_timeout"
    assert result["last_stage"] == "execute"
    assert result["provider_attempted"] is True


def test_categorize_timeout_non_reasoner_stage_is_stage_deadlock() -> None:
    """Last stage NOT in {decompose, execute} (e.g. select_topology,
    learn) with no provider witness → stage_deadlock, NOT
    reasoner_thinking_overflow even if heartbeats are regular.
    """
    progress = [
        _cli_progress("select_topology", 5_000),
        _cli_progress("select_topology", 25_000),
        _cli_progress("select_topology", 55_000),
    ]
    result = categorize_timeout(
        progress_events=progress,
        model_assigned_events=[],
        node_started_events=[],
        routing_decision_events=[_routing_decision("deepseek-v4-flash")],
        elapsed_total_ms=60_000.0,
    )
    assert result["reason_code"] == "stage_deadlock"
    assert result["last_stage"] == "select_topology"


# ─────────────────────────────────────────────────────────────────────────────
# Block `canary-stage-timing-budget` (cgpro DESIGN 2026-05-11): empty-patch
# reason-code enum + classifier for non-timeout cases. Together with
# ``categorize_timeout`` outputs, these feed the pre-grader gate
# (``canary_pre_grader_gate.py``).
# ─────────────────────────────────────────────────────────────────────────────


def test_timeout_reason_codes_match_categorize_timeout_outputs() -> None:
    """The exported ``TIMEOUT_REASON_CODES`` enum must equal the set of
    ``reason_code`` values that ``categorize_timeout`` can actually
    return. Drift here means the pre-grader gate would either accept
    invalid codes or reject valid ones.
    """
    expected = {
        "scoring_boot_impossible",
        "provider_call_timeout",
        "reasoner_thinking_overflow",
        "stage_deadlock",
    }
    assert TIMEOUT_REASON_CODES == expected
    assert isinstance(TIMEOUT_REASON_CODES, frozenset)


def test_empty_patch_reason_codes_superset_of_timeout_codes() -> None:
    """``EMPTY_PATCH_REASON_CODES`` must be a strict superset of
    ``TIMEOUT_REASON_CODES`` and add the four non-timeout codes
    (three from cgpro DESIGN 2026-05-11 + ``repo_unavailable`` from
    RESOLUTION_UNBLOCKERS 2026-06-10).
    """
    assert TIMEOUT_REASON_CODES.issubset(EMPTY_PATCH_REASON_CODES)
    extras = EMPTY_PATCH_REASON_CODES - TIMEOUT_REASON_CODES
    assert extras == {
        "no_patch_extracted",
        "task_budget_exhausted",
        "no_patch_to_verify",
        "repo_unavailable",
    }
    assert isinstance(EMPTY_PATCH_REASON_CODES, frozenset)


def test_classify_repo_unavailable_dominates_everything() -> None:
    """RESOLUTION_UNBLOCKERS (cgpro Q2): a fail-closed repo skip is an
    infra failure and must never be reclassified as a budget or model
    signal, whatever else is set."""
    assert (
        classify_non_timeout_empty_patch(
            budget_exhausted=True,
            diff_verifier_outcome="no_patch_to_verify",
            repo_unavailable=True,
        )
        == "repo_unavailable"
    )
    assert (
        classify_non_timeout_empty_patch(
            budget_exhausted=False,
            repo_unavailable=True,
        )
        == "repo_unavailable"
    )


def test_classify_non_timeout_empty_patch_budget_exhausted_wins() -> None:
    """Budget-exhausted dominates other signals."""
    assert (
        classify_non_timeout_empty_patch(
            budget_exhausted=True,
            diff_verifier_outcome="no_patch_to_verify",
        )
        == "task_budget_exhausted"
    )
    assert (
        classify_non_timeout_empty_patch(
            budget_exhausted=True,
            diff_verifier_outcome=None,
        )
        == "task_budget_exhausted"
    )


def test_classify_non_timeout_empty_patch_verifier_signal() -> None:
    """Diff verifier explicit signal wins over the fallback."""
    assert (
        classify_non_timeout_empty_patch(
            budget_exhausted=False,
            diff_verifier_outcome="no_patch_to_verify",
        )
        == "no_patch_to_verify"
    )


def test_classify_non_timeout_empty_patch_fallback_no_patch_extracted() -> None:
    """Default case when neither budget nor verifier explains it."""
    assert (
        classify_non_timeout_empty_patch(
            budget_exhausted=False,
            diff_verifier_outcome=None,
        )
        == "no_patch_extracted"
    )
    # Unknown verifier outcome strings do NOT short-circuit; they fall
    # through to the no_patch_extracted fallback so the gate has
    # something canonical to match against.
    assert (
        classify_non_timeout_empty_patch(
            budget_exhausted=False,
            diff_verifier_outcome="match_ok",
        )
        == "no_patch_extracted"
    )


def test_classify_non_timeout_empty_patch_returns_only_allowed_codes() -> None:
    """Sanity: every reachable return value is a member of
    ``EMPTY_PATCH_REASON_CODES``.
    """
    cases = [
        {"budget_exhausted": True, "diff_verifier_outcome": None},
        {"budget_exhausted": False, "diff_verifier_outcome": "no_patch_to_verify"},
        {"budget_exhausted": False, "diff_verifier_outcome": None},
        {"budget_exhausted": False, "diff_verifier_outcome": "anything_else"},
    ]
    for case in cases:
        assert (
            classify_non_timeout_empty_patch(**case)  # type: ignore[arg-type]
            in EMPTY_PATCH_REASON_CODES
        )
