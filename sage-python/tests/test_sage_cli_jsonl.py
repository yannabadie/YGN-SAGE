"""Cycle-12 prelude: tests for ``sage run --jsonl`` v0 protocol components.

Per ``docs/contracts/SAGE_CLI_PROTOCOL.md``. These tests exercise the
COMPONENTS of the CLI driver (envelope emission, sink mirroring, approval
bridge, argparse contract). Full integration tests that boot the real
pipeline + assert byte-identical golden JSONL across runs are Cycle-12
Phase B work (the ADR-015 decomposition needs to land first so each stage's
event emissions are contractual rather than incidental).

What this file covers:
  1. CLI-shell envelope shape (``_emit_cli_event``).
  2. Tee sink mirrors writes to both file and stdout (``_CliMirrorSinkHandle``).
  3. Sequence counter monotonic without gaps (``_SeqCounter``).
  4. Approval bridge round-trip (``_CliApprovalBridge``):
     - approve_tool_call resolves the future to True
     - deny_tool_call resolves to False
     - timeout falls back to deny
  5. argparse contract: ``--jsonl`` required, empty task returns 2.

What this file does NOT cover (deferred to Cycle-12 Phase B):
  - Full ``sage run --jsonl`` integration with real pipeline boot.
  - Golden JSONL byte-identical snapshots across multi-stage runs.
  - End-to-end ``cancel`` mid-run drains + emits ``cli_complete(outcome=cancelled)``.
"""
from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace
from typing import Any

import pytest

from sage.cli import run as cli_run


# ────────────────────────────────────────────────────────────────────
# 1. CLI-shell envelope shape
# ────────────────────────────────────────────────────────────────────


def test_emit_cli_event_envelope_shape() -> None:
    """``_emit_cli_event`` produces the documented v0 envelope."""
    buf = io.StringIO()
    cli_run._emit_cli_event(
        buf,
        "cli_started",
        run_id="01TESTRUN0000000000000001",
        payload={"protocol_version": "v0", "task": "hello"},
        seq=0,
    )
    line = buf.getvalue()

    # JSONL discipline: exactly one line, LF-terminated.
    assert line.endswith("\n"), "frame must end with LF"
    assert line.count("\n") == 1, "exactly one LF per frame"
    assert "\r" not in line, "no CR in stdout (LF-only protocol)"

    frame = json.loads(line.rstrip("\n"))
    # Envelope per docs/contracts/SAGE_CLI_PROTOCOL.md
    assert frame["protocol_version"] == "v0"
    assert frame["event_type"] == "cli_started"
    assert frame["seq"] == 0
    assert frame["run_id"] == "01TESTRUN0000000000000001"
    assert frame["payload_schema_version"] == "cli_v1"
    assert isinstance(frame["ts_ms"], int)
    assert frame["payload"]["protocol_version"] == "v0"
    assert frame["payload"]["task"] == "hello"


def test_emit_cli_event_compact_separators() -> None:
    """Frames use compact JSON (no whitespace) so frame size is minimal.

    pi-mono RPC consumers parse line-by-line; whitespace inside the JSON
    object is allowed by JSON spec but inflates wire size on a TUI that
    streams every keystroke. Compact = ``json.dumps(..., separators=(",", ":"))``.
    """
    buf = io.StringIO()
    cli_run._emit_cli_event(
        buf,
        "cli_complete",
        run_id="01TESTRUN0000000000000002",
        payload={"exit_code": 0, "outcome": "success"},
        seq=1,
    )
    line = buf.getvalue().rstrip("\n")
    # Compact separators: no spaces around ``:`` or ``,``.
    assert ", " not in line
    assert ": " not in line


# ────────────────────────────────────────────────────────────────────
# 2. Tee sink mirrors writes to both file and stdout
# ────────────────────────────────────────────────────────────────────


class _RecordingFile:
    """Test double for ``_SinkHandle``-shaped object."""

    def __init__(self) -> None:
        self.writes: list[str] = []
        self.flushed = 0
        self.closed = False

    def write(self, value: str) -> int:
        self.writes.append(value)
        return len(value)

    def flush(self) -> None:
        self.flushed += 1

    def fileno(self) -> int:
        return -1

    def tell(self) -> int:
        return sum(len(w) for w in self.writes)

    def truncate(self, size: int | None = None) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


def test_tee_sink_writes_file_verbatim_and_renumbers_stdout_seq() -> None:
    """File write is byte-identical (forensic), stdout seq is rewritten.

    Stage A contract (cgpro `cgpro_cli_protocol_gaps_20260507`): the forensic
    archive keeps the RuntimeEventLog internal seq domain unchanged, while
    the stdout mirror is renumbered through the unified ``_StdoutSeqCounter``
    so the frontend can reconcile the stream via ``cli_complete.payload.final_seq``.
    """
    file = _RecordingFile()
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()
    tee = cli_run._CliMirrorSinkHandle(file, stdout, counter)

    tee.write('{"event_type":"task_started","seq":42}\n')

    # File: byte-identical to RuntimeEventLog's emit (forensic preserved).
    assert file.writes == ['{"event_type":"task_started","seq":42}\n']
    # Stdout: same fields but seq replaced with the next counter value (0).
    assert stdout.getvalue() == '{"event_type":"task_started","seq":0}\n'
    # Mirror tracks the rewritten seq so cli_complete can pull final_seq.
    assert tee.last_stdout_seq == 0


def test_tee_sink_flush_propagates_to_both() -> None:
    """``flush()`` must hit both sinks so frames are visible to the consumer."""
    file = _RecordingFile()
    stdout = io.StringIO()
    tee = cli_run._CliMirrorSinkHandle(file, stdout, cli_run._StdoutSeqCounter())

    tee.write("data\n")
    tee.flush()

    assert file.flushed == 1


def test_tee_sink_close_only_closes_file_not_stdout() -> None:
    """``close()`` releases the file handle but NEVER closes stdout
    (the parent process owns it; closing it would break the next test
    or any subsequent CLI output)."""

    class _StdoutDouble:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

        def write(self, _value: str) -> int:
            return 0

        def flush(self) -> None:
            pass

    file = _RecordingFile()
    stdout = _StdoutDouble()
    tee = cli_run._CliMirrorSinkHandle(file, stdout, cli_run._StdoutSeqCounter())

    tee.close()

    assert file.closed is True
    assert stdout.closed is False, "tee.close() must NOT close stdout"


def test_tee_sink_swallows_stdout_broken_pipe() -> None:
    """If the consumer disconnects mid-run, stdout writes raise OSError.
    The tee MUST NOT propagate — the file write (forensic) keeps going."""

    class _BrokenStdout:
        def write(self, _value: str) -> int:
            raise OSError("Broken pipe")

        def flush(self) -> None:
            raise OSError("Broken pipe")

    file = _RecordingFile()
    stdout = _BrokenStdout()
    tee = cli_run._CliMirrorSinkHandle(file, stdout, cli_run._StdoutSeqCounter())

    # Should NOT raise — the file write completes.
    n = tee.write('{"event_type":"task_started","seq":0}\n')

    # File write succeeded; stdout was best-effort.
    assert file.writes == ['{"event_type":"task_started","seq":0}\n']
    assert n == len('{"event_type":"task_started","seq":0}\n')


# ────────────────────────────────────────────────────────────────────
# 3. Seq counter monotonic without gaps
# ────────────────────────────────────────────────────────────────────


def test_seq_counter_monotonic() -> None:
    counter = cli_run._SeqCounter()
    seqs = [counter.next() for _ in range(5)]
    assert seqs == [0, 1, 2, 3, 4]


# ────────────────────────────────────────────────────────────────────
# 3.5 Global stdout seq + final_seq reconciliation (Stage A)
# ────────────────────────────────────────────────────────────────────


def test_stdout_global_seq_monotonic_across_cli_and_mirror() -> None:
    """Stage A contract: every stdout frame, including mirrored runtime events,
    is assigned one monotonic per-run stdout seq.

    Replays a realistic frame sequence:
    cli_started → 4 runtime events with raw seq starting at 0 (would
    duplicate if we passed through verbatim) → run_frame_summary →
    cli_complete. Asserts every stdout frame's seq is contiguous 0..N
    with no duplicates / gaps.
    """
    file = _RecordingFile()
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()

    # cli_started (CLI envelope): emitted directly via _emit_cli_event.
    cli_run._emit_cli_event(
        stdout, "cli_started", run_id="R", payload={}, seq=counter.next(),
    )

    # Mirror sink renumbers runtime events.
    tee = cli_run._CliMirrorSinkHandle(file, stdout, counter)
    for raw_seq in range(4):
        tee.write(
            json.dumps({"event_type": "task_started", "seq": raw_seq}) + "\n"
        )
    # run_frame_summary is the last runtime event.
    tee.write('{"event_type":"run_frame_summary","seq":99}\n')

    # Pull seq from each stdout line.
    lines = stdout.getvalue().splitlines()
    seqs = [json.loads(line)["seq"] for line in lines]
    assert seqs == [0, 1, 2, 3, 4, 5], f"non-contiguous stdout seqs: {seqs}"
    # Mirror tracks the last rewritten seq → seed for cli_complete.final_seq.
    assert tee.last_stdout_seq == 5


def test_file_seq_unchanged_after_mirror_rewrite() -> None:
    """Forensic archive (file) preserves the RuntimeEventLog internal seq
    domain BYTE-IDENTICAL. Only stdout is renumbered."""
    file = _RecordingFile()
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()
    counter.next()  # simulate cli_started already emitted (stdout seq 0).
    tee = cli_run._CliMirrorSinkHandle(file, stdout, counter)

    # Runtime emits its own seq=7 from a long-running session.
    tee.write('{"event_type":"task_started","seq":7}\n')

    # File side: byte-identical to the runtime emit.
    assert file.writes == ['{"event_type":"task_started","seq":7}\n']
    # Stdout side: renumbered to the next stdout slot (1, since 0 was cli_started).
    assert json.loads(stdout.getvalue())["seq"] == 1


def test_final_seq_equals_last_mirrored_frame_seq() -> None:
    """Protocol invariant 5 (docs/contracts/SAGE_CLI_PROTOCOL.md):
    ``cli_complete.payload.final_seq`` equals the stdout seq of the last
    mirrored frame (run_frame_summary on the success path)."""
    file = _RecordingFile()
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()
    cli_run._emit_cli_event(
        stdout, "cli_started", run_id="R", payload={}, seq=counter.next(),
    )
    tee = cli_run._CliMirrorSinkHandle(file, stdout, counter)
    tee.write('{"event_type":"task_started","seq":0}\n')
    tee.write('{"event_type":"run_frame_summary","seq":99}\n')

    final_seq_from_mirror = tee.last_stdout_seq
    assert final_seq_from_mirror == 2  # 0=cli_started, 1=task_started, 2=run_frame_summary

    # Now emit cli_complete with final_seq = mirror's last_stdout_seq.
    cli_run._emit_cli_event(
        stdout,
        "cli_complete",
        run_id="R",
        payload={
            "exit_code": 0,
            "outcome": "success",
            "final_seq": final_seq_from_mirror,
        },
        seq=counter.next(),
    )

    lines = stdout.getvalue().splitlines()
    cli_complete_frame = json.loads(lines[-1])
    assert cli_complete_frame["event_type"] == "cli_complete"
    # cli_complete.seq is final_seq + 1 (it's the frame AFTER the last mirror frame).
    assert cli_complete_frame["seq"] == final_seq_from_mirror + 1
    # final_seq matches the run_frame_summary seq.
    assert cli_complete_frame["payload"]["final_seq"] == final_seq_from_mirror
    # cli_complete is the LAST frame.
    assert all(
        json.loads(line).get("event_type") != "cli_complete"
        for line in lines[:-1]
    )


def test_final_seq_tracks_cli_tool_request_after_mirror() -> None:
    """Per cgpro VERIFY round-2: ``final_seq`` MUST come from the global
    stdout counter (``_StdoutSeqCounter.last``), NOT from the mirror's
    own ``last_stdout_seq``. CLI-shell frames like ``cli_tool_request``
    can fire AFTER the last mirrored runtime event (mid-tool-call cancel
    path); the mirror tracker would miss those, but the counter never
    misses.

    Sequence:
      cli_started        seq=0
      mirrored task_started   seq=1
      cli_tool_request   seq=2
      cli_complete       seq=3, payload.final_seq=2
    """
    file = _RecordingFile()
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()

    cli_run._emit_cli_event(
        stdout, "cli_started", run_id="R", payload={}, seq=counter.next(),
    )
    tee = cli_run._CliMirrorSinkHandle(file, stdout, counter)
    tee.write('{"event_type":"task_started","seq":0}\n')
    cli_run._emit_cli_event(
        stdout,
        "cli_tool_request",
        run_id="R",
        payload={"correlation_id": "c1", "tool_name": "noop"},
        seq=counter.next(),
    )

    # Pre cli_complete: counter.last is the cli_tool_request seq (2),
    # mirror.last_stdout_seq is the task_started seq (1). The terminal
    # frame must use counter.last.
    assert counter.last == 2
    assert tee.last_stdout_seq == 1
    final_seq = counter.last

    cli_run._emit_cli_event(
        stdout,
        "cli_complete",
        run_id="R",
        payload={"exit_code": 0, "outcome": "success", "final_seq": final_seq},
        seq=counter.next(),
    )

    lines = stdout.getvalue().splitlines()
    seqs = [json.loads(line)["seq"] for line in lines]
    assert seqs == [0, 1, 2, 3]
    cli_complete_frame = json.loads(lines[-1])
    assert cli_complete_frame["event_type"] == "cli_complete"
    assert cli_complete_frame["seq"] == 3
    assert cli_complete_frame["payload"]["final_seq"] == 2


def test_set_budget_before_prompt_emits_failure_with_reason_budget_before_prompt() -> None:
    """Stage B: ``set_budget`` arriving before ``prompt`` (no active context)
    is rejected non-terminally with ``failure(error_type="budget_before_prompt")``.
    """
    captured_failures: list[dict[str, Any]] = []

    class _RecordingEventLog:
        def emit_failure(
            self, *, kind: str, error_type: str, message: str, node_id: str = "",
        ) -> None:
            captured_failures.append(
                {"kind": kind, "error_type": error_type, "message": message},
            )

        def emit_budget(self, **kwargs: Any) -> None:
            captured_failures.append({"_unexpected_budget_event": kwargs})

    cli_run._handle_set_budget(
        {"command": "set_budget", "args": {"budget_usd": 5.0}},
        pipeline=None,  # no boot yet
        event_log=_RecordingEventLog(),
    )
    assert len(captured_failures) == 1
    assert captured_failures[0]["error_type"] == "budget_before_prompt"
    assert captured_failures[0]["kind"] == "cli_command"


def test_set_budget_invalid_value_emits_budget_invalid_value() -> None:
    """Non-numeric / non-finite ``budget_usd`` is rejected before reaching
    the pipeline. CostTracker root guard isn't even consulted."""
    captured_failures: list[str] = []

    class _RecordingEventLog:
        def emit_failure(
            self, *, kind: str, error_type: str, message: str, node_id: str = "",
        ) -> None:
            captured_failures.append(error_type)

        def emit_budget(self, **kwargs: Any) -> None:
            captured_failures.append("UNEXPECTED_budget_event")

    for bad in [None, "12.34", True]:
        cli_run._handle_set_budget(
            {"command": "set_budget", "args": {"budget_usd": bad}},
            pipeline=object(),
            event_log=_RecordingEventLog(),
        )

    assert captured_failures == [
        "budget_invalid_value", "budget_invalid_value", "budget_invalid_value",
    ]


def test_set_budget_zero_rejected_as_budget_invalid_value() -> None:
    """Per cgpro Stage B VERIFY round-2 trap: ``budget_usd == 0`` is
    rejected at the CostTracker root guard because the tracker's
    unlimited sentinel is ``budget_usd <= 0``; accepting zero would
    silently keep the tracker unlimited rather than freeze remaining."""

    class _ActivePipeline:
        _active_context = type("_C", (), {"cost_tracker": __import__(
            "sage.contracts.cost_tracker", fromlist=["CostTracker"]
        ).CostTracker(budget_usd=10.0)})()

        def tighten_budget(self, new_remaining_usd: float) -> Any:
            return self._active_context.cost_tracker.tighten_remaining_budget(
                new_remaining_usd
            )

    captured: list[str] = []

    class _RecordingEventLog:
        def emit_failure(self, **kwargs: Any) -> None:
            captured.append(("failure", kwargs.get("error_type")))

        def emit_budget(self, **kwargs: Any) -> None:
            captured.append(("budget", kwargs.get("kind")))

    cli_run._handle_set_budget(
        {"command": "set_budget", "args": {"budget_usd": 0}},
        pipeline=_ActivePipeline(),
        event_log=_RecordingEventLog(),
    )
    # Exactly one failure event, no budget event.
    assert captured == [("failure", "budget_invalid_value")]


def test_set_budget_loosen_emits_failure_budget_loosen_rejected() -> None:
    """When the pipeline's ``tighten_budget`` rejects with
    ``budget_loosen_rejected``, the dispatcher emits a non-terminal
    ``failure`` event with that error_type."""
    from sage.contracts.cost_tracker import BudgetUpdateResult

    class _FakePipeline:
        _active_context = object()  # truthy

        def tighten_budget(self, new_remaining_usd: float) -> BudgetUpdateResult:
            return BudgetUpdateResult(
                accepted=False, reason="budget_loosen_rejected",
                budget_usd=10.0, remaining=2.0, total_spent=8.0,
            )

    captured: list[dict[str, Any]] = []

    class _RecordingEventLog:
        def emit_failure(self, **kwargs: Any) -> None:
            captured.append({"failure": kwargs})

        def emit_budget(self, **kwargs: Any) -> None:
            captured.append({"budget": kwargs})

    cli_run._handle_set_budget(
        {"command": "set_budget", "args": {"budget_usd": 100.0}},
        pipeline=_FakePipeline(),
        event_log=_RecordingEventLog(),
    )
    assert len(captured) == 1
    assert "failure" in captured[0]
    assert captured[0]["failure"]["error_type"] == "budget_loosen_rejected"


def test_set_budget_accepted_emits_budget_event_kind_tightened() -> None:
    """Accepted tighten emits a ``budget(kind="budget_tightened", ...)`` event
    (NOT a CLI-shell event — protocol invariant: shell event set is fixed at 4)."""
    from sage.contracts.cost_tracker import BudgetUpdateResult

    class _FakePipeline:
        _active_context = object()

        def tighten_budget(self, new_remaining_usd: float) -> BudgetUpdateResult:
            return BudgetUpdateResult(
                accepted=True, reason="budget_tightened",
                budget_usd=5.0, remaining=2.0, total_spent=3.0,
            )

    captured: list[dict[str, Any]] = []

    class _RecordingEventLog:
        def emit_failure(self, **kwargs: Any) -> None:
            captured.append({"failure": kwargs})

        def emit_budget(self, **kwargs: Any) -> None:
            captured.append({"budget": kwargs})

    cli_run._handle_set_budget(
        {"command": "set_budget", "args": {"budget_usd": 2.0}},
        pipeline=_FakePipeline(),
        event_log=_RecordingEventLog(),
    )
    assert len(captured) == 1
    assert "budget" in captured[0]
    assert captured[0]["budget"]["kind"] == "budget_tightened"
    assert captured[0]["budget"]["budget_limit_usd"] == 5.0
    assert captured[0]["budget"]["budget_remaining_usd"] == 2.0
    assert captured[0]["budget"]["cost_so_far_usd"] == 3.0


# ────────────────────────────────────────────────────────────────────
# 3.6 cli_progress idle heartbeat (Stage C)
#
# Fake-clock tests: production uses ``loop.time()`` (asyncio's monotonic),
# tests inject ``now`` (or ``now_fn``) explicitly so the timer logic is
# fully deterministic. NO real 5/10-second sleeps anywhere.
# ────────────────────────────────────────────────────────────────────


def test_maybe_emit_cli_progress_does_not_emit_before_idle_threshold() -> None:
    """Below the 10s idle floor, no progress frame fires."""
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()
    state = cli_run._CliProgressState(
        stage="execute",
        started_at=0.0,
        last_non_progress_frame_at=0.0,
        last_progress_frame_at=0.0,
    )
    fired = cli_run._maybe_emit_cli_progress(
        stdout=stdout, run_id="R", seq_counter=counter,
        progress_state=state, now=5.0,
    )
    assert fired is False
    assert stdout.getvalue() == ""


def test_maybe_emit_cli_progress_fires_after_idle_threshold() -> None:
    """At >=10s idle, a single progress frame fires with the current stage."""
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()
    counter.next()  # simulate cli_started already at seq=0
    state = cli_run._CliProgressState(
        stage="execute",
        started_at=0.0,
        last_non_progress_frame_at=0.0,
        last_progress_frame_at=0.0,
    )
    fired = cli_run._maybe_emit_cli_progress(
        stdout=stdout, run_id="R", seq_counter=counter,
        progress_state=state, now=10.5,
    )
    assert fired is True
    frame = json.loads(stdout.getvalue().strip())
    assert frame["event_type"] == "cli_progress"
    assert frame["seq"] == 1
    assert frame["payload"]["stage"] == "execute"
    assert frame["payload"]["elapsed_ms"] == 10500
    assert "Still running" in frame["payload"]["human_readable"]
    # last_progress_frame_at advanced; last_non_progress_frame_at NOT
    # advanced — the load-bearing trap.
    assert state.last_progress_frame_at == 10.5
    assert state.last_non_progress_frame_at == 0.0


def test_maybe_emit_cli_progress_repeats_every_5s_while_idle() -> None:
    """While no non-progress frame arrives, heartbeat repeats every 5s."""
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()
    counter.next()  # cli_started seq=0
    state = cli_run._CliProgressState(
        stage="learn",
        started_at=0.0,
        last_non_progress_frame_at=0.0,
        last_progress_frame_at=0.0,
    )
    # First fire at t=10
    assert cli_run._maybe_emit_cli_progress(
        stdout=stdout, run_id="R", seq_counter=counter,
        progress_state=state, now=10.0,
    ) is True
    # 4s later: idle 14s but only 4s since last progress → SUPPRESS.
    assert cli_run._maybe_emit_cli_progress(
        stdout=stdout, run_id="R", seq_counter=counter,
        progress_state=state, now=14.0,
    ) is False
    # 6s after first: 5s since last progress → FIRE.
    assert cli_run._maybe_emit_cli_progress(
        stdout=stdout, run_id="R", seq_counter=counter,
        progress_state=state, now=15.0,
    ) is True
    # 5s later: should fire again.
    assert cli_run._maybe_emit_cli_progress(
        stdout=stdout, run_id="R", seq_counter=counter,
        progress_state=state, now=20.0,
    ) is True
    # Stdout has 3 progress frames seq 1, 2, 3.
    seqs = [json.loads(line)["seq"] for line in stdout.getvalue().splitlines()]
    assert seqs == [1, 2, 3]


def test_runtime_mirror_frame_resets_idle_guard_and_suppresses_progress() -> None:
    """A mirrored runtime event updates ``last_non_progress_frame_at`` so
    the next heartbeat tick re-starts the 10s idle clock from that point."""
    file = _RecordingFile()
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()
    counter.next()  # cli_started seq=0
    fake_now = [0.0]
    state = cli_run._CliProgressState(
        stage="boot",
        started_at=0.0,
        last_non_progress_frame_at=0.0,
        last_progress_frame_at=0.0,
    )
    tee = cli_run._CliMirrorSinkHandle(
        file, stdout, counter,
        progress_state=state,
        now_fn=lambda: fake_now[0],
    )
    # t=8: a mirrored runtime frame arrives; updates the guard.
    fake_now[0] = 8.0
    tee.write('{"event_type":"task_started","seq":0}\n')
    assert state.last_non_progress_frame_at == 8.0
    # t=15: only 7s since the last non-progress frame → idle threshold
    # NOT met. No heartbeat.
    fired = cli_run._maybe_emit_cli_progress(
        stdout=stdout, run_id="R", seq_counter=counter,
        progress_state=state, now=15.0,
    )
    assert fired is False
    # t=18.5: 10.5s since last non-progress → idle met, heartbeat fires.
    fired = cli_run._maybe_emit_cli_progress(
        stdout=stdout, run_id="R", seq_counter=counter,
        progress_state=state, now=18.5,
    )
    assert fired is True


def test_unparseable_mirror_write_does_NOT_reset_idle_guard() -> None:
    """Verbatim-fallback mirror writes (malformed JSON) must NOT reset the
    idle clock — they're not real protocol frames, just byte passthrough
    for forensic preservation."""
    file = _RecordingFile()
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()
    fake_now = [0.0]
    state = cli_run._CliProgressState(
        stage="boot", started_at=0.0,
        last_non_progress_frame_at=0.0, last_progress_frame_at=0.0,
    )
    tee = cli_run._CliMirrorSinkHandle(
        file, stdout, counter,
        progress_state=state,
        now_fn=lambda: fake_now[0],
    )
    fake_now[0] = 5.0
    tee.write("not-json\n")
    assert state.last_non_progress_frame_at == 0.0  # unchanged


def test_emit_cli_event_updates_last_non_progress_for_envelope_events() -> None:
    """``cli_started`` / ``cli_tool_request`` / ``cli_complete`` (non-progress
    CLI envelope events) update ``last_non_progress_frame_at`` so they
    reset the idle guard alongside mirrored runtime frames."""
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()
    state = cli_run._CliProgressState(
        stage="boot", started_at=0.0,
        last_non_progress_frame_at=0.0, last_progress_frame_at=0.0,
    )
    fake_now = [3.5]
    cli_run._emit_cli_event(
        stdout, "cli_tool_request", run_id="R",
        payload={"correlation_id": "c1"},
        seq=counter.next(),
        progress_state=state,
        now_fn=lambda: fake_now[0],
    )
    assert state.last_non_progress_frame_at == 3.5


def test_emit_cli_progress_does_NOT_update_last_non_progress() -> None:
    """The cli_progress emission path passes ``progress_state=None`` to
    ``_emit_cli_event`` so it doesn't accidentally update the idle guard
    (the load-bearing trap from cgpro Stage C lock — would forever
    reset the heartbeat into a 10s cadence)."""
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()
    counter.next()  # cli_started seq=0
    state = cli_run._CliProgressState(
        stage="execute", started_at=0.0,
        last_non_progress_frame_at=0.0, last_progress_frame_at=0.0,
    )
    cli_run._maybe_emit_cli_progress(
        stdout=stdout, run_id="R", seq_counter=counter,
        progress_state=state, now=10.0,
    )
    assert state.last_non_progress_frame_at == 0.0
    assert state.last_progress_frame_at == 10.0


@pytest.mark.asyncio
async def test_orchestrator_sets_progress_stage_labels_in_canonical_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_internal`` MUST call ``_set_cli_progress_stage(pipeline, ...)``
    before each high-level stage with one of the 6 canonical labels in
    the canonical order. cgpro Stage C lock 2026-05-07 acceptance test.

    The orchestrator's stage calls are LOCAL-imported (Phase 2.1
    discipline), so module-level monkeypatch on
    ``sage.pipeline_v2.orchestrator._set_cli_progress_stage`` intercepts
    every call site.
    """
    captured: list[str] = []

    def _capture(_pipeline: Any, stage: str) -> None:
        captured.append(stage)

    monkeypatch.setattr(
        "sage.pipeline_v2.orchestrator._set_cli_progress_stage", _capture,
    )

    # Build a minimal S1 single-agent pipeline inline (mirrors the
    # ``_single_agent_pipeline`` helper in test_pipeline_budget.py
    # without the cross-test import).
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from sage.llm.base import LLMResponse
    from sage.pipeline import CognitiveOrchestrationPipeline

    class _Router:
        system = 1

        def assess_complexity(self, task: str) -> Any:
            return SimpleNamespace(system=1)

        def route(self, profile: Any) -> Any:
            return SimpleNamespace(system=profile.system)

    class _Provider:
        async def generate(self, *args: Any, **kwargs: Any) -> LLMResponse:
            return LLMResponse(content="ok", model="stub")

    pipeline = CognitiveOrchestrationPipeline(
        router=_Router(), engine=None, assigner=None,
        provider_pool=MagicMock(), llm_provider=_Provider(),
    )
    await pipeline.run("trivial s1 task")

    # The orchestrator calls _set_cli_progress_stage UNCONDITIONALLY before
    # each of the 6 stages (decompose, assign_models etc. don't gate the
    # label call — they always fire even on S1 fast paths). Stage 5 (learn)
    # appears once per run on the success path; on the failure-recovery
    # path it could appear twice (see orchestrator try/except). For the
    # success-path test, the canonical sequence is exactly:
    canonical = [
        "classify", "decompose", "select_topology",
        "assign_models", "execute", "learn",
    ]
    assert captured == canonical, (
        f"stage labels diverged from canonical sequence: {captured}"
    )


@pytest.mark.asyncio
async def test_cli_progress_heartbeat_task_cancellable_before_terminal() -> None:
    """The heartbeat async task respects ``stop_event`` + ``cancel()`` and
    never emits a frame after cancellation. Stage C lock: no
    cli_progress after cli_complete (terminal-frame contract)."""
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()
    state = cli_run._CliProgressState(
        stage="execute", started_at=0.0,
        last_non_progress_frame_at=0.0, last_progress_frame_at=0.0,
    )
    stop_event = asyncio.Event()
    fake_now = [100.0]  # well past idle threshold

    async def _tick(_seconds: float) -> None:
        # Yield once so the task gets to enter its loop, then return.
        await asyncio.sleep(0)

    task = asyncio.create_task(
        cli_run._cli_progress_heartbeat(
            stdout=stdout, run_id="R", seq_counter=counter,
            progress_state=state, stop_event=stop_event,
            sleep_fn=_tick, now_fn=lambda: fake_now[0],
            idle_after_s=10.0, heartbeat_interval_s=5.0,
        )
    )
    # Let the loop run a few iterations.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    # Cancel + signal stop.
    stop_event.set()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # No subsequent emission can happen after the task is cancelled — write
    # something now and assert no NEW progress frames appear.
    bytes_at_cancel = len(stdout.getvalue())
    await asyncio.sleep(0)
    assert len(stdout.getvalue()) == bytes_at_cancel


def test_boot_initialization_prevents_first_heartbeat_before_idle_threshold() -> None:
    """After ``cli_started``, ``last_non_progress_frame_at`` is initialized
    to the cli-start time so the first boot-stage heartbeat cannot fire
    before the 10s idle threshold elapses (cgpro Stage C VERIFY round-2:
    boot-timing initialization regression test)."""
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()
    counter.next()  # cli_started just fired at seq=0, time t=cli_start_t.
    cli_start_t = 100.0
    state = cli_run._CliProgressState(
        stage="boot",
        started_at=cli_start_t,
        last_non_progress_frame_at=cli_start_t,
        last_progress_frame_at=cli_start_t,
    )
    # Tick through every second from t=cli_start_t to t=cli_start_t+9.99 and
    # assert no heartbeat ever fires before the idle threshold. That covers
    # the entire boot phase (typically 1-3s on this machine) without
    # assuming a specific boot duration.
    for offset in [0.0, 0.5, 1.0, 2.0, 5.0, 9.0, 9.99]:
        fired = cli_run._maybe_emit_cli_progress(
            stdout=stdout, run_id="R", seq_counter=counter,
            progress_state=state, now=cli_start_t + offset,
        )
        assert fired is False, f"heartbeat fired prematurely at offset={offset}"
    assert stdout.getvalue() == ""
    # At t = cli_start_t + 10.0 the threshold is met and the heartbeat fires
    # exactly once.
    assert cli_run._maybe_emit_cli_progress(
        stdout=stdout, run_id="R", seq_counter=counter,
        progress_state=state, now=cli_start_t + 10.0,
    ) is True


# ────────────────────────────────────────────────────────────────────
# 3.7 cancel command + cooperative v0 cancellation (Stage D)
#
# Direct asyncio integration: monkeypatch ``boot_agent_system`` to return
# a fake System whose pipeline.run hangs on an asyncio.Event; feed
# stdin a JSONL ``cancel`` command and assert the full stream contract:
# exactly one ``failure(kind="cli_cancel", error_type="cancelled")``
# frame followed by ``cli_complete(outcome="cancelled", exit_code=130)``,
# stream idempotency on double-cancel, no progress after terminal,
# final_seq reconciliation, exit code 130.
# ────────────────────────────────────────────────────────────────────


class _CancelHangingPipeline:
    """Fake Pipeline.run that hangs on an asyncio.Event.

    Tracks whether cancellation was observed (asyncio.CancelledError raised
    out of the wait). This is the v0 cooperative cancellation contract:
    Python's ``asyncio.Task.cancel()`` raises CancelledError at the next
    ``await`` boundary, which is exactly the Event.wait() here.
    """

    def __init__(self) -> None:
        self.cancelled = False
        self.last_context: Any = None
        self._never_set = asyncio.Event()
        self._active_context: Any = None
        self._cli_progress_state: Any = None

    async def run(
        self, task: str,
        budget_usd: float | None = None, system_hint: int | None = None,
    ) -> str:
        try:
            await self._never_set.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return "never reached"

    def tighten_budget(self, _new: float) -> Any:
        from sage.contracts.cost_tracker import BudgetUpdateResult
        return BudgetUpdateResult(
            accepted=False, reason="budget_before_prompt",
            budget_usd=0.0, remaining=0.0, total_spent=0.0,
        )


class _CancelFakeSystem:
    """Fake ``System`` returned by patched ``boot_agent_system``."""

    def __init__(self, pipeline: _CancelHangingPipeline) -> None:
        self.pipeline = pipeline


async def _drive_cancel_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdin_lines: list[str],
) -> tuple[int, str, _CancelHangingPipeline]:
    """Helper: spin up ``run_jsonl_async`` with a fake boot + fake stdin
    populated with ``stdin_lines`` (each is a complete JSONL command
    line ending in ``\\n``)."""
    fake_pipeline = _CancelHangingPipeline()

    def _fake_boot(*_args: Any, **_kwargs: Any) -> _CancelFakeSystem:
        return _CancelFakeSystem(fake_pipeline)

    # Patch the boot function. ``run_jsonl_async`` does
    # ``from sage.boot import boot_agent_system`` LOCALLY — patch
    # the source module so the local import picks up the fake.
    monkeypatch.setattr("sage.boot.boot_agent_system", _fake_boot)
    # Ensure no real .sage state interferes.
    monkeypatch.setenv("SAGE_BOOT_BYPASS_EPOCH_GUARD", "1")
    monkeypatch.setenv("SAGE_BOOT_BYPASS_REASON", "cli stage-D test")
    monkeypatch.setenv("SAGE_OPERATOR_ID", "test")

    stdin_buf = io.StringIO("".join(stdin_lines))
    stdout_buf = io.StringIO()
    exit_code = await cli_run.run_jsonl_async(
        "fake task", stdin=stdin_buf, stdout=stdout_buf,
    )
    return exit_code, stdout_buf.getvalue(), fake_pipeline


def _parse_jsonl_frames(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_main_prompt_command_starts_run_from_stdin_first_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first stdin JSONL ``prompt`` command starts the run in command mode."""
    calls: list[dict[str, Any]] = []

    async def _fake_run_jsonl_async(
        task: str,
        *,
        budget_usd: float | None = None,
        system_hint: int | None = None,
        stdin: Any = None,
        **_kwargs: Any,
    ) -> int:
        calls.append(
            {
                "task": task,
                "budget_usd": budget_usd,
                "system_hint": system_hint,
                "stdin": stdin,
            }
        )
        return 17

    stdin = io.StringIO(
        json.dumps(
            {
                "command": "prompt",
                "args": {"task": "hello", "budget_usd": 1.25, "system_hint": 2},
            }
        )
        + "\n"
    )
    monkeypatch.setattr("sys.stdin", stdin)
    monkeypatch.setattr(cli_run, "run_jsonl_async", _fake_run_jsonl_async)

    rc = cli_run.main(["--jsonl"])

    assert rc == 17
    assert calls == [
        {
            "task": "hello",
            "budget_usd": 1.25,
            "system_hint": 2,
            "stdin": stdin,
        }
    ]


def test_main_plain_stdin_batch_mode_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-command stdin remains the legacy one-shot task text mode."""
    calls: list[dict[str, Any]] = []

    async def _fake_run_jsonl_async(
        task: str,
        *,
        budget_usd: float | None = None,
        system_hint: int | None = None,
        **_kwargs: Any,
    ) -> int:
        calls.append(
            {"task": task, "budget_usd": budget_usd, "system_hint": system_hint}
        )
        return 0

    monkeypatch.setattr("sys.stdin", io.StringIO("first line\nsecond line\n"))
    monkeypatch.setattr(cli_run, "run_jsonl_async", _fake_run_jsonl_async)

    rc = cli_run.main(["--jsonl"])

    assert rc == 0
    assert calls == [
        {"task": "first line\nsecond line", "budget_usd": None, "system_hint": None}
    ]


def test_main_first_stdin_command_must_be_prompt(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Command-mode stdin starts only with ``prompt``."""
    called = False

    async def _fake_run_jsonl_async(*_args: Any, **_kwargs: Any) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps({"command": "cancel"}) + "\n")
    )
    monkeypatch.setattr(cli_run, "run_jsonl_async", _fake_run_jsonl_async)

    rc = cli_run.main(["--jsonl"])
    captured = capsys.readouterr()

    assert rc == 2
    assert called is False
    assert "first stdin command must be prompt" in captured.err


def test_main_prompt_command_requires_non_empty_task(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initial ``prompt`` must carry a non-empty task string."""
    called = False

    async def _fake_run_jsonl_async(*_args: Any, **_kwargs: Any) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps({"command": "prompt", "args": {"task": ""}}) + "\n"
        ),
    )
    monkeypatch.setattr(cli_run, "run_jsonl_async", _fake_run_jsonl_async)

    rc = cli_run.main(["--jsonl"])
    captured = capsys.readouterr()

    assert rc == 2
    assert called is False
    assert "prompt task must be a non-empty string" in captured.err


def test_main_prompt_command_rejects_argv_budget_loosen(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stdin prompt cannot loosen a process-level budget cap."""
    called = False

    async def _fake_run_jsonl_async(*_args: Any, **_kwargs: Any) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {
                    "command": "prompt",
                    "args": {"task": "hello", "budget_usd": 2.0},
                }
            )
            + "\n"
        ),
    )
    monkeypatch.setattr(cli_run, "run_jsonl_async", _fake_run_jsonl_async)

    rc = cli_run.main(["--jsonl", "--budget-usd", "1.0"])
    captured = capsys.readouterr()

    assert rc == 2
    assert called is False
    assert "prompt budget_usd cannot exceed --budget-usd" in captured.err


@pytest.mark.asyncio
async def test_cancel_emits_failure_then_terminal_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cancel`` command produces exactly one
    ``failure(kind="cli_cancel", error_type="cancelled")`` frame followed
    by ``cli_complete(outcome="cancelled", exit_code=130)`` — terminal."""
    exit_code, stdout, fake_pipeline = await _drive_cancel_run(
        monkeypatch,
        stdin_lines=[
            json.dumps({"command": "cancel", "args": {"reason": "user clicked"}}) + "\n",
        ],
    )
    assert exit_code == 130
    frames = _parse_jsonl_frames(stdout)
    # Runtime ``failure`` events from RuntimeEventLog use the cycle-7 R6.1c
    # FLAT redacted shape: ``kind`` / ``error_type`` are at the top level
    # (NOT nested under ``payload``). CLI envelope events (cli_started /
    # cli_complete) keep the ``payload`` nesting. Stage A's stdout seq
    # rewrite preserves this distinction.
    cancel_failures = [
        f for f in frames
        if f["event_type"] == "failure" and f.get("kind") == "cli_cancel"
    ]
    assert len(cancel_failures) == 1
    assert cancel_failures[0]["error_type"] == "cancelled"
    # ``message`` is hashed into payload_hash for the redacted forensic
    # archive but the flat-redacted stdout frame doesn't carry it. The
    # forensic file (under trace_dir) retains the full payload.
    # cli_complete is the LAST frame.
    assert frames[-1]["event_type"] == "cli_complete"
    assert frames[-1]["payload"]["outcome"] == "cancelled"
    assert frames[-1]["payload"]["exit_code"] == 130
    # The fake pipeline observed the CancelledError.
    assert fake_pipeline.cancelled is True


@pytest.mark.asyncio
async def test_cancel_final_seq_points_to_cancel_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cli_complete.payload.final_seq`` equals the cancel failure
    frame's stdout seq (the failure is the immediately previous
    stdout frame on the cancel path)."""
    exit_code, stdout, _ = await _drive_cancel_run(
        monkeypatch,
        stdin_lines=[json.dumps({"command": "cancel"}) + "\n"],
    )
    assert exit_code == 130
    frames = _parse_jsonl_frames(stdout)
    cli_complete = frames[-1]
    final_seq = cli_complete["payload"]["final_seq"]
    # The frame at seq=final_seq MUST be the cli_cancel failure.
    pen_ultimate = [f for f in frames if f["seq"] == final_seq]
    assert len(pen_ultimate) == 1
    assert pen_ultimate[0]["event_type"] == "failure"
    # Runtime failure shape is FLAT (cycle-7 R6.1c redacted form),
    # not nested under ``payload``.
    assert pen_ultimate[0].get("kind") == "cli_cancel"


@pytest.mark.asyncio
async def test_cancel_idempotent_double_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two ``cancel`` commands produce exactly ONE
    ``failure(kind="cli_cancel")`` and ONE ``cli_complete``. Stream-level
    idempotency, regardless of stdin-drain semantics."""
    exit_code, stdout, _ = await _drive_cancel_run(
        monkeypatch,
        stdin_lines=[
            json.dumps({"command": "cancel"}) + "\n",
            json.dumps({"command": "cancel"}) + "\n",
        ],
    )
    assert exit_code == 130
    frames = _parse_jsonl_frames(stdout)
    cancel_failures = [
        f for f in frames
        if f["event_type"] == "failure" and f.get("kind") == "cli_cancel"
    ]
    cli_completes = [f for f in frames if f["event_type"] == "cli_complete"]
    assert len(cancel_failures) == 1
    assert len(cli_completes) == 1


@pytest.mark.asyncio
async def test_cancel_before_pipeline_wait_still_cancels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cancel`` queued before pipeline_task starts waiting still cancels
    the run. The dispatcher sets ``cancel_requested`` during/before boot
    and the wait races immediately (FIRST_COMPLETED) on the cancel side."""
    exit_code, stdout, fake_pipeline = await _drive_cancel_run(
        monkeypatch,
        # Cancel arrives first — before any other command.
        stdin_lines=[json.dumps({"command": "cancel"}) + "\n"],
    )
    assert exit_code == 130
    frames = _parse_jsonl_frames(stdout)
    assert frames[-1]["event_type"] == "cli_complete"
    assert frames[-1]["payload"]["outcome"] == "cancelled"
    assert fake_pipeline.cancelled is True


@pytest.mark.asyncio
async def test_cancel_stops_progress_before_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``cli_progress`` frame appears AFTER the terminal
    ``cli_complete`` on the cancel path. The heartbeat task is cancelled
    + awaited in ``finally`` before ``cli_complete`` is emitted (Stage C
    contract preserved through the cancel-then-cli_complete sequence)."""
    exit_code, stdout, _ = await _drive_cancel_run(
        monkeypatch,
        stdin_lines=[json.dumps({"command": "cancel"}) + "\n"],
    )
    assert exit_code == 130
    frames = _parse_jsonl_frames(stdout)
    # Find cli_complete index; assert nothing follows.
    cli_complete_idx = next(
        i for i, f in enumerate(frames) if f["event_type"] == "cli_complete"
    )
    assert cli_complete_idx == len(frames) - 1, "cli_complete is not the last frame"


@pytest.mark.asyncio
async def test_run_jsonl_rejects_subsequent_prompt_with_nonterminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-start ``prompt`` is rejected without becoming a second run."""
    exit_code, stdout, _ = await _drive_cancel_run(
        monkeypatch,
        stdin_lines=[
            json.dumps({"command": "prompt", "args": {"task": "second"}}) + "\n",
            json.dumps({"command": "cancel"}) + "\n",
        ],
    )

    assert exit_code == 130
    frames = _parse_jsonl_frames(stdout)
    prompt_failures = [
        f for f in frames
        if f["event_type"] == "failure"
        and f.get("kind") == "cli_command"
        and f.get("error_type") == "prompt_already_started"
    ]
    assert len(prompt_failures) == 1
    assert frames[-1]["event_type"] == "cli_complete"


@pytest.mark.asyncio
async def test_run_jsonl_unknown_command_emits_nonterminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown commands are protocol-visible non-terminal failures."""
    exit_code, stdout, _ = await _drive_cancel_run(
        monkeypatch,
        stdin_lines=[
            json.dumps({"command": "bogus"}) + "\n",
            json.dumps({"command": "cancel"}) + "\n",
        ],
    )

    assert exit_code == 130
    frames = _parse_jsonl_frames(stdout)
    unknown_failures = [
        f for f in frames
        if f["event_type"] == "failure"
        and f.get("kind") == "cli_command"
        and f.get("error_type") == "unknown_command"
    ]
    assert len(unknown_failures) == 1
    assert frames[-1]["event_type"] == "cli_complete"


def test_prompt_mode_cli_started_first_and_cli_complete_last(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt command mode still preserves terminal stream invariants."""

    class _FastPipeline:
        last_context = None

        async def run(
            self,
            task: str,
            budget_usd: float | None = None,
            system_hint: int | None = None,
        ) -> str:
            assert task == "prompt task"
            assert budget_usd == 1.0
            assert system_hint == 2
            return "ok"

    class _FastSystem:
        pipeline = _FastPipeline()

    def _fake_boot(*_args: Any, **_kwargs: Any) -> _FastSystem:
        return _FastSystem()

    monkeypatch.setattr("sage.boot.boot_agent_system", _fake_boot)
    monkeypatch.setenv("SAGE_BOOT_BYPASS_EPOCH_GUARD", "1")
    monkeypatch.setenv("SAGE_BOOT_BYPASS_REASON", "cli prompt test")
    monkeypatch.setenv("SAGE_OPERATOR_ID", "test")
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {
                    "command": "prompt",
                    "args": {
                        "task": "prompt task",
                        "budget_usd": 1.0,
                        "system_hint": 2,
                    },
                }
            )
            + "\n"
        ),
    )

    rc = cli_run.main(["--jsonl"])
    captured = capsys.readouterr()

    assert rc == 0
    frames = _parse_jsonl_frames(captured.out)
    assert frames[0]["event_type"] == "cli_started"
    assert frames[0]["seq"] == 0
    assert frames[0]["payload"]["task"] == "prompt task"
    assert frames[-1]["event_type"] == "cli_complete"


def test_set_cli_progress_stage_helper_no_op_without_state() -> None:
    """``_set_cli_progress_stage`` is a no-op when no CLI is attached
    (running pipeline outside the CLI doesn't pay any cost)."""
    from sage.pipeline_v2.orchestrator import _set_cli_progress_stage

    class _NoCliPipeline:
        pass

    pipeline = _NoCliPipeline()
    # Does not raise; nothing to assert on the pipeline side.
    _set_cli_progress_stage(pipeline, "classify")
    # And when state IS present, it updates the label.
    pipeline._cli_progress_state = cli_run._CliProgressState(stage="boot")  # type: ignore[attr-defined]
    _set_cli_progress_stage(pipeline, "execute")
    assert pipeline._cli_progress_state.stage == "execute"  # type: ignore[attr-defined]


def test_mirror_falls_through_on_unparseable_line() -> None:
    """Mirror is best-effort: malformed JSON is written verbatim so the
    forensic file's bytes still reach stdout (parser bugs are caller-side)."""
    file = _RecordingFile()
    stdout = io.StringIO()
    counter = cli_run._StdoutSeqCounter()
    tee = cli_run._CliMirrorSinkHandle(file, stdout, counter)

    tee.write("not-json\n")

    assert file.writes == ["not-json\n"]
    assert stdout.getvalue() == "not-json\n"
    # No seq was rewritten — counter unchanged.
    assert tee.last_stdout_seq is None


# ────────────────────────────────────────────────────────────────────
# 4. Approval bridge round-trip
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_approval_bridge_approve_resolves_true() -> None:
    """``approve_tool_call`` with matching id → callback returns True."""
    stdout = io.StringIO()
    queue: "asyncio.Queue[Any]" = asyncio.Queue()
    bridge = cli_run._CliApprovalBridge(
        stdout=stdout,
        run_id="01TESTRUN0000000000000003",
        seq_counter=cli_run._SeqCounter(),
        command_queue=queue,
        timeout_s=2.0,
    )

    decision = SimpleNamespace(action="upgrade_model", node_id="n1", model_id="claude-sonnet")

    # Run the callback in the background; meanwhile dispatch an approve.
    cb_task = asyncio.create_task(bridge.callback(decision))
    await asyncio.sleep(0.01)  # let the callback emit cli_tool_request
    # Approval bridge stores correlation_id "approval-1" (next_id starts at 1).
    bridge.dispatch_command({"command": "approve_tool_call", "id": "approval-1"})

    result = await asyncio.wait_for(cb_task, timeout=1.0)

    assert result is True
    # Verify the cli_tool_request frame was emitted
    line = stdout.getvalue().rstrip("\n")
    frame = json.loads(line)
    assert frame["event_type"] == "cli_tool_request"
    assert frame["payload"]["correlation_id"] == "approval-1"
    assert frame["payload"]["action"] == "upgrade_model"


@pytest.mark.asyncio
async def test_approval_bridge_deny_resolves_false() -> None:
    """``deny_tool_call`` with matching id → callback returns False."""
    stdout = io.StringIO()
    queue: "asyncio.Queue[Any]" = asyncio.Queue()
    bridge = cli_run._CliApprovalBridge(
        stdout=stdout,
        run_id="01TESTRUN0000000000000004",
        seq_counter=cli_run._SeqCounter(),
        command_queue=queue,
        timeout_s=2.0,
    )

    decision = SimpleNamespace(action="prune_node", node_id="n2", model_id="")

    cb_task = asyncio.create_task(bridge.callback(decision))
    await asyncio.sleep(0.01)
    bridge.dispatch_command({"command": "deny_tool_call", "id": "approval-1"})

    result = await asyncio.wait_for(cb_task, timeout=1.0)
    assert result is False


@pytest.mark.asyncio
async def test_approval_bridge_timeout_defaults_to_deny() -> None:
    """If no inbound approve/deny arrives in ``timeout_s``, callback falls back to deny.

    Important for pi-mono consumers that disconnect mid-run — the runner must
    not hang indefinitely. v0 default is 60s; we override to a short value
    here for fast tests.
    """
    stdout = io.StringIO()
    queue: "asyncio.Queue[Any]" = asyncio.Queue()
    bridge = cli_run._CliApprovalBridge(
        stdout=stdout,
        run_id="01TESTRUN0000000000000005",
        seq_counter=cli_run._SeqCounter(),
        command_queue=queue,
        timeout_s=0.1,  # ← short timeout for the test
    )

    decision = SimpleNamespace(action="continue", node_id="n3", model_id="")

    result = await bridge.callback(decision)

    # Timed out → defaulted to deny (False)
    assert result is False


@pytest.mark.asyncio
async def test_approval_bridge_ignores_mismatched_correlation_id() -> None:
    """An ``approve_tool_call`` with a non-matching id is silently ignored
    (idempotent semantics) — does NOT crash, does NOT spuriously resolve
    other pending approvals.
    """
    stdout = io.StringIO()
    queue: "asyncio.Queue[Any]" = asyncio.Queue()
    bridge = cli_run._CliApprovalBridge(
        stdout=stdout,
        run_id="01TESTRUN0000000000000006",
        seq_counter=cli_run._SeqCounter(),
        command_queue=queue,
        timeout_s=0.5,
    )

    decision = SimpleNamespace(action="continue", node_id="n4", model_id="")
    cb_task = asyncio.create_task(bridge.callback(decision))
    await asyncio.sleep(0.01)

    # Wrong id — does NOT release the pending future.
    bridge.dispatch_command({"command": "approve_tool_call", "id": "approval-99"})
    await asyncio.sleep(0.05)
    assert not cb_task.done(), "wrong id MUST NOT resolve the pending approval"

    # Correct id releases it.
    bridge.dispatch_command({"command": "approve_tool_call", "id": "approval-1"})
    result = await asyncio.wait_for(cb_task, timeout=0.5)
    assert result is True


def test_approval_bridge_dispatch_returns_false_for_non_approval_commands() -> None:
    """``dispatch_command`` returns False for ``cancel`` / ``set_budget`` etc
    so the caller (the dispatcher loop) can route them to other handlers.
    """
    stdout = io.StringIO()
    queue: "asyncio.Queue[Any]" = asyncio.Queue()
    bridge = cli_run._CliApprovalBridge(
        stdout=stdout,
        run_id="01TESTRUN0000000000000007",
        seq_counter=cli_run._SeqCounter(),
        command_queue=queue,
    )

    assert bridge.dispatch_command({"command": "cancel"}) is False
    assert bridge.dispatch_command({"command": "set_budget", "args": {"budget_usd": 1.0}}) is False
    assert bridge.dispatch_command({"command": "prompt", "args": {"task": "x"}}) is False


# ────────────────────────────────────────────────────────────────────
# 5. argparse contract: --jsonl required, empty task returns 2
# ────────────────────────────────────────────────────────────────────


def test_main_requires_jsonl_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """``sage run`` (no flag) is reserved; v0 requires explicit ``--jsonl``."""
    rc = cli_run.main(["task text"])
    captured = capsys.readouterr()

    assert rc == 2
    assert "--jsonl is required" in captured.err


def test_main_empty_task_returns_2(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No positional + empty stdin → exits 2 with operator-readable message."""
    # Empty stdin
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    rc = cli_run.main(["--jsonl"])
    captured = capsys.readouterr()

    assert rc == 2
    assert "empty task" in captured.err


def test_main_help_does_not_boot_pipeline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``sage run --help`` exits cleanly without booting the pipeline.

    Boot is heavy (Rust extension load, provider pool init). ``--help`` must
    short-circuit at the argparse level before any imports of ``sage.boot``.
    """
    with pytest.raises(SystemExit) as exc_info:
        cli_run.main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "sage run" in captured.out
    assert "--jsonl" in captured.out
    assert "--budget-usd" in captured.out
    assert "--system-hint" in captured.out


# ────────────────────────────────────────────────────────────────────
# Integration smoke (lightweight): root dispatcher routes ``run`` to cli/run.py
# ────────────────────────────────────────────────────────────────────


def test_root_dispatcher_routes_run_subcommand(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sage run`` (via root dispatcher) reaches cli.run.main.

    Verifies the wire-up in ``sage/cli/__init__.py`` — independent of the
    implementation in cli.run.main itself.
    """
    captured_argv: dict[str, Any] = {}

    def _fake_main(argv: list[str]) -> int:
        captured_argv["argv"] = list(argv)
        return 7  # arbitrary distinguishable code

    monkeypatch.setattr(cli_run, "main", _fake_main)

    from sage.cli import main as root_main

    rc = root_main(["run", "--jsonl", "hello"])

    assert rc == 7
    assert captured_argv["argv"] == ["--jsonl", "hello"]
