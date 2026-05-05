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


def test_tee_sink_writes_to_both_file_and_stdout() -> None:
    """Every write is mirrored: file gets it, stdout gets it."""
    file = _RecordingFile()
    stdout = io.StringIO()
    tee = cli_run._CliMirrorSinkHandle(file, stdout)

    tee.write('{"event_type":"task_started","seq":0}\n')

    assert file.writes == ['{"event_type":"task_started","seq":0}\n']
    assert stdout.getvalue() == '{"event_type":"task_started","seq":0}\n'


def test_tee_sink_flush_propagates_to_both() -> None:
    """``flush()`` must hit both sinks so frames are visible to the consumer."""
    file = _RecordingFile()
    stdout = io.StringIO()
    tee = cli_run._CliMirrorSinkHandle(file, stdout)

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
    tee = cli_run._CliMirrorSinkHandle(file, stdout)

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
    tee = cli_run._CliMirrorSinkHandle(file, stdout)

    # Should NOT raise — the file write completes.
    n = tee.write("data\n")

    # File write succeeded; stdout was best-effort.
    assert file.writes == ["data\n"]
    assert n == len("data\n")


# ────────────────────────────────────────────────────────────────────
# 3. Seq counter monotonic without gaps
# ────────────────────────────────────────────────────────────────────


def test_seq_counter_monotonic() -> None:
    counter = cli_run._SeqCounter()
    seqs = [counter.next() for _ in range(5)]
    assert seqs == [0, 1, 2, 3, 4]


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
