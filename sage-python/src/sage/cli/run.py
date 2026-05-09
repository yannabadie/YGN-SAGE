"""``sage run --jsonl`` — machine-readable backend for the pi-mono pivot (v0).

See ``docs/contracts/SAGE_CLI_PROTOCOL.md`` for the full v0 spec. This module
implements the contract:

  - 15 inherited events from ``RuntimeEventLog`` v0 (tee'd to stdout, with
    stdout seq rewritten through the unified ``_StdoutSeqCounter``;
    forensic file keeps its own internal seq).
  - 4 CLI-shell envelope events emitted directly: ``cli_started``,
    ``cli_progress``, ``cli_tool_request``, ``cli_complete``.
  - 5 inbound commands parsed from stdin: ``prompt``, ``approve_tool_call``,
    ``deny_tool_call``, ``cancel``, ``set_budget``.
  - Strict JSONL: LF-only delimiters, UTF-8, fail-close on protocol_version
    mismatch.

Cycle-13 K post-Phase-2.2 closure status:
  - Stage A (`2d557b15`): unified stdout seq + ``cli_complete.payload.final_seq``.
  - Stage B (`7bd48c17`): tightening-only ``set_budget`` command wired
    through ``CostTracker.tighten_remaining_budget`` root guard.
  - Stage C (`2ce3c877`): ``cli_progress`` idle heartbeat — timer-based
    5s cadence with 10s idle guard, stage labels driven by orchestrator,
    no piggyback.
  - Stage D (this commit): cooperative Python cancellation hardening.
    The cancel path emits one terminal
    ``failure(kind="cli_cancel", error_type="cancelled")`` frame before
    ``cli_complete(outcome="cancelled", exit_code=130)``. Idempotent
    at the stream level.

Known v0 limitations (documented in ``docs/contracts/SAGE_CLI_PROTOCOL.md``):
  - Cancellation is cooperative at Python ``await`` boundaries.
    ``asyncio.Task.cancel()`` raises ``CancelledError`` at the next
    opportunity. In-flight provider HTTP calls, blocking tool calls,
    and Rust ``TopologyExecutor`` work do NOT support fine-grained
    interruption in v0. Frontends SHOULD show "cancellation
    requested" until ``cli_complete.payload.outcome == "cancelled"``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from sage.runtime.event_log.redaction import _hash_payload

CLI_PROTOCOL_VERSION = "v0"

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# CLI progress state (Stage C — cli_progress idle heartbeat)
#
# The state object is created in ``run_jsonl_async``, threaded into the
# mirror sink + ``_emit_cli_event`` so every NON-progress stdout frame
# updates ``last_non_progress_frame_at``. The heartbeat task reads the
# state through ``_maybe_emit_cli_progress`` and emits ``cli_progress``
# only when ``now - last_non_progress_frame_at >= idle_after_s`` AND
# ``now - last_progress_frame_at >= heartbeat_interval_s``. Two timestamps
# are required to prevent the progress emission itself from forever
# resetting the idle guard — that's the cgpro Stage C lock trap.
# ────────────────────────────────────────────────────────────────────────


@dataclass
class _CliProgressState:
    """CLI-owned progress state. Attached to the pipeline at runtime via
    ``setattr(pipeline, "_cli_progress_state", state)``. NOT a declared
    Pipeline attribute — keeps the public façade (293 LOC at
    Stage F closure, < 300 LOC HARD GATE) untouched.

    ``stage`` follows the canonical 7-label vocabulary:
    ``boot`` → ``classify`` → ``decompose`` → ``select_topology`` →
    ``assign_models`` → ``execute`` → ``learn``. The orchestrator updates
    it before each stage call via ``_set_cli_progress_stage(pipeline, stage)``.
    """

    stage: str = "boot"
    started_at: float = 0.0
    last_non_progress_frame_at: float = 0.0
    last_progress_frame_at: float = 0.0


@dataclass(frozen=True)
class _InitialPromptCommand:
    task: str
    budget_usd: float | None
    system_hint: int | None


def _maybe_emit_cli_progress(
    *,
    stdout: TextIO,
    run_id: str,
    seq_counter: "_StdoutSeqCounter",
    progress_state: _CliProgressState,
    now: float,
    idle_after_s: float = 10.0,
    heartbeat_interval_s: float = 5.0,
) -> bool:
    """Emit one ``cli_progress`` frame iff the idle conditions are met.

    Conditions (cgpro Stage C lock):
      - ``now - last_non_progress_frame_at >= idle_after_s`` — the stream
        has been silent for at least the idle threshold.
      - ``now - last_progress_frame_at >= heartbeat_interval_s`` — the
        previous heartbeat was at least one full interval ago.

    On emit: writes a ``cli_progress`` frame through the unified seq
    counter (so ``cli_complete.payload.final_seq`` reconciles correctly),
    updates ``last_progress_frame_at`` only (NOT
    ``last_non_progress_frame_at`` — that would forever reset the idle
    guard, the load-bearing trap). Returns ``True`` if a frame was
    emitted, ``False`` otherwise.
    """
    if now - progress_state.last_non_progress_frame_at < idle_after_s:
        return False
    if now - progress_state.last_progress_frame_at < heartbeat_interval_s:
        return False
    elapsed_ms = max(0, int((now - progress_state.started_at) * 1000))
    payload = {
        "stage": progress_state.stage,
        "elapsed_ms": elapsed_ms,
        "human_readable": f"Still running: {progress_state.stage}",
    }
    _emit_cli_event(
        stdout,
        "cli_progress",
        run_id=run_id,
        payload=payload,
        seq=seq_counter.next(),
        progress_state=None,  # cli_progress does NOT update last_non_progress.
    )
    progress_state.last_progress_frame_at = now
    return True


# ────────────────────────────────────────────────────────────────────────
# Sink mirroring: tees RuntimeEventLog file writes to stdout (LF-only)
# ────────────────────────────────────────────────────────────────────────


class _CliMirrorSinkHandle:
    """Tees writes to a forensic file AND stdout, with stdout-side seq rewrite.

    Mirror of the ``_SinkHandle`` interface from
    ``sage-python/src/sage/runtime/event_log/writer.py:110-136``. The forensic
    file is preserved BYTE-IDENTICAL — the runtime contract artifact keeps its
    own ``RuntimeEventLog`` internal seq domain. The stdout mirror, on the other
    hand, RE-NUMBERS each frame's ``seq`` field through a single per-run
    ``_StdoutSeqCounter`` shared with the CLI envelope emitter. This unifies
    the stdout stream into one monotonic seq domain so the frontend can
    reconcile via ``cli_complete.payload.final_seq`` (protocol invariant 5 at
    ``docs/contracts/SAGE_CLI_PROTOCOL.md`` — frontends use this as the
    drop-detection check).

    The write semantics: ``RuntimeEventLog._emit`` already appends ``"\\n"``
    to every JSON record. We split on the trailing newline, parse the JSON,
    rewrite ``seq``, re-serialize, and write to stdout. Parse failures
    fall through verbatim (best-effort: the forensic file already has the
    correct bytes; a malformed mirror line is preferable to silent loss).

    Per ``docs/contracts/SAGE_CLI_PROTOCOL.md``: stdout uses LF-only
    delimiters. On Windows, opening stdout in text mode would normally
    translate ``\\n`` → ``\\r\\n``, so callers MUST reconfigure stdout to
    ``newline=""`` (or use the binary buffer) before constructing the tee.
    See ``run_jsonl_async`` for the setup.
    """

    def __init__(
        self,
        file_handle: Any,
        stdout_stream: TextIO,
        stdout_seq_counter: "_StdoutSeqCounter",
        progress_state: _CliProgressState | None = None,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._file = file_handle
        self._stdout = stdout_stream
        self._stdout_seq_counter = stdout_seq_counter
        self._progress_state = progress_state
        self._now_fn = now_fn or time.monotonic
        self.last_stdout_seq: int | None = None

    def _rewrite_seq_for_stdout(self, value: str) -> str:
        """Parse ``value`` (one JSON line + ``\\n``), rewrite ``seq``, re-serialize.

        Falls back to verbatim ``value`` if parsing fails — the forensic file
        already has the correct line; the mirror is best-effort.
        """
        if not value:
            return value
        suffix = ""
        body = value
        if body.endswith("\n"):
            suffix = "\n"
            body = body[:-1]
        body_stripped = body.strip()
        if not body_stripped:
            return value
        try:
            frame = json.loads(body_stripped)
        except (ValueError, TypeError):
            return value
        if not isinstance(frame, dict) or "seq" not in frame:
            return value
        new_seq = self._stdout_seq_counter.next()
        frame["seq"] = new_seq
        frame["protocol_version"] = CLI_PROTOCOL_VERSION
        if "ts_ms" not in frame:
            timestamp_ns = frame.get("timestamp_ns")
            if (
                isinstance(timestamp_ns, int)
                and not isinstance(timestamp_ns, bool)
                and timestamp_ns >= 0
            ):
                frame["ts_ms"] = timestamp_ns // 1_000_000
            else:
                frame["ts_ms"] = int(time.time() * 1000)
        self.last_stdout_seq = new_seq
        return json.dumps(frame, separators=(",", ":")) + suffix

    def write(self, value: str) -> int:
        # File first (forensic archive — runtime contract artifact, preserved
        # byte-identical with the original RuntimeEventLog seq domain). The
        # file write MUST succeed; if it fails the runtime contract is
        # violated and RuntimeEventLog's ``_handle_sink_failure`` will
        # take over (it's wrapped above the _SinkHandle).
        n = self._file.write(value)
        # Stdout mirror — CLI protocol consumer. The seq is rewritten into
        # the unified per-run stdout domain; bytes downstream of ``"seq":N``
        # are otherwise preserved.
        seq_before = self.last_stdout_seq
        rewritten = self._rewrite_seq_for_stdout(value)
        seq_after = self.last_stdout_seq
        try:
            self._stdout.write(rewritten)
            # Flush stdout immediately so frames are visible to the
            # consumer without waiting for buffer flush.
            self._stdout.flush()
        except (OSError, ValueError):
            # stdout closed mid-run (e.g. pi-mono frontend hang-up).
            # Best-effort: keep writing to the file. Subsequent writes
            # will keep trying — the OS will keep raising EPIPE; we keep
            # eating it.
            pass
        # Stage C: a successfully-parsed mirrored runtime frame counts as
        # a non-progress stdout frame for the heartbeat idle guard.
        # ``seq_after != seq_before`` is the parse-success witness — the
        # rewrite only advances ``last_stdout_seq`` when JSON parse +
        # ``"seq"`` extraction succeeded. Verbatim-fallback writes
        # (malformed JSON) leave ``last_stdout_seq`` unchanged and
        # therefore don't reset the idle clock.
        if self._progress_state is not None and seq_after != seq_before:
            self._progress_state.last_non_progress_frame_at = self._now_fn()
        return n

    def flush(self) -> None:
        try:
            self._file.flush()
        except (OSError, ValueError):
            pass
        try:
            self._stdout.flush()
        except (OSError, ValueError):
            pass

    @property
    def closed(self) -> bool:
        return self._file.closed

    def fileno(self) -> int:
        return self._file.fileno()

    def tell(self) -> int:
        return self._file.tell()

    def truncate(self, size: int | None = None) -> int:
        return self._file.truncate(size)

    def close(self) -> None:
        # Close the file but NOT stdout (the process owns it).
        self._file.close()


# ────────────────────────────────────────────────────────────────────────
# CLI-shell envelope events (cli_started, cli_progress, cli_tool_request,
# cli_complete) — protocol-layer, NOT runtime-layer.
# ────────────────────────────────────────────────────────────────────────


def _emit_cli_event(
    stream: TextIO,
    event_type: str,
    *,
    run_id: str,
    payload: Mapping[str, Any],
    seq: int,
    progress_state: _CliProgressState | None = None,
    now_fn: Callable[[], float] | None = None,
) -> None:
    """Emit a CLI-shell envelope event directly (bypasses RuntimeEventLog).

    Per ``docs/contracts/SAGE_CLI_PROTOCOL.md``: the 4 CLI-shell events are
    protocol-layer, NOT runtime-layer. They MUST NOT appear in
    ``RuntimeEventLog`` v0's ``EVENT_TYPES``. Their schema is independently
    versioned (``cli_v1`` initially).

    When ``progress_state`` is provided AND ``event_type != "cli_progress"``,
    ``last_non_progress_frame_at`` is updated to ``now_fn()`` so the
    heartbeat idle guard sees this frame. ``cli_progress`` itself MUST
    pass ``progress_state=None`` (its caller, ``_maybe_emit_cli_progress``,
    updates ``last_progress_frame_at`` directly) — otherwise the idle
    guard would forever reset and the heartbeat would never repeat.
    """
    frame = {
        "protocol_version": CLI_PROTOCOL_VERSION,
        "event_type": event_type,
        "seq": seq,
        "run_id": run_id,
        "ts_ms": int(time.time() * 1000),
        "payload_schema_version": "cli_v1",
        "payload": dict(payload),
    }
    stream.write(json.dumps(frame, separators=(",", ":")) + "\n")
    try:
        stream.flush()
    except (OSError, ValueError):
        pass
    if progress_state is not None and event_type != "cli_progress":
        progress_state.last_non_progress_frame_at = (
            now_fn() if now_fn is not None else time.monotonic()
        )


def _emit_cli_cancel_failure_fallback(
    stream: TextIO,
    *,
    run_id: str,
    seq: int,
    reason: str,
) -> None:
    """Emit the flat v0 cancel failure on stdout if RuntimeEventLog misses it."""
    _ = reason  # operator-readable text is forensic-only on the primary path.
    redacted_payload = {
        "kind": "cli_cancel",
        "error_type": "cancelled",
        "message": "<redacted>",
    }
    frame = {
        "protocol_version": CLI_PROTOCOL_VERSION,
        "event_type": "failure",
        "seq": seq,
        "run_id": run_id,
        "ts_ms": int(time.time() * 1000),
        "payload_schema_version": "v1",
        "kind": "cli_cancel",
        "error_type": "cancelled",
        "node_id": "",
        "redaction_state": "redacted",
        "payload_hash": _hash_payload("failure", redacted_payload),
    }
    stream.write(json.dumps(frame, separators=(",", ":")) + "\n")
    try:
        stream.flush()
    except (OSError, ValueError):
        pass


# ────────────────────────────────────────────────────────────────────────
# Stdin command parser
# ────────────────────────────────────────────────────────────────────────


async def _read_stdin_commands(
    stdin: TextIO,
    queue: "asyncio.Queue[Mapping[str, Any] | None]",
    stop_event: threading.Event,
) -> None:
    """Read JSONL commands from stdin and enqueue them.

    One command per line. Parse errors are NON-fatal — the malformed line is
    skipped (no failure event is emitted to avoid amplifying garbage). EOF on
    stdin closes the queue with ``None``. ``stop_event`` is a thread-safe flag
    because the reader loop runs in an OS thread.
    """
    loop = asyncio.get_running_loop()
    finished = asyncio.Event()

    def _put(item: Mapping[str, Any] | None) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            pass

    def _mark_finished() -> None:
        try:
            loop.call_soon_threadsafe(finished.set)
        except RuntimeError:
            pass

    def _reader() -> None:
        try:
            while not stop_event.is_set():
                try:
                    line = stdin.readline()
                except (OSError, ValueError):
                    break
                if not line:
                    _put(None)
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("sage run: stdin parse error on line: %r", line[:120])
                    continue
                if not isinstance(cmd, dict) or "command" not in cmd:
                    log.warning(
                        "sage run: stdin command missing 'command' field: %r",
                        line[:120],
                    )
                    continue
                _put(cmd)
        finally:
            _mark_finished()

    # Never use the loop's default executor for stdin.readline(): if the
    # frontend keeps stdin open after cancel, Python waits for the default
    # executor during asyncio.run() shutdown and the process hangs after the
    # terminal cli_complete frame. A daemon reader can stay blocked on the pipe
    # without delaying process exit; command delivery still happens through the
    # event loop via call_soon_threadsafe.
    reader = threading.Thread(
        target=_reader,
        name="sage-cli-stdin-reader",
        daemon=True,
    )
    reader.start()
    try:
        await finished.wait()
    except asyncio.CancelledError:
        stop_event.set()
        raise


# ────────────────────────────────────────────────────────────────────────
# Approval callback bridge: TopologyRunner.approval_callback ↔ stdin commands
# ────────────────────────────────────────────────────────────────────────


class _CliApprovalBridge:
    """Bridges ``TopologyRunner.approval_callback`` to the cli_tool_request
    sub-protocol.

    When the runner asks for approval, we:
      1. emit ``cli_tool_request`` with a fresh ``correlation_id``;
      2. await an inbound ``approve_tool_call`` / ``deny_tool_call`` whose
         ``id`` matches; or
      3. fall back to "deny" after ``timeout`` seconds (default 60s — pi-mono
         consumers that disconnect should not hang the run).

    The actual TopologyRunner ``approval_callback`` signature is
    ``async (decision) -> bool``. We translate `True`→approved, `False`→denied.
    """

    def __init__(
        self,
        stdout: TextIO,
        run_id: str,
        seq_counter: "_SeqCounter",
        command_queue: "asyncio.Queue[Mapping[str, Any] | None]",
        timeout_s: float = 60.0,
    ) -> None:
        self._stdout = stdout
        self._run_id = run_id
        self._seq_counter = seq_counter
        self._queue = command_queue
        self._timeout_s = timeout_s
        self._next_id = 0
        # Pending requests keyed by correlation_id; value is an asyncio.Future
        # the dispatcher (read_stdin_commands) resolves on matching reply.
        self._pending: dict[str, asyncio.Future[bool]] = {}

    async def callback(self, decision: Any) -> bool:
        """The async approval callback wired into ``TopologyRunner``."""
        self._next_id += 1
        correlation_id = f"approval-{self._next_id}"
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[correlation_id] = fut

        action = getattr(decision, "action", "<unknown>")
        node_id = getattr(decision, "node_id", "")
        model_id = getattr(decision, "model_id", "")

        _emit_cli_event(
            self._stdout,
            "cli_tool_request",
            run_id=self._run_id,
            payload={
                "correlation_id": correlation_id,
                "action": str(action),
                "node_id": str(node_id),
                "model_id": str(model_id),
                "details_redacted": "<see runtime event for full details>",
            },
            seq=self._seq_counter.next(),
        )

        try:
            return await asyncio.wait_for(fut, timeout=self._timeout_s)
        except asyncio.TimeoutError:
            log.warning(
                "sage run: approval timed out after %.1fs; defaulting to deny "
                "(correlation_id=%s)",
                self._timeout_s,
                correlation_id,
            )
            return False
        finally:
            self._pending.pop(correlation_id, None)

    def dispatch_command(self, cmd: Mapping[str, Any]) -> bool:
        """Route an inbound command. Returns True if the command was an
        approval response; False otherwise (caller should handle)."""
        verb = cmd.get("command")
        if verb not in ("approve_tool_call", "deny_tool_call"):
            return False
        correlation_id = cmd.get("id")
        if not isinstance(correlation_id, str):
            return True
        fut = self._pending.get(correlation_id)
        if fut is None or fut.done():
            return True
        fut.set_result(verb == "approve_tool_call")
        return True


# ────────────────────────────────────────────────────────────────────────
# Stdout seq counter — unified per-run monotonic across CLI envelope frames
# and mirrored RuntimeEventLog frames. The forensic file keeps its own
# RuntimeEventLog internal seq; only the stdout mirror is renumbered, so
# the frontend can reconcile via cli_complete.payload.final_seq (protocol
# invariant 5).
# ────────────────────────────────────────────────────────────────────────


class _StdoutSeqCounter:
    """Monotonic counter for ALL stdout frames (CLI envelope + mirrored runtime).

    The terminal ``cli_complete.payload.final_seq`` reads ``self.last`` to
    find the seq of the frame immediately preceding ``cli_complete`` — this
    is correct for both the success path (where the last mirrored frame is
    ``run_frame_summary``) AND the cancel/failure paths (where the last
    frame may be any CLI-shell event, e.g. ``cli_tool_request``).
    """

    def __init__(self) -> None:
        self._n = 0
        self._last: int | None = None

    def next(self) -> int:
        n = self._n
        self._n += 1
        self._last = n
        return n

    @property
    def last(self) -> int | None:
        """Seq of the last frame issued through ``next()``; ``None`` if never called."""
        return self._last


# Backward-compat alias for tests / external callers that imported
# ``_SeqCounter``. Same behavior; the stdout-domain rename happens in
# ``cli_gaps_stage_a``.
_SeqCounter = _StdoutSeqCounter


# ────────────────────────────────────────────────────────────────────────
# set_budget command handler (Stage B — tightening-only contract)
# ────────────────────────────────────────────────────────────────────────


def _handle_set_budget(
    cmd: Mapping[str, Any],
    *,
    pipeline: Any,
    event_log: Any,
) -> None:
    """Apply a ``set_budget`` command, emitting a runtime event on the result.

    Per ``docs/contracts/SAGE_CLI_PROTOCOL.md`` invariant 7
    (TIGHTEN-ONLY) and Stage B lock 2026-05-07:

      - Pre-run (``pipeline._active_context is None``): emit
        ``failure(error_type="budget_before_prompt")``.
      - ``args.budget_usd`` missing or non-numeric: emit
        ``failure(error_type="budget_invalid_value")``.
      - Otherwise call ``pipeline.tighten_budget(new_remaining_usd)``;
        accept emits a ``budget(kind="budget_tightened", ...)`` event,
        reject emits ``failure(kind="cli_command", error_type=<reason>)``.
    """
    args_obj = cmd.get("args")
    args: Mapping[str, Any] = args_obj if isinstance(args_obj, Mapping) else {}
    raw_budget = args.get("budget_usd")
    if not isinstance(raw_budget, (int, float)) or isinstance(raw_budget, bool):
        if event_log is not None:
            event_log.emit_failure(
                kind="cli_command",
                error_type="budget_invalid_value",
                message=(
                    "set_budget rejected: 'budget_usd' must be a finite number"
                ),
            )
        return
    if pipeline is None or getattr(pipeline, "_active_context", None) is None:
        if event_log is not None:
            event_log.emit_failure(
                kind="cli_command",
                error_type="budget_before_prompt",
                message="set_budget rejected before the run started",
            )
        return
    result = pipeline.tighten_budget(float(raw_budget))
    if not result.accepted:
        if event_log is not None:
            event_log.emit_failure(
                kind="cli_command",
                error_type=result.reason,
                message=(
                    f"set_budget rejected: {result.reason} "
                    f"(budget_usd={result.budget_usd}, remaining={result.remaining})"
                ),
            )
        return
    if event_log is not None:
        event_log.emit_budget(
            kind="budget_tightened",
            budget_limit_usd=float(result.budget_usd),
            budget_remaining_usd=float(result.remaining),
            cost_so_far_usd=float(result.total_spent),
        )


# ────────────────────────────────────────────────────────────────────────
# Background heartbeat task (Stage C)
# ────────────────────────────────────────────────────────────────────────


async def _cli_progress_heartbeat(
    *,
    stdout: TextIO,
    run_id: str,
    seq_counter: "_StdoutSeqCounter",
    progress_state: _CliProgressState,
    stop_event: asyncio.Event,
    sleep_fn: Callable[[float], "asyncio.Future[None] | Any"] | None = None,
    now_fn: Callable[[], float] | None = None,
    idle_after_s: float = 10.0,
    heartbeat_interval_s: float = 5.0,
) -> None:
    """Background task: poll every ``heartbeat_interval_s`` and emit a
    ``cli_progress`` frame iff ``_maybe_emit_cli_progress`` returns True.

    Uses the asyncio loop's monotonic clock by default (``loop.time()``).
    Tests inject ``now_fn`` + ``sleep_fn`` for deterministic timing.
    The strong reference to this task MUST be retained by the caller and
    ``cancel()``-ed in ``finally`` before ``cli_complete`` (per cgpro
    Stage C lock + Python asyncio docs warning on GC of background tasks).
    """
    if sleep_fn is None:
        sleep_fn = asyncio.sleep
    if now_fn is None:
        loop = asyncio.get_event_loop()
        now_fn = loop.time
    while not stop_event.is_set():
        try:
            await sleep_fn(heartbeat_interval_s)
        except asyncio.CancelledError:
            return
        if stop_event.is_set():
            return
        _maybe_emit_cli_progress(
            stdout=stdout,
            run_id=run_id,
            seq_counter=seq_counter,
            progress_state=progress_state,
            now=now_fn(),
            idle_after_s=idle_after_s,
            heartbeat_interval_s=heartbeat_interval_s,
        )


def _set_cli_progress_stage(pipeline: Any, stage: str) -> None:
    """Update the CLI progress stage label if the pipeline has a state.

    Called by ``pipeline_v2.orchestrator.run_internal`` before each
    high-level stage. Safe to call when no CLI is attached — the
    ``getattr`` falls through to ``None`` and the helper is a no-op
    (running outside the CLI does not pay the cost of a state object).
    """
    state = getattr(pipeline, "_cli_progress_state", None)
    if state is not None:
        state.stage = stage


# ────────────────────────────────────────────────────────────────────────
# Main entry: run_jsonl_async + main(argv)
# ────────────────────────────────────────────────────────────────────────


def _looks_like_stdin_command(line: str) -> bool:
    try:
        parsed = json.loads(line)
    except (TypeError, ValueError):
        return False
    return isinstance(parsed, dict) and "command" in parsed


def _parse_initial_prompt_command(line: str) -> _InitialPromptCommand:
    try:
        parsed = json.loads(line)
    except (TypeError, ValueError) as exc:
        raise ValueError("first stdin command must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("first stdin command must be a JSON object")
    if parsed.get("command") != "prompt":
        raise ValueError("first stdin command must be prompt")

    args = parsed.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("prompt args must be an object")

    task = args.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("prompt task must be a non-empty string")

    budget_usd: float | None = None
    if "budget_usd" in args:
        raw_budget = args["budget_usd"]
        if not isinstance(raw_budget, (int, float)) or isinstance(raw_budget, bool):
            raise ValueError("prompt budget_usd must be a finite positive number")
        budget_usd = float(raw_budget)
        if not math.isfinite(budget_usd) or budget_usd <= 0.0:
            raise ValueError("prompt budget_usd must be a finite positive number")

    system_hint: int | None = None
    if "system_hint" in args:
        raw_hint = args["system_hint"]
        if not isinstance(raw_hint, int) or isinstance(raw_hint, bool):
            raise ValueError("prompt system_hint must be one of 1, 2, 3")
        if raw_hint not in (1, 2, 3):
            raise ValueError("prompt system_hint must be one of 1, 2, 3")
        system_hint = raw_hint

    return _InitialPromptCommand(
        task=task.strip(), budget_usd=budget_usd, system_hint=system_hint,
    )


def _resolve_prompt_budget(
    *, argv_budget_usd: float | None, prompt_budget_usd: float | None,
) -> float | None:
    if prompt_budget_usd is None:
        return argv_budget_usd
    if argv_budget_usd is None:
        return prompt_budget_usd
    if prompt_budget_usd > argv_budget_usd:
        raise ValueError("prompt budget_usd cannot exceed --budget-usd")
    return prompt_budget_usd


def _resolve_prompt_system_hint(
    *, argv_system_hint: int | None, prompt_system_hint: int | None,
) -> int | None:
    if prompt_system_hint is None:
        return argv_system_hint
    if argv_system_hint is None:
        return prompt_system_hint
    if prompt_system_hint != argv_system_hint:
        raise ValueError("prompt system_hint must match --system-hint")
    return prompt_system_hint


async def run_jsonl_async(
    task: str,
    *,
    budget_usd: float | None = None,
    system_hint: int | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Run pipeline.run() with JSONL output on stdout.

    Returns exit code (0=success, 1=failure, 130=cancelled).
    """
    if stdin is None:
        stdin = sys.stdin
    if stdout is None:
        stdout = sys.stdout

    # Ensure stdout uses LF-only line endings on Windows. Without this,
    # text-mode stdout on Windows translates ``\n`` to ``\r\n``, which
    # breaks pi-mono's strict-JSONL parser.
    if hasattr(stdout, "reconfigure"):
        try:
            stdout.reconfigure(newline="")
        except (TypeError, OSError):
            pass

    # Lazy imports — avoid loading heavy boot machinery on argparse-only
    # invocations like ``sage run --help``.
    from sage.boot import boot_agent_system
    from sage.runtime.event_log import RuntimeEventLog, install_event_log
    from sage.pipeline import _new_runtime_run_id

    seq_counter = _StdoutSeqCounter()
    run_id = _new_runtime_run_id()

    # Stage C: CLI-owned progress state. Started immediately so the boot
    # phase has liveness too (heartbeat task starts after cli_started, sees
    # ``stage="boot"`` until orchestrator updates it). ``loop.time()`` is
    # the canonical clock per asyncio docs.
    loop = asyncio.get_event_loop()
    progress_state = _CliProgressState(
        stage="boot",
        started_at=loop.time(),
        last_non_progress_frame_at=loop.time(),
        last_progress_frame_at=loop.time(),
    )

    # Emit cli_started BEFORE booting (so consumers see "sage is starting"
    # immediately even if boot takes seconds).
    sage_version = "0.1.0"  # TODO(cycle-12): pull from pyproject.toml
    sage_commit_sha = os.environ.get("SAGE_COMMIT_SHA", "")
    _emit_cli_event(
        stdout,
        "cli_started",
        run_id=run_id,
        payload={
            "protocol_version": CLI_PROTOCOL_VERSION,
            "sage_version": sage_version,
            "sage_commit_sha": sage_commit_sha,
            "task": task[:200],  # cap to avoid huge frames
            "budget_usd": budget_usd,
            "system_hint": system_hint,
            "tier": os.environ.get("SAGE_LLM_TIER", ""),
        },
        seq=seq_counter.next(),
        progress_state=progress_state,
        now_fn=loop.time,
    )

    # Tempdir for the forensic JSONL archive (RuntimeEventLog file). The
    # mirror tee will replicate writes to stdout. We keep the file even
    # on cancel for post-mortem inspection.
    trace_dir = Path(tempfile.mkdtemp(prefix="sage_cli_run_"))
    eventlog = RuntimeEventLog(run_id=run_id, trace_dir=trace_dir)

    # If the writer initialized successfully (sink is open), splice in
    # the stdout-mirror tee. If the writer is disabled (e.g. permissions),
    # we still emit cli_started/cli_complete but no runtime events.
    #
    # Cycle-12 prelude (cgpro CI debug 2026-05-05): the assignment line
    # carries `[assignment]` (not `[attr-defined]`) because mypy sees the
    # type drift `_SinkHandle | None` → `_CliMirrorSinkHandle`. The
    # `_CliMirrorSinkHandle` IS a structural subtype of `_SinkHandle`
    # (matching write/flush/close/closed/fileno/tell/truncate), but
    # since `_SinkHandle` is a concrete class not a Protocol, mypy
    # requires the assignment cast. Switching `_SinkHandle` to a
    # `typing.Protocol` is an option for cycle-12 Phase B; for now
    # the targeted ignore narrows the type-safety escape to one site.
    mirror_sink: _CliMirrorSinkHandle | None = None
    if eventlog._fh is not None:  # type: ignore[attr-defined]
        mirror_sink = _CliMirrorSinkHandle(
            eventlog._fh, stdout, seq_counter,
            progress_state=progress_state,
            now_fn=loop.time,
        )
        eventlog._fh = mirror_sink  # type: ignore[assignment]

    # Stage C: heartbeat task. Strong reference retained so it isn't GC'd
    # mid-run; cancelled in ``finally`` before cli_complete (see Python
    # asyncio task-lifetime docs + cgpro Stage C lock).
    heartbeat_stop = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        _cli_progress_heartbeat(
            stdout=stdout,
            run_id=run_id,
            seq_counter=seq_counter,
            progress_state=progress_state,
            stop_event=heartbeat_stop,
        )
    )

    log_token = install_event_log(eventlog)

    # Stdin command queue + reader task.
    command_queue: asyncio.Queue[Mapping[str, Any] | None] = asyncio.Queue()
    stop_stdin_event = threading.Event()
    stdin_task = asyncio.create_task(
        _read_stdin_commands(stdin, command_queue, stop_stdin_event)
    )

    # Approval bridge — wired into the system after boot.
    approval_bridge = _CliApprovalBridge(
        stdout=stdout,
        run_id=run_id,
        seq_counter=seq_counter,
        command_queue=command_queue,
    )

    # Cancel handling: the dispatcher cancels the pipeline task when
    # ``cancel`` arrives.
    cancel_requested = asyncio.Event()

    # Pipeline reference shared with the dispatcher. Populated after boot
    # below so that set_budget arriving early (before prompt fully starts)
    # is rejected with reason="budget_before_prompt" (Stage B lock).
    pipeline_for_dispatcher: list[Any] = [None]

    # Stage D: cancel reason captured from ``args.reason``. Surfaced in the
    # terminal ``failure(kind="cli_cancel")`` frame's ``message`` field so
    # the operator-readable reason is recorded both forensically and on the
    # stdout protocol stream.
    cancel_reason: list[str] = [""]

    async def _command_dispatcher() -> None:
        while True:
            cmd = await command_queue.get()
            if cmd is None:  # EOF on stdin
                return
            verb = cmd.get("command")
            # Approval responses route to the bridge first.
            if approval_bridge.dispatch_command(cmd):
                continue
            if verb == "cancel":
                args = cmd.get("args")
                if isinstance(args, Mapping):
                    raw_reason = args.get("reason")
                    if isinstance(raw_reason, str) and raw_reason:
                        cancel_reason[0] = raw_reason
                cancel_requested.set()
                return
            if verb == "set_budget":
                _handle_set_budget(
                    cmd,
                    pipeline=pipeline_for_dispatcher[0],
                    event_log=eventlog,
                )
                continue
            if verb == "prompt":
                eventlog.emit_failure(
                    kind="cli_command",
                    error_type="prompt_already_started",
                    message="prompt command rejected after run started",
                )
                continue
            eventlog.emit_failure(
                kind="cli_command",
                error_type="unknown_command",
                message=f"unsupported command: {verb!r}",
            )

    dispatcher_task = asyncio.create_task(_command_dispatcher())

    exit_code = 0
    outcome = "success"
    total_cost_usd = 0.0
    total_latency_ms = 0.0

    pipeline_task: asyncio.Task[Any] | None = None
    try:
        # Boot the system. boot_agent_system returns a System object whose
        # .pipeline.run(task, ...) is the canonical entry point.
        system = boot_agent_system()
        pipeline = getattr(system, "pipeline", None)
        if pipeline is None:
            raise RuntimeError("boot_agent_system did not return a pipeline")
        # Share the pipeline with the dispatcher so set_budget can reach
        # ``pipeline.tighten_budget`` (Stage B lock). The dispatcher reads
        # ``pipeline_for_dispatcher[0]`` lazily on each command, so runs
        # that finish before any set_budget arrives never observe it.
        pipeline_for_dispatcher[0] = pipeline
        # Stage C: attach the CLI progress state so ``run_internal`` can
        # update ``state.stage`` before each high-level stage. ``setattr``
        # at runtime keeps the public ``Pipeline`` façade (293 LOC at
        # Stage F closure, < 300 LOC HARD GATE) untouched per cgpro
        # Stage C lock.
        setattr(pipeline, "_cli_progress_state", progress_state)

        # Wire the approval callback into the topology runner. The runner
        # is constructed per-run inside _stage_execute, so we set a default
        # on the system that the pipeline picks up.
        # The cleanest path here is via a system attribute that
        # TopologyRunner reads. For the prelude POC, we leave the runner
        # callback as None and surface approval via cli_tool_request only
        # on bypass paths (see TODO below).
        #
        # TODO(cycle-12 Phase B): plumb approval_bridge.callback into
        # TopologyRunner via the agent_loop_factory_kwargs path.

        # Run the pipeline, racing against cancel.
        async def _run() -> str:
            return await pipeline.run(
                task, budget_usd=budget_usd, system_hint=system_hint,
            )

        pipeline_task = asyncio.create_task(_run())
        cancel_wait = asyncio.create_task(cancel_requested.wait())
        done, pending = await asyncio.wait(
            {pipeline_task, cancel_wait}, return_when=asyncio.FIRST_COMPLETED,
        )

        if cancel_wait in done and not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except (asyncio.CancelledError, Exception):
                pass
            outcome = "cancelled"
            exit_code = 130
        else:
            cancel_wait.cancel()
            try:
                _result = pipeline_task.result()
                outcome = "success"
            except asyncio.CancelledError:
                outcome = "cancelled"
                exit_code = 130
            except Exception as exc:  # noqa: BLE001 — terminal-frame guarantee
                log.error("sage run: pipeline failed: %s", exc)
                outcome = "failure"
                exit_code = 1

        # Pull cost / latency telemetry from the last context if present.
        ctx = getattr(pipeline, "last_context", None)
        if ctx is not None:
            total_cost_usd = float(getattr(ctx, "cost", 0.0) or 0.0)
            total_latency_ms = float(getattr(ctx, "latency_ms", 0.0) or 0.0)

    finally:
        # Stop background tasks before emitting the terminal frame.
        # Stage C: heartbeat MUST be stopped + awaited before cli_complete
        # so no cli_progress can fire after the terminal frame.
        stop_stdin_event.set()
        heartbeat_stop.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except (asyncio.CancelledError, Exception):
            pass
        for t in (stdin_task, dispatcher_task):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        # Stage D: emit one terminal ``failure(kind="cli_cancel")`` frame
        # BEFORE closing the eventlog so it lands on stdout immediately
        # before ``cli_complete``. ``cli_complete.payload.final_seq`` will
        # then point at this failure frame on the cancel path. Guarded so
        # only one frame fires even if the cancel handler ran multiple
        # times (idempotent stream contract).
        if outcome == "cancelled":
            seq_before_cancel_failure = seq_counter.last
            try:
                eventlog.emit_failure(
                    kind="cli_cancel",
                    error_type="cancelled",
                    message=cancel_reason[0] or "cancel requested",
                )
            except Exception:  # noqa: BLE001
                pass
            if seq_counter.last == seq_before_cancel_failure:
                _emit_cli_cancel_failure_fallback(
                    stdout,
                    run_id=run_id,
                    seq=seq_counter.next(),
                    reason=cancel_reason[0] or "cancel requested",
                )

        # Close eventlog (flushes the tee → final runtime events on stdout).
        try:
            install_event_log(None)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            pass
        try:
            eventlog.close()
        except (AttributeError, OSError):
            pass
        try:
            log_token  # noqa: B018 — keep token alive in case of nested usage
        except Exception:  # noqa: BLE001
            pass

        # Emit terminal CLI envelope frame. ALWAYS the last frame.
        # ``final_seq`` is the stdout seq of the frame IMMEDIATELY PRECEDING
        # this cli_complete (protocol invariant 5). On the success path
        # that's the stdout-mirrored ``run_frame_summary``; on cancel /
        # failure / mid-tool-call paths that's whichever stdout frame fired
        # last — which may itself be a CLI-shell event like
        # ``cli_tool_request``. Frontends use ``final_seq`` as a stream
        # reconciliation gate: any frame with seq > final_seq after
        # cli_complete is a stream-level violation.
        #
        # We pull from ``seq_counter.last`` (NOT ``mirror_sink.last_stdout_seq``)
        # because the counter is the single source of truth for the unified
        # stdout domain; the mirror only tracks its own subset.
        last_emitted_seq = seq_counter.last
        # cli_started always advances the counter to 0 before this point,
        # so ``last`` is never None here. Defensive fallback for completeness.
        if last_emitted_seq is None:
            last_emitted_seq = 0
        _emit_cli_event(
            stdout,
            "cli_complete",
            run_id=run_id,
            payload={
                "exit_code": exit_code,
                "outcome": outcome,
                "total_cost_usd": total_cost_usd,
                "total_latency_ms": total_latency_ms,
                "trace_dir": str(trace_dir),
                "final_seq": last_emitted_seq,
            },
            seq=seq_counter.next(),
        )

    return exit_code


def main(argv: list[str]) -> int:
    """argparse + dispatch for ``sage run --jsonl ...``."""
    parser = argparse.ArgumentParser(
        prog="sage run",
        description="Machine-readable backend for pi-mono / TUI / IDE front-ends. "
        "See docs/contracts/SAGE_CLI_PROTOCOL.md for the v0 protocol.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Required: emit JSONL events on stdout.",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="Optional task-level spend cap in USD.",
    )
    parser.add_argument(
        "--system-hint",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="Optional Stage 0 routing override (S1=trivial, S2=code, S3=reasoner).",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="",
        help="Task text. If empty, reads from stdin until EOF (single-shot).",
    )
    args = parser.parse_args(argv)

    if not args.jsonl:
        print(
            "sage run: --jsonl is required in v0. Bare ``sage run`` is reserved.",
            file=sys.stderr,
        )
        return 2

    task = args.task
    stdin_for_run: TextIO | None = None
    budget_usd = args.budget_usd
    system_hint = args.system_hint
    if not task:
        first_line = sys.stdin.readline()
        if not first_line:
            print("sage run: empty task (no positional + empty stdin).", file=sys.stderr)
            return 2
        if _looks_like_stdin_command(first_line):
            try:
                prompt = _parse_initial_prompt_command(first_line)
                budget_usd = _resolve_prompt_budget(
                    argv_budget_usd=args.budget_usd,
                    prompt_budget_usd=prompt.budget_usd,
                )
                system_hint = _resolve_prompt_system_hint(
                    argv_system_hint=args.system_hint,
                    prompt_system_hint=prompt.system_hint,
                )
            except ValueError as exc:
                print(f"sage run: {exc}", file=sys.stderr)
                return 2
            task = prompt.task
            stdin_for_run = sys.stdin
        else:
            # Preserve the legacy one-shot batch mode: non-command stdin is
            # task text, including any remaining lines.
            task = (first_line + sys.stdin.read()).strip()
            if not task:
                print("sage run: empty task (no positional + empty stdin).", file=sys.stderr)
                return 2

    try:
        return asyncio.run(
            run_jsonl_async(
                task,
                budget_usd=budget_usd,
                system_hint=system_hint,
                stdin=stdin_for_run,
            )
        )
    except KeyboardInterrupt:
        return 130
