"""``sage run --jsonl`` — machine-readable backend for the pi-mono pivot (v0).

See ``docs/contracts/SAGE_CLI_PROTOCOL.md`` for the full v0 spec. This module
implements the contract:

  - 14 inherited events from ``RuntimeEventLog`` v0 (tee'd to stdout).
  - 4 CLI-shell envelope events emitted directly: ``cli_started``,
    ``cli_progress``, ``cli_tool_request``, ``cli_complete``.
  - 5 inbound commands parsed from stdin: ``prompt``, ``approve_tool_call``,
    ``deny_tool_call``, ``cancel``, ``set_budget``.
  - Strict JSONL: LF-only delimiters, UTF-8, fail-close on protocol_version
    mismatch.

Cycle-12 prelude scope (this commit):
  - ``cli_started`` + ``cli_complete`` envelope frames around the run.
  - RuntimeEventLog file ↔ stdout TEE via ``_CliMirrorSinkHandle``.
  - ``prompt`` + ``cancel`` commands.
  - ``approve_tool_call`` / ``deny_tool_call`` round-trip via
    ``TopologyRunner.approval_callback``.

Out of scope for this commit (lands in Cycle-12 Phase B alongside the ADR-015
decomposition):
  - ``cli_progress`` heartbeat (the runner doesn't yet expose a stage-level
    progress signal; we'd add a noisy timer here without the upstream signal).
  - ``set_budget`` (the ``CostTracker`` mid-run mutation point isn't exposed
    yet; needs a small ``Pipeline`` API addition).
  - Cancellation token threading into Rust ``TopologyRunner`` (today the
    Python pipeline can ``asyncio.CancelledError`` itself; deeper cancellation
    is Cycle-13 work).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, TextIO

CLI_PROTOCOL_VERSION = "v0"

log = logging.getLogger(__name__)


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
    ) -> None:
        self._file = file_handle
        self._stdout = stdout_stream
        self._stdout_seq_counter = stdout_seq_counter
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
        rewritten = self._rewrite_seq_for_stdout(value)
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
) -> None:
    """Emit a CLI-shell envelope event directly (bypasses RuntimeEventLog).

    Per ``docs/contracts/SAGE_CLI_PROTOCOL.md``: the 4 CLI-shell events are
    protocol-layer, NOT runtime-layer. They MUST NOT appear in
    ``RuntimeEventLog`` v0's ``EVENT_TYPES``. Their schema is independently
    versioned (``cli_v1`` initially).
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


# ────────────────────────────────────────────────────────────────────────
# Stdin command parser
# ────────────────────────────────────────────────────────────────────────


async def _read_stdin_commands(
    stdin: TextIO,
    queue: "asyncio.Queue[Mapping[str, Any] | None]",
    stop_event: asyncio.Event,
) -> None:
    """Read JSONL commands from stdin and enqueue them.

    One command per line. Parse errors are NON-fatal — the malformed line is
    skipped (no failure event is emitted to avoid amplifying garbage). EOF on
    stdin closes the queue with ``None`` and sets the stop event.
    """
    loop = asyncio.get_running_loop()
    while not stop_event.is_set():
        # ``readline`` blocks; run in executor so we don't block the loop.
        try:
            line = await loop.run_in_executor(None, stdin.readline)
        except (OSError, ValueError):
            break
        if not line:
            # EOF
            await queue.put(None)
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
            log.warning("sage run: stdin command missing 'command' field: %r", line[:120])
            continue
        await queue.put(cmd)


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
# Main entry: run_jsonl_async + main(argv)
# ────────────────────────────────────────────────────────────────────────


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
        mirror_sink = _CliMirrorSinkHandle(eventlog._fh, stdout, seq_counter)
        eventlog._fh = mirror_sink  # type: ignore[assignment]

    log_token = install_event_log(eventlog)

    # Stdin command queue + reader task.
    command_queue: asyncio.Queue[Mapping[str, Any] | None] = asyncio.Queue()
    stop_stdin_event = asyncio.Event()
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
                cancel_requested.set()
                return
            if verb == "set_budget":
                _handle_set_budget(
                    cmd,
                    pipeline=pipeline_for_dispatcher[0],
                    event_log=eventlog,
                )
                continue
            # Other commands are NOT supported in this commit (multi-prompt).
            # Surface them as warnings but don't fail.
            log.info("sage run: unsupported command in v0 prelude: %r", verb)

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
        stop_stdin_event.set()
        for t in (stdin_task, dispatcher_task):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

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
    if not task:
        # Read task text from stdin until EOF (one-shot mode for batch use).
        # The stdin command parser is NOT used in this branch.
        task = sys.stdin.read().strip()
        if not task:
            print("sage run: empty task (no positional + empty stdin).", file=sys.stderr)
            return 2

    try:
        return asyncio.run(
            run_jsonl_async(
                task,
                budget_usd=args.budget_usd,
                system_hint=args.system_hint,
            )
        )
    except KeyboardInterrupt:
        return 130
