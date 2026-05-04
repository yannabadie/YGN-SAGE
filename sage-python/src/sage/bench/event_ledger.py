"""Append-only NDJSON event ledger for benchmarks.

Design from cgpro 2026-05-04 cycle-9 recovery analysis (conv
``cgpro_a3_recovery_20260504``). The A3 N=50 ablation died after 34/300
tasks because:

1. ``BigCodeBenchBench.run()`` only emits ``BenchReport`` at the end,
   and ``_run_ablation()`` only records ``all_results[config.label]``
   after ``await bench.run(...)`` returns. Process death = total loss
   of partial signal.
2. ``asyncio.wait_for(timeout=120)`` does not enforce wall-clock when
   the asyncio loop itself is suspended (Windows Modern Standby S0
   DRIPS). BCB/273 reported ``elapsed_wall_ms=20278211`` (5h 38min)
   without firing the 120s cap.

This module implements the append-only event ledger described in
cgpro's recovery plan Step 1: every state change is written to disk
immediately and ``fsync``-ed before the bench moves on. The final
report should be reducible from the ledger, not the other way round.

Wire-format: NDJSON, one event per line. Schema is intentionally flat
so downstream analysis tools (jq, duckdb, pandas read_json) can scan it
without a custom parser.

Mandatory fields on every event:

- ``event``: one of ``RUN_START``, ``CONFIG_START``, ``TASK_START``,
  ``TASK_END``, ``TASK_TIMEOUT``, ``TASK_ABORT``, ``CONFIG_END``,
  ``RUN_END``, ``RUN_ABORT``.
- ``ts``: ISO-8601 UTC timestamp at emit time.
- ``run_id``: ULID identifying the run.

Optional fields are documented per-event below. The ledger does NOT
validate the schema -- callers are expected to pass through structured
``**fields`` and the ledger writes whatever it gets. This keeps the
ledger a transport, not a contract; the contract belongs in
``docs/contracts/runtime-integrity-ledger.md``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import ulid

__all__ = [
    "BenchEventLedger",
    "build_run_meta",
]


def _git_sha(repo_root: Path | None = None) -> str:
    """Return the current HEAD short SHA, or ``"unknown"`` if not a repo."""
    cwd = repo_root or Path.cwd()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode("utf-8", errors="replace").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "unknown"


def _git_dirty_diff_hash(repo_root: Path | None = None) -> str:
    """Return SHA-256 of ``git diff HEAD`` (working tree changes), or ``"clean"`` / ``"unknown"``.

    The hash is the audit anchor cgpro asked for: if the working tree
    was dirty when the run started, this lets the post-hoc analyst tie
    a ledger back to the exact diff that produced it.
    """
    cwd = repo_root or Path.cwd()
    try:
        out = subprocess.check_output(
            ["git", "diff", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "unknown"
    if not out:
        return "clean"
    return hashlib.sha256(out).hexdigest()[:16]


def build_run_meta(
    *,
    bench_type: str,
    tier: str,
    timeout_s: float,
    limit: int | None = None,
    repo_root: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the static run-metadata dict to embed in ``RUN_START``.

    The values here do not change for the duration of a run, so we
    capture them once. Pass through ``extra`` for caller-specific
    fields (e.g. ``--subset`` for BigCodeBench).
    """
    meta: dict[str, Any] = {
        "bench_type": bench_type,
        "tier": tier,
        "timeout_s": timeout_s,
        "limit": limit,
        "git_sha": _git_sha(repo_root),
        "git_dirty_hash": _git_dirty_diff_hash(repo_root),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
    }
    if extra:
        meta.update(extra)
    return meta


class BenchEventLedger:
    """Append-only NDJSON event sink.

    Lifecycle:

    1. Construct with the output path (typically ``<output>.events.jsonl``
       sibling of the bench report JSON) and the run metadata.
    2. Call ``emit_run_start(...)`` once.
    3. For each config: ``emit_config_start`` ... per-task events ...
       ``emit_config_end``.
    4. ``emit_run_end(...)`` (or ``emit_run_abort(...)`` on a known
       crash path) and ``close()``.

    Every emit appends one JSON line followed by ``\\n``, then flushes
    and ``fsync``-s the file descriptor. This is slower than a buffered
    writer but matches the single-source-of-truth contract: if the
    process dies between the ``write`` and the next event, the ledger
    on disk is still a valid prefix of the run.

    Thread-safety: NOT thread-safe. The bench loop is single-threaded
    asyncio; the OS-level append (``open(..., "a")``) plus per-line
    fsync is sufficient for that case. If a future caller needs
    multi-threaded emit, wrap ``emit`` in a lock.
    """

    def __init__(
        self,
        output_path: str | Path,
        run_meta: dict[str, Any],
    ) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._run_meta = dict(run_meta)
        self._run_id = str(ulid.new()) if hasattr(ulid, "new") else str(ulid.ULID())
        self._fp: Any = open(self._path, "a", encoding="utf-8", buffering=1)
        self._closed = False

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event: str, **fields: Any) -> None:
        """Append one event line and fsync to disk.

        ``event`` is the discriminator (``RUN_START`` / ``TASK_END`` / ...).
        Caller-supplied ``fields`` are merged into the line as-is. The
        ledger always injects ``event``, ``ts`` (UTC ISO-8601), and
        ``run_id``.
        """
        if self._closed:
            raise RuntimeError("BenchEventLedger.emit called after close")
        record: dict[str, Any] = {
            "event": event,
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "run_id": self._run_id,
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        self._fp.write(line)
        self._fp.flush()
        try:
            os.fsync(self._fp.fileno())
        except (OSError, ValueError):
            # ValueError on closed fd; OSError on filesystems that
            # don't support fsync (rare but possible on network
            # mounts). Ledger durability is best-effort beyond write()
            # in those cases.
            pass

    def emit_run_start(self, **extra: Any) -> None:
        """RUN_START event. Embed run metadata so the ledger is self-describing."""
        self.emit("RUN_START", run_meta=self._run_meta, **extra)

    def emit_config_start(self, *, config_label: str, config_dict: dict[str, Any], **extra: Any) -> None:
        """CONFIG_START event. Embed the AblationConfig dict for traceability."""
        self.emit(
            "CONFIG_START",
            config_label=config_label,
            config=config_dict,
            **extra,
        )

    def emit_task_start(
        self,
        *,
        config_label: str,
        idx: int,
        task_id: str,
        timeout_s: float,
        **extra: Any,
    ) -> None:
        """TASK_START event. ``idx`` is 1-indexed; ``timeout_s`` echoed for analysis."""
        self.emit(
            "TASK_START",
            config_label=config_label,
            idx=idx,
            task_id=task_id,
            timeout_s=timeout_s,
            **extra,
        )

    def emit_task_end(
        self,
        *,
        config_label: str,
        idx: int,
        task_id: str,
        status: str,
        elapsed_wall_ms: float,
        passed: bool,
        **extra: Any,
    ) -> None:
        """TASK_END event. ``status`` ∈ {PASS, FAIL, ERROR}.

        Reserved fields under ``**extra``:
        - ``host_suspend_or_event_loop_stall``: bool, set when the
          watchdog detected wall_elapsed > timeout * grace_factor.
          Tasks with this flag MUST be excluded from gate-quality
          pass-rate aggregation.
        - ``control_surface``: dict of ``executed_template``,
          ``node_count``, ``controller_attached``, skip flags,
          ``frugal_cascade_attempted``, ``controller_decision_count``.
        """
        self.emit(
            "TASK_END",
            config_label=config_label,
            idx=idx,
            task_id=task_id,
            status=status,
            elapsed_wall_ms=elapsed_wall_ms,
            passed=passed,
            **extra,
        )

    def emit_task_timeout(
        self,
        *,
        config_label: str,
        idx: int,
        task_id: str,
        elapsed_wall_ms: float,
        **extra: Any,
    ) -> None:
        """TASK_TIMEOUT event. Distinguishes asyncio.TimeoutError from generic FAIL."""
        self.emit(
            "TASK_TIMEOUT",
            config_label=config_label,
            idx=idx,
            task_id=task_id,
            elapsed_wall_ms=elapsed_wall_ms,
            **extra,
        )

    def emit_task_abort(
        self,
        *,
        config_label: str,
        idx: int,
        task_id: str,
        reason: str,
        elapsed_wall_ms: float,
        **extra: Any,
    ) -> None:
        """TASK_ABORT event. ``reason`` ∈ {host_suspend_detected, exception, ...}.

        Use this when the task did not produce a normal PASS/FAIL/TIMEOUT
        result — e.g. ``host_suspend_or_event_loop_stall=true``. Tasks
        with TASK_ABORT MUST be excluded from pass-rate stats.
        """
        self.emit(
            "TASK_ABORT",
            config_label=config_label,
            idx=idx,
            task_id=task_id,
            reason=reason,
            elapsed_wall_ms=elapsed_wall_ms,
            **extra,
        )

    def emit_config_end(
        self,
        *,
        config_label: str,
        passed: int,
        total: int,
        aborted: int = 0,
        **extra: Any,
    ) -> None:
        """CONFIG_END event. ``passed``/``total`` exclude aborted tasks."""
        self.emit(
            "CONFIG_END",
            config_label=config_label,
            passed=passed,
            total=total,
            aborted=aborted,
            **extra,
        )

    def emit_run_end(self, **extra: Any) -> None:
        """RUN_END event. Normal termination."""
        self.emit("RUN_END", **extra)

    def emit_run_abort(self, *, reason: str, **extra: Any) -> None:
        """RUN_ABORT event. Abnormal termination (e.g. host suspend, KeyboardInterrupt)."""
        self.emit("RUN_ABORT", reason=reason, **extra)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._fp.flush()
            try:
                os.fsync(self._fp.fileno())
            except (OSError, ValueError):
                pass
        finally:
            try:
                self._fp.close()
            except OSError:
                pass
            self._closed = True

    def __enter__(self) -> "BenchEventLedger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            try:
                self.emit_run_abort(reason=f"exception:{exc_type.__name__}")
            except Exception:  # noqa: BLE001
                pass
        self.close()
