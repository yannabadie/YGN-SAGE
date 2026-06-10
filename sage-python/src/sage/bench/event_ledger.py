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
import uuid
from pathlib import Path
from typing import Any

try:  # ulid-py: ulid.new() returns a ULID. Keep as preferred path.
    import ulid as _ulid_module
except ImportError:  # pragma: no cover - ulid is a project dep, but defensive
    _ulid_module = None  # type: ignore[assignment]

__all__ = [
    "BenchEventLedger",
    "EMPTY_PATCH_REASON_CODES",
    "TIMEOUT_REASON_CODES",
    "build_run_meta",
    "categorize_timeout",
    "classify_non_timeout_empty_patch",
]


# Block A5 (cgpro DESIGN 2026-05-10, conv
# `cgpro_ygn_sage_global_analysis_20260510`).
# Heartbeat freshness threshold for distinguishing reasoner_thinking_overflow
# (regular heartbeats) from stage_deadlock (silent past this gap).
_DEFAULT_HEARTBEAT_MAX_GAP_MS = 30_000

# Stages where a long stay with regular heartbeats is consistent with a
# reasoner generating thinking tokens (decompose: planner LLM call;
# execute: per-node provider call).
_REASONER_STAGES = frozenset({"decompose", "execute"})


# Block `canary-stage-timing-budget` (cgpro DESIGN 2026-05-11, same conv).
# Reason codes produced by ``categorize_timeout`` — timeout-derived only.
TIMEOUT_REASON_CODES: frozenset[str] = frozenset(
    {
        "scoring_boot_impossible",
        "provider_call_timeout",
        "reasoner_thinking_overflow",
        "stage_deadlock",
    }
)

# Full enum of allowed empty-patch reasons for the pre-grader gate
# (``canary_pre_grader_gate.py``). A canary task that produced ``patch=""``
# must carry one of these codes; absence is a gate failure.
#
# Union of timeout-derived codes (above) plus non-timeout codes:
#
# - ``no_patch_extracted`` — task completed within budget but the canary
#   could not extract a unified diff from the agent output.
# - ``task_budget_exhausted`` — task hit the per-task or global budget
#   cap before producing a patch.
# - ``no_patch_to_verify`` — emitted by the diff verifier when the patch
#   is empty so the verifier has nothing to check (follow-up not yet
#   shipped at the writer side; the gate accepts it once wired).
# - ``repo_unavailable`` — the canary could not materialize the task's
#   repo worktree (clone/fetch/checkout failure or timeout) and the
#   fail-closed default skipped generation entirely (RESOLUTION_UNBLOCKERS
#   2026-06-10, cgpro Q2: an infra failure must NOT be conflated with the
#   model failing to produce a patch).
EMPTY_PATCH_REASON_CODES: frozenset[str] = TIMEOUT_REASON_CODES | frozenset(
    {
        "no_patch_extracted",
        "task_budget_exhausted",
        "no_patch_to_verify",
        "repo_unavailable",
    }
)


def classify_non_timeout_empty_patch(
    *,
    budget_exhausted: bool,
    diff_verifier_outcome: str | None = None,
    repo_unavailable: bool = False,
) -> str:
    """Classify an empty-patch outcome for a task that did NOT time out.

    Used by the canary runner when a task completed within ``--task-timeout-s``
    but produced ``extracted_patch_present=False``. Returns one of:

    - ``"repo_unavailable"`` — generation was skipped fail-closed because
      the repo worktree could not be set up (infra failure, NOT an LLM
      quality signal; takes precedence over every other classification).
    - ``"task_budget_exhausted"`` — caller signaled budget cap hit.
    - ``"no_patch_to_verify"`` — diff verifier output already says so.
    - ``"no_patch_extracted"`` — fallback for any other empty-patch case.

    Timeout-derived reason codes are produced by ``categorize_timeout``
    and are NOT returned here. All returned values are members of
    ``EMPTY_PATCH_REASON_CODES``.
    """
    if repo_unavailable:
        return "repo_unavailable"
    if budget_exhausted:
        return "task_budget_exhausted"
    if diff_verifier_outcome == "no_patch_to_verify":
        return "no_patch_to_verify"
    return "no_patch_extracted"


def categorize_timeout(
    *,
    progress_events: list[dict[str, Any]],
    model_assigned_events: list[dict[str, Any]],
    node_started_events: list[dict[str, Any]],
    routing_decision_events: list[dict[str, Any]],
    elapsed_total_ms: float,
    heartbeat_max_gap_ms: int = _DEFAULT_HEARTBEAT_MAX_GAP_MS,
) -> dict[str, Any]:
    """Categorize a task timeout from per-task event log contents.

    Block A5 (cgpro DESIGN 2026-05-10) — the same 5/5-timeout pattern in
    cycle-13 canaries (`2026-05-10-canary-n5-real-{ec0b775e,8844c42e}`)
    can hide three distinct root causes that the bench layer must
    distinguish so a future timeout reports something actionable rather
    than just "timed out at 120s".

    Returns a dict with keys:

    - ``last_stage`` — string from the last ``cli_progress`` event's
      ``payload.stage``, or ``None`` if there were no progress frames.
    - ``elapsed_ms_by_stage`` — per-stage duration computed from
      ``cli_progress.payload.elapsed_ms`` (which is cumulative since
      task start). Stage duration = last ``elapsed_ms`` minus first
      ``elapsed_ms`` for events of that stage.
    - ``provider_attempted`` — bool. ``True`` ONLY if at least one
      ``node_started`` event was emitted (or a provider execution
      witness, future). ``model_assigned`` alone proves assignment, NOT
      a call attempt — per cgpro DESIGN correction.
    - ``model_id_final`` — payload.model_id of the last ``model_assigned``
      event, or ``None``.
    - ``provider_final`` — payload.provider_id of the last
      ``model_assigned`` event, or ``None``.
    - ``reason_code`` — one of:
      - ``"scoring_boot_impossible"`` — no progress, no routing, no
        assignment.
      - ``"reasoner_thinking_overflow"`` — last_stage in
        {decompose, execute}, majority TEMPORAL of the run spent in
        that stage, with recent heartbeats (last cli_progress within
        ``heartbeat_max_gap_ms`` of the timeout).
      - ``"provider_call_timeout"`` — ``provider_attempted=True`` and
        ``last_stage="execute"``. Distinct from generic stage_deadlock
        because we have positive evidence the call was started.
      - ``"stage_deadlock"`` — progression then silence (gap >=
        heartbeat_max_gap_ms) OR non-reasoner stage with long block.
    """
    provider_attempted = bool(node_started_events)

    model_id_final: str | None = None
    provider_final: str | None = None
    if model_assigned_events:
        last_assigned = model_assigned_events[-1]
        payload = last_assigned.get("payload") or {}
        model_id_final = (
            payload.get("model_id")
            or last_assigned.get("model_id")
        )
        provider_final = (
            payload.get("provider_id")
            or last_assigned.get("provider_id")
        )

    elapsed_ms_by_stage: dict[str, int] = {}
    last_stage: str | None = None
    last_progress_elapsed_ms: int = 0
    by_stage_first_last: dict[str, tuple[int, int]] = {}
    for ev in progress_events:
        payload = ev.get("payload") or {}
        stage = payload.get("stage")
        elapsed_ms_raw = payload.get("elapsed_ms")
        if not stage or elapsed_ms_raw is None:
            continue
        try:
            elapsed_ms = int(elapsed_ms_raw)
        except (TypeError, ValueError):
            continue
        if stage in by_stage_first_last:
            first_ms, _ = by_stage_first_last[stage]
            by_stage_first_last[stage] = (first_ms, elapsed_ms)
        else:
            by_stage_first_last[stage] = (elapsed_ms, elapsed_ms)
        last_stage = stage
        last_progress_elapsed_ms = elapsed_ms

    for stage, (first_ms, last_ms) in by_stage_first_last.items():
        elapsed_ms_by_stage[stage] = max(0, last_ms - first_ms)

    no_progress = not progress_events
    no_routing = not routing_decision_events
    no_assignment = not model_assigned_events

    # 1. scoring_boot_impossible: pipeline never produced any usable
    #    signal — no progress, no routing decision, no assignment.
    if no_progress and no_routing and no_assignment:
        reason_code = "scoring_boot_impossible"
    # 2. provider_call_timeout: positive evidence the provider call was
    #    started (node_started present) AND we ended in the execute
    #    stage. Distinguishes provider RPC hang from local reasoner.
    elif provider_attempted and last_stage == "execute":
        reason_code = "provider_call_timeout"
    elif last_stage in _REASONER_STAGES and elapsed_total_ms > 0:
        # Time-based heuristic per cgpro DESIGN correction: events count
        # is NOT a proxy for time. Use payload.elapsed_ms (cumulative
        # task time) to compute share of the run spent in last_stage.
        time_in_stage = elapsed_ms_by_stage.get(last_stage, 0)
        share_in_stage = time_in_stage / float(elapsed_total_ms)
        time_since_heartbeat = elapsed_total_ms - last_progress_elapsed_ms
        recent_heartbeat = time_since_heartbeat < heartbeat_max_gap_ms
        if share_in_stage > 0.5 and recent_heartbeat:
            reason_code = "reasoner_thinking_overflow"
        else:
            reason_code = "stage_deadlock"
    else:
        # Some progress in a non-reasoner stage, or progress went silent
        # past the heartbeat window. Both fall under stage_deadlock.
        reason_code = "stage_deadlock"

    return {
        "last_stage": last_stage,
        "elapsed_ms_by_stage": elapsed_ms_by_stage,
        "provider_attempted": provider_attempted,
        "model_id_final": model_id_final,
        "provider_final": provider_final,
        "reason_code": reason_code,
    }


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
        # ulid-py preferred (lexicographically sortable, monotonically
        # increasing, embeds creation time). uuid4 fallback only if the
        # ulid module is unavailable; both are 26+ chars so downstream
        # consumers don't need to know which was emitted.
        if _ulid_module is not None and hasattr(_ulid_module, "new"):
            self._run_id = str(_ulid_module.new())
        else:
            self._run_id = str(uuid.uuid4())
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
