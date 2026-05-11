#!/usr/bin/env python3
"""Cycle-13 E Tier 2.1 — Arm D smoke runner.

Reads N task metadata from `instances.json` (produced by
`swebench_pro_fetch.py`), runs each task through `python -m sage.cli
run --jsonl`, captures the `final_result` event, extracts a unified
diff (if any), formats as SWE-bench Pro record, writes
`predictions.json` + per-task event traces.

Per cgpro DESIGN E (2026-05-05, conv `cgpro_pi_mono_pivot_20260505`,
verdict GO_TIER_1_PLUS_2 sub-stage 2.1):

  Tier 2.1 acceptance: 1/1 graded real Arm D task minimum, 2/2 only
  if Docker/image/runtime is not the bottleneck. Hard cutoff: if
  Docker pull/eval > 15 min OR API spend > $5, stop.

Modes:
  --mock              Synthetic empty patch per task (NO API spend).
                      Validates fetch script + format_patch wiring.
                      Tier 2.0 expansion. Acceptance: shape-valid
                      predictions.json produced.
  (default)           Real LLM via `python -m sage.cli run --jsonl`.
                      Default model: budget tier (deepseek-v4-flash);
                      override via SAGE_LLM_TIER env var.
                      Acceptance: shape-valid predictions.json +
                      per-task RuntimeEventLog file present.

Output:
  <output-dir>/predictions.json     — Pro grader input format
  <output-dir>/per_task/<id>.events.jsonl  — full event trace per task
  <output-dir>/summary.json         — aggregated metrics

Grading is OUT OF SCOPE here. The grader (`swe_bench_pro_eval.py`
in scaleapi/SWE-bench_Pro-os) requires Docker daemon running OR
Modal account + per-instance dockerfiles + run_scripts. Run grader
separately on the produced predictions.json.

Usage:
  # Mock smoke (no API):
  python -m sage_python.scripts.run_dryrun_arm_d \\
      --instances-json sage-python/data/swebench_pro/n10/instances.json \\
      --limit 1 --mock \\
      --output-dir sage-python/data/swebench_pro/arm_d_smoke_mock_n1

  # Real smoke (1 task, ~$0.50-1):
  python -m sage_python.scripts.run_dryrun_arm_d \\
      --instances-json sage-python/data/swebench_pro/n10/instances.json \\
      --limit 1 --budget-usd 5.0 \\
      --output-dir sage-python/data/swebench_pro/arm_d_smoke_real_n1
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("sage.bench.run_dryrun_arm_d")

# Default model when running in real mode. Per cgpro plan §"Models per
# arm": cycle-13 main run forces SAGE_LLM_TIER=reasoner (Opus 4.7) for
# fair vs Claude Code. The smoke uses `budget` (deepseek-v4-flash) by
# default for cost — overridable via env or --tier.
_DEFAULT_TIER = "budget"

# Block `canary-stage-timing-budget` (cgpro DESIGN 2026-05-11, conv
# `cgpro_ygn_sage_global_analysis_20260510`) slice 3.
#
# Named timeout profiles. The B2 step 4 N=1 canary timed out at 300s on
# a substantial Vuls Trivy upgrade task without extracting a patch.
# cgpro recommended a 600-1200s envelope for the graded-patch-generation
# profile; 900s is the midpoint and gives the agent room without
# inviting unbounded reasoner_thinking_overflow.
#
# `default` preserves the historical 120s — the existing per-task budget
# for non-graded smokes (mock runs, plumbing checks, A5 timing triage
# unit work) — so unflagged callers are byte-equivalent to pre-slice-3.
_TIMEOUT_PROFILES: dict[str, float] = {
    "default": 120.0,
    "graded_patch_generation": 900.0,
}
_DEFAULT_PROFILE = "default"

# Path to the Pro patch format helper (loaded as module).
_FORMAT_PATCH_PATH = (
    Path(__file__).parent / "swebench_pro_format_patch.py"
).resolve()
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CANARY_MANIFEST_PATH = (
    _REPO_ROOT / "docs" / "benchmarks" / "cycle-13-canary-manifest.md"
)

_PREDICTION_AUDIT_SCHEMA_VERSION = "swebench_pro_canary_prediction_v1"
_PREDICTION_AUDIT_FIELDS = (
    "_verifier_repair_budget_usd",
    "_diff_verifier_mismatches",
    "_diff_verifier_outcome",
    "model_id_final",
    "provider_final",
    "_observed_model_ids",
    "_observed_providers",
    "_observed_event_cost_usd",
    "total_cost_usd",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _git_status_short() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=_REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _extract_manifest_commit(manifest_text: str) -> str | None:
    match = re.search(
        r"\|\s*Commit SHA\s*\|\s*`?([^`|]+)`?\s*\|",
        manifest_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip()


def _event_value(event: dict[str, Any], key: str) -> Any:
    payload = event.get("payload")
    if isinstance(payload, dict) and payload.get(key) is not None:
        return payload.get(key)
    return event.get(key)


def _event_audit_from_file(events_path: Path) -> dict[str, Any]:
    model_id_final: str | None = None
    provider_final: str | None = None
    observed_model_ids: set[str] = set()
    observed_providers: set[str] = set()
    assigned_providers: set[str] = set()
    execution_providers: set[str] = set()
    assigned_model_ids: set[str] = set()
    execution_model_ids: set[str] = set()
    provider_policy_failure_seen = False
    observed_cost_usd = 0.0

    if not events_path.is_file():
        return {
            "model_id_final": None,
            "provider_final": None,
            "_observed_model_ids": [],
            "_observed_providers": [],
            "_assigned_model_ids": [],
            "_assigned_providers": [],
            "_execution_model_ids": [],
            "_execution_providers": [],
            "_provider_policy_failure_seen": False,
            "_observed_event_cost_usd": 0.0,
        }

    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            ev_type = event.get("event_type")
            if ev_type == "failure":
                kind = _event_value(event, "kind")
                error_type = _event_value(event, "error_type")
                if kind == "provider_policy" or error_type == "provider_policy_violation":
                    provider_policy_failure_seen = True
            if ev_type in {
                "routing_decision",
                "model_assigned",
                "node_started",
                "node_completed",
            }:
                model_id = _event_value(event, "model_id")
                if isinstance(model_id, str) and model_id:
                    model_id_final = model_id
                    observed_model_ids.add(model_id)
                    if ev_type == "model_assigned":
                        assigned_model_ids.add(model_id)
                    if ev_type in {"node_started", "node_completed"}:
                        execution_model_ids.add(model_id)
                provider_id = _event_value(event, "provider_id") or _event_value(
                    event, "provider"
                )
                if isinstance(provider_id, str) and provider_id:
                    provider_final = provider_id
                    observed_providers.add(provider_id)
                    if ev_type == "model_assigned":
                        assigned_providers.add(provider_id)
                    if ev_type in {"node_started", "node_completed"}:
                        execution_providers.add(provider_id)
            cost_usd = _event_value(event, "cost_usd")
            if isinstance(cost_usd, (int, float)) and not isinstance(cost_usd, bool):
                observed_cost_usd += float(cost_usd)

    return {
        "model_id_final": model_id_final,
        "provider_final": provider_final,
        "_observed_model_ids": sorted(observed_model_ids),
        "_observed_providers": sorted(observed_providers),
        "_assigned_model_ids": sorted(assigned_model_ids),
        "_assigned_providers": sorted(assigned_providers),
        "_execution_model_ids": sorted(execution_model_ids),
        "_execution_providers": sorted(execution_providers),
        "_provider_policy_failure_seen": provider_policy_failure_seen,
        "_observed_event_cost_usd": observed_cost_usd,
    }


# ── Synthetic patch generators (mock mode) ───────────────────────────────────


def _synthetic_empty_patch() -> str:
    """Empty patch: agent gave up. Pro grader treats as non-resolution.

    Per cgpro DESIGN E trap Q5 (validate_record accepts empty patch):
    this proves the runner produces shape-valid output even when the
    agent fails entirely.
    """
    return ""


def _synthetic_minimal_patch(instance_id: str) -> str:
    """Minimal but well-formed unified diff for shape validation.

    Per `gather_patches.py` in scaleapi/SWE-bench_Pro-os: the grader
    accepts plain-text patches. This synthetic patch is shape-valid
    but won't actually resolve any real test (designed to fail
    gracefully under the grader, not pass).
    """
    return (
        f"diff --git a/synthetic.py b/synthetic.py\n"
        f"index 0000000..1111111 100644\n"
        f"--- a/synthetic.py\n"
        f"+++ b/synthetic.py\n"
        f"@@ -1,1 +1,1 @@\n"
        f"-# placeholder for {instance_id}\n"
        f"+# patched by ygn-sage arm-d smoke (mock mode)\n"
    )


# ── Patch extraction from agent output ───────────────────────────────────────


_DIFF_HEADER_RE = re.compile(
    r"^diff --git a/.+ b/.+$",
    re.MULTILINE,
)


def _extract_patch_from_text(text: str) -> str:
    """Find the first unified-diff block in `text` and return it.

    Heuristic — matches a `diff --git ...` header and returns from
    that line to the end (or up to a fenced code block close if
    present). Empty-string returned when no diff found.

    This is the Tier 2.1 dumb-extractor. Cycle-13 main run can use
    a more sophisticated extractor (sage's existing diff parsing) if
    this proves insufficient.
    """
    if not text:
        return ""

    # Strip markdown code fences if the diff is wrapped.
    fence_match = re.search(
        r"```(?:diff|patch)?\n(.*?)```",
        text,
        re.DOTALL,
    )
    candidate = fence_match.group(1) if fence_match else text

    header_match = _DIFF_HEADER_RE.search(candidate)
    if not header_match:
        return ""

    return candidate[header_match.start():].strip() + "\n"


# ── Sage CLI subprocess (real mode) ──────────────────────────────────────────


async def _run_sage_cli(
    task_text: str,
    budget_usd: float,
    output_events_path: Path,
    tier: str,
    provider_allowlist: tuple[str, ...] = (),
    provider_denylist: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Invoke `python -m sage.cli run --jsonl` as subprocess.

    Uses the JSONL stdin protocol (matching the direct CLI path):
    writes ``{"command":"prompt","args":{"task":...}}`` to stdin,
    then drains stdout events + stderr in parallel (cgpro 2026-05-09
    fix — previously passed the task as a positional arg, which
    bypasses the canonical JSONL prompt channel and deadlocks on
    stderr buffering).
    """
    output_events_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SAGE_LLM_TIER"] = tier
    env["SAGE_DIFF_VERIFIER_MODE"] = "observe"
    env["SAGE_OTEL_EXPORTER"] = "none"
    env.setdefault("SAGE_BOOT_BYPASS_EPOCH_GUARD", "1")
    env.setdefault("SAGE_BOOT_BYPASS_REASON",
        "cycle-13 E Tier 2.1 arm D smoke run; bypass disables atexit "
        "save so smokes do not pollute ~/.sage/ across consecutive runs")
    env.setdefault("SAGE_OPERATOR_ID", "ygn-sage-arm-d-smoke")

    cmd = [
        sys.executable, "-m", "sage.cli", "run", "--jsonl",
        "--budget-usd", str(budget_usd),
    ]
    if provider_allowlist:
        cmd.extend(["--provider-allowlist", ",".join(provider_allowlist)])
    if provider_denylist:
        cmd.extend(["--provider-denylist", ",".join(provider_denylist)])
    log.info("Spawning sage CLI: %s", " ".join(cmd))

    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # Exercise the canonical inbound JSONL protocol, not the legacy
    # raw-stdin fallback.
    assert proc.stdin is not None
    prompt_command = {
        "command": "prompt",
        "args": {
            "task": task_text,
            "budget_usd": budget_usd,
        },
    }
    proc.stdin.write(
        (json.dumps(prompt_command, separators=(",", ":")) + "\n").encode("utf-8")
    )
    await proc.stdin.drain()
    proc.stdin.close()

    # Drain stderr in parallel to prevent pipe deadlock.
    async def _drain_stderr() -> bytes:
        if proc.stderr is None:
            return b""
        chunks: list[bytes] = []
        async for chunk in proc.stderr:
            chunks.append(chunk)
        return b"".join(chunks)

    stderr_task = asyncio.create_task(_drain_stderr())

    final_result_payload: Any = None
    cli_complete_payload: dict[str, Any] | None = None
    cli_complete_run_id: str | None = None
    model_id_final: str | None = None
    provider_final: str | None = None

    try:
        with output_events_path.open("w", encoding="utf-8", newline="\n") as event_log:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    continue
                event_log.write(line + "\n")
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("Non-JSON line on stdout: %r", line[:80])
                    continue
                ev_type = event.get("event_type")
                payload = event.get("payload", {})
                if isinstance(payload, dict):
                    if ev_type == "routing_decision":
                        model_id_final = payload.get("model_id") or model_id_final
                    elif ev_type in {"model_assigned", "node_started", "node_completed"}:
                        model_id_final = payload.get("model_id") or model_id_final
                        provider_final = (
                            payload.get("provider_id")
                            or payload.get("provider")
                            or provider_final
                        )
                if ev_type == "final_result":
                    final_result_payload = payload
                elif ev_type == "cli_complete":
                    cli_complete_payload = payload if isinstance(payload, dict) else {}
                    event_run_id = event.get("run_id")
                    cli_complete_run_id = (
                        event_run_id if isinstance(event_run_id, str) else None
                    )

        exit_code = await proc.wait()
        latency_s = time.monotonic() - start
        stderr_bytes = await stderr_task
    except asyncio.CancelledError:
        if proc.returncode is None:
            proc.terminate()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=5)
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        stderr_task.cancel()
        with suppress(asyncio.CancelledError):
            await stderr_task
        raise

    if cli_complete_payload is None:
        stderr_text = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        log.error(
            "CLI ended without cli_complete; exit_code=%s stderr=%s",
            exit_code, stderr_text[-4000:],
        )

    event_audit = _event_audit_from_file(output_events_path)

    return {
        "exit_code": exit_code,
        "latency_ms": int(latency_s * 1000),
        "final_result_payload": final_result_payload,
        "cli_complete_payload": cli_complete_payload,
        "total_cost_usd": (
            (cli_complete_payload or {}).get("total_cost_usd")
            if cli_complete_payload
            else None
        ),
        "run_id": cli_complete_run_id,
        "model_id_final": event_audit.get("model_id_final") or model_id_final,
        "provider_final": event_audit.get("provider_final") or provider_final,
        "_observed_model_ids": event_audit.get("_observed_model_ids", []),
        "_observed_providers": event_audit.get("_observed_providers", []),
        "_assigned_model_ids": event_audit.get("_assigned_model_ids", []),
        "_assigned_providers": event_audit.get("_assigned_providers", []),
        "_execution_model_ids": event_audit.get("_execution_model_ids", []),
        "_execution_providers": event_audit.get("_execution_providers", []),
        "_provider_policy_failure_seen": event_audit.get(
            "_provider_policy_failure_seen",
            False,
        ),
        "_observed_event_cost_usd": event_audit.get("_observed_event_cost_usd", 0.0),
        "trace_dir": (
            cli_complete_payload.get("trace_dir")
            if isinstance(cli_complete_payload, dict)
            else None
        ),
        "stderr": stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else "",
    }


# ── Per-task runner ──────────────────────────────────────────────────────────


def _learning_evidence_not_requested() -> dict[str, Any]:
    return {
        "claimed": False,
        "status": "skipped",
        "reason_code": "not_claimed",
    }


def _safe_artifact_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "task"


def _learning_evidence_no_go(
    *,
    reason_code: str,
    detail: str,
    run_id: str | None,
    source_trace_dir: str | Path | None,
    archived_trace_dir: Path | None,
    expect_default_pipeline_learn: bool,
) -> dict[str, Any]:
    return {
        "claimed": True,
        "status": "no_go",
        "reason_code": reason_code,
        "detail": detail,
        "mode": "evidence-boundary",
        "expect_default_pipeline_learn": expect_default_pipeline_learn,
        "run_id": run_id,
        "source_trace_dir": str(source_trace_dir) if source_trace_dir else None,
        "trace_dir": str(archived_trace_dir) if archived_trace_dir else None,
        "records": 0,
    }


def _validate_learning_evidence(
    trace_dir: str | Path | None,
    run_id: str | None,
    *,
    archive_trace_dir: Path,
    expect_default_pipeline_learn: bool,
) -> dict[str, Any]:
    """Run the post-run learning side-effect evidence boundary.

    The task has already completed. This controls benchmark artifact
    acceptability only; it does not authorize or block runtime learning.
    """
    if not trace_dir or not run_id:
        return _learning_evidence_no_go(
            reason_code="missing_trace_identity",
            detail="cli_complete did not provide both run_id and trace_dir",
            run_id=run_id,
            source_trace_dir=trace_dir,
            archived_trace_dir=archive_trace_dir,
            expect_default_pipeline_learn=expect_default_pipeline_learn,
        )

    source_trace_dir = Path(trace_dir)
    if not source_trace_dir.is_dir():
        return _learning_evidence_no_go(
            reason_code="trace_dir_missing",
            detail=f"trace_dir not found: {source_trace_dir}",
            run_id=run_id,
            source_trace_dir=source_trace_dir,
            archived_trace_dir=archive_trace_dir,
            expect_default_pipeline_learn=expect_default_pipeline_learn,
        )

    try:
        if archive_trace_dir.exists():
            shutil.rmtree(archive_trace_dir)
        shutil.copytree(source_trace_dir, archive_trace_dir)
    except OSError as exc:
        return _learning_evidence_no_go(
            reason_code="trace_archive_failed",
            detail=f"{type(exc).__name__}: {exc}",
            run_id=run_id,
            source_trace_dir=source_trace_dir,
            archived_trace_dir=archive_trace_dir,
            expect_default_pipeline_learn=expect_default_pipeline_learn,
        )

    cmd = [
        sys.executable,
        "-m",
        "sage.runtime.credit_assignment.validate",
        str(archive_trace_dir),
        "--run-id",
        run_id,
        "--mode",
        "evidence-boundary",
    ]
    if expect_default_pipeline_learn:
        cmd.append("--expect-default-pipeline-learn")

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _learning_evidence_no_go(
            reason_code="validator_error",
            detail=f"{type(exc).__name__}: {exc}",
            run_id=run_id,
            source_trace_dir=source_trace_dir,
            archived_trace_dir=archive_trace_dir,
            expect_default_pipeline_learn=expect_default_pipeline_learn,
        )

    status = "pass" if proc.returncode == 0 else "no_go"
    return {
        "claimed": True,
        "status": status,
        "reason_code": "validated" if status == "pass" else "validator_failed",
        "mode": "evidence-boundary",
        "expect_default_pipeline_learn": expect_default_pipeline_learn,
        "run_id": run_id,
        "source_trace_dir": str(source_trace_dir),
        "trace_dir": str(archive_trace_dir),
        "validator_exit_code": proc.returncode,
        "validator_command": cmd,
        "validator_stdout": proc.stdout[-4000:],
        "validator_stderr": proc.stderr[-4000:],
    }


def _load_format_patch_module() -> Any:
    """Load `swebench_pro_format_patch` from scripts/ as a sibling module."""
    spec = importlib.util.spec_from_file_location(
        "swebench_pro_format_patch", _FORMAT_PATCH_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("swebench_pro_format_patch", mod)
    spec.loader.exec_module(mod)
    return mod


def _prediction_audit_record(
    record: dict[str, Any],
    summary: dict[str, Any],
    *,
    task_index: int,
) -> dict[str, Any]:
    audit_record = dict(record)
    audit_record["_audit_schema_version"] = _PREDICTION_AUDIT_SCHEMA_VERSION
    audit_record["_task_index"] = task_index
    audit_record["_mock"] = bool(summary.get("mock"))
    audit_record["_timeout"] = bool(summary.get("timeout"))
    audit_record["_exit_code"] = summary.get("exit_code")
    audit_record["_latency_ms"] = summary.get("latency_ms")
    audit_record["_events_path"] = summary.get("events_path")
    for field in _PREDICTION_AUDIT_FIELDS:
        audit_record[field] = summary.get(field)
    return audit_record


def _write_predictions_jsonl(
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, (record, summary) in enumerate(zip(records, summaries, strict=True)):
            handle.write(
                json.dumps(
                    _prediction_audit_record(record, summary, task_index=index),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _write_aggregate_events(
    summaries: list[dict[str, Any]],
    *,
    output_dir: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for summary in summaries:
            rel_path = summary.get("events_path")
            if not isinstance(rel_path, str):
                continue
            source = output_dir / rel_path
            if not source.is_file():
                continue
            with source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    destination.write(line.rstrip("\r\n") + "\n")


def _load_grader_gate(grader_preflight_path: Path | None) -> dict[str, Any]:
    if grader_preflight_path is None:
        return {
            "status": "BLOCKED",
            "reason": "grader_preflight_artifact_not_supplied",
            "path": None,
        }
    if not grader_preflight_path.is_file():
        return {
            "status": "BLOCKED",
            "reason": "grader_preflight_artifact_missing",
            "path": str(grader_preflight_path),
        }
    try:
        data = json.loads(grader_preflight_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "BLOCKED",
            "reason": "grader_preflight_artifact_unreadable",
            "path": str(grader_preflight_path),
            "detail": f"{type(exc).__name__}: {exc}",
        }
    decision = data.get("decision")
    ready = bool(data.get("local_grading_ready")) or decision in {
        "READY_LOCAL_DOCKER",
        "READY_REMOTE_MODAL",
    }
    return {
        "status": "PASS" if ready else "BLOCKED",
        "reason": None if ready else "grader_preflight_not_ready",
        "path": str(grader_preflight_path),
        "sha256": _sha256_file(grader_preflight_path),
        "decision": decision,
        "blockers": data.get("blockers", []),
    }


def _load_ci_gate(ci_green_artifact: Path | None, *, git_head: str | None) -> dict[str, Any]:
    if ci_green_artifact is None:
        return {
            "status": "BLOCKED",
            "reason": "ci_green_artifact_not_supplied",
            "path": None,
        }
    if not ci_green_artifact.is_file():
        return {
            "status": "BLOCKED",
            "reason": "ci_green_artifact_missing",
            "path": str(ci_green_artifact),
        }
    try:
        data = json.loads(ci_green_artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "BLOCKED",
            "reason": "ci_green_artifact_unreadable",
            "path": str(ci_green_artifact),
            "detail": f"{type(exc).__name__}: {exc}",
        }
    status = str(data.get("status") or data.get("conclusion") or "").lower()
    commit = data.get("commit") or data.get("head_sha")
    passed = status in {"pass", "passed", "success", "green"} and (
        git_head is None or commit == git_head
    )
    reasons: list[str] = []
    if status not in {"pass", "passed", "success", "green"}:
        reasons.append("ci_status_not_green")
    if git_head is not None and commit != git_head:
        reasons.append("ci_commit_mismatch")
    return {
        "status": "PASS" if passed else "BLOCKED",
        "reason": None if passed else ",".join(reasons),
        "path": str(ci_green_artifact),
        "sha256": _sha256_file(ci_green_artifact),
        "commit": commit,
        "reported_status": status,
    }


def _write_launch_manifest(
    *,
    output_dir: Path,
    instances_json: Path,
    manifest_path: Path | None,
    budget_usd: float,
    global_budget_usd: float,
    task_timeout_s: float,
    profile: str = _DEFAULT_PROFILE,
    profile_timeout_override: bool = False,
    provider_allowlist: tuple[str, ...],
    provider_denylist: tuple[str, ...],
    grader_preflight_path: Path | None,
    ci_green_artifact: Path | None,
) -> dict[str, Any]:
    git_head = _git_head()
    git_status_short = _git_status_short()
    manifest_exists = manifest_path is not None and manifest_path.is_file()
    manifest_text = (
        manifest_path.read_text(encoding="utf-8") if manifest_exists and manifest_path else ""
    )
    manifest_commit = _extract_manifest_commit(manifest_text) if manifest_text else None
    manifest_reasons: list[str] = []
    if not manifest_exists:
        manifest_reasons.append("manifest_missing")
    elif manifest_commit in {None, "", "<SET_AT_LAUNCH>"}:
        manifest_reasons.append("manifest_commit_not_frozen")
    elif git_head is not None and manifest_commit != git_head:
        manifest_reasons.append("manifest_commit_mismatch")

    if manifest_exists and manifest_path is not None:
        shutil.copyfile(manifest_path, output_dir / "launch_manifest.md")

    launch_manifest = {
        "schema_version": "swebench_pro_canary_launch_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": {
            "path": str(_REPO_ROOT),
            "head": git_head,
            "dirty": bool(git_status_short),
            "status_short": git_status_short,
        },
        "inputs": {
            "instances_json": {
                "path": str(instances_json),
                "sha256": _sha256_file(instances_json),
            },
            "manifest": {
                "path": str(manifest_path) if manifest_path else None,
                "exists": bool(manifest_exists),
                "sha256": _sha256_file(manifest_path) if manifest_exists and manifest_path else None,
                "declared_commit": manifest_commit,
                "copied_to": "launch_manifest.md" if manifest_exists else None,
            },
        },
        "budget": {
            "budget_usd_per_task": budget_usd,
            "global_budget_usd": global_budget_usd,
            "task_timeout_s": task_timeout_s,
            "effective_profile": profile,
            "profile_timeout_default_s": _TIMEOUT_PROFILES.get(profile),
            "profile_timeout_override": profile_timeout_override,
        },
        "providers": {
            "allowlist": list(provider_allowlist),
            "denylist": list(provider_denylist),
        },
        "manifest_gate": {
            "status": "PASS" if not manifest_reasons else "BLOCKED",
            "reasons": manifest_reasons,
        },
        "grading_gate": _load_grader_gate(grader_preflight_path),
        "ci_gate": _load_ci_gate(ci_green_artifact, git_head=git_head),
    }
    (output_dir / "launch_manifest.json").write_text(
        json.dumps(launch_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    return launch_manifest


def _combine_canary_decision(gates: dict[str, dict[str, Any]]) -> str:
    statuses = {name: gate.get("status") for name, gate in gates.items()}
    if any(status == "NO_GO" for status in statuses.values()):
        return "NO_GO"
    if any(status == "BLOCKED" for status in statuses.values()):
        return "BLOCKED"
    if all(status == "PASS" for status in statuses.values()):
        return "PENDING_REVIEW"
    return "BLOCKED"


def _provider_gate(
    summaries: list[dict[str, Any]],
    *,
    mock: bool,
    provider_allowlist: tuple[str, ...],
    provider_denylist: tuple[str, ...],
) -> dict[str, Any]:
    if mock:
        return {
            "status": "BLOCKED",
            "reason": "mock_mode_no_provider_observation",
            "observed_providers": [],
            "provider_allowlist": list(provider_allowlist),
            "provider_denylist": list(provider_denylist),
        }
    observed_set: set[str] = set()
    assigned_set: set[str] = set()
    execution_set: set[str] = set()
    policy_failure_seen = False
    for summary in summaries:
        raw_observed = summary.get("_observed_providers")
        if isinstance(raw_observed, list):
            observed_set.update(str(item) for item in raw_observed if item)
        elif isinstance(raw_observed, tuple):
            observed_set.update(str(item) for item in raw_observed if item)
        raw_assigned = summary.get("_assigned_providers")
        if isinstance(raw_assigned, (list, tuple)):
            assigned_set.update(str(item) for item in raw_assigned if item)
        raw_execution = summary.get("_execution_providers")
        if isinstance(raw_execution, (list, tuple)):
            execution_set.update(str(item) for item in raw_execution if item)
        policy_failure_seen = policy_failure_seen or bool(
            summary.get("_provider_policy_failure_seen")
        )
        provider_final = summary.get("provider_final")
        if provider_final:
            observed_set.add(str(provider_final))
    observed = sorted(observed_set)
    assigned = sorted(assigned_set)
    execution = sorted(execution_set)
    missing = [
        str(summary.get("instance_id"))
        for summary in summaries
        if (
            not summary.get("_provider_policy_failure_seen")
            and (not summary.get("provider_final") or not summary.get("model_id_final"))
        )
    ]
    denyset = set(provider_denylist)
    allowset = set(provider_allowlist)
    assigned_denied = [provider for provider in assigned if provider in denyset]
    execution_denied = [provider for provider in execution if provider in denyset]
    assigned_outside_allowlist = [
        provider for provider in assigned if allowset and provider not in allowset
    ]
    execution_outside_allowlist = [
        provider for provider in execution if allowset and provider not in allowset
    ]
    assigned_policy_violation = bool(assigned_denied or assigned_outside_allowlist)
    execution_policy_violation = bool(execution_denied or execution_outside_allowlist)
    if execution_policy_violation or missing:
        status = "NO_GO"
    elif assigned_policy_violation and not policy_failure_seen:
        status = "NO_GO"
    else:
        status = "PASS"
    reason = None
    if status == "NO_GO":
        reason = "provider_audit_failed"
    elif assigned_policy_violation and policy_failure_seen:
        reason = "runtime_provider_policy_enforced"
    return {
        "status": status,
        "reason": reason,
        "observed_providers": observed,
        "assigned_providers": assigned,
        "execution_providers": execution,
        "provider_policy_failure_seen": policy_failure_seen,
        "missing_provider_or_model": missing,
        "assigned_denied_providers": assigned_denied,
        "execution_denied_providers": execution_denied,
        "assigned_outside_allowlist": assigned_outside_allowlist,
        "execution_outside_allowlist": execution_outside_allowlist,
        "provider_allowlist": list(provider_allowlist),
        "provider_denylist": list(provider_denylist),
    }


def _categorize_timeout_from_events(
    events_path: Path,
    *,
    task_timeout_s: float,
) -> dict[str, Any]:
    """Block A5 (cgpro DESIGN 2026-05-10): categorize a runner timeout.

    Reads per-task events file and forwards typed event lists to
    `sage.bench.event_ledger.categorize_timeout`. Returns the
    categorization dict (last_stage / elapsed_ms_by_stage /
    provider_attempted / model_id_final / provider_final / reason_code)
    so the runner summary can carry it as a first-class field.

    Per cgpro DESIGN correction: heuristic is TIME-based (uses
    cli_progress.payload.elapsed_ms which is cumulative since task
    start), not count-based; provider_attempted is true only when
    node_started events exist (model_assigned alone is not proof of a
    call attempt).
    """
    progress_events: list[dict[str, Any]] = []
    model_assigned_events: list[dict[str, Any]] = []
    node_started_events: list[dict[str, Any]] = []
    routing_decision_events: list[dict[str, Any]] = []

    if events_path.is_file():
        for raw in events_path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                ev = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            event_type = ev.get("event_type")
            if event_type == "cli_progress":
                progress_events.append(ev)
            elif event_type == "model_assigned":
                model_assigned_events.append(ev)
            elif event_type == "node_started":
                node_started_events.append(ev)
            elif event_type == "routing_decision":
                routing_decision_events.append(ev)

    from sage.bench.event_ledger import categorize_timeout

    return categorize_timeout(
        progress_events=progress_events,
        model_assigned_events=model_assigned_events,
        node_started_events=node_started_events,
        routing_decision_events=routing_decision_events,
        elapsed_total_ms=task_timeout_s * 1000.0,
    )


def _timeout_task_result(
    task: dict[str, Any],
    output_dir: Path,
    *,
    prefix: str,
    fmt_module: Any,
    task_timeout_s: float,
    expect_default_pipeline_learn: bool,
) -> dict[str, Any]:
    instance_id = task["instance_id"]
    per_task_events = output_dir / "per_task" / f"{instance_id}.events.jsonl"
    per_task_events.parent.mkdir(parents=True, exist_ok=True)
    needs_leading_newline = False
    if per_task_events.is_file() and per_task_events.stat().st_size > 0:
        with per_task_events.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            needs_leading_newline = existing.read(1) not in {b"\n", b"\r"}
    timeout_line = json.dumps(
            {
                "event_type": "runner_timeout",
                "instance_id": instance_id,
                "task_timeout_s": task_timeout_s,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    with per_task_events.open("a", encoding="utf-8", newline="\n") as handle:
        if needs_leading_newline:
            handle.write("\n")
        handle.write(timeout_line + "\n")
    event_audit = _event_audit_from_file(per_task_events)
    # Block A5 categorization — read AFTER the runner_timeout line is
    # appended so the events file is in its final shape.
    timeout_categorization = _categorize_timeout_from_events(
        per_task_events,
        task_timeout_s=task_timeout_s,
    )
    summary = {
        "instance_id": instance_id,
        "exit_code": None,
        "latency_ms": int(task_timeout_s * 1000),
        "total_cost_usd": 0.0,
        "extracted_patch_present": False,
        "extracted_patch_chars": 0,
        "mock": False,
        "timeout": True,
        "events_path": str(per_task_events.relative_to(output_dir)),
        "_verifier_repair_budget_usd": None,
        "_diff_verifier_mismatches": None,
        "_diff_verifier_outcome": None,
        "model_id_final": (
            timeout_categorization.get("model_id_final")
            or event_audit.get("model_id_final")
        ),
        "provider_final": (
            timeout_categorization.get("provider_final")
            or event_audit.get("provider_final")
        ),
        "_observed_model_ids": event_audit.get("_observed_model_ids", []),
        "_observed_providers": event_audit.get("_observed_providers", []),
        "_assigned_model_ids": event_audit.get("_assigned_model_ids", []),
        "_assigned_providers": event_audit.get("_assigned_providers", []),
        "_execution_model_ids": event_audit.get("_execution_model_ids", []),
        "_execution_providers": event_audit.get("_execution_providers", []),
        "_provider_policy_failure_seen": event_audit.get(
            "_provider_policy_failure_seen",
            False,
        ),
        "_observed_event_cost_usd": event_audit.get("_observed_event_cost_usd", 0.0),
        # Block A5: timeout categorization (cgpro DESIGN 2026-05-10).
        # Distinguishes scoring_boot_impossible / reasoner_thinking_overflow /
        # provider_call_timeout / stage_deadlock so a timeout reports
        # something actionable rather than just "120s exceeded".
        "timeout_categorization": {
            "last_stage": timeout_categorization["last_stage"],
            "elapsed_ms_by_stage": timeout_categorization["elapsed_ms_by_stage"],
            "provider_attempted": timeout_categorization["provider_attempted"],
            "reason_code": timeout_categorization["reason_code"],
        },
        "learning_evidence_boundary": _learning_evidence_no_go(
            reason_code="task_timeout",
            detail=f"task exceeded timeout_s={task_timeout_s}",
            run_id=None,
            source_trace_dir=None,
            archived_trace_dir=None,
            expect_default_pipeline_learn=expect_default_pipeline_learn,
        ),
    }
    return {
        "summary": summary,
        "record": fmt_module.format_patch(instance_id, "", prefix=prefix),
    }


async def _run_one_task(
    task: dict[str, Any],
    output_dir: Path,
    *,
    mock: bool,
    budget_usd: float,
    tier: str,
    provider_allowlist: tuple[str, ...],
    provider_denylist: tuple[str, ...],
    prefix: str,
    fmt_module: Any,
    claim_default_pipeline_learning_evidence: bool,
    expect_default_pipeline_learn: bool,
) -> dict[str, Any]:
    """Run one task end-to-end. Returns a per-task summary dict."""
    instance_id = task["instance_id"]
    log.info("Running task %s (mock=%s)", instance_id, mock)

    per_task_events = output_dir / "per_task" / f"{instance_id}.events.jsonl"

    if mock:
        # No subprocess — synthesize a patch + minimal "summary".
        patch = _synthetic_minimal_patch(instance_id)
        summary = {
            "instance_id": instance_id,
            "exit_code": 0,
            "latency_ms": 0,
            "total_cost_usd": 0.0,
            "extracted_patch_present": bool(patch),
            "extracted_patch_chars": len(patch),
            "mock": True,
            "timeout": False,
            "events_path": str(per_task_events.relative_to(output_dir)),
            "learning_evidence_boundary": _learning_evidence_not_requested(),
        }
        # Write a dummy events file so per_task dir is uniform.
        per_task_events.parent.mkdir(parents=True, exist_ok=True)
        per_task_events.write_text(
            json.dumps(
                {
                    "event_type": "synthetic_mock",
                    "instance_id": instance_id,
                    "note": "mock mode — no real CLI invocation",
                }
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    else:
        # Build the prompt: problem_statement + minimal "produce a
        # unified diff" framing (NOT a sophisticated SWE-bench prompt
        # — Tier 2.1 smoke only validates wire-up).
        problem = task.get("problem_statement", "").strip()
        prompt = (
            f"{problem}\n\n"
            f"---\n\n"
            f"Produce a unified diff (`diff --git`) that resolves the "
            f"issue above. Output ONLY the diff."
        )
        cli_result = await _run_sage_cli(
            prompt,
            budget_usd=budget_usd,
            output_events_path=per_task_events,
            tier=tier,
            provider_allowlist=provider_allowlist,
            provider_denylist=provider_denylist,
        )

        # Extract patch from final_result
        agent_output = ""
        payload = cli_result.get("final_result_payload") or {}
        if isinstance(payload, str):
            agent_output = payload
        elif isinstance(payload, dict):
            for key in ("result", "output", "text", "content", "answer"):
                val = payload.get(key)
                if isinstance(val, str) and val.strip():
                    agent_output = val
                    break

        patch = _extract_patch_from_text(agent_output)
        if claim_default_pipeline_learning_evidence:
            cli_complete_payload = cli_result.get("cli_complete_payload")
            cli_outcome = (
                cli_complete_payload.get("outcome")
                if isinstance(cli_complete_payload, dict)
                else None
            )
            archive_trace_dir = (
                output_dir
                / "per_task"
                / f"{_safe_artifact_stem(instance_id)}.trace"
            )
            if cli_outcome != "success":
                learning_evidence = _learning_evidence_no_go(
                    reason_code="cli_outcome_not_success",
                    detail=f"cli_complete outcome was {cli_outcome!r}",
                    run_id=cli_result.get("run_id"),
                    source_trace_dir=cli_result.get("trace_dir"),
                    archived_trace_dir=archive_trace_dir,
                    expect_default_pipeline_learn=expect_default_pipeline_learn,
                )
            else:
                learning_evidence = _validate_learning_evidence(
                    cli_result.get("trace_dir"),
                    cli_result.get("run_id"),
                    archive_trace_dir=archive_trace_dir,
                    expect_default_pipeline_learn=expect_default_pipeline_learn,
                )
        else:
            learning_evidence = _learning_evidence_not_requested()

        summary = {
            "instance_id": instance_id,
            "exit_code": cli_result["exit_code"],
            "latency_ms": cli_result["latency_ms"],
            "total_cost_usd": cli_result.get("total_cost_usd"),
            "extracted_patch_present": bool(patch),
            "extracted_patch_chars": len(patch),
            "mock": False,
            "timeout": False,
            "events_path": str(per_task_events.relative_to(output_dir)),
            "model_id_final": cli_result.get("model_id_final"),
            "provider_final": cli_result.get("provider_final"),
            "_observed_model_ids": cli_result.get("_observed_model_ids", []),
            "_observed_providers": cli_result.get("_observed_providers", []),
            "_assigned_model_ids": cli_result.get("_assigned_model_ids", []),
            "_assigned_providers": cli_result.get("_assigned_providers", []),
            "_execution_model_ids": cli_result.get("_execution_model_ids", []),
            "_execution_providers": cli_result.get("_execution_providers", []),
            "_provider_policy_failure_seen": cli_result.get(
                "_provider_policy_failure_seen",
                False,
            ),
            "_observed_event_cost_usd": cli_result.get("_observed_event_cost_usd", 0.0),
            "_verifier_repair_budget_usd": None,
            "_diff_verifier_mismatches": None,
            "_diff_verifier_outcome": None,
            "stderr_chars": len(cli_result.get("stderr", "")),
            "learning_evidence_boundary": learning_evidence,
        }

    record = fmt_module.format_patch(instance_id, patch, prefix=prefix)
    return {"summary": summary, "record": record}


# ── Main ────────────────────────────────────────────────────────────────────


async def run(
    instances_json: Path,
    output_dir: Path,
    *,
    mock: bool,
    limit: int,
    budget_usd: float,
    tier: str,
    prefix: str,
    global_budget_usd: float = 25.0,
    task_timeout_s: float = 120.0,
    profile: str = _DEFAULT_PROFILE,
    profile_timeout_override: bool = False,
    manifest_path: Path | None = None,
    grader_preflight_path: Path | None = None,
    ci_green_artifact: Path | None = None,
    provider_allowlist: tuple[str, ...] = ("google", "deepseek"),
    provider_denylist: tuple[str, ...] = ("openai",),
    claim_default_pipeline_learning_evidence: bool = False,
    expect_default_pipeline_learn: bool = False,
) -> int:
    if mock and claim_default_pipeline_learning_evidence:
        log.error(
            "--claim-default-pipeline-learning-evidence requires real mode; "
            "mock has no runtime trace"
        )
        return 2

    fmt_module = _load_format_patch_module()

    instances_text = instances_json.read_text(encoding="utf-8")
    instances = json.loads(instances_text)
    if not isinstance(instances, list) or not instances:
        log.error("instances.json empty or wrong shape (expected list)")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    launch_manifest = _write_launch_manifest(
        output_dir=output_dir,
        instances_json=instances_json,
        manifest_path=manifest_path,
        budget_usd=budget_usd,
        global_budget_usd=global_budget_usd,
        task_timeout_s=task_timeout_s,
        profile=profile,
        profile_timeout_override=profile_timeout_override,
        provider_allowlist=provider_allowlist,
        provider_denylist=provider_denylist,
        grader_preflight_path=grader_preflight_path,
        ci_green_artifact=ci_green_artifact,
    )

    # Apply limit
    selected = instances[:limit] if limit > 0 else instances
    log.info(
        "Selected %d/%d tasks (mock=%s, tier=%s, budget_usd=%.2f, prefix=%s)",
        len(selected), len(instances), mock, tier, budget_usd, prefix,
    )

    started_at = datetime.now(timezone.utc)
    cumulative_cost = 0.0
    cumulative_latency_ms = 0
    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    budget_stop_reasons: list[str] = []

    for i, task in enumerate(selected):
        if not mock and cumulative_cost >= global_budget_usd:
            budget_stop_reasons.append("global_budget_exhausted_before_task")
            log.warning(
                "Global budget exhausted: cumulative_cost=$%.2f >= $%.2f. "
                "Stopping early at task %d/%d.",
                cumulative_cost, global_budget_usd, i, len(selected),
            )
            break
        if not mock and cumulative_cost + budget_usd > global_budget_usd:
            budget_stop_reasons.append("task_budget_would_exceed_global_cap")
            log.warning(
                "Task budget would exceed global cap: $%.2f + $%.2f > $%.2f. "
                "Stopping early at task %d/%d.",
                cumulative_cost, budget_usd, global_budget_usd, i, len(selected),
            )
            break
        try:
            result = await asyncio.wait_for(
                _run_one_task(
                    task,
                    output_dir,
                    mock=mock,
                    budget_usd=budget_usd,
                    tier=tier,
                    provider_allowlist=provider_allowlist,
                    provider_denylist=provider_denylist,
                    prefix=prefix,
                    fmt_module=fmt_module,
                    claim_default_pipeline_learning_evidence=claim_default_pipeline_learning_evidence,
                    expect_default_pipeline_learn=expect_default_pipeline_learn,
                ),
                timeout=task_timeout_s,
            )
        except asyncio.TimeoutError:
            log.error("Task %s exceeded timeout_s=%.1f", task["instance_id"], task_timeout_s)
            result = _timeout_task_result(
                task,
                output_dir,
                prefix=prefix,
                fmt_module=fmt_module,
                task_timeout_s=task_timeout_s,
                expect_default_pipeline_learn=expect_default_pipeline_learn,
            )
        summaries.append(result["summary"])
        records.append(result["record"])
        cost = result["summary"].get("total_cost_usd") or 0.0
        cumulative_cost += cost if isinstance(cost, (int, float)) else 0.0
        cumulative_latency_ms += result["summary"].get("latency_ms", 0)

    # Write predictions.json (Pro shape)
    predictions_path = output_dir / "predictions.json"
    fmt_module.write_predictions(records, predictions_path)
    predictions_jsonl_path = output_dir / "predictions.jsonl"
    _write_predictions_jsonl(records, summaries, predictions_jsonl_path)
    events_path = output_dir / "events.jsonl"
    _write_aggregate_events(summaries, output_dir=output_dir, output_path=events_path)

    # Write summary
    summary_doc: dict[str, Any] = {
        "run_started_at_utc": started_at.isoformat(),
        "run_ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "mock" if mock else "real",
        "tier": tier if not mock else None,
        "tasks_run": len(summaries),
        "tasks_in_set": len(instances),
        "cumulative_cost_usd": cumulative_cost,
        "cumulative_latency_ms": cumulative_latency_ms,
        "budget": {
            "budget_usd_per_task": budget_usd,
            "global_budget_usd": global_budget_usd,
            "task_timeout_s": task_timeout_s,
            "effective_profile": profile,
            "profile_timeout_default_s": _TIMEOUT_PROFILES.get(profile),
            "profile_timeout_override": profile_timeout_override,
            "stop_reasons": budget_stop_reasons,
        },
        "predictions_path": str(predictions_path.relative_to(output_dir)),
        "predictions_jsonl_path": str(predictions_jsonl_path.relative_to(output_dir)),
        "events_path": str(events_path.relative_to(output_dir)),
        "per_task_dir": "per_task/",
        "patches_extracted": sum(
            1 for s in summaries if s["extracted_patch_present"]
        ),
        "patches_empty": sum(
            1 for s in summaries if not s["extracted_patch_present"]
        ),
        "task_summaries": summaries,
    }
    evidence_items = [
        item
        for item in (summary.get("learning_evidence_boundary") for summary in summaries)
        if isinstance(item, dict)
    ]
    summary_doc["learning_evidence_gate"] = {
        "claimed": claim_default_pipeline_learning_evidence,
        "status": (
            "NO_GO"
            if any(item.get("status") == "no_go" for item in evidence_items)
            else "PASS" if claim_default_pipeline_learning_evidence else "NOT_CLAIMED"
        ),
        "expect_default_pipeline_learn": expect_default_pipeline_learn,
        "passed": sum(1 for item in evidence_items if item.get("status") == "pass"),
        "failed": sum(1 for item in evidence_items if item.get("status") == "no_go"),
        "skipped": sum(1 for item in evidence_items if item.get("status") == "skipped"),
    }
    budget_gate = {
        "status": "NO_GO" if cumulative_cost > global_budget_usd else "PASS",
        "cumulative_cost_usd": cumulative_cost,
        "global_budget_usd": global_budget_usd,
        "stop_reasons": budget_stop_reasons,
    }
    timeout_gate = {
        "status": (
            "NO_GO"
            if sum(1 for summary in summaries if summary.get("timeout")) > 1
            else "PASS"
        ),
        "timeouts": sum(1 for summary in summaries if summary.get("timeout")),
    }
    summary_doc["budget_gate"] = budget_gate
    summary_doc["timeout_gate"] = timeout_gate
    provider_gate = _provider_gate(
        summaries,
        mock=mock,
        provider_allowlist=provider_allowlist,
        provider_denylist=provider_denylist,
    )
    summary_doc["launch_manifest_path"] = "launch_manifest.json"
    summary_doc["acceptance_gate_results"] = {
        "manifest_gate": launch_manifest["manifest_gate"],
        "budget_gate": budget_gate,
        "timeout_gate": timeout_gate,
        "provider_gate": provider_gate,
        "grading_gate": launch_manifest["grading_gate"],
        "ci_gate": launch_manifest["ci_gate"],
    }
    summary_doc["canary_decision"] = _combine_canary_decision(
        summary_doc["acceptance_gate_results"]
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    log.info(
        "Done. predictions=%d patches_extracted=%d cost=$%.4f latency=%.1fs",
        len(records),
        summary_doc["patches_extracted"],
        cumulative_cost,
        cumulative_latency_ms / 1000,
    )
    if claim_default_pipeline_learning_evidence and (
        not summaries
        or any(
            (summary.get("learning_evidence_boundary") or {}).get("status") == "no_go"
            for summary in summaries
        )
    ):
        log.error("Learning side-effect evidence boundary failed")
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances-json",
        type=Path,
        required=True,
        help="path to instances.json from swebench_pro_fetch.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="output dir for predictions.json + per_task/ + summary.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="run at most N tasks (default 1, smoke discipline)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="synthetic patches (NO API spend); validates wire-up only",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=5.0,
        help="per-task spend cap forwarded to sage CLI (default 5.0)",
    )
    parser.add_argument(
        "--global-budget-usd",
        type=float,
        default=25.0,
        help="global spend cap for this runner invocation (default 25.0)",
    )
    parser.add_argument(
        "--task-timeout-s",
        type=float,
        default=None,
        help=(
            "per-task wall-clock timeout in seconds. Sentinel ``None`` "
            "means: resolve from --profile. Explicit value overrides the "
            "profile-driven default and is reported as a profile_override."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=sorted(_TIMEOUT_PROFILES),
        default=_DEFAULT_PROFILE,
        help=(
            "Named timeout profile. ``default`` keeps 120s for plumbing "
            "smokes; ``graded_patch_generation`` gives the agent 900s for "
            "real-LLM SWE-bench Pro canaries (cgpro DESIGN 2026-05-11 "
            "envelope 600-1200s)."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=_DEFAULT_CANARY_MANIFEST_PATH,
        help=(
            "human canary manifest to copy/hash into launch artifacts "
            f"(default {_DEFAULT_CANARY_MANIFEST_PATH})"
        ),
    )
    parser.add_argument(
        "--grader-preflight-path",
        type=Path,
        default=None,
        help="optional SWE-bench Pro grader preflight artifact for GO gating",
    )
    parser.add_argument(
        "--ci-green-artifact",
        type=Path,
        default=None,
        help="optional machine-readable CI-green artifact for GO gating",
    )
    parser.add_argument(
        "--provider-allowlist",
        default="google,deepseek",
        help="comma-separated provider allowlist forwarded to sage run and audit",
    )
    parser.add_argument(
        "--provider-denylist",
        default="openai",
        help="comma-separated provider denylist forwarded to sage run and audit",
    )
    parser.add_argument(
        "--tier",
        default=_DEFAULT_TIER,
        help=f"SAGE_LLM_TIER override (default {_DEFAULT_TIER})",
    )
    parser.add_argument(
        "--prefix",
        default="ygn-sage-arm-d-smoke",
        help="prefix label written into each Pro record",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--claim-default-pipeline-learning-evidence",
        action="store_true",
        help=(
            "Claim oracle-enabled default-pipeline learning evidence; after "
            "each real task, archive the canonical RuntimeEventLog trace, run "
            "the evidence-boundary validator, and fail the harness if it fails."
        ),
    )
    parser.add_argument(
        "--expect-default-pipeline-learn",
        action="store_true",
        help=(
            "With --claim-default-pipeline-learning-evidence, require the "
            "minimal current default Stage 5 learning decision set."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.instances_json.is_file():
        log.error(
            "--instances-json not found: %s. "
            "Run `swebench_pro_fetch.py` first.",
            args.instances_json,
        )
        return 2
    if (
        args.expect_default_pipeline_learn
        and not args.claim_default_pipeline_learning_evidence
    ):
        parser.error(
            "--expect-default-pipeline-learn requires "
            "--claim-default-pipeline-learning-evidence"
        )
    if args.mock and args.claim_default_pipeline_learning_evidence:
        parser.error(
            "--claim-default-pipeline-learning-evidence requires real mode; "
            "remove --mock"
        )

    profile_timeout = _TIMEOUT_PROFILES[args.profile]
    if args.task_timeout_s is None:
        effective_task_timeout_s = profile_timeout
        timeout_override = False
    else:
        effective_task_timeout_s = float(args.task_timeout_s)
        timeout_override = True
        log.info(
            "Explicit --task-timeout-s=%.1f overrides profile %r default %.1f",
            effective_task_timeout_s,
            args.profile,
            profile_timeout,
        )

    return asyncio.run(
        run(
            args.instances_json,
            args.output_dir,
            mock=args.mock,
            limit=args.limit,
            budget_usd=args.budget_usd,
            global_budget_usd=args.global_budget_usd,
            task_timeout_s=effective_task_timeout_s,
            profile=args.profile,
            profile_timeout_override=timeout_override,
            manifest_path=args.manifest_path,
            grader_preflight_path=args.grader_preflight_path,
            ci_green_artifact=args.ci_green_artifact,
            provider_allowlist=tuple(
                item.strip() for item in args.provider_allowlist.split(",") if item.strip()
            ),
            provider_denylist=tuple(
                item.strip() for item in args.provider_denylist.split(",") if item.strip()
            ),
            tier=args.tier,
            prefix=args.prefix,
            claim_default_pipeline_learning_evidence=(
                args.claim_default_pipeline_learning_evidence
            ),
            expect_default_pipeline_learn=args.expect_default_pipeline_learn,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
