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

# Path to the Pro patch format helper (loaded as module).
_FORMAT_PATCH_PATH = (
    Path(__file__).parent / "swebench_pro_format_patch.py"
).resolve()


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
) -> dict[str, Any]:
    """Invoke `python -m sage.cli run --jsonl` as subprocess.

    Uses the JSONL stdin protocol (matching the direct CLI path):
    writes ``{"command":"prompt","payload":{"task":...}}`` to stdin,
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
    log.info("Spawning sage CLI: %s", " ".join(cmd))

    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # The CLI reads stdin as raw task text in one-shot mode
    # (when no positional arg is given). Write the task text,
    # then close stdin to signal EOF.
    assert proc.stdin is not None
    proc.stdin.write(task_text.encode("utf-8"))
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
            if ev_type == "final_result":
                final_result_payload = event.get("payload", {})
            elif ev_type == "cli_complete":
                cli_complete_payload = event.get("payload", {})
                event_run_id = event.get("run_id")
                cli_complete_run_id = (
                    event_run_id if isinstance(event_run_id, str) else None
                )

    exit_code = await proc.wait()
    latency_s = time.monotonic() - start
    stderr_bytes = await stderr_task

    if cli_complete_payload is None:
        stderr_text = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        log.error(
            "CLI ended without cli_complete; exit_code=%s stderr=%s",
            exit_code, stderr_text[-4000:],
        )

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


async def _run_one_task(
    task: dict[str, Any],
    output_dir: Path,
    *,
    mock: bool,
    budget_usd: float,
    tier: str,
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

    for i, task in enumerate(selected):
        # Per cgpro DESIGN E hard cutoff: $5 USD total.
        if not mock and cumulative_cost > 5.0:
            log.warning(
                "Hard cutoff hit: cumulative_cost=$%.2f > $5.00. "
                "Stopping early at task %d/%d.",
                cumulative_cost, i, len(selected),
            )
            break
        result = await _run_one_task(
            task,
            output_dir,
            mock=mock,
            budget_usd=budget_usd,
            tier=tier,
            prefix=prefix,
            fmt_module=fmt_module,
            claim_default_pipeline_learning_evidence=claim_default_pipeline_learning_evidence,
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

    # Write summary
    summary_doc = {
        "run_started_at_utc": started_at.isoformat(),
        "run_ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "mock" if mock else "real",
        "tier": tier if not mock else None,
        "tasks_run": len(summaries),
        "tasks_in_set": len(instances),
        "cumulative_cost_usd": cumulative_cost,
        "cumulative_latency_ms": cumulative_latency_ms,
        "predictions_path": str(predictions_path.relative_to(output_dir)),
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

    return asyncio.run(
        run(
            args.instances_json,
            args.output_dir,
            mock=args.mock,
            limit=args.limit,
            budget_usd=args.budget_usd,
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
