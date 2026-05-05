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
    """Invoke `python -m sage.cli run --jsonl <task>` as subprocess.

    Captures stdout to `output_events_path` (line-by-line JSONL).
    Returns a dict summarizing the run: total_cost, latency_ms,
    final_result_payload, exit_code.
    """
    output_events_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SAGE_LLM_TIER"] = tier
    env["SAGE_DIFF_VERIFIER_MODE"] = "observe"
    env["SAGE_OTEL_EXPORTER"] = "none"
    # Cycle-13 E Tier 2.1 smoke discovery 2026-05-05: per directive #8
    # (A14 posterior epoch guard fail-closed), the post-save manifest
    # gap (advisor-flagged 2026-05-04, separate ticket) leaves
    # `~/.sage/` with state files but no `topology_state_manifest.json`
    # after every successful pipeline run. Subsequent boots fail. The
    # bypass env disables atexit save AND skips the boot guard — for
    # bench/smoke runs this is the right semantics (no state pollution
    # across smokes; learning is irrelevant for shape validation).
    env.setdefault("SAGE_BOOT_BYPASS_EPOCH_GUARD", "1")
    env.setdefault(
        "SAGE_BOOT_BYPASS_REASON",
        "cycle-13 E Tier 2.1 arm D smoke run; bypass disables atexit "
        "save so smokes do not pollute ~/.sage/ across consecutive runs",
    )
    env.setdefault("SAGE_OPERATOR_ID", "ygn-sage-arm-d-smoke")

    cmd = [
        sys.executable,
        "-m",
        "sage.cli",
        "run",
        "--jsonl",
        "--budget-usd",
        str(budget_usd),
        task_text,
    ]
    log.info("Spawning sage CLI: %s", " ".join(cmd[:5]))

    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    final_result_payload: dict[str, Any] | None = None
    cli_complete_payload: dict[str, Any] | None = None

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

    exit_code = await proc.wait()
    latency_s = time.monotonic() - start
    stderr_bytes = await proc.stderr.read() if proc.stderr is not None else b""

    return {
        "exit_code": exit_code,
        "latency_ms": int(latency_s * 1000),
        "final_result_payload": final_result_payload,
        "cli_complete_payload": cli_complete_payload,
        "total_cost_usd": (
            (cli_complete_payload or {}).get("total_cost") if cli_complete_payload else None
        ),
        "stderr": stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else "",
    }


# ── Per-task runner ──────────────────────────────────────────────────────────


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
        for key in ("result", "output", "text", "content", "answer"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                agent_output = val
                break

        patch = _extract_patch_from_text(agent_output)

        summary = {
            "instance_id": instance_id,
            "exit_code": cli_result["exit_code"],
            "latency_ms": cli_result["latency_ms"],
            "total_cost_usd": cli_result.get("total_cost_usd"),
            "extracted_patch_present": bool(patch),
            "extracted_patch_chars": len(patch),
            "mock": False,
            "stderr_chars": len(cli_result.get("stderr", "")),
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
) -> int:
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

    return asyncio.run(
        run(
            args.instances_json,
            args.output_dir,
            mock=args.mock,
            limit=args.limit,
            budget_usd=args.budget_usd,
            tier=args.tier,
            prefix=args.prefix,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
