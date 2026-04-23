"""SWE-bench typed-vs-bash parity smoke — red-team plan §5 decision gate.

Runs the same SWE-bench Lite slice twice:

  * Arm A (baseline): SAGE_DANGEROUS_TOOLS=1 — execute_bash registered,
    matches the pre-2026-04-22 default.
  * Arm B (typed-only): SAGE_DANGEROUS_TOOLS=0 — execute_bash NOT
    registered. Agent must solve using the P0.1 typed repo tools
    (read_file, search_repo, list_files, run_tests, apply_patch,
    git_diff).

The decision gate for flipping `AgentConfig.dangerous_tools=False`
default in boot.py is: does typed-only mode *function*? Per the
red-team plan §5, "±2 pp parity" at N=50 is below the noise floor
(per-task variance ~10 pp → combined SE ~2 pp even at N=50). The
honest, measurable criterion at smoke scale is:

  * typed-only arm produces >0 patches → safe to flip; scale to
    larger N for confidence interval if needed.
  * typed-only arm produces 0 patches → template / tool-description
    is the blocker, not capability. Fix prereq before flipping.

Usage:

    # Gen-only sanity (fast, cheap) — N=10, ~50 min, ~$10
    python sage-python/scripts/swebench_parity_smoke.py --limit 10 --generate-only

    # Full paired with Docker eval — N=30, ~4h, ~$30-50
    python sage-python/scripts/swebench_parity_smoke.py --limit 30

Outputs (saved under docs/benchmarks/<date>-swebench-parity-smoke/):
    * {date}-swebench-parity-bash.json   — Arm A bench report
    * {date}-swebench-parity-typed.json  — Arm B bench report
    * {date}-swebench-parity-summary.md  — side-by-side comparison
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_arm(
    *,
    repo_root: Path,
    label: str,
    dangerous_tools: str,
    limit: int,
    offset: int,
    out_json: Path,
    predictions_dest: Path,
    generate_only: bool,
) -> int:
    """Run one arm of the parity smoke. Returns the bench process
    exit code. In --generate-only mode the bench writes predictions
    to a temp dir and ignores --output; we parse the stdout for the
    temp path and copy the JSONL + meta into `predictions_dest`."""
    env = os.environ.copy()
    env["SAGE_DANGEROUS_TOOLS"] = dangerous_tools
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable, "-m", "sage.bench",
        "--type", "swebench",
        "--dataset", "lite",
        "--limit", str(limit),
        "--offset", str(offset),
        "--output", str(out_json),
    ]
    if generate_only:
        cmd.append("--generate-only")

    print(f"\n{'=' * 60}")
    print(f"  Arm: {label}")
    print(f"  SAGE_DANGEROUS_TOOLS={dangerous_tools}")
    print(f"  Cmd: {' '.join(cmd)}")
    print(f"  Output (docker-eval mode): {out_json}")
    print(f"  Predictions dest:          {predictions_dest}")
    print(f"{'=' * 60}\n", flush=True)

    # Capture stdout so we can grep the "Predictions saved to:" line —
    # in --generate-only mode that's the only record of where the JSONL
    # actually landed, because --output is only honored in full-eval mode.
    proc = subprocess.Popen(
        cmd, cwd=str(repo_root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    rc = proc.wait()

    if generate_only:
        predictions_src = _extract_predictions_path(captured)
        if predictions_src and predictions_src.exists():
            predictions_dest.parent.mkdir(parents=True, exist_ok=True)
            predictions_dest.write_bytes(predictions_src.read_bytes())
            meta_src = predictions_src.with_name("predictions_meta.json")
            if meta_src.exists():
                meta_dest = predictions_dest.with_name(
                    predictions_dest.stem + "-meta.json"
                )
                meta_dest.write_bytes(meta_src.read_bytes())
            print(f"\n  Copied predictions to: {predictions_dest}")
        else:
            print(
                "\n  WARN: could not locate predictions file from bench stdout — "
                "summary will be empty.", file=sys.stderr,
            )
    return rc


def _extract_predictions_path(lines: list[str]) -> Path | None:
    """Parse the bench's 'Predictions saved to: <path>' line out of
    captured stdout. Returns the Path or None if absent."""
    marker = "Predictions saved to:"
    for line in reversed(lines):
        if marker in line:
            return Path(line.split(marker, 1)[1].strip())
    return None


def summarise_predictions_jsonl(path: Path) -> dict:
    """Summarise a predictions.jsonl file (the generate-only output).
    Each line is a dict with at least instance_id and model_patch."""
    if not path.exists():
        return {"error": f"Predictions file not found: {path}"}
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    n = len(lines)
    patches = sum(1 for r in lines if r.get("model_patch", "").strip())
    per_task = [
        (r["instance_id"], "PATCH" if r.get("model_patch", "").strip() else "EMPTY")
        for r in lines
    ]
    return {
        "n": n,
        "patches": patches,
        "patch_rate": patches / n if n else 0.0,
        "empty": n - patches,
        "per_task": per_task,
    }


def write_summary(
    *,
    out_md: Path,
    bash_predictions: Path,
    typed_predictions: Path,
    limit: int,
    offset: int,
    generate_only: bool,
) -> None:
    b = summarise_predictions_jsonl(bash_predictions)
    t = summarise_predictions_jsonl(typed_predictions)
    if "error" in b or "error" in t:
        print(f"WARN: {b.get('error', '')} {t.get('error', '')}", file=sys.stderr)
        return

    typed_functions = t["patches"] > 0
    patch_gap = abs(b["patch_rate"] - t["patch_rate"]) * 100.0

    lines = [
        "# SWE-bench typed-vs-bash parity smoke",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Slice: Lite, limit={limit}, offset={offset}",
        f"Mode: {'generate-only' if generate_only else 'full Docker eval'}",
        "",
        "## Headline",
        "",
        "| Metric | Arm A (bash) | Arm B (typed-only) |",
        "|---|---|---|",
        f"| N | {b['n']} | {t['n']} |",
        f"| Patches produced | {b['patches']} ({b['patch_rate']:.0%}) | {t['patches']} ({t['patch_rate']:.0%}) |",
        f"| Empty | {b['empty']} | {t['empty']} |",
        "",
        "## Per-task breakdown",
        "",
        "| instance_id | Arm A | Arm B |",
        "|---|---|---|",
    ]
    # Per-task is indexed by instance_id — join the two arms on that key.
    b_map = dict(b["per_task"])
    t_map = dict(t["per_task"])
    for iid in sorted(set(b_map) | set(t_map)):
        lines.append(f"| {iid} | {b_map.get(iid, '?')} | {t_map.get(iid, '?')} |")
    lines += [
        "",
        "## Decision gate (functional criterion)",
        "",
        f"Typed-only arm produces patches at all: **{'YES' if typed_functions else 'NO'}**",
        "",
    ]
    if typed_functions:
        lines.append(
            "=> Safe to flip `AgentConfig.dangerous_tools=False` default on the "
            "functional criterion. Scale to Docker eval at larger N if a "
            "resolved-rate confidence interval matters."
        )
    else:
        lines.append(
            "=> DO NOT FLIP. Typed-only arm produced zero patches; the template "
            "or tool descriptions are blocking the model. Fix that before re-smoking."
        )
    lines += [
        "",
        "## Statistical caveat",
        "",
        f"Observed patch-rate gap: {patch_gap:.1f} pp.",
        "",
        "Per-task variance is ~10 pp. At N=10 the combined arm-gap standard error "
        "is ~15 pp — a 10 pp gap is inside noise. The red-team plan's '±2 pp at "
        "N=50' criterion is below the noise floor even at N=50 (combined SE ~2 pp); "
        "confirming ±2 pp parity statistically would need N~600 per arm.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10,
                    help="Tasks per arm (default 10 for sanity smoke)")
    ap.add_argument("--offset", type=int, default=0,
                    help="HF dataset offset (default 0)")
    ap.add_argument("--generate-only", action="store_true",
                    help="Skip Docker eval — patch-rate only (faster, cheaper)")
    ap.add_argument("--skip-bash", action="store_true",
                    help="Skip Arm A (bash). Use if you already have a "
                         "baseline report from an earlier run.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    today = datetime.now().strftime("%Y-%m-%d")
    bench_dir = repo_root / "docs" / "benchmarks" / f"{today}-swebench-parity-smoke"
    bench_dir.mkdir(parents=True, exist_ok=True)

    bash_json = bench_dir / f"{today}-parity-bash.json"
    typed_json = bench_dir / f"{today}-parity-typed.json"
    bash_predictions = bench_dir / f"{today}-parity-bash-predictions.jsonl"
    typed_predictions = bench_dir / f"{today}-parity-typed-predictions.jsonl"
    summary_md = bench_dir / f"{today}-parity-summary.md"

    if not args.skip_bash:
        rc = run_arm(
            repo_root=repo_root,
            label="A (bash baseline)",
            dangerous_tools="1",
            limit=args.limit,
            offset=args.offset,
            out_json=bash_json,
            predictions_dest=bash_predictions,
            generate_only=args.generate_only,
        )
        if rc != 0:
            print(f"\nArm A exited {rc}; continuing to Arm B.", file=sys.stderr)

    rc = run_arm(
        repo_root=repo_root,
        label="B (typed-only)",
        dangerous_tools="0",
        limit=args.limit,
        offset=args.offset,
        out_json=typed_json,
        predictions_dest=typed_predictions,
        generate_only=args.generate_only,
    )
    if rc != 0:
        print(f"\nArm B exited {rc}.", file=sys.stderr)

    write_summary(
        out_md=summary_md,
        bash_predictions=bash_predictions,
        typed_predictions=typed_predictions,
        limit=args.limit,
        offset=args.offset,
        generate_only=args.generate_only,
    )
    print(f"\nSummary written to: {summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
