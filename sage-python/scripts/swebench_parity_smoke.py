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
    generate_only: bool,
) -> int:
    """Run one arm of the parity smoke. Returns the bench process
    exit code."""
    env = os.environ.copy()
    env["SAGE_DANGEROUS_TOOLS"] = dangerous_tools
    # Unbuffered so the log file tails cleanly.
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
    print(f"  Output: {out_json}")
    print(f"{'=' * 60}\n", flush=True)

    return subprocess.call(cmd, cwd=str(repo_root), env=env)


def load_report(path: Path) -> dict:
    if not path.exists():
        return {"error": f"Report file not found: {path}"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse failed: {e}"}


def summarise(report: dict) -> dict:
    """Extract the headline numbers we care about for parity."""
    results = report.get("results") or []
    n = len(results)

    def _count(key: str) -> int:
        return sum(1 for r in results if r.get(key))

    patches = _count("prediction")
    resolved = sum(
        1 for r in results
        if r.get("eval") and r["eval"].get("resolved")
    )
    empty = sum(
        1 for r in results
        if not r.get("prediction")
    )
    errors = _count("error")

    return {
        "n": n,
        "patches": patches,
        "patch_rate": patches / n if n else 0.0,
        "resolved": resolved,
        "resolved_rate": resolved / n if n else 0.0,
        "empty": empty,
        "errors": errors,
    }


def write_summary(
    *,
    out_md: Path,
    bash_report: dict,
    typed_report: dict,
    limit: int,
    offset: int,
    generate_only: bool,
) -> None:
    b = summarise(bash_report)
    t = summarise(typed_report)

    # Functional criterion (the real decision gate at smoke scale):
    typed_functions = t["patches"] > 0

    # Parity band (±2pp) — informational only; smoke N cannot confirm
    # it statistically.
    patch_gap = abs(b["patch_rate"] - t["patch_rate"]) * 100.0
    resolved_gap = abs(b["resolved_rate"] - t["resolved_rate"]) * 100.0

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
        f"| Resolved (Docker) | {b['resolved']} ({b['resolved_rate']:.0%}) | {t['resolved']} ({t['resolved_rate']:.0%}) |",
        f"| Empty | {b['empty']} | {t['empty']} |",
        f"| Errors | {b['errors']} | {t['errors']} |",
        "",
        "## Decision gate (functional criterion)",
        "",
        f"Typed-only arm produces patches at all: **{'YES' if typed_functions else 'NO'}**",
        "",
    ]
    if typed_functions:
        lines.append(
            "=> **Safe to flip `AgentConfig.dangerous_tools=False` default** on "
            "the functional-parity criterion. Scale to a larger N if a confidence "
            "interval on the pass-rate gap matters for your decision."
        )
    else:
        lines.append(
            "=> **DO NOT FLIP.** Typed-only arm produced zero patches; the template "
            "or tool descriptions are blocking the model. Fix that before re-smoking."
        )
    lines.extend([
        "",
        "## Statistical caveat",
        "",
        f"Observed patch-rate gap: {patch_gap:.1f} pp "
        f"({'within' if patch_gap <= 2 else 'outside'} the ±2 pp parity band).",
        f"Observed resolved-rate gap: {resolved_gap:.1f} pp.",
        "",
        "Per-task variance is ~10 pp. Combined arm-gap SE at this N is too wide "
        "to confirm ±2 pp parity statistically. These gaps are descriptive, not "
        "inferential. The functional criterion above is the actual decision gate "
        "at smoke scale.",
        "",
    ])

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

    bash_json = bench_dir / f"{today}-swebench-parity-bash.json"
    typed_json = bench_dir / f"{today}-swebench-parity-typed.json"
    summary_md = bench_dir / f"{today}-swebench-parity-summary.md"

    if not args.skip_bash:
        rc = run_arm(
            repo_root=repo_root,
            label="A (bash baseline)",
            dangerous_tools="1",
            limit=args.limit,
            offset=args.offset,
            out_json=bash_json,
            generate_only=args.generate_only,
        )
        if rc != 0:
            print(f"\nArm A (bash) exited with code {rc}. Continuing to Arm B.",
                  file=sys.stderr)

    rc = run_arm(
        repo_root=repo_root,
        label="B (typed-only)",
        dangerous_tools="0",
        limit=args.limit,
        offset=args.offset,
        out_json=typed_json,
        generate_only=args.generate_only,
    )
    if rc != 0:
        print(f"\nArm B (typed-only) exited with code {rc}.", file=sys.stderr)

    bash_report = load_report(bash_json)
    typed_report = load_report(typed_json)

    write_summary(
        out_md=summary_md,
        bash_report=bash_report,
        typed_report=typed_report,
        limit=args.limit,
        offset=args.offset,
        generate_only=args.generate_only,
    )
    print(f"\nSummary written to: {summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
