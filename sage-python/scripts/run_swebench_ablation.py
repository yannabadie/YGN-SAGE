"""Run a SWE-bench ablation across four configurations.

Configs
-------
- full               : everything on (topology + ToolForge + sage_recurse)
- no_sage_recurse    : recursion disabled, rest on
- no_toolforge       : tool synthesis disabled, rest on
- bare               : topology + recursion + toolforge all off

Each config is a fresh subprocess so env vars take effect cleanly (boot
registers tools based on SAGE_ABLATION_* at import-time).

Usage
-----
    python scripts/run_swebench_ablation.py \\
        --dataset lite --limit 10 --tier reasoner --out docs/benchmarks/

Outputs:
- docs/benchmarks/{date}-swebench-{dataset}-ablation.json
- docs/benchmarks/{date}-swebench-{dataset}-predictions-{config}.jsonl

Notes
-----
- `--generate-only` is always passed to the child runs; Docker evaluation
  must be done separately on Linux (see swebench_bench.evaluate_predictions).
- Total cost is roughly 4 * (single-config cost). Budget accordingly.
- The script does NOT validate that each config actually took effect in
  the child boot — assume `SAGE_ABLATION_*` plumbing is correct (covered
  by tests/test_system_hint.py + tests/test_toolforge_wiring.py).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Config:
    name: str
    env: dict[str, str]
    description: str


CONFIGS: list[Config] = [
    Config(
        name="full",
        env={},
        description="Everything on (topology, ToolForge, sage_recurse).",
    ),
    Config(
        name="no_sage_recurse",
        env={"SAGE_ABLATION_NO_RECURSE": "1"},
        description="sage_recurse tool unregistered, rest on.",
    ),
    Config(
        name="no_toolforge",
        env={"SAGE_ABLATION_NO_TOOLFORGE": "1"},
        description="ToolForge synthesis disabled, rest on.",
    ),
    Config(
        name="bare",
        env={
            "SAGE_ABLATION_NO_RECURSE": "1",
            "SAGE_ABLATION_NO_TOOLFORGE": "1",
            "SAGE_ABLATION_NO_TOPOLOGY": "1",
        },
        description="Topology + ToolForge + sage_recurse all off (LLM + basic tools).",
    ),
]


def _most_recent_tempdir() -> Path | None:
    """Return the newest /tmp/sage_swebench_* directory, if any exist.

    `sage.bench ... --generate-only` writes predictions to
    `tempfile.mkdtemp(prefix="sage_swebench_")` and returns that path only
    via stdout. We poll the filesystem immediately after the subprocess
    exits — on Windows / POSIX this is a reliable way to pick up the
    fresh dir without parsing the child's log.
    """
    import tempfile
    root = Path(tempfile.gettempdir())
    candidates = sorted(
        root.glob("sage_swebench_*"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _synth_report_from_predictions(
    preds_path: Path, wall_time_s: float, config_name: str, dataset: str,
) -> dict:
    """Compute a synthetic bench report from a predictions JSONL.

    `--generate-only` mode doesn't write a report (no Docker eval); the
    ablation still needs a pass_rate signal for decide_next_phase.py.
    We use a lightweight proxy: a patch is considered "valid" if it
    contains `diff --git` AND is >= 100 chars. Empty strings and the F2
    `[sage: agent exited after N steps with no content]` sentinel both
    fail. This is NOT the Docker-eval score — it's a generator-side
    signal that the agent actually produced a diff.
    """
    total, valid = 0, 0
    results = []
    if preds_path.exists():
        with open(preds_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                patch = p.get("model_patch", "") or ""
                is_diff = "diff --git" in patch and len(patch) >= 100
                if is_diff:
                    valid += 1
                results.append({
                    "task_id": p.get("instance_id", ""),
                    "passed": is_diff,
                    "patch_len": len(patch),
                })
    return {
        "benchmark": f"swebench_{dataset}",
        "config": config_name,
        "total": total,
        "passed": valid,
        # NB: pass_rate here = "generator produced a non-empty diff" —
        # real SWE-bench resolved_rate requires the Docker harness.
        "pass_rate": valid / total if total > 0 else 0.0,
        "wall_time_s": wall_time_s,
        "results": results,
        "_note": "synthetic: diff --git presence + len>=100, not Docker-verified",
    }


def _run_one(
    config: Config,
    dataset: str,
    limit: int,
    tier: str,
    out_dir: Path,
    date: str,
) -> dict:
    """Run one configuration via a child process and collect results."""
    base_env = os.environ.copy()
    base_env.update(config.env)

    # Deterministic output path per config.
    pred_path = out_dir / f"{date}-swebench-{dataset}-predictions-{config.name}.jsonl"
    report_path = out_dir / f"{date}-swebench-{dataset}-{config.name}.json"

    cmd = [
        sys.executable, "-m", "sage.bench",
        "--type", "swebench",
        "--dataset", dataset,
        "--limit", str(limit),
        "--tier", tier,
        "--generate-only",
        "--output", str(out_dir),
    ]

    print(f"\n{'=' * 70}", flush=True)
    print(f"  Config: {config.name}", flush=True)
    print(f"  Env:    {config.env or '(none)'}", flush=True)
    print(f"  Cmd:    {' '.join(cmd)}", flush=True)
    print('=' * 70, flush=True)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, env=base_env, capture_output=False)
    elapsed = time.perf_counter() - t0

    # Find the subprocess's tempdir, copy predictions.jsonl to our
    # stable location, and synthesize a bench report so downstream tools
    # (decide_next_phase.py) can read it.
    tempdir = _most_recent_tempdir()
    if tempdir is not None:
        src_preds = tempdir / "predictions.jsonl"
        if src_preds.exists():
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(src_preds, pred_path)
        src_meta = tempdir / "predictions_meta.json"
        if src_meta.exists():
            meta_dest = out_dir / f"{date}-swebench-{dataset}-predictions-{config.name}-meta.json"
            import shutil
            shutil.copy2(src_meta, meta_dest)
    synthetic_report = _synth_report_from_predictions(
        pred_path, wall_time_s=round(elapsed, 1),
        config_name=config.name, dataset=dataset,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(synthetic_report, indent=2), encoding="utf-8",
    )

    return {
        "config": config.name,
        "description": config.description,
        "env": config.env,
        "exit_code": result.returncode,
        "wall_time_s": round(elapsed, 1),
        "predictions_path": str(pred_path),
        "report_path": str(report_path),
        "synthetic_pass_rate": synthetic_report["pass_rate"],
        "synthetic_total": synthetic_report["total"],
        "synthetic_passed": synthetic_report["passed"],
    }


def main():
    parser = argparse.ArgumentParser(description="SWE-bench ablation runner")
    parser.add_argument("--dataset", choices=["lite", "verified", "full", "pro"],
                        default="lite", help="SWE-bench dataset (default: lite)")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of tasks per config (default: 5)")
    parser.add_argument("--tier", default="reasoner",
                        help="LLM tier (default: reasoner — needed for S3 work)")
    parser.add_argument("--out", default="docs/benchmarks/",
                        help="Output directory for reports (default: docs/benchmarks/)")
    parser.add_argument("--configs", default="full,no_sage_recurse,no_toolforge,bare",
                        help="Comma-separated subset of config names to run.")
    args = parser.parse_args()

    selected = {c.strip() for c in args.configs.split(",") if c.strip()}
    to_run = [c for c in CONFIGS if c.name in selected]
    if not to_run:
        print(f"No matching configs: {selected}. Valid: {[c.name for c in CONFIGS]}")
        sys.exit(2)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Overall report.
    ablation_report = {
        "benchmark": f"swebench_{args.dataset}",
        "date": date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tier": args.tier,
        "limit": args.limit,
        "configs_run": [c.name for c in to_run],
        "results": [],
    }

    for cfg in to_run:
        result = _run_one(cfg, args.dataset, args.limit, args.tier, out_dir, date)
        ablation_report["results"].append(result)

    out_path = out_dir / f"{date}-swebench-{args.dataset}-ablation.json"
    out_path.write_text(json.dumps(ablation_report, indent=2), encoding="utf-8")
    print(f"\nAblation report saved to {out_path}")

    # Quick summary — exit codes + synthetic generator-side pass rate.
    print("\nSummary:")
    for r in ablation_report["results"]:
        status = "OK" if r["exit_code"] == 0 else f"FAIL ({r['exit_code']})"
        passed = r.get("synthetic_passed", "?")
        total = r.get("synthetic_total", "?")
        rate = r.get("synthetic_pass_rate", 0.0) or 0.0
        print(
            f"  {r['config']:<20} {status:<12} "
            f"gen-pass={passed}/{total} ({rate:.1%}) "
            f"({r['wall_time_s']:.1f}s)",
        )


if __name__ == "__main__":
    main()
