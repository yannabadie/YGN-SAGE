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

    return {
        "config": config.name,
        "description": config.description,
        "env": config.env,
        "exit_code": result.returncode,
        "wall_time_s": round(elapsed, 1),
        "predictions_path": str(pred_path),
        "report_path": str(report_path),
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

    # Quick summary — exit codes.
    print("\nSummary:")
    for r in ablation_report["results"]:
        status = "OK" if r["exit_code"] == 0 else f"FAIL ({r['exit_code']})"
        print(f"  {r['config']:<20} {status:<12} ({r['wall_time_s']:.1f}s)")


if __name__ == "__main__":
    main()
