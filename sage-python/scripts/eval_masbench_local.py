#!/usr/bin/env python3
"""N2 Evaluator: MASBENCH depth with Path 6 local model.

Sets SAGE_ENABLE_PATH6=1 and runs MASBENCH depth benchmark.
Cost: ~$0.50 per run. Duration: ~10 min.

Usage:
    python scripts/eval_masbench_local.py --adapter models/local_qwen3_4b_grpo/sft_checkpoint --limit 20
"""
import argparse
import json
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("eval_n2")


def main():
    parser = argparse.ArgumentParser(description="N2: MASBENCH depth evaluation")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    env = os.environ.copy()
    env["SAGE_ENABLE_PATH6"] = "1"
    env["SAGE_PATH6_ADAPTER"] = os.path.abspath(args.adapter)

    log.info("Running MASBENCH depth (limit=%d) with Path 6 from %s", args.limit, args.adapter)

    result = subprocess.run(
        [sys.executable, "-m", "sage.bench",
         "--type", "masbench", "--axis", "depth", "--limit", str(args.limit),
         "--output-json", args.output or "experiments/n2_latest.json"],
        env=env, capture_output=True, text=True, timeout=1200,
    )

    print(result.stdout)
    if result.returncode != 0:
        log.error("MASBENCH failed: %s", result.stderr[-500:] if result.stderr else "no stderr")
        return None

    if args.output and os.path.exists(args.output):
        with open(args.output) as f:
            metrics = json.load(f)
        log.info("N2 MASBENCH depth: %s", json.dumps(metrics, indent=2)[:500])
        return metrics

    return None


if __name__ == "__main__":
    main()
