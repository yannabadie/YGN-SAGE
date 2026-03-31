#!/usr/bin/env python3
"""N2 Evaluator: MASBENCH ablation — bare model vs SAGE+Path6.

Runs MASBenchAblation directly (sage.bench CLI doesn't support masbench yet).
Cost: ~$0.50 per run. Duration: ~15 min.

Usage:
    SAGE_SSL_VERIFY=false SAGE_ENABLE_PATH6=1 \
    SAGE_PATH6_ADAPTER=models/toolcall_qwen3_4b_phase_c/sft_checkpoint \
    python scripts/eval_masbench_local.py --limit 20
"""
import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
import time

# SSL bypass for corporate proxy (HuggingFace datasets download)
if os.environ.get("SAGE_SSL_VERIFY", "").lower() == "false":
    import urllib3
    urllib3.disable_warnings()
    import requests
    _orig_send = requests.adapters.HTTPAdapter.send
    def _patched_send(self, request, **kwargs):
        kwargs["verify"] = False
        return _orig_send(self, request, **kwargs)
    requests.adapters.HTTPAdapter.send = _patched_send
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["CURL_CA_BUNDLE"] = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
log = logging.getLogger("eval_n2")


async def run_masbench(limit: int, axis: str, output: str):
    """Run MASBENCH ablation: bare model vs SAGE full engine."""
    from sage.boot import boot_agent_system
    from sage.bench.masbench import MASBenchAblation

    log.info("Booting SAGE system...")
    system = boot_agent_system()

    log.info("Running MASBENCH ablation (axis=%s, limit=%d)...", axis, limit)
    bench = MASBenchAblation(system=system, axis=axis)
    t0 = time.perf_counter()
    reports = await bench.run(limit=limit)
    elapsed = time.perf_counter() - t0

    # Save results
    results = {}
    for name, report in reports.items():
        results[name] = {
            "pass_rate": report.pass_rate,
            "passed": report.passed,
            "total": report.total,
            "avg_latency_ms": report.avg_latency_ms,
        }

    # Add delta
    if "bare" in results and "sage_full" in results:
        delta = results["sage_full"]["pass_rate"] - results["bare"]["pass_rate"]
        results["topology_delta_pp"] = round(delta * 100, 1)

    results["axis"] = axis
    results["limit"] = limit
    results["elapsed_sec"] = round(elapsed, 1)
    results["path6_adapter"] = os.environ.get("SAGE_PATH6_ADAPTER", "none")

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    log.info("Saved to %s", output)
    log.info("=== Results ===")
    for name in ["bare", "sage_full"]:
        if name in results:
            r = results[name]
            log.info("  %s: %.1f%% (%d/%d)", name, r["pass_rate"] * 100, r["passed"], r["total"])
    if "topology_delta_pp" in results:
        log.info("  TOPOLOGY DELTA: %+.1fpp", results["topology_delta_pp"])

    return results


def main():
    parser = argparse.ArgumentParser(description="N2: MASBENCH ablation evaluation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--axis", default="depth", choices=["breadth", "depth", "horizon", "parallel", "robustness"])
    parser.add_argument("--output", default="experiments/n2_masbench_phase_c.json")
    args = parser.parse_args()

    asyncio.run(run_masbench(args.limit, args.axis, args.output))


if __name__ == "__main__":
    main()
