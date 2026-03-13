#!/usr/bin/env python3
"""Merge multiple TopologyBench result files into a single comprehensive file.

Usage:
    python scripts/merge_topologybench.py \
        --files data/topologybench_164_real.json \
               data/topologybench_164_brainstorming.json \
               data/topologybench_164_parallel.json \
        --output data/topologybench_164_all.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path


def load_topologies(path: str) -> dict:
    """Load topologies from a results file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("topologies", {})


def mcnemar_test(results_a: list[dict], results_b: list[dict]) -> dict:
    """McNemar's test on paired task results."""
    import math

    a_by_id = {r["task_id"]: r["passed"] for r in results_a}
    b_by_id = {r["task_id"]: r["passed"] for r in results_b}
    common = sorted(set(a_by_id) & set(b_by_id))

    if not common:
        return {"chi2": 0, "p_value": 1.0, "b": 0, "c": 0, "n": 0}

    b = sum(1 for tid in common if a_by_id[tid] and not b_by_id[tid])
    c = sum(1 for tid in common if not a_by_id[tid] and b_by_id[tid])

    if b + c == 0:
        return {"chi2": 0, "p_value": 1.0, "b": b, "c": c, "n": len(common)}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2))  # chi2(1) survival function

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p, 6),
        "significant": p < 0.05,
        "b": b,
        "c": c,
        "n": len(common),
    }


def main():
    parser = argparse.ArgumentParser(description="Merge TopologyBench results")
    parser.add_argument("--files", nargs="+", required=True, help="Input JSON files")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    # Merge all topologies
    all_topos = {}
    for path in args.files:
        topos = load_topologies(path)
        for name, data in topos.items():
            if name in all_topos:
                print(f"  Warning: topology '{name}' exists in multiple files, using latest")
            all_topos[name] = data
            passed = data.get("passed", 0)
            total = data.get("total", 0)
            rate = data.get("pass_rate", 0) * 100
            print(f"  {name}: {rate:.1f}% ({passed}/{total})")

    # Run pairwise McNemar tests
    topo_names = sorted(all_topos.keys())
    pairwise = {}
    for t1, t2 in combinations(topo_names, 2):
        r1 = all_topos[t1].get("results", [])
        r2 = all_topos[t2].get("results", [])
        if r1 and r2:
            test = mcnemar_test(r1, r2)
            key = f"{t1}_vs_{t2}"
            pairwise[key] = test

    # Failure analysis
    failures = {}
    all_tasks = set()
    for name, data in all_topos.items():
        results = data.get("results", [])
        fails = set()
        for r in results:
            all_tasks.add(r["task_id"])
            if not r.get("passed", True):
                fails.add(r["task_id"])
        failures[name] = fails

    # Jaccard matrix
    jaccard = {}
    for t1, t2 in combinations(topo_names, 2):
        overlap = failures[t1] & failures[t2]
        union = failures[t1] | failures[t2]
        j = len(overlap) / len(union) if union else 0.0
        jaccard[f"{t1}_vs_{t2}"] = round(j, 4)

    # Oracle ensemble
    oracle_pass = sum(
        1 for tid in all_tasks
        if not all(tid in failures[t] for t in topo_names)
    )

    # Build output
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_count": max(d.get("total", 0) for d in all_topos.values()),
        "topology_count": len(all_topos),
        "partial": False,
        "topologies": all_topos,
        "statistical_tests": {
            "pairwise_mcnemar": pairwise,
            "jaccard_similarity": jaccard,
            "oracle_ensemble": {
                "pass_count": oracle_pass,
                "total": len(all_tasks),
                "pass_rate": round(oracle_pass / len(all_tasks), 4) if all_tasks else 0,
            },
        },
    }

    # Summary
    print(f"\n{'='*60}")
    print(f"Merged {len(all_topos)} topologies, {len(all_tasks)} tasks")
    print(f"{'='*60}")

    for name in sorted(all_topos, key=lambda n: -all_topos[n].get("pass_rate", 0)):
        d = all_topos[name]
        print(f"  {name:20s}: {d['pass_rate']*100:5.1f}% ({d['passed']}/{d['total']})")

    print(f"\nOracle ensemble: {oracle_pass}/{len(all_tasks)} = {oracle_pass/len(all_tasks)*100:.1f}%")

    sig = {k: v for k, v in pairwise.items() if v.get("significant")}
    if sig:
        print(f"\nSignificant pairs ({len(sig)}):")
        for k, v in sorted(sig.items(), key=lambda x: x[1]["p_value"]):
            print(f"  {k}: chi2={v['chi2']}, p={v['p_value']:.4f}, b={v['b']}, c={v['c']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
