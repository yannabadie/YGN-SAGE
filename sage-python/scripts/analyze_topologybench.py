#!/usr/bin/env python3
"""Analyze TopologyBench results: statistical tests + failure pattern analysis.

Usage:
    python scripts/analyze_topologybench.py data/topologybench_164_real.json
    python scripts/analyze_topologybench.py data/topologybench_20_real.json
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def analyze(data: dict) -> None:
    topos = data.get("topologies", {})
    n_topos = len(topos)
    partial = data.get("partial", False)

    print(f"{'='*60}")
    print(f"TopologyBench Analysis — {n_topos} topologies, partial={partial}")
    print(f"{'='*60}\n")

    # --- Pass rates ---
    print("=== PASS RATES ===")
    for t, r in sorted(topos.items(), key=lambda x: -x[1].get("pass_rate", 0)):
        pr = r.get("pass_rate", 0) * 100
        p = r.get("passed", 0)
        tot = r.get("total", 0)
        print(f"  {t:20s}: {pr:5.1f}% ({p}/{tot})")

    rates = [r["pass_rate"] * 100 for r in topos.values()]
    print(f"\n  Mean: {np.mean(rates):.1f}%")
    print(f"  Spread: {max(rates) - min(rates):.1f}pp ({min(rates):.1f}% - {max(rates):.1f}%)")

    # --- Failure analysis ---
    print("\n=== FAILURE ANALYSIS ===")
    failures: dict[str, set[str]] = {}
    all_tasks: set[str] = set()

    for topo, result in topos.items():
        tasks = result.get("tasks", result.get("results", []))
        if isinstance(tasks, dict):
            tasks = list(tasks.values())
        fails = set()
        for t in tasks:
            if isinstance(t, dict):
                task_id = t.get("task_id", t.get("id", "?"))
                passed = t.get("passed", t.get("pass", None))
                all_tasks.add(task_id)
                if passed is False or passed == 0:
                    fails.add(task_id)
        failures[topo] = fails

    all_fails = set()
    for f in failures.values():
        all_fails.update(f)

    print(f"Total unique failing tasks: {len(all_fails)}")
    for t in sorted(all_fails):
        fails_in = [topo for topo, f in failures.items() if t in f]
        print(f"  {t}: fails in {len(fails_in)}/{n_topos} topologies -> {fails_in}")

    # --- Overlap matrix ---
    print("\n=== OVERLAP MATRIX (Jaccard similarity) ===")
    topo_names = sorted(topos.keys())
    total_overlap = 0
    total_pairs = 0
    for i, t1 in enumerate(topo_names):
        for j, t2 in enumerate(topo_names):
            if i < j:
                overlap = failures[t1] & failures[t2]
                union = failures[t1] | failures[t2]
                jaccard = len(overlap) / len(union) if union else 0.0
                total_overlap += len(overlap)
                total_pairs += 1
                if jaccard > 0:
                    print(f"  {t1} vs {t2}: {len(overlap)} shared, Jaccard={jaccard:.3f}")

    if total_overlap == 0:
        print("  ALL Jaccard = 0.00 (perfectly disjoint failures)")

    # --- Pairwise McNemar ---
    print("\n=== PAIRWISE McNEMAR TESTS ===")
    from scipy.stats import chi2 as chi2_dist

    sig_count = 0
    for t1, t2 in combinations(topo_names, 2):
        r1 = topos[t1]
        r2 = topos[t2]
        tasks1 = {
            t.get("task_id", t.get("id")): t.get("passed", t.get("pass", False))
            for t in (r1.get("tasks", r1.get("results", [])))
        }
        tasks2 = {
            t.get("task_id", t.get("id")): t.get("passed", t.get("pass", False))
            for t in (r2.get("tasks", r2.get("results", [])))
        }

        b = sum(1 for tid in tasks1 if tasks1.get(tid) and not tasks2.get(tid, True))
        c = sum(1 for tid in tasks1 if not tasks1.get(tid) and tasks2.get(tid, True))

        if b + c > 0:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            p = 1.0 - chi2_dist.cdf(chi2, df=1)
        else:
            chi2 = 0
            p = 1.0

        rate1 = r1["pass_rate"] * 100
        rate2 = r2["pass_rate"] * 100
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "ns"
        if p < 0.05:
            sig_count += 1
        print(f"  {t1}({rate1:.0f}%) vs {t2}({rate2:.0f}%): chi2={chi2:.2f}, p={p:.4f} [{sig}]  (b={b}, c={c})")

    n_pairs = len(list(combinations(topo_names, 2)))
    print(f"\n  Significant pairs: {sig_count}/{n_pairs}")

    # --- Cohen's d (best vs worst) ---
    best_topo = max(topos.items(), key=lambda x: x[1]["pass_rate"])
    worst_topo = min(topos.items(), key=lambda x: x[1]["pass_rate"])
    if best_topo[0] != worst_topo[0]:
        best_tasks = {
            t.get("task_id"): 1 if t.get("passed") else 0
            for t in best_topo[1].get("tasks", [])
        }
        worst_tasks = {
            t.get("task_id"): 1 if t.get("passed") else 0
            for t in worst_topo[1].get("tasks", [])
        }
        common = set(best_tasks.keys()) & set(worst_tasks.keys())
        if common:
            diffs = [best_tasks[k] - worst_tasks[k] for k in common]
            mean_d = np.mean(diffs)
            std_d = np.std(diffs, ddof=1)
            cohens_d = mean_d / std_d if std_d > 0 else (float("inf") if mean_d > 0 else 0.0)
            label = (
                "negligible" if abs(cohens_d) < 0.2
                else "small" if abs(cohens_d) < 0.5
                else "medium" if abs(cohens_d) < 0.8
                else "large"
            )
            print(f"\n=== EFFECT SIZE ===")
            print(f"  Best ({best_topo[0]}) vs Worst ({worst_topo[0]})")
            print(f"  Cohen's d = {cohens_d:.4f} ({label})")

    # --- Oracle ensemble ---
    oracle_pass = sum(
        1 for tid in all_tasks
        if not all(tid in failures[t] for t in topo_names)
    )
    print(f"\n=== ORACLE ENSEMBLE ===")
    print(f"  Oracle (best topo per task): {oracle_pass}/{len(all_tasks)} = {oracle_pass/len(all_tasks)*100:.1f}%")
    print(f"  Best single: {best_topo[0]} = {best_topo[1]['pass_rate']*100:.1f}%")
    uplift = (oracle_pass / len(all_tasks) - best_topo[1]["pass_rate"]) * 100
    print(f"  Oracle uplift: +{uplift:.1f}pp over best single topology")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_topologybench.py <results.json>")
        sys.exit(1)
    data = load_results(sys.argv[1])
    analyze(data)
