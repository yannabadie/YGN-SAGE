#!/usr/bin/env python3
"""TopologyBench-Reasoning: benchmark topologies on math reasoning tasks.

Proves topology significance on reasoning tasks (GSM8K) where multi-agent
debate/verification should outperform sequential, unlike code generation
where the spread was only 4.3pp (not significant).

Key insight (Du et al. ICML 2024, Wang et al. ICLR 2023):
- Debate gives +4-15pp on math reasoning
- Self-consistency gives +17.9pp on GSM8K
- Code generation is fundamentally single-agent (write a function)
- Math reasoning benefits from verification, diverse paths, and aggregation

Usage::

    # Quick smoke test (10 tasks, 3 topologies)
    python scripts/run_topologybench_reasoning.py --limit 10 --topologies sequential,debate,brainstorming

    # Full benchmark (200 tasks x 6 topologies)
    python scripts/run_topologybench_reasoning.py --limit 200

    # Dry run (estimate cost only)
    python scripts/run_topologybench_reasoning.py --limit 200 --dry-run

    # Resume from partial results
    python scripts/run_topologybench_reasoning.py --limit 200 --resume data/topobench_reasoning.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- SSL bypass for corporate proxy ---
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# Prevent HuggingFace 401 errors (snowflake-arctic-embed-m download fails behind proxy)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_src = Path(__file__).resolve().parent.parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("topobench-reasoning")

# ---------------------------------------------------------------------------
# Topologies to test — focus on those expected to differ on reasoning
# ---------------------------------------------------------------------------

TOPOLOGIES = [
    "sequential",     # Baseline single-path
    "parallel",       # Parallel workers + aggregator (self-consistency analog)
    "debate",         # Adversarial verification (Du et al.)
    "brainstorming",  # Diverse reasoning paths + synthesis
    "selfmoa",        # Self-Mixture-of-Agents
    "avr",            # Act-Verify-Refine
]

_COST_PER_CALL_USD = 0.003
_CALLS_PER_TASK = {
    "sequential": 3, "parallel": 4, "avr": 5,
    "selfmoa": 5, "debate": 5, "brainstorming": 5,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id: str
    passed: bool
    predicted: str = ""
    expected: str = ""
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class TopologyReport:
    name: str
    task_count: int = 0
    passed: int = 0
    pass_rate: float = 0.0
    avg_latency_ms: float = 0.0
    errors: int = 0
    results: list[TaskResult] = field(default_factory=list)

    def compute(self) -> None:
        self.task_count = len(self.results)
        if self.task_count == 0:
            return
        self.passed = sum(1 for r in self.results if r.passed)
        self.errors = sum(1 for r in self.results if r.error)
        self.pass_rate = self.passed / self.task_count
        self.avg_latency_ms = (
            sum(r.latency_ms for r in self.results) / self.task_count
        )


# ---------------------------------------------------------------------------
# GSM8K dataset loading
# ---------------------------------------------------------------------------

def load_gsm8k(limit: int | None = None) -> list[dict[str, Any]]:
    """Load GSM8K test set from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        tasks = []
        for i, item in enumerate(ds):
            if limit is not None and i >= limit:
                break
            answer_str = item["answer"].split("####")[-1].strip()
            # Normalize: remove commas, convert to number
            answer_str = answer_str.replace(",", "").strip()
            tasks.append({
                "task_id": f"GSM8K/{i}",
                "question": item["question"],
                "answer": answer_str,
                "full_solution": item["answer"],
            })
        log.info("Loaded %d GSM8K tasks", len(tasks))
        return tasks
    except ImportError:
        raise RuntimeError(
            "datasets package required: pip install datasets"
        )


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_NUMBER_PATTERN = re.compile(
    r"(?:####?\s*)?(-?\d[\d,]*\.?\d*)\s*$", re.MULTILINE
)
_BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}")
_FINAL_ANSWER_PATTERN = re.compile(
    r"(?:final\s+answer|answer\s+is|the\s+answer)\s*(?:is|:)?\s*"
    r"[:\s]*\$?(-?\d[\d,]*\.?\d*)",
    re.IGNORECASE,
)


def extract_number(response: str) -> str | None:
    """Extract the final numerical answer from an LLM response.

    Tries multiple patterns:
    1. \\boxed{N} (LaTeX)
    2. "#### N" (GSM8K format)
    3. "final answer is N" / "the answer is N"
    4. Last number in the response
    """
    if not response:
        return None

    # 1. \boxed{N}
    m = _BOXED_PATTERN.search(response)
    if m:
        val = m.group(1).replace(",", "").strip()
        if _is_number(val):
            return val

    # 2. #### N
    if "####" in response:
        parts = response.split("####")
        val = parts[-1].strip().replace(",", "").strip()
        if _is_number(val):
            return val

    # 3. "final answer is N" / "the answer is N"
    m = _FINAL_ANSWER_PATTERN.search(response)
    if m:
        val = m.group(1).replace(",", "").strip()
        if _is_number(val):
            return val

    # 4. Last number in the response
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", response)
    if numbers:
        val = numbers[-1].replace(",", "").strip()
        if _is_number(val):
            return val

    return None


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def answers_match(predicted: str | None, expected: str) -> bool:
    """Check if predicted answer matches expected (numerical comparison)."""
    if predicted is None:
        return False
    try:
        p = float(predicted.replace(",", ""))
        e = float(expected.replace(",", ""))
        # Exact match for integers, 1e-4 tolerance for floats
        if p == e:
            return True
        if abs(e) > 0:
            return abs(p - e) / abs(e) < 1e-4
        return abs(p - e) < 1e-6
    except (ValueError, TypeError):
        return predicted.strip() == expected.strip()


# ---------------------------------------------------------------------------
# Statistical analysis (with scipy for Wilcoxon)
# ---------------------------------------------------------------------------

def run_statistical_analysis(
    reports: dict[str, TopologyReport],
) -> dict[str, Any]:
    """Comprehensive statistical analysis: bootstrap CI, McNemar, Wilcoxon."""
    import numpy as np
    stats: dict[str, Any] = {}
    topo_names = sorted(reports.keys())

    # Bootstrap CIs per topology
    cis: dict[str, dict[str, float]] = {}
    for name in topo_names:
        report = reports[name]
        passed = [1 if r.passed else 0 for r in report.results]
        if not passed:
            cis[name] = {"lower": 0.0, "upper": 0.0, "pass_rate": 0.0}
            continue
        rng = np.random.RandomState(42)
        boots = [
            np.mean(rng.choice(passed, size=len(passed), replace=True))
            for _ in range(10000)
        ]
        boots.sort()
        cis[name] = {
            "lower": round(float(np.percentile(boots, 2.5)), 4),
            "upper": round(float(np.percentile(boots, 97.5)), 4),
            "pass_rate": round(report.pass_rate, 4),
        }
    stats["bootstrap_ci_95"] = cis

    # Pairwise tests
    pairwise: dict[str, Any] = {}
    for i, name_a in enumerate(topo_names):
        for name_b in topo_names[i + 1:]:
            ra = reports[name_a]
            rb = reports[name_b]
            a_by_id = {r.task_id: r.passed for r in ra.results}
            b_by_id = {r.task_id: r.passed for r in rb.results}
            common = sorted(set(a_by_id) & set(b_by_id))
            if not common:
                continue

            a_vec = np.array([int(a_by_id[t]) for t in common])
            b_vec = np.array([int(b_by_id[t]) for t in common])

            # McNemar's test
            b_ct = int(np.sum((a_vec == 1) & (b_vec == 0)))
            c_ct = int(np.sum((a_vec == 0) & (b_vec == 1)))
            if b_ct + c_ct > 0:
                chi2 = (abs(b_ct - c_ct) - 1) ** 2 / (b_ct + c_ct)
                p_mcnemar = float(math.erfc(math.sqrt(chi2 / 2)))
            else:
                chi2 = 0.0
                p_mcnemar = 1.0

            # Cohen's d
            pooled_p = (ra.pass_rate + rb.pass_rate) / 2
            sd = math.sqrt(pooled_p * (1 - pooled_p)) if 0 < pooled_p < 1 else 0.001
            d = (ra.pass_rate - rb.pass_rate) / sd

            # Wilcoxon signed-rank on per-task binary outcomes
            diff = a_vec - b_vec
            if np.any(diff != 0):
                from scipy.stats import wilcoxon
                try:
                    w_stat, p_wilcoxon = wilcoxon(
                        a_vec, b_vec, alternative="two-sided"
                    )
                    p_wilcoxon = float(p_wilcoxon)
                    w_stat = float(w_stat)
                except Exception:
                    w_stat, p_wilcoxon = 0.0, 1.0
            else:
                w_stat, p_wilcoxon = 0.0, 1.0

            key = f"{name_a}_vs_{name_b}"
            pairwise[key] = {
                "mcnemar_chi2": round(chi2, 4),
                "mcnemar_p": round(p_mcnemar, 6),
                "mcnemar_significant": p_mcnemar < 0.05,
                "wilcoxon_stat": round(w_stat, 4),
                "wilcoxon_p": round(p_wilcoxon, 6),
                "wilcoxon_significant": p_wilcoxon < 0.05,
                "cohens_d": round(d, 4),
                "effect_size": (
                    "large" if abs(d) >= 0.8 else
                    "medium" if abs(d) >= 0.5 else
                    "small" if abs(d) >= 0.2 else
                    "negligible"
                ),
                "delta_pp": round((ra.pass_rate - rb.pass_rate) * 100, 2),
                "b": b_ct,
                "c": c_ct,
            }
    stats["pairwise"] = pairwise

    # Summary: how many significant pairs?
    sig_pairs = [
        k for k, v in pairwise.items()
        if v.get("wilcoxon_significant") or v.get("mcnemar_significant")
    ]
    stats["significant_pairs_count"] = len(sig_pairs)
    stats["total_pairs"] = len(pairwise)
    stats["significant_pairs"] = sig_pairs

    return stats


# ---------------------------------------------------------------------------
# Topology graph builder (reuse from run_topologybench.py)
# ---------------------------------------------------------------------------

def _get_topology_graph(topo_name: str) -> Any | None:
    """Build topology graph for the given name."""
    model_id = "default"
    try:
        from sage_core import PyTemplateStore
        store = PyTemplateStore()
        return store.create(topo_name, model_id)
    except (ImportError, Exception):
        pass

    # Fallback to Python graph
    try:
        from sage.topology.py_graph import PyTopologyGraph
        g = PyTopologyGraph()
        _build_py_graph(g, topo_name, model_id)
        return g
    except Exception as e:
        log.warning("Cannot build topology %s: %s", topo_name, e)
        return None


def _build_py_graph(g: Any, topo_name: str, model_id: str) -> None:
    """Populate a PyTopologyGraph with the given template."""
    if topo_name == "sequential":
        n0 = g.add_node(role="solver", model_id=model_id, system=2)
        n1 = g.add_node(role="formatter", model_id=model_id, system=1)
        g.add_edge(n0, n1)
    elif topo_name == "parallel":
        src = g.add_node(role="source", model_id=model_id, system=1)
        w0 = g.add_node(role="solver_0", model_id=model_id, system=2)
        w1 = g.add_node(role="solver_1", model_id=model_id, system=2)
        agg = g.add_node(role="aggregator", model_id=model_id, system=2)
        g.add_edge(src, w0); g.add_edge(src, w1)
        g.add_edge(w0, agg); g.add_edge(w1, agg)
    elif topo_name == "debate":
        topic = g.add_node(role="topic_setter", model_id=model_id, system=1)
        da = g.add_node(role="debater_a", model_id=model_id, system=2)
        db = g.add_node(role="debater_b", model_id=model_id, system=2)
        judge = g.add_node(role="judge", model_id=model_id, system=2)
        g.add_edge(topic, da); g.add_edge(topic, db)
        g.add_edge(da, judge); g.add_edge(db, judge)
    elif topo_name == "brainstorming":
        p = g.add_node(role="prompt", model_id=model_id, system=1)
        t0 = g.add_node(role="thinker_0", model_id=model_id, system=2)
        t1 = g.add_node(role="thinker_1", model_id=model_id, system=2)
        t2 = g.add_node(role="thinker_2", model_id=model_id, system=2)
        s = g.add_node(role="synthesizer", model_id=model_id, system=2)
        g.add_edge(p, t0); g.add_edge(p, t1); g.add_edge(p, t2)
        g.add_edge(t0, s); g.add_edge(t1, s); g.add_edge(t2, s)
    elif topo_name == "selfmoa":
        d = g.add_node(role="dispatcher", model_id=model_id, system=1)
        a0 = g.add_node(role="agent_0", model_id=model_id, system=2)
        a1 = g.add_node(role="agent_1", model_id=model_id, system=2)
        a2 = g.add_node(role="agent_2", model_id=model_id, system=2)
        m = g.add_node(role="mixer", model_id=model_id, system=2)
        g.add_edge(d, a0); g.add_edge(d, a1); g.add_edge(d, a2)
        g.add_edge(a0, m); g.add_edge(a1, m); g.add_edge(a2, m)
    elif topo_name == "avr":
        actor = g.add_node(role="actor", model_id=model_id, system=2)
        verifier = g.add_node(role="verifier", model_id=model_id, system=2)
        output = g.add_node(role="output", model_id=model_id, system=1)
        g.add_edge(actor, verifier)
        g.add_edge(verifier, output)
        g.add_edge(verifier, actor)


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

async def run_topology_on_tasks(
    llm_provider: Any,
    llm_config: Any,
    topo_name: str,
    tasks: list[dict[str, Any]],
    topology_graph: Any | None,
) -> TopologyReport:
    """Run all GSM8K tasks through TopologyRunner directly.

    Bypasses the full AgentSystem pipeline (routing, memory, guardrails)
    to isolate the pure topology effect on task accuracy.
    """
    from sage.topology.runner import TopologyRunner
    from sage_core import TopologyExecutor as PyTopologyExecutor

    report = TopologyReport(name=topo_name)

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        question = task["question"]
        expected = task["answer"]

        prompt = (
            "Solve this math problem step by step. "
            "At the end, write your final numerical answer after ####.\n\n"
            f"{question}"
        )

        t0 = time.perf_counter()
        error = ""
        predicted = None

        try:
            executor = PyTopologyExecutor(topology_graph)
            runner = TopologyRunner(
                graph=topology_graph,
                executor=executor,
                llm_provider=llm_provider,
                llm_config=llm_config,
            )
            response = await asyncio.wait_for(
                runner.run(prompt), timeout=60.0
            )
            predicted = extract_number(response)
            if predicted is None:
                log.warning(
                    "[%s] extract_number returned None. Preview: %s",
                    task_id, repr(response[:200]) if response else "(empty)",
                )
        except asyncio.TimeoutError:
            error = "TIMEOUT"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            log.error("[%s][%d/%d] %s: %s", topo_name, i+1, len(tasks), task_id, error)

        latency = (time.perf_counter() - t0) * 1000
        passed = answers_match(predicted, expected)

        result = TaskResult(
            task_id=task_id,
            passed=passed,
            predicted=predicted or "",
            expected=expected,
            latency_ms=latency,
            error=error,
        )
        report.results.append(result)

        status = "PASS" if passed else "FAIL"
        pred_str = predicted or "NONE"
        print(
            f"  [{topo_name}][{i+1}/{len(tasks)}] {task_id}: "
            f"{status} (pred={pred_str}, expected={expected}, {latency:.0f}ms)"
        )

    report.compute()
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="TopologyBench-Reasoning (GSM8K)")
    parser.add_argument("--limit", type=int, default=200, help="Number of tasks (default 200)")
    parser.add_argument("--topologies", type=str, default=None, help="Comma-separated topology list")
    parser.add_argument("--output", type=str, default="data/topobench_reasoning.json")
    parser.add_argument("--resume", type=str, default=None, help="Resume from partial results")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost only")
    parser.add_argument("--model", type=str, default=None, help="Override model ID (e.g. gemini-2.5-flash-lite)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    topos = [t.strip() for t in args.topologies.split(",")] if args.topologies else TOPOLOGIES

    if args.dry_run:
        n_tasks = args.limit
        total_calls = sum(_CALLS_PER_TASK.get(t, 4) * n_tasks for t in topos)
        total_cost = total_calls * _COST_PER_CALL_USD
        print(f"Dry run: {n_tasks} tasks x {len(topos)} topologies")
        print(f"Estimated LLM calls: {total_calls}")
        print(f"Estimated cost: ${total_cost:.2f}")
        return

    # Load tasks
    tasks = load_gsm8k(args.limit)

    # Load partial results if resuming
    completed_topos: set[str] = set()
    existing_reports: dict[str, TopologyReport] = {}
    if args.resume and Path(args.resume).exists():
        with open(args.resume) as f:
            data = json.load(f)
        for tname, tdata in data.get("topologies", {}).items():
            if tname in topos:
                report = TopologyReport(name=tname)
                for rd in tdata.get("results", []):
                    report.results.append(TaskResult(**rd))
                report.compute()
                existing_reports[tname] = report
                completed_topos.add(tname)
                log.info("Resumed %s: %.1f%% (%d/%d)", tname,
                         report.pass_rate * 100, report.passed, report.task_count)

    remaining = [t for t in topos if t not in completed_topos]
    if not remaining:
        log.info("All topologies already completed!")
    else:
        log.info("Topologies to run: %s", remaining)

    # Create LLM provider directly (no full system boot needed).
    # TopologyRunner calls the LLM directly per-node — no orchestrator,
    # no routing, no memory overhead. This isolates the pure topology effect.
    llm_provider = None
    llm_config = None
    if remaining:
        from sage.llm.google import GoogleProvider
        from sage.llm.base import LLMConfig
        model_id = args.model or os.environ.get("SAGE_MODEL_FAST", "gemini-2.5-flash")
        os.environ["_TOPOBENCH_MODEL_ID"] = model_id
        llm_provider = GoogleProvider()
        llm_config = LLMConfig(provider="google", model=model_id, temperature=0.3)
        log.info("LLM provider: Google %s (direct, no agent overhead)", model_id)

    all_reports = dict(existing_reports)

    for idx, topo_name in enumerate(remaining):
        log.info("[%d/%d] Running topology: %s", idx + 1, len(remaining), topo_name)
        topo_graph = _get_topology_graph(topo_name)
        if topo_graph is None:
            log.warning("Topology %s unavailable, skipping", topo_name)
            continue

        report = await run_topology_on_tasks(llm_provider, llm_config, topo_name, tasks, topo_graph)
        all_reports[topo_name] = report

        log.info(
            "%s: %.1f%% (%d/%d), %d errors, avg %.1fs",
            topo_name, report.pass_rate * 100, report.passed,
            report.task_count, report.errors, report.avg_latency_ms / 1000,
        )

        # Save after each topology
        _save_results(args.output, all_reports, tasks, partial=True)

    # Final save with statistics
    _save_results(args.output, all_reports, tasks, partial=False)

    # Print summary
    print("\n" + "=" * 80)
    print("TopologyBench-Reasoning (GSM8K) — Final Results")
    print("=" * 80)
    for name in sorted(all_reports, key=lambda n: all_reports[n].pass_rate, reverse=True):
        r = all_reports[name]
        print(f"  {name:<15}: {r.pass_rate*100:5.1f}% ({r.passed}/{r.task_count}), "
              f"{r.errors} errors, avg {r.avg_latency_ms/1000:.1f}s")

    rates = [r.pass_rate for r in all_reports.values()]
    if rates:
        spread = (max(rates) - min(rates)) * 100
        mean = sum(rates) / len(rates) * 100
        print(f"\n  Mean: {mean:.1f}%, Spread: {spread:.1f}pp")

        if spread > 4.3:
            print(f"\n  >>> SPREAD {spread:.1f}pp > 4.3pp (HumanEval code). "
                  "Topology matters MORE for reasoning!")
        else:
            print(f"\n  >>> SPREAD {spread:.1f}pp <= 4.3pp. "
                  "Topology effect comparable to code generation.")

    # Statistical significance
    if len(all_reports) >= 2:
        stats = run_statistical_analysis(all_reports)
        n_sig = stats["significant_pairs_count"]
        n_total = stats["total_pairs"]
        print(f"\n  Significant pairs: {n_sig}/{n_total}")
        for pair in stats.get("significant_pairs", []):
            info = stats["pairwise"][pair]
            print(f"    {pair}: Wilcoxon p={info['wilcoxon_p']:.4f}, "
                  f"delta={info['delta_pp']:+.1f}pp, "
                  f"Cohen's d={info['cohens_d']:.3f} ({info['effect_size']})")

        if n_sig == 0:
            print("\n  NOTE: No statistically significant differences found.")
            print("  Consider: more tasks (--limit 500) or different topologies.")


def _save_results(
    output_path: str,
    reports: dict[str, TopologyReport],
    tasks: list[dict],
    partial: bool,
) -> None:
    """Save results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "benchmark": "TopologyBench-Reasoning",
        "dataset": "GSM8K",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_count": len(tasks),
        "topology_count": len(reports),
        "partial": partial,
        "model_id": os.environ.get("_TOPOBENCH_MODEL_ID", "unknown"),
        "topologies": {},
    }

    for name, report in sorted(reports.items()):
        data["topologies"][name] = {
            "pass_rate": round(report.pass_rate, 4),
            "passed": report.passed,
            "total": report.task_count,
            "errors": report.errors,
            "avg_latency_ms": round(report.avg_latency_ms, 1),
            "results": [asdict(r) for r in report.results],
        }

    # Add statistics if not partial
    if not partial and len(reports) >= 2:
        try:
            data["statistical_tests"] = run_statistical_analysis(reports)
        except Exception as e:
            log.error("Statistical analysis failed: %s", e)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Results saved to %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
