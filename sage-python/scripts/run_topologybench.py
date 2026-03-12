#!/usr/bin/env python3
"""TopologyBench: benchmark multi-agent topologies on coding tasks.

Proves the "topology > model" hypothesis (AdaptOrch, arXiv 2602.16873) by
running HumanEval+ tasks across 10 different topologies and measuring
performance differences with statistical rigor.

Usage::

    # Quick smoke test (5 tasks, 2 topologies)
    python scripts/run_topologybench.py --tasks 5 --topologies sequential,avr

    # Full benchmark (200 tasks x 10 topologies)
    python scripts/run_topologybench.py --tasks 200

    # Dry run (estimate cost only)
    python scripts/run_topologybench.py --tasks 200 --dry-run

    # Resume from partial results
    python scripts/run_topologybench.py --tasks 200 --resume data/topologybench_results.json

    # Custom output path
    python scripts/run_topologybench.py --tasks 50 --output data/topo_50.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- SSL bypass for corporate proxy (CLAUDE.md protocol) ---
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Add sage-python/src to path if running from scripts/
_src = Path(__file__).resolve().parent.parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("topologybench")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All 10 topologies: 8 templates + 2 dynamic (evolved, llm_synthesized)
ALL_TOPOLOGIES = [
    "sequential",
    "parallel",
    "avr",
    "selfmoa",
    "hierarchical",
    "hub",
    "debate",
    "brainstorming",
    "evolved",
    "llm_synthesized",
]

# 8 template topologies (always available via PyTemplateStore or PyTopologyGraph)
TEMPLATE_TOPOLOGIES = ALL_TOPOLOGIES[:8]

# Approximate cost per LLM call (USD) — varies by topology agent count
# Used for --dry-run estimation
_COST_PER_CALL_USD = 0.003  # ~gemini-2.5-flash average

# Approximate LLM calls per task per topology
_CALLS_PER_TASK = {
    "sequential": 3,     # input_processor + worker + output_formatter
    "parallel": 4,       # source + 2 workers + aggregator (best-of)
    "avr": 5,            # actor + verifier + possible retry + output
    "selfmoa": 5,        # dispatcher + 3 agents + mixer
    "hierarchical": 4,   # manager + 2 workers + collect
    "hub": 5,            # coordinator + 3 spokes + collect
    "debate": 5,         # topic_setter + 2 debaters + judge + synthesize
    "brainstorming": 5,  # prompt + 3 thinkers + synthesizer
    "evolved": 5,        # ~avg of templates
    "llm_synthesized": 6,  # extra LLM call for synthesis + execution
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TopologyTaskResult:
    """Result of a single task under a specific topology."""
    task_id: str
    passed: bool
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    system_used: int = 0
    error: str = ""
    topology_nodes: int = 0
    topology_edges: int = 0


@dataclass
class TopologyReport:
    """Aggregated results for one topology."""
    name: str
    task_count: int = 0
    passed: int = 0
    pass_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    node_count: int = 0
    edge_count: int = 0
    results: list[TopologyTaskResult] = field(default_factory=list)
    errors: int = 0

    def compute(self) -> None:
        """Recompute aggregates from results list."""
        self.task_count = len(self.results)
        if self.task_count == 0:
            return
        self.passed = sum(1 for r in self.results if r.passed)
        self.errors = sum(1 for r in self.results if r.error)
        self.pass_rate = self.passed / self.task_count
        self.avg_latency_ms = sum(r.latency_ms for r in self.results) / self.task_count
        self.avg_cost_usd = sum(r.cost_usd for r in self.results) / self.task_count
        self.total_cost_usd = sum(r.cost_usd for r in self.results)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def _mcnemar_test(results_a: list[bool], results_b: list[bool]) -> dict[str, Any]:
    """McNemar's test for paired binary outcomes.

    Tests whether topology A and B have significantly different error rates
    on the same set of tasks.
    """
    assert len(results_a) == len(results_b), "Result vectors must be same length"
    n = len(results_a)
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant": False, "n": 0}

    # Contingency: b=pass in A but fail in B, c=fail in A but pass in B
    b = sum(1 for a, bb in zip(results_a, results_b) if a and not bb)
    c = sum(1 for a, bb in zip(results_a, results_b) if not a and bb)

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant": False, "n": n, "b": b, "c": c}

    # McNemar's chi-squared with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # Approximate p-value from chi2(1) distribution using survival function
    # For chi2(1): P(X > x) ~ erfc(sqrt(x/2)) / 2, but we use a simpler approx
    p_value = _chi2_sf(chi2, df=1)

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "n": n,
        "b": b,
        "c": c,
    }


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function for chi-squared distribution (no scipy needed).

    Uses the regularized incomplete gamma function approximation for df=1.
    For df=1: chi2 ~ N(0,1)^2, so P(chi2 > x) = 2 * (1 - Phi(sqrt(x))).
    """
    if x <= 0:
        return 1.0
    if df != 1:
        # Fallback: very rough approximation for df != 1
        return max(0.0, 1.0 - x / (df + 2 * math.sqrt(df)))
    # df=1: use complementary error function
    return math.erfc(math.sqrt(x / 2))


def _bootstrap_ci(
    passed: list[bool], n_boot: int = 10000, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for pass rate.

    Returns (lower, upper) bounds.
    """
    if not passed:
        return (0.0, 0.0)

    n = len(passed)
    rng = random.Random(42)  # Reproducible
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = [rng.choice(passed) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    lo_idx = int(n_boot * (alpha / 2))
    hi_idx = int(n_boot * (1 - alpha / 2))
    return (round(boot_means[lo_idx], 4), round(boot_means[min(hi_idx, n_boot - 1)], 4))


def _cohens_d(rate_a: float, rate_b: float, n: int) -> float:
    """Cohen's d effect size for two proportions.

    d = (p1 - p2) / pooled_sd.  Small: 0.2, medium: 0.5, large: 0.8.
    """
    if n == 0:
        return 0.0
    pooled_p = (rate_a + rate_b) / 2
    sd = math.sqrt(pooled_p * (1 - pooled_p)) if 0 < pooled_p < 1 else 0.001
    return round((rate_a - rate_b) / sd, 4)


def run_statistical_analysis(
    reports: dict[str, TopologyReport],
) -> dict[str, Any]:
    """Run pairwise McNemar tests, bootstrap CIs, and Cohen's d."""
    stats: dict[str, Any] = {}
    topo_names = sorted(reports.keys())

    # Bootstrap CIs per topology
    cis: dict[str, dict[str, float]] = {}
    for name in topo_names:
        report = reports[name]
        passed_list = [r.passed for r in report.results]
        lo, hi = _bootstrap_ci(passed_list)
        cis[name] = {"lower": lo, "upper": hi, "pass_rate": report.pass_rate}
    stats["bootstrap_ci_95"] = cis

    # Pairwise McNemar + Cohen's d
    pairwise: dict[str, Any] = {}
    for i, name_a in enumerate(topo_names):
        for name_b in topo_names[i + 1:]:
            report_a = reports[name_a]
            report_b = reports[name_b]

            # Align results by task_id
            a_by_id = {r.task_id: r.passed for r in report_a.results}
            b_by_id = {r.task_id: r.passed for r in report_b.results}
            common_ids = sorted(set(a_by_id.keys()) & set(b_by_id.keys()))

            if not common_ids:
                continue

            a_vec = [a_by_id[tid] for tid in common_ids]
            b_vec = [b_by_id[tid] for tid in common_ids]

            mcnemar = _mcnemar_test(a_vec, b_vec)
            d = _cohens_d(report_a.pass_rate, report_b.pass_rate, len(common_ids))

            key = f"mcnemar_{name_a}_vs_{name_b}"
            pairwise[key] = {
                **mcnemar,
                "cohens_d": d,
                "effect_size": (
                    "large" if abs(d) >= 0.8 else
                    "medium" if abs(d) >= 0.5 else
                    "small" if abs(d) >= 0.2 else
                    "negligible"
                ),
            }
    stats["pairwise"] = pairwise

    return stats


# ---------------------------------------------------------------------------
# Topology execution strategies
# ---------------------------------------------------------------------------

def _get_topology_graph(
    topo_name: str,
    topology_engine: Any = None,
) -> Any | None:
    """Get a TopologyGraph for the given topology name.

    Returns a sage_core.TopologyGraph (preferred) or PyTopologyGraph (fallback),
    or None if the topology is not available.
    """
    model_id = "default"  # Placeholder; actual model is determined by the agent system

    # Try Rust PyTemplateStore first
    if topo_name in TEMPLATE_TOPOLOGIES:
        try:
            from sage_core import PyTemplateStore
            store = PyTemplateStore()
            return store.create(topo_name, model_id)
        except ImportError:
            pass

        # Fallback: build PyTopologyGraph manually
        from sage.topology.py_graph import PyTopologyGraph
        return _build_py_graph(topo_name, model_id)

    if topo_name == "evolved":
        if topology_engine:
            try:
                # Try to get best from MAP-Elites archive
                result = topology_engine.generate("benchmark task", None, 2, 0.0)
                if result and result.source != "template_fallback":
                    return result.topology
            except Exception as e:
                log.debug("Evolved topology unavailable: %s", e)
        return None

    if topo_name == "llm_synthesized":
        # LLM synthesis requires async + live LLM; handled at run time
        return None

    return None


def _build_py_graph(topo_name: str, model_id: str) -> Any:
    """Build a PyTopologyGraph for a template topology (fallback when sage_core unavailable)."""
    from sage.topology.py_graph import PyTopologyGraph

    g = PyTopologyGraph()

    if topo_name == "sequential":
        n0 = g.add_node(role="input_processor", model_id=model_id, system=1)
        n1 = g.add_node(role="worker", model_id=model_id, system=2)
        n2 = g.add_node(role="output_formatter", model_id=model_id, system=1)
        g.add_edge(n0, n1)
        g.add_edge(n1, n2)

    elif topo_name == "parallel":
        src = g.add_node(role="source", model_id=model_id, system=1)
        w0 = g.add_node(role="worker_0", model_id=model_id, system=2)
        w1 = g.add_node(role="worker_1", model_id=model_id, system=2)
        agg = g.add_node(role="aggregator", model_id=model_id, system=1)
        g.add_edge(src, w0)
        g.add_edge(src, w1)
        g.add_edge(w0, agg)
        g.add_edge(w1, agg)

    elif topo_name == "avr":
        actor = g.add_node(role="actor", model_id=model_id, system=2)
        verifier = g.add_node(role="verifier", model_id=model_id, system=2)
        output = g.add_node(role="output", model_id=model_id, system=1)
        g.add_edge(actor, verifier)
        g.add_edge(verifier, output)
        g.add_edge(verifier, actor)  # repair back-edge

    elif topo_name == "selfmoa":
        disp = g.add_node(role="dispatcher", model_id=model_id, system=1)
        a0 = g.add_node(role="agent_0", model_id=model_id, system=2)
        a1 = g.add_node(role="agent_1", model_id=model_id, system=2)
        a2 = g.add_node(role="agent_2", model_id=model_id, system=2)
        mixer = g.add_node(role="mixer", model_id=model_id, system=2)
        g.add_edge(disp, a0)
        g.add_edge(disp, a1)
        g.add_edge(disp, a2)
        g.add_edge(a0, mixer)
        g.add_edge(a1, mixer)
        g.add_edge(a2, mixer)

    elif topo_name == "hierarchical":
        parent = g.add_node(role="parent", model_id=model_id, system=2)
        c0 = g.add_node(role="child_0", model_id=model_id, system=1)
        c1 = g.add_node(role="child_1", model_id=model_id, system=1)
        g.add_edge(parent, c0)
        g.add_edge(parent, c1)
        g.add_edge(c0, parent)
        g.add_edge(c1, parent)

    elif topo_name == "hub":
        coord = g.add_node(role="coordinator", model_id=model_id, system=2)
        s0 = g.add_node(role="spoke_0", model_id=model_id, system=1)
        s1 = g.add_node(role="spoke_1", model_id=model_id, system=1)
        s2 = g.add_node(role="spoke_2", model_id=model_id, system=1)
        for s in (s0, s1, s2):
            g.add_edge(coord, s)
            g.add_edge(s, coord)

    elif topo_name == "debate":
        topic = g.add_node(role="topic_setter", model_id=model_id, system=1)
        da = g.add_node(role="debater_a", model_id=model_id, system=2)
        db = g.add_node(role="debater_b", model_id=model_id, system=2)
        judge = g.add_node(role="judge", model_id=model_id, system=2)
        g.add_edge(topic, da)
        g.add_edge(topic, db)
        g.add_edge(da, judge)
        g.add_edge(db, judge)

    elif topo_name == "brainstorming":
        prompt = g.add_node(role="prompt", model_id=model_id, system=1)
        t0 = g.add_node(role="thinker_0", model_id=model_id, system=2)
        t1 = g.add_node(role="thinker_1", model_id=model_id, system=2)
        t2 = g.add_node(role="thinker_2", model_id=model_id, system=2)
        synth = g.add_node(role="synthesizer", model_id=model_id, system=2)
        g.add_edge(prompt, t0)
        g.add_edge(prompt, t1)
        g.add_edge(prompt, t2)
        g.add_edge(t0, synth)
        g.add_edge(t1, synth)
        g.add_edge(t2, synth)

    return g


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks(limit: int | None = None) -> list[dict[str, Any]]:
    """Load HumanEval+ tasks for benchmarking."""
    try:
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
        task_ids = list(problems.keys())
        if limit is not None:
            task_ids = task_ids[:limit]
        return [{"task_id": tid, **problems[tid]} for tid in task_ids]
    except ImportError:
        log.warning("evalplus not installed, falling back to bundled HumanEval")
        from sage.bench.humaneval import load_problems
        return load_problems(limit)


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

async def run_topology_benchmark(
    system: Any,
    topo_name: str,
    tasks: list[dict[str, Any]],
    topology_graph: Any | None = None,
) -> TopologyReport:
    """Run all tasks through the agent system with a specific topology forced.

    The topology is injected into the agent loop before each task so the system
    uses the prescribed multi-agent structure.
    """
    report = TopologyReport(name=topo_name)
    node_count = 0
    edge_count = 0

    if topology_graph is not None:
        node_count = topology_graph.node_count()
        edge_count = topology_graph.edge_count()
        report.node_count = node_count
        report.edge_count = edge_count

    for i, task_data in enumerate(tasks):
        task_id = task_data["task_id"]
        prompt = task_data["prompt"]
        entry_point = task_data["entry_point"]

        # Force the topology onto the agent loop before running
        if topology_graph is not None:
            system.agent_loop._current_topology = topology_graph
            if system.topology_engine:
                try:
                    system.topology_engine.cache_topology(topology_graph)
                except Exception:
                    pass

        task_prompt = (
            "Complete this Python function. "
            "Return ONLY the complete function, no explanation.\n\n"
            f"```python\n{prompt}\n```"
        )

        t0 = time.perf_counter()
        error = ""
        system_used = 0

        try:
            response = await asyncio.wait_for(
                system.run(task_prompt),
                timeout=90.0,
            )
            system_used = (
                getattr(system.agent_loop, "_last_routing_system", 0) or 2
            )

            # Extract and evaluate code
            from sage.bench.humaneval import extract_code, run_test
            completion = extract_code(response, entry_point)

            # Evaluate against tests
            test_code = task_data.get("test", "")
            if test_code:
                passed, test_error = run_test(
                    prompt, completion, test_code, entry_point, timeout=15.0,
                )
                if not passed and not error:
                    error = test_error[:200]
            else:
                # EvalPlus-style evaluation with base + plus inputs
                try:
                    from sage.bench.evalplus_bench import EvalPlusBench
                    bench = EvalPlusBench(dataset="humaneval")
                    eval_result = bench.evaluate_task(completion, task_data)
                    passed = eval_result["base_passed"] and eval_result["plus_passed"]
                    if not passed:
                        error = eval_result.get("error", "evaluation_failed")[:200]
                except Exception as eval_err:
                    # Fallback: if EvalPlus evaluation fails, just check syntax
                    passed = False
                    error = f"eval_error: {eval_err}"[:200]

        except asyncio.TimeoutError:
            passed = False
            error = "timeout_90s"
            log.warning("[%s][%s] Task timed out after 90s", topo_name, task_id)
        except Exception as e:
            passed = False
            error = str(e)[:200]
            log.error("[%s][%s] Task failed: %s", topo_name, task_id, error)

        latency_ms = (time.perf_counter() - t0) * 1000
        cost_usd = getattr(system.agent_loop, "total_cost_usd", 0.0)

        result = TopologyTaskResult(
            task_id=task_id,
            passed=passed,
            latency_ms=round(latency_ms, 1),
            cost_usd=round(cost_usd, 6),
            system_used=system_used,
            error=error,
            topology_nodes=node_count,
            topology_edges=edge_count,
        )
        report.results.append(result)

        status = "PASS" if passed else "FAIL"
        print(
            f"  [{topo_name}][{i+1}/{len(tasks)}] {task_id}: {status} "
            f"({latency_ms:.0f}ms)",
            flush=True,
        )

    report.compute()
    return report


# ---------------------------------------------------------------------------
# LLM-synthesized topology
# ---------------------------------------------------------------------------

async def try_llm_synthesis(system: Any, task_sample: str) -> Any | None:
    """Attempt to synthesize a topology via LLM (Path 3 of DynamicTopologyEngine).

    Returns a TopologyGraph or None if synthesis fails.
    """
    try:
        from sage.topology.llm_caller import synthesize_topology
        graph = await synthesize_topology(
            system.agent_loop._llm,
            task_sample,
            max_agents=4,
            available_models=["gemini-2.5-flash", "gemini-3-flash-preview"],
        )
        if graph and graph.node_count() > 0:
            log.info("LLM synthesis produced %d-node topology", graph.node_count())
            return graph
    except Exception as e:
        log.warning("LLM topology synthesis failed: %s", e)
    return None


# ---------------------------------------------------------------------------
# Cost estimation (dry run)
# ---------------------------------------------------------------------------

def estimate_cost(
    task_count: int,
    topologies: list[str],
) -> dict[str, Any]:
    """Estimate total LLM calls and cost for a benchmark run."""
    total_calls = 0
    per_topo: dict[str, dict[str, Any]] = {}
    for topo in topologies:
        calls = _CALLS_PER_TASK.get(topo, 5) * task_count
        cost = calls * _COST_PER_CALL_USD
        per_topo[topo] = {
            "tasks": task_count,
            "calls_per_task": _CALLS_PER_TASK.get(topo, 5),
            "total_calls": calls,
            "estimated_cost_usd": round(cost, 2),
        }
        total_calls += calls

    total_cost = total_calls * _COST_PER_CALL_USD
    # Estimate time: ~5s per task average (includes LLM latency + evaluation)
    est_time_min = (task_count * len(topologies) * 5) / 60

    return {
        "task_count": task_count,
        "topology_count": len(topologies),
        "total_llm_calls": total_calls,
        "estimated_total_cost_usd": round(total_cost, 2),
        "estimated_time_minutes": round(est_time_min, 1),
        "per_topology": per_topo,
    }


# ---------------------------------------------------------------------------
# Partial results persistence (resumability)
# ---------------------------------------------------------------------------

def save_partial(
    output_path: Path,
    reports: dict[str, TopologyReport],
    completed_topologies: list[str],
    task_count: int,
    start_time: str,
) -> None:
    """Save partial results for resumability."""
    data = _build_output(reports, task_count, start_time, partial=True)
    data["_completed_topologies"] = completed_topologies
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("Partial results saved to %s", output_path)


def load_partial(path: Path) -> tuple[dict[str, TopologyReport], list[str]]:
    """Load partial results from a previous run."""
    if not path.exists():
        return {}, []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    completed = data.get("_completed_topologies", [])
    reports: dict[str, TopologyReport] = {}

    for topo_name, topo_data in data.get("topologies", {}).items():
        results = [
            TopologyTaskResult(**r) for r in topo_data.get("results", [])
        ]
        report = TopologyReport(
            name=topo_name,
            node_count=topo_data.get("node_count", 0),
            edge_count=topo_data.get("edge_count", 0),
            results=results,
        )
        report.compute()
        reports[topo_name] = report

    log.info("Loaded %d completed topologies from %s", len(completed), path)
    return reports, completed


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _build_output(
    reports: dict[str, TopologyReport],
    task_count: int,
    start_time: str,
    partial: bool = False,
) -> dict[str, Any]:
    """Build the output JSON structure."""
    topologies_out: dict[str, Any] = {}
    for name, report in sorted(reports.items()):
        topologies_out[name] = {
            "pass_rate": round(report.pass_rate, 4),
            "passed": report.passed,
            "total": report.task_count,
            "avg_latency_ms": round(report.avg_latency_ms, 1),
            "avg_cost_usd": round(report.avg_cost_usd, 6),
            "total_cost_usd": round(report.total_cost_usd, 4),
            "node_count": report.node_count,
            "edge_count": report.edge_count,
            "errors": report.errors,
            "results": [asdict(r) for r in report.results],
        }

    output: dict[str, Any] = {
        "timestamp": start_time,
        "task_count": task_count,
        "topology_count": len(reports),
        "partial": partial,
        "topologies": topologies_out,
    }

    if not partial and len(reports) >= 2:
        output["statistical_tests"] = run_statistical_analysis(reports)

    return output


def print_summary_table(reports: dict[str, TopologyReport]) -> None:
    """Print a formatted summary table to stdout."""
    if not reports:
        print("No results to display.")
        return

    print("\n" + "=" * 90)
    print("TOPOLOGY BENCHMARK RESULTS")
    print("=" * 90)
    print(
        f"{'Topology':<16} {'Pass Rate':>10} {'Passed':>8} {'Total':>6} "
        f"{'Avg Latency':>12} {'Avg Cost':>10} {'Nodes':>6} {'Edges':>6}"
    )
    print("-" * 90)

    # Sort by pass rate descending
    sorted_reports = sorted(reports.values(), key=lambda r: r.pass_rate, reverse=True)

    for report in sorted_reports:
        print(
            f"{report.name:<16} {report.pass_rate:>9.1%} "
            f"{report.passed:>8} {report.task_count:>6} "
            f"{report.avg_latency_ms:>10.0f}ms "
            f"${report.avg_cost_usd:>8.4f} "
            f"{report.node_count:>6} {report.edge_count:>6}"
        )

    print("-" * 90)

    # Total cost
    total_cost = sum(r.total_cost_usd for r in reports.values())
    print(f"{'Total cost':>76} ${total_cost:>8.2f}")

    # Best vs worst
    if len(sorted_reports) >= 2:
        best = sorted_reports[0]
        worst = sorted_reports[-1]
        delta = best.pass_rate - worst.pass_rate
        print(f"\nBest:  {best.name} ({best.pass_rate:.1%})")
        print(f"Worst: {worst.name} ({worst.pass_rate:.1%})")
        print(f"Delta: {delta:+.1%} ({delta*100:+.1f} percentage points)")

        if delta > 0.05:
            print(
                "\n** FINDING: Topology choice matters. "
                f"{delta:.0%} pass rate difference observed. "
                "Consistent with AdaptOrch (arXiv 2602.16873) hypothesis."
            )
        else:
            print(
                "\nNote: Small topology delta observed. "
                "May need more tasks or different task types to reveal differences."
            )

    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    """Async main entry point."""
    start_time = datetime.now(timezone.utc).isoformat()
    output_path = Path(args.output)

    # Parse topology list
    if args.topologies == "all":
        topologies = list(ALL_TOPOLOGIES)
    else:
        topologies = [t.strip() for t in args.topologies.split(",")]
        unknown = [t for t in topologies if t not in ALL_TOPOLOGIES]
        if unknown:
            log.error("Unknown topologies: %s. Available: %s", unknown, ALL_TOPOLOGIES)
            sys.exit(1)

    task_limit = args.limit if args.limit else args.tasks

    # --- Dry run ---
    if args.dry_run:
        estimate = estimate_cost(task_limit, topologies)
        print("\n=== DRY RUN: Cost Estimate ===")
        print(f"Tasks per topology: {task_limit}")
        print(f"Topologies: {len(topologies)}")
        print(f"Total LLM calls: ~{estimate['total_llm_calls']}")
        print(f"Estimated cost: ~${estimate['estimated_total_cost_usd']:.2f}")
        print(f"Estimated time: ~{estimate['estimated_time_minutes']:.0f} minutes")
        print("\nPer topology:")
        for topo, info in estimate["per_topology"].items():
            print(
                f"  {topo:<16}: ~{info['total_calls']:>5} calls, "
                f"~${info['estimated_cost_usd']:.2f}"
            )
        print("\nTo run for real, remove --dry-run.")
        return

    # --- Load partial results if resuming ---
    reports: dict[str, TopologyReport] = {}
    completed: list[str] = []
    if args.resume:
        resume_path = Path(args.resume)
        reports, completed = load_partial(resume_path)
        log.info("Resuming: skipping %d already-completed topologies", len(completed))

    # --- Boot SAGE system ---
    log.info("Booting SAGE agent system...")
    from sage.boot import boot_agent_system
    system = boot_agent_system(llm_tier="auto")
    log.info("SAGE system booted successfully")

    # --- Load tasks ---
    log.info("Loading HumanEval+ tasks (limit=%s)...", task_limit)
    tasks = load_tasks(limit=task_limit)
    log.info("Loaded %d tasks", len(tasks))

    # --- Run each topology ---
    remaining = [t for t in topologies if t not in completed]
    total_topologies = len(remaining)
    topo_start_time = time.perf_counter()

    for topo_idx, topo_name in enumerate(remaining):
        print(
            f"\n{'='*60}\n"
            f"[{topo_idx+1}/{total_topologies}] Running topology: {topo_name}\n"
            f"{'='*60}",
            flush=True,
        )

        # Get or build topology graph
        topology_graph = None
        if topo_name == "llm_synthesized":
            # Use a sample task to guide LLM synthesis
            sample_prompt = tasks[0]["prompt"] if tasks else "Write a Python function"
            topology_graph = await try_llm_synthesis(system, sample_prompt)
            if topology_graph is None:
                log.warning(
                    "LLM synthesis unavailable; skipping llm_synthesized topology"
                )
                continue
        elif topo_name == "evolved":
            topology_graph = _get_topology_graph(
                topo_name, topology_engine=system.topology_engine
            )
            if topology_graph is None:
                log.warning(
                    "Evolved topology unavailable (empty MAP-Elites archive); "
                    "skipping"
                )
                continue
        else:
            topology_graph = _get_topology_graph(topo_name)

        if topology_graph is not None:
            log.info(
                "Topology %s: %d nodes, %d edges",
                topo_name,
                topology_graph.node_count(),
                topology_graph.edge_count(),
            )

        try:
            report = await run_topology_benchmark(
                system, topo_name, tasks, topology_graph
            )
            reports[topo_name] = report
            completed.append(topo_name)

            # Progress ETA
            elapsed = time.perf_counter() - topo_start_time
            avg_per_topo = elapsed / (topo_idx + 1)
            remaining_count = total_topologies - (topo_idx + 1)
            eta_min = (avg_per_topo * remaining_count) / 60

            print(
                f"\n  >> {topo_name}: {report.pass_rate:.1%} pass rate "
                f"({report.passed}/{report.task_count}), "
                f"avg {report.avg_latency_ms:.0f}ms, "
                f"total ${report.total_cost_usd:.2f}",
                flush=True,
            )
            if remaining_count > 0:
                print(
                    f"  >> ETA: ~{eta_min:.0f} min remaining "
                    f"({remaining_count} topologies left)",
                    flush=True,
                )

            # Save partial results after each topology
            save_partial(output_path, reports, completed, task_limit, start_time)

        except Exception as e:
            log.error(
                "Topology %s failed with unrecoverable error: %s\n%s",
                topo_name, e, traceback.format_exc(),
            )
            # Continue with next topology

    # --- Final output ---
    output = _build_output(reports, task_limit, start_time, partial=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Final results written to %s", output_path)

    # --- Summary ---
    print_summary_table(reports)

    # Print statistical highlights
    if "statistical_tests" in output:
        stats = output["statistical_tests"]
        pairwise = stats.get("pairwise", {})
        significant = {k: v for k, v in pairwise.items() if v.get("significant")}
        if significant:
            print(f"\nStatistically significant differences (p < 0.05): {len(significant)}")
            for key, val in sorted(significant.items(), key=lambda x: x[1]["p_value"]):
                print(
                    f"  {key}: p={val['p_value']:.4f}, "
                    f"Cohen's d={val['cohens_d']:.2f} ({val['effect_size']})"
                )
        else:
            print(
                "\nNo statistically significant differences found. "
                "Consider running with more tasks (--tasks 200)."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TopologyBench: benchmark multi-agent topologies on coding tasks. "
            "Proves the 'topology > model' hypothesis (AdaptOrch, arXiv 2602.16873)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_topologybench.py --tasks 5 --topologies sequential,avr\n"
            "  python scripts/run_topologybench.py --tasks 200\n"
            "  python scripts/run_topologybench.py --tasks 200 --dry-run\n"
            "  python scripts/run_topologybench.py --tasks 200 --resume data/topologybench_results.json\n"
        ),
    )
    parser.add_argument(
        "--tasks", type=int, default=200,
        help="Number of tasks to run per topology (default: 200)",
    )
    parser.add_argument(
        "--topologies", type=str, default="all",
        help=(
            "Comma-separated topology names or 'all' (default: all). "
            f"Available: {', '.join(ALL_TOPOLOGIES)}"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Per-topology task limit (overrides --tasks for quick tests)",
    )
    parser.add_argument(
        "--output", type=str, default="data/topologybench_results.json",
        help="Output JSON path (default: data/topologybench_results.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Estimate cost and exit without running any tasks",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from a partial results file",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
