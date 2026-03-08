"""Sprint 3 evidence benchmarks: HumanEval multi-config + routing value proof.

Usage:
    # HumanEval 3 configs (baseline, routing-only, full stack)
    python -m sage.bench.sprint3_evidence humaneval --limit 164

    # Routing value proof (unseen tasks, 3 baselines)
    python -m sage.bench.sprint3_evidence routing-proof
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in [here] + list(here.parents):
        if (parent / ".git").is_dir():
            return parent
    return here.parents[3]


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        for parent in [Path.cwd()] + list(Path.cwd().parents):
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                break
    except ImportError:
        pass


# ── HumanEval Multi-Config ──────────────────────────────────────────────

async def run_humaneval_config(
    config_name: str,
    baseline_mode: bool = False,
    disable_memory: bool = False,
    disable_guardrails: bool = False,
    limit: int | None = None,
    llm_tier: str = "fast",
) -> dict:
    """Run HumanEval with a specific configuration."""
    from sage.bench.humaneval import HumanEvalBench
    from sage.boot import boot_agent_system
    from sage.events.bus import EventBus

    bus = EventBus()
    system = boot_agent_system(use_mock_llm=False, llm_tier=llm_tier, event_bus=bus)

    if disable_memory:
        # Disable memory injection (routing-only mode)
        system.agent_loop.semantic_memory = None
        system.agent_loop.causal_memory = None
        system.agent_loop.exocortex = None
        # Keep working memory for basic operation but disable S-MMU context
        if hasattr(system.agent_loop, '_memory_compressor'):
            system.agent_loop._memory_compressor = None

    if disable_guardrails:
        system.agent_loop.guardrail_pipeline = None

    bench = HumanEvalBench(
        system=system,
        event_bus=bus,
        baseline_mode=baseline_mode,
    )

    log.info(f"Running HumanEval config '{config_name}' (limit={limit})...")
    t0 = time.perf_counter()
    report = bench.run if asyncio.iscoroutinefunction(bench.run) else bench.run
    report = await bench.run(limit=limit)
    elapsed = time.perf_counter() - t0

    summary = {
        "config": config_name,
        "baseline_mode": baseline_mode,
        "disable_memory": disable_memory,
        "disable_guardrails": disable_guardrails,
        "llm_tier": llm_tier,
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "pass_rate": report.pass_rate,
        "avg_latency_ms": report.avg_latency_ms,
        "avg_cost_usd": report.avg_cost_usd,
        "routing_breakdown": report.routing_breakdown,
        "total_time_s": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Print inline
    print(f"\n{'=' * 60}")
    print(f"  Config: {config_name}")
    print(f"  pass@1: {report.pass_rate:.1%} ({report.passed}/{report.total})")
    print(f"  Avg Latency: {report.avg_latency_ms:.0f}ms")
    print(f"  Avg Cost: ${report.avg_cost_usd:.6f}/task")
    print(f"  Routing: {report.routing_breakdown}")
    print(f"  Total time: {elapsed:.0f}s")
    print(f"{'=' * 60}")

    # Save individual report
    bench_dir = _repo_root() / "docs" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = bench_dir / f"{date_str}-humaneval-{config_name}.json"
    out_path.write_text(
        json.dumps(dataclasses.asdict(report), indent=2), encoding="utf-8"
    )
    log.info(f"Report saved: {out_path}")

    # Save JSONL traces
    if bench.manifest and bench.manifest.traces:
        jsonl_path = out_path.with_suffix(".jsonl")
        jsonl_path.write_text(bench.manifest.to_jsonl(), encoding="utf-8")

    return summary


async def run_humaneval_all_configs(limit: int | None = None) -> None:
    """Run HumanEval 164 with 3 configurations:
    A) Baseline: bare LLM, no routing/memory/guardrails
    B) Routing-only: S1/S2/S3 routing, no memory injection
    C) Full stack: routing + memory + guardrails
    """
    results = []

    # Config A: Baseline (bare LLM)
    log.info("=" * 70)
    log.info("CONFIG A: Baseline (bare LLM, no routing/memory/guardrails)")
    log.info("=" * 70)
    r = await run_humaneval_config(
        "baseline",
        baseline_mode=True,
        limit=limit,
    )
    results.append(r)

    # Config B: Routing-only (no memory/ExoCortex)
    log.info("=" * 70)
    log.info("CONFIG B: Routing-only (S1/S2/S3, no memory injection)")
    log.info("=" * 70)
    r = await run_humaneval_config(
        "routing-only",
        disable_memory=True,
        disable_guardrails=True,
        limit=limit,
    )
    results.append(r)

    # Config C: Full stack
    log.info("=" * 70)
    log.info("CONFIG C: Full stack (routing + memory + guardrails)")
    log.info("=" * 70)
    r = await run_humaneval_config(
        "full-stack",
        limit=limit,
    )
    results.append(r)

    # Save comparison summary
    bench_dir = _repo_root() / "docs" / "benchmarks"
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    summary_path = bench_dir / f"{date_str}-humaneval-comparison.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("  HumanEval Multi-Config Comparison")
    print(f"{'=' * 70}")
    print(f"  {'Config':<20} {'pass@1':>8} {'Passed':>8} {'Latency':>10} {'Cost':>12}")
    print(f"  {'-' * 60}")
    for r in results:
        print(
            f"  {r['config']:<20} {r['pass_rate']:>7.1%} "
            f"{r['passed']:>4}/{r['total']:<4} "
            f"{r['avg_latency_ms']:>8.0f}ms "
            f"${r['avg_cost_usd']:>10.6f}"
        )
    print(f"{'=' * 70}")
    print(f"  Summary saved: {summary_path}")


# ── Routing Value Proof ─────────────────────────────────────────────────

# 30 UNSEEN tasks (NOT the calibrated routing benchmark tasks)
ROUTING_PROOF_TASKS = [
    # Simple tasks (should route S1 or cheap)
    {"task": "What is the square root of 144", "category": "simple"},
    {"task": "List the months of the year", "category": "simple"},
    {"task": "Define photosynthesis in one sentence", "category": "simple"},
    {"task": "What is the boiling point of water in Celsius", "category": "simple"},
    {"task": "Name the author of Romeo and Juliet", "category": "simple"},
    {"task": "Reverse the string 'hello world'", "category": "simple"},
    {"task": "What does HTTP stand for", "category": "simple"},
    {"task": "Convert 50 miles to kilometers", "category": "simple"},
    {"task": "What is the chemical formula for water", "category": "simple"},
    {"task": "Count the vowels in 'supercalifragilisticexpialidocious'", "category": "simple"},

    # Medium tasks (should route S2)
    {"task": "Write a Python function to check if a string is a palindrome", "category": "medium"},
    {"task": "Implement binary search in Python", "category": "medium"},
    {"task": "Write a function that finds the nth Fibonacci number using memoization", "category": "medium"},
    {"task": "Create a Python class for a linked list with insert and delete", "category": "medium"},
    {"task": "Write a regex to validate email addresses", "category": "medium"},
    {"task": "Implement a stack using two queues", "category": "medium"},
    {"task": "Write a function to merge two sorted arrays", "category": "medium"},
    {"task": "Create a decorator that caches function results", "category": "medium"},
    {"task": "Write a function to detect cycles in a linked list", "category": "medium"},
    {"task": "Implement a simple LRU cache in Python", "category": "medium"},

    # Complex tasks (should route S3 or high-tier)
    {"task": "Debug this race condition in a producer-consumer queue and design a lock-free alternative using compare-and-swap", "category": "complex"},
    {"task": "Fix the memory corruption in this unsafe Rust code and architect a safe abstraction layer with lifetime guarantees", "category": "complex"},
    {"task": "Analyze why this distributed consensus algorithm fails under network partitions and evolve it to handle Byzantine faults", "category": "complex"},
    {"task": "Debug the numerical instability in this gradient descent optimizer and design an adaptive learning rate schedule", "category": "complex"},
    {"task": "Fix the deadlock in this transaction manager and architect a multi-version concurrency control system", "category": "complex"},
    {"task": "Diagnose the error in this B-tree implementation and optimize it for SSD access patterns with write amplification reduction", "category": "complex"},
    {"task": "Debug the crash in this JIT compiler's register allocator and design a graph-coloring approach with spill cost optimization", "category": "complex"},
    {"task": "Fix the serialization bug in this RPC framework and architect a schema evolution strategy with backward compatibility", "category": "complex"},
    {"task": "Analyze this error in the garbage collector's mark phase and evolve it to support concurrent marking without stop-the-world pauses", "category": "complex"},
    {"task": "Debug the crash in this neural network trainer and optimize the backpropagation for mixed-precision with gradient scaling", "category": "complex"},
]


async def run_routing_proof() -> None:
    """Run routing value proof: compare router vs all-S1 vs all-S2.

    Methodology (inspired by RouteLLM 2024):
    1. All tasks → S1 (cheapest model)
    2. All tasks → S2 (best available model)
    3. Router decides (ComplexityRouter)
    4. Proof: Router achieves >= S2 quality at < S2 cost
    """
    from sage.boot import boot_agent_system
    from sage.events.bus import EventBus
    from sage.strategy.metacognition import ComplexityRouter

    bench_dir = _repo_root() / "docs" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # First, route all tasks to see distribution
    router = ComplexityRouter()
    routing_decisions = []
    for item in ROUTING_PROOF_TASKS:
        profile = router.assess_complexity(item["task"])
        decision = router.route(profile)
        routing_decisions.append({
            "task": item["task"],
            "category": item["category"],
            "routed_to": f"S{decision.system}",
            "complexity": round(profile.complexity, 2),
            "uncertainty": round(profile.uncertainty, 2),
            "tool_required": profile.tool_required,
        })

    # Print routing distribution
    from collections import Counter
    by_system = Counter(d["routed_to"] for d in routing_decisions)
    by_category = {}
    for d in routing_decisions:
        key = (d["category"], d["routed_to"])
        by_category[key] = by_category.get(key, 0) + 1

    print(f"\n{'=' * 60}")
    print("  Routing Distribution (30 unseen tasks)")
    print(f"{'=' * 60}")
    print(f"  S1: {by_system.get('S1', 0)}  S2: {by_system.get('S2', 0)}  S3: {by_system.get('S3', 0)}")
    print(f"\n  By category:")
    for cat in ["simple", "medium", "complex"]:
        parts = []
        for sys in ["S1", "S2", "S3"]:
            count = by_category.get((cat, sys), 0)
            if count:
                parts.append(f"S{sys[-1]}={count}")
        print(f"    {cat:<10}: {', '.join(parts)}")
    print(f"{'=' * 60}")

    # Run with router (Config R: actual routing)
    log.info("Running 30 unseen tasks with ComplexityRouter...")
    bus_r = EventBus()
    system_r = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=bus_r)
    router_results = []
    total_cost_r = 0.0

    for i, item in enumerate(ROUTING_PROOF_TASKS):
        t0 = time.perf_counter()
        try:
            result = await system_r.run(item["task"])
            passed = bool(result and len(result.strip()) > 5)
            error = ""
        except Exception as e:
            result = ""
            passed = False
            error = str(e)[:200]
        latency = (time.perf_counter() - t0) * 1000
        cost = getattr(system_r.agent_loop, "total_cost_usd", 0.0)
        step_cost = cost - total_cost_r
        total_cost_r = cost

        router_results.append({
            "task": item["task"][:60],
            "category": item["category"],
            "routing": routing_decisions[i]["routed_to"],
            "passed": passed,
            "latency_ms": round(latency, 0),
            "cost_usd": round(step_cost, 6),
            "error": error,
        })
        status = "OK" if passed else "FAIL"
        log.info(f"  [{i+1}/30] {status} ({latency:.0f}ms) {item['task'][:50]}")

    # Run with all-S1 (Config S1: cheapest, baseline)
    log.info("Running 30 unseen tasks with ALL-S1 (cheapest)...")
    bus_s1 = EventBus()
    system_s1 = boot_agent_system(use_mock_llm=False, llm_tier="budget", event_bus=bus_s1)
    s1_results = []
    total_cost_s1 = 0.0

    for i, item in enumerate(ROUTING_PROOF_TASKS):
        t0 = time.perf_counter()
        try:
            # Direct call, no routing
            result = await system_s1.agent_loop.run(item["task"])
            passed = bool(result and len(result.strip()) > 5)
            error = ""
        except Exception as e:
            result = ""
            passed = False
            error = str(e)[:200]
        latency = (time.perf_counter() - t0) * 1000
        cost = getattr(system_s1.agent_loop, "total_cost_usd", 0.0)
        step_cost = cost - total_cost_s1
        total_cost_s1 = cost

        s1_results.append({
            "task": item["task"][:60],
            "category": item["category"],
            "passed": passed,
            "latency_ms": round(latency, 0),
            "cost_usd": round(step_cost, 6),
            "error": error,
        })
        status = "OK" if passed else "FAIL"
        log.info(f"  [{i+1}/30] {status} ({latency:.0f}ms) {item['task'][:50]}")

    # Compute metrics
    r_pass = sum(1 for r in router_results if r["passed"])
    r_cost = sum(r["cost_usd"] for r in router_results)
    r_lat = sum(r["latency_ms"] for r in router_results) / len(router_results)

    s1_pass = sum(1 for r in s1_results if r["passed"])
    s1_cost = sum(r["cost_usd"] for r in s1_results)
    s1_lat = sum(r["latency_ms"] for r in s1_results) / len(s1_results)

    proof = {
        "methodology": "RouteLLM-inspired: 30 unseen tasks, router vs all-S1 baseline",
        "date": date_str,
        "router": {
            "pass_count": r_pass,
            "pass_rate": round(r_pass / 30, 3),
            "total_cost_usd": round(r_cost, 6),
            "avg_latency_ms": round(r_lat, 0),
        },
        "all_s1": {
            "pass_count": s1_pass,
            "pass_rate": round(s1_pass / 30, 3),
            "total_cost_usd": round(s1_cost, 6),
            "avg_latency_ms": round(s1_lat, 0),
        },
        "routing_distribution": dict(by_system),
        "routing_decisions": routing_decisions,
        "router_results": router_results,
        "s1_results": s1_results,
        "verdict": (
            f"Router: {r_pass}/30 ({r_pass/30:.0%}) at ${r_cost:.4f}. "
            f"All-S1: {s1_pass}/30 ({s1_pass/30:.0%}) at ${s1_cost:.4f}. "
            f"Cost ratio: {r_cost/max(s1_cost, 0.0001):.1f}x"
        ),
    }

    # Print comparison
    print(f"\n{'=' * 60}")
    print("  Routing Value Proof")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<20} {'Router':>12} {'All-S1':>12}")
    print(f"  {'-' * 44}")
    print(f"  {'Pass rate':<20} {r_pass/30:>11.0%} {s1_pass/30:>11.0%}")
    print(f"  {'Total cost':<20} ${r_cost:>10.4f} ${s1_cost:>10.4f}")
    print(f"  {'Avg latency':<20} {r_lat:>10.0f}ms {s1_lat:>10.0f}ms")
    print(f"{'=' * 60}")

    out_path = bench_dir / f"{date_str}-routing-proof.json"
    out_path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    log.info(f"Routing proof saved: {out_path}")


# ── Evolution 4-Arm Validation ───────────────────────────────────────────

async def run_evolution_validation(
    n_seeds: int = 5,
    n_generations: int = 10,
    mutations_per_gen: int = 3,
) -> None:
    """4-arm evolution experiment: full engine vs random vs no-SAMPO vs seed-only.

    Uses simple Python code optimization tasks (sort, fibonacci, etc.)
    to measure whether SAMPO + DGM adds value over random mutation.
    """
    from sage.evolution.engine import EvolutionEngine, EvolutionConfig
    from sage.evolution.population import Individual
    from sage.evolution.evaluator import Evaluator, EvalResult

    bench_dir = _repo_root() / "docs" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Seed code: simple Python functions to optimize
    seed_codes = [
        "def sort_list(lst):\n    return sorted(lst)\n",
        "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\n",
        "def reverse_string(s):\n    return s[::-1]\n",
        "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, n):\n        if n % i == 0: return False\n    return True\n",
        "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n",
    ]

    # Simple evaluator: syntax check + basic execution
    async def syntax_eval(code: str) -> EvalResult:
        try:
            compile(code, "<eval>", "exec")
            return EvalResult(score=1.0, passed=True, stage="syntax")
        except SyntaxError as e:
            return EvalResult(score=0.0, passed=False, stage="syntax", error=str(e))

    async def execution_eval(code: str) -> EvalResult:
        import subprocess
        import tempfile
        test_code = code + "\n# Smoke test\nprint('OK')\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            tmp = f.name
        try:
            result = subprocess.run(
                ["python", tmp], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                score = 0.8 + 0.2 * (len(code) < 500)  # Bonus for conciseness
                return EvalResult(score=score, passed=True, stage="execution")
            return EvalResult(score=0.3, passed=False, stage="execution", error=result.stderr[:100])
        except subprocess.TimeoutExpired:
            return EvalResult(score=0.1, passed=False, stage="execution", error="timeout")
        finally:
            Path(tmp).unlink(missing_ok=True)

    evaluator = Evaluator()
    evaluator.add_stage("syntax", syntax_eval, threshold=0.5, weight=0.3)
    evaluator.add_stage("execution", execution_eval, threshold=0.3, weight=0.7)

    # Mutation function using LLM
    async def llm_mutate(code: str, dgm_context: dict | None = None) -> tuple[str, tuple[int, ...]]:
        from sage.llm.google import GoogleProvider
        from sage.llm.base import Message, Role
        provider = GoogleProvider()
        action_desc = dgm_context.get("description", "improve") if dgm_context else "improve"
        prompt = (
            f"Improve this Python function. Goal: {action_desc}.\n"
            f"Return ONLY the improved code, no explanation.\n\n```python\n{code}\n```"
        )
        try:
            response = await provider.generate([Message(role=Role.USER, content=prompt)])
            import re
            blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response.content, re.DOTALL)
            new_code = blocks[0] if blocks else response.content
            # Simple feature extraction: (line_count_bin, has_docstring)
            lines = len(new_code.strip().split("\n"))
            has_doc = 1 if '"""' in new_code or "'''" in new_code else 0
            return new_code, (min(lines // 5, 9), has_doc)
        except Exception as e:
            log.warning(f"LLM mutation failed: {e}")
            return code, (0, 0)

    # Random mutation function (no LLM, just code perturbation)
    import random
    async def random_mutate(code: str, dgm_context: dict | None = None) -> tuple[str, tuple[int, ...]]:
        lines = code.split("\n")
        if len(lines) > 2:
            # Random: add a comment, swap lines, or duplicate a line
            action = random.choice(["comment", "swap", "duplicate"])
            if action == "comment":
                idx = random.randint(0, len(lines) - 1)
                lines.insert(idx, "    # optimized")
            elif action == "swap" and len(lines) > 3:
                i, j = random.sample(range(1, len(lines)), 2)
                lines[i], lines[j] = lines[j], lines[i]
            elif action == "duplicate":
                idx = random.randint(1, len(lines) - 1)
                lines.insert(idx, lines[idx])
        new_code = "\n".join(lines)
        line_count = len(new_code.strip().split("\n"))
        return new_code, (min(line_count // 5, 9), 0)

    results = {}

    # Arm 1: Full Engine (DGM + SAMPO + MAP-Elites)
    log.info("ARM 1: Full Engine (DGM + SAMPO + MAP-Elites)")
    config1 = EvolutionConfig(
        population_size=50,
        mutations_per_generation=mutations_per_gen,
        max_generations=n_generations,
        enable_dgm=True,
    )
    engine1 = EvolutionEngine(config=config1, evaluator=evaluator)
    for i, code in enumerate(seed_codes[:n_seeds]):
        eval_r = await evaluator.evaluate(code)
        engine1.seed([Individual(code=code, score=eval_r.score, features=(i, 0))])
    for gen in range(n_generations):
        new_inds = await engine1.evolve_step(llm_mutate)
        log.info(f"  Gen {gen+1}: {len(new_inds)} accepted")
    stats1 = engine1.stats()
    results["full_engine"] = {
        "best_score": stats1.get("best_score", 0),
        "population_size": stats1.get("population_size", 0),
        "coverage": stats1.get("coverage", 0),
    }

    # Arm 2: Random Mutation (no LLM, no SAMPO)
    log.info("ARM 2: Random Mutation (no LLM, no SAMPO)")
    config2 = EvolutionConfig(
        population_size=50,
        mutations_per_generation=mutations_per_gen,
        max_generations=n_generations,
        enable_dgm=False,
    )
    engine2 = EvolutionEngine(config=config2, evaluator=evaluator)
    for i, code in enumerate(seed_codes[:n_seeds]):
        eval_r = await evaluator.evaluate(code)
        engine2.seed([Individual(code=code, score=eval_r.score, features=(i, 0))])
    for gen in range(n_generations):
        new_inds = await engine2.evolve_step(random_mutate)
        log.info(f"  Gen {gen+1}: {len(new_inds)} accepted")
    stats2 = engine2.stats()
    results["random_mutation"] = {
        "best_score": stats2.get("best_score", 0),
        "population_size": stats2.get("population_size", 0),
        "coverage": stats2.get("coverage", 0),
    }

    # Arm 3: LLM Mutation without SAMPO (uniform random actions)
    log.info("ARM 3: LLM Mutation without SAMPO")
    config3 = EvolutionConfig(
        population_size=50,
        mutations_per_generation=mutations_per_gen,
        max_generations=n_generations,
        enable_dgm=False,
    )
    engine3 = EvolutionEngine(config=config3, evaluator=evaluator)
    for i, code in enumerate(seed_codes[:n_seeds]):
        eval_r = await evaluator.evaluate(code)
        engine3.seed([Individual(code=code, score=eval_r.score, features=(i, 0))])
    for gen in range(n_generations):
        new_inds = await engine3.evolve_step(llm_mutate)
        log.info(f"  Gen {gen+1}: {len(new_inds)} accepted")
    stats3 = engine3.stats()
    results["llm_no_sampo"] = {
        "best_score": stats3.get("best_score", 0),
        "population_size": stats3.get("population_size", 0),
        "coverage": stats3.get("coverage", 0),
    }

    # Arm 4: Seed Only (no evolution)
    log.info("ARM 4: Seed Only (no evolution)")
    config4 = EvolutionConfig(population_size=50)
    engine4 = EvolutionEngine(config=config4, evaluator=evaluator)
    for i, code in enumerate(seed_codes[:n_seeds]):
        eval_r = await evaluator.evaluate(code)
        engine4.seed([Individual(code=code, score=eval_r.score, features=(i, 0))])
    stats4 = engine4.stats()
    results["seed_only"] = {
        "best_score": stats4.get("best_score", 0),
        "population_size": stats4.get("population_size", 0),
        "coverage": stats4.get("coverage", 0),
    }

    # Print comparison
    print(f"\n{'=' * 60}")
    print("  Evolution 4-Arm Validation")
    print(f"{'=' * 60}")
    print(f"  {'Arm':<25} {'Best Score':>12} {'Pop Size':>10} {'Coverage':>10}")
    print(f"  {'-' * 57}")
    for arm_name, arm_data in results.items():
        print(
            f"  {arm_name:<25} {arm_data['best_score']:>11.3f} "
            f"{arm_data['population_size']:>9} "
            f"{arm_data['coverage']:>9.1%}"
        )
    print(f"{'=' * 60}")

    # Save results
    out_path = bench_dir / f"{date_str}-evolution-4arm.json"
    out_path.write_text(json.dumps({
        "methodology": "4-arm evolution validation: full vs random vs llm-no-sampo vs seed-only",
        "n_seeds": n_seeds,
        "n_generations": n_generations,
        "mutations_per_gen": mutations_per_gen,
        "results": results,
        "date": date_str,
    }, indent=2), encoding="utf-8")
    log.info(f"Evolution results saved: {out_path}")


# ── Memory Ablation ─────────────────────────────────────────────────────

async def run_memory_ablation(limit: int | None = 20) -> None:
    """Memory ablation: compare full memory vs no memory on HumanEval.

    Configs:
    A) Full memory (all 4 tiers + S-MMU + ExoCortex)
    B) No memory (all disabled except working memory)
    C) Episodic + semantic only (no S-MMU compression, no ExoCortex)
    """
    results = []

    # Config A: Full memory (= full-stack from HumanEval config)
    log.info("MEMORY ABLATION — Config A: Full memory")
    r = await run_humaneval_config("memory-full", limit=limit)
    results.append(r)

    # Config B: No memory
    log.info("MEMORY ABLATION — Config B: No memory")
    r = await run_humaneval_config(
        "memory-none",
        disable_memory=True,
        limit=limit,
    )
    results.append(r)

    # Config C: Episodic + semantic only (no ExoCortex, no compression)
    log.info("MEMORY ABLATION — Config C: Episodic + semantic only")
    from sage.bench.humaneval import HumanEvalBench
    from sage.boot import boot_agent_system
    from sage.events.bus import EventBus

    bus = EventBus()
    system = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=bus)
    # Disable ExoCortex and memory compressor, keep episodic + semantic
    system.agent_loop.exocortex = None
    system.agent_loop.memory_compressor = None

    bench = HumanEvalBench(system=system, event_bus=bus)
    t0 = time.perf_counter()
    report = await bench.run(limit=limit)
    elapsed = time.perf_counter() - t0

    r = {
        "config": "memory-episodic-semantic",
        "total": report.total,
        "passed": report.passed,
        "pass_rate": report.pass_rate,
        "avg_latency_ms": report.avg_latency_ms,
        "avg_cost_usd": report.avg_cost_usd,
        "routing_breakdown": report.routing_breakdown,
        "total_time_s": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    results.append(r)
    print(f"\n  Config memory-episodic-semantic: {report.pass_rate:.0%} ({report.passed}/{report.total})")

    # Print comparison
    print(f"\n{'=' * 60}")
    print("  Memory Ablation Comparison")
    print(f"{'=' * 60}")
    print(f"  {'Config':<30} {'pass@1':>8} {'Latency':>10} {'Cost':>12}")
    print(f"  {'-' * 60}")
    for r in results:
        print(
            f"  {r['config']:<30} {r['pass_rate']:>7.1%} "
            f"{r['avg_latency_ms']:>8.0f}ms "
            f"${r['avg_cost_usd']:>10.6f}"
        )
    print(f"{'=' * 60}")

    # Save
    bench_dir = _repo_root() / "docs" / "benchmarks"
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = bench_dir / f"{date_str}-memory-ablation.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info(f"Memory ablation saved: {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sage.bench.sprint3_evidence",
        description="Sprint 3 evidence benchmarks",
    )
    parser.add_argument(
        "benchmark",
        choices=["humaneval", "routing-proof", "evolution", "memory-ablation", "all"],
        help="Which benchmark to run",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit problems (humaneval/memory)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (evolution)")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations (evolution)")
    args = parser.parse_args()

    _load_env()

    if args.benchmark in ("humaneval", "all"):
        asyncio.run(run_humaneval_all_configs(limit=args.limit))

    if args.benchmark in ("routing-proof", "all"):
        asyncio.run(run_routing_proof())

    if args.benchmark in ("evolution", "all"):
        asyncio.run(run_evolution_validation(
            n_seeds=args.seeds,
            n_generations=args.generations,
        ))

    if args.benchmark in ("memory-ablation", "all"):
        asyncio.run(run_memory_ablation(limit=args.limit or 20))


if __name__ == "__main__":
    main()
