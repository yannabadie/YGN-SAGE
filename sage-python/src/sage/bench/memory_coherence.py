"""Memory coherence benchmark — prove archive→retrieve→quality+ gain.

Thesis (CORAL arXiv 2604.01658 + MEM1): when an agent has previously solved
a semantically close task, retrieving that episode yields higher quality
OR lower latency/cost than re-solving from scratch.

Method:
  Phase 1 — PRIME:    solve N anchor tasks, store each (task, solution)
                       in episodic memory.
  Phase 2 — PROBE:    for N companion tasks (semantically near anchors),
                       measure two runs:
                         a) cold_run    — memory cleared, solve from scratch
                         b) primed_run  — anchor episodes preloaded, episodic
                                           search injected into context
  Report:            per-probe (quality_cold, quality_primed, latency_cold,
                       latency_primed). Aggregate delta on pass_rate, quality,
                       latency.

Quality rubric (LLM-free for CI portability):
  score = (
      0.4 if len(solution.strip()) > 30 else 0.0
    + 0.3 if "def " in solution or "class " in solution else 0.0
    + 0.3 if ast_parses_ok(solution) else 0.0
  )

The bench deliberately uses compile-checkable quality to avoid an LLM judge
loop. For full rubric (test execution) use evalplus / bigcodebench instead.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Task pairs — anchor + semantically-close companion
# ----------------------------------------------------------------------
# Each pair is designed so that a reasonable episodic retrieval would
# yield a directly useful starter. Tasks are intentionally small to
# keep bench turn-around under ~10 minutes on a fast provider.

TASK_PAIRS: list[tuple[str, str]] = [
    (
        "Write a Python function `is_palindrome(s)` that returns True if the string reads the same forwards and backwards (ignore case).",
        "Write a Python function `is_palindromic_number(n)` that returns True if the integer n is a palindrome in base 10.",
    ),
    (
        "Implement `gcd(a, b)` using Euclid's algorithm in Python.",
        "Implement `lcm(a, b)` in Python using the relation lcm = a*b/gcd(a,b).",
    ),
    (
        "Write a Python function `reverse_list(xs)` that returns a new list with the elements in reverse order without using the reversed built-in.",
        "Write a Python function `reverse_string(s)` that returns the string reversed without slicing notation.",
    ),
    (
        "Implement binary search `bsearch(xs, target)` over a sorted list, returning the index or -1.",
        "Implement a function `lower_bound(xs, target)` over a sorted list, returning the first index i where xs[i] >= target.",
    ),
    (
        "Write `fibonacci(n)` returning the nth Fibonacci number (iterative, O(n)).",
        "Write `tribonacci(n)` returning the nth Tribonacci number where T(0)=0, T(1)=1, T(2)=1.",
    ),
]


@dataclass
class ProbeResult:
    pair_idx: int
    anchor_task: str
    probe_task: str
    cold_quality: float = 0.0
    primed_quality: float = 0.0
    cold_latency_ms: float = 0.0
    primed_latency_ms: float = 0.0
    cold_tokens: int = 0
    primed_tokens: int = 0
    cold_error: str | None = None
    primed_error: str | None = None

    @property
    def quality_delta(self) -> float:
        return self.primed_quality - self.cold_quality

    @property
    def latency_delta_ms(self) -> float:
        return self.primed_latency_ms - self.cold_latency_ms


@dataclass
class MemoryCoherenceReport:
    benchmark: str = "memory_coherence"
    total: int = 0
    cold_pass: int = 0
    primed_pass: int = 0
    avg_cold_quality: float = 0.0
    avg_primed_quality: float = 0.0
    avg_cold_latency_ms: float = 0.0
    avg_primed_latency_ms: float = 0.0
    quality_gain: float = 0.0
    latency_gain_ms: float = 0.0
    probes: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""
    passed: int = 0  # alias for primed_pass (runner convention)
    pass_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0
    results: list[dict[str, Any]] = field(default_factory=list)
    routing_breakdown: dict[str, int] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

def _score_solution(solution: str) -> float:
    """Cheap, deterministic quality score. Returns in [0, 1]."""
    if not solution:
        return 0.0
    stripped = solution.strip()
    if not stripped:
        return 0.0
    score = 0.0
    if len(stripped) > 30:
        score += 0.4
    if "def " in stripped or "class " in stripped:
        score += 0.3
    try:
        tree = ast.parse(stripped)
        if tree.body:  # empty modules parse OK but carry no content
            score += 0.3
    except SyntaxError:
        pass
    return score


def _extract_code(response: str) -> str:
    """Pull the first ```python block if present, else return raw response."""
    if "```" not in response:
        return response
    parts = response.split("```")
    for i, part in enumerate(parts):
        if i % 2 == 1:  # inside a fence
            stripped = part.strip()
            if stripped.startswith("python"):
                stripped = stripped[len("python"):].lstrip("\n")
            return stripped
    return response


# ----------------------------------------------------------------------
# Bench
# ----------------------------------------------------------------------

async def _run_one(system: Any, task: str, context_prefix: str = "") -> tuple[str, float, int]:
    """Execute a task through the given AgentSystem. Returns (response, latency_ms, token_estimate)."""
    full_task = f"{context_prefix}\n\n{task}" if context_prefix else task
    t0 = time.perf_counter()
    response = await system.run(full_task)
    latency_ms = (time.perf_counter() - t0) * 1000
    # Rough token estimate (no provider tokenizer): 4 chars per token
    tokens = (len(full_task) + len(response or "")) // 4
    return (response or ""), latency_ms, tokens


async def run_memory_coherence(
    boot_fn,
    pairs: list[tuple[str, str]] | None = None,
    limit: int | None = None,
) -> MemoryCoherenceReport:
    """Run the memory coherence benchmark.

    Args:
        boot_fn: coroutine returning an AgentSystem. Called twice per pair
                 (cold + primed) — isolated instances for fair comparison.
        pairs:   override TASK_PAIRS (for small-run tests).
        limit:   truncate to first *limit* pairs (CLI convenience).
    """
    pair_set = pairs or TASK_PAIRS
    if limit is not None:
        pair_set = pair_set[:limit]

    results: list[ProbeResult] = []

    for idx, (anchor, probe) in enumerate(pair_set):
        probe_result = ProbeResult(pair_idx=idx, anchor_task=anchor, probe_task=probe)

        # ── COLD RUN: fresh system, no memory ──
        try:
            cold_system = await boot_fn()
            cold_resp, cold_lat, cold_tok = await _run_one(cold_system, probe)
            probe_result.cold_quality = _score_solution(_extract_code(cold_resp))
            probe_result.cold_latency_ms = cold_lat
            probe_result.cold_tokens = cold_tok
        except Exception as exc:  # pragma: no cover — live bench best-effort
            probe_result.cold_error = f"{type(exc).__name__}: {exc}"
            _log.warning("Pair %d cold run failed: %s", idx, exc)

        # ── PRIMED RUN: fresh system, anchor solution pre-stored in episodic ──
        try:
            primed_system = await boot_fn()
            # 1. Solve anchor first (this populates episodic via the agent loop)
            anchor_resp, _, _ = await _run_one(primed_system, anchor)
            anchor_code = _extract_code(anchor_resp)
            # 2. Explicit episodic store for robustness across boot configs
            loop = getattr(primed_system, "agent_loop", None)
            if loop is not None and getattr(loop, "episodic_memory", None) is not None:
                try:
                    await loop.episodic_memory.store(
                        key=f"anchor-{idx}",
                        content=anchor_code,
                        metadata={"task": anchor, "source": "memory_coherence_prime"},
                    )
                except Exception as exc:  # pragma: no cover
                    _log.debug("Episodic store fallback failed: %s", exc)
            # 3. Retrieve and inject as context for the probe
            retrieved: list[dict[str, Any]] = []
            if loop is not None and getattr(loop, "episodic_memory", None) is not None:
                try:
                    retrieved = await loop.episodic_memory.search(probe, top_k=3)
                except Exception as exc:  # pragma: no cover
                    _log.debug("Episodic search failed: %s", exc)
            context_prefix = ""
            if retrieved:
                snippets = "\n\n".join(
                    f"Prior solution: {r.get('content', '')[:400]}" for r in retrieved
                )
                context_prefix = (
                    "You previously solved a related task. Use the solution "
                    "pattern if applicable, adapt it to the new requirement.\n"
                    f"{snippets}"
                )
            primed_resp, primed_lat, primed_tok = await _run_one(
                primed_system, probe, context_prefix=context_prefix
            )
            probe_result.primed_quality = _score_solution(_extract_code(primed_resp))
            probe_result.primed_latency_ms = primed_lat
            probe_result.primed_tokens = primed_tok
        except Exception as exc:  # pragma: no cover
            probe_result.primed_error = f"{type(exc).__name__}: {exc}"
            _log.warning("Pair %d primed run failed: %s", idx, exc)

        results.append(probe_result)
        _log.info(
            "Pair %d: cold q=%.2f (%.0fms) → primed q=%.2f (%.0fms) "
            "Δq=%+.2f Δlat=%+.0fms",
            idx,
            probe_result.cold_quality,
            probe_result.cold_latency_ms,
            probe_result.primed_quality,
            probe_result.primed_latency_ms,
            probe_result.quality_delta,
            probe_result.latency_delta_ms,
        )

    # ── Aggregate ──
    n = len(results)
    cold_q = [r.cold_quality for r in results if r.cold_error is None]
    primed_q = [r.primed_quality for r in results if r.primed_error is None]
    cold_lat = [r.cold_latency_ms for r in results if r.cold_error is None]
    primed_lat = [r.primed_latency_ms for r in results if r.primed_error is None]

    report = MemoryCoherenceReport(
        total=n,
        cold_pass=sum(1 for q in cold_q if q >= 0.7),
        primed_pass=sum(1 for q in primed_q if q >= 0.7),
        avg_cold_quality=sum(cold_q) / len(cold_q) if cold_q else 0.0,
        avg_primed_quality=sum(primed_q) / len(primed_q) if primed_q else 0.0,
        avg_cold_latency_ms=sum(cold_lat) / len(cold_lat) if cold_lat else 0.0,
        avg_primed_latency_ms=sum(primed_lat) / len(primed_lat) if primed_lat else 0.0,
        probes=[
            {
                "pair_idx": r.pair_idx,
                "cold_quality": r.cold_quality,
                "primed_quality": r.primed_quality,
                "cold_latency_ms": r.cold_latency_ms,
                "primed_latency_ms": r.primed_latency_ms,
                "quality_delta": r.quality_delta,
                "latency_delta_ms": r.latency_delta_ms,
                "cold_error": r.cold_error,
                "primed_error": r.primed_error,
            }
            for r in results
        ],
    )
    report.quality_gain = report.avg_primed_quality - report.avg_cold_quality
    report.latency_gain_ms = report.avg_cold_latency_ms - report.avg_primed_latency_ms
    # Runner-convention aliases
    report.passed = report.primed_pass
    report.pass_rate = report.primed_pass / n if n else 0.0
    report.avg_latency_ms = report.avg_primed_latency_ms
    report.results = report.probes
    from datetime import datetime, timezone
    report.timestamp = datetime.now(timezone.utc).isoformat()
    return report


# ----------------------------------------------------------------------
# Runnable entry point
# ----------------------------------------------------------------------

async def _default_boot():
    """Default boot: full SAGE system with fresh DB per call (no cross-run bleed)."""
    from sage.boot import boot_agent_system
    import tempfile
    import os
    tmpdir = tempfile.mkdtemp(prefix="sage_memcoh_")
    os.environ["SAGE_DB_PATH"] = os.path.join(tmpdir, "episodic.db")
    return boot_agent_system(use_mock_llm=False)


def main() -> None:
    import argparse
    import dataclasses
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description="Memory coherence benchmark")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of task pairs (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    report = asyncio.run(run_memory_coherence(_default_boot, limit=args.limit))

    print(f"\n{'=' * 60}")
    print("  Benchmark: memory_coherence")
    print(f"{'=' * 60}")
    print(f"  Pairs run          : {report.total}")
    print(f"  Cold   pass@0.7    : {report.cold_pass}/{report.total}")
    print(f"  Primed pass@0.7    : {report.primed_pass}/{report.total}")
    print(f"  Avg quality        : cold={report.avg_cold_quality:.3f}  primed={report.avg_primed_quality:.3f}  Δ={report.quality_gain:+.3f}")
    print(f"  Avg latency (ms)   : cold={report.avg_cold_latency_ms:.0f}  primed={report.avg_primed_latency_ms:.0f}  Δ={report.latency_gain_ms:+.0f}")
    print(f"{'=' * 60}\n")

    if args.output:
        Path(args.output).write_text(json.dumps(dataclasses.asdict(report), indent=2), encoding="utf-8")
        print(f"  Report saved to: {args.output}")


if __name__ == "__main__":
    main()
