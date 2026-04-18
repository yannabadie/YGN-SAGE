"""Microbenchmark: where does TopologyRunner.run() actually spend time?

Question (2026-04-18): is the Python-side fan-out dispatch overhead big
enough to justify rewriting TopologyRunner in Rust? Directive #1
("Rust first") is a principle, not an axe: Rust pays off on CPU-bound
loops, JSON-parsing hot paths, zero-copy buffers, or schedulers that
beat asyncio on real workloads. A fan-out whose per-node work is an
HTTP call to OpenAI is IO-bound — asyncio releases the GIL on every
await, so the theoretical Rust win is a few μs / node / batch.

This script isolates that claim with a fake LLMProvider that returns
after an artificial latency, so the ONLY real cost is dispatch +
bookkeeping. Run:

    python tests/perf/profile_topology_runner_dispatch.py

Report: total wall-clock vs. sum of per-node artificial latencies. If
``dispatch_overhead = wall_clock - max(latencies)`` is <1% of the
per-node latency, Rust-ifying won't help.

Uses SimpleNamespace + AsyncMock instead of the real topology graph
(avoids sage-core dependency). The point is to measure the Python
orchestration layer — the graph walk, context snapshot, gather,
aggregation — not the actual LLM stack.
"""
from __future__ import annotations

import asyncio
import sys
import time
import types as _types
from statistics import mean, median, stdev
from unittest.mock import AsyncMock, MagicMock

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = _types.ModuleType("sage_core")


def _make_graph(n_nodes: int) -> MagicMock:
    graph = MagicMock()
    graph.node_count.return_value = n_nodes
    nodes = []
    for i in range(n_nodes):
        n = MagicMock()
        n.role = f"actor-{i}"
        n.model_id = ""
        n.prompt = ""
        n.node_type = "llm"
        n.required_capabilities = []
        nodes.append(n)
    graph.get_node = lambda idx: nodes[idx]
    graph.get_predecessors = lambda idx: []
    return graph


def _make_executor(batches: list[list[int]]) -> MagicMock:
    ex = MagicMock()
    state = [0]

    def _next_ready(_g):
        if state[0] < len(batches):
            out = batches[state[0]]
            state[0] += 1
            return out
        return []

    ex.next_ready = _next_ready
    ex.is_done = lambda: state[0] >= len(batches)
    return ex


async def _run_single_batch(n_nodes: int, per_node_latency_s: float, repetitions: int) -> dict:
    """Run TopologyRunner with N parallel nodes; each node sleeps `latency_s`."""
    from sage.topology.runner import TopologyRunner

    wall_clocks: list[float] = []
    overheads: list[float] = []

    for _ in range(repetitions):
        # Fresh fake loops each repetition so there's no state leak
        fake_loops = []
        for _ in range(n_nodes):
            loop = MagicMock()

            async def _fake_run(task: str, *, _lat=per_node_latency_s) -> str:
                await asyncio.sleep(_lat)
                return "done"

            loop.run = AsyncMock(side_effect=_fake_run)
            loop.total_cost_usd = 0.0
            loop.tool_call_count = 0
            loop.tool_turn_count = 0
            loop.executed_commands = []
            fake_loops.append(loop)

        call_count = [0]

        def _factory(**_kwargs):
            loop = fake_loops[call_count[0]]
            call_count[0] += 1
            return loop

        # One batch of N parallel nodes
        batches = [list(range(n_nodes))]
        runner = TopologyRunner(
            graph=_make_graph(n_nodes),
            executor=_make_executor(batches),
            llm_provider=MagicMock(),
            agent_loop_factory=_factory,
        )

        t0 = time.perf_counter()
        await runner.run("bench task")
        wall = time.perf_counter() - t0

        # Dispatch overhead ≈ wall_clock - per_node_latency (because the max of
        # N concurrent awaits of duration L is just L; anything above is the
        # orchestration tax).
        overhead = max(0.0, wall - per_node_latency_s)
        wall_clocks.append(wall * 1000)  # ms
        overheads.append(overhead * 1000)  # ms

    return {
        "n_nodes": n_nodes,
        "per_node_latency_ms": per_node_latency_s * 1000,
        "repetitions": repetitions,
        "wall_clock_ms": {
            "median": median(wall_clocks),
            "mean": mean(wall_clocks),
            "min": min(wall_clocks),
            "max": max(wall_clocks),
        },
        "dispatch_overhead_ms": {
            "median": median(overheads),
            "mean": mean(overheads),
            "stdev": stdev(overheads) if len(overheads) > 1 else 0.0,
            "min": min(overheads),
            "max": max(overheads),
        },
        "overhead_pct": (median(overheads) / (per_node_latency_s * 1000)) * 100.0,
    }


async def main() -> None:
    print("TopologyRunner dispatch microbenchmark")
    print("=" * 60)
    print()
    print("Scenarios mirror SAGE's real topology shapes:")
    print("  • selfmoa / parallel_fanout: 3-5 parallel nodes")
    print("  • hub / debate:              5-7 parallel per round")
    print("  • hierarchical:              up to 11 in the widest layer")
    print()
    print("Per-node latency simulates a real LLM call (1-3 s typical).")
    print()

    scenarios = [
        # (n_nodes, per_node_latency_s, repetitions)
        (3, 1.0, 20),     # selfmoa pattern, fast LLM
        (5, 2.0, 10),     # parallel_fanout, medium LLM
        (7, 3.0, 5),      # hub/debate, slow reasoning LLM
        (11, 3.0, 3),     # hierarchical widest, slow
    ]

    rows: list[dict] = []
    for n, lat, reps in scenarios:
        print(f"  Running N={n}  latency={lat:.1f}s  reps={reps} ...", flush=True)
        result = await _run_single_batch(n, lat, reps)
        rows.append(result)

    print()
    print(
        f"{'N':>4} | {'latency':>8} | {'wall_ms':>10} | {'overhead_ms':>12} | "
        f"{'overhead_pct':>12}"
    )
    print("-" * 60)
    for r in rows:
        w = r["wall_clock_ms"]["median"]
        o = r["dispatch_overhead_ms"]["median"]
        pct = r["overhead_pct"]
        print(
            f"{r['n_nodes']:>4} | {r['per_node_latency_ms']:>6.0f} ms | "
            f"{w:>7.1f} ms | {o:>9.2f} ms | {pct:>10.3f}%"
        )

    print()
    print("Verdict heuristic:")
    max_pct = max(r["overhead_pct"] for r in rows)
    if max_pct < 1.0:
        print(
            f"  Max overhead observed: {max_pct:.3f}% — "
            "fan-out dispatch is NOT a bottleneck."
        )
        print("  Rust-ifying TopologyRunner.run() will give a marginal")
        print("  wall-clock gain that is invisible under a real LLM call.")
        print()
        print("  Higher-leverage Rust targets:")
        print("    • aggregation of streaming chunks across nodes")
        print("    • per-call JSON serde if running tool-heavy tasks")
        print("    • subprocess tool execution (already partly in sage-core)")
    elif max_pct < 5.0:
        print(
            f"  Max overhead observed: {max_pct:.3f}% — "
            "dispatch is measurable but not dominant."
        )
        print("  Rust would cap out at a ~5% wall-clock improvement per batch.")
    else:
        print(
            f"  Max overhead observed: {max_pct:.3f}% — "
            "fan-out dispatch IS material."
        )
        print("  Rust prototype justified. Next step: port _execute_node + the")
        print("  asyncio.gather loop to a tokio runtime, benchmark the same shapes.")


if __name__ == "__main__":
    asyncio.run(main())
