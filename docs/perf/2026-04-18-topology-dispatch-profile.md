# Perf study — is TopologyRunner dispatch a Rust-rewrite target?

**Date**: 2026-04-18
**Question**: user suggested Rust-ifying `TopologyRunner.run()` in sage-python
as a follow-up to directive #1 (Rust-first). The claim: "supprimer le saut
GIL Python↔Rust au fan-out."
**Method**: microbenchmark the Python orchestration layer with a synthetic
LLMProvider that sleeps `L` seconds per call. Everything above
`max(L_i)` over a concurrent batch is dispatch + bookkeeping overhead.
Script: `tests/perf/profile_topology_runner_dispatch.py`.

## Result

| N nodes | per-node latency | wall-clock | dispatch overhead | overhead % |
|---------|------------------|------------|-------------------|------------|
| 3 | 1 s | 1012 ms | 12 ms | **1.21 %** |
| 5 | 2 s | 2014 ms | 14 ms | 0.70 % |
| 7 | 3 s | 3014 ms | 15 ms | 0.50 % |
| 11 | 3 s | 3013 ms | 13 ms | 0.44 % |

Dispatch overhead is **~13-15 ms per batch regardless of N**, dominated by
AsyncMock allocation + `asyncio.gather` orchestration. On real LLM calls
(2-30 s per node), the percentage floor is **0.05-0.5 %**.

## Conclusion

Rust-ifying `TopologyRunner.run()` is **not a leveraged target**. A full
rewrite would eliminate at most ~15 ms per topology batch. A typical
SWE-bench task has 5-10 batches → 75-150 ms saved out of 100-300 s
total task time = **0.05 % wall-clock improvement**.

Why: asyncio releases the GIL on every await, so GIL-pessimism on
network-IO fan-out is a myth. Python's overhead on `gather(*coros)` is
the coroutine-object creation + state-machine dispatch — single-digit
microseconds per coroutine. The per-node work (HTTP to OpenAI, tool
execution, JSON parsing) dwarfs it.

## Where Rust DOES pay off (already applied)

Directive #1 is not being ignored — it's already pointed at the right
spots:

- **ModelAssigner** — Rust `sage-core/src/routing/model_assigner.rs`.
  Scoring over N models × M profiles, CPU-bound.
- **QualityLabeler / Z3** — Rust `sage-core/src/quality/`. Formal
  verification, CPU-bound.
- **SystemRouter + kNN** — Rust `sage-core/src/routing/`. Embedding
  similarity, CPU-bound.
- **Topology generation (6-path + MAP-Elites)** — Rust
  `sage-core/src/topology/`. Graph search, CPU-bound.
- **Tree-sitter sandbox** — Rust `sage-core/src/sandbox/`. AST parsing,
  CPU-bound.

These all have measurable wins because they run tight loops that Python
stalls on under the GIL.

## Alternative Rust investments we SHOULD consider

Ordered by expected leverage, each with a concrete signal that would
un-park it:

1. **Rust LiteLLM replacement** — `litellm-rs` (crates.io): not yet
   (see `docs/perf/2026-04-18-litellm-rs-review.md` for the decline
   rationale). Revisit when litellm-rs hits v1.0 + xAI provider +
   per-call `response_cost`.
2. **Streaming chunk aggregator** — if we ever stream heavily across
   multiple nodes concurrently and need to merge token streams. Not
   today. Signal: "token/s per node" < 30% of provider's max.
3. **Ablation sweep runner** — the one script that actually loops
   O(100) candidates and does CPU-bound scoring. Only worth it if the
   sweep is a regular ops action, not a one-shot. Signal: we start
   running `scripts/ablate_*.py` more than once a week.

## Revisit trigger

Rerun this benchmark when ANY of the following changes:

- Real-LLM topology runs show a `_par_latency_ms - max_per_node_ms`
  > 5 % of wall-clock (check `ctx.tool_turn_count` + runtime trace).
- A `scripts/bench_topology_overhead.py` script lands that measures on
  live providers (not simulated) and shows the same <1 % result.
- Python < 3.13.12 (GIL-free mode default) adoption changes asyncio
  semantics enough to invalidate the measurement.

Until then, the dispatch layer stays Python. Directive #1 is satisfied
by the Rust work already in `sage-core`.
