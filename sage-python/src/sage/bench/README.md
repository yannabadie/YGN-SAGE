# Benchmarks

Benchmark suite for measuring YGN-SAGE agent performance. **Primary benchmarks are BigCodeBench Hard and ablation study** — HumanEval+ is saturated (99%+ SOTA) and does not measure framework value.

## Invocation

```bash
# Primary (use these to prove framework value)
python -m sage.bench --type routing_gt                            # kNN routing GT (60-task, instant)
python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 20
python -m sage.bench --type ablation --limit 10 --tier budget    # 6-config ablation (BCB-Hard)
python -m sage.bench --type ablation --limit 50 --tier budget    # Full A3 gate run

# SWE-bench (observe mode is default for every smoke)
SAGE_DIFF_VERIFIER_MODE=observe \
  python -m sage.bench --type swebench --dataset lite --limit 10 \
    --output docs/benchmarks/$(date +%F)-observe.json

# Optional: older benchmarks
python -m sage.bench --type humaneval --limit 20    # HumanEval (saturated — not representative)
python -m sage.bench --type evalplus --dataset humaneval
python -m sage.bench --type apps --limit 10 --difficulty interview
```

## Modules

### `bigcodebench_bench.py` — BigCodeBench Benchmark

Primary benchmark. Hard split: 148 tasks, non-saturated, complex library usage. Instruct split (default): NL task description → complete function. Calls official Docker evaluation via `bigcodebench.evaluate` for verified pass@1. Internal evaluation available for fast smoke runs. Results go to `docs/benchmarks/`.

### `ablation.py` — Ablation Framework

6-config ablation study over BCB-Hard tasks:
- `full` — all cognitive pillars active (reference)
- `baseline` — bare LLM call, pipeline disabled
- `no-memory` — memory tier disabled
- `no-avr` — adaptive topology disabled
- `no-routing` — kNN/Rust router disabled (heuristic only)
- `no-guardrails` — guardrails disabled

Gate: ≥4/10 PASS on `full` config → proceed to A3 N=50. Use `--tier budget` for deepseek-v4-flash.

### `__main__.py` — CLI Entry Point

`python -m sage.bench` dispatches all bench types. Key flags:
- `--type` — bench type (routing_gt, bigcodebench, ablation, swebench, evalplus, humaneval, apps, gaia)
- `--limit N` — cap task count
- `--tier budget|fast|reasoner` — model tier for bench run
- `--output PATH` — report JSON output
- `--subset hard|complete` — BCB subset (default: hard)
- `--split instruct|complete` — BCB split (default: instruct)
- `--dataset lite|verified` — SWE-bench dataset

### `swebench_bench.py` — SWE-bench Benchmark

All-5-pillars benchmark. Generates patches via the full agent pipeline + pre-emission diff-context verifier (`SAGE_DIFF_VERIFIER_MODE=observe` annotates predictions.jsonl with `_diff_verifier_mismatches`). Docker evaluation via official SWE-bench harness. Gen log at `<output-stem>-gen.log`.

### `routing_gt.py` — Routing Ground Truth

60-task stratified set (20 S1/20 S2/20 S3, human-labeled 2026-03-11). Measures kNN router accuracy vs Rust SystemRouter vs heuristic fallback — historic figures (kNN ~92%, SystemRouter ~88%, heuristic ~34%) are non-autoritative; current authoritative status: `routing.knn_92pct` / `routing.system_router_88pct` `evidence_pending` in `docs/CLAIMS.yaml`. No API keys needed. Runs in seconds.

### `runner.py` — BenchmarkRunner

Core infrastructure: `BenchmarkRunner`, `BenchReport`, `TaskResult`. Supports `baseline_mode` for before/after comparison, `truth_pack` for JSONL traces.

### `humaneval.py` / `evalplus_bench.py`

Legacy benchmarks. HumanEval is saturated (99%+ SOTA, 0 framework signal). Use BCB Hard instead.

### `routing_quality.py` / `routing_downstream.py`

Extended routing accuracy (45-task set) and downstream quality metrics: tier precision, escalation rate, latency P50/P99.

### `swebench_patch_repair.py`

Two-stage patch repair for SWE-bench: (1) LLM one-shot repair of malformed hunks, (2) context-mismatch repair guided by diff-context verifier mismatches.

## Data Files

- `humaneval_data.json` — 164 HumanEval problems (bundled; not the primary bench)

## Bench Signals to Monitor

```bash
grep -c "PASS" output.log              # tasks passed
grep -c "search_repo\|read_file" log   # tool calls (should be 0 for BCB ablation)
grep -c "120.*timeout\|TimeoutError" log  # timeouts (should be 0 with proper boot)
grep -c "oracle_verdict.*trainable" log   # learning signals emitted
```
