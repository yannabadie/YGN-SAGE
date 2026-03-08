# Benchmarks

Benchmark suite for measuring YGN-SAGE agent performance across coding tasks and routing accuracy.

## Modules

### `runner.py` -- BenchmarkRunner

Core benchmark infrastructure. Defines `BenchmarkRunner` (orchestrates benchmark execution), `BenchReport` (aggregated results with pass rate, latency, cost), and `TaskResult` (per-problem outcome). Supports baseline mode for reproducible comparisons.

### `humaneval.py` -- HumanEval Benchmark

Implements the OpenAI HumanEval benchmark: 164 Python programming problems evaluated via pass@1. Each problem is executed in a subprocess sandbox. Results include per-problem pass/fail, latency, and cost. Problem data is bundled in `humaneval_data.json`.

### `routing.py` -- Routing Accuracy Benchmark

Measures the ComplexityRouter's classification precision across 30 labeled tasks: 10 S1 (simple/fast), 10 S2 (reasoning), and 10 S3 (formal verification). Runs locally without API keys. Reports per-tier accuracy and confusion matrix.

### `truth_pack.py` -- Truth Pack

`BenchmarkManifest` for reproducible benchmarks. Generates per-task JSONL traces with timestamps, model used, routing tier, and raw outputs. Enables before/after comparison across code changes.

### `__main__.py` -- CLI Entry Point

CLI interface invoked via `python -m sage.bench`. Supports `--type routing|humaneval`, `--limit N` for partial runs, and `--baseline` for evidence-first comparison mode.

## Usage

```bash
python -m sage.bench --type routing                  # Instant, no API key
python -m sage.bench --type humaneval --limit 20     # Smoke test (20 problems)
python -m sage.bench --type humaneval                # Full 164 problems
```

## Data Files

- `humaneval_data.json` -- 164 HumanEval problems (prompt, entry point, test cases)
