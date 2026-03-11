# Benchmarks

Benchmark suite for measuring YGN-SAGE agent performance across coding tasks and routing accuracy.

## Modules

### `runner.py` -- BenchmarkRunner

Core benchmark infrastructure. Defines `BenchmarkRunner` (orchestrates benchmark execution), `BenchReport` (aggregated results with pass rate, latency, cost), and `TaskResult` (per-problem outcome). Supports baseline mode for reproducible comparisons.

### `humaneval.py` -- HumanEval Benchmark

Implements the OpenAI HumanEval benchmark: 164 Python programming problems evaluated via pass@1. Each problem is executed in a subprocess sandbox. Results include per-problem pass/fail, latency, and cost. Problem data is bundled in `humaneval_data.json`.

### `routing.py` -- Routing Accuracy Benchmark

Measures the ComplexityRouter's classification precision across 30 labeled tasks: 10 S1 (simple/fast), 10 S2 (reasoning), and 10 S3 (formal verification). Runs locally without API keys. Reports per-tier accuracy and confusion matrix.

### `routing_quality.py` -- Routing Quality Benchmark

Extended routing accuracy with 45 labeled tasks for both ComplexityRouter and AdaptiveRouter ground truth evaluation.

### `routing_downstream.py` -- Downstream Quality Evaluator

DownstreamEvaluator: tracks tier precision, escalation rate (<20% target), routing P50/P99 latency (<50ms target), and quality metrics per routing decision.

### `evalplus_bench.py` -- EvalPlus Adapter

Official EvalPlus HumanEval+ (164 problems, 80x harder tests with up to 999 plus_inputs) and MBPP+ (378 problems, 35x harder). Subprocess sandbox evaluator, Windows-compatible.

### `ablation.py` -- Ablation Framework

6-configuration ablation study: full, baseline, no-memory, no-avr, no-routing, no-guardrails. Quantifies each cognitive pillar's contribution vs bare LLM baseline.

### `eval_protocol.py` -- Official Evaluation Protocol

Comprehensive benchmark evaluation with full error logging. Captures every error with traceback, phase, model ID, routing decision, and timing. Produces structured JSON reports and JSONL error logs for post-mortem analysis. Supports error replay for debugging.

### `truth_pack.py` -- Truth Pack

`BenchmarkManifest` for reproducible benchmarks. Generates per-task JSONL traces with timestamps, model used, routing tier, and raw outputs. Enables before/after comparison across code changes.

### `__main__.py` -- CLI Entry Point

CLI interface invoked via `python -m sage.bench`. Supports `--type routing|humaneval|evalplus|ablation`, `--dataset humaneval|mbpp` (for evalplus), `--limit N` for partial runs, and `--baseline` for evidence-first comparison mode.

## Usage

```bash
python -m sage.bench --type routing                        # Routing self-consistency (instant)
python -m sage.bench --type humaneval --limit 20           # Legacy HumanEval (20 problems)
python -m sage.bench --type evalplus --dataset humaneval   # EvalPlus HumanEval+ (80x harder)
python -m sage.bench --type evalplus --dataset mbpp        # EvalPlus MBPP+ (35x harder)
python -m sage.bench --type ablation --limit 20            # Ablation study (6 configs)
python -m sage.bench.eval_protocol --suite humaneval -v    # Official evaluation protocol
python -m sage.bench.eval_protocol --replay errors.jsonl   # Post-mortem error analysis
```

## Data Files

- `humaneval_data.json` -- 164 HumanEval problems (prompt, entry point, test cases)
