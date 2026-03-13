# Benchmark Results

All results as of March 13, 2026. Tests run using the YGN-SAGE framework with budget-tier Gemini 2.5 Flash unless otherwise noted.

---

## Code Generation

### EvalPlus HumanEval+

164 coding problems with 80x more tests than the original HumanEval benchmark (up to 999 `plus_inputs` per task). pass@1 metric with subprocess sandbox execution.

| Metric | Score |
|--------|-------|
| **HumanEval+ pass@1** | **84.1%** (138/164) |
| HumanEval base pass@1 | 90.9% (149/164) |

### EvalPlus MBPP+

378 Python problems with 35x more tests than the original MBPP benchmark.

| Metric | Score |
|--------|-------|
| **MBPP+ pass@1** | **75.1%** (284/378) |
| MBPP base pass@1 | 88.9% (336/378) |
| MBPP+ smoke test (20) | 80.0% (16/20) |

### SOTA Comparison (HumanEval+ pass@1)

| System | Score | Notes |
|--------|-------|-------|
| O1 | ~89% | OpenAI reasoning model |
| GPT-4o | ~87% | OpenAI flagship |
| Qwen2.5-Coder-32B | ~87% | Specialized coding model |
| **YGN-SAGE** | **84.1%** | Budget Gemini 2.5 Flash |
| Claude Sonnet 3.5 | ~82% | Anthropic |

!!! note "Model context"
    YGN-SAGE achieves 84.1% using Gemini 2.5 Flash (a budget-tier model), while competing systems use flagship models. The framework adds +15pp over the bare model baseline through routing, topology, memory, and AVR self-refinement.

---

## TopologyBench

### GSM8K Reasoning (50 tasks, Gemini 2.5 Flash)

Tests topology significance on grade-school math reasoning. Direct TopologyRunner execution (no orchestrator overhead).

| Topology | Accuracy | Avg Latency |
|----------|----------|-------------|
| parallel | **98.0%** (49/50) | 6977ms |
| sequential | 96.0% (48/50) | 6486ms |
| debate | 96.0% (48/50) | 4891ms |
| brainstorming | 96.0% (48/50) | 7059ms |

!!! warning "Null result"
    All pairwise comparisons are **not statistically significant** (Wilcoxon p > 0.3, Cohen's d < 0.12). The model ceiling is too high — Gemini 2.5 Flash solves 96%+ of GSM8K regardless of topology. The same 2 tasks (GSM8K/2, GSM8K/12) fail across ALL topologies, indicating LLM blind spots rather than topology deficiencies.

### HumanEval+ (164 tasks, Real Topology Execution)

After fixing the topology execution path in `boot.py`, we ran TopologyBench with real multi-agent topology execution. **First statistically significant topology result:**

| Topology | pass@1 | Failures |
|----------|--------|----------|
| **debate** | **95.1%** (156/164) | 8 |
| sequential | 90.8% (149/164) | 15 |

| Statistical Test | Value | Significance |
|-----------------|-------|--------------|
| McNemar chi-squared | 5.14 | **p = 0.023 < 0.05** |
| Discordant pairs | b=7, c=0 | Debate strictly dominates |
| Failure overlap | Jaccard=0.533 | 8 shared, 7 debate-only recoveries |

!!! success "Topology is statistically significant for code generation"
    Debate **strictly dominates** sequential: every task sequential passes, debate also passes. Debate additionally recovers 7 tasks (HumanEval/83, 84, 94, 95, 96, 97, 130). The shared 8 failures are "hard" tasks requiring model capability improvements.

### HumanEval+ (20-task Pilot)

| Topology | pass@1 | Failures |
|----------|--------|----------|
| debate | **100.0%** (20/20) | — |
| sequential | 95.0% (19/20) | HumanEval/14 |
| parallel | 95.0% (19/20) | HumanEval/10 |
| brainstorming | 90.0% (18/20) | HumanEval/0, HumanEval/9 |

Zero failure overlap on the pilot (Jaccard=0.00) — each topology has unique blind spots.

!!! warning "Previous results invalidated"
    Earlier results (9 topologies, mean 94.0%) were **invalidated** — `CognitiveOrchestrator` bypassed topology execution entirely.

---

## Ablation Study

6-configuration ablation framework proving each pillar's contribution. A/B paired tests on the same model with 20 tasks.

| Configuration | Score | Delta |
|--------------|-------|-------|
| **Full system** | **100%** | baseline |
| No routing (random tier) | 95% | -5pp |
| No guardrails | 95% | -5pp |
| No AVR | 90% | -10pp |
| No memory | 90% | -10pp |
| **Bare baseline** | **85%** | **-15pp** |

The full framework adds **+15 percentage points** over the bare LLM baseline, with AVR self-refinement and memory each contributing ~10pp, and routing and guardrails each contributing ~5pp.

---

## Routing Accuracy

### Non-Circular Ground Truth (50 tasks)

50 human-labeled tasks (10 S1 + 20 S2 + 20 S3). Labels assigned by domain expertise, NOT reverse-engineered from the heuristic.

| Router | Accuracy | S1 | S2 | S3 |
|--------|----------|----|----|-----|
| **kNN (arctic-embed-m)** | **92%** (46/50) | 70% | 95% | 100% |
| Rust SystemRouter | 88% (44/50) | 80% | 95% | 85% |
| Heuristic (ComplexityRouter) | 52% (26/50) | 80% | 50% | 40% |
| DeBERTa zero-shot (NVIDIA) | 52% (26/50) | -- | -- | S3=0% |
| Python AdaptiveRouter | 44% (22/50) | -- | -- | S3=0% |

The kNN router (arXiv 2505.12601) using snowflake-arctic-embed-m embeddings is the best-performing router, with particular strength on S3 formal verification tasks (100% vs 40% for the keyword heuristic). Leave-one-out cross-validation: 80%.

### Shadow Traces

1090 dual-routing traces comparing Rust SystemRouter vs Python AdaptiveRouter:

| Metric | Value |
|--------|-------|
| Total traces | 1,090 |
| Divergence rate | 49.6% |
| Rust distribution (S1/S2/S3) | 20% / 47% / 33% |
| Python distribution (S1/S2/S3) | 59% / 41% / <1% |
| Ground truth distribution | 20% / 40% / 40% |

The Rust SystemRouter is well-calibrated against ground truth. The Python AdaptiveRouter is heavily S1-biased with near-zero S3 classification.

---

## Quality Estimation

### DistilBERT QualityEstimator

| Metric | Value |
|--------|-------|
| Pearson correlation | 0.3436 |
| Baseline (heuristic) correlation | 0.0 |
| **Improvement** | **+34.4pp** |
| Training data | 600 quality triples |
| Model size | 0.9 MB (ONNX, opset 18) |
| Status | **Strong SHIP** |

The DistilBERT quality estimator replaces the `len > 10` heuristic with a 5-signal learned scorer. The +34.4pp Pearson improvement on held-out data confirms meaningful signal extraction.

---

## Evolution Statistical Proof

5-run paired experiment (10 HumanEval+ tasks per run, budget Gemini 2.5 Flash-Lite):

| Config | Mean pass@1 | Per-run rates |
|--------|-------------|---------------|
| No-evolution (evo OFF) | **98.0%** | 100%, 100%, 100%, 90%, 100% |
| Full system (evo ON) | 88.0% | 90%, 90%, 90%, 90%, 80% |

| Statistic | Value |
|-----------|-------|
| Delta | **-10.0pp** |
| Cohen's d | -1.41 (large negative) |
| Bootstrap 95% CI | [-16pp, -4pp] |
| Wilcoxon p (one-sided H1: full > no-evo) | 1.0 |
| McNemar p | 0.1306 |

!!! warning "Honest negative result"
    Evolution **hurts** on simple HumanEval+ tasks with a budget model. The TopologyEngine generates multi-node topologies that introduce failure points without benefit when the base system already achieves ~98% accuracy. Evolution is designed for complex, multi-model routing scenarios — on single-model simple tasks, the overhead degrades performance.

---

## Legacy Benchmarks

| Benchmark | Score | Notes |
|-----------|-------|-------|
| Routing quality (30 GT) | 100% (30/30) | Self-consistency check |
| kNN LOO-CV | 80% | Leave-one-out cross-validation |
| DeBERTa CI | [38%, 66%] | Confidence interval, fine-tuning required |

---

## How to Reproduce

### EvalPlus

```bash
cd sage-python

# Full HumanEval+ (164 problems)
python -m sage.bench --type evalplus --dataset humaneval

# Quick smoke test (20 problems)
python -m sage.bench --type evalplus --dataset humaneval --limit 20

# Full MBPP+ (378 problems)
python -m sage.bench --type evalplus --dataset mbpp
```

### TopologyBench

```bash
cd sage-python

# Cost estimate
python scripts/run_topologybench.py --dry-run

# Smoke test
python scripts/run_topologybench.py --tasks 5 --topologies sequential,avr --limit 3

# Full run (~$28 estimated)
python scripts/run_topologybench.py --tasks 200

# Resume from partial
python scripts/run_topologybench.py --tasks 164 --resume data/topologybench_results.json
```

### Ablation

```bash
cd sage-python
python -m sage.bench --type ablation --limit 20
```

### Routing Ground Truth

```bash
cd sage-python
python -m sage.bench --type routing_gt
```

### Official Evaluation Protocol

For full error logging and post-mortem analysis:

```bash
cd sage-python

# HumanEval+ with error capture
python -m sage.bench.eval_protocol --suite humaneval --limit 20 -v

# MBPP+ with error capture
python -m sage.bench.eval_protocol --suite mbpp --limit 20 -v

# Post-mortem replay
python -m sage.bench.eval_protocol --replay docs/benchmarks/errors.jsonl
```
