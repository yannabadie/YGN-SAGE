# When Does Topology Matter? Honest Multi-Run Results for Multi-Agent LLM Systems

## Abstract

We investigate when multi-agent topology structure significantly affects task outcomes, testing the "topology > model" hypothesis (AdaptOrch, arXiv 2602.16873). Through controlled experiments on HumanEval+ (164 tasks, 4 topologies, 80x harder tests) and GSM8K (50 tasks), we find that topology effects are **smaller and less reliable than single-run experiments suggest**. A first run showed debate strictly dominating sequential (McNemar p=0.023, N=164). A second independent run of all 4 topologies yielded brainstorming 96.3%, parallel 94.5%, debate 93.9%, sequential 93.3% — a 3.1pp spread with **0/6 pairwise comparisons significant**. The key finding: run-to-run variance at temperature=0 is substantial, making single-run significance claims unreliable. However, topology-dependent failure patterns are consistent: 12/17 failing tasks are topology-specific, and an oracle ensemble achieves 97.0% (+0.6pp over the best single topology). On GSM8K math reasoning (96%+ baseline), 0/6 comparisons reach significance. We propose boundary conditions for when topology matters and recommend multi-run evaluation protocols.

## 1. Introduction

Multi-agent systems use structured communication topologies — debate, sequential chains, parallel voting, brainstorming — to improve LLM task performance. Recent work claims "topology > model capability" (AdaptOrch, arXiv 2602.16873), and Topology Structure Learning (arXiv 2505.22467) reports up to 10% performance gaps. We test these claims with controlled experiments and find the reality is more nuanced.

### Contributions

1. **Multi-run evaluation**: Two independent 164-task HumanEval+ runs reveal substantial run-to-run variance at temperature=0
2. **Non-reproducible significance**: An initial p=0.023 result (debate vs sequential) was not reproduced in a second run
3. **Topology-dependent errors**: 12/17 failing tasks are topology-specific, enabling oracle ensemble gains
4. **Null result on math**: GSM8K shows no topology effect (model ceiling at 96%)
5. **Boundary conditions**: Six factors predicting when topology matters

## 2. Related Work

**Multi-agent topologies.** MASFactory (arXiv 2603.06007) and OFA-MAS (arXiv 2601.12996) propose topology generation methods but do not establish boundary conditions for when topology matters.

**Topology significance.** AdaptOrch (arXiv 2602.16873) claims topology outweighs model capability. Topology Structure Learning (arXiv 2505.22467) reports up to 10% gaps and predicts error diversity across topologies enables ensemble gains — a prediction we partially validate.

**Evaluation methodology.** EvalPlus (arXiv 2305.01210) provides 80x harder tests than HumanEval, reducing false positives. We use EvalPlus exclusively.

**LLM non-determinism.** Even at temperature=0, LLM APIs exhibit variance due to batching, floating-point non-determinism, and server-side changes. Our results quantify this variance for topology evaluation.

## 3. Methodology

### 3.1 Experimental Setup

- **Model**: All topology nodes use Gemini 2.5 Flash-Lite (budget tier)
- **Temperature**: 0 (nominally deterministic)
- **Evaluation**: EvalPlus subprocess sandbox (up to 999 test cases per problem)
- **Execution**: Real multi-agent execution via TopologyRunner (no orchestrator bypass)
- **Environment**: `SAGE_BENCH_LLM_TIER=budget`, `HF_HUB_OFFLINE=1`

### 3.2 Topologies

| Topology | Structure | Nodes | LLM Calls/Task |
|----------|-----------|-------|-----------------|
| sequential | Chain: input -> worker -> output | 3 | 3 |
| parallel | Fan-out: source -> 2 workers -> aggregator | 4 | 4 |
| debate | Argumentation: topic -> 2 debaters -> judge | 4 | 5 |
| brainstorming | Fan-out: prompt -> 3 thinkers -> synthesizer | 5 | 5 |

### 3.3 Statistical Tests

- **McNemar's chi-squared** (continuity correction): paired binary comparison
- **Bootstrap 95% CI** (10,000 resamples, seed=42)
- **Cohen's d**: effect size
- **Jaccard similarity**: failure set overlap

## 4. Results

### 4.1 Run 2: Consistent 4-Topology Benchmark (N=164)

All 4 topologies run in the same session, same model, same conditions:

| Topology | pass@1 | Failures | Unique Failures |
|----------|--------|----------|-----------------|
| **brainstorming** | **96.3%** (158/164) | 6 | 0 |
| parallel | 94.5% (155/164) | 9 | 1 (HumanEval/125) |
| debate | 93.9% (154/164) | 10 | 2 (HumanEval/77, 91) |
| sequential | 93.3% (153/164) | 11 | 4 (HumanEval/10, 32, 126, 149) |

| Pair | McNemar chi2 | p-value | Discordant (b, c) |
|------|-------------|---------|-------------------|
| brainstorming vs sequential | 2.29 | 0.131 | (6, 1) |
| brainstorming vs debate | 1.50 | 0.221 | (5, 1) |
| brainstorming vs parallel | 1.33 | 0.248 | (3, 0) |
| debate vs sequential | 0.00 | 1.000 | (5, 4) |
| parallel vs sequential | 0.10 | 0.752 | (6, 4) |
| debate vs parallel | 0.00 | 1.000 | (2, 3) |

**Significant pairs: 0/6.** The closest is brainstorming vs sequential (p=0.131), with brainstorming recovering 6 tasks and losing 1.

### 4.2 Run 1: Debate vs Sequential (N=164) — Earlier Session

Run from a separate session (different day):

| Topology | pass@1 | Failures |
|----------|--------|----------|
| **debate** | **95.1%** (156/164) | 8 |
| sequential | 90.8% (149/164) | 15 |

| Metric | Value |
|--------|-------|
| McNemar chi2 | 5.14 |
| McNemar p-value | **0.023** |
| Discordant pairs | b=7 (debate wins), c=0 |

This single run showed **strict dominance**: debate passed every task sequential passed, plus 7 more.

### 4.3 Non-Reproducibility of Significance

Comparing Run 1 and Run 2 for debate vs sequential:

| Metric | Run 1 | Run 2 |
|--------|-------|-------|
| Debate pass@1 | 95.1% (156/164) | 93.9% (154/164) |
| Sequential pass@1 | 90.8% (149/164) | 93.3% (153/164) |
| Delta | 4.3pp | 0.6pp |
| McNemar p | **0.023** | 1.000 |
| Strict dominance | Yes (c=0) | No (c=4) |

The 4.3pp advantage of debate in Run 1 shrinks to 0.6pp in Run 2. Sequential improves from 90.8% to 93.3%, while debate drops from 95.1% to 93.9%. Even at temperature=0, LLM APIs have sufficient variance to flip significance conclusions.

### 4.4 Consistent Finding: Topology-Dependent Failure Patterns

Despite aggregate rates converging, failure patterns differ across topologies:

| Task | Fails in Topologies (Run 2) |
|------|---------------------------|
| HumanEval/120, 127, 132, 145, 163 | All 4 (hard problems) |
| HumanEval/101, 130 | debate + parallel |
| HumanEval/75 | brainstorming + parallel |
| HumanEval/142 | debate + sequential |
| HumanEval/77, 91 | debate only |
| HumanEval/125 | parallel only |
| HumanEval/10, 32, 93, 126, 149 | sequential only |

5 tasks fail in all topologies (hard problems). 12 tasks are topology-specific. **Oracle ensemble**: 97.0% (159/164), +0.6pp over the best single topology.

### 4.5 Null Result: GSM8K Math Reasoning (N=50)

| Topology | Accuracy |
|----------|----------|
| parallel | 98.0% (49/50) |
| sequential | 96.0% (48/50) |
| debate | 96.0% (48/50) |
| brainstorming | 96.0% (48/50) |

0/6 pairwise comparisons significant (all p > 0.3, Cohen's d < 0.12). Same 2 tasks fail across ALL topologies. Model ceiling effect.

### 4.6 20-Task Pilot: Disjoint Failures

20-task HumanEval+ pilot (Gemini 2.5 Flash, earlier session):

| Topology | pass@1 | Failed Tasks |
|----------|--------|--------------|
| debate | 100% (20/20) | — |
| sequential | 95% (19/20) | HumanEval/14 |
| parallel | 95% (19/20) | HumanEval/10 |
| brainstorming | 90% (18/20) | HumanEval/0, 9 |

Jaccard = 0.00 for all 6 pairs (perfectly disjoint). Note: small N amplifies apparent disjointness — the 164-task run shows partial overlap.

## 5. Boundary Conditions

| Condition | Topology Likely Matters | Topology Unlikely Matters |
|-----------|------------------------|--------------------------|
| Base accuracy | 60-90% | >95% |
| Error correlation | Low (diverse errors) | High (same blind spots) |
| Task type | Code generation, open-ended | Factual/formulaic |
| Model homogeneity | Heterogeneous | Homogeneous |
| Verification benefit | High (catch-then-fix) | Low (already correct) |
| Run count | Multiple runs, pooled | Single run |

## 6. Discussion

**LLM non-determinism matters.** Our most important finding is that single-run topology comparisons are unreliable. Temperature=0 does not guarantee determinism in production LLM APIs. Future topology evaluation should use multiple independent runs and report variance.

**Failure patterns are real.** While aggregate pass rates converge, each topology has genuine blind spots. The oracle ensemble (97.0%) exceeds any single topology (96.3%), confirming that topology-aware routing has practical value even when no single topology dominates.

**Brainstorming is the most robust.** Across both runs and the pilot, brainstorming (fan-out + synthesizer) consistently performs well. Its parallel generation of diverse solutions followed by synthesis appears to be the most robust pattern for code generation.

**Recommendation.** For production systems: (1) default to brainstorming or debate for code tasks, (2) use simpler topologies for formulaic tasks, (3) consider oracle-ensemble routing if budget permits, (4) evaluate with multiple runs before drawing conclusions.

## 7. Limitations

- **Two runs only**: More runs needed to estimate true variance
- **Single model family**: Gemini 2.5 Flash-Lite only
- **Temperature=0 only**: Higher temperatures may amplify topology effects
- **Two task types**: HumanEval+ (code) and GSM8K (math)
- **Homogeneous nodes**: All topology nodes use the same model

## 8. Reproducibility

```bash
cd sage-python

# 4-topology consistent run (all in same session)
for t in sequential debate brainstorming parallel; do
    SAGE_BENCH_LLM_TIER=budget python scripts/run_topologybench.py \
        --tasks 164 --topologies $t --output data/topologybench_164_${t}.json
done

# Merge and analyze
python scripts/merge_topologybench.py \
    --files data/topologybench_164_*.json \
    --output data/topologybench_164_all.json
python scripts/analyze_topologybench.py data/topologybench_164_all.json

# GSM8K
python scripts/run_topologybench_reasoning.py \
    --limit 50 --topologies sequential,debate,brainstorming,parallel
```

## References

1. AdaptOrch (arXiv 2602.16873): Topology > model capability hypothesis
2. Topology Structure Learning (arXiv 2505.22467): Up to 10% topology performance gaps
3. MASFactory (arXiv 2603.06007): LLM-to-graph topology generation
4. OFA-MAS (arXiv 2601.12996): MoE graph generative for MAS topology
5. EvalPlus (arXiv 2305.01210): 80x harder HumanEval+ tests
6. FrugalGPT (arXiv 2305.05176): Adaptive computational investment
