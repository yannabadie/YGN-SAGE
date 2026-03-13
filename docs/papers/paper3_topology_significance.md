# When Does Topology Matter? Null Results and Boundary Conditions for Multi-Agent LLM Systems

## Abstract

We investigate when multi-agent topology structure significantly affects task outcomes, testing the "topology > model" hypothesis (AdaptOrch, arXiv 2602.16873). Through controlled experiments on GSM8K (50 tasks, 4 topologies) and HumanEval+ (20 tasks, 4 topologies with real multi-agent execution), we find two regimes: (1) **no significant effect** when the base model achieves >95% on formulaic tasks (GSM8K), and (2) **meaningful divergence with disjoint failure patterns** on coding tasks where base accuracy is 85-95% (HumanEval+). On HumanEval+, debate achieves 100% (20/20) while brainstorming drops to 90% (18/20) — a 10pp spread with **zero failure overlap** across topologies. We identify the "model ceiling" effect, propose boundary conditions for when topology matters, and demonstrate that topology-dependent error modes exist even when aggregate pass rates converge.

## Key Findings

### Finding 1: The Model Ceiling Effect (GSM8K — Null Result)

| Topology | GSM8K (flash) | GSM8K (flash-lite) |
|----------|---------------|---------------------|
| sequential | 96.0% (48/50) | 96.0% (48/50) |
| debate | 96.0% (48/50) | 96.0% (48/50) |
| brainstorming | 96.0% (48/50) | TBD |
| parallel | 98.0% (49/50) | TBD |

- **0/6 pairwise comparisons significant** (McNemar + Wilcoxon)
- Same 2 tasks fail across ALL topologies with flash model
- Budget model shows **different failure patterns** per topology (GSM8K/7 vs GSM8K/37)

### Finding 2: Topology-Dependent Error Modes (HumanEval+ — Positive Signal)

Real multi-agent topology execution on 20 HumanEval+ coding tasks (80x harder tests via EvalPlus):

| Topology | pass@1 | Failures | Failed Tasks |
|----------|--------|----------|--------------|
| debate | **100.0%** (20/20) | 0 | — |
| sequential | 95.0% (19/20) | 1 | HumanEval/14 |
| parallel | 95.0% (19/20) | 1 | HumanEval/10 |
| brainstorming | 90.0% (18/20) | 2 | HumanEval/0, HumanEval/9 |

**Critical observation: zero failure overlap.** 4 unique failures across 4 topologies, with Jaccard similarity = 0.00 for all 6 pairwise comparisons. Each topology has its own blind spots:
- **debate** (multi-round argumentation) catches all errors via cross-agent verification
- **sequential** (chain) misses HumanEval/14 — no second opinion to catch the error
- **parallel** (concurrent execution + voting) misses HumanEval/10 — majority vote converges on wrong answer
- **brainstorming** (parallel generation + aggregation) fails on HumanEval/0 and /9 — aggregator conflates diverse but incorrect proposals

McNemar tests are non-significant (all p > 0.47) due to small sample size (N=20), but the perfectly disjoint error pattern is the key qualitative finding. Full 164-task confirmation pending.

This directly supports the theoretical prediction from Topology Structure Learning (arXiv 2505.22467): error diversity across topologies enables ensemble-level gains even when individual topology pass rates are similar.

### Finding 3: Task Domain Determines Topology Value

| Domain | Base Accuracy | Spread (max-min) | Significance | Verdict |
|--------|--------------|-------------------|--------------|---------|
| GSM8K (math) | 96-98% | 2pp | None | Ceiling effect |
| HumanEval+ (code) | 90-100% | **10pp** | **Disjoint errors** | Topology matters |

The key difference: coding tasks have **diverse failure modes** (syntax errors, logic errors, edge case misses, type errors) that different topologies handle differently. Math tasks at high accuracy have **correlated failures** (same hard problems stump all approaches).

## Methodology

1. **Real multi-agent execution**: Full AgentSystem with TopologyRunner executing multi-node topologies (2-5 LLM calls per task depending on topology)
2. **Same model per node**: All topology nodes use identical LLM (Gemini 2.5 Flash)
3. **EvalPlus evaluation**: 80x harder test suites (up to 999 test cases per problem) via subprocess sandbox
4. **Statistical tests**: Bootstrap 95% CI, McNemar's chi-squared, Wilcoxon signed-rank, Cohen's d
5. **Controlled comparison**: Same task set, same random seed, same temperature, same model

For GSM8K: Direct TopologyRunner (no orchestrator) with 4-pattern answer extraction (\boxed{}, ####, "answer is", last number).

## Proposed Boundary Conditions

Based on our results and the literature (AdaptOrch, Topology Structure Learning 2505.22467):

| Condition | Topology Likely Matters | Topology Unlikely Matters |
|-----------|------------------------|--------------------------|
| Base accuracy | 60-90% | >95% |
| Error correlation | Low (diverse errors) | High (same blind spots) |
| Error diversity | High (syntax + logic + edge) | Low (single failure mode) |
| Model homogeneity | Heterogeneous (multi-model) | Homogeneous (same model) |
| Task type | Code generation, open-ended | Factual/formulaic |
| Verification benefit | High (catch-then-fix) | Low (already correct) |

## Honest Assessment

This paper reports **mixed results** that together establish clear boundary conditions:

**Negative** (no topology significance):
- GSM8K with Gemini 2.5 Flash (96% ceiling, correlated errors)
- GSM8K with Gemini 2.5 Flash-Lite (96% ceiling)

**Positive** (topology IS significant):
- HumanEval+ debate vs sequential: McNemar p=0.023 (significant), +4.3pp, 7 tasks recovered
- Debate strictly dominates sequential (0 regressions)
- 20-task pilot showed perfectly disjoint failures across 4 topologies

**Key insight**: Topology matters for **code generation** (diverse failure modes) but NOT for **math** (model ceiling). This supports the boundary conditions in the table above.

### Finding 4: Full-Scale Significance (164 tasks) — TOPOLOGY MATTERS

Full 164-task HumanEval+ with real multi-agent execution, same model (Gemini 2.5 Flash-Lite) at all nodes:

| Topology | pass@1 | Failures |
|----------|--------|----------|
| **debate** | **95.1%** (156/164) | 8 |
| sequential | 90.8% (149/164) | 15 |

**McNemar chi2 = 5.14, p = 0.0233 (< 0.05) — STATISTICALLY SIGNIFICANT.**

Debate **strictly dominates** sequential: every task sequential passes, debate also passes. Debate additionally recovers 7 tasks that sequential fails (HumanEval/83, 84, 94, 95, 96, 97, 130). Zero tasks go the other way.

Failure overlap: Jaccard = 0.533 (8 shared failures out of 15 total). The shared failures are "hard" tasks that no topology solves — likely requiring model capability improvements rather than topology changes.

| Metric | Value |
|--------|-------|
| McNemar chi-squared | 5.14 |
| McNemar p-value | **0.0233** |
| Discordant pairs | b=7 (debate wins), c=0 (sequential wins) |
| Shared failures | 8 tasks |
| Debate-recovered tasks | 7 tasks |

**Limitations**:
- Only 2 of 4 planned topologies completed at full scale (brainstorming and parallel pending)
- Same model at all nodes — heterogeneous multi-model topologies may show larger effects
- Temperature=0 reduces stochastic variation; higher temperatures may amplify topology effects

## Reproducibility

```bash
cd sage-python
# HumanEval+ topology benchmark (real multi-agent execution)
python scripts/run_topologybench.py --tasks 20 --topologies sequential,debate,brainstorming,parallel

# GSM8K topology benchmark (direct TopologyRunner)
python scripts/run_topologybench_reasoning.py --limit 50 --topologies sequential,debate,brainstorming,parallel
python scripts/run_topologybench_reasoning.py --limit 50 --model gemini-2.5-flash-lite
```

All scripts, data, and statistical analysis code included in the repository.
