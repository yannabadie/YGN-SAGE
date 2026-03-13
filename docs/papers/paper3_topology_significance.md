# When Does Topology Matter? Null Results and Boundary Conditions for Multi-Agent LLM Systems

## Abstract

We investigate when multi-agent topology structure significantly affects task outcomes, testing the "topology > model" hypothesis (AdaptOrch, arXiv 2602.16873). Through controlled experiments on GSM8K (50 tasks, 4 topologies) and HumanEval+ (pending), we find that topology choice has **no statistically significant effect** when the base model achieves >95% accuracy. All pairwise comparisons yield Cohen's d < 0.12 (negligible) and Wilcoxon p > 0.3. We identify the "model ceiling" effect: when a single LLM solves problems correctly on the first try, multi-agent debate, brainstorming, and parallel execution cannot improve upon it. We propose boundary conditions for when topology matters and suggest that heterogeneous multi-model topologies may be required for significance.

## Key Finding: The Model Ceiling Effect

| Topology | GSM8K (flash) | GSM8K (flash-lite) |
|----------|---------------|---------------------|
| sequential | 96.0% (48/50) | 96.0% (48/50) |
| debate | 96.0% (48/50) | 96.0% (48/50) |
| brainstorming | 96.0% (48/50) | TBD |
| parallel | 98.0% (49/50) | TBD |

- **0/6 pairwise comparisons significant** (McNemar + Wilcoxon)
- Same 2 tasks fail across ALL topologies with flash model
- Budget model shows **different failure patterns** per topology (GSM8K/7 vs GSM8K/37)

## Methodology

1. **Direct TopologyRunner execution**: No orchestrator, no routing, no memory — isolates pure topology effect
2. **Same model per node**: All topology nodes use identical LLM (Gemini 2.5 Flash or Flash-Lite)
3. **4-pattern answer extraction**: \boxed{}, ####, "answer is", last number
4. **Statistical tests**: Bootstrap 95% CI, McNemar's chi-squared, Wilcoxon signed-rank, Cohen's d
5. **Controlled comparison**: Same 50 GSM8K tasks, same random seed, same temperature

## Proposed Boundary Conditions

Based on our results and the literature (AdaptOrch, Topology Structure Learning 2505.22467):

| Condition | Topology Likely Matters | Topology Unlikely Matters |
|-----------|------------------------|--------------------------|
| Base accuracy | 60-80% | >90% |
| Error correlation | Low (diverse errors) | High (same blind spots) |
| Model homogeneity | Heterogeneous (multi-model) | Homogeneous (same model) |
| Task type | Ambiguous reasoning | Factual/formulaic |
| Verification benefit | High (catch-then-fix) | Low (already correct) |

## Honest Assessment

This paper reports primarily **negative results**. We did NOT find topology significance on:
- GSM8K with Gemini 2.5 Flash (96% ceiling)
- GSM8K with Gemini 2.5 Flash-Lite (96% ceiling)
- HumanEval+ with 9 topologies (previous results INVALIDATED; re-run pending)

The one positive signal: budget model shows topology-dependent failure patterns (different tasks fail under different topologies), suggesting topology CAN redirect error modes even when overall accuracy is unchanged.

## Reproducibility

```bash
cd sage-python
python scripts/run_topologybench_reasoning.py --limit 50 --topologies sequential,debate,brainstorming,parallel
python scripts/run_topologybench_reasoning.py --limit 50 --model gemini-2.5-flash-lite
```

All scripts, data, and statistical analysis code included in the repository.
