# SWE-bench Ablation Protocol (Sprint 5)

**Status:** Protocol defined, runner shipped, execution gated on Docker + API budget.
**Date:** 2026-04-17

## Goal

Measure the marginal contribution of each SAGE "framework delta" component
on SWE-bench by ablating components one at a time and comparing pass rates.
This answers: *does each pillar earn its place, or is a subset sufficient?*

## Configurations (4)

| # | Name | Flags | What stays on |
|---|------|-------|----------------|
| 1 | `full` | *(none)* | topology + ToolForge + sage_recurse + kNN + bandit + formal verification |
| 2 | `no_sage_recurse` | `SAGE_ABLATION_NO_RECURSE=1` | full minus recursive self-invocation |
| 3 | `no_toolforge` | `SAGE_ABLATION_NO_TOOLFORGE=1` | full minus autonomous tool synthesis |
| 4 | `bare` | `SAGE_ABLATION_NO_{RECURSE,TOOLFORGE,TOPOLOGY}=1` | LLM + basic tools only, no topology |

The `bare` config is the baseline for measuring total framework delta.
`full` is the headline number. The two middle configs isolate each
capability.

## Running

```bash
# Small smoke (5 tasks per config) — ~20 LLM calls, ~5 min per config with the
# reasoner tier; ~$1-2 total.
python scripts/run_swebench_ablation.py \
    --dataset lite --limit 5 --tier reasoner \
    --out docs/benchmarks/

# Full Pro run (50 tasks) — ~$30-60, ~3h with parallel Docker evaluation.
# REQUIRES Docker Desktop + WSL2 backend on Windows.
python scripts/run_swebench_ablation.py \
    --dataset pro --limit 50 --tier reasoner \
    --out docs/benchmarks/
```

The runner writes:
- One predictions JSONL per config (generate-only).
- One bench report JSON per config.
- One ablation summary JSON: `{date}-swebench-{dataset}-ablation.json`.

Docker evaluation is a separate step (the generate step runs on any OS;
the evaluator needs Linux containers):

```bash
python -m swebench.harness.run_evaluation \
    --predictions_path docs/benchmarks/2026-04-17-swebench-pro-predictions-full.jsonl \
    --dataset_name ScaleAI/SWE-bench_Pro \
    --run_id sage-ablation-full-20260417
```

## Gates

Before claiming anything from the ablation, require:

1. **Zero regressions in Python tests** (`pytest tests/ -q`) on `main`.
2. **Reproducibility**: same commit SHA, same dataset split, same seed where
   applicable. Record the git SHA in the ablation JSON.
3. **Minimum N**: `full` vs `bare` must use **≥20 tasks** before quoting a
   delta. Lite N=5 is smoke-test only.
4. **Statistical test**: McNemar's test on paired pass/fail vectors between
   `full` and each ablated config. p < 0.05 + Cohen's d > 0.2 to claim that
   a component is "load-bearing."
5. **Cost tracking**: log per-task `total_cost_usd` — the `full` config
   must not be >3× the `bare` config to be credible.

## Interpretation matrix

| Outcome pattern | Interpretation |
|-----------------|----------------|
| `full > no_toolforge ≈ no_recurse > bare` | topology is the dominant contributor |
| `full ≈ no_toolforge > no_recurse ≈ bare` | recursion is the dominant contributor |
| `full ≈ no_recurse > no_toolforge ≈ bare` | ToolForge is the dominant contributor |
| `full > no_toolforge > no_recurse > bare` | all three contribute, monotonic |
| `full ≈ bare` | none of the framework components helped — scope miss |

Any pattern where `full < bare` is a regression to investigate before
shipping Sprint 6.

## What this sprint does NOT deliver

- Running the ablation on a real Docker host. Scheduled for the user's
  next session with API budget + Docker up.
- Statistical significance testing — runner collects paired outcomes but
  the analysis script (McNemar + bootstrap CIs) lives in `sage.bench.ablation`
  and is not yet plumbed through for SWE-bench specifically.
- SWE-bench Pro official leaderboard submission. That happens after
  Sprint 6 decision.
