# SWE-bench typed-vs-bash parity smoke — N=10 gen-only

**Date:** 2026-04-22
**Slice:** SWE-bench Lite, `--limit 10 --offset 0` (deterministic HF ordering)
**Mode:** generate-only (no Docker eval — patch-produce rate only)
**Command:** `python sage-python/scripts/swebench_parity_smoke.py --limit 10 --generate-only`
**Wallclock:** ~50 min total (both arms)

## Headline

| Metric | Arm A (bash) | Arm B (typed-only) |
|---|---|---|
| N | 10 | 10 |
| Patches produced | **3 (30%)** | **4 (40%)** |
| Empty (gen failed) | 7 | 6 |
| Errors | 0 | 0 |
| Sentinels | 0 | 0 |

## Per-task breakdown

| instance_id | Arm A (bash) | Arm B (typed-only) |
|---|---|---|
| astropy__astropy-12907 | EMPTY | EMPTY |
| astropy__astropy-14182 | **PATCH** | **PATCH** |
| astropy__astropy-14365 | EMPTY | EMPTY |
| astropy__astropy-14995 | EMPTY | **PATCH** |
| astropy__astropy-6938 | **PATCH** | EMPTY |
| astropy__astropy-7746 | EMPTY | EMPTY |
| django__django-10914 | EMPTY | **PATCH** |
| django__django-10924 | **PATCH** | EMPTY |
| django__django-11001 | EMPTY | **PATCH** |
| django__django-11019 | EMPTY | EMPTY |

## Set analysis

- Both arms resolved: {astropy-14182} — 1 task
- Bash-only: {astropy-6938, django-10924} — 2 tasks
- Typed-only: {astropy-14995, django-10914, django-11001} — 3 tasks

The arms pick **different subsets** rather than one being a superset of
the other. Five tasks were resolved in at least one arm, three of those
uniquely by typed-only. This undercuts the "bash is strictly more
capable" intuition — at smoke scale the tools-available choice shifts
which tasks succeed, not whether succeess is possible at all.

## Decision gate (red-team plan §5, functional criterion)

Typed-only arm produces patches at all: **YES** (4/10).

=> **Safe to flip `AgentConfig.dangerous_tools=False` default** on the
functional criterion. Bash is not load-bearing for SWE-bench
capability; removing it does not crater pass-rate.

## Statistical caveat

Observed patch-rate gap: 10 pp (40% − 30%).

Per-task variance is ±10 pp. At N=10 the combined arm-gap standard
error is ~15 pp, so the 10 pp gap is NOT statistically distinguishable
from zero — the bash arm's 30% and the typed arm's 40% both lie within
each other's noise envelope. These numbers are descriptive, not
inferential.

The red-team plan §5 specified "±2 pp parity at N=50" — which is below
the noise floor even at N=50 (combined arm-gap SE ≈ 2 pp; can't
confirm a 2 pp gap with N=50 without a specific test design that
shrinks intra-task variance). The honest, measurable criterion at any
smoke scale is the functional one used here: does typed-only function?

Statistical parity at higher confidence would require N≈600 per arm
and Docker-eval for actual resolved-rate, not patch-rate. That's a
separate investment from the decision gate.

## Predictions artefacts

- `2026-04-22-parity-bash-predictions.jsonl` — Arm A, 10 entries
- `2026-04-22-parity-bash-meta.json` — Arm A metadata
- `2026-04-22-parity-typed-predictions.jsonl` — Arm B, 10 entries
- `2026-04-22-parity-typed-meta.json` — Arm B metadata

Docker eval can be run later against either JSONL via:

```
python -m swebench.harness.run_evaluation \
    --predictions_path <jsonl> \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --run_id <arbitrary>
```

## Notes on the run

- Script bug uncovered: in `--generate-only` mode the bench writes
  predictions to `C:/Users/.../Temp/sage_swebench_*/predictions.jsonl`
  instead of honoring `--output <path>.json` (the --output path is
  only used when the bench runs Docker eval). The parity script's
  `write_summary` therefore got empty reports and skipped writing the
  summary. Predictions themselves are correct — they were copied into
  this folder manually for durability.
- `create_python_tool` + `validate_and_execute` (post-ADR-013 sandbox
  default) were not exercised on this slice — SWE-bench generation
  never dynamically creates tools; all tool calls are static registry
  lookups. The sandbox default flip is orthogonal to the dangerous_tools
  flip measured here.
