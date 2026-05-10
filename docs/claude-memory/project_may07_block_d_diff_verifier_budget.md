---
name: project_may07_block_d_diff_verifier_budget
description: Block D cycle-13 diff-verifier repair budget closed — 3 HARD_STOP fixes + 44 tests green
type: project
originSessionId: 98c5d292-7098-4166-9e24-6d083e057a81
---
# Block D — diff-verifier repair budget (cycle-13 K, 2026-05-07)

## Status: CLOSED ✓

cgpro `NEXT_BLOCK_ID = D` verdict GO_SHIP at commit `a7a0025f` + HARD_STOP #2 fix `8fbeeb1f`.

## Contract (cgpro DESIGN 2026-05-07)
When `SAGE_DIFF_VERIFIER_MODE=repair` and mismatches exist, repair must either:
- A. receive explicit `repair_budget_usd` cap and record it in metadata, OR
- B. be skipped with deterministic `verifier_repair_budget_exhausted` reason

Never silently spend unbounded extra repair budget.

## Files changed
- `sage-python/src/sage/bench/swebench_diff_verifier.py`: added `repair_budget_usd` param + timeout derivation
- `sage-python/src/sage/bench/swebench_bench.py`: budget computation at callsite + `llm=None` guard + metadata fields
- `sage-python/tests/test_swebench_diff_verifier_budget.py`: RED-first tests

## HARD_STOP chain (3 rounds)
1. **#1** (`a7a0025f`): positive budget not enforced + prediction dict missing fields → fixed
2. **#1b**: test `assert 0.0 is None` fail → fixed to `assert == 0.0` (budget cap always exposed when startswith match)
3. **#2** (`8fbeeb1f`): budget-exhausted skip could fall through to downstream `try_repair_patch` with live LLM → fixed with `llm=None` guard + reason computed from pre-chain `verifier_repair_stage`

## Key code patterns
```python
# swebench_bench.py callsite guard
_llm_for_repair = None if verifier_repair_stage == "verifier_repair_skipped" else llm_handle

# prediction dict
"_verifier_repair_budget_usd": repair_budget_usd if verifier_repair_stage.startswith("verifier_repair") else None,
"_verifier_repair_skipped_reason": "budget_exhausted" if verifier_repair_stage == "verifier_repair_skipped" else None,
```

## Verification
- 44/44 pytest green
- mypy 0, ruff clean
- cgpro VERIFY post-fix `8fbeeb1f` not yet called (paused per user directive)

## Follow-up: Block A held pending explicit budget approval