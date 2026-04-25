---
name: CodeContests Test Cases Fix
description: CodeContests uses stdin/stdout format, not Python unittest. Need harness that patches sys.stdin and compares stdout. 165 tasks, up to 200 test pairs each. Ready to implement in next cycle.
type: project
---

## Problem
CodeContests tasks get RUNS_OK (0.3) instead of PASSED (1.5) because _load_test_cases() doesn't load them.

## Key Facts
- Parquet: `data/code_contests_test.parquet` (165 rows)
- Test format: `{input: [...], output: [...]}` in public_tests, private_tests, generated_tests
- stdin/stdout format (competitive programming), NOT Python unittest
- Task IDs: "CodeContests/0" through "CodeContests/164" (DataFrame row index)

## Solution
Generate a test harness per task that:
1. Patches sys.stdin with StringIO(input)
2. Captures stdout
3. Compares stripped output to expected
4. Cap at 20 test pairs per task (30s timeout)

## Status
- Code ready (agent produced full implementation)
- NOT applied to current run — save for next cycle
- Needs review: exec() based harness has security concerns

**How to apply:** Add CodeContests block to _load_test_cases() in execution_reward.py after HumanEval block.
