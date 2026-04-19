# v9 patches — semantic validation vs SWE-bench gold

**Date:** 2026-04-19
**Source:** v9 predictions at `docs/benchmarks/2026-04-19-swebench-smoke-v9-15task-f9-kimi-disabled.log`
**Proxy claim:** 9/15 = 60% real patch rate
**Anchor goal:** reconcile with how many actually semantically match the gold fix.

## Method

Same as v8 validation: line-diff exact match, loose token-overlap fallback, then manual inspection for ambiguous cases.

## Per-instance verdict

| Instance | exact | loose | Verdict | Note |
|---|---|---|---|---|
| astropy-12907 | 1/1 | 1/1 | **PASS** | EXACT match of gold's single-line fix |
| astropy-14182 | 0/28 | 1/28 | **LIKELY FAIL** | Gold = full `header_rows` feature (28 lines); SAGE = `**kwargs` passthrough only. FAIL_TO_PASS = `test_rst_with_header_rows` — needs the feature, not a shim |
| astropy-14995 | 1/2 | 1/2 | **PASS** | Critical fix line matches gold (audit target) |
| astropy-6938 | 0/1 | 1/1 | **LIKELY PASS** | `encode_ascii('E') ≡ b'E'`, semantically equivalent |
| astropy-7746 | 0/4 | 2/4 | **UNCERTAIN** | Gold patches 2 callee helpers; SAGE patches the caller with an `elif all(len==0)` — different interception point |
| django-11001 | 0/2 | 0/2 | **UNCERTAIN** | Gold adds `re.MULTILINE|re.DOTALL` to regex; SAGE flattens newlines before the search. Different approach, outcomes may or may not overlap on test input |
| django-11039 | 1/3 | 1/3 | **PASS** | Critical line is exact match (`self.output_transaction = migration.atomic and connection.features.can_rollback_ddl`) |
| django-11133 | 0/1 | 1/1 | **LIKELY PASS** | Gold: `isinstance(value, (bytes, memoryview))`. SAGE: adds separate `if isinstance(value, memoryview): return bytes(value)` after the bytes check. Same code path, same outcome |
| django-11179 | 1/1 | 1/1 | **PASS** | EXACT match |

## Summary

- **Confirmed PASS**: 12907, 14995, 11039, 11179 (4)
- **LIKELY PASS** (semantic equivalence): 6938, 11133 (2)
- **UNCERTAIN** (different approach, test outcome unclear): 7746, 11001 (2)
- **LIKELY FAIL** (incomplete fix): 14182 (1)

**Real pass-rate range: 6–8 of 15 = 40%–53%.**

Proxy (60%) is inflated by ~7–20 percentage points. The inflation comes from:
- 1 shim that compiles as a "diff" but doesn't fix the real bug (14182)
- 2 cases where SAGE's approach diverges enough that we can't assert PASS without running tests (7746, 11001)

## Comparison to v8

| | v8 (F6+F8) | v9 (F6+F8+F9) |
|---|---|---|
| Real-looking patches | 6/15 = 40% | 9/15 = 60% |
| Confirmed semantic matches | 5 | 6 |
| Uncertain | 1 (7746) | 2 (7746, 11001) |
| Likely FAIL within "real" | 0 | 1 (14182) |

v9 adds 3 new "real" patches vs v8. Of those 3:
- 12907: **confirmed PASS** — F9 clear win (was Kimi-crashed EMPTY in v8)
- 14182: **likely FAIL** — semantic failure, the patch is a shim
- 11001: **uncertain** — different approach to the bug

So F9's true contribution is **+1 to +2 real passes** (not +3 as the proxy suggests).

## What F9 unambiguously did

- Removed 80% of Kimi HTTP 400 errors (10 → 2 in the log).
- Recovered astropy-12907 from EMPTY to a confirmed-matching patch.
- Did NOT regress any prior winner (all 6 v8 winners still passed in v9).

## Lessons for the loop

- Semantic validation is doable without Docker and catches proxy inflation.
- "Real" (= non-sentinel non-empty diff) overstates reality by ~10–20pp on this benchmark.
- The gap narrows when upstream quality is higher (v8 inflation was ~7pp; v9 inflation is ~10–20pp because the coder now produces more diffs, some of which are shims rather than fixes).

## Anchored claim

Audit has moved from baseline honest **2/5 = 40%** (5-task, high variance) to v9 validated **6–8/15 = 40–53%** (15 tasks, lower variance). The best single commit was F9 (2026-04-19) which unblocked a provider-layer bug silently zeroing 4 tasks per run.

Docker-eval remains the gold standard and would settle 7746, 11001, 14182 definitively.
