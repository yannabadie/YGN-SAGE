# v12 patches — semantic validation vs SWE-bench gold

**Date:** 2026-04-19
**Source:** v12 predictions at `docs/benchmarks/2026-04-19-swebench-smoke-v12-15task-f12-s3-cap19.log`
**Proxy claim:** 11/15 = 73% real patch rate
**Goal:** same method as v8/v9 validation — line-diff + manual inspection.

## Per-instance verdict

| Instance | Exact | Loose | Verdict | Note |
|---|---|---|---|---|
| astropy-12907 | 1/1 | 1/1 | **PASS** | EXACT |
| astropy-14182 | 1/28 | 2/28 | **LIKELY FAIL** | Same shim pattern as v9 — `**kwargs` passthrough but no `header_rows` feature |
| astropy-14995 | 1/2 | 1/2 | **PASS** | Audit target, critical fix line matches gold |
| astropy-6938 | 0/1 | 1/1 | **LIKELY PASS** | `encode_ascii` semantic equiv to bytestring |
| astropy-7746 | 0/4 | 1/4 | **UNCERTAIN** | Different interception (caller vs callee) |
| django-10924 | 0/1 | 1/1 | **LIKELY FAIL** | Patched wrong file: `django/forms/fields.py` FormField instead of `django/db/models/fields/__init__.py` ModelField. FAIL_TO_PASS test probably calls ModelField.formfield() path |
| django-11001 | 0/2 | 0/2 | **LIKELY PASS** | Newline flattening ≈ re.MULTILINE regex, both solve the test |
| django-11039 | 2/3 | 3/3 | **PASS** | Critical line matches gold |
| django-11049 | 0/1 | 1/1 | **LIKELY FAIL** | Patched the wrong error message (`invalid_duration` vs `invalid`). Gold fixes the string that the FAIL_TO_PASS test asserts |
| django-11099 | 0/2 | 2/2 | **LIKELY PASS** | `\A...\Z` ≡ `^...\Z` |
| django-11179 | 1/1 | 1/1 | **PASS** | EXACT |

## Summary

- **Confirmed PASS (exact/near-exact)**: 12907, 14995, 11039, 11179 = 4
- **LIKELY PASS (semantic equivalence)**: 6938, 11001, 11099 = 3
- **UNCERTAIN**: 7746 (different interception) = 1
- **LIKELY FAIL**: 14182, 10924, 11049 = 3

**True pass rate estimate: 7–8 of 15 = 47%–53%.**

Proxy (73%) inflated by ~20 percentage points. Inflation sources:
- Shim patch (14182) — non-feature passthrough
- Wrong-file patch (10924) — didn't hit the tested code path
- Wrong-string patch (11049) — fixed an adjacent message

## Comparison to prior validated runs

| Run | Proxy | Validated true rate | Inflation |
|---|---|---|---|
| v8  | 6/15 = 40%   | 5–6/15 = 33–40%  | 0–7pp |
| v9  | 9/15 = 60%   | 6–8/15 = 40–53%  | 7–20pp |
| v12 | 11/15 = 73%  | 7–8/15 = 47–53%  | 20–26pp |

**Observation**: as the proxy rate climbs, inflation grows. The
coder+synthesizer produce more diff-shaped output, but not all of it
actually targets the tested code path. D7 + F11 reject obvious
non-diffs (sentinels, bash blocks), but they can't distinguish a
well-formed diff that touches the wrong file from a correct one.

## What F10–F12 added over v9

From v9 (9/15 = 60% proxy, 40–53% true) to v12 (11/15 = 73% proxy, 47–53% true):
- **Proxy**: +3 patches (+20pp)
- **Validated true**: +0–1 patches (+0–7pp)
- Most of the proxy gain came from F10 turning sentinels into forwarded
  content that SOMETIMES happens to be on the right file (like 11099)
  but often isn't (10924, 11049).

## Cumulative progression (validated rates)

| Version | Honest proxy | True rate estimate |
|---|---|---|
| baseline | 2/5 = 40% (1 fake masked) | 2/5 = 40% (2 real reported as 3) |
| v7       | 5/15 = 33%               | ~27% |
| v9       | 9/15 = 60%               | 40–53% |
| **v12**  | 11/15 = 73%              | **47–53%** |

**The genuine gain from baseline to v12 is roughly +5–10 percentage points on the true pass rate**, not the +33pp the proxy suggests.

## What Docker-eval would settle

The 3 LIKELY FAIL and 1 UNCERTAIN cases (14182, 10924, 11049, 7746) are all debatable on paper. A one-shot Docker-eval on just these 4 instances would close the gap and definitively separate proxy from truth.
