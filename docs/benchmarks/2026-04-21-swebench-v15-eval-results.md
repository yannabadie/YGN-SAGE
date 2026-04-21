# SWE-bench v15 Eval — First Real Docker-Graded Pass-Rate

**Date**: 2026-04-21
**Branch**: `main` @ `bcade10`
**Dataset**: SWE-bench Lite, same 10 instances as v13
**Command**: `python sage-python/scripts/swebench_eval_only_v14.py`
**Input**: v13-generated predictions (5 PATCH + 5 EMPTY, no regeneration)
**Stack**: CRLF fix (efb8afd) + UTF-8 open fix (bcade10) + Directive #3 gating (482ea28)

## Headline

**1/10 resolved (10%)** — **1/5 on real-patch submissions (20%)**. First
non-zero Docker-graded result in this repo. Supersedes the v13 0/10
(which was environment-broken, not a SAGE quality signal).

## Per-task breakdown

| # | Instance | v13 | v14 | **v15** | Note |
|---|---|---|---|---|---|
| 1 | astropy__astropy-12907 | unresolved* | error (cp1252) | ✅ **resolved** | 468 chars, correct fix |
| 2 | astropy__astropy-14182 | unresolved* | error (cp1252) | unresolved | 457 chars, applied OK, tests fail (missing converter logic) |
| 3 | astropy__astropy-14365 | empty | empty | empty | minimax 529 storm |
| 4 | astropy__astropy-14995 | empty | empty | empty | minimax 529 storm |
| 5 | astropy__astropy-6938 | empty | empty | empty | minimax circuit breaker |
| 6 | astropy__astropy-7746 | error (hunk hdr) | error (hunk hdr) | **error (hunk hdr)** | 3375 chars, `@@ -1264,28 +1279,35 @@` malformed |
| 7 | django__django-10914 | error (context) | error (context) | **error (context)** | 2089 chars, hunk mismatch at line 25 |
| 8 | django__django-10924 | empty | empty | empty | minimax 529 |
| 9 | django__django-11001 | unresolved* | unresolved | unresolved | 825 chars, applied OK, tests fail |
| 10 | django__django-11019 | empty | empty | empty | minimax 529 |

\* v13 "unresolved" was a false positive — eval.sh died before pytest
ran (CRLF crash), test_output.txt wrote an empty/ASCII "command not
found" string, swebench classified as unresolved (applied-but-failed).

## Journey: v13 → v14 → v15

| Metric | v13 | v14 | v15 |
|---|---|---|---|
| Predictions submitted | 10 | 10 (reuse) | 10 (reuse) |
| Empty patches | 5 | 5 | 5 |
| Errors (patch-apply) | 2 | 4 | **2** |
| Completed (patch applied) | 3 | 1 | **3** |
| **Resolved (tests pass)** | **0** | 0 | **1** |
| Unresolved (tests fail) | 3* | 1 | 2 |

Each iteration removed one layer of Windows-infra obstruction:
- **v13 → v14**: CRLF fix (`Path.write_text` LF-safe for `.sh`/`.bash`)
  → eval.sh now parses correctly in the Linux container, conda activates,
  pytest runs. Side-effect: 2 previously "completed" tasks flipped to
  ERROR because pytest's tree-output emits Unicode box chars, and
  swebench's `open(path, "w")` wrote them through Windows cp1252.
- **v14 → v15**: UTF-8 open fix (monkey-patch `run_evaluation.open`
  to default text-write encoding to utf-8) → 2 errors reverted to
  completed, and one of them (astropy-12907) **passed all tests**.

## astropy-12907 — the resolved case

Instance: `TypeError` when stacking models via `_cstack` with
`Mapping | Mapping` where the right operand was a plain matrix.
Real human fix: `astropy/modeling/separable.py`, change `cright[...] = 1`
to `cright[...] = right`. SAGE produced the identical fix:

```diff
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right
```

FAIL_TO_PASS tests (3) all passed; PASS_TO_PASS (9) held stable.

## What's left (real SAGE quality signal)

### 2 unresolved — genuine quality gaps

- **astropy-14182**: SAGE added `header_rows=None` parameter to
  `RST.__init__`, but the test wants the whole `header_rows` feature
  (multi-line RST headers with unit rows). Patch is a syntactic
  addition, not a semantic implementation. Reasoner tier under-budget?
- **django-11001**: 825 chars, applies cleanly, tests fail. Haven't
  inspected root cause; likely similar under-specification.

### 2 errors — malformed unified diffs

- **astropy-7746** (3375 chars): hunk header `@@ -1264,28 +1279,35 @@`
  has wrong line counts for the hunk body. LLM hallucinated line
  numbers in a 1300+-line file.
- **django-10914** (2089 chars): context lines don't match file at
  line 348. LLM had stale context for that version of the file.

Both failures are consistent with what Codex's controlled experiment
predicted: CRLF is not the cause here, malformed hunk headers / stale
context are. Mitigation (post-v15):
- Option B (fast): add `patch --dry-run` validator in
  `swebench_bench.py` before writing predictions — per-patch
  quality gate, logs APPLY_PATCH_FAIL as diagnostic signal.
- Option C (real fix): switch SAGE's patch emission from unified-diff
  to search-and-replace blocks. Avoids line-number hallucination.

### 5 empty — minimax 529 storm

Upstream provider operational issue on 2026-04-21 morning (Paris).
Not a SAGE bug. Remediation ideas (out of scope for this smoke):
tighten TTL exclusion for cascades, or fall back to a different
provider earlier when 5xx storms are detected.

## Verdict

The 70 % "SWE-bench Lite smoke" rate previously quoted in CLAUDE.md
§Benchmarks was a **patch-generation** rate (v5d/v5e, how many real
patches came out of SAGE on a different 10-task subset, April 18). No
Docker-graded pass-rate existed until today.

**v15 is the anchor point**: **10 % resolved (1/10)** — 20 % on
real-patch submissions. This is a realistic first-light number for a
small, single-model (`gemini-3.1-flash-lite-preview`), no-retry,
cost-constrained configuration on the Lite set.

Next levers to move the needle (ordered by expected lift):
1. Unblock the 5 EMPTYs (provider-health / fallback policy).
2. Fix the 2 malformed diffs (switch to search-and-replace).
3. Improve the 2 unresolved-but-applied cases (deeper planner budget,
   test-first agent loop?).
4. Scale to 50 / 100 tasks once the infra is proven stable.

## Artifacts

- `docs/benchmarks/2026-04-21-swebench-v15-eval-report.json` —
  summary JSON
- `docs/benchmarks/2026-04-21-swebench-v15-eval.log` — full run log
- `logs/run_evaluation/sage-v14-crlf-20260421-092637/` — per-instance
  container logs (Windows-style path in the report is the swebench
  harness run_id; the v14 label in there comes from the same script
  being reused for v15 — the run_id differs so artifacts don't collide)
- Commits: `efb8afd` (CRLF), `bcade10` (UTF-8), `482ea28` (gating)
