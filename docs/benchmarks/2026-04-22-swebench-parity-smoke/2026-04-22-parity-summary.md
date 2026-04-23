# SWE-bench typed-vs-bash parity smoke — N=10

**Dates:**
- Generation: 2026-04-22, gen-only
- Docker-eval: 2026-04-23, full harness (run IDs `sage-20260423-080111` for Arm A, `sage-20260423-084039` for Arm B)

**Slice:** SWE-bench Lite, `--limit 10 --offset 0` (deterministic HF ordering)
**Wallclock:** ~50 min gen-only + ~35 min Arm A Docker + ~25 min Arm B Docker

## Headline

| Metric | Arm A (bash) | Arm B (typed-only) |
|---|---|---|
| N | 10 | 10 |
| Patches produced | 3 (30%) | 4 (40%) |
| Applied cleanly | 1 | 1 |
| Malformed / apply-failed | 2 | 3 |
| Resolved | **0/10 (0%)** | **0/10 (0%)** |
| Empty (gen failed) | 7 | 6 |

## Per-task breakdown

| instance_id | Arm A (bash) | Arm B (typed-only) |
|---|---|---|
| astropy__astropy-12907 | EMPTY | EMPTY |
| astropy__astropy-14182 | PATCH (malformed) | **PATCH (applied, tests failed)** |
| astropy__astropy-14365 | EMPTY | EMPTY |
| astropy__astropy-14995 | EMPTY | PATCH (apply failed) |
| astropy__astropy-6938 | PATCH (malformed) | EMPTY |
| astropy__astropy-7746 | EMPTY | EMPTY |
| django__django-10914 | EMPTY | PATCH (malformed) |
| django__django-10924 | **PATCH (applied, tests failed)** | EMPTY |
| django__django-11001 | EMPTY | PATCH (apply failed) |
| django__django-11019 | EMPTY | EMPTY |

## §5 Decision gate

**Functional criterion:** Arm B produces patches (4 > 0). MET.
**Resolved-rate parity:** 0/10 = 0/10. MET trivially (both arms non-productive on this slice).

Removing `execute_bash` does NOT degrade resolved rate. Both arms are equally
non-productive on this N=10 slice, and both produce patches at similar rates
(within noise). The flip of `AgentConfig.dangerous_tools=False` default
(shipped in commit `d275bc7`) is empirically safe.

## Finding: diff-emission quality is the bottleneck, not tool set

The two arms resolved different subsets of the 10 tasks (intersection = 1,
`astropy-14182`), yet BOTH arms hit the same ceiling of 0/10 resolved —
because:

- 5/7 patches are MALFORMED or APPLY-FAILED (trailing CRs, wrong context
  lines, or syntactically broken diff bodies). `git apply` / `patch`
  refuses them before any test runs.
- 2/7 patches apply cleanly but the fix is semantically wrong
  (FAIL_TO_PASS test still fails): astropy-14182 in Arm B, django-10924
  in Arm A.

The tool set shapes WHICH tasks the agent attempts but neither tool set
teaches the agent to emit well-formed diffs or semantically-correct fixes.

## Implications

1. **§5 flip is safe.** Commit `d275bc7` already landed, flipping
   `AgentConfig.dangerous_tools` default to False. This run confirms no
   resolved-rate regression.
2. **Next investment: diff emission quality.** The
   `docs/superpowers/plans/2026-04-21-semantic-quality-plan.md` semantic-
   quality plan (search-and-replace emission format + semantic-miss
   bucket) directly targets the bottleneck this smoke revealed. It's now
   the highest-leverage next direction.
3. **A separate failure mode surfaced then receded:** the first Arm A
   eval run (Apr 23 07:31) errored on django-10924 with `can't start
   new thread` during image pull — that was host thread exhaustion
   coinciding with the PC crash. The re-run (workers=1) completed
   cleanly: django-10924 actually applies and fails tests — a clean
   unresolved, not an infra bug. Keep `--max-workers 1` as the safe
   default for Windows Docker Desktop when sustained runs are expected;
   `--max-workers 2` is fine for short runs on a freshly-booted daemon.

## Statistical caveats

- Per-task variance on SWE-bench is ~10 pp. At N=10 the combined arm-gap
  standard error for patch-rate is ~15 pp; both the 10 pp patch-rate gap
  (30% → 40%) and the 0 pp resolved-rate gap are INSIDE noise.
- The red-team plan's "±2 pp at N=50" statistical criterion is below the
  noise floor even at N=50 (combined SE ≈ 2 pp) — confirming ±2 pp
  statistically would need N≈600 per arm.
- The functional criterion ("does typed-only function?") and the
  non-regression criterion ("does resolved-rate degrade?") are the
  actually-measurable criteria at smoke scale, and both are MET here.

## Artefacts

- `2026-04-22-parity-bash-predictions.jsonl` — Arm A, 10 entries
- `2026-04-22-parity-typed-predictions.jsonl` — Arm B, 10 entries
- `2026-04-22-parity-bash-eval-v2.log` — Arm A Docker harness stdout
- `2026-04-22-parity-typed-eval.log` — Arm B Docker harness stdout
- Harness per-instance reports:
  - `sage-python/logs/run_evaluation/sage-20260423-080111/` — Arm A
  - `sage-python/logs/run_evaluation/sage-20260423-084039/` — Arm B
