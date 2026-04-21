# V16 Repair Pipeline — Post-Mortem (0 lift, learning logged)

**Date**: 2026-04-21
**Commit tested**: `24e51af` (patch validator + two-stage repair)
**Result**: **1/10 resolved — identical to v15**. Zero lift.

## What was shipped

Two-stage repair pipeline in `sage.bench.swebench_patch_repair`:

1. **Programmatic counts-fix**: recomputes `@@ -s,c +s,c @@` from the
   real hunk body line counts. Zero-cost, deterministic.
2. **LLM one-shot repair**: direct call on `agent_loop._llm` with the
   `git apply --check` stderr as feedback. Aider-style apply-with-
   feedback. No tools, no agent-loop state reuse.

Pipeline runs after `_extract_patch` in `generate_patches`. Per-task
metadata adds `_repair_stage` ∈ {"", "unchanged", "programmatic_counts",
"llm_repair", "failed"} for post-run analysis.

## V16 stage distribution

| Stage | Count | Tasks |
|---|---|---|
| `""` (skipped, empty patch) | 5 | the minimax-529 bucket |
| `"programmatic_counts"` | 2 | astropy-12907, astropy-14182 |
| `"failed"` | 3 | astropy-7746, django-10914, **django-11001** |
| `"llm_repair"` | 0 | — |
| `"unchanged"` | 0 | — |

**All 5 real patches failed `git apply --check`** in my validator.
Programmatic counts-fix "resolved" 2 to git-apply-acceptable. 3 remained
broken; LLM repair couldn't produce a valid fix for any of those 3.

## V15 → V16 resolved deltas — all zero

| Instance | V15 | V16 | Why no change |
|---|---|---|---|
| astropy-12907 | resolved | resolved | Programmatic-counts-fix rewrote header; body identical; semantically same fix; still passes tests. |
| astropy-14182 | unresolved | unresolved | Programmatic fix made the patch git-apply-clean; tests still fail because the SAGE patch is semantically shallow (added `header_rows=None` param, didn't implement the logic). |
| astropy-7746 | error | error | Programmatic fix failed (patch also had stale context lines, not just wrong counts); LLM stage returned nothing usable. |
| django-10914 | error | error | Same — stale context lines, LLM stage couldn't fix in one shot. |
| django-11001 | unresolved | unresolved | `git apply --check` rejected it (stricter than container's `patch --fuzz=5`); programmatic+LLM repair both declared "failed"; original patch passed through → container applied it fine via `patch` fallback → tests still fail for the original SAGE quality reason. |

## Root cause of the 0 lift

**My validator (`git apply --check`) is STRICTER than what swebench's
Docker container actually uses.** The container's apply chain is:

```
1. git apply --verbose                (strict)
2. git apply --verbose --reject       (still strict)
3. patch -p1 --fuzz=5 -i patch.diff   (lenient — fuzz tolerance)
```

The 3rd step — GNU `patch --fuzz=5` — accepts:
- Slightly wrong hunk header counts (it adjusts them)
- Context lines with small drift (fuzz looks 5 lines up/down for match)
- Trailing CRs (the `Stripping trailing CRs from patch` line we saw)

My validator only tests step 1. So patches that would have been fine
with steps 2 or 3 got flagged as "broken", sent to the repair pipeline,
and the pipeline either:
- Produced a programmatic-fixed version that was cosmetically different
  but semantically identical (astropy-12907, -14182 — no gain)
- Failed outright and emitted the original anyway (astropy-7746, -10914,
  django-11001 — no change)

The repair pipeline *is* correct code-wise. It just never gets a chance
to save a task the container wasn't already going to save.

## Takeaway

**Patch-repair is the wrong lever for this failure mix.** The 2 "error"
tasks fail because of semantic LLM hallucination in the unified-diff
body (wrong context lines, wrong hunk ordering) — problems that only a
content-aware tool can fix, not a counts recomputation. The gemini-
flash-lite-preview budget model couldn't one-shot-repair them either
(advisor had predicted this: the repair prompt lacks source-file content;
Aider's pattern works because they feed the file).

**Orthogonal fix that should actually lift**: the v17 pipeline-level
fallback change (commit `4f90a98`). Stage 4 multi-agent failure used to
fall through to `self.llm_provider.generate()` — which on a minimax 529
storm was the same dead minimax. The fix routes to a healthy alternate
from `provider_pool` and raises on empty content rather than emitting
an empty patch. Targets the 5/10 EMPTY bucket; measured separately by
v17 full smoke (task #42).

## What stays in the repo

- `sage.bench.swebench_patch_repair` — module retained; tests pass
  (14/14). It's correct code; the validator just needs to be relaxed
  later to match container behavior (add `patch --fuzz=5` fallback),
  and the LLM repair prompt needs source-file content. Both are
  follow-up tasks, not blockers for shipping other work.
- `sage.bench.swebench_bench.generate_patches` wiring — retained. Runs
  a fast `git apply --check`; when it succeeds (stage="unchanged")
  that's confirmation the patch is trivially valid. On failure, the
  current pipeline at worst does nothing harmful (returns original).

## What to do next before re-running a smoke

- **Do NOT revert `24e51af`**. Costs nothing on the happy path and
  leaves the scaffolding in place for a later prompt / validator fix.
- **Relax validator** in a follow-up: either run `patch -p1 --dry-run
  --fuzz=5 -` as a second check (declare valid if EITHER tool accepts),
  or match swebench's chain exactly.
- **Fix LLM repair prompt** in a follow-up: read ±15 lines around the
  failing hunk's line numbers from the cloned repo and include them
  in the prompt. This is what Aider does.

## Artifacts

- Report JSON: `docs/benchmarks/2026-04-21-swebench-v16-repair-eval-report.json`
- Run log: `docs/benchmarks/2026-04-21-swebench-v16-repair-eval.log`
- Module: `sage-python/src/sage/bench/swebench_patch_repair.py`
- Script: `sage-python/scripts/swebench_repair_and_eval_v16.py`
- Commits touched: `24e51af` (repair pipeline), `4f90a98` (Stage 4 fallback fix — separate)
