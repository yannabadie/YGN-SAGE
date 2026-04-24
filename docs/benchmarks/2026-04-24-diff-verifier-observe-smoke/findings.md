# 2026-04-24 — Diff-verifier observe-mode smoke (N=10)

**Session:** continuation from 2026-04-23. Goal is A1 (`roadmap.md`) — accumulate observe-mode diff-verifier data toward the ≥10 PATCH / ≥10 clean gate that unlocks repair-mode flip.

**Command:**

```bash
set -a && source .env && set +a
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
SAGE_DIFF_VERIFIER_MODE=observe \
  python -m sage.bench --type swebench --dataset lite --limit 10 \
    --output docs/benchmarks/2026-04-24-diff-verifier-observe-smoke/observe.json
```

Ran ~35 min (generation) + ~8 min (Docker grading).

## Results at a glance

| Phase | Counts |
|---|---|
| Generation (N=10) | 2 PATCH / 8 EMPTY |
| Docker-graded (PATCHes only) | 0 resolved / 2 apply-failures |
| Verifier flagged | 1 (astropy-14365, 2 content_mismatch hunks) |
| Verifier missed | 1 (astropy-6938, malformed-hunk-header class — see below) |
| Fast-abort (<90 s) | 2 deterministic + 1 borderline |

## Per-task breakdown

| # | Instance | Outcome | Latency | Tool calls | Verifier | Docker |
|---|---|---|---|---|---|---|
| 1 | astropy-12907 | EMPTY | 266.5 s | 35 | — | empty_patch |
| 2 | astropy-14182 | EMPTY | 57.8 s | **0** (fast-abort, same as 2026-04-23) | — | empty_patch |
| 3 | astropy-14365 | **PATCH** | 180.7 s | 36 | ✅ **2 × content_mismatch** | `2/2 hunks FAILED` |
| 4 | astropy-14995 | EMPTY | 173.9 s | ? | — | empty_patch |
| 5 | astropy-6938 | **PATCH** | 154.1 s | ? | ❌ not flagged | `malformed patch at line 15` |
| 6 | astropy-7746 | EMPTY | 71.5 s | 0-ish (fast-abort, same as 2026-04-23) | — | empty_patch |
| 7 | django-10914 | EMPTY | 323.3 s | ? | — | empty_patch (was PATCH+flagged on 2026-04-23 — task-level flakiness) |
| 8 | django-10924 | EMPTY | 87.1 s | ? | — | empty_patch (borderline fast-abort) |
| 9 | django-11001 | EMPTY | 193.4 s | ? | — | empty_patch |
| 10 | django-11019 | EMPTY | 155.1 s | ? | — | empty_patch |

## New finding — verifier blind spot

The 2026-04-23 smoke validated the content-mismatch detector. Today's smoke surfaces a **second** failure class the current verifier does NOT detect:

### Malformed hunk header (line-count mismatch)

`astropy-6938` emitted a patch where the hunk header claims
`@@ -1541,10 +1541,4 @@` (10 old lines, 4 new lines), but the diff body
has context+removed lines that don't add up to those counts. `patch`'s parser
aborts with `malformed patch at line 15`. No verifier annotation was produced.

Extract:

```diff
@@ -1541,10 +1541,4 @@
             # `output_field` should be modified.
             output_field = np.char.strip(output_field)

-        # Replace exponent separator in floating point numbers
-        if 'D' in format:
-            # Although the FITS standard indicates D should be used, it's not clear
...
-            output_field.replace(encode_ascii('E'), encode_ascii('D'))
```

The verifier walks each hunk to compare context/removed lines against file bytes. That check can only run if `patch` can parse the hunk in the first place. When the header line counts lie, the verifier never gets a chance to compare.

**Coverage gap:** verifier currently catches context-hallucination within well-formed hunks. It doesn't catch hunk-header arithmetic lies.

**Repair-mode implication:** an LLM one-shot repair on `content_mismatch` is cleaner (the file's actual context is known) than a repair on `malformed_header` (which requires re-counting the hunk). Different repair policies per class.

## Cumulative observe data (A1 gate tracking)

| Smoke | N | PATCH | EMPTY | Verifier flagged | Verifier-missed fails |
|---|---|---|---|---|---|
| 2026-04-23 | 10 | 2 | 8 | 2/2 | 0 |
| 2026-04-24 | 10 | 2 | 8 | 1/2 | 1 (malformed header) |
| **Total** | **20** | **4** | **16** | **3** | **1** |

**A1 gate:** ≥10 PATCH, ≥10 clean needed. **Currently 4 PATCH, 3 flagged, 0 confirmed clean (all 4 patches failed Docker apply).**

## Gaps + confirmed patterns

### Deterministic fast-abort tasks (A2 signal)

Both smokes produced EMPTY in < 90 s for:

- **astropy-14182**: 58 s (0 tool calls). Now N=2/2.
- **astropy-7746**: 72 s. Now N=2/2.

Two deterministic fast-aborts out of 10 is a reproducible 20% floor. That's the A2 investigation target — it silently eats 20% of smoke budget. Per roadmap A2: "essentially zero $" cost to diagnose.

### Task-level flakiness

- **django-10914**: 2026-04-23 = PATCH (caught the headerless-diff parser bug), 2026-04-24 = EMPTY. Same task, different outcome — the planner+coder decomposition is not deterministic on this task.

### Zero clean patches observed

After 2 smokes, 4 PATCHes, not a single one passed Docker apply cleanly. The 2026-04-23 smoke had 2/2 flagged (both content_mismatch); today 1/2 flagged + 1/2 failed for a class we don't yet detect. **The "clean PATCH" sample is currently N=0.**

If the Arm B model (`gemini-3.1-flash-lite-preview` bandit-assigned to the coder role) has a ~0% clean-patch rate on SWE-bench lite, repair-mode with the current content_mismatch detector won't move the needle — the patches that DO get generated all have diff-hygiene issues, and half of them are outside the verifier's current coverage.

Two options surface:

1. **Broaden the verifier** to include hunk-header arithmetic validation before the content walk. Straightforward — count `-`/` ` lines against old count, count `+`/` ` lines against new count, flag on mismatch. Adds ~20 LOC to the parser.
2. **Upstream the problem**: tighten the search-replace → diff emission so the generator can't produce bad line counts. That's ADR material, not a verifier extension.

## Next actions

1. **Keep accumulating.** Every future SWE-bench smoke already includes `SAGE_DIFF_VERIFIER_MODE=observe` per the CLAUDE.md example. Two more smokes of this shape gets us to N=40 with ~8 PATCHes — still under the A1 "10 clean" bar but closer.
2. **A2 fast-abort diagnosis.** Grep gen log for astropy-14182 / astropy-7746 to see which stage aborts at 0 tool calls. Cheap, fixes the small-N problem on everything else.
3. **Extend verifier (B option for repair).** Add hunk-header arithmetic validation — small scope, closes today's blind spot. Would let repair-mode cover both failure classes instead of just one.
4. **Arm B model question.** If the coder-role model emits unapplicable diffs deterministically, the experiment should either (a) try a different coder-role assignment or (b) accept that the verifier is measuring emission hygiene of *this* model, not a model-agnostic signal.

## Artefacts

- `observe.json` — benchmark report
- `observe.jsonl` — per-task truth pack (no verifier annotations — those only appear in the operator log, not the truth pack)
- `observe-gen.log` — full generation + Docker log (128 KB)
- `observe-summary.json` — rollup
- Patches under `sage-python/logs/run_evaluation/sage-20260424-065103/sage__gemini-3.1-flash-lite-preview/`:
  - `astropy__astropy-14365/patch.diff` + `run_instance.log` (2 × content_mismatch, both hunks FAILED)
  - `astropy__astropy-6938/patch.diff` + `run_instance.log` (malformed header, patch aborted at line 15)
