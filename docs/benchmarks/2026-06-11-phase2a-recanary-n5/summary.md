# RESOLUTION_UNBLOCKERS re-canary — N=5 repair-mode graded (2026-06-11)

**Decision: `INFRA_VALIDATED_NO_GO_2B_YET`** — every plumbing class built
by the RESOLUTION_UNBLOCKERS block worked in production; official
resolution is 0/5 and the run does NOT meet the locked 2.b criteria
(≥1/5 resolved OR a clean reasoning-class 0/5). Three NEW cheap levers
identified below; 2.b and N=50 stay NO_GO.

- **Frozen commit**: `64fb986fd38610486323e60c5b234d999c4517c0`
  (CI/Security/coherence/coverage/fuzz ALL green; Security green via the
  documented PYSEC-2025-194 torch quarantine)
- **Authorization**: Yann's standing GO (2026-06-10 evening) covering the
  re-canary; **real spend $0.7318 generation + $0.0416 Modal grading =
  $0.7734** (caps $5/task gen + $0.50/task repair + $30 global — never
  approached)
- **Config delta vs 2026-06-10**: `--verifier-mode repair
  --repair-budget-usd 0.50` (the new chain), same sha-pinned instances

## What the block PROVED in production

| Block capability | Evidence |
|---|---|
| Targeted-fetch recovery | `fetch_fallback_used=true` on **5/5** repos — including teleport, unrecoverable on 2026-06-10 (clone timeout → blind generation). All 5 worktrees `ready`. The June-10 teleport failure class is CLOSED. |
| Fail-closed accounting | `patches_empty_infra=0 / patches_empty_model=3` — zero blind generation, the infra/model split reports honestly |
| Repair chain live | tutanota-219: pre `hunk_body_count_mismatch` → **mechanical recount fired** (post shows `content_mismatch` = counts fixed, content exposed) → ONE LLM pass → `verifier_repair_empty` → **clean-strict kept the original**. NodeBB: `content_mismatch` → LLM → empty → original kept. Exactly the cgpro-locked semantics. |
| Audited repair channel | `_verifier_repair_provider=deepseek`, `_verifier_repair_model=deepseek-v4-flash`, budget $0.50 recorded per attempt (usage field: see lever 3) |
| Gates | 6/6 acceptance PASS, learning 5/5, pre-grader gate 5/5 (explicit reasons on all 3 empties), 0 timeouts |
| Post-grader taxonomy (first production use) | `BUILD_FAILED=1, EMPTY_PATCH=3, GRADER_OUTPUT_WRITE_FAILED=1` with causal `first_compiler_error` extracted — the write-crash class was bucketized instead of lost |

## Per task

| Instance | Patch | Verifier → repair | Graded verdict |
|---|---|---|---|
| protonmail | — ($0.052) | `skipped_no_patch` | EMPTY_PATCH (grader's local log write ALSO crashed — see lever 1 — verdict survived via taxonomy) |
| teleport | — ($0.108) | `skipped_no_patch` | EMPTY_PATCH — **real worktree this time**; honest no-patch beats June-10's $0.21 blind build-breaker |
| tutanota-219 | ✓ ($0.302) | counts fixed mechanically, LLM empty, original kept | BUILD_FAILED — **same `make ***[sqlite3.target.mk]` break as 2026-06-10 with a DIFFERENT patch** → suspected instance-image build fragility, not patch-caused (lever 4) |
| NodeBB | ✓ ($0.232, first patch ever on this instance) | `content_mismatch`, LLM empty, original kept | GRADER_OUTPUT_WRITE_FAILED (cp1252 class again) |
| tutanota-db90 | — ($0.039) | `skipped_no_patch` | EMPTY_PATCH |

## New levers (ranked by cost)

1. **`PYTHONUTF8=1` on grader invocations** (trivial, root-caused this
   run): `PYTHONIOENCODING` only covers stdio; the grader writes log
   FILES via `open()` without encoding → locale cp1252 → two write
   crashes this run. UTF-8 mode changes the `open()` default. Wrapper
   env fix, zero upstream edits.
2. **Repair LLM tier → reasoner**: the chain wiring is proven; the
   budget-tier model (deepseek-v4-flash) returned empty/unusable on both
   real mismatch feedbacks. One param + a re-capped budget line in the
   manifest.
3. **Usage-recorder shape fix** (telemetry): `_verifier_repair_usage`
   stayed None despite a real call — the live provider's response usage
   is not a plain dict. Small adapter in `_RepairUsageRecorder`.
4. **Instance-health screen for tutanota-219**: same native build break
   two runs straight with different patches. One empty-patch Modal
   sandbox (~$0.01) would establish the baseline; if the image is
   build-broken, flag/replace the instance in the pinned set (with cgpro
   sign-off — set change breaks comparability).

## Decision detail

Locked 2.b criteria (cgpro 2026-06-10 post-run): "≥1/5 resolved OR a
clean 0/5 whose failures are genuinely reasoning/test-class". This run:
0/5 resolved; remaining non-reasoning failure classes = grader-side
encoding (lever 1), suspected instance-side build fragility (lever 4),
repair-model quality (lever 2). **NO_GO for 2.b**; rerun the canary after
levers 1-3 (free + one param) — levers are hours, not days.
