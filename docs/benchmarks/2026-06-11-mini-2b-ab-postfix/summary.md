# MINI_2B re-test after EMISSION_FIXES (2026-06-11 evening)

**Decision: `EMISSION_FIXED_GROUNDING_IS_THE_RESIDUAL` — defects A+B are
closed in production; the next product block is diff grounding.**

| Metric (N=10 paired, same instances) | Arm A (Agentless-lite) | **Arm D pre-fix** (morning) | **Arm D post-fix** |
|---|---|---|---|
| patch_non_empty | 8/10 | 2/10 | **9/10** |
| `git apply --check` OK | **6/10** | 0/10 | 0/10 |
| verifier clean | 1/10 | 0/10 | 1/10 |
| D failure classes | — | 8 EMPTY, 2 COUNT | 1 EMPTY, 6 COUNT_MISMATCH, 2 CONTENT_MISMATCH, 1 APPLY_FAILED |

## What the fixes proved

The cgpro-locked EMISSION_FIXES block (F1 rescue + F2 artifact
pass-through + F3 forced-final turn + F4 bypass patch profile, commits
up to `ba125f8a`) **multiplied patch emission 4.5×** (2/10 → 9/10): the
pipeline was producing value and discarding it in the last mile, exactly
as the forensics said. The "[sage: agent exited with no content]" class
is down to 1/10.

## The residual — and it is ONE class

Every emitted D patch fails `git apply` on context grounding
(count/content mismatches), while arm A — whose emitter receives the
**verbatim file bytes in its prompt** — applies 6/10. D's emitters write
diffs from conversational memory of tool outputs; A's emitter writes
diffs from the actual bytes. The delta is no longer emission, budget,
auth, transit or selection: it is **grounding at generation time**.

Next product block (cgpro consultation): ground the emitter — when the
artifact profile is active, the emitting node's context must carry the
LOCALIZED FILE CONTENTS verbatim (arm A's winning pattern inside the
pipeline: localize → read bytes → emit against bytes), and/or a
mechanical post-emission reground pass (rewrite hunk context from the
real file at the claimed positions — the existing repair chain's
mechanical recount extended to context lines).

## Decision rule status

cgpro rule (D ≤ A → diagnosis; D > A clearly → graded 2.b): D now WINS
non-empty (9v8) but loses applyability (0v6) — still not "D > A
clairement" ⇒ no graded 2.b yet; the diagnosis pivot continues with
grounding as the single named target.

Telemetry note: the mini driver does not yet surface the canary's
`_patch_source` / `_raw_final_patch_status` fields in its per-pair view
(they live in the per-task canary events) — driver follow-up, not
gate-bearing.

Spend: ~$2.1 this re-run (cumulative session ≈ $9.5 of $30).
