# MINI_2B re-test after GROUNDING block (2026-06-11 night)

**Decision: `FIRST_APPLIES_G2_CRLF_DRIVEN__G1_REACH_GAP_IDENTIFIED`.**

| Metric (N=10 paired) | Arm A | D pre-fix (am) | D post-emission (pm) | **D post-grounding** |
|---|---|---|---|---|
| patch_non_empty | 8/10 | 2/10 | 9/10 | 9/10 |
| `git apply` OK | **8/10** | 0/10 | 0/10 | **3/10** |
| verifier clean | 7/10 | 0/10 | 1/10 | 3/10 |
| D classes | — | 8 EMPTY 2 COUNT | 1E 6C 2CONTENT 1A | 1E **5C** 0CONTENT 1A 3✓ |

## What moved — and what moved it

**Arm D applies for the first time in project history: 0 → 3/10**
(protonmail, teleport, openlibrary). Honest attribution from the
per-task stages: protonmail and openlibrary repaired at
`positional_reground` (G2) and flipt's reground also fired; the CRLF
clone-fidelity fix (`core.autocrlf=false`) lifted BOTH arms' apply
ceilings — **arm A rose 6→8/10 on the same tide**. The CONTENT_MISMATCH
class went 2→0.

**G1's effect is NOT verifiable in this run** — two gaps, both now
precisely identified:

1. **Telemetry export gap**: `grounding_telemetry` /
   `last_grounding_telemetry` are built but never exported through the
   CLI events or canary summaries — attachment is invisible post-hoc.
2. **Emitter-matcher reach gap**: the actual topologies this run used
   debate/mixer/actor templates (roles: topic_setter, debater_a/b,
   judge, mixer, actor, reviewer…). `is_emitter_role` matched ONLY
   `coder` (4 node occurrences across 10 tasks) — the de-facto emitters
   of the other templates were name-excluded. G1 reached at most a
   minority of emitting turns.

## Decision rule status

D (3 apply) < A (8 apply): not "D > A clairement" ⇒ no graded 2.b. The
trajectory across the day stands at D apply 0% → 0% → 33% with each
mechanism shipped against measured causes.

## Locked next iteration (cheap, before any further paid run)

1. **Export grounding telemetry** (event or summary channel) — directive
   #9: an unverifiable mechanism is an unproven one.
2. **Effective-emitter matching**: under the verified patch profile,
   invert to denylist-only (exclude planner/judge/verifier/topic_setter;
   include actor/debaters/mixer) — on a patch task, whoever produces the
   artifact needs the bytes, whatever its template name.
3. **Product question surfaced upward**: the pipeline selected DEBATE
   topologies for single-file bugfix tasks. Grounding cannot fix
   topology mis-selection; this goes to the routing/selection review.

## Spend

This run ≈ $2.1; session cumulative ≈ $11.6 of the $30 cap.
