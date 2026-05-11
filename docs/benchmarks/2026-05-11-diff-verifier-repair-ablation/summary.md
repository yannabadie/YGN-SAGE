# Slice 10B — diff-verifier-repair-ablation findings

**cgpro VERIFY 2026-05-11** Fix #5 verdict was MODIFY — paired ablation observe-vs-repair on slice 9 patches, NOT a flip to repair-as-default. This delivers the paired ablation.

**Inputs**:
- baseline patches: `docs/benchmarks/2026-05-11-canary-patch-focused-prompt-profile/run/predictions.json` (slice 9, 5/5 non-empty, 0/5 Modal-resolved)
- instances metadata: `docs/benchmarks/2026-05-11-canary-n5-graded/instances/instances.json`
- llm tier for repair: `budget` (deepseek-v4-flash via `init_llm_provider`)
- repair_budget_usd cap: $0.50 per task

**Outputs** in this directory:
- `repair_summary.json` — per-task ablation record (machine-readable)
- `predictions.json` — repaired patches (mode=repair output)
- `eval/eval_results.json` — Modal grading verdict on repaired patches (in progress)

## Headline

**Repair mode did NOT reduce the AVR-induced path-hallucination mismatches.** Per cgpro's pre-emptive warning:

> "Ne pas prétendre que repair améliore la compréhension du bug : il peut corriger format/contexte, pas la logique."

The slice 9 AVR runs (webclients, tutanota db90) produced patches whose `--- a/<path>` and `+++ b/<path>` line referenced files that don't exist in the base_commit cloned repo (slice 10C empirically proved this is judge/output role regeneration). The diff verifier in `observe` mode CORRECTLY flagged these as `file_missing` mismatches. The LLM repair call DID generate a new patch with slightly different content, but the repair LLM (deepseek-v4-flash budget tier) **could not invent the correct file path** — it doesn't have repo access either, it only sees the broken patch and the mismatch report.

## Per-task ablation

| Task | Topology | Mismatches (obs) | Mismatches (post-repair) | Repair stage | Patch chars Δ | Verdict |
|---|---|---|---|---|---|---|
| protonmail/webclients | **AVR** | 1 | **1** (no reduction) | `verifier_repair` | 1145 → 1093 (-52) | `repair_did_not_reduce_mismatches` |
| gravitational/teleport | sequential | 0 | n/a | `skipped_no_mismatches` | unchanged | `no_repair_needed` |
| tutao/tutanota 219bc | sequential | 1 | n/a | `verifier_repair_empty` | unchanged | LLM timed out at 60s, no repair patch |
| NodeBB | sequential | 0 | n/a | `skipped_no_mismatches` | unchanged | `no_repair_needed` |
| tutao/tutanota db90 | **AVR** | 4 | **4** (no reduction) | `verifier_repair` | 1574 → 1558 (-16) | `repair_did_not_reduce_mismatches` |

**Total**: 6 mismatches before repair → 5 mismatches after (the 1 reduction is the timed-out task that returned empty, not a real fix).

## Verdict by topology

- **Sequential** (3/5 tasks):
  - 2/3 patches have 0 mismatches — diff verifier clean, no repair needed
  - 1/3 has 1 mismatch (tutanota 219bc) — but repair LLM timed out, so we can't claim repair-helps-on-sequential from this run
- **AVR** (2/5 tasks):
  - **2/2 patches have 1-4 mismatches**, all `file_missing` type (path hallucination)
  - **2/2 repair attempts FAILED to reduce mismatches** — LLM rewrote the diff with similar character count but didn't fix the underlying wrong-file-path issue

This matches slice 10C's diagnosis: **AVR's judge/output role regenerates the diff with hallucinated paths**, and the verifier+repair toolchain cannot fix path hallucination because the repair LLM doesn't have repo access.

## What this rules out

- ❌ "Activate `SAGE_DIFF_VERIFIER_MODE=repair` as default to fix 0/5 resolved" — repair as currently wired does NOT improve patch correctness when the underlying error is structural (wrong file path) rather than format (bad context lines).
- ❌ "Repair mode is a free fix for SWE-bench Pro 0/5 resolved" — false. It costs an extra LLM call per mismatched task ($0.01-0.50 depending on tier) and yields no improvement on the dominant failure mode in this run.

## What this DOES confirm

- ✅ The diff verifier `observe` mode is working — it correctly flagged 6 mismatches across 5 tasks, with `file_missing` being the dominant `kind` for AVR tasks.
- ✅ The repair LLM call mechanics work — 3 LLM calls completed (2 returned a modified patch, 1 timed out), zero crashes, budget cap respected.
- ✅ Sequential topology produces near-zero verifier mismatches (2/3 clean, 1/3 minor) — corroborating slice 10C's finding that sequential synth preserves the coder's diff.

## Modal grader on repaired predictions — DEFINITIVE

Run completed. `eval/eval_results.json`:

| Instance | Resolved (slice 9 baseline) | Resolved (slice 10B repaired) |
|---|---|---|
| protonmail/webclients | False | **False** |
| gravitational/teleport | False | **False** |
| tutao/tutanota 219bc | False | **False** |
| NodeBB | None (UTF-8 emoji exception, recurring) | **False** |
| tutao/tutanota db90 | False | **False** |

**0/5 → 0/5 resolved. Repair mode produced zero improvement on resolution rate.**

This is the empirical confirmation of cgpro's pre-warning: repair fixes format/context, not logic. The AVR judge's path hallucination is a **structural** error (wrong file path entirely), not a **formatting** error the repair LLM can address from the mismatch report alone.

Modal cost: $0.048 incremental (this run; cumulative today $0.133 after slice 9's $0.085).

## Cost accounting

- Canary baseline (slice 9 already produced): not counted here, already attributed to slice 9 ($0.81).
- Repair LLM calls: 3 attempts × deepseek-v4-flash budget tier ≈ $0.01-0.05 total (precise capture pending Modal billing API call).
- Modal grader on repaired predictions: ~$0.05 expected.
- Total slice 10B incremental cost: < $0.20.

## Recommendation

Per cgpro VERIFY, **slice 10B's ACCEPTANCE is met** — we have the paired observe-vs-repair evidence. Repair mode does NOT close the 0/5 resolved gap because the underlying failure mode is **AVR judge/output path hallucination**, not hunk-format corruption.

The next steps (NOT part of slice 10B):
1. **Fix AVR judge prompt** to preserve actor's diff verbatim (separate cgpro DESIGN needed). This is the high-leverage fix.
2. **Investigate the sequential 1-mismatch case** (tutanota 219bc) once we have a non-timed-out repair attempt — could be a true repair-mode use case.
3. **Do NOT** flip repair as default (cgpro VERIFY NON_GOAL).

This slice is informational — no production code change. The ablation script is reusable for future paired runs.
