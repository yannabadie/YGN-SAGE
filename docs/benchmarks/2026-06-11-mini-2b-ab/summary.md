# MINI_2B — paired ungraded Arm A vs Arm D (2026-06-11)

**Decision: the cgpro rule fires unambiguously — `PIVOT_TO_C_PRODUCT_DIAGNOSIS`.**

| Metric (N=10 paired, same instances) | **Arm A** (Agentless-lite, 2 calls) | **Arm D** (full SAGE pipeline) |
|---|---|---|
| patch_non_empty | **9/10** | 2/10 |
| `git apply --check` OK | **4/10** | 0/10 |
| verifier clean | 0/10 | 0/10 |
| PLAUSIBLE_PATCH | **4/10** | 0/10 |
| failure classes | 5 COUNT_MISMATCH, 1 EMPTY | 8 EMPTY, 2 COUNT_MISMATCH |

cgpro decision rule (locked 2026-06-11): *"si Arm D ≤ Arm A sur non-empty +
applyability + verifier-clean → bascule C, diagnostic produit
patch-generation."* Arm D is ≤ Arm A on EVERY axis, by a wide margin —
**two direct reasoner calls with naive `git ls-files` localization beat
the full adaptive orchestration 9-vs-2 on patch production and 4-vs-0 on
applicability**, at comparable per-task cost (~$0.10-0.15 both arms).
This is the project's first paired internal evidence for the
minimal-scaffold thesis (Mini-SWE-Agent/Live-SWE-agent class) the
2026-06-10 analysis flagged from leaderboards.

## Arms

- **Arm A v3** — Agentless-lite controlled baseline: call 1 localizes
  files over the capped `git ls-files` tree; call 2 emits a strict
  unified diff over the selected file contents (≤6 files, ≤60k chars);
  same worktree, verifier/repair chain, and `git apply --check` as D.
  Localization quality was striking: e.g. tutanota-219 → its LLM picked
  `EventBusClient.ts`, the exact file of the known TS2551 baseline error.
  NOT the cycle-13 wiring doc's Claude-Code arm A (that one asks a
  product-competitiveness question; this one isolates the
  pure-orchestration delta).
- **Arm D** — the canary runner unchanged (reasoner tier, repair mode,
  $5/task), uniform post-hoc apply-check.

## Caveats (recorded, none change the verdict's direction)

- Arm D's 2/10 here vs 3/5 on 2026-06-11 morning (same first-5): same-day
  variance and provider-window effects are real; even taking D's BEST
  observed day (3/5 non-empty, 0/5 apply), A's 9/10 + 4/10-applying
  dominates.
- Resolution is UNGRADED by design (mini contract). The 4 applying arm-A
  patches are one `remote-grading.yml` dispatch (~$0.04) away from a
  resolution datapoint — proposed as the first evidence item of the C
  block.
- Two invalid arm-A attempts preceded v3 (archived `-invalid-ssl`; SSL
  in-process truststore + asyncio single-loop fixes, both committed).

## Spend

Mini block total ≈ **$2.8** (D $1.0 + A attempts ~$1.8), within the
~$3 envelope; cumulative Phase 2.a+mini ≈ $5.0 of the $30 cap.

## Next (C block, cgpro consultation pending)

Product diagnosis of arm D's patch generation. The mini's data already
points at suspects: D produces EMPTY 8/10 where A produces patches from
the SAME models — the orchestration's prompt/contract chain (topology
node prompts, output requirements, step budgets) is destroying emission,
not the models. Candidate first probes: (1) grade A's 4 applying patches
(resolution ceiling check); (2) diff the EXACT prompts/budgets the
pipeline's coder node received vs arm A's two prompts on one instance;
(3) the patch_focused contract inside multi-node context.
