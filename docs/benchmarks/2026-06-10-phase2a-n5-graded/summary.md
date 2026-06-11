# Phase 2.a — N=5 SWE-bench Pro graded canary (2026-06-10)

**Decision: `GATE_2A_MET_NO_GO_N50`** — the Phase 2.a plumbing gate is met
(3/5 patches, all 6 acceptance gates PASS, every B2 fix validated live), the
official resolution signal is 0/5, so the N=50 4-arm run ($240-460) stays
NO_GO and Phase 2.b needs the resolution bottlenecks addressed first (see
"Next steps").

- **Frozen commit**: `b2b40b3021d844d009fc35bc690ccb9a561f05c0` (HEAD at
  launch; CI/Security/coverage/fuzz/coherence ALL green — `ci_green.json`)
- **Authorization**: explicit paid GO from Yann 2026-06-10 (expected ~$1,
  hard caps $5/task + $30 global per the refrozen
  `docs/benchmarks/cycle-13-canary-manifest.md`)
- **Instances**: same sha-pinned 5 as 2026-05-12
  (`b0ea91c4…`, comparability run-over-run)
- **Config**: tier `reasoner`, prompt `patch_focused`, profile
  `graded_patch_generation` (900s/task), allowlist google+deepseek,
  denylist openai
- **Generation cost**: **$0.4748** (vs $0.79 real on 2026-05-12); wall 751s.
  Grading cost: **$0.0415 measured** (`modal_billing.json`
  status=measured, 1 billing row for app `ap-LhzIeBC5TBPQ4BumsWJEeD`,
  re-captured 2026-06-10 evening after Modal aggregation lag).
  **Total Phase 2.a real spend: $0.5163.**

## Headline vs 2026-05-12 (the B2 rerun this is)

| Metric | 2026-05-12 | 2026-06-10 |
|---|---|---|
| Patches extracted | 1/5 | **3/5** (gate ≥3/5 MET) |
| provider_gate | NO_GO (`unknown` execution) | **PASS** (`[deepseek, google]`, zero unknown) |
| Cost integrity | $0.30 reported vs $0.79 real (2.6×) | `_total_cost_usd_source` explicit on 5/5 (`cli_complete` everywhere; no under-report observed) |
| `_diff_verifier_outcome` | None on 5/5 | non-null on 5/5 (2 real diagnoses + 3 explicit skips) |
| Timeouts | 1 (+ the rest of the run degraded) | 0 |
| Official resolved | not graded (gates failed) | **0/5** (first fully-graded N=5 in project history) |

All three B2_RERUN_UNBLOCKERS fixes are validated in production. The
runtime policy chain also showed its I-11 witness behavior live: the Rust
SystemRouter proposed `gpt-5.4-pro` (openai) as routing candidate on 2 tasks;
policy re-evaluation rejected it; zero openai execution; the witness chain
recorded candidate → rejection explicitly.

## Per task

| Instance | Patch | Cost | Verifier | Graded |
|---|---|---|---|---|
| protonmail__webclients-0200ce0f | — | $0.0328 | `skipped_no_patch` | false (empty patch) |
| gravitational__teleport-6eaaf3a2 | ✓ 4.9KB | $0.2085 | `skipped_no_repo_dir` (clone timeout → agent ran from YGN root) | false — patch applied but **build failed** (`lib/benchmark [build failed]`): blind-generated patch broke compilation |
| tutao__tutanota-219bc8f0 | ✓ 10.6KB | $0.0824 | `hunk_body_count_mismatch` | false — `make` exit 2 after apply; the verifier-flagged context mismatch materialized |
| NodeBB__NodeBB-76c6e302 | — | $0.0715 | `skipped_no_patch` | false (empty patch; grader's local result-write crashed on cp1252 vs emoji — see incidents — result still false) |
| tutao__tutanota-db90ac26 | ✓ 7.9KB | $0.0795 | `hunk_body_count_mismatch` | false — applied, tests ran, f2p not resolved (genuine difficulty) |

**The observe-mode diff verifier predicted both application-class failures
in advance** (`hunk_body_count_mismatch` on exactly the two patches that
broke at apply/build time). This is the strongest empirical argument to date
for flipping repair mode (budget wiring shipped in cycle-13 K Block D,
`8fbeeb1f`).

## Acceptance gates (manifest)

manifest_gate PASS · ci_gate PASS · grading_gate PASS (READY_MODAL) ·
provider_gate PASS · budget_gate PASS ($0.47/$30) · timeout_gate PASS (0) ·
learning_evidence_gate PASS (5/5) · pre-grader gate PASS (5/5 classified,
empty patches carry explicit `no_patch_extracted`).

## Resolution bottlenecks (ranked by evidence)

1. **Repo-context loss** — teleport's clone timed out (180s clone budget);
   the agent generated a plausible-looking patch with NO repo grounding and
   it broke the build. Fix: raise/parallelize clone budget, fail-closed to
   skip-patch instead of blind generation, or pre-fetch repos.
2. **Patch application quality** — both tutanota patches carried verifier
   mismatches that materialized at the grader. Fix: flip
   `SAGE_DIFF_VERIFIER_MODE=repair` (one-shot LLM realign with mismatch
   feedback, already built + budget-capped).
3. **Genuine difficulty** — tutanota-db90 applied cleanly and still failed
   f2p. Budget-tier models on Pro tasks; SOTA is 59.1% with frontier
   scaffolds. No cheap fix; this is what Phase 2.b measures comparatively.

## Incidents (environment, not runtime)

- **Local pydantic lockstep skew** (first launch, $0 lost): an interrupted
  pip retry during the disk-full window had left `pydantic-ai-slim` 1.106 /
  `pydantic-graph` 1.84.1 → CLI subprocess boot-failed with
  `No module named 'pydantic_graph.util'`. The B2 fixes reported it cleanly
  (kind=boot failure event, honest $0, run continued). Fixed by aligning
  the full pinned set from `sage-python/constraints.txt`; CI was never
  affected (clean installs).
- **Corporate-network TLS interception**: Python 3.13 strict validation
  rejects the proxy's MITM CA (missing AKI). SAGE itself is immune
  (`boot.py:33` truststore). The Modal CLI is not — fixed with a
  truststore-injecting shim (`.tmp/modal-shim/modal.bat`) used for
  preflight, grading and billing. Lesson recorded in memory.
- **Grader cp1252 crash** (cosmetic): `swe_bench_pro_eval.py` crashed
  writing NodeBB's emoji-bearing log on Windows (UnicodeEncodeError) AFTER
  the sandbox ran; `output.json` for NodeBB lost locally, result (false)
  unaffected. Next grading run should set `PYTHONIOENCODING=utf-8`.
- 4f08b2fd CI Windows flakes (sandbox::subprocess, host-python) → issue #26.

## cgpro post-run verdict (2026-06-10, conv `cgpro_b2_unblockers_verify`)

`GO_NEXT_BLOCK` / `NO_GO_DIRECT_2B` / `NO_GO_N50`. Sequencing validated,
with two real traps caught in my proposed "repair flip":

1. **Env-flip alone is a NO-OP on the canary path** —
   `_annotate_diff_verifier` only calls `verify_diff_context_with_reasons`
   in both modes; the actual LLM repair (`repair_with_verifier_feedback`)
   must be wired explicitly post-extraction pre-cleanup.
2. **`hunk_body_count_mismatch` is a structural reason event, not a
   mismatch object** — the repair helper skips on an empty `mismatches`
   list, so naive repair would skip exactly the two cases that predicted
   the graded failures. Repair input must consume `_diff_verifier_reasons`
   (or convert reason events to repairable diagnostics).

Four extra signals cgpro read in the graded bundle: db90's stdout carries
precise TS2554/TS2345 test-compile errors (repair-exploitable, not opaque
difficulty); 219's first causal signal is `TS2551 '_onMessage' →
'_message'` before the make noise; teleport's `output.json` collapses to
`NO_TESTS_FOUND_OR_PARSING_ERROR` while stdout has the real
`[build failed]` (post-grader parser should distinguish BUILD_FAILED /
PATCH_APPLY_FAILED / TEST_FAILED / EMPTY_PATCH / GRADER_ENCODING_FAILURE);
negative grading cases should always write a minimal machine-readable
output.json wrapper-side.

## Next block (locked)

**`RESOLUTION_UNBLOCKERS_REPAIR_WIRING_PLUS_REPO_CONTEXT_FAIL_CLOSED`** —
exit criteria before any re-canary:

- [ ] no repo ⇒ skip generation (no paid blind patch);
      `--allow-blind-generation-on-repo-failure` escape, off by default
- [ ] clone/fetch failure reason explicit; `repo_unavailable` reported
      separately from `model_no_patch` (infra failure ≠ LLM-quality signal);
      partial/shallow fetch policy rather than just a bigger timeout
- [ ] repair path actually invoked in the canary (not env-only)
- [ ] `hunk_body_count_mismatch` ⇒ actionable repair feedback or explicit
      non-repairable outcome
- [ ] `first_compiler_error` / `first_test_error` captured per instance —
      **scope note (cgpro VERIFY 2026-06-11 edit 3)**: these are
      POST-GRADER advisory fields (extracted by
      `swebench_pro_post_grader_parse.py`), inputs to a FUTURE
      grading-feedback regenerate loop; the generation-time repair chain
      of this block consumes diff-verifier feedback only
- [ ] all negative grading cases write machine-readable output

Then: re-canary N=5 (paid, same sha-pinned instances, ~$0.50, fresh GO
required). 2.b decision after: ≥1/5 resolved OR a clean 0/5 whose failures
are genuinely reasoning/test-class, not plumbing. N=50 stays NO_GO.

Also queued: re-capture Modal billing once rows aggregate.
