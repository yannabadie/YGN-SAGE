# Clean re-canary N=5 — set v2, repair-mode reasoner (2026-06-11)

**Decision: `CLEAN_ZERO_RESOLVED_PRODUCT_CLASS` — the cgpro "clean 0/5"
arm is REACHED.** Real grades obtained via the new `remote-grading.yml`
GitHub-runner workflow (run 27339674783) after the local network blocked
Modal streams: **0/5 resolved, and for the first time EVERY failure is
product-class, none plumbing**:

| Instance | Graded verdict | Class |
|---|---|---|
| protonmail | **TEST_FAILED** (its first-ever patch APPLIED, tests RAN, f2p unresolved) | reasoning/patch quality |
| NodeBB | **TEST_FAILED** (applied, tests ran, `AssertionError` in suite) | reasoning/patch quality |
| teleport | BUILD_FAILED (count-mismatched patch broke the Go build; repair not-improved) | patch quality |
| qutebrowser | EMPTY_PATCH | model (no patch) |
| tutanota-db90 | EMPTY_PATCH | model (no patch) |

Mechanical stop-rule (0/5 AND model-empties ≥3 AND no infra): **does NOT
trigger** (model-empties = 2). The cgpro 2.b criterion "un 0/5 propre où
les échecs restants sont réellement des échecs de raisonnement/test" is
**satisfied** — the plumbing campaign is COMPLETE; the next decision
(Phase 2.b arm A-vs-D vs product diagnosis of patch generation) is a
cgpro/Yann strategy call, not an engineering blocker.

Grading spend today (Modal, measured): $0.0939 across the blocked
attempts + the successful GH-runner run.

- **Frozen commit**: `9bd885ce20629bd18e7ca031ac48a08513361b33`
  (CI/Security/coherence/coverage/fuzz ALL green)
- **Set**: `phase2a_n5_v2` (qutebrowser replacing the baseline-failed
  tutanota-219; sha256 `35ddf390…`)
- **Config**: `--verifier-mode repair --repair-tier reasoner
  --repair-budget-usd 0.50 --repair-timeout-s 180`, keep-awake guard
  active (held through the full 985s run after two suspend-killed
  attempts)
- **Generation spend**: **$0.5308** (+ ~$0.16 archived partial from the
  suspend-killed first attempt at `run-interrupted-0833-suspend/`;
  grading sandbox spend ~$0.08 across the two blocked attempts)

## Generation results (VALID)

| Instance | Patch | Verifier → repair chain |
|---|---|---|
| protonmail | ✓ ($0.297) — **first patch ever on this instance** | `unsupported_no_opinion` (verifier had no opinion; repair correctly not attempted) |
| teleport | ✓ ($0.073) | `hunk_body_count_mismatch` → mechanical recount → **reasoner LLM replied at 180s** → re-verified NOT clean → `verifier_repair_not_improved`, original kept (clean-strict working as locked) |
| qutebrowser | — ($0.058) | `skipped_no_patch` (model-class empty) |
| NodeBB | ✓ ($0.017) | `hunk_body_count_mismatch` → `verifier_repair_empty` |
| tutanota-db90 | — ($0.086) | `skipped_no_patch` (model-class empty) |

patches_empty: 2 (infra=0, model=2) · 6/6 acceptance gates PASS ·
learning 5/5 · 0 timeouts · keep-awake `ES_SYSTEM_REQUIRED` set and
cleared cleanly.

Repair-chain progress vs the morning run: the 401 class is FIXED (the
reasoner call authenticates and, at 180s, actually returns content —
teleport's reply was extracted, re-verified and rejected by clean-strict,
exactly the locked semantics). `_verifier_repair_usage` still lands None
on live providers — telemetry follow-up, not gate-bearing.

## Grading status (BLOCKED — not a resolution verdict)

Two grading attempts (workers=3, then workers=1 + `--redo`) failed with
`grpclib StreamTerminatedError: Stream reset by remote party` on the
long-lived Modal sandbox streams; per-instance output collection returned
None for the patched instances. The same grader invocation succeeded at
06:50 from a different network — this is the corporate-middlebox
gRPC/h2 class diagnosed 2026-06-10 (short RPCs like `modal token info`
pass; long streams die). `eval_results.json`'s `false` entries for the
patched instances are COLLECTION failures, not grades; the post-grader
parser correctly bucketized them as `GRADER_OUTPUT_WRITE_FAILED`
(3 patched) + `EMPTY_PATCH` (2) — the taxonomy's last bucket doing its
job.

**Re-grade from a clean network** (exact command, ~$0.04, ~10 min):

```
cd external/SWE-bench_Pro-os && \
MODAL_TOKEN_ID=… MODAL_TOKEN_SECRET=… PYTHONUTF8=1 \
python -c "import sys, truststore, runpy; truststore.inject_into_ssl(); \
sys.argv=['swe_bench_pro_eval.py', \
 '--raw_sample_path', '<bundle>/grader_n5.csv', \
 '--patch_path', '<bundle>/run/predictions.json', \
 '--output_dir', '<bundle>/grading', \
 '--dockerhub_username', 'jefzda', \
 '--scripts_dir', '<repo>/external/SWE-bench_Pro-os/run_scripts', \
 '--num_workers', '3', '--redo']; \
runpy.run_path('swe_bench_pro_eval.py', run_name='__main__')"
```

Then: `swebench_pro_post_grader_parse.py` + the cgpro stop-rule
evaluation (0/5 resolved + `patches_empty_model ≥ 3` + no infra class ⇒
STOP canary loop; here model-empties = 2, so a 0/5 would NOT trigger the
mechanical stop — cgpro consultation either way).

## Incident log

- First v2 attempt (frozen `97fff357`) killed mid-task-3 by Windows
  Modern Standby → keep-awake guard added (`prevent_os_sleep`, cycle-9)
  + `--repair-timeout-s` (reasoner repair was timing out at the
  budget-derived 60s). Partial archived.
- Grading attempt 1 launched against a missing v2 grader CSV (caught in
  seconds, $0); CSV built from the v2 ids, relaunched.
- Grading attempts 2-3 network-blocked as above.
