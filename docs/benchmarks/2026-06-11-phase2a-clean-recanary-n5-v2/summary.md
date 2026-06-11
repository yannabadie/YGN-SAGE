# Clean re-canary N=5 — set v2, repair-mode reasoner (2026-06-11)

**Decision: `GENERATION_VALID_GRADING_BLOCKED_NETWORK`** — the generation
run is complete and gate-clean (**3/5 patches** on the health-screened v2
set, all infra classes green); the Modal grading is BLOCKED at the current
network location by reproducible gRPC stream resets and must be re-run
from a clean network before any resolution verdict. The cgpro stop-rule
CANNOT be evaluated yet (it needs real grades).

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
