# B2 step 4 — Modal grader N=1 infra validation (empty patch)

**Status**: B2 step 4 CLOSE. B2 step 5 (post-run gates) + B2 final close-out (canary timeout extension OR task-difficulty triage → N=5 with non-empty patches) remain open.
**HEAD at run**: `df0f4e5fc91d543b15ba8c29668596d185265756` (working tree clean except pre-existing `.claude/settings.local.json` from previous session).
**Plan reference**: `docs/superpowers/plans/2026-05-10-handoff-recovery-plan.md` block B2 `first-graded-swebench-pro-n5`.

## What this run proves

End-to-end Modal grader infra works on a real SWE-bench Pro prediction:
`predictions.json → swe_bench_pro_eval.py (Modal mode) → Modal Sandbox → image pull → entryscript → output.json → eval_results.json`.

This is the **first real SWE-bench Pro Modal-graded result** in YGN-SAGE history. Prior cycle-13 E result (`b5fbe064`) was local-Docker on a different code path.

## What this run does NOT prove

- Not a B2 ACCEPTANCE_GATE close. B2 requires N=5+ with provider/model non-null for each attempt and `grader résultat ≠ NO_GO` on tasks that **produced actual patches**.
- Resolution metric here (0/1) is uninformative because the SAGE canary that generated the prediction timed out at 300s before extracting any patch. Empty patch → predictable resolved=0.
- Not a runtime contract change. Pure infra + docs ticket.

## Run timeline

- 09:33:42 UTC — canary `run_dryrun_arm_d.py --limit 1 --task-timeout-s 300` started (commit `df0f4e5f`)
- 09:38:42 UTC — canary completed, 0 patches extracted, prediction written with `patch=""`
- ~09:54   UTC — first grader launch from repo root → failed (working dir issue: `dockerfiles/base_dockerfile/<iid>/Dockerfile` is hardcoded relative in `swe_bench_pro_eval.py:58`)
- ~09:55   UTC — second grader launch from `external/SWE-bench_Pro-os/` with absolute paths
- 10:00   UTC approx — Modal Sandbox image pull (`jefzda/sweap-images:future-architect.vuls-...`)
- ~10:02   UTC — entryscript ran, output.json produced, sandbox torn down
- Run wall-clock: 177.33s (Modal sandbox creation + image pull + Go test setup + collect)

## Artefacts

```
docs/benchmarks/2026-05-11-canary-n1-graded-eval/
├── summary.md                          (this file)
├── eval_results.json                   ({instance: false})
└── instance_future-architect__vuls-139f3a81b66c47e6d8f70ce6c4afe7a9196a6ea8/
    ├── sage_patch.diff                 (0 bytes — empty patch, as sent)
    ├── sage_entryscript.sh             (3084 bytes — git reset/checkout/apply/run, no errors at script level)
    ├── sage_stdout.log                 (4973 bytes — Go test setup failure, see below)
    ├── sage_stderr.log                 (0 bytes — no script-level errors)
    └── sage_output.json                ({"tests": []} — parser returned empty test list)
```

All file mtimes are AFTER the residual `eval_results.json` from run 1 was deleted at epoch 1778486287, confirming these are fresh run-2 artefacts.

## Why resolution=0 in detail

The entryscript executes:
1. `git reset --hard d1a617cfff04` (base commit)
2. `git checkout d1a617cfff04`
3. `git apply -v /workspace/patch.diff` (our empty patch — silent no-op)
4. `git checkout <target_commit> -- scanner/base_test.go` (restore the test file from the target commit — SWE-bench Pro's test_patch mechanism)
5. `bash /workspace/run_script.sh <fail-to-pass test list>` (run tests)
6. `python /workspace/parser.py stdout.log stderr.log output.json`

With the empty patch, step 4 restores a test file that imports `github.com/aquasecurity/trivy/pkg/fanal/analyzer/language/dotnet/deps` — a package only present after the canary's *intended* patch (Vuls Trivy 0.30.x upgrade). Without the patch, `go test` setup fails:

```
scanner/base_test.go:7:2: no required module provides package github.com/aquasecurity/trivy/pkg/fanal/analyzer/language/dotnet/deps
FAIL github.com/future-architect/vuls/scanner [setup failed]
```

All non-`scanner` packages run with `[no tests to run]` because the test filter is scoped to fail-to-pass test names that live in `scanner/`. `parser.py` then emits `{"tests": []}` since no test executions were recorded, and the grader marks `eval_results[instance] = false`.

This is the **infra-OK + empty-patch = resolved-0** outcome documented in the post-run gate triage. The alternative branch (sandbox unreachable / Modal token expired / image pull fail) would have left `sage_output.json` absent and `eval_results[instance] = False` from a different code path (`swe_bench_pro_eval.py:547`, "future.result() returned None"). We're in the former, not the latter.

## Acceptance gate status

| Sub-gate | Status | Evidence |
|---|---|---|
| Modal preflight `READY_MODAL` | PASS | `docs/benchmarks/2026-05-11-grader-preflight-5a7ed115-modal.json` |
| Predictions shape `{instance_id, patch, prefix}` | PASS | `predictions.json` matches grader `main():492` filter |
| Modal Sandbox creation + image pull | PASS | `Using Docker Hub image: jefzda/sweap-images:future-architect.vuls-...` |
| Entryscript ran end-to-end | PASS | stderr.log = 0 bytes, exit code 0, output.json produced |
| Resolution metric meaningful | N/A | Empty patch by construction; metric reserved for B2 final close |
| B2 final ACCEPTANCE_GATE | NOT MET | B2 final remains open: requires N=5+ graded run with provider/model non-null, prediction outcomes classified, grader result not NO_GO, and any empty patches/errors explicitly accounted for. This run validates the grader path only. |

### Infra-OK vs silent-fail discriminator (full evidence tuple)

Per cgpro VERIFY 2026-05-11: presence of `sage_output.json` alone is a primary signal but not sufficient. The full evidence tuple distinguishing **case A (grader executed, produced empty test result)** from **case B (sandbox output missing, `collect_outputs_modal` returned None at `swe_bench_pro_eval.py:243-246`, then line 547 writes `False`)** is:

1. `eval_results.json` exists and marks the instance `false` ✓
2. `sage_output.json` exists with `{"tests": []}` ✓
3. `sage_stdout.log` exists and contains the Go scanner setup failure ✓
4. `sage_entryscript.sh` exists (3084 bytes, full reset/checkout/apply/run flow) ✓
5. Modal sandbox creation + image pull occurred (per stdout: `Using Docker Hub image: jefzda/sweap-images:future-architect.vuls-...`) ✓
6. **No** `output.json not found / collect_outputs_modal returned None` path used (`swe_bench_pro_eval.py:243-246` would print `Warning: output.json not found for {uid}`; absent from our stdout) ✓

This is **case A**. The upstream evaluator at `swe_bench_pro_eval.py:236-247` computes pass/fail from `output["tests"]` when output exists, and returns `None` when output.json is missing (`FileNotFoundError` branch). Our run took the former.

## Cost / latency

- `modal_app_id = ap-LhzIeBC5TBPQ4BumsWJEeD` (`swe-bench-pro-eval`, deployed under workspace `yann-abadie` 2026-05-11 09:55 Paris)
- `modal_sandbox_id` = sandbox_id_not_captured (the upstream evaluator does not log it to stdout; `modal app history` and `modal app stats` do not expose per-sandbox cost)
- `modal_cost_usd` = unmeasured. Modal prices Sandbox compute per second; this single sandbox ran ~177s wall but the actual billed-CPU-second and image-pull bandwidth cost is only visible on the Modal web dashboard. **Cost must be checked in Modal dashboard before N=5 budget accounting.**
- Local cost: 0 LLM tokens (no provider call during grading).
- Wall-clock: 177.33s (per tqdm).

## What's still open for B2 final close

1. **Canary timeout extension** — the 300s budget at `scripts/run_dryrun_arm_d.py --task-timeout-s 300` was insufficient on the Vuls Trivy task (substantial codebase upgrade). Options: (a) raise to 600-1200s for canaries on hard tasks, (b) stratify task selection by base-commit→target-commit diff size so canary N=5 picks lighter tasks first.
2. **Re-run canary N=5** with one of the strategies above so we get N=5 predictions with at least some non-empty `patch` fields and non-null `provider_final`.
3. **Modal grader on N=5** — same invocation, just a larger predictions.json.
4. **Post-run gate** — write the N=5 graded summary doc (this template scales to N=5).
5. **cgpro VERIFY post-N=5** — resume `cgpro_ygn_sage_global_analysis_20260510` with the graded artefacts.

## Operational notes

- **Working-directory assumption in `swe_bench_pro_eval.py`** (NOT a definitive upstream bug; the SWE-bench Pro README shows usage from inside the evaluator repo): `load_base_docker(iid)` and `instance_docker(iid)` at lines 58 and 62 use the relative path `dockerfiles/base_dockerfile/{iid}/Dockerfile`. Invoked from YGN repo root, this lookup fails and `swe_bench_pro_eval.py:547` writes `False` without surfacing the dockerfile error. **YGN wrapper must `cd external/SWE-bench_Pro-os/` and pass absolute paths for `--raw_sample_path`, `--patch_path`, `--output_dir`.** Optional upstream robustness PR (make the grader cwd-independent) is a possible follow-up alongside the existing Windows CRLF/UTF-8 spec at `.tmp/swebench_pro_windows_newline_fix_spec.md`, but not required for B2 close.
- **HF dataset CSV** at `.tmp/swebench_pro_data/swe_bench_pro_full.csv` (23.9 MB, 731 rows, gitignored). Re-gen via `python -c "from datasets import load_dataset; load_dataset('ScaleAI/SWE-bench_Pro', split='test').to_csv('<path>')"`.
- **`_diff_verifier_outcome=None` in predictions.jsonl**: per `.claude/rules/development.md`, `SAGE_DIFF_VERIFIER_MODE=observe` is the recommended default. The canary at step 2 produced `_diff_verifier_mismatches=None` and `_diff_verifier_outcome=None` because the patch was empty (no hunks to verify). **This means no verifier decision was produced for the empty patch — it does NOT imply the verifier passed.** Follow-up (not a blocker for step 4 infra validation): emit explicit `reason_code="no_patch_to_verify"` for the empty-patch case before B2 final close or block B3 `provider-execution-evidence-v0`, so empty-patch runs leave a positive verifier trace instead of a `None` ambiguity.
