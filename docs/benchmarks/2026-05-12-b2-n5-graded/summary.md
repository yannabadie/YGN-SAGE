# B2 N=5 SWE-bench Pro graded canary — 2026-05-12

**Status**: `NO_GO_N50` — patches_extracted=1/5 below the manifest acceptance threshold (≥3/5 required).

**Commit**: `7928dd560bb6a278e3a8b1713b36c5c52422766a` (CI green: 6/6 workflows, 11/11 CI jobs)
**Manifest**: `docs/benchmarks/cycle-13-canary-manifest.md` (frozen at full 40-char SHA, all 3 acceptance gates PASS: manifest_gate / grading_gate / ci_gate)
**Decision**: `canary_decision: NO_GO` (combined from `provider_gate: NO_GO` + `patches_extracted=1`)

## Pre-launch posture

- HEAD `7928dd56` after 4 commits this session:
  - `0cdf253c` drop pydantic-ai umbrella (mistralai PyPI quarantine)
  - `c3d36b62` regen constraints from per-package cwd
  - `31510440` 2 latent gate fixes:
    1. `security.yml` audit filter skips `pywin32==` (Windows-only, no Linux wheel)
    2. `_load_grader_gate` accepts `READY_MODAL` decision string + `modal_grading_ready=True` boolean (preflight script emits `READY_MODAL` since `a7474306` 2026-05-10, but gate had been speculatively typed `READY_REMOTE_MODAL` since `ec0b775e` same day — every Modal preflight ever produced was silently rejected with `grader_preflight_not_ready` until this fix)
  - `7928dd56` Linux-regen constraints under WSL Ubuntu 24.04 + cap `mypy<2` (mypy 2.x narrows `isinstance(x, (list, tuple))` to `tuple[Any,...] | list[Any]` breaking `extended_drift.py:229`)
- cgpro VERIFY `cgpro_i11_design_20260511` returned `EDIT_REQUIRED` with 5 conditions; all 5 satisfied before launch (manifest aligned, `--swebench-prompt-profile patch_focused` added, `--task-timeout-s` dropped, `ci_green.json` derived from real GitHub Actions runs, fresh `grader_preflight.json` post-edits).
- Launch parameters: tier=reasoner, budget=$5/task, global=$30, timeout=900s, profile=graded_patch_generation, prompt_profile=patch_focused.

## Per-task results

| # | Instance | Outcome | Cost | Latency | Patch | LEB | Final provider |
|---|----------|---------|------|---------|-------|-----|----------------|
| 1 | protonmail/webclients-0200ce0fc | `success` | $0.099 | 183s | **3607 chars** | pass | deepseek (`deepseek-v4-flash`) |
| 2 | gravitational/teleport-6eaaf3a2 | `failure` | $0.000 | — | 0 | **no_go** `cli_outcome_not_success` | google (`gemini-2.5-flash`) |
| 3 | tutao/tutanota-219bc8f0 | `failure` | $0.000 | — | 0 | **no_go** `cli_outcome_not_success` | **unknown** (`gemini-3.1-pro-preview` reasoner) |
| 4 | NodeBB/NodeBB-76c6e302 | `success` | $0.066 | 401s | 0 | pass | deepseek (`deepseek-v4-flash`) |
| 5 | tutao/tutanota-db90ac26 | `success` | $0.134 | — | 0 | pass | google (`gemini-3-flash-preview`) |

**Cumulative**: $0.298 spent ($30 cap), 1649s = 27.5 min wall-clock.

## Findings

1. **Provider gate failed on the reasoner tier** (`unknown` in `execution_outside_allowlist`). Task 3 (tutanota 219bc8f) emitted `provider_final=unknown` when `model_id_final=gemini-3.1-pro-preview` (the explicit reasoner). The provider catalog in `sage-core/config/cards.toml` should map gemini-3.1-pro-preview → google, but the runtime saw `unknown`. This is a real production bug: the reasoner tier is the configured target per Option B (2026-05-12), and getting `unknown` provider means the routing/policy layer lost the provider binding for that model. Follow-up ticket required: trace where the `unknown` provider label is emitted in the runtime (likely a missing entry in cards.toml or a `_provider_for_model` lookup fall-through for the `-preview` suffix).

2. **Patch extraction rate 1/5 (20%)** — same order of magnitude as cycle-13 E single-task smokes (1 patch / 1 task with NodeBB at HEAD `db304bc6` on 2026-05-06) but a stark drop from the cgpro DESIGN E theoretical 3/5 baseline. With only 5 samples the 95% CI on pass rate is wide; this is a directional signal, not a definitive measurement.

3. **2 fast-fail tasks** (gravitational + tutanota 219bc8f) — `cli_outcome_not_success` with $0 cost suggests the CLI returned outcome=failure before significant LLM work, consistent with either (a) early provider-policy failure or (b) the model giving up at the EMPTY_STEP_SENTINEL (the bidirectional trap from Option B — but Option B identified that for fast/budget tiers, not reasoner). The tutanota 219bc8f failure is especially worrying because the `unknown` provider tag indicates the runtime never had a working provider for the reasoner — provider lookup failed BEFORE any LLM call.

4. **Diff verifier never fired** — `_diff_verifier_outcome=None` across all 5 tasks even though `SAGE_DIFF_VERIFIER_MODE=observe` was set. Either the env didn't propagate to the canary subprocess or the verifier is gated on something else. Investigation needed.

5. **Learning evidence boundary**: 3/5 pass / 2/5 no_go, gated correctly by `cli_outcome_not_success` for the 2 hard failures. The two empty-patch successes (NodeBB, tutanota db90ac26) DID pass LEB — meaning the agent gave up cleanly with proper trace records, just no patch. The gate did its job: blocked the no-success traces from claiming learning evidence.

## Cycle-13 closeout impact

This is the first real B2 N=5 SWE-bench Pro graded canary that closed all preflight gates and produced replayable, auditable artefacts on real LLM spend. The verdict `NO_GO_N50` is the correct gating decision — but it surfaced two real production bugs (unknown provider on reasoner, diff verifier silent) that were not visible from the prior cycle-13 E single-task smoke.

The decision NOT to escalate to N=50 is gated by:
- patches_extracted=1 < 3 threshold (manifest stop condition #1 — empty patches)
- provider_gate NO_GO (unknown provider in execution path)

Next actions (not in this commit):
- File issue: `unknown` provider when `model_id_final=gemini-3.1-pro-preview`
- File issue: `_diff_verifier_outcome=None` when `SAGE_DIFF_VERIFIER_MODE=observe`
- After fixes: re-run B2 N=5 with the same canary manifest at the new HEAD

## Artefacts

- `run/predictions.json` — Pro grader input format (5 records, 1 with patch_len=3607, 4 empty)
- `run/predictions.jsonl` — internal trace format
- `run/events.jsonl` — aggregate runtime events
- `run/summary.json` — gates + canary_decision + task_summaries
- `run/launch_manifest.json` — frozen at run-start, all 3 acceptance gates PASS
- `run/launch_manifest.md` — copy of canary manifest at run-start
- `run/per_task/*.events.jsonl` — per-task event traces (5 files)
- `run/per_task/*.trace/` — per-task SAGE CLI run dirs
- `grader_preflight.json` — `READY_MODAL` at HEAD 7928dd56
- `ci_green.json` — 6 workflow runs all success at HEAD 7928dd56 (derived from GitHub Actions runs, not auto-attested)
- `cycle-13-canary-manifest.md` (root) — manifest frozen at full SHA `7928dd56...`
