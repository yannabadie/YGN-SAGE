# Slice 9 closure — canary-patch-focused-prompt-profile N=5

**Status**: cgpro DESIGN ACCEPTANCE GATE met — first canary chain to produce non-empty patches on all 5 instances.
**HEAD at run**: `3b182682` (slice 9 commit) — verified via `_observed_event_cost_usd` sums.
**Plan reference**: `docs/superpowers/plans/2026-05-10-handoff-recovery-plan.md` block B2 follow-up; cgpro PIVOT DESIGN 2026-05-11 NEW_SLICE.
**cgpro conv**: `cgpro_ygn_sage_global_analysis_20260510` (id `6a00a5a1-96e8-8396-ad88-46d0c6b46623`).

## Empirical chain (cycle-13 canary, 5 slices)

| Slice | Commit | Result | Root cause exposed |
|---|---|---|---|
| 1-5 (foundation) | `05b01d4c` ... `77b6dd98` | Pre-grader gate + profile + difficulty triage + Modal cost capture | scaffolding only |
| 6 (post-cli_complete hang) | `6e0609ba` | wall 75min → 24min on N=5 | subprocess held stdout open after cli_complete |
| 7 (SWEBENCH_SYSTEM_TEMPLATE) | `b28cccc1` | 0/5 patches, output_length 51-1301 (sentinel/short) | mandate ≥3 tool calls + no repo on disk |
| 8 (repo clone + cwd + dotenv + TRACE_RAW) | `6317d42a` | provider gate passes, output 1890 chars (avr) / 51 (debate sentinel) | tool calls succeed but bandit picks debate which can't satisfy tool quota |
| **9 (patch_focused prompt)** | **`3b182682`** | **5/5 patches, $0.81, 21min wall** | **DROPS "≥3 tool calls" mandate while keeping repo-grounding + strict diff** |

## Slice 9 acceptance — full evidence

| Sub-gate (cgpro DESIGN) | Required | Actual | Status |
|---|---|---|---|
| 5/5 `repo_context_status=ready` | yes | 5/5 ready, fetch_fallback used per instance | ✅ |
| `topology_override_used=false` for every task | yes | 5/5 false (summary.prompt + per-task prompt_metadata) | ✅ |
| `system_hint_forced=false` for every task | yes | 5/5 false | ✅ |
| Empty patches with explicit reason codes | n/a | 0 empty patches — moot | ✅ |
| At least 1/5 non-empty patch | yes | **5/5 non-empty** | ✅ exceeded |
| Pre-grader gate PASS | yes | 5/5 verdict=pass:non_empty_patch (exit 0) | ✅ |
| Modal preflight READY_MODAL | yes | confirmed earlier today | ✅ |

## Per-task evidence

| Instance | Topology | Nodes | Output chars | Patch chars | Cost | Latency |
|---|---|---|---|---|---|---|
| protonmail/webclients | sequential | 2 | (redacted, but >1145) | 1145 | $0.16 | 215s |
| gravitational/teleport | sequential | 3 | ~1891 | 1880 | $0.15 | 333s |
| tutao/tutanota 219bc | sequential | 3 | ~3017 | 3006 | $0.12 | 267s |
| NodeBB | sequential | 2 | ~1206 | 1195 | $0.19 | 211s |
| tutao/tutanota db90 | (TBD, likely sequential) | ? | ~1574 | 1574 | $0.18 | ~250s |

**Topology distribution**: 4-5/5 sequential, 0-1/5 avr, 0/5 debate. Confirms cgpro PIVOT diagnostic — the canonical SWEBENCH_SYSTEM_TEMPLATE's "≥3 tool calls" mandate was incompatible with debate topology bandit picks. Patch-focused profile + adaptive topology converges toward sequential/avr (tool-friendly).

**Prompt metadata** (all 5 tasks):
- `prompt_profile = "patch_focused"`
- `prompt_sha256` differs per task (instance-specific content)
- `topology_override_used = False`
- `system_hint_forced = False`

## What this DOES prove

1. The canary harness can now produce gradable predictions on SWE-bench Pro instances.
2. The patch-focused prompt drops a hard tool-call mandate without compromising:
   - Repo grounding (still says "repo is checked out in current working directory")
   - Strict diff output contract (still requires fenced ```diff with diff --git / --- a/ / +++ b/ / matching context)
3. Adaptive topology selection (NO override, NO system_hint force) reaches tool-friendly topologies under this prompt.
4. The 5-slice canary-stage-timing-budget chain has cleared the prediction-generation pre-requisite for B2 final close.

## What this does NOT prove

- **Patch quality / resolution rate.** Modal grader run pending — will likely return 0-1/5 resolved because budget tier (`deepseek-v4-flash` / `gemini-3-flash-preview`) on out-of-context SWE-bench Pro hard tasks rarely produces resolving diffs. The slice 9 acceptance criterion is "non-empty patch", NOT "resolves". A separate ticket (premium-tier ablation) would be needed for genuine resolution-rate evidence.
- **Topology comparison across prompt profiles.** This run was a single sample under patch_focused. Pre-slice-9 N=5 runs were under canonical with different bandit posterior states. A future paired run (same instances, same posteriors, canonical vs patch_focused) would isolate the prompt-profile effect from topology variance.

## NOT done — flagged as Windows-specific known issue

- All 5 tasks have `repo_dir_cleanup_status="failed"`. Windows `shutil.rmtree(ignore_errors=True)` cannot remove `.git/objects` readonly files without escalation. Disk impact ~250 MB transient per N=5 run. The `atexit` registry catches the rest at process exit; tempdir leakage in interrupted runs is the residual risk.

## Modal grader status

Run launched 2026-05-11 14:00 UTC via `cd external/SWE-bench_Pro-os && python swe_bench_pro_eval.py --raw_sample_path ... --patch_path ... --output_dir ... --num_workers 1 --dockerhub_username jefzda`. ETA ~20-25min. Eval artefacts will land in `docs/benchmarks/2026-05-11-canary-patch-focused-prompt-profile/eval/`.

## Cost ledger

- Canary 5×real-LLM: $0.8061 (5× `deepseek-v4-flash` / `gemini-3-flash-preview` on `graded_patch_generation` 900s profile)
- Modal grader N=5: TBD (last N=1 cost = $0.0135; expect N=5 ≈ $0.05-0.20)
- **Total slice 9 evidence cost**: ~$0.85-1.00

## NEXT (for cgpro post-push)

- cgpro VERIFY this slice 9 closure on resume `cgpro_ygn_sage_global_analysis_20260510`
- B2 final close — strict reading of B2 ACCEPTANCE_GATE met if Modal grader returns valid eval_results
- Post-Modal: capture cost via `sage.bench.modal_billing.capture_modal_app_cost_usd` (slice 5)
- Decision: ship the patch_focused profile as default for B2 canaries, or keep canonical as default + patch_focused as opt-in
