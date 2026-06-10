# B2_RERUN_UNBLOCKERS — postmortem & fixes (2026-06-10)

Closes the 3 production bugs exposed by the 2026-05-12 N=5 graded canary
(`canary_decision: NO_GO_N50`, cgpro `NEXT_BLOCK_ID=
B2_RERUN_UNBLOCKERS_PROVIDER_COST_DIFF_VERIFIER`). Plan:
`docs/superpowers/plans/2026-06-10-b2-rerun-unblockers.md`. All fixes are
TDD'd against the committed task artifacts of the N=5 run (ground truth:
`run/per_task/*.events.jsonl`).

| # | Bug (N=5 symptom) | Root cause (file:line at fix time) | Fix | Commit | Tests |
|---|---|---|---|---|---|
| 1 | `provider_final=unknown` on the reasoner tier (task #3, `gemini-3.1-pro-preview`) → provider_gate NO_GO via `execution_outside_allowlist=["unknown"]`, while cards.toml declares `provider="google"` | Two layers: ALL 5 entries of `sage-python/config/model_profiles.toml` lacked a `provider` key, so `registry._profile_from_toml` (providers/registry.py:318) stamped `provider="unknown"` for curated-but-not-live-discovered models; `ProviderPool.infer_provider` (llm/provider_pool.py:359) trusted any truthy registry value, so the sentinel beat the correct `gemini-*` → `google` prefix fallback. Live-discovered models (deepseek-v4-flash on the same run) took the discovery merge path (`dm.provider`) and dodged the bug — which is why only the reasoner model was hit. The `"provider_id":"unknown"` literal appears exactly twice in the task #3 events, both on `node_started` of node 1 (coder). | `infer_provider` treats `"unknown"` as unresolved and falls through; explicit `provider` added to all 5 curated entries (values per cards.toml, directive #7); tripwire test pins every future entry to declare a provider that agrees with cards.toml. Gate semantics untouched: `unknown` execution still NO_GO (contract locks added). | `355b7ae9` | `test_provider_pool.py` 22/22 (4 new) + 2 gate contract locks |
| 2 | Real spend ~$0.79 vs reported $0.30 (2.6×): hard failures reported `total_cost_usd=0` while `_observed_event_cost_usd` carried $0.16/$0.18; successes under-reported (tutanota db90ac26: $0.134 vs $0.266) | The `eeb3a7fb` recovery existed ONLY in `_timeout_task_result`; the nominal path trusted `cli_complete.total_cost_usd` alone (run_dryrun_arm_d.py, summary construction) | Shared `_resolve_total_cost` helper used by all paths: larger source wins (never summed — cgpro stop-condition), `_total_cost_usd_source` explicit everywhere (`cli_complete` / `event_audit_observed_event_cost_usd` / `no_cost_evidence`), `cost_integrity_warning` on `cli_complete_cost_missing` / `cli_complete_cost_underreport` / existing `llm_execution_observed_zero_cost`. Timeout parity: `cli_complete_expected=False` keeps eeb3a7fb behavior byte-compatible (its 3 tests pass unchanged). | `7c726e53` | 4 new resolver tests; 61/61 file |
| 3 | `_diff_verifier_outcome=None` on every task despite `SAGE_DIFF_VERIFIER_MODE=observe` in the launch env | Architectural: the env flag WAS propagated to the subprocess, but the verifier lives in `SWEBenchBench._run_one_instance`, which the canary never instantiates — no consumer on the canary path. Worse, patch extraction ran AFTER the `finally` that deletes the cloned worktree, so a wired verifier would have had no file bytes to compare. | `_CANARY_DIFF_VERIFIER_MODE` single-source constant (env builder + annotation); `_annotate_diff_verifier` launcher-side annotation with EXPLICIT skip outcomes (`skipped_no_patch` / `skipped_no_repo_dir` / `skipped_mode_off` / `skipped_timeout` / `unsupported_no_opinion`) — never None, because manifest stop condition #5 keys off non-null fields; extraction+verification moved inside the try-block before worktree cleanup; mock branch carries the same explicit fields. Predictions audit shape stays `swebench_pro_canary_prediction_v1` (no new fields there; `_diff_verifier_reasons` is summary-only). | `5bd351d2` | 6 new wiring tests; 67/67 file |

## Dry-run proof (free, no API spend)

`docs/benchmarks/2026-06-10-b2-unblockers-dryrun/` — `--mock --limit 1`:

- `_diff_verifier_outcome: "skipped_no_repo_dir"` (explicit, non-null) ✓
- `_total_cost_usd_source: "no_cost_evidence"` (explicit) ✓
- `provider_final` is not observable in mock mode (no subprocess, gate
  BLOCKED by design) — the provider fix is proven at the unit level
  instead: `infer_provider("gemini-3.1-pro-preview") == "google"` against a
  registry double returning the `unknown` sentinel, plus the data tripwire.

## Contract coverage (cgpro 9 required tests)

1. gemini-3.1-pro-preview→google mapping ✓ (`test_registry_unknown_sentinel_falls_through_to_prefix`)
2. provider_gate stays NO_GO on unknown ✓ (`test_provider_gate_still_no_go_on_unknown_provider`)
3. provider_gate PASS on google+deepseek-only ✓ (`test_provider_gate_pass_on_google_deepseek_only_execution`)
4. non-timeout failure cost recovered from event audit ✓ (`test_resolve_total_cost_recovers_hard_failure_from_event_audit`)
5. total_cost_usd source explicit ✓ (`test_resolve_total_cost_uses_cli_complete_when_consistent` + source field on every path)
6. SAGE_DIFF_VERIFIER_MODE propagates to subprocess ✓ (`test_diff_verifier_env_propagates_to_subprocess`)
7. patch + observe → non-null outcome ✓ (`test_patch_with_observe_mode_yields_non_null_verifier_outcome`)
8. no patch → explicit skipped_no_patch ✓ (`test_no_patch_yields_explicit_skipped_no_patch`)
9. 55/55 pre-existing run_dryrun tests stay green ✓ (file now 67/67)

## Scope notes for VERIFY

- `sage/llm/provider_pool.py` matches the allowed glob (`provider*.py`);
  `providers/registry.py:318` (the sentinel factory) was deliberately NOT
  edited — the pool-side guard closes the class without touching the
  out-of-glob file. `topology/runner.py` was NOT edited either (the
  emission chain is correct once attribution resolves).
- Value-level change inside the v1 prediction shape: `_diff_verifier_outcome`
  goes from always-None to explicit strings; field SET unchanged
  (`_PREDICTION_AUDIT_SCHEMA_VERSION` not bumped).
- Next step per the master roadmap: explicit paid GO for Phase 2.a
  (canary N=5, ~$1) — NOT executed in this block.
