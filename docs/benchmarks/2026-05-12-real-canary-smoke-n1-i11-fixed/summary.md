# Real-canary smoke N=1 (post-fix) — I-11 chain end-to-end evidence

**Date**: 2026-05-12
**HEAD** (at fix-commit time, NOT yet committed): `381445fd` + uncommitted prod-drift fix in `runtime_events.py:507-540`.
**cgpro auth**: explicit "Yes, launch N=1 budget tier" + autonomous-dev-with-cgpro-HITL mode.
**Cost**: $0.0149 + $0.00 (2 runs, 2nd timed out at 300s with no API spend captured).
**Task**: SWE-bench Pro N=1 stratified — `instance_future-architect__vuls-139f3a81b66c47e6d8f70ce6c4afe7a9196a6ea8`
**Budget**: $2.0 per-task / $3.0 global / 300s timeout (`--profile graded_patch_generation`).

## Bug discovered in pre-fix canary

Pre-fix run (`docs/benchmarks/2026-05-12-real-canary-smoke-n1-i11/`) produced:
- ✅ witness event v1_1 emitted (schema correct)
- ✅ 18 rust_filter_rejections recorded (Phase 2 instrumentation alive)
- ❌ **`policy.active=False`** with `decision=allowed, reason=no_policy_active` — wrong!
- ❌ **0 runtime_integrity_assertion events** — I-11 inline binding skipped via the `no_policy_active` exclusion

Root cause: `runtime_emit_provider_execution_witness` read the policy from `ctx.provider_allowlist` / `ctx.provider_denylist` (legacy/test source), but the CLI sets the policy via `configure_pipeline_provider_policy` which writes to `pipeline._provider_allowlist` (underscore-prefixed). The witness saw an empty ctx-level policy and reported `policy.active=False`, even though Rust had clearly applied the policy (18 real rejections).

Production drift class: **single-source-of-truth violation between witness emit (Python) and policy enforcement (Python + Rust)**.

## Fix

`runtime_events.py:507-540` now reads via `effective_provider_policy(pipeline)` (the same helper `enforce_provider_policy` uses) as the primary source. Falls back to ctx attrs only if pipeline-level policy is inactive — preserves backward compat for legacy/test callers.

Regression test: `test_witness_reads_policy_from_pipeline_underscore_attrs` in `tests/test_invariant_i11.py` — fake pipeline with `_provider_allowlist`/`_provider_denylist` set, ctx with NO `provider_*` attrs, witness MUST report `policy.active=True` with the correct allowlist/denylist.

## Post-fix canary trace (this run)

11 event types in 65 events:
- 1 task_started
- 1 routing_decision
- 1 topology_selected
- 3 model_assigned (initial + 2 FrugalGPT upgrade cascade re-assignments)
- **1 provider_execution_witness** (seq=8, phase=initial)
- **1 runtime_integrity_assertion** (seq=9, I-11, verdict=pass)
- 51 cli_progress (idle heartbeat)
- 1 cli_started
- 2 node_started + 1 node_completed
- 1 runner_timeout (300s budget exceeded — separate issue)

### Witness payload (seq=8)

```json
{
  "policy": {
    "active": true,                                      ← FIXED: was false pre-fix
    "source": "cli",
    "allowlist": ["deepseek", "google"],
    "denylist": ["openai"],
    "routing_candidate_decision": "blocked",
    "routing_candidate_reason_code": "provider_in_denylist"
  },
  "substitution_summary": {
    "rust_filter_details_observed": true,                ← Phase 2 alive
    "rust_filter_rejections_truncated": true,            ← cap hit at 20
    "rust_filter_rejections": [/* 20 entries */],
    "routing_candidate_blocked_by_policy": true,
    "routing_model_distinct_from_assignments": true
  }
}
```

### Rust filter rejections (20, truncated from 24)

| Reason code | Count |
|---|---|
| provider_excluded_policy_allowlist | 9 |
| provider_excluded_policy_denylist | 7 |
| card_inactive | 4 |

3 of the 8 reason codes from cgpro DESIGN_LOCKED Q2 surfaced in real data. The other 5 (`provider_excluded_dead`, `excluded_by_caller`, `capability_mismatch`, `cost_above_budget`, `provider_excluded_policy_unknown_provider`) need different triggers (dead-provider list, FrugalGPT cascade, capability mismatch, tight budget, unknown provider). Not in scope for this single-task smoke.

### Runtime integrity assertion (seq=9, I-11)

```json
{
  "invariant_id": "I-11",
  "verdict": "pass",
  "declared_decision": "blocked",
  "verified_decision": "blocked",
  "phase": "initial",
  "declared_reason_code": "provider_in_denylist",
  "verified_reason_code": "denylist",
  "fail_closed": false,
  "witness_seq": 8
}
```

**This is the first production runtime_integrity_assertion event** — the I-11 inline binding fired correctly:
- Read the witness state from `event_log._last_witness_state` (seq=8, decision=blocked)
- Re-evaluated the active policy against the routing candidate's provider (openai)
- Got `verified_decision=blocked` with `verified_reason_code=denylist`
- `declared == verified` → verdict=pass
- `fail_closed=False` (SAGE_TRACE_FAIL_CLOSED not set) → no escalation, just record + continue

## What this evidence proves

| Acceptance gate (cgpro Q3 ledger row) | Status |
|---|---|
| witness v1_1 emitted in production CLI run | ✅ |
| rust_filter_details_observed=true with real data | ✅ (20 entries, 3 reason classes) |
| rust_filter_rejections_truncated=true at cap=20 | ✅ |
| I-11 inline binding fires when policy active | ✅ (verdict=pass) |
| witness.policy.active matches enforce_provider_policy view | ✅ (post-fix) |
| Phase ordering: witness → assertion → … | ✅ (seq 8 < seq 9) |
| ProviderPolicyViolation path preservation | n/a (per-node assignments passed, no violation) |

## Open follow-ups

- **runner_timeout / $0.00 cost** — separate issue. Agent ran (1 node_completed) but the cost tracker recorded $0.0000. Cost-attribution path may not be wired for the budget-tier providers. NOT slice 10D scope; ticket separately.
- **5/8 reason codes not exercised** — need traces with dead-provider exclusions, FrugalGPT cascade, capability mismatch, tight budget, unknown provider. Future canary chain can target these.
- **cgpro NEXT_HARDENING_ID=I11_FAILURE_CORRELATION_METADATA** — non-blocker. Enrich `emit_failure` with `correlation_witness_seq` so close-time audit can pair by witness identity instead of window semantics.

## Reproduce

```bash
PYTHONIOENCODING=utf-8 \
SAGE_BOOT_BYPASS_EPOCH_GUARD=1 \
SAGE_BOOT_BYPASS_REASON="real-canary-smoke-N1 i11 evidence" \
SAGE_OPERATOR_ID="$(whoami)" \
SAGE_TRACE_RAW=1 \
SAGE_LLM_TIER=budget \
python sage-python/scripts/run_dryrun_arm_d.py \
  --instances-json sage-python/data/swebench_pro/n10/instances.json \
  --limit 1 \
  --budget-usd 2.0 --global-budget-usd 3.0 \
  --task-timeout-s 300 \
  --profile graded_patch_generation \
  --swebench-prompt-profile patch_focused \
  --output-dir docs/benchmarks/<date>-real-canary-smoke-n1-i11
```

Then inspect `per_task/*.events.jsonl` for `event_type=provider_execution_witness` (substitution_summary) and `event_type=runtime_integrity_assertion` (I-11 verdict).
