# Cycle-13 Canary Manifest — N=5 preflight before N=50

**Status**: preflight gate. **cgpro audit 2026-05-08**: CONDITIONAL_GO N≤5.

## Frozen at launch

| Parameter | Value |
|-----------|-------|
| Commit SHA | `<SET_AT_LAUNCH>` |
| Dataset | SWE-bench Pro test split, stratified N=5 |
| Seed | 42 |
| Provider allowlist | `google`, `deepseek` |
| OpenAI | **EXCLUDED** (gpt-5.5-pro/gpt-5.4 silent fail, routing bug fixed `c5605439`, pending re-verification) |

## Budget & timeouts

| Parameter | Value |
|-----------|-------|
| Budget per task | $5.00 |
| **Global budget** | **$25.00** |
| Timeout per task | 120s |
| Hard stop | First task exceeding $25 total or >2h wall clock |

## Environment

```bash
SAGE_LLM_TIER=budget
SAGE_DIFF_VERIFIER_MODE=observe
SAGE_OTEL_EXPORTER=none
SAGE_BOOT_BYPASS_EPOCH_GUARD=1
SAGE_BOOT_BYPASS_REASON=cycle-13-canary-n5
SAGE_OPERATOR_ID=ygn-sage-arm-d-canary
SAGE_DANGEROUS_TOOLS=0
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

## Runner invocation

For a real Arm D canary that claims default-pipeline learning-integrity
evidence, the runner must enable the post-run evidence boundary:

```bash
python sage-python/scripts/run_dryrun_arm_d.py \
  --instances-json <instances.json> \
  --limit 5 \
  --budget-usd 5.0 \
  --output-dir <artifact-dir> \
  --claim-default-pipeline-learning-evidence \
  --expect-default-pipeline-learn
```

If this gate fails, keep the generated artifacts and classify the canary as
NO_GO/blocked. Do not infer a successful learning-integrity claim from
`oracle_verdict` presence alone.

## Stop conditions (abort N=50 if any trigger)

1. Cumulative cost > $25
2. >2 tasks with empty patch (patches_extracted=0)
3. >1 task timeout (120s exceeded)
4. Any `cli_complete` with `outcome != "success"`
5. Zero tasks produce `_diff_verifier_outcome` in predictions.jsonl

## Acceptance gates (GO for N=50 if all pass)

- [ ] CI green at frozen commit
- [ ] ≥3/5 tasks produce non-empty patches
- [ ] Every predictions.jsonl entry has: `_verifier_repair_budget_usd`, `_diff_verifier_mismatches`
- [ ] Every events.jsonl has: `cli_started`, `cli_progress`, `task_started`, `routing_decision`, `final_result`, `cli_complete`
- [ ] Every real-task `summary.json.task_summaries[].learning_evidence_boundary.status == "pass"` and top-level `summary.json.learning_evidence_gate.status == "PASS"` when default-pipeline learning-integrity evidence is claimed
- [ ] Cost tracking: every task has `total_cost_usd > 0` in `cli_complete`
- [ ] No `model_id_final` empty or `provider_final` absent

## Post-canary

- [ ] Archive all artifacts (predictions.jsonl, events.jsonl, summary.json)
- [ ] cgpro VERIFY with canary results
- [ ] Decision: GO / NO_GO for N=50
