# Cycle-13 Canary Manifest — N=5 preflight before N=50

**Status**: preflight gate. **cgpro audit 2026-05-08**: CONDITIONAL_GO N≤5.
**Refrozen 2026-06-10** for the Phase 2.a rerun (B2_RERUN_UNBLOCKERS closed:
provider attribution, cost recovery, diff-verifier wiring — cgpro VERIFY
EDIT_REQUIRED applied; post-push NEXT_BLOCK_ID=
PHASE2A_PAID_N5_CANARY_AUTH_MANIFEST_FREEZE_AND_RUN; explicit paid GO from
Yann 2026-06-10, expected ~$1, hard caps unchanged below). Same pinned
instance set as 2026-05-12 for comparability.

## Frozen at launch

| Parameter | Value |
|-----------|-------|
| Commit SHA | `b2b40b3021d844d009fc35bc690ccb9a561f05c0` |
| Dataset | SWE-bench Pro test split, stratified N=10; first 5 records taken via `--limit 5` (deterministic file order — `instances.json` at `docs/benchmarks/2026-05-11-canary-n5-graded/instances/instances.json`) |
| Seed | 42 |
| Provider allowlist | `google`, `deepseek` |
| Provider denylist | `openai` |
| OpenAI | **Not in the budget canary allowlist by default.** Direct live provider smoke for `gpt-5.4` and `gpt-5.5-pro` passed on 2026-05-10 (`docs/benchmarks/2026-05-10-provider-preflight-post-model-catalog.json`), but a canary launch must still declare any OpenAI execution explicitly in its launch manifest. |
| Prompt profile | `patch_focused` (cycle-13 K, slice 9: canonical produced 0/5 patches; patch_focused drops the ≥3 tool calls mandate, keeps STRICT unified-diff output) |
| LLM tier | `reasoner` (gemini-3.1-pro-preview, per Option B finding 2026-05-12 — fast/budget tiers terminate with EMPTY_STEP_SENTINEL; reasoner is the only tier reaching the controller and emitting patches on Pro tasks) |

## Budget & timeouts

| Parameter | Value |
|-----------|-------|
| Budget per task | $5.00 |
| **Global budget** | **$30.00** |
| Timeout per task | 900s (`--profile graded_patch_generation`) |
| Hard stop | cumulative cost exceeding $30 total, any task exceeding $5 per-task cap, or >2h wall clock |

## Environment

```bash
SAGE_LLM_TIER=reasoner
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
  --global-budget-usd 30.0 \
  --profile graded_patch_generation \
  --swebench-prompt-profile patch_focused \
  --tier reasoner \
  --provider-allowlist google,deepseek \
  --provider-denylist openai \
  --grader-preflight-path <grader_preflight.json> \
  --ci-green-artifact <ci_green.json> \
  --output-dir <artifact-dir> \
  --claim-default-pipeline-learning-evidence \
  --expect-default-pipeline-learn
```

If this gate fails, keep the generated artifacts and classify the canary as
NO_GO/blocked. Do not infer a successful learning-integrity claim from
`oracle_verdict` presence alone.

## Stop conditions (abort N=50 if any trigger)

1. Cumulative cost > $30
2. >2 tasks with empty patch (patches_extracted=0)
3. >1 task timeout (900s exceeded)
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
