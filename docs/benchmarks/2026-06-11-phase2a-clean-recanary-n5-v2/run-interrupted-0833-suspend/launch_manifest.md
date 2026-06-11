# Cycle-13 Canary Manifest — N=5 preflight before N=50

**Status**: preflight gate. **cgpro audit 2026-05-08**: CONDITIONAL_GO N≤5.
**Refrozen 2026-06-11 (second refreeze — CLEAN re-canary per cgpro
`PHASE2A_CLEAN_RECANARY_UTF8_USAGE_HEALTH_REASONER_REPAIR`)**. Changes vs
the morning run: (1) repair LLM 401 root-caused and fixed (explicit
connector api_key; the morning's `verifier_repair_empty` stages were
unauthenticated calls, NOT model-quality evidence); (2)
`--repair-tier reasoner` (audited identity google/gemini stays inside the
allowlist); (3) `PYTHONUTF8=1` on grader invocations (file writes were
cp1252; PYTHONIOENCODING only covers stdio); (4) **instance set v2**:
`tutao/tutanota-219bc8f0…` is marked `INVALID_INFRA_BASELINE_FAILED` —
its EMPTY-PATCH Modal baseline reproduces the same
`make ***[sqlite3.target.mk] Error 1` native build break and yields only
the opaque `test/unknown | error` bucket, so the image cannot grade ANY
patch. Replacement per the cgpro protocol (deterministic next pinned
instance with a non-represented repo): `qutebrowser/qutebrowser-96b99780…`,
admitted on clone/fetch PASS (checkout == base_commit `2e65f731…`) +
empty-patch baseline PASS (178 real named tests, 161 PASSED / 17 FAILED
f2p). **Dual reporting**: `phase2a_n5_v1` (original set) scores stay
historical/contaminated (2026-05-12: 1/5 patches gate-blocked;
2026-06-10: 3/5 patches 0/5 resolved; 2026-06-11 morning: 2/5 patches
0/5 resolved); `phase2a_n5_v2` starts fresh and is NOT directly
comparable. v2 instances file:
`docs/benchmarks/phase2a-n5-v2-instances/instances.json` (sha256
`35ddf3905ff27b739ff1b1bdb39a98ad0b7fff9db5a825f3c529f1a2542c0f76`).
Hard-cap accounting: generation global $30.00 + repair worst-case
5 × $0.50 = $2.50 + Modal grading ~$0.10. Expected ~$0.75-1.
**Stop-rule (cgpro)**: if this clean N=5 still resolves 0/5 with
`patches_empty_model ≥ 3` and no infra class, STOP the canary loop and
pivot to product diagnosis (patch-generation prompt/agent, retrieval/
context, or a cheap ungraded mini Arm-A-vs-D).

## Frozen at launch

| Parameter | Value |
|-----------|-------|
| Commit SHA | `97fff357115a7f14e02a39dfe30ac97bd32460cd` |
| Dataset | SWE-bench Pro test split; **set v2** (health-screened) — 5 records via `--limit 5` from `docs/benchmarks/phase2a-n5-v2-instances/instances.json` (sha256 `35ddf390…`; v1 file at `docs/benchmarks/2026-05-11-canary-n5-graded/instances/instances.json` retired for resolution metrics after the tutanota-219 baseline failure) |
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
