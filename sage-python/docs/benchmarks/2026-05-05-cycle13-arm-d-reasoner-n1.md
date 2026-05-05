# Cycle-13 E Tier 2.1 — Arm D reasoner-tier graded result (N=1, REAL Pro grader)

**Date**: 2026-05-05 evening (autonomous segment after Yann "continues en toute autonomie").
**Status**: Tier 2.1 acceptance gate **CLOSED** — `GO_FINISH_N1_THEN_STOP` per cgpro VERIFY 2026-05-05 (`cgpro_pi_mono_pivot_20260505`).
**Companion**: `2026-05-05-cycle13-arm-d-smoke-N1.md` (Tier 2.0 NO-API canary + Tier 2.1 partial).

---

## What this run is

The first **end-to-end real reasoner-tier YGN-SAGE prediction** through the full SWE-bench Pro grader:

```
sage.bench --type swebench --dataset pro --limit 1 --tier reasoner --generate-only
  → predictions.jsonl (Lite shape)
  → predictions.pro.json (via swebench_pro_lite_to_pro_predictions.py)
  → grader_reasoner_n1.csv (via swebench_pro_build_grader_csv.py)
  → swe_bench_pro_eval.py --use_local_docker
  → grader_summary.json
```

All shipped in the cycle-13 prelude commit chain (`6710eb0b..3c88ca3b`).

---

## REAL graded result

```json
{
  "total_tests": 300,
  "status_counts": {"PASSED": 297, "FAILED": 3},
  "fail_to_pass_count": 3,
  "pass_to_pass_count": 288,
  "f2p_resolved": 0,
  "f2p_failed_or_missing": [
    "test/database.js | Test database test/database/keys.js::Key methods should return empty array if keys is empty array or falsy",
    "test/database.js | Test database test/database/keys.js::Key methods should return multiple keys and null if key doesn't exist",
    "test/user/emails.js | email confirmation (library methods) canSendValidation should return true if it has been long enough to re-send confirmation"
  ],
  "p2p_regressed": []
}
```

**Resolution rate**: 0/3 fail_to_pass tests → **task FAILED**.
**Regression rate**: 0/288 pass_to_pass tests → **no harm done**.
**Overall pass@1**: 0% (0/1 task resolved).

---

## What this proves

1. **Full multi-agent harness works end-to-end on a real Pro task.** Stage 0 routing → Stage 2 sequential template → Stage 3 ModelAssigner → Stage 4 agent_loop on each of 3 nodes (planner / coder / synthesizer).

2. **Real prod cost on a real task**: $0.749 USD for 203.7 sec wall-clock at reasoner tier (Opus-class models). 46 tool calls across 28 turns. Cycle-13 main run (N=50) cost-estimate: ~$37 at this rate per task — within plan envelope ($240-460).

3. **Diff verifier integration works**: `metadata.json:log_signals.patch_validation_failed=true` + `patch_repair_timed_out=true` were CAPTURED. cycle-9 diff verifier observed the patch issue mid-run (didn't auto-repair within budget — which is the documented behavior at observe mode).

4. **Pro grader integration works**: 300 tests run inside the Docker container. 297 PASSED, 3 FAILED. The grader produced a clean JSON summary with f2p / p2p / regression breakdown.

5. **No regression**: 288/288 pass_to_pass tests remained green. The agent's diff didn't break working code — meaningful for "harness lift" framing in cycle-13 main run (a smarter harness should not regress while attempting the issue).

---

## Run metrics (per `metadata.json`)

| Field | Value |
|---|---|
| Instance | `instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan` |
| Repo | NodeBB/NodeBB |
| Known-bug excluded | **false** (cgpro flagged `00c70ce7...` — different SHA; this instance is NOT the dataset bad-instance) |
| Model used | `sage/gemini-3.1-pro-preview` (Opus replacement at reasoner tier) |
| Patch chars | 3289 |
| Extraction method | `unified` |
| Repair stage | `failed` (diff_verifier timed out before producing an auto-repaired diff) |
| Latency | 203.7 sec |
| Cost | $0.749 USD |
| Tool calls | 46 across 28 turns |
| System | S3 (reasoner) |
| Execution path | `pipeline` (full multi-agent) |
| Topology templates | `["sequential"]` (3 nodes: planner → coder → synthesizer) |
| Wrote predictions | true |

Stage trail (compressed from 1083 log lines) confirms FrugalGPT cascade retry pattern — Stage 4 quality=0.00 below threshold 0.3 fired the cascade, agent re-routed and tried again. Multiple Stage 0/2/3 sequences within the single bench task — adaptive runtime layer (TopologyController) actively engaged.

---

## What the bench DIDN'T do

- **Did not resolve the actual issue.** 3 fail_to_pass tests in NodeBB still fail. The agent produced a syntactically-applicable diff but not the right semantic fix.
- **Did not break anything.** 288 pass_to_pass tests stay green.

---

## What we learned during the smoke (real bugs caught)

1. **Pipeline event_log shadowing** (`d3fc6fe0`): cycle-12 prelude `sage run --jsonl` was emitting ZERO RuntimeEventLog events because `pipeline.py:763` shadowed the CLI's installed eventlog. Fixed via `current_event_log()` first + 80-LOC regression test. Without this fix, cycle-13 main run would have produced predictions with no telemetry, defeating cgpro DESIGN E secondary metrics (oracle.trainable rate, bandit_attribution_mismatch rate, controller_decision distribution).

2. **CSV builder missing `repo` column** (Yann's fix in `sage-python/scripts/swebench_pro_build_grader_csv.py`): the original CSV builder I wrote didn't include the `repo` column — the grader needs it to resolve `jefzda/sweap-images:<repo-derived-tag>` via `helper_code/image_uri.py`. Without it: grader pulled wrong / missing image. Yann iterated 14 times on the grader invocation to find the issue (`grader.prefix_missing_repo.stdout.log` is the pre-fix evidence). Now CSV writes BOTH lowercase (`fail_to_pass`) and uppercase (`FAIL_TO_PASS`) test sets per the grader's inconsistent column reads.

3. **Patch repair timed out**: cycle-9 diff verifier set to `observe` mode; observed the patch invalid + flagged it but didn't auto-repair within budget. Real production-flow signal.

4. **NodeBB known-bug clarification**: cgpro DESIGN E originally suggested avoiding NodeBB broadly. cgpro VERIFY refined this — only `instance_NodeBB__NodeBB-00c70ce7b0541cfc94afe567921d7668cdc8f4ac-vnan` has the truncated fail_to_pass test names that cause false-negative grading. Other NodeBB instances are fair game.

---

## What is now unblocked for cycle-13 main run

- Real cost-per-task estimate: ~$0.75 at reasoner tier with full 3-node topology.
- Real latency-per-task estimate: ~204 sec.
- Cycle-13 main run N=50 cost projection: ~$37.5 + grader Docker overhead.
- Cycle-13 main run N=50 wall-clock: ~3-4 hours pure agent + grader iteration.
- Confirmed the full sage.bench → Lite/JSONL → Pro/JSON → grader → result pipeline.
- Confirmed the multi-agent harness produces shape-valid + semantically-attempting (not just empty) diffs at reasoner tier.

---

## What is still TODO for cycle-13 main

1. **Wire arm A** (Claude Code direct) — needs subprocess invocation pattern + token usage metering.
2. **Wire arm B** (pi-mono coding-agent direct) — needs `pi` CLI + the env-hygiene exports (`PI_OFFLINE=1` etc.).
3. **Wire arm C** (YGN-SAGE via pi-mono CLI) — needs `clients/pi-ygn-sage/` adapter implementation (cycle-13 main work, NYI).
4. **Approve N=50 budget** (~$240-460 across all 4 arms).
5. **Decide grading strategy**: local Docker (this smoke) vs Modal (faster but needs account setup).

cgpro VERIFY explicit: "Do not start N=10 tonight. N=10 should be a fresh block after the single real reasoner path is documented and any protocol/patch-format drift is fixed."

---

## Acceptance gate per cgpro DESIGN E

- "1/1 graded real Arm D task minimum" — ✅ MET (this run).
- "2/2 only if Docker/image/runtime is not the bottleneck" — DEFERRED to cycle-13 main per cgpro VERIFY.
- Hard cutoff: $5 OR Docker > 15 min — observed: $0.75 + ~3 min Docker. SAFE.
- Patch-format trap (Q5): closed by Tier 2.0 + grader output.
- Telemetry trap (Q5): closed by event_log fix in `d3fc6fe0` + grader output preserved in `metadata.json` sidecar (cgpro Q3 advice).

---

## Artifacts (all under `sage-python/data/swebench_pro/arm_d_reasoner_n1/` — gitignored)

```
predictions.jsonl              -- Lite shape from sage.bench (1 record)
predictions.pro.json           -- Pro shape from lite_to_pro converter (1 record)
predictions_meta.json          -- sage.bench metadata
metadata.json                  -- forensic sidecar (Yann)
grader_summary.json            -- Pro grader pass/fail (Yann)
gen.log                        -- 1083-line bench gen log
debug_eval.py                  -- direct grader invocation (Yann)
grader.<various>.{stdout,stderr}.log   -- 14 grader iteration logs
```

Plus persistent grader-friendly artifacts:
```
sage-python/data/swebench_pro/grader_reasoner_n1.csv   -- raw_sample CSV (Yann's improved builder)
external/SWE-bench_Pro-os/                              -- cloned grader repo (gitignored)
```

---

## References

- Cycle-13 plan: `docs/benchmarks/2026-05-05-cli-baseline-plan.md`.
- Arm wiring contract: `docs/benchmarks/2026-05-05-cycle13-arm-wiring.md`.
- Tier 2.0/2.1 partial: `2026-05-05-cycle13-arm-d-smoke-N1.md` (this doc's predecessor).
- SAGE_CLI_PROTOCOL v0: `docs/contracts/SAGE_CLI_PROTOCOL.md`.
- Runtime integrity ledger (9 invariants): `docs/contracts/runtime-integrity-ledger.md`.
- SWE-bench Pro repo: `github.com/scaleapi/SWE-bench_Pro-os` (cloned at `external/SWE-bench_Pro-os/`).
- HuggingFace dataset: `huggingface.co/datasets/ScaleAI/SWE-bench_Pro`.
- cgpro DESIGN E + VERIFY rounds: conv `cgpro_pi_mono_pivot_20260505` 2026-05-05.

## Status

- 2026-05-05 evening (this commit): Tier 2.1 closed with REAL graded reasoner-tier result. 0/3 f2p resolved on NodeBB-04998908 instance. 0/288 p2p regressed. $0.75 cost / 204 sec / 46 tool calls. Cycle-13 main run unblocked.
- TBD (cycle-13 main): N=50 4-arm with arms A + B + C wired.
