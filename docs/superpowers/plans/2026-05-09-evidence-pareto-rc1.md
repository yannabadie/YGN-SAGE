# Evidence-Pareto RC1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn YGN-SAGE into a replayable agent-runtime that can prove Pareto improvement over strong simple baselines on real tasks without weakening runtime integrity or scientific evidence standards.

**Architecture:** Keep JSONL event evidence as the canonical forensic layer, then build benchmark comparisons on top of it. Claims and docs remain strict gates, but performance optimization is accepted only when a paired run shows better resolved/pass@1 or equal resolved/pass@1 at lower cost/latency, with no unverified learning.

**Tech Stack:** Python 3.13, Rust/PyO3 `sage_core`, `sage run --jsonl`, TypeScript `clients/pi-ygn-sage`, SWE-bench Pro adapters, GitHub Actions, cgpro review.

---

### Task 1: Real-Backend CLI JSONL Smoke

**Files:**
- Modify: `sage-python/tests/test_sage_cli_jsonl.py`
- Modify only if the red test exposes a real gap: `sage-python/src/sage/cli/run.py`
- Verify adapter compatibility: `clients/pi-ygn-sage/test/contract.test.ts`

- [x] **Step 1: Write the failing subprocess test**

Add a test that launches `python -m sage.cli run --jsonl` as a subprocess with stdin command mode. It must send `prompt`, wait for `cli_started`, send a tightening `set_budget`, send `cancel` while execution is active, and parse stdout as LF-only JSONL.

- [x] **Step 2: Verify RED**

Run:

```powershell
cd sage-python
python -m pytest tests/test_sage_cli_jsonl.py::test_subprocess_jsonl_cancel_stream_is_adapter_compatible -q
```

Observed before the fix: the subprocess emitted the terminal `cli_complete` frame but did not exit while stdin stayed open.

- [x] **Step 3: Implement the minimal fix**

If the subprocess stream violates the contract, change only `sage-python/src/sage/cli/run.py` or the smallest directly-owned helper needed to restore: contiguous seq, first `cli_started`, exactly one `failure(kind="cli_cancel")`, final `cli_complete(outcome="cancelled", exit_code=130)`, `final_seq` pointing to the prior frame, and no frames after terminal.

- [x] **Step 4: Verify GREEN**

Run:

```powershell
cd sage-python
python -m pytest tests/test_sage_cli_jsonl.py::test_subprocess_jsonl_cancel_stream_is_adapter_compatible -q
cd ..\clients\pi-ygn-sage
npm test
```

Expected: new subprocess smoke passes and adapter fixture/contract tests remain green.

Evidence archived under `docs/superpowers/evidence/2026-05-09-rc1-cli-subprocess-cancel/`.

### Task 2: Paired Canary Run Pack

**Files:**
- Modify: `sage-python/scripts/run_swebench_ablation.py`
- Possibly create: `sage-python/scripts/run_swebench_pareto_canary.py`
- Create artifacts under: `docs/benchmarks/YYYY-MM-DD-pareto-canary-n5/`

- [ ] **Step 1: Add a two-arm manifest**

Freeze task IDs, commit SHA, provider/model policy, budget, timeout, diff verifier mode, tool policy, and official evaluator command. Do not replace `<SET_AT_LAUNCH>` in the existing cycle-13 manifest unless the actual canary launches at that commit.

- [ ] **Step 2: Run N=5 instrumentation canary**

Run direct baseline and YGN-SAGE direct on the same SWE-bench Pro tasks. Archive predictions, events, summaries, logs, and the manifest.

- [ ] **Step 3: Decide GO/NO_GO for four-arm ablation**

Proceed only if every task has a complete trace, every prediction has verifier metadata, and the summary separates empty patch, invalid patch, official failure, timeout, provider failure, and cancel.

### Task 3: Credit Event Schema v0

**Files:**
- Modify: `sage-python/src/sage/runtime/events/payload_schemas.py`
- Modify: `sage-python/src/sage/pipeline_v2/execute.py`
- Modify: `sage-python/src/sage/pipeline_v2/learn.py`
- Test: `sage-python/tests/test_runtime_event_contracts.py`

- [ ] **Step 1: Add a schema-level test**

The golden payload must include task id, arm id, topology id, node id, optional edge id, parent event ids, model id, provider, token counts, cost, duration, verifier/oracle deltas, and `trainable`.

- [ ] **Step 2: Emit credit records**

Emit credit records after node/tool execution boundaries without changing learning behavior.

- [ ] **Step 3: Keep learning fail-closed**

Add a test proving a credit record with missing or unverified evidence cannot update bandit, MAP-Elites, training memory, or online evolution.

### Task 4: Topology Claim Split

**Files:**
- Modify: `docs/claims/topology.yaml`
- Modify: `docs/contracts/runtime-integrity-ledger.md` only if a side-effect label changes
- Modify: `README.md`
- Modify: `AI-ARCHITECTURE.md`
- Run: `python scripts/regenerate_claims_index.py`

- [ ] **Step 1: Split claim semantics**

Separate `sources_exposed`, `source_reachability`, and `source_utility` so a reachability smoke cannot masquerade as performance evidence.

- [ ] **Step 2: Preserve evidence gates**

Keep deterministic S-MMU/MCTS reachability as `evidence_pending` unless a deterministic seam proves it.

### Task 5: Benchmark Infrastructure Decision

**Files:**
- Modify: `docs/benchmarks/cycle-13-canary-manifest.md` only if launching that manifest
- Possibly create: `docs/benchmarks/2026-05-09-docker-grading-blocker.md`

- [ ] **Step 1: Classify Docker blocker**

If Docker Desktop Linux daemon remains unavailable, archive a `NO_GO_LOCAL_DOCKER` record with exact command/error and select cloud/Linux runner options.

- [ ] **Step 2: Select grading path**

Use local Docker only if the daemon is healthy. Otherwise select Modal, GitHub larger runner, or another Linux runner and document the cost and reproducibility tradeoff before launching N=5.
