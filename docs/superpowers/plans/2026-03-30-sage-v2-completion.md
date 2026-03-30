# SAGE V2 Completion Plan — From MASBENCH Validation to Publication

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Nemotron-8B training (DAPO Phase A+B → Phase C micro-decisions), optimize the SAGE pipeline for latency, validate on MASBENCH (5 axes) and GAIA, publish the trained model on HuggingFace with full precision + GGUF, and enable Path 6 in production.

**Architecture:** Three parallel workstreams converging to publication:
1. **GPU workstream** — DAPO training (running) → Phase C GiGPO → post-training merge
2. **CPU workstream** — Pipeline latency optimization → MASBENCH full 50-task rerun → GAIA evaluation
3. **Research workstream** — Function-calling topology (MAS-Orchestra P1) → edge credit upgrade (Graph-GRPO P4)

All workstreams produce artifacts that feed into the final publication: trained model + benchmark results + documentation.

**Tech Stack:** verl 0.7.1 (DAPO), sage-core (Rust), Nemotron-Orchestrator-8B, 2x H100 NVL, 7 API providers, MASBENCH (Salesforce), GAIA (HuggingFace), SageTopologyEnv (Phase C).

---

## Current State (March 30, 2026)

| Component | Status | Key Metric |
|-----------|--------|------------|
| DAPO training | RUNNING step 111/1920 | reward 0.181, best 0.987 |
| MASBENCH depth (fixed runner) | DONE | **SAGE 67% vs bare 40% (+27pp)** |
| Pipeline runner fixes | DONE | DeepSeek fallback, 60s/node timeout |
| Model updates | DONE | gpt-5.4, gemini-3.1, deepseek-chat |
| GAIA dataset | ACCESS GRANTED | Not yet tested |
| Phase C scripts | READY | train_phase_c_custom.py |
| Post-training pipeline | READY | post_training_pipeline.py |
| HF backups | DONE | 49.7 GB (FSDP + LoRA + SFT) |

## File Structure

| File | Responsibility | Status |
|------|---------------|--------|
| `sage-python/src/sage/topology/runner.py` | TopologyRunner (execution engine) | Modified (fallback + timeout) |
| `sage-python/src/sage/bench/masbench.py` | MASBENCH adapter | Exists |
| `sage-python/src/sage/bench/gaia_bench.py` | GAIA adapter | Needs level filtering |
| `sage-python/scripts/verl/train_topology_targeted.sh` | DAPO training | Running |
| `sage-python/scripts/verl/train_phase_c_custom.py` | Phase C GiGPO | Ready |
| `sage-python/scripts/verl/post_training_pipeline.py` | Export → merge → HF → GGUF | Ready |
| `sage-python/scripts/verl/monitor_training.sh` | Autonomous monitor | Running |
| `sage-python/src/sage/verl/topology_env.py` | 4-state machine (Phase C) | Ready |
| `sage-python/src/sage/verl/edge_credit.py` | Graph-GRPO credit | Needs continuous reward upgrade |
| `sage-python/src/sage/verl/reward.py` | Reward function | Phase A simplified + exec |
| `CLAUDE.md` | Project status | Updated March 30 |
| `RUNPOD_PLAN.md` | Training plan | Updated March 30 |
| `TRAINING_LOG.md` | Full training history | Updated March 30 |

---

## WORKSTREAM 1: GPU — Training Pipeline (sequential, on GPU)

### Task 1: Monitor DAPO training to convergence gate

**Files:** Monitor logs only (no code changes)

The DAPO training is running (step 111/1920, ~30h remaining). Monitor until convergence gate.

- [ ] **Step 1: Check training every 2h via background monitor**

The monitor (`monitor_training.sh`) handles:
- Metrics push to GitHub every 10 min
- Checkpoint rotation (keep 2) on NVMe
- LoRA + FSDP upload to HF at each save_freq=100
- Auto-detect crashes and log errors

```bash
tail -f /workspace/monitor.log
```

- [ ] **Step 2: Evaluate convergence gate at step 500**

```bash
ray_dir=$(ls -dt /tmp/ray/session_* | head -1)
big=$(ls -S "$ray_dir"/logs/worker-*.out | head -1)
# Check: reward > 0.3? exec_hits > 15%? grad_norm stable?
grep "critic/score/mean:" "$big" | tail -20
```

Convergence gate:
- reward > 0.3 → proceed to Phase C
- reward < 0.25 after 500 steps → increase lr to 5e-6 or switch to full dataset
- reward flat for 200+ steps → stop, use best checkpoint

- [ ] **Step 3: Save final Phase A+B checkpoint**

Verify FSDP complete on HF:
```bash
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='<token>')
files = [f for f in api.list_repo_tree('yannabadie/sage-topology-policy-v2', recursive=True)
         if hasattr(f,'rfilename') and 'checkpoint' in f.rfilename and '.pt' in f.rfilename]
print(f'{len(files)} FSDP files on HF')
"
```

- [ ] **Step 4: Commit convergence results**

```bash
# Update TRAINING_LOG.md with final Phase A+B metrics
git add TRAINING_LOG.md
git commit -m "results: DAPO Phase A+B converged at step N, reward X"
git push origin main
```

---

### Task 2: Phase C — GiGPO multi-step micro-decisions

**Files:**
- Run: `sage-python/scripts/verl/train_phase_c_custom.py`
- Use: `sage-python/src/sage/verl/topology_env.py`
- Use: `sage-python/src/sage/verl/reward.py` (SAGE_TRAINING_PHASE=C)
- Modify: `sage-python/src/sage/verl/edge_credit.py` (upgrade to continuous rewards)

**Depends on:** Task 1 convergence gate passed.

Phase C is SAGE's core differentiator. The model operates the topology at runtime:
```
awaiting_yaml → executing → awaiting_decision (upgrade/continue/reroute) → terminal
```

- [ ] **Step 1: Upgrade edge_credit.py to continuous rewards**

Current: binary success (`reward > 0.5 ? 1 : 0`)
Target: continuous reward (Graph-GRPO arXiv 2603.02701)

```python
# In edge_credit.py, replace:
success = 1.0 if reward > 0.5 else 0.0
# With:
success = max(0.0, min(1.0, reward))  # continuous [0,1]
```

- [ ] **Step 2: Set SAGE_TRAINING_PHASE=C and launch Phase C**

```bash
export SAGE_TRAINING_PHASE=C
cd sage-python
python3 scripts/verl/train_phase_c_custom.py \
    --model /workspace/sft_merged_model \
    --checkpoint /home/yann/verl_checkpoints \
    --data data/verl_topology_train.parquet \
    --output /home/yann/verl_checkpoints_phase_c \
    --epochs 3 --lr 5e-7 --k 4 --batch-size 4 \
    2>&1 | tee /workspace/train_phase_c.log
```

- [ ] **Step 3: Monitor Phase C success signals**

Must see ALL of these in logs:
- `step_advantage` values != 0 (GiGPO multi-step working)
- Anchor keys like `decision:coder:moderate:low` (model taking decisions)
- Mix of `upgrade` and `continue` decisions (not all same)
- reward_mean improving from initial negative values

- [ ] **Step 4: Backup Phase C checkpoint to HF**

```bash
python3 scripts/verl/upload_checkpoint.py --step <latest>
```

- [ ] **Step 5: Commit**

```bash
git add TRAINING_LOG.md src/sage/verl/edge_credit.py
git commit -m "results: Phase C GiGPO — micro-decisions with step-level advantages"
git push origin main
```

---

### Task 3: Post-training — merge + HF push + GGUF

**Files:**
- Run: `sage-python/scripts/verl/post_training_pipeline.py all`

**Depends on:** Task 2 Phase C completed (or Task 1 if Phase C skipped).

**CRITICAL:** Merge LoRA ONLY at this stage, never during training.

- [ ] **Step 1: Run full post-training pipeline**

```bash
cd sage-python
python3 scripts/verl/post_training_pipeline.py all
```

Pipeline steps:
1. Export LoRA from latest checkpoint
2. Merge LoRA into Nemotron-8B base (float16, ~16GB)
3. Push merged model to HF root
4. Quantize to GGUF Q8_0 (~8.5GB)
5. Push GGUF to HF `/gguf/`

- [ ] **Step 2: Verify both versions on HF**

```bash
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='<token>')
files = list(api.list_repo_tree('yannabadie/sage-topology-policy-v2', recursive=True))
safetensors = [f for f in files if hasattr(f,'rfilename') and f.rfilename.endswith('.safetensors') and '/' not in f.rfilename]
gguf = [f for f in files if hasattr(f,'rfilename') and f.rfilename.endswith('.gguf')]
print(f'Full precision: {len(safetensors)} shards')
print(f'GGUF: {len(gguf)} files')
"
```

Expected: 4 safetensors shards + 1 GGUF Q8_0.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: post-training complete — merged model + GGUF on HuggingFace"
git push origin main
```

---

## WORKSTREAM 2: CPU — Pipeline Optimization + Benchmarks (parallel to GPU)

### Task 4: MASBENCH full 50-task rerun with fixed pipeline

**Files:**
- Run: `sage-python/src/sage/bench/masbench.py`
- Output: `docs/benchmarks/2026-03-30-masbench-full-fixed.json`

**Depends on:** Runner fixes already committed (DeepSeek fallback + 60s timeout).

- [ ] **Step 1: Run 50-task MASBENCH (10 per axis, bare vs SAGE)**

```bash
cd sage-python
python -m sage.bench --type masbench --limit 10
# Or run the ablation script directly:
python3 -c "
from sage.bench.masbench import MASBenchAblation
from sage.boot import boot_agent_system
import asyncio
system = boot_agent_system(use_mock_llm=False, llm_tier='budget')
ablation = MASBenchAblation(system=system, axis='depth')
asyncio.run(ablation.run(limit=10))
"
```

Run for ALL 5 axes: depth, breadth, horizon, parallel, robustness.

- [ ] **Step 2: Save results JSON**

```bash
# Save structured results to docs/benchmarks/
python3 -c "import json; ..."  # Parse output into JSON
```

- [ ] **Step 3: Statistical analysis**

For axes with N >= 10: McNemar's test + Cohen's d.
Document which axes show significant topology advantage.

- [ ] **Step 4: Commit**

```bash
git add docs/benchmarks/2026-03-30-masbench-full-fixed.json
git commit -m "results: MASBENCH 50-task full evaluation with fixed pipeline"
git push origin main
```

---

### Task 5: GAIA Level 1 evaluation

**Files:**
- Modify: `sage-python/src/sage/bench/gaia_bench.py`
- Output: `docs/benchmarks/2026-03-30-gaia-level1.json`

**Depends on:** GAIA access accepted (done).

- [ ] **Step 1: Enhance gaia_bench.py with level filtering**

Add `level` parameter to filter by GAIA Level (1, 2, 3).
Add exact-match scoring (GAIA standard — no partial credit).
Add JSONL output for leaderboard submission.

- [ ] **Step 2: Run GAIA Level 1**

```bash
cd sage-python
python -m sage.bench --type gaia --level 1 --tier budget
```

Expected: ~50 Level 1 tasks, ~2h runtime, ~$5 API cost.

- [ ] **Step 3: Run bare model baseline on same tasks**

```bash
python -m sage.bench --type gaia --level 1 --tier budget --baseline
```

- [ ] **Step 4: Compare and document**

```bash
# Delta analysis: SAGE vs bare on GAIA Level 1
git add docs/benchmarks/2026-03-30-gaia-level1.json src/sage/bench/gaia_bench.py
git commit -m "results: GAIA Level 1 — SAGE vs bare model"
git push origin main
```

---

### Task 6: Pipeline latency optimization (reduce 191s → <90s)

**Files:**
- Modify: `sage-python/src/sage/topology/runner.py`
- Modify: `sage-python/src/sage/strategy/knn_router.py`
- Modify: `sage-python/src/sage/boot.py`

The runner already parallelizes multi-ready nodes. The bottleneck is:
1. Routing embedding computed every call (not cached)
2. Large context passed between nodes (no compression)
3. Full system prompt repeated per node

- [ ] **Step 1: Cache routing embeddings**

In `knn_router.py`, add LRU cache for task embeddings:
```python
from functools import lru_cache

@lru_cache(maxsize=256)
def _cached_embed(self, task_hash: str, task: str):
    return self.embedder.embed(task)
```

- [ ] **Step 2: Truncate predecessor context to 1000 chars**

In `runner.py`, `_gather_predecessor_context()`:
```python
# Truncate each predecessor output to 1000 chars
context = "\n".join(
    f"[{role}]: {output[:1000]}"
    for idx, output in self._node_outputs.items()
    if idx in predecessor_indices
)
```

- [ ] **Step 3: Benchmark latency improvement**

```bash
# Run same 3 depth tasks, compare timing
python3 -c "..." # Same test as before, report avg latency
```

Target: < 90s average (from 191s).

- [ ] **Step 4: Commit**

```bash
git add src/sage/topology/runner.py src/sage/strategy/knn_router.py
git commit -m "perf: cache routing embeddings + truncate context (191s → Xs)"
git push origin main
```

---

## WORKSTREAM 3: Research — Future Improvements (background)

### Task 7: Function-calling topology format (MAS-Orchestra P1)

**Files:**
- Create: `sage-python/src/sage/verl/topology_fc.py`
- Modify: `sage-python/src/sage/verl/reward.py`

**Priority:** P1 — highest impact research improvement.
**Status:** Research only in this plan. Implementation in next sprint.

Instead of generating free-form YAML, the model emits function calls:
```python
# Instead of YAML generation:
"nodes:\n  - role: coder\n    model_tier: budget\n..."

# Function-calling format:
[
    {"name": "add_node", "args": {"role": "coder", "model_tier": "budget", "prompt": "..."}},
    {"name": "add_node", "args": {"role": "reviewer", "model_tier": "fast", "prompt": "..."}},
    {"name": "add_edge", "args": {"from": 0, "to": 1}},
    {"name": "set_reasoning", "args": {"text": "..."}}
]
```

Benefits:
- Leverages Nemotron-8B's existing FC training (GRPO-trained for tool selection)
- Malformed FC is rarer than malformed YAML (~5% vs ~60% error rate)
- Action space is discrete (finite set of functions) vs infinite (free text)

- [ ] **Step 1: Research — read MAS-Orchestra paper implementation details**

```bash
# Use ExoCortex or web search
python3 -c "..."
```

- [ ] **Step 2: Prototype topology_fc.py — convert YAML schema to FC schema**

- [ ] **Step 3: Test with 5 prompts — compare FC generation quality vs YAML**

- [ ] **Step 4: Document findings**

```bash
git add src/sage/verl/topology_fc.py docs/
git commit -m "research: function-calling topology prototype (MAS-Orchestra P1)"
git push origin main
```

---

## CONVERGENCE: Final Validation + Publication

### Task 8: Final benchmarks with Path 6 enabled

**Files:**
- Output: `docs/benchmarks/2026-03-XX-final-results.json`
- Modify: `CLAUDE.md`, `TRAINING_LOG.md`

**Depends on:** Task 3 (post-training complete).

- [ ] **Step 1: Enable Path 6 and run MASBENCH**

```bash
export SAGE_ENABLE_PATH6=1
python -m sage.bench --type masbench --limit 10
```

Compare: Path 6 vs templates vs bare model.

- [ ] **Step 2: Enable Path 6 and run GAIA Level 1**

```bash
export SAGE_ENABLE_PATH6=1
python -m sage.bench --type gaia --level 1 --limit 20
```

- [ ] **Step 3: Run BigCodeBench Hard (the CLAUDE.md reference benchmark)**

```bash
export SAGE_ENABLE_PATH6=1
python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 20
```

Target: > 40% (beat The Conductor's reported 40%).

- [ ] **Step 4: Update all documentation with final results**

Update: CLAUDE.md, TRAINING_LOG.md, RUNPOD_PLAN.md, HF model card.

- [ ] **Step 5: Final commit and push**

```bash
git add CLAUDE.md TRAINING_LOG.md RUNPOD_PLAN.md docs/benchmarks/
git commit -m "results: SAGE V2 complete — MASBENCH/GAIA/BigCodeBench with Path 6"
git push origin main
```

---

## Success Criteria

| Criterion | Target | Verification |
|-----------|--------|-------------|
| DAPO reward convergence | > 0.3 by step 500 | Ray metrics |
| DAPO exec hit rate | > 20% | Monitor logs |
| Phase C step_advantage | Non-zero | Training logs |
| Phase C decision variety | upgrade + continue + reroute observed | Training logs |
| MASBENCH depth (full, fixed) | > +20pp vs bare | 50-task evaluation |
| GAIA Level 1 | > 30% (competitive) | HF leaderboard |
| BigCodeBench Hard + Path 6 | > 40% | Benchmark run |
| Pipeline latency | < 90s avg (from 191s) | Timing test |
| Full precision model on HF | 4 safetensors shards | HF repo |
| GGUF Q8_0 on HF | 1 file ~8.5GB | HF repo |
| FSDP backup on HF | Every 100 steps | HF repo |
| All results documented | CLAUDE.md + TRAINING_LOG + HF card | Git history |

## Dependency Graph

```
WORKSTREAM 1 (GPU):      Task 1 (DAPO monitor) ──→ Task 2 (Phase C) ──→ Task 3 (post-training) ──→ Task 8 (final bench)
                              │                                                                          ↑
WORKSTREAM 2 (CPU):      Task 4 (MASBENCH full) ─────────────────────────────────────────────────────────┤
                         Task 5 (GAIA Level 1)  ──────────────────────────────────────────────────────────┤
                         Task 6 (latency opt)   ──────────────────────────────────────────────────────────┘
                              │
WORKSTREAM 3 (Research): Task 7 (FC topology) ── feeds into next training cycle
```

Tasks 4, 5, 6 run on CPU and are **independent of GPU training**. They can start immediately.
Task 7 is research — outputs feed into the next training iteration, not this one.
Task 8 requires Task 3 (post-training) AND Tasks 4/5 (benchmarks for comparison).

## Cost Estimate

| Task | GPU hours | API cost | Total |
|------|-----------|----------|-------|
| Task 1 (DAPO training) | ~30h H100 | $0 | ~$90 |
| Task 2 (Phase C) | ~5h H100 | ~$30 | ~$45 |
| Task 3 (Post-training) | ~1h | $0 | ~$3 |
| Task 4 (MASBENCH 50) | 0 | ~$5 | $5 |
| Task 5 (GAIA Level 1) | 0 | ~$10 | $10 |
| Task 6 (Latency opt) | 0 | ~$2 | $2 |
| Task 7 (FC research) | 0 | ~$5 | $5 |
| Task 8 (Final bench) | 0 | ~$15 | $15 |
| **Total** | **~36h** | **~$67** | **~$175** |

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|------------|-----------|
| DAPO training plateau | Medium | Increase lr or K, switch to dynamic sampling |
| Phase C doesn't converge | Medium | Use Phase A+B checkpoint as Path 6 (still +27pp value) |
| GAIA score too low | Medium | Focus on MASBENCH (already validated) |
| Pipeline still too slow | Low | Already 191s → target 90s, runner parallelization exists |
| Pod crashes during training | Medium | FSDP on HF, can resume from any checkpoint |
| vLLM TP=2 crash | Medium | Using TP=1, proven stable for 1000+ steps |
| BigCodeBench < 40% | Medium | Value is in MASBENCH depth (+27pp), not single-turn code |
