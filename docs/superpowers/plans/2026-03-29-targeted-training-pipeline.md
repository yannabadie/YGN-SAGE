# Targeted Nemotron-8B Training Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Nemotron-Orchestrator-8B through 3 progressive phases to generate adaptive topologies that improve SAGE on depth/horizon/robustness tasks, culminating in **Phase C micro-decisions** (upgrade/continue/reroute at runtime) — SAGE's core differentiator vs all competitors.

**Architecture:** Four-stage pipeline aligned with RUNPOD_PLAN:
1. **Validate** — MASBENCH/GAIA evaluation to confirm which axes benefit from topology
2. **Phase A+B targeted** — Structural + execution reward on filtered depth/horizon/robustness dataset
3. **Phase C (THE differentiator)** — GiGPO multi-step with `SageTopologyEnv` 4-state machine. Model learns WHEN to upgrade model_tier, reroute, or continue at checkpoints. Uses 5-signal reward (structural 20% + execution 35% + rewardflow 20% + resilience 15% + cost 10%).
4. **Post-training** — Merge + HF push (full precision + GGUF) + final MASBENCH/GAIA benchmark

**Phase C is what no competitor has:** The Conductor generates static topologies. AgentConductor evolves offline. SAGE's model takes **runtime decisions** at checkpoints based on node output quality. This is the publishable innovation.

**Tech Stack:** verl 0.7.1, GRPO (Phase A/B) + GiGPO (Phase C), Nemotron-Orchestrator-8B, sage_core (Rust: TopologyExecutor, HybridVerifier, TopologyDensity), SageTopologyEnv (4-state machine), 2x H100 NVL, 7 providers, MASBENCH, GAIA.

---

## File Structure

| File | Responsibility | Status |
|------|---------------|--------|
| `sage-python/src/sage/bench/masbench.py` | MASBENCH adapter | Exists |
| `sage-python/src/sage/bench/gaia_bench.py` | GAIA adapter | Exists, needs enhancement |
| `sage-python/scripts/verl/train_topology_targeted.sh` | NEW: Targeted training script | Create |
| `sage-python/scripts/verl/filter_training_data.py` | NEW: Filter dataset by task type | Create |
| `sage-python/src/sage/verl/reward.py` | Reward function (Phase A simplified) | Exists, may need targeted adjustments |
| `sage-python/scripts/verl/post_training_pipeline.py` | Export + merge + push HF + GGUF | Exists |
| `sage-python/scripts/verl/monitor_training.sh` | Autonomous monitor | Exists |
| `sage-python/scripts/verl/upload_checkpoint.py` | HF backup (LoRA + FSDP) | Exists |
| `docs/benchmarks/2026-03-29-masbench-full.json` | Full MASBENCH results | Generated |
| `docs/benchmarks/2026-03-30-gaia-results.json` | GAIA results | Generated |
| `TRAINING_LOG.md` | Training documentation | Update |
| `CLAUDE.md` | Project status | Update |

---

### Task 1: Collect MASBENCH 50-task results and document

**Files:**
- Output: `docs/benchmarks/2026-03-29-masbench-full.json`
- Modify: `TRAINING_LOG.md`

The 50-task MASBENCH test is currently running (PID 856577). Wait for completion, save results.

- [ ] **Step 1: Wait for MASBENCH test to complete**

```bash
# Check every 5 min until done
while ps -p 856577 > /dev/null 2>&1; do sleep 300; done
cat /tmp/claude-1000/-workspace-YGN-SAGE/*/tasks/bpk8l49ue.output | grep -v "^WARNING\|UserWarning"
```

- [ ] **Step 2: Save results to JSON**

```bash
# Parse output into structured JSON
python3 -c "
# Extract results from test output and save as JSON
import json
results = {
    'benchmark': 'MASBENCH',
    'date': '2026-03-29',
    'tasks_per_axis': 10,
    'total_tasks': 50,
    # Fill from actual output
}
with open('docs/benchmarks/2026-03-29-masbench-full.json', 'w') as f:
    json.dump(results, f, indent=2)
"
```

- [ ] **Step 3: Update TRAINING_LOG.md with full results table**

- [ ] **Step 4: Commit and push**

```bash
git add docs/benchmarks/2026-03-29-masbench-full.json TRAINING_LOG.md
git commit -m "results: MASBENCH full 50-task evaluation (5 axes, bare vs SAGE)"
git push origin main
```

---

### Task 2: Run GAIA Level 1 evaluation

**Files:**
- Modify: `sage-python/src/sage/bench/gaia_bench.py`
- Output: `docs/benchmarks/2026-03-30-gaia-level1.json`

**Prerequisites:** User must accept GAIA dataset terms on HuggingFace.

- [ ] **Step 1: Verify GAIA dataset access**

```bash
python3 -c "
from datasets import load_dataset
import os
os.environ['HF_TOKEN'] = '<token>'
ds = load_dataset('gaia-benchmark/GAIA', '2023_all', split='validation', token=os.environ['HF_TOKEN'])
print(f'GAIA: {len(ds)} tasks')
"
```

Expected: ~165 tasks loaded.

- [ ] **Step 2: Enhance gaia_bench.py with level filtering and exact-match scoring**

Add `level` parameter, GAIA-standard exact match, JSONL output for leaderboard submission.

- [ ] **Step 3: Run GAIA Level 1 (bare model vs SAGE)**

```bash
cd sage-python
python -m sage.bench --type gaia --level 1 --tier budget
```

- [ ] **Step 4: Save results and commit**

```bash
git add docs/benchmarks/2026-03-30-gaia-level1.json src/sage/bench/gaia_bench.py
git commit -m "results: GAIA Level 1 evaluation — bare vs SAGE"
git push origin main
```

---

### Task 3: Filter training dataset for depth/horizon/robustness tasks

**Files:**
- Create: `sage-python/scripts/verl/filter_training_data.py`
- Output: `sage-python/data/verl_topology_targeted.parquet`

The 12K training dataset contains tasks of varying difficulty. We need to filter for tasks where topology matters: complex multi-step tasks (difficulty=complex, node_count>=3).

- [ ] **Step 1: Analyze training data distribution**

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/verl_topology_train.parquet')
# Count by difficulty
for d in ['simple', 'medium', 'complex']:
    subset = df[df['extra_info'].apply(lambda x: x.get('difficulty','') == d)]
    print(f'{d}: {len(subset)} entries')
# Count by node_count
for n in range(1, 8):
    subset = df[df['extra_info'].apply(lambda x: x.get('node_count', 0) == n)]
    if len(subset) > 0:
        print(f'  nodes={n}: {len(subset)}')
"
```

- [ ] **Step 2: Create filter script**

```python
# filter_training_data.py
# Keep only medium/complex tasks with node_count >= 3
# These are the tasks where topology decomposition matters
# Target: ~4000-6000 entries (balanced, not all complex)
```

- [ ] **Step 3: Generate targeted parquet**

```bash
python3 scripts/verl/filter_training_data.py \
    --input data/verl_topology_train.parquet \
    --output data/verl_topology_targeted.parquet \
    --min-nodes 3 \
    --difficulty medium,complex
```

- [ ] **Step 4: Commit filtered dataset**

```bash
git add scripts/verl/filter_training_data.py data/verl_topology_targeted.parquet
git commit -m "feat(training): filtered dataset for depth/horizon tasks (medium+complex, nodes>=3)"
git push origin main
```

---

### Task 4: Create targeted training script

**Files:**
- Create: `sage-python/scripts/verl/train_topology_targeted.sh`

Based on lessons learned:
- Resume from FSDP checkpoint step 100 on HF (download if not local)
- Use SFT merged model as base (NOT double-merged)
- SAGE_TRAINING_PHASE=A (simple reward, fast convergence)
- SAGE_VERL_EXEC=1 (execution reward when YAML valid)
- Targeted dataset (medium/complex, nodes>=3)
- 5 epochs (~1000 steps), save_freq=100, max_actor_ckpt_to_keep=2
- Checkpoints on NVMe, FULL FSDP uploaded to HF at each save
- NEVER merge LoRA during training

- [ ] **Step 1: Write the training script**

```bash
#!/bin/bash
# train_topology_targeted.sh
# Targeted training on depth/horizon tasks
# Resume from FSDP checkpoint or start fresh

set -euo pipefail
# ... (full script with all lessons learned)
```

- [ ] **Step 2: Test script in smoke mode (2 steps, verify plumbing)**

```bash
bash scripts/verl/train_topology_targeted.sh --smoke
```

Expected: 2 steps complete, checkpoint saved, no errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/verl/train_topology_targeted.sh
git commit -m "feat(training): targeted training script for depth/horizon topologies"
git push origin main
```

---

### Task 5: Download FSDP checkpoint from HF and resume training

**Files:**
- Modify: `sage-python/scripts/verl/monitor_training.sh`

- [ ] **Step 1: Download FSDP checkpoint from HF**

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'yannabadie/sage-topology-policy-v2',
    allow_patterns='checkpoints/combined_step_100/*',
    local_dir='/home/yann/verl_checkpoints_restore',
    token='<HF_TOKEN>'
)
# Restructure for verl resume
import shutil
shutil.move(
    '/home/yann/verl_checkpoints_restore/checkpoints/combined_step_100',
    '/home/yann/verl_checkpoints/global_step_100'
)
"
```

- [ ] **Step 2: Launch targeted training**

```bash
cd sage-python
nohup bash scripts/verl/train_topology_targeted.sh > /workspace/train_targeted.log 2>&1 &
```

- [ ] **Step 3: Restart monitor**

```bash
echo "A" > /tmp/training_phase
nohup bash scripts/verl/monitor_training.sh > /workspace/monitor.log 2>&1 &
```

- [ ] **Step 4: Verify first step and checkpoint**

```bash
# Wait for first step, verify reward, check FSDP saves
sleep 300
ray_dir=$(ls -dt /tmp/ray/session_* | head -1)
grep "training/global_step:" "$ray_dir"/logs/worker-*.out | tail -1
```

- [ ] **Step 5: Commit monitor update**

```bash
git add scripts/verl/monitor_training.sh
git commit -m "feat(training): monitor adapted for targeted training pipeline"
git push origin main
```

---

### Task 6: Monitor training and backup to HuggingFace

**Files:**
- Output: checkpoints on NVMe + HF

The monitor handles:
- Every 10 min: parse Ray metrics, push to GitHub
- Every save_freq=100: upload FULL FSDP checkpoint to HF (~34GB)
- Every save_freq=100: upload LoRA adapter to HF (~667MB)
- Detect crashes: restart from latest HF checkpoint

- [ ] **Step 1: Verify monitor uploads FSDP to HF after first checkpoint**

```bash
# After step 100:
grep "FSDP checkpoint uploaded" /workspace/monitor.log
# Verify on HF:
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='<token>')
files = [f for f in api.list_repo_tree('yannabadie/sage-topology-policy-v2', recursive=True)
         if hasattr(f,'rfilename') and 'targeted' in f.rfilename]
print(f'{len(files)} targeted checkpoint files on HF')
"
```

- [ ] **Step 2: Monitor reward progression — expect improvement over baseline**

Target metrics at step 500:
- reward/mean > 0.3 (vs 0.22 Phase A)
- exec_hits > 20% (vs 11% combined training)
- score/max > 0.9 (vs 0.864 combined)

---

### Task 7: Phase C — Multi-step micro-decisions (THE differentiator)

**Files:**
- Run: `sage-python/scripts/verl/train_phase_c_custom.py`
- Use: `sage-python/src/sage/verl/topology_env.py` (4-state machine)
- Use: `sage-python/src/sage/verl/rewardflow.py` (PageRank per-node credit)
- Use: `sage-python/src/sage/verl/edge_credit.py` (Graph-GRPO)
- Use: `sage-python/src/sage/verl/step_reward.py` (StepRewardVector for GiGPO anchoring)

**What makes Phase C unique:** The model doesn't just generate a topology — it OPERATES it. At each checkpoint node, it sees the output quality and decides:
- `continue` — quality OK, proceed to next node
- `upgrade` — quality low, re-run node with fallback_tier (e.g., budget → reasoner)
- `reroute` — topology is failing, abort and restart with different structure

This is the `SageTopologyEnv` 4-state machine:
```
awaiting_yaml → executing → awaiting_decision → terminal
                    ↑              ↓ (continue)
                    └──────────────┘
```

**Reward (5 signals):**
```
R = 0.20 × R_structural      + 0.35 × R_execution
  + 0.20 × R_rewardflow      + 0.15 × R_resilience
  + 0.10 × R_cost_efficiency
```

**MASBENCH alignment:** Phase C targets the `robustness` axis (0% bare model) — the model learns to recover from node failures through upgrade/reroute.

- [ ] **Step 1: Verify Phase C dependencies**

```bash
python3 -c "
from sage.verl.topology_env import SageTopologyEnv
from sage.verl.step_reward import StepRewardVector
from sage.verl.rewardflow import RewardFlowPropagator
print('Phase C deps: OK')
"
```

- [ ] **Step 2: Set SAGE_TRAINING_PHASE=C and launch Phase C**

```bash
cd sage-python
export SAGE_TRAINING_PHASE=C
python3 scripts/verl/train_phase_c_custom.py \
    --model /workspace/sft_merged_model \
    --checkpoint /home/yann/verl_checkpoints \
    --data data/verl_topology_targeted.parquet \
    --output /home/yann/verl_checkpoints_phase_c \
    --epochs 3 --lr 5e-7 --k 4 --batch-size 4 \
    2>&1 | tee /workspace/train_phase_c.log
```

Key success signals:
- `step_advantage` non-zero (GiGPO multi-step working)
- Anchor states `decision:*` appear (model takes decisions)
- `upgrade` chosen when quality < threshold
- `continue` chosen when quality > threshold

- [ ] **Step 3: Monitor Phase C metrics**

Watch for:
- reward_mean > -0.5 (improving from initial -1.5)
- At least 20% episodes with successful adaptation
- Edge credit distribution (not all reward to first node)

- [ ] **Step 4: Backup Phase C checkpoint to HF**

```bash
python3 scripts/verl/upload_checkpoint.py --step <latest>
```

- [ ] **Step 5: Commit Phase C results**

```bash
git add TRAINING_LOG.md docs/benchmarks/
git commit -m "results: Phase C micro-decisions — GiGPO multi-step with SageTopologyEnv"
git push origin main
```

---

### Task 8: Post-training pipeline (merge + push HF + GGUF)

**Files:**
- Run: `sage-python/scripts/verl/post_training_pipeline.py all`

Only run AFTER training converges (reward > 0.5 or no improvement for 500 steps).

- [ ] **Step 1: Run post-training pipeline**

```bash
cd sage-python
python3 scripts/verl/post_training_pipeline.py all
```

This does: export LoRA → merge into Nemotron-8B (float16) → push to HF → quantize GGUF Q8_0 → push GGUF to HF.

- [ ] **Step 2: Verify HF repo has both versions**

```bash
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='<token>')
files = list(api.list_repo_tree('yannabadie/sage-topology-policy-v2', recursive=True))
safetensors = [f for f in files if hasattr(f,'rfilename') and f.rfilename.endswith('.safetensors') and not f.rfilename.startswith('checkpoint')]
gguf = [f for f in files if hasattr(f,'rfilename') and f.rfilename.endswith('.gguf')]
print(f'Full precision: {len(safetensors)} files')
print(f'GGUF: {len(gguf)} files')
"
```

Expected: 4 safetensors shards (~16GB total) + 1 GGUF Q8_0 (~8.5GB).

- [ ] **Step 3: Enable Path 6 and run final benchmark**

```bash
export SAGE_ENABLE_PATH6=1
python -m sage.bench --type masbench --axis depth --limit 20
python -m sage.bench --type gaia --level 1 --limit 20
```

Compare with pre-training results. This is the final validation.

- [ ] **Step 4: Update CLAUDE.md, TRAINING_LOG.md, HF model card with final results**

- [ ] **Step 5: Final commit and push**

```bash
git add CLAUDE.md TRAINING_LOG.md docs/benchmarks/
git commit -m "results: targeted training complete — MASBENCH/GAIA with Path 6 enabled"
git push origin main
```

---

## Success Criteria

| Criterion | Target | Verification |
|-----------|--------|-------------|
| MASBENCH depth delta | > +20pp (SAGE vs bare) | 50-task evaluation |
| GAIA Level 1 score | > 30% (competitive with leaderboard) | HF leaderboard submission |
| Training reward convergence | > 0.4 by step 500 | Ray metrics |
| Execution hit rate | > 25% | Monitor logs |
| FSDP backup on HF | Every 100 steps | HF repo check |
| Full precision model on HF | 16GB safetensors | HF repo |
| GGUF on HF | Q8_0 ~8.5GB | HF repo |
| Phase C step_advantage | Non-zero | GiGPO anchor logs |
| Phase C decision anchors | `decision:*` in logs | Training output |
| Phase C adaptation rate | > 20% episodes | Monitor logs |
| MASBENCH robustness delta | > +10pp (0% baseline) | Post-Phase C benchmark |
| Path 6 benchmark delta | > +10pp vs templates | Final benchmark |
| All results documented | TRAINING_LOG.md + HF card | Git history |

## Cost Estimate

| Phase | GPU hours | API cost | Total |
|-------|-----------|----------|-------|
| MASBENCH 50 tasks | 0 | ~$2 | $2 |
| GAIA Level 1 | 0 | ~$5 | $5 |
| Targeted training (1000 steps) | ~18h H100 | $0 | ~$55 |
| Post-training + GGUF | ~1h | $0 | ~$3 |
| Final benchmarks | 0 | ~$5 | $5 |
| **Total** | **~19h** | **~$12** | **~$70** |

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Training reward plateau (like Phase A) | Simple Conductor-style reward + targeted dataset should converge faster |
| FUSE checkpoint corruption | All saves on NVMe, FSDP uploaded to HF |
| LoRA merge destroys quality | NEVER merge during training, merge ONLY in post-training |
| vLLM TP=2 crash | TP=1, save_freq=100, can resume from HF |
| Provider API failures | Fallback = deepseek-chat (reliable, no rate limit) |
| MASBENCH shows no topology value | Pivot to function-calling approach (MAS-Orchestra) |
