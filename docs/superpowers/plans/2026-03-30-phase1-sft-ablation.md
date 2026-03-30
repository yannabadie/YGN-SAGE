# Phase 1: SFT Ablation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find the optimal SFT checkpoint as RL starting point by ablating one variable at a time (epochs, LR, data mix, LoRA rank). Each experiment uses the autoresearch loop (10-20 min budget).

**Architecture:** Create experiment configs, run `autoresearch_loop.py --sft-only`, evaluate with N1 holdout. The SFT data has 1880 examples (72% simple, 20% moderate, 8% complex). Ablation uses 400 samples for speed (~10 min/experiment), final champion trains on full 1880.

**Tech Stack:** autoresearch_loop.py (Phase 0), train_local_qwen3_4b.py, eval_reward_holdout.py

**Baseline:** avg reward 0.391, simple 0.567, moderate 0.441, complex 0.148

---

### File Structure

```
sage-python/
  scripts/
    filter_sft_data.py           # NEW: filter SFT JSONL by source/difficulty
  experiments/
    configs/
      ablation_epochs_1.json     # 1 epoch
      ablation_epochs_3.json     # 3 epochs
      ablation_lr_1e5.json       # lr=1e-5
      ablation_lr_5e5.json       # lr=5e-5
      ablation_mix_complex.json  # complex-heavy data mix
      ablation_mix_code.json     # BigCodeBench-only
      ablation_rank_16.json      # LoRA rank 16
      ablation_rank_64.json      # LoRA rank 64
    data/
      sft_complex_heavy.jsonl    # Upsampled complex examples
      sft_code_only.jsonl        # BigCodeBench only
```

---

### Task 1: Data Mix Filter Script

**Files:**
- Create: `sage-python/scripts/filter_sft_data.py`

- [ ] **Step 1: Write the filter script**

```python
#!/usr/bin/env python3
"""Filter/resample SFT data by source or difficulty.

Creates data mix variants for ablation experiments.

Usage:
    # BigCodeBench only
    python scripts/filter_sft_data.py --source BigCodeBench --output experiments/data/sft_code_only.jsonl

    # Complex-heavy: upsample complex 5x, moderate 2x
    python scripts/filter_sft_data.py --upsample-complex --output experiments/data/sft_complex_heavy.jsonl
"""
import argparse
import json
import random

random.seed(42)

SFT_DATA = "data/topology_sft_v2_combined.jsonl"


def load_all(path: str) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def filter_by_source(entries: list[dict], source: str) -> list[dict]:
    return [e for e in entries if e.get("task_id", "").startswith(source)]


def upsample_complex(entries: list[dict]) -> list[dict]:
    """Upsample: complex 5x, moderate 2x, simple 1x."""
    result = []
    for e in entries:
        diff = e.get("difficulty", "simple")
        if diff == "complex":
            result.extend([e] * 5)
        elif diff == "moderate":
            result.extend([e] * 2)
        else:
            result.append(e)
    random.shuffle(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Filter/resample SFT data")
    parser.add_argument("--input", default=SFT_DATA)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default=None, help="Filter by task_id prefix")
    parser.add_argument("--upsample-complex", action="store_true",
                        help="Upsample complex 5x, moderate 2x")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    entries = load_all(args.input)
    print(f"Loaded {len(entries)} entries")

    if args.source:
        entries = filter_by_source(entries, args.source)
        print(f"Filtered to {len(entries)} ({args.source})")

    if args.upsample_complex:
        entries = upsample_complex(entries)
        print(f"Upsampled to {len(entries)}")

    if args.max_samples > 0:
        entries = entries[:args.max_samples]

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    by_diff = {}
    for e in entries:
        d = e.get("difficulty", "unknown")
        by_diff[d] = by_diff.get(d, 0) + 1

    print(f"Wrote {len(entries)} entries to {args.output}")
    for k, v in sorted(by_diff.items()):
        print(f"  {k}: {v} ({100*v/len(entries):.0f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create data mix variants**

```bash
cd sage-python
mkdir -p experiments/data
python scripts/filter_sft_data.py --source BigCodeBench --output experiments/data/sft_code_only.jsonl
python scripts/filter_sft_data.py --upsample-complex --output experiments/data/sft_complex_heavy.jsonl
```

Expected:
- `sft_code_only.jsonl`: ~1140 entries (BigCodeBench only)
- `sft_complex_heavy.jsonl`: ~2839 entries (complex 5x=745, moderate 2x=734, simple 1x=1364)

- [ ] **Step 3: Commit**

```bash
git add scripts/filter_sft_data.py experiments/data/
git commit -m "feat: SFT data filter/resample for ablation experiments"
```

---

### Task 2: Create Ablation Configs

**Files:**
- Create: 8 JSON configs in `sage-python/experiments/configs/`

All configs use `sft_max_samples: 400` and `sft_only: true` (implied by running with `--sft-only`).
Output dirs are unique per experiment to avoid overwriting checkpoints.

- [ ] **Step 1: Create epoch ablation configs**

`experiments/configs/ablation_epochs_1.json`:
```json
{
  "model": "Qwen/Qwen3-4B",
  "sft_data": "data/topology_sft_v2_combined.jsonl",
  "sft_epochs": 1,
  "sft_lr": 2e-5,
  "sft_max_samples": 400,
  "output": "models/ablation/epochs_1",
  "lora_rank": 32
}
```

`experiments/configs/ablation_epochs_3.json`:
```json
{
  "model": "Qwen/Qwen3-4B",
  "sft_data": "data/topology_sft_v2_combined.jsonl",
  "sft_epochs": 3,
  "sft_lr": 2e-5,
  "sft_max_samples": 400,
  "output": "models/ablation/epochs_3",
  "lora_rank": 32
}
```

- [ ] **Step 2: Create LR ablation configs**

`experiments/configs/ablation_lr_1e5.json`:
```json
{
  "model": "Qwen/Qwen3-4B",
  "sft_data": "data/topology_sft_v2_combined.jsonl",
  "sft_epochs": 2,
  "sft_lr": 1e-5,
  "sft_max_samples": 400,
  "output": "models/ablation/lr_1e5",
  "lora_rank": 32
}
```

`experiments/configs/ablation_lr_5e5.json`:
```json
{
  "model": "Qwen/Qwen3-4B",
  "sft_data": "data/topology_sft_v2_combined.jsonl",
  "sft_epochs": 2,
  "sft_lr": 5e-5,
  "sft_max_samples": 400,
  "output": "models/ablation/lr_5e5",
  "lora_rank": 32
}
```

- [ ] **Step 3: Create data mix configs**

`experiments/configs/ablation_mix_code.json`:
```json
{
  "model": "Qwen/Qwen3-4B",
  "sft_data": "experiments/data/sft_code_only.jsonl",
  "sft_epochs": 2,
  "sft_lr": 2e-5,
  "sft_max_samples": 400,
  "output": "models/ablation/mix_code",
  "lora_rank": 32
}
```

`experiments/configs/ablation_mix_complex.json`:
```json
{
  "model": "Qwen/Qwen3-4B",
  "sft_data": "experiments/data/sft_complex_heavy.jsonl",
  "sft_epochs": 2,
  "sft_lr": 2e-5,
  "sft_max_samples": 400,
  "output": "models/ablation/mix_complex",
  "lora_rank": 32
}
```

- [ ] **Step 4: Create LoRA rank configs**

`experiments/configs/ablation_rank_16.json`:
```json
{
  "model": "Qwen/Qwen3-4B",
  "sft_data": "data/topology_sft_v2_combined.jsonl",
  "sft_epochs": 2,
  "sft_lr": 2e-5,
  "sft_max_samples": 400,
  "output": "models/ablation/rank_16",
  "lora_rank": 16
}
```

`experiments/configs/ablation_rank_64.json`:
```json
{
  "model": "Qwen/Qwen3-4B",
  "sft_data": "data/topology_sft_v2_combined.jsonl",
  "sft_epochs": 2,
  "sft_lr": 2e-5,
  "sft_max_samples": 400,
  "output": "models/ablation/rank_64",
  "lora_rank": 64
}
```

- [ ] **Step 5: Add ablation output dirs to .gitignore**

Append to `.gitignore`:
```
sage-python/models/ablation/
```

- [ ] **Step 6: Commit**

```bash
git add experiments/configs/ablation_*.json .gitignore
git commit -m "feat: 8 ablation configs for Phase 1 SFT sweep"
```

---

### Task 3: Run Epoch Ablation (exp-003, exp-004)

Each experiment: SFT with one variable changed → N1 eval on holdout.

- [ ] **Step 1: Run 1-epoch ablation**

```bash
cd sage-python
nvidia-smi -lgc 3105
python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_epochs_1.json \
  --hypothesis "1 SFT epoch (vs baseline 2): faster but may underfit YAML format" \
  --phase "1-sft-ablation" \
  --budget 20 \
  --sft-only
```

Wait for completion. Check: `tail -1 experiments/journal.jsonl | python -m json.tool`

- [ ] **Step 2: Run 3-epoch ablation**

```bash
python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_epochs_3.json \
  --hypothesis "3 SFT epochs (vs baseline 2): more exposure but risk overfitting" \
  --phase "1-sft-ablation" \
  --budget 30 \
  --sft-only
```

- [ ] **Step 3: Commit journal**

```bash
git add experiments/journal.jsonl
git commit -m "metrics: Phase 1 epoch ablation (1 vs 2 vs 3 epochs)"
```

---

### Task 4: Run LR Ablation (exp-005, exp-006)

- [ ] **Step 1: Run lr=1e-5 ablation**

```bash
python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_lr_1e5.json \
  --hypothesis "lr=1e-5 (vs baseline 2e-5): slower convergence, potentially more stable" \
  --phase "1-sft-ablation" \
  --budget 20 \
  --sft-only
```

- [ ] **Step 2: Run lr=5e-5 ablation**

```bash
python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_lr_5e5.json \
  --hypothesis "lr=5e-5 (vs baseline 2e-5): faster convergence, risk of instability with 4-bit" \
  --phase "1-sft-ablation" \
  --budget 20 \
  --sft-only
```

- [ ] **Step 3: Commit journal**

```bash
git add experiments/journal.jsonl
git commit -m "metrics: Phase 1 LR ablation (1e-5 vs 2e-5 vs 5e-5)"
```

---

### Task 5: Run Data Mix Ablation (exp-007, exp-008)

- [ ] **Step 1: Run code-only ablation**

```bash
python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_mix_code.json \
  --hypothesis "BigCodeBench-only SFT data: specialized for coding tasks (drop GSM8K math)" \
  --phase "1-sft-ablation" \
  --budget 20 \
  --sft-only
```

- [ ] **Step 2: Run complex-heavy ablation**

```bash
python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_mix_complex.json \
  --hypothesis "Complex-heavy data (5x complex, 2x moderate): address 0.148 complex weakness" \
  --phase "1-sft-ablation" \
  --budget 20 \
  --sft-only
```

- [ ] **Step 3: Commit journal**

```bash
git add experiments/journal.jsonl
git commit -m "metrics: Phase 1 data mix ablation (code-only vs complex-heavy)"
```

---

### Task 6: Run LoRA Rank Ablation (exp-009, exp-010)

- [ ] **Step 1: Run rank=16 ablation**

```bash
python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_rank_16.json \
  --hypothesis "LoRA rank 16 (vs baseline 32): less capacity, faster training, lower VRAM" \
  --phase "1-sft-ablation" \
  --budget 20 \
  --sft-only
```

- [ ] **Step 2: Run rank=64 ablation**

```bash
python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_rank_64.json \
  --hypothesis "LoRA rank 64 (vs baseline 32): more capacity, may learn complex topologies better" \
  --phase "1-sft-ablation" \
  --budget 20 \
  --sft-only
```

- [ ] **Step 3: Commit journal**

```bash
git add experiments/journal.jsonl
git commit -m "metrics: Phase 1 LoRA rank ablation (16 vs 32 vs 64)"
```

---

### Task 7: Analyze Results & Train Champion

- [ ] **Step 1: Print ablation summary**

```bash
cd sage-python
python -c "
import json
entries = []
with open('experiments/journal.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))

print(f'{'ID':<10} {'Phase':<20} {'N1 Avg':>8} {'N1 Max':>8} {'P>0.3':>6} {'Clip':>6} | Hypothesis')
print('-' * 100)
for e in entries:
    m = e.get('metrics', {})
    avg = m.get('n1_reward_avg', 0)
    mx = m.get('n1_reward_max', 0)
    above = m.get('n1_above_03', 0)
    clip = m.get('n1_clipped_ratio', 0)
    hyp = e.get('hypothesis', '')[:50]
    print(f'{e[\"id\"]:<10} {e.get(\"phase\",\"\"):<20} {avg:>8.4f} {mx:>8.4f} {above:>6.1%} {clip:>6.1%} | {hyp}')
"
```

- [ ] **Step 2: Select champion config**

Based on results, pick the config with highest N1 avg reward. If complex reward improved significantly, weight that.

Create `experiments/configs/champion_sft.json` combining the best values from ablation.

- [ ] **Step 3: Train champion on full data**

```bash
python scripts/autoresearch_loop.py \
  --config experiments/configs/champion_sft.json \
  --hypothesis "Champion SFT: best hyperparams from ablation on full 1880 samples" \
  --phase "1-champion" \
  --budget 90 \
  --sft-only
```

This trains on ALL 1880 samples (not 400) with the best hyperparams. ~60 min.

- [ ] **Step 4: Commit and push**

```bash
git add experiments/ .gitignore
git commit -m "feat: Phase 1 complete — SFT champion selected from 8 ablations"
git push origin local
```

---

### Summary: Phase 1 Exit Criteria

| Criteria | How to verify |
|----------|---------------|
| 8 ablation experiments in journal | `grep -c "1-sft-ablation" experiments/journal.jsonl` >= 8 |
| Champion config exists | `cat experiments/configs/champion_sft.json` |
| Champion trained on full data | `ls models/ablation/champion/sft_checkpoint/adapter_config.json` |
| Champion N1 > baseline (0.391) | Check journal entry for champion |
| Complex reward improved | Compare complex per-difficulty vs 0.148 baseline |

### Expected Timeline

| Experiment | GPU Time | Total with Overhead |
|-----------|----------|---------------------|
| 8 ablation runs (400 samples each) | 8 × 10 min | ~2h |
| N1 eval per run (50 prompts) | 8 × 40 min | ~5h |
| Champion full training | 60 min | 1.5h |
| Champion N1 eval | 40 min | 1h |
| **Total** | | **~9-10h GPU time** |

Can be run overnight. Each experiment is independent — can be interrupted and resumed.
