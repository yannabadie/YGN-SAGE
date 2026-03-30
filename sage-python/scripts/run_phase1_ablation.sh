#!/bin/bash
# Phase 1: Run all 8 SFT ablation experiments sequentially.
# Each experiment: SFT (~5-10 min) + N1 eval (~40 min) = ~50 min
# Total: ~7 hours. Run overnight.
#
# Usage: cd sage-python && bash scripts/run_phase1_ablation.sh

set -e
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

nvidia-smi -lgc 3105

echo "=== Phase 1: SFT Ablation (8 experiments) ==="
echo "Start: $(date)"

# Epoch ablation
python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_epochs_1.json \
  --hypothesis "1 SFT epoch (vs baseline 2): faster but may underfit YAML format" \
  --phase "1-sft-ablation" --budget 60 --sft-only
echo "--- epochs_1 done: $(date) ---"

python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_epochs_3.json \
  --hypothesis "3 SFT epochs (vs baseline 2): more exposure but risk overfitting" \
  --phase "1-sft-ablation" --budget 60 --sft-only
echo "--- epochs_3 done: $(date) ---"

# LR ablation
python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_lr_1e5.json \
  --hypothesis "lr=1e-5 (vs baseline 2e-5): slower convergence, potentially more stable" \
  --phase "1-sft-ablation" --budget 60 --sft-only
echo "--- lr_1e5 done: $(date) ---"

python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_lr_5e5.json \
  --hypothesis "lr=5e-5 (vs baseline 2e-5): faster convergence, risk of instability with 4-bit" \
  --phase "1-sft-ablation" --budget 60 --sft-only
echo "--- lr_5e5 done: $(date) ---"

# Data mix ablation
python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_mix_code.json \
  --hypothesis "BigCodeBench-only SFT data: specialized for coding tasks (drop GSM8K math)" \
  --phase "1-sft-ablation" --budget 60 --sft-only
echo "--- mix_code done: $(date) ---"

python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_mix_complex.json \
  --hypothesis "Complex-heavy data (5x complex, 2x moderate): address 0.148 complex weakness" \
  --phase "1-sft-ablation" --budget 60 --sft-only
echo "--- mix_complex done: $(date) ---"

# LoRA rank ablation
python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_rank_16.json \
  --hypothesis "LoRA rank 16 (vs baseline 32): less capacity, faster training, lower VRAM" \
  --phase "1-sft-ablation" --budget 60 --sft-only
echo "--- rank_16 done: $(date) ---"

python scripts/autoresearch_loop.py \
  --config experiments/configs/ablation_rank_64.json \
  --hypothesis "LoRA rank 64 (vs baseline 32): more capacity, may learn complex topologies better" \
  --phase "1-sft-ablation" --budget 60 --sft-only
echo "--- rank_64 done: $(date) ---"

echo "=== Phase 1 complete: $(date) ==="
echo "Check results: python -c \"import json; [print(json.loads(l)['id'], json.loads(l).get('metrics',{}).get('n1_reward_avg',0)) for l in open('experiments/journal.jsonl')]\""

# Commit results
git add experiments/journal.jsonl
git commit -m "metrics: Phase 1 complete — 8 SFT ablation experiments"
git push origin local
