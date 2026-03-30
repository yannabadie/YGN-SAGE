#!/bin/bash
# Phase 1: Direct ablation runner — no autoresearch_loop overhead.
# Runs SFT then N1 eval directly for each config.
# Total: ~8 experiments × ~50 min = ~7h. Run overnight.
#
# Usage: cd sage-python && bash scripts/run_phase1_direct.sh

set -euo pipefail
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

nvidia-smi -lgc 3105

CONFIGS=(
  "ablation_epochs_3"
  "ablation_lr_1e5"
  "ablation_lr_5e5"
  "ablation_mix_code"
  "ablation_mix_complex"
  "ablation_rank_16"
  "ablation_rank_64"
  "ablation_model_instruct"
)

echo "=== Phase 1 Direct Runner: ${#CONFIGS[@]} experiments ==="
echo "Start: $(date)"

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "============================================"
  echo "=== $cfg ==="
  echo "=== $(date) ==="
  echo "============================================"

  CONFIG_FILE="experiments/configs/${cfg}.json"
  OUTPUT_DIR=$(python -c "import json; print(json.load(open('$CONFIG_FILE'))['output'])")
  SFT_CHECKPOINT="${OUTPUT_DIR}/sft_checkpoint"

  # Step 1: SFT training
  echo "--- SFT training: $CONFIG_FILE ---"
  python -u scripts/train_local_qwen3_4b.py --config "$CONFIG_FILE" --sft-only

  # Step 2: N1 eval
  if [ -f "${SFT_CHECKPOINT}/adapter_config.json" ]; then
    echo "--- N1 eval: $SFT_CHECKPOINT ---"
    python -u scripts/eval_reward_holdout.py \
      --adapter "$SFT_CHECKPOINT" \
      --output "experiments/n1_${cfg}.json"
    echo "--- N1 result ---"
    python -c "import json; d=json.load(open('experiments/n1_${cfg}.json')); print(f'  avg={d[\"n1_reward_avg\"]:.4f} max={d[\"n1_reward_max\"]:.4f} P>0.3={d[\"n1_above_03\"]*100:.0f}%')"
  else
    echo "ERROR: No checkpoint at $SFT_CHECKPOINT"
  fi

  echo "--- $cfg done: $(date) ---"
done

echo ""
echo "=== Phase 1 complete: $(date) ==="
echo "Results:"
for cfg in "${CONFIGS[@]}"; do
  if [ -f "experiments/n1_${cfg}.json" ]; then
    python -c "import json; d=json.load(open('experiments/n1_${cfg}.json')); print(f'  ${cfg}: avg={d[\"n1_reward_avg\"]:.4f} P>0.3={d[\"n1_above_03\"]*100:.0f}%')"
  else
    echo "  ${cfg}: NO RESULT"
  fi
done

# Commit
git add experiments/n1_*.json experiments/journal.jsonl 2>/dev/null
git commit -m "metrics: Phase 1 SFT ablation — ${#CONFIGS[@]} experiments complete" 2>/dev/null
git push origin local 2>/dev/null
