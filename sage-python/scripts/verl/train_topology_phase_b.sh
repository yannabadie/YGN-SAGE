#!/bin/bash
# ============================================================
# YGN-SAGE Phase B — Execution Reward (single-turn)
# ============================================================
# Follows Phase A. Adds REAL topology execution via 7 providers.
# Model learns which topologies actually WORK, not just format.
#
# Key differences from Phase A:
#   - SAGE_VERL_EXEC=1 → real API calls (DeepSeek, Google, OpenAI...)
#   - Dataset: curated 498 entries (adaptive-priority subset)
#   - LR: 1e-6 (conservative to preserve Phase A structural knowledge)
#   - Reward: 30% structural + 70% execution
#   - Cost: ~$30-50 in API calls
#   - Duration: ~1-2h (47 steps × 2-5 min/step with API latency)
#
# Usage:
#   bash scripts/verl/train_topology_phase_b.sh
#   bash scripts/verl/train_topology_phase_b.sh --checkpoint /path/to/step_N
# ============================================================

set -euo pipefail

# Load .env (API keys needed for execution)
if [ -f "/workspace/YGN-SAGE/.env" ]; then
    set -a && source /workspace/YGN-SAGE/.env && set +a
fi

cd /workspace/YGN-SAGE/sage-python

export PYTHONPATH="/workspace/verl-071:${PYTHONPATH:-}"
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99

# ── EXECUTION MODE ON ──
export SAGE_VERL_EXEC=1

# ── Find Phase A checkpoint ──
CHECKPOINT_ARG=""
if [ "${1:-}" = "--checkpoint" ] && [ -n "${2:-}" ]; then
    CHECKPOINT="$2"
    shift 2
else
    # Auto-find latest checkpoint across both locations
    CHECKPOINT=""
    for dir in /home/yann/verl_checkpoints /workspace/topology_verl_output; do
        if [ -d "$dir" ]; then
            latest=$(find "$dir" -maxdepth 1 -name "global_step_*" -type d 2>/dev/null | sort -t_ -k3 -n | tail -1)
            if [ -n "$latest" ]; then
                step=$(basename "$latest" | grep -oP '\d+')
                if [ -z "$CHECKPOINT" ] || [ "$step" -gt "$(basename "$CHECKPOINT" | grep -oP '\d+')" ]; then
                    CHECKPOINT="$latest"
                fi
            fi
        fi
    done
fi

if [ -z "$CHECKPOINT" ] || [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: No Phase A checkpoint found. Run Phase A first."
    exit 1
fi

STEP=$(basename "$CHECKPOINT" | grep -oP '\d+')
echo "=== Phase B: Execution Reward (single-turn) ==="
echo "  Checkpoint: $CHECKPOINT (step $STEP)"
echo "  Mode: SAGE_VERL_EXEC=1 (real API execution)"
echo "  Dataset: data/verl_topology_curated.parquet (498 entries)"
echo "  LR: 1e-6 (preserve Phase A structural knowledge)"
echo "  Cost: ~\$30-50 API calls"
echo ""

# ── Verify API keys ──
python3 -c "
import os
keys = ['DEEPSEEK_API_KEY', 'GOOGLE_API_KEY', 'OPENAI_API_KEY']
missing = [k for k in keys if not os.environ.get(k)]
if missing:
    print(f'WARNING: Missing API keys: {missing}')
    print('Phase B needs at least one provider for execution reward.')
else:
    print(f'API keys OK: {len(keys)} primary providers configured')
    optional = [k for k in ['GROK_API_KEY', 'MINIMAX_API_KEY', 'KIMI_API_KEY', 'OPEN_ROUTER_API_KEY'] if os.environ.get(k)]
    print(f'  + {len(optional)} optional providers')
"

MODEL="/workspace/sft_merged_model"
OUTPUT="/home/yann/verl_checkpoints"
REWARD_SCRIPT="/workspace/YGN-SAGE/sage-python/src/sage/verl/reward.py"

mkdir -p "$OUTPUT"

# ── Phase B training ──
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=0.95 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    \
    data.train_files=data/verl_topology_curated.parquet \
    data.val_files=data/verl_topology_curated.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    \
    custom_reward_function.path="$REWARD_SCRIPT" \
    custom_reward_function.name=compute_score \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$CHECKPOINT" \
    trainer.total_epochs=3 \
    trainer.project_name=sage_topology \
    trainer.experiment_name=grpo_phase_b_exec \
    trainer.default_local_dir="$OUTPUT" \
    'trainer.logger=["console"]'

echo ""
echo "=== Phase B complete ==="
echo "Checkpoint: $OUTPUT"
echo "Next: Phase C (train_phase_c_custom.py)"
