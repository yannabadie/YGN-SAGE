#!/bin/bash
# ============================================================
# YGN-SAGE Combined Training — Phase A (YAML) + Phase B (Exec)
# ============================================================
# Model: Phase A single-merged (SFT + step 1050 LoRA baked in)
# Algorithm: GRPO via verl 0.7.1
# Strategy: 10 epochs on full dataset (12K entries)
#   - SAGE_TRAINING_PHASE=A: simple Conductor-style reward
#   - SAGE_VERL_EXEC=1: execution reward fires when YAML is valid
#   - Combined: model learns format AND execution simultaneously
#
# CRITICAL LESSONS:
#   - NEVER merge LoRA before training is done
#   - Save FULL FSDP checkpoint (not just LoRA)
#   - Use NVMe for checkpoints (FUSE corrupts torch.save)
#   - max_actor_ckpt_to_keep=2 (always have a fallback)
#
# Duration: ~10h on 2x H100 NVL (3840 steps × ~10s/step)
# ============================================================

set -euo pipefail

# Load .env
if [ -f "/workspace/YGN-SAGE/.env" ]; then
    set -a && source /workspace/YGN-SAGE/.env && set +a
fi

cd /workspace/YGN-SAGE/sage-python

export PYTHONPATH="/workspace/verl-071:${PYTHONPATH:-}"
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99

# Phase A reward (simple Conductor-style) + execution when YAML valid
export SAGE_TRAINING_PHASE=A
export SAGE_VERL_EXEC=1

# Use Phase A single-merged model (SFT + step 1050 LoRA)
# verl creates fresh LoRA on top — the merged weights are the base
if [ -d "/workspace/sft_merged_model_phase_a" ]; then
    MODEL="/workspace/sft_merged_model_phase_a"
    echo "Base: Phase A merged (SFT + step 1050 LoRA)"
else
    MODEL="/workspace/sft_merged_model"
    echo "Base: SFT only (no Phase A LoRA)"
fi

OUTPUT="/home/yann/verl_checkpoints"
REWARD_SCRIPT="/workspace/YGN-SAGE/sage-python/src/sage/verl/reward.py"

mkdir -p "$OUTPUT"

echo "=== Combined Training (Phase A reward + execution) ==="
echo "  Model: $MODEL"
echo "  Dataset: 12303 entries (full)"
echo "  Epochs: 10 (~3840 steps)"
echo "  Reward: SAGE_TRAINING_PHASE=A (simple) + SAGE_VERL_EXEC=1"
echo "  Checkpoints: $OUTPUT (NVMe, FSDP complete, keep=2)"
echo ""

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=0.95 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    \
    data.train_files=data/verl_topology_train.parquet \
    data.val_files=data/verl_topology_curated.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
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
    actor_rollout_ref.rollout.temperature=0.7 \
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
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.total_epochs=10 \
    trainer.project_name=sage_topology \
    trainer.experiment_name=grpo_combined_v1 \
    trainer.default_local_dir="$OUTPUT" \
    'trainer.logger=["console"]'

echo ""
echo "=== Combined training complete ==="
echo "Checkpoint: $OUTPUT"
echo "Next: post_training_pipeline.py all"
