#!/bin/bash
# ============================================================
# YGN-SAGE Targeted Training — depth/horizon/robustness
# ============================================================
# Nemotron-Orchestrator-8B trained specifically for tasks where
# topology matters (base accuracy < 60% per AdaptOrch 2602.16873).
#
# Key improvements over combined training:
#   1. DAPO token-level loss (fixes GRPO entropy collapse + reward hacking)
#   2. Targeted dataset (medium/complex tasks, nodes >= 3)
#   3. Resume from HF FSDP checkpoint (no lost training)
#   4. SAGE_TRAINING_PHASE=A (simple Conductor-style reward)
#   5. SAGE_VERL_EXEC=1 (execution reward when YAML valid)
#   6. Full FSDP checkpoint backup to HF at each save
#
# Research backing:
#   - DAPO (arXiv 2503.14476): token-level loss, asymmetric clipping
#   - The Conductor (arXiv 2512.04388): binary reward converges in 200 iters
#   - AdaptOrch (arXiv 2602.16873): topology matters when base accuracy < 60%
#   - MASBENCH (Salesforce): depth/horizon/robustness = topology-sensitive axes
#
# Usage:
#   bash scripts/verl/train_topology_targeted.sh          # Full run
#   bash scripts/verl/train_topology_targeted.sh --resume  # Resume from HF
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

# Phase A reward (simple) + execution when YAML valid
export SAGE_TRAINING_PHASE=A
export SAGE_VERL_EXEC=1

# Model: SFT merged (verl creates fresh LoRA on top)
MODEL="/workspace/sft_merged_model"
OUTPUT="/home/yann/verl_checkpoints"
REWARD_SCRIPT="/workspace/YGN-SAGE/sage-python/src/sage/verl/reward.py"

# Dataset: targeted (medium/complex, nodes >= 3) or full if targeted doesn't exist
if [ -f "data/verl_topology_targeted.parquet" ]; then
    DATA="data/verl_topology_targeted.parquet"
    echo "Dataset: targeted (medium/complex tasks)"
else
    DATA="data/verl_topology_train.parquet"
    echo "Dataset: full (12K entries)"
fi

# Resume from HF checkpoint if --resume flag
RESUME_ARGS=""
if [ "${1:-}" = "--resume" ]; then
    CKPT=$(ls -dt "$OUTPUT"/global_step_* 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        echo "Resuming from: $CKPT"
        RESUME_ARGS="trainer.resume_mode=resume_path trainer.resume_from_path=$CKPT"
    else
        echo "No local checkpoint found. Download from HF first:"
        echo "  python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('yannabadie/sage-topology-policy-v2', allow_patterns='checkpoints/*', local_dir='/home/yann/verl_checkpoints_restore')\""
        exit 1
    fi
fi

mkdir -p "$OUTPUT"

echo "=== Targeted Training (DAPO + depth/horizon/robustness) ==="
echo "  Model: $MODEL"
echo "  Dataset: $DATA"
echo "  Reward: Phase A (simple) + execution"
echo "  Loss: DAPO token-level (not GRPO seq-level)"
echo "  Checkpoints: $OUTPUT (NVMe, FSDP complete, keep=2)"
echo ""

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=0.95 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    \
    data.train_files="$DATA" \
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
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
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
    trainer.total_epochs=5 \
    trainer.project_name=sage_topology \
    trainer.experiment_name=targeted_dapo_v1 \
    trainer.default_local_dir="$OUTPUT" \
    'trainer.logger=["console"]' \
    $RESUME_ARGS

echo ""
echo "=== Targeted training complete ==="
echo "Next: Phase C (train_phase_c_custom.py) or post_training_pipeline.py all"
