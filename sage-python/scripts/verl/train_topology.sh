#!/bin/bash
# ============================================================
# YGN-SAGE Topology Training via veRL + GiGPO on H100
# ============================================================
# Model: Qwen/Qwen3-8B (or Qwen3.5 when available)
# Algorithm: GiGPO (Group-in-Group Policy Optimization)
# Hardware: 1x H100 80GB
# ============================================================

set -euo pipefail

# ── Config ───────────────────────────────────────────────────
MODEL=${SAGE_MODEL:-"Qwen/Qwen3-8B"}  # Override with SAGE_MODEL env var
DATA=${SAGE_DATA:-"/workspace/YGN-SAGE/sage-python/data/verl_topology_train.parquet"}
OUTPUT=${SAGE_OUTPUT:-"/workspace/YGN-SAGE/sage-python/models/topology_verl_gigpo/"}
REWARD_SCRIPT="/workspace/YGN-SAGE/sage-python/scripts/verl/reward_topology.py"

echo "=== YGN-SAGE veRL + GiGPO Training ==="
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Output: $OUTPUT"
echo ""

# ── Training ─────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$DATA" \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.4 \
    \
    custom_reward_function.path="$REWARD_SCRIPT" \
    custom_reward_function.name=compute_score \
    \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=3 \
    trainer.save_freq=50 \
    trainer.project_name=sage_topology \
    trainer.experiment_name=gigpo_qwen3_9b \
    trainer.default_local_dir="$OUTPUT" \
    \
    +trainer.val_before_train=False

echo ""
echo "=== Training complete ==="
echo "Model saved to: $OUTPUT"
echo "Next: Export LoRA adapter for local inference"
