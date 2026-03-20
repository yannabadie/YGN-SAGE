#!/bin/bash
# ============================================================
# YGN-SAGE Topology Training via veRL GRPO on H100
# ============================================================
# Model: Qwen/Qwen3.5-9B (9B, Apache 2.0, hybrid attention)
# Algorithm: GRPO (GiGPO is multi-turn only, irrelevant here)
# Hardware: 1x H100 80GB
# Docker: verlai/verl:vllm017.latest
# ============================================================

set -euo pipefail

# ── Config ───────────────────────────────────────────────────
MODEL=${SAGE_MODEL:-"Qwen/Qwen3.5-9B"}
DATA=${SAGE_DATA:-"/workspace/YGN-SAGE/sage-python/data/verl_topology_train.parquet"}
OUTPUT=${SAGE_OUTPUT:-"/workspace/YGN-SAGE/sage-python/models/topology_verl/"}
REWARD_SCRIPT="/workspace/YGN-SAGE/sage-python/scripts/verl/reward_topology.py"

echo "=== YGN-SAGE veRL GRPO Training ==="
echo "Model:  $MODEL"
echo "Data:   $DATA"
echo "Output: $OUTPUT"
echo "Reward: $REWARD_SCRIPT"
echo ""

# ── VRAM budget (H100 80GB) ─────────────────────────────────
# Model bf16:     ~18 GB
# LoRA r=64:      ~0.2 GB
# Ref (offload):  0 GB (CPU)
# vLLM KV cache:  ~48 GB (gpu_memory_utilization=0.6)
# Optimizer:      ~0.8 GB (LoRA only)
# Activations:    ~4 GB (grad checkpoint)
# Total:          ~71 GB → fits H100 80GB
# ─────────────────────────────────────────────────────────────

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$DATA" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    \
    custom_reward_function.path="$REWARD_SCRIPT" \
    custom_reward_function.name=compute_score \
    \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.project_name=sage_topology \
    trainer.experiment_name=grpo_qwen35_9b \
    trainer.default_local_dir="$OUTPUT" \
    'trainer.logger=["console","wandb"]'

echo ""
echo "=== Training complete ==="
echo "Model saved to: $OUTPUT"
echo "Next: python scripts/verl/export_for_local.py --checkpoint $OUTPUT"
