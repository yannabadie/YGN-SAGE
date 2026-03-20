#!/bin/bash
# ============================================================
# YGN-SAGE Topology Training via veRL GRPO on H100
# ============================================================
# Model: Qwen/Qwen3.5-9B (dense 9B, Apache 2.0, hybrid GatedDeltaNet+attention)
# Algorithm: GRPO standard
# Hardware: 1x H100 80GB
# Docker: verlai/verl:vllm017.latest
# ============================================================
# STRATEGY: Mixed reward (structural → execution)
#   Phase A: 5 epochs structural only (format/structure/density, $0 API)
#   Phase B: 5 epochs execution with real multi-provider ProviderPool
#            (TopologyRunner executes each node with its assigned provider)
#
# Total cost: ~$60-80 (API) + ~$30-50 (RunPod H100)
# ============================================================

set -euo pipefail

# Load .env if present
if [ -f "/workspace/YGN-SAGE/.env" ]; then
    set -a && source /workspace/YGN-SAGE/.env && set +a
fi

# ── Config ───────────────────────────────────────────────────
MODEL=${SAGE_MODEL:-"Qwen/Qwen3.5-9B"}
OUTPUT=${SAGE_OUTPUT:-"/workspace/YGN-SAGE/sage-python/models/topology_verl/"}
REWARD_SCRIPT="/workspace/YGN-SAGE/sage-python/src/sage/verl/reward.py"

# Phase A uses FULL dataset (1965 prompts, $0 API — learn format on max diversity)
# Phase B uses CURATED dataset (499 prompts, ~$50 API — learn execution on best data)
DATA_FULL=${SAGE_DATA:-"data/verl_topology_train.parquet"}
DATA_CURATED="data/verl_topology_curated.parquet"
if [ ! -f "$DATA_CURATED" ]; then
    DATA_CURATED="$DATA_FULL"
    echo "WARNING: curated parquet not found, using full for both phases"
fi

echo "=== YGN-SAGE veRL GRPO Training ==="
echo "Model:    $MODEL"
echo "Phase A:  $DATA_FULL (structural, full dataset)"
echo "Phase B:  $DATA_CURATED (execution, curated)"
echo "Output:   $OUTPUT"
echo "Reward:   $REWARD_SCRIPT"
echo ""

# ── Common training args ─────────────────────────────────────
COMMON_ARGS=(
    algorithm.adv_estimator=grpo

    data.train_batch_size=64
    data.max_prompt_length=512
    data.max_response_length=512
    data.filter_overlong_prompts=True
    data.truncation=error
    data.return_raw_chat=True

    actor_rollout_ref.model.path="$MODEL"
    actor_rollout_ref.model.lora_rank=64
    actor_rollout_ref.model.lora_alpha=32
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.04
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.fsdp_config.param_offload=True

    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.temperature=0.4
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.rollout.num_speculative_tokens=0

    custom_reward_function.path="$REWARD_SCRIPT"
    custom_reward_function.name=compute_score

    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.n_gpus_per_node=1
    trainer.nnodes=1
    trainer.save_freq=20
    trainer.test_freq=5
    trainer.project_name=sage_topology
    trainer.default_local_dir="$OUTPUT"
    'trainer.logger=["console","wandb"]'
)

# ── Phase A: Structural (5 epochs, $0 API) ────────────────────
echo ""
echo "=== Phase A: Structural reward (5 epochs) ==="
echo "Learning YAML format, structure, and density. No API calls."
echo ""

export SAGE_VERL_EXEC=0

python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    data.train_files="$DATA_FULL" \
    trainer.total_epochs=5 \
    trainer.experiment_name=grpo_qwen35_structural

echo ""
echo "=== Phase A complete. Starting Phase B ==="
echo ""

# ── Phase B: Execution (5 epochs, multi-provider) ─────────────
echo "=== Phase B: Execution reward (5 epochs) ==="
echo "Real topology execution via TopologyRunner + ProviderPool."
echo "Each node executed by its assigned provider (multi-provider)."
echo "Requires: DEEPSEEK_API_KEY and/or GOOGLE_API_KEY"
echo ""

export SAGE_VERL_EXEC=1

python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    data.train_files="$DATA_CURATED" \
    trainer.total_epochs=5 \
    trainer.experiment_name=grpo_qwen35_execution

echo ""
echo "=== Training complete ==="
echo "Model saved to: $OUTPUT"
echo "Next: python scripts/verl/export_for_local.py --checkpoint $OUTPUT"
