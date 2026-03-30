#!/bin/bash
# ============================================================
# YGN-SAGE Tool-Call Training — Nemotron Native Format V2
# ============================================================
# Nemotron-Orchestrator-8B is a native JSON tool-caller.
# <tool_call> token id=151657, </tool_call> id=151658.
# <think> token id=151667 (BANNED via logit_bias in verl).
#
# V2 improvements over V1:
#   - Tool definitions baked into system prompt (triggers <tool_call>)
#   - 7 SAGE tools: create_topology, route_task, assign_models,
#     verify_topology, adapt_topology, execute_code, manage_memory
#   - 13K examples (5K simple + 5K moderate + 3K complex)
#   - max_prompt_length=2048 (tool schemas need ~1800 tokens)
#   - max_model_len=4096 (prompt + response fits)
#   - <think> banned via logit_bias in verl vllm_async_server.py
#
# Base model: nvidia/Nemotron-Orchestrator-8B (ORIGINAL weights)
#   - NOT sft_merged_model (YAML-damaged by SFT warmup)
#   - GRPO-trained by NVIDIA for JSON tool orchestration
#
# Training: DAPO token-level loss
#   - No SFT warmup needed (model already knows tool-calling)
#   - Reward: format (tool_call tags) + structure (valid topology)
#
# HuggingFace:
#   - Model: yannabadie/sage-topology-orchestrator
#   - Dataset: yannabadie/sage-topology-dataset
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

# Phase A reward (simple Conductor-style) + execution when valid
export SAGE_TRAINING_PHASE=A
export SAGE_VERL_EXEC=1

# ── CRITICAL: Use ORIGINAL NVIDIA weights ──
MODEL="/home/yann/nemotron_original"

# V2 tool-call dataset (generated from scratch with SAGE tools)
DATA="data/verl_topology_train_toolcall.parquet"
VAL="data/verl_topology_curated_toolcall.parquet"

OUTPUT="/home/yann/verl_checkpoints"
REWARD_SCRIPT="/workspace/YGN-SAGE/sage-python/src/sage/verl/reward.py"

mkdir -p "$OUTPUT"

echo "============================================================"
echo "  Tool-Call Training V2 — Nemotron Native + SAGE Tools"
echo "============================================================"
echo "  Model:   $MODEL (ORIGINAL NVIDIA weights)"
echo "  Dataset: $DATA (13K examples, 7 SAGE tools)"
echo "  Reward:  Phase A + tool_call format bonus"
echo "  Loss:    DAPO token-level"
echo "  Output:  $OUTPUT"
echo "============================================================"
echo ""

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=0.95 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    \
    data.train_files="$DATA" \
    data.val_files="$VAL" \
    data.train_batch_size=16 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_shm=False \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
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
    trainer.experiment_name=toolcall_v2 \
    trainer.default_local_dir="$OUTPUT" \
    'trainer.logger=["console"]'

echo ""
echo "=== Tool-call training V2 complete ==="
echo "Next: Phase C (train_phase_c_custom.py) or post_training_pipeline.py all"
