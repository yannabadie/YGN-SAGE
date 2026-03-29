#!/bin/bash
# ============================================================
# YGN-SAGE Topology Training V5 — Post-mortem fixes from V3/V4
# ============================================================
# Model: nvidia/Nemotron-Orchestrator-8B (SFT-merged)
# Algorithm: GRPO via verl 0.7.1 (single-turn, NOT GiGPO — see TRAINING_LOG.md)
# Hardware: 2x H100 NVL 94GB (adapted from 1x)
# ============================================================
#
# FIXES FROM V3/V4 POST-MORTEM:
#   1. max_response_length: 512 → 1024
#      V4 clip_ratio=0.97-1.0 — all responses hit 512 token cap.
#      SFT YAML topologies are 400-800 tokens → truncated → unparseable → reward=0.
#      Must allow at least 1024 tokens for complete YAML.
#
#   2. lr: 5e-5 → 1e-6
#      V4 KL divergence: 0.0 → 0.020 in 18 steps (catastrophic drift).
#      5e-5 destroys SFT warmup signal in <20 steps.
#      RUNPOD_PLAN specified 1e-6. V3 script had wrong value.
#
#   3. max_model_len: 1024 → 2048
#      Must be >= max_prompt_length + max_response_length (512 + 1024 = 1536).
#      Set to 2048 for headroom.
#
#   4. Reward shaping added to reward.py
#      _score_format now gives partial credit for truncated-but-YAML-like text.
#      Reduces reward sparsity from ~97% zeros to ~60% (estimated).
#
#   5. gpu_memory_utilization: 0.3 → 0.35
#      V4 was stable at 64.65 GB / 94 GB. Slight increase for longer sequences.
#
# KEPT FROM V3/V4:
#   - batch_size=32 (stable, no OOM in V4)
#   - TORCH_SDPA backend (Qwen3 arch, no flashinfer)
#
# 2-GPU ADAPTATIONS:
#   - trainer.n_gpus_per_node=2 (FSDP across both GPUs)
#   - param_offload=False, optimizer_offload=True (activations need GPU RAM)
#   - tensor_model_parallel_size=2 (vLLM uses both GPUs for rollout)
#   - micro_batch_size_per_gpu=4 (conservative: 1024 response_len = large activations)
#   - gpu_memory_utilization=0.35 (shared with FSDP actor)
#   - rollout.n=4 (K=4 for GRPO grouping)
#   - temperature=0.7 (diversity for grouping)
#   - RAY memory guard disabled (container limit != real limit)
#
# ESTIMATED DURATION:
#   ~576 steps × ~90s/step ÷ 3600 = ~14h on 2x H100 NVL
#   (2x throughput: FSDP across 2 GPUs, no CPU offload needed)
# ============================================================

set -euo pipefail

# Load .env
if [ -f "/workspace/YGN-SAGE/.env" ]; then
    set -a && source /workspace/YGN-SAGE/.env && set +a
fi

cd /workspace/YGN-SAGE/sage-python

# verl 0.7.1 vanilla for single-turn (Phase A/B)
export PYTHONPATH="/workspace/verl-071:${PYTHONPATH:-}"

# Force SDPA attention backend (no flashinfer needed for Qwen3)
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

# Prevent Ray from killing workers due to container memory limit
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99

# ── Config ───────────────────────────────────────────────────
if [ -d "/workspace/sft_merged_model_v2" ]; then
    MODEL="/workspace/sft_merged_model_v2"
    echo "Using SFT+RL-merged model v2 (step 160 LoRA baked in)"
elif [ -d "/workspace/sft_merged_model" ]; then
    MODEL="/workspace/sft_merged_model"
    echo "Using SFT-merged model (YAML-aware base)"
else
    MODEL=${SAGE_MODEL:-"/workspace/patched_nemotron_orchestrator"}
    echo "WARNING: Using base model (no SFT warmup) — expect slower convergence"
fi
OUTPUT="/workspace/topology_verl_output"
REWARD_SCRIPT="/workspace/YGN-SAGE/sage-python/src/sage/verl/reward.py"
DATA_FULL="data/verl_topology_train.parquet"
DATA_CURATED="data/verl_topology_curated.parquet"

mkdir -p "$OUTPUT"

echo "=== YGN-SAGE verl 0.7.1 Training V5 (post-mortem fixes) ==="
echo "Model:    $MODEL (nvidia/Nemotron-Orchestrator-8B)"
echo "Data:     $DATA_FULL"
echo "Output:   $OUTPUT"
echo "Fixes:    max_response_length=1024, lr=1e-6, reward shaping"
echo "GPUs:     2x H100 NVL (FSDP, no CPU offload, TP=2 for vLLM)"
echo ""

# ── Step 0: Verify stack ────────────────────────────────────
echo "[0] Verifying stack..."
python3 -c "
import sys; sys.path.insert(0, '/workspace/verl-071')
import verl; print(f'verl: {verl.__version__}')
import vllm; print(f'vLLM: {vllm.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')

# Verify GRPO
from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
assert 'grpo' in ADV_ESTIMATOR_REGISTRY, 'GRPO NOT registered!'
print(f'GRPO: registered')

# Verify model
from transformers import AutoConfig
config = AutoConfig.from_pretrained('$MODEL')
print(f'model: {config.model_type} ({config.architectures})')
assert config.model_type in ('qwen3', 'qwen2'), f'Expected qwen3/qwen2, got {config.model_type}'

# Verify tokenizer has no <think>
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL', trust_remote_code=True)
test = tok.apply_chat_template(
    [{'role': 'user', 'content': 'test'}],
    tokenize=False, add_generation_prompt=True
)
assert '<think>' not in test, f'Tokenizer still has <think>! Patch failed.'
print(f'tokenizer: patched (no <think>)')

# Verify reward (with shaping)
from sage.verl.reward import compute_score, _score_format
# Full YAML should score high
score_full = compute_score('t', 'nodes:\n- role: coder\n  model_tier: budget\nreasoning: test', '', {})
# Truncated YAML should get partial credit (not -2.0)
score_trunc = _score_format('nodes:\n- role: coder\n  model_tier: budget\n  prompt: write code that')
print(f'reward: full={score_full:.3f}, truncated_format={score_trunc:.3f}')
assert score_trunc > -2.0, 'Reward shaping not working — truncated YAML should get partial credit'

# Verify data
import pandas as pd
df = pd.read_parquet('$DATA_FULL')
print(f'training data: {len(df)} entries')

print('=== All checks passed ===')
"

# ── Phase A: Structural GRPO (V5 fixes) ──────────────────────
echo ""
echo "=== Phase A V5: Structural GRPO (3 epochs, Nemotron-Orchestrator-8B) ==="
echo "    max_response_length=1024, lr=1e-6, reward shaping active"
echo ""

export SAGE_VERL_EXEC=0

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=0.95 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    data.train_files="$DATA_FULL" \
    data.val_files="$DATA_CURATED" \
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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
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
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.total_epochs=3 \
    trainer.project_name=sage_topology \
    trainer.experiment_name=grpo_nemotron_orch_8b_v5_r2 \
    trainer.default_local_dir="/home/yann/verl_checkpoints" \
    'trainer.logger=["console"]'

echo ""
echo "=== Phase A V5 complete ==="
echo "Model saved to: $OUTPUT"
echo "Next: python3 scripts/verl/post_training_pipeline.py all"
