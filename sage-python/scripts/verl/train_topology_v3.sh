#!/bin/bash
# ============================================================
# YGN-SAGE Topology Training V3 — verl 0.7.1 + vLLM + GiGPO
# ============================================================
# Model: Qwen/Qwen3-8B
#         (Apache 2.0 license, Qwen3-8B transformer)
# Algorithm: GiGPO via verl 0.7.1 registry plugin
# Hardware: 1x H100 80GB
# ============================================================
#
# Research-optimized hyperparameters:
#   - kl_loss_coef=0.001 (all verl official configs, not 0.04)
#   - temperature=0.7 (rollout diversity for GiGPO grouping)
#   - entropy_coeff=0.001 (exploration bonus)
#   - similarity_thresh=0.85 (GiGPO anchor grouping for variable hashes)
#   - param_offload + optimizer_offload (standard for LoRA on H100)
#   - layered_summon + use_shm (memory-efficient weight sharing)
#
# Memory budget (Qwen3-8B bf16 on H100 80GB):
#   Actor (FSDP): ~16GB model + 1GB LoRA/optim = ~17GB
#   vLLM rollout: ~16GB model + KV cache = ~22GB at util=0.3
#   Ref: offloaded to CPU
#   Total: ~39GB / 81GB — comfortable margin
# ============================================================

set -euo pipefail

# Load .env
if [ -f "/workspace/YGN-SAGE/.env" ]; then
    set -a && source /workspace/YGN-SAGE/.env && set +a
fi

cd /workspace/YGN-SAGE/sage-python

# GiGPO plugin must be importable
export PYTHONPATH="/workspace/verl-071:${PYTHONPATH:-}"

# Force SDPA attention backend (no flashinfer needed)
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

# ── Config ───────────────────────────────────────────────────
MODEL=${SAGE_MODEL:-"/workspace/patched_qwen3_8b"}
OUTPUT="/workspace/topology_verl_output"
REWARD_SCRIPT="/workspace/YGN-SAGE/sage-python/src/sage/verl/reward.py"
DATA_FULL="data/verl_topology_train.parquet"
DATA_CURATED="data/verl_topology_curated.parquet"

mkdir -p "$OUTPUT"

echo "=== YGN-SAGE verl 0.7.1 GiGPO Training V3 ==="
echo "Model:    $MODEL (Qwen/Qwen3-8B)"
echo "Data:     $DATA_FULL (2225 entries)"
echo "Output:   $OUTPUT"
echo ""

# ── Step 0: Verify stack ────────────────────────────────────
echo "[0] Verifying stack..."
python3 -c "
import sys; sys.path.insert(0, '/workspace/verl-071')
import verl; print(f'verl: {verl.__version__}')
import vllm; print(f'vLLM: {vllm.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')

# Register GiGPO
import gigpo
from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
assert 'gigpo' in ADV_ESTIMATOR_REGISTRY, 'GiGPO NOT registered!'
print(f'GiGPO: registered')

# Verify model
from transformers import AutoConfig
config = AutoConfig.from_pretrained('$MODEL')
print(f'model: {config.model_type} ({config.architectures})')
assert config.model_type == 'qwen3', f'Expected qwen3, got {config.model_type}'

# Verify tokenizer has no <think>
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL', trust_remote_code=True)
test = tok.apply_chat_template(
    [{'role': 'user', 'content': 'test'}],
    tokenize=False, add_generation_prompt=True
)
assert '<think>' not in test, f'Tokenizer still has <think>! Patch failed.'
print(f'tokenizer: patched (no <think>)')

# Verify reward
from sage.verl.reward import compute_score
score = compute_score('t', 'nodes:\n- role: coder', '', {})
print(f'reward function: {score:.3f}')

# Verify data
import pandas as pd
df = pd.read_parquet('$DATA_FULL')
print(f'training data: {len(df)} entries')

print('=== All checks passed ===')
"

# ── Phase A: Structural GiGPO ────────────────────────────────
echo ""
echo "=== Phase A: Structural GiGPO (5 epochs, 2225 entries, Qwen3-8B) ==="
echo ""

export SAGE_VERL_EXEC=0

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    algorithm.gigpo.enable_similarity=True \
    algorithm.gigpo.similarity_thresh=0.85 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_norm \
    algorithm.gamma=0.95 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    data.train_files="$DATA_FULL" \
    data.val_files="$DATA_CURATED" \
    data.train_batch_size=64 \
    data.val_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=768 \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.max_model_len=1280 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1280 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    \
    custom_reward_function.path="$REWARD_SCRIPT" \
    custom_reward_function.name=compute_score \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    trainer.project_name=sage_topology \
    trainer.experiment_name=gigpo_qwen3_8b_v3 \
    trainer.default_local_dir="$OUTPUT" \
    'trainer.logger=["console"]'

echo ""
echo "=== Phase A complete ==="
echo "Model saved to: $OUTPUT"
echo "Next: python3 scripts/verl/post_training_pipeline.py all"
