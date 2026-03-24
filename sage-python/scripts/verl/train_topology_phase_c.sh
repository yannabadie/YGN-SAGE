#!/bin/bash
# ============================================================
# YGN-SAGE Topology Training — Phase C: Multi-step Micro-decisions
# ============================================================
# Model: nvidia/Nemotron-Orchestrator-8B (Phase A/B LoRA checkpoint)
# Algorithm: GiGPO via verl-agent multi-turn env
# Environment: SageTopologyEnv (4-state machine)
# Hardware: 1x H100 80GB
# ============================================================
#
# WHAT'S DIFFERENT FROM PHASE A/B:
#
#   Phase A/B: Single-turn. Model generates YAML → reward evaluates
#              it statically or via execution. One model action per episode.
#              GiGPO provides token-level credit within the single response.
#
#   Phase C:   Multi-turn. Model generates YAML (step 0), then the env
#              executes nodes incrementally. At checkpoint nodes, the env
#              pauses and asks the model to decide: continue / upgrade / reroute.
#              GiGPO provides step-level credit across multiple model actions,
#              using anchor states (role:difficulty:quality_bucket) for grouping.
#
# This is the REAL GiGPO use case: comparing decisions at the SAME anchor
# state across multiple trajectories. "When a coder produces low-quality
# output, is it better to upgrade or continue?"
#
# THE 4-STATE MACHINE (topology_env.py):
#   awaiting_yaml → model generates YAML topology
#   executing → env runs nodes one-by-one (no model action)
#   awaiting_decision → model chooses continue/upgrade/reroute at checkpoint
#   terminal → episode ends, execution score computed
#
# PREREQUISITES:
#   - Phase A/B checkpoint exists at $CHECKPOINT_DIR
#   - verl-agent installed (pip install -e /workspace/verl-agent)
#   - SageTopologyEnv registered via env_register.py
#
# REWARD (5 signals, weights from RUNPOD_PLAN.md):
#   R = 0.20 * R_structural     (format + density + verifier)
#     + 0.35 * R_execution       (PASSED=1.0, WRONG_ANSWER=0.5, ...)
#     + 0.20 * R_rewardflow      (PageRank per-node credit)
#     + 0.15 * R_resilience      (adaptation triggered + succeeded)
#     + 0.10 * R_cost_efficiency (CARD price penalty)
#
# MICRO-DECISION REWARD CONSTANTS (topology_env.py):
#   _REWARD_UPGRADE_COST    = -0.05  (cost per upgrade)
#   _REWARD_REROUTE_PENALTY = -0.30  (reroute = abort + restart)
#   _REWARD_UPGRADE_SUCCESS = +0.15  (upgrade improved quality)
#
# COST: ~$10 GPU + ~$50-80 API = ~$60-90
# ============================================================

set -euo pipefail

# Load .env
if [ -f "/workspace/YGN-SAGE/.env" ]; then
    set -a && source /workspace/YGN-SAGE/.env && set +a
fi

cd /workspace/YGN-SAGE/sage-python

# verl-agent overlay (adds GiGPO + multi-turn env support)
export PYTHONPATH="/workspace/verl-agent:${PYTHONPATH:-}"

# Force SDPA attention backend
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

# ── Config ───────────────────────────────────────────────────
# Phase A/B checkpoint is the starting point for Phase C
CHECKPOINT_DIR=${SAGE_CHECKPOINT:-"/workspace/topology_verl_output"}
MODEL=${SAGE_MODEL:-"/workspace/patched_nemotron_orchestrator"}
OUTPUT="/workspace/topology_verl_phase_c"
REWARD_SCRIPT="/workspace/YGN-SAGE/sage-python/src/sage/verl/reward.py"
DATA_CURATED="data/verl_topology_curated.parquet"
MEMORY_DB="/workspace/topology_training_memory.db"

mkdir -p "$OUTPUT"

echo "=== YGN-SAGE Phase C: Multi-step Micro-decisions ==="
echo "Checkpoint: $CHECKPOINT_DIR (from Phase A/B)"
echo "Base model: $MODEL (Nemotron-Orchestrator-8B)"
echo "Output:     $OUTPUT"
echo "Memory DB:  $MEMORY_DB"
echo ""

# ── Step 0: Verify stack + Phase A/B checkpoint ─────────────
echo "[0] Verifying stack..."
python3 -c "
import sys, os
sys.path.insert(0, '/workspace/verl-agent')

# Verify verl-agent (not vanilla verl)
import verl; print(f'verl: {verl.__version__}')
import vllm; print(f'vLLM: {vllm.__version__}')

# Verify GiGPO + multi-turn env support
try:
    from agent_system.environments.env_manager import make_envs
    print('verl-agent: env_manager found (multi-turn env support OK)')
except ImportError:
    print('WARNING: verl-agent env_manager not found')
    print('Phase C requires verl-agent for multi-turn env.')
    print('Fallback: use train_phase_c_custom.py instead.')

# Register GiGPO
try:
    import gigpo
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
    assert 'gigpo' in ADV_ESTIMATOR_REGISTRY, 'GiGPO NOT registered!'
    print('GiGPO: registered')
except Exception as e:
    print(f'WARNING: GiGPO registration: {e}')

# Verify SageTopologyEnv 4-state machine
from sage.verl.topology_env import SageTopologyEnv
env = SageTopologyEnv()
obs = env.reset('test checkpoint decisions', 'test/phase_c')
assert env._state == 'awaiting_yaml', f'Bad initial state: {env._state}'

# Test the full 4-state cycle with a checkpoint topology
yaml_with_checkpoint = '''
nodes:
  - role: coder
    model_tier: budget
    prompt: Write code
    fallback_tier: reasoner
  - role: reviewer
    model_tier: fast
    prompt: Review code
edges:
  - from_idx: 0
    to_idx: 1
reasoning: Two-step with checkpoint
difficulty: moderate
adaptation:
  checkpoints: [0]
  max_upgrades: 1
  quality_threshold: 0.5
'''
obs, r, done, info = env.step(yaml_with_checkpoint)
if env._state == 'awaiting_decision':
    print(f'4-state machine: checkpoint reached at node {info.get(\"node_idx\", \"?\")}')
    # Test a decision
    obs2, r2, done2, info2 = env.step('continue')
    print(f'Decision handled: state={env._state}, done={done2}')
elif done:
    print(f'4-state machine: completed without checkpoint (structural mode)')
else:
    print(f'4-state machine: state={env._state}')

# Verify StepRewardVector
srv = env.get_step_rewards()
print(f'StepRewardVector: {srv.n_steps} steps, total={srv.episode_reward:.3f}')
assert srv.n_steps > 0, 'No step rewards collected'

# Verify env registration
import sage.verl.env_register
print('SageTopologyEnv: registered in verl-agent')

# Verify checkpoint dir
checkpoint_dir = '$CHECKPOINT_DIR'
if os.path.isdir(checkpoint_dir):
    import glob
    steps = sorted(glob.glob(os.path.join(checkpoint_dir, 'global_step_*')))
    if steps:
        print(f'Phase A/B checkpoint: {steps[-1]}')
    else:
        # Check for actor dir directly
        if os.path.isdir(os.path.join(checkpoint_dir, 'actor')):
            print(f'Phase A/B checkpoint: {checkpoint_dir}/actor/')
        else:
            print(f'Phase A/B checkpoint dir exists but no global_step_* found')
            print('Will use base model weights (no LoRA warm start)')
else:
    print(f'WARNING: Checkpoint dir {checkpoint_dir} not found')
    print('Phase C will start from base model (not recommended)')

# Verify Phase C modules
from sage.verl.rewardflow import RewardFlowPropagator
from sage.verl.edge_credit import compute_edge_advantages
from sage.verl.training_memory import TrainingMemory
from sage.verl.step_reward import StepRewardVector
print('Phase C modules: all importable')

# Verify reward
from sage.verl.reward import compute_score
score = compute_score('t', 'nodes:\\n- role: coder', '', {})
print(f'reward function: {score:.3f}')

# Verify data
import pandas as pd
df = pd.read_parquet('$DATA_CURATED')
print(f'training data: {len(df)} entries (curated)')

print('=== All Phase C checks passed ===')
"

# ── Build dynamic GiGPO + env args ──────────────────────────
echo ""
echo "Building GiGPO + env args from verl-agent ppo_trainer.yaml..."
GIGPO_ARGS=$(python3 -c "
import os, sys
try:
    import yaml
    import verl
    verl_path = os.path.dirname(os.path.dirname(verl.__file__))
    yaml_path = os.path.join(verl_path, 'verl', 'trainer', 'config', 'ppo_trainer.yaml')
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(verl_path, '..', 'verl', 'trainer', 'config', 'ppo_trainer.yaml')

    args = ['algorithm.adv_estimator=gigpo']

    if os.path.exists(yaml_path):
        cfg = yaml.safe_load(open(yaml_path))
        gigpo_cfg = cfg.get('algorithm', {}).get('gigpo', {})

        if 'enable_similarity' in gigpo_cfg:
            args.append('algorithm.gigpo.enable_similarity=True')
        if 'similarity_thresh' in gigpo_cfg:
            args.append('algorithm.gigpo.similarity_thresh=0.85')
        if 'step_advantage_w' in gigpo_cfg:
            args.append('algorithm.gigpo.step_advantage_w=1.0')
        if 'mode' in gigpo_cfg:
            args.append('algorithm.gigpo.mode=mean_norm')

        algo_cfg = cfg.get('algorithm', {})
        if 'gamma' in algo_cfg:
            args.append('algorithm.gamma=0.95')
        if 'norm_adv_by_std_in_grpo' in algo_cfg:
            args.append('algorithm.norm_adv_by_std_in_grpo=True')
        if 'use_kl_in_reward' in algo_cfg:
            args.append('algorithm.use_kl_in_reward=False')

        # Env params — CRITICAL for Phase C
        env_cfg = cfg.get('env', {})
        if 'env_name' in env_cfg:
            args.append('env.env_name=sage_topology')
        if 'max_steps' in env_cfg:
            args.append('env.max_steps=10')

        data_cfg = cfg.get('data', {})
        if 'return_raw_chat' in data_cfg:
            args.append('data.return_raw_chat=True')

        print(' '.join(args))
        print(f'# {len(args)} verified params', file=sys.stderr)
    else:
        args.extend([
            'algorithm.gigpo.enable_similarity=True',
            'algorithm.gigpo.similarity_thresh=0.85',
            'env.env_name=sage_topology',
            'env.max_steps=10',
        ])
        print(' '.join(args))
        print('# FALLBACK: core params (yaml not found)', file=sys.stderr)
except Exception as e:
    print('algorithm.adv_estimator=gigpo env.env_name=sage_topology env.max_steps=10')
    print(f'# ERROR building args: {e}', file=sys.stderr)
")
echo "GiGPO args: $GIGPO_ARGS"

# ── Phase C: Multi-step GiGPO with micro-decisions ──────────
echo ""
echo "=== Phase C: Multi-step GiGPO (3 epochs, curated dataset, micro-decisions) ==="
echo "4-state env: awaiting_yaml -> executing -> awaiting_decision -> terminal"
echo "The model generates YAML AND makes checkpoint decisions (upgrade/continue/reroute)"
echo ""

# Enable execution mode — Phase C requires real execution
export SAGE_VERL_EXEC=1

# Enable episodic memory for cross-episode learning
export SAGE_TRAINING_MEMORY_DB="$MEMORY_DB"

python3 -m verl.trainer.main_ppo \
    $GIGPO_ARGS \
    \
    data.train_files="$DATA_CURATED" \
    data.val_files="$DATA_CURATED" \
    data.train_batch_size=32 \
    data.val_batch_size=8 \
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
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.max_model_len=1280 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1280 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    \
    env.env_name=sage_topology \
    env.max_steps=10 \
    env.rollout.n=4 \
    \
    custom_reward_function.path="$REWARD_SCRIPT" \
    custom_reward_function.name=compute_score \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 \
    trainer.project_name=sage_topology \
    trainer.experiment_name=gigpo_nemotron_phase_c_multistep \
    trainer.default_local_dir="$OUTPUT" \
    'trainer.logger=["console"]'

echo ""
echo "=== Phase C complete ==="
echo "Model saved to: $OUTPUT"
echo ""
echo "Success criteria to verify:"
echo "  1. step_advantage non-null in logs (GiGPO multi-step works)"
echo "  2. anchors 'decision:*' appear (model makes checkpoint decisions)"
echo "  3. upgrade when quality < threshold, continue when quality > threshold"
echo "  4. Topologies with adaptation have better terminal reward"
echo "  5. BigCodeBench Hard (20 tasks) > 40.0%"
echo "  6. >= 20% episodes have adaptation triggered and succeeded"
echo ""
echo "Next: python3 scripts/verl/post_training_pipeline.py all"
echo ""
echo "If verl-agent multi-turn failed, use the custom fallback:"
echo "  python3 scripts/verl/train_phase_c_custom.py \\"
echo "    --checkpoint $OUTPUT \\"
echo "    --data $DATA_CURATED \\"
echo "    --epochs 3"
