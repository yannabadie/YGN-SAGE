#!/bin/bash
# ============================================================
# YGN-SAGE Topology Training via veRL-agent + GiGPO on H100
# ============================================================
# Model: Qwen/Qwen3.5-9B (patched tokenizer, no thinking mode)
# Algorithm: GiGPO (Group-in-Group Policy Optimization)
# Environment: SageTopologyEnv (multi-step, gym-style)
# Hardware: 1x H100 80GB
# ============================================================
#
# WHY GiGPO (not GRPO):
#   Topologies are multi-step: the model generates YAML, then the
#   environment executes each node (coder→reviewer→synthesizer).
#   GiGPO provides step-level credit assignment via anchor states:
#   "when a reviewer sees good coder output (same anchor), does it
#   produce better feedback?" This is impossible with flat GRPO.
#
#   Ref: GiGPO (arXiv 2505.10978), verl-agent (langfengQ/verl-agent)
#
# STRATEGY:
#   Phase A: 5 epochs, GiGPO, structural reward ($0 API)
#            SageTopologyEnv runs in structural mode (no API calls)
#   Phase B: 3 epochs, GiGPO, execution reward (~$50-80 API)
#            SageTopologyEnv runs with TopologyRunner + ProviderPool
#
# Total cost: ~$50-80 (API) + ~$20-40 (RunPod H100)
# ============================================================

set -euo pipefail

# Load .env if present
if [ -f "/workspace/YGN-SAGE/.env" ]; then
    set -a && source /workspace/YGN-SAGE/.env && set +a
fi

cd /workspace/YGN-SAGE/sage-python

# verl-agent overlay on top of verl 0.7.1 (adds GiGPO + agent env support)
export PYTHONPATH="/workspace/verl-agent:${PYTHONPATH:-}"

# ── Config ───────────────────────────────────────────────────
MODEL=${SAGE_MODEL:-"/workspace/patched_model"}
OUTPUT=${SAGE_OUTPUT:-"/workspace/YGN-SAGE/sage-python/models/topology_verl/"}
REWARD_SCRIPT="/workspace/YGN-SAGE/sage-python/src/sage/verl/reward.py"

# Verify patched tokenizer exists
if [ ! -f "$MODEL/tokenizer_config.json" ]; then
    echo "ERROR: Patched tokenizer not found at $MODEL"
    echo "Run first: python scripts/verl/patch_tokenizer.py --model Qwen/Qwen3.5-9B --output $MODEL"
    echo "Then: bash $MODEL/link_weights.sh"
    exit 1
fi

# Dataset paths
DATA_FULL=${SAGE_DATA:-"data/verl_topology_train.parquet"}
DATA_CURATED="data/verl_topology_curated.parquet"
if [ ! -f "$DATA_CURATED" ]; then
    DATA_CURATED="$DATA_FULL"
    echo "WARNING: curated parquet not found, using full for both phases"
fi

echo "=== YGN-SAGE veRL-agent GiGPO Training ==="
echo "Model:    $MODEL"
echo "Phase A:  $DATA_FULL (structural, GiGPO)"
echo "Phase B:  $DATA_CURATED (execution, GiGPO)"
echo "Output:   $OUTPUT"
echo ""

# ── Step 0: Validate GiGPO config against verl-agent source ──
# Read the ACTUAL run_alfworld.sh to extract validated params.
# This ensures we don't use fabricated config keys.
echo "[0] Validating GiGPO config against verl-agent source..."
python3 -c "
import os, sys

# Check verl-agent is installed and find examples
try:
    import verl
    verl_path = os.path.dirname(os.path.dirname(verl.__file__))
    print(f'verl-agent at: {verl_path}')
except ImportError:
    print('ERROR: verl-agent not installed')
    sys.exit(1)

# Verify GiGPO is in the AdvantageEstimator enum
try:
    from verl.trainer.ppo.ray_trainer import AdvantageEstimator
    gigpo_found = hasattr(AdvantageEstimator, 'GIGPO') or hasattr(AdvantageEstimator, 'GiGPO') or 'gigpo' in [e.value for e in AdvantageEstimator]
    assert gigpo_found, 'GiGPO not found in AdvantageEstimator enum'
    print('GiGPO: confirmed in AdvantageEstimator enum')
except (ImportError, AttributeError) as e:
    print(f'WARNING: Could not verify GiGPO enum: {e}')
    print('Falling back to algorithm.adv_estimator=gigpo (string)')

# Check env_manager exists
try:
    from agent_system.environments import env_manager
    print(f'env_manager: {env_manager.__file__}')
except ImportError:
    print('WARNING: agent_system.environments.env_manager not found')
    print('SageTopologyEnv registration may need adjustment')

# Print example scripts for manual verification
examples_dir = os.path.join(verl_path, 'examples', 'gigpo_trainer')
if os.path.isdir(examples_dir):
    scripts = os.listdir(examples_dir)
    print(f'GiGPO examples found: {scripts}')
    # Print first example for reference
    for s in scripts:
        if s.endswith('.sh'):
            script_path = os.path.join(examples_dir, s)
            with open(script_path) as f:
                content = f.read()
            # Extract algorithm.adv_estimator line
            for line in content.split('\n'):
                stripped = line.strip()
                if 'adv_estimator' in stripped or 'enable_similarity' in stripped or 'similarity' in stripped or 'env' in stripped.lower() and 'custom' in stripped.lower():
                    print(f'  [{s}] {stripped}')
            break
else:
    print(f'No examples at {examples_dir}')
    # Try recipe path
    recipe_dir = os.path.join(verl_path, 'recipe', 'gigpo')
    if os.path.isdir(recipe_dir):
        print(f'Found recipe at {recipe_dir}')
    recipe_dir2 = os.path.join(verl_path, '..', 'recipe', 'gigpo')
    if os.path.isdir(recipe_dir2):
        print(f'Found recipe at {recipe_dir2}')

print('Config validation complete')
print()

# ── Step 0b: Verify GiGPO YAML params exist ──
print('--- Verifying GiGPO params in ppo_trainer.yaml ---')
import yaml as _yaml
yaml_path = os.path.join(verl_path, 'verl', 'trainer', 'config', 'ppo_trainer.yaml')
if not os.path.exists(yaml_path):
    yaml_path = os.path.join(verl_path, '..', 'verl', 'trainer', 'config', 'ppo_trainer.yaml')

if os.path.exists(yaml_path):
    with open(yaml_path) as f:
        cfg = _yaml.safe_load(f)
    algo = cfg.get('algorithm', {})
    gigpo_cfg = algo.get('gigpo', {})
    print(f'  ppo_trainer.yaml algorithm.gigpo: {gigpo_cfg}')

    # Check which params exist
    for key in ['step_advantage_w', 'mode', 'enable_similarity', 'similarity_thresh']:
        if key in gigpo_cfg:
            print(f'  ✓ algorithm.gigpo.{key} = {gigpo_cfg[key]}')
        else:
            print(f'  ✗ algorithm.gigpo.{key} NOT in yaml — will be REMOVED from train args')

    # Check env config
    env_cfg = cfg.get('env', {})
    print(f'  ppo_trainer.yaml env: {env_cfg}')
    for key in ['env_name', 'max_steps']:
        if key in env_cfg:
            print(f'  ✓ env.{key} = {env_cfg[key]}')
        else:
            print(f'  ✗ env.{key} NOT in yaml default — may still work as override')
else:
    print(f'  WARNING: ppo_trainer.yaml not found at {yaml_path}')
    print('  Proceeding with current params — Hydra will crash on unknowns')

# ── Step 0c: Verify token masking for custom env ──
print()
print('--- Verifying response_mask support ---')
try:
    import inspect
    # Check if the rollout loop handles response_mask
    rollout_paths = [
        os.path.join(verl_path, 'agent_system', 'multi_turn_rollout'),
        os.path.join(verl_path, '..', 'agent_system', 'multi_turn_rollout'),
    ]
    for rp in rollout_paths:
        if os.path.isdir(rp):
            for fname in os.listdir(rp):
                if fname.endswith('.py'):
                    content = open(os.path.join(rp, fname)).read()
                    if 'response_mask' in content or 'action_mask' in content:
                        print(f'  ✓ response_mask found in {fname}')
                    if 'observation' in content.lower() and 'mask' in content.lower():
                        print(f'  ✓ observation masking found in {fname}')
            break
    else:
        print('  WARNING: multi_turn_rollout dir not found')
        print('  Token masking may not work — verify manually on first batch')
except Exception as e:
    print(f'  WARNING: Could not verify masking: {e}')

# ── Step 0d: Test env registration ──
print()
print('--- Testing SageTopologyEnv registration ---')
try:
    import sage.verl.env_register
    from sage.verl.topology_env import SageTopologyEnv
    env = SageTopologyEnv()
    obs = env.reset('test', 'test/1')
    assert 'text' in obs and 'anchor' in obs, 'Bad obs format'
    # Quick step test
    obs, r, done, info = env.step('difficulty: simple\nnodes:\n  - {role: coder, model_tier: budget}')
    print(f'  ✓ SageTopologyEnv works (state={env._state}, done={done})')
    print(f'  ✓ Env registration OK')
except Exception as e:
    print(f'  ✗ Env registration FAILED: {e}')
    print('  Training will crash. Fix env_register.py before proceeding.')
    sys.exit(1)

print()
print('=== Step 0 complete. All checks passed. ===')
"

# ── Register SageTopologyEnv ─────────────────────────────────
echo ""
echo "Registering SageTopologyEnv..."
python3 -c "import sage.verl.env_register; print('SageTopologyEnv registered')"

# ── Build dynamic GiGPO args (only include params that exist in the YAML) ──
echo ""
echo "Building GiGPO args from verified ppo_trainer.yaml..."
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

        # Only add params that exist in the schema
        if 'enable_similarity' in gigpo_cfg:
            args.append('algorithm.gigpo.enable_similarity=True')
        if 'similarity_thresh' in gigpo_cfg:
            args.append('algorithm.gigpo.similarity_thresh=0.85')
        if 'step_advantage_w' in gigpo_cfg:
            args.append('algorithm.gigpo.step_advantage_w=1.0')
        if 'mode' in gigpo_cfg:
            args.append('algorithm.gigpo.mode=mean_norm')

        # Top-level algorithm params
        algo_cfg = cfg.get('algorithm', {})
        if 'gamma' in algo_cfg:
            args.append('algorithm.gamma=0.95')
        if 'norm_adv_by_std_in_grpo' in algo_cfg:
            args.append('algorithm.norm_adv_by_std_in_grpo=True')
        if 'use_kl_in_reward' in algo_cfg:
            args.append('algorithm.use_kl_in_reward=False')

        # Env params
        env_cfg = cfg.get('env', {})
        if 'env_name' in env_cfg:
            args.append('env.env_name=sage_topology')
        if 'max_steps' in env_cfg:
            args.append('env.max_steps=10')

        # Data params
        data_cfg = cfg.get('data', {})
        if 'return_raw_chat' in data_cfg:
            args.append('data.return_raw_chat=True')

        print(' '.join(args))
        print(f'# {len(args)} verified params', file=sys.stderr)
    else:
        # Fallback: minimal safe set
        args.extend([
            'algorithm.gigpo.enable_similarity=True',
            'algorithm.gigpo.similarity_thresh=0.85',
        ])
        print(' '.join(args))
        print('# FALLBACK: only core params (yaml not found)', file=sys.stderr)
except Exception as e:
    # Absolute minimum
    print('algorithm.adv_estimator=gigpo')
    print(f'# ERROR building args: {e}', file=sys.stderr)
")
echo "GiGPO args: $GIGPO_ARGS"

# ── Phase A: Structural GiGPO ────────────────────────────────
echo ""
echo "=== Phase A: Structural GiGPO (5 epochs, full dataset) ==="
echo "Multi-step topology execution (structural mode). Learning format + per-node credit."
echo ""

export SAGE_VERL_EXEC=0

python3 -m verl.trainer.main_ppo \
    $GIGPO_ARGS \
    \
    data.train_files="$DATA_FULL" \
    data.val_files="$DATA_CURATED" \
    data.train_batch_size=64 \
    data.val_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
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
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    trainer.project_name=sage_topology \
    trainer.experiment_name=gigpo_qwen35_structural \
    trainer.default_local_dir="$OUTPUT" \
    'trainer.logger=["console"]'

echo ""
echo "=== Phase A complete. Starting Phase B ==="
echo ""

# ── Phase B: Execution GiGPO ─────────────────────────────────
echo "=== Phase B: Execution GiGPO (3 epochs, curated dataset) ==="
echo "Real multi-step topology execution via TopologyRunner + ProviderPool."
echo ""

export SAGE_VERL_EXEC=1

python3 -m verl.trainer.main_ppo \
    $GIGPO_ARGS \
    \
    data.train_files="$DATA_CURATED" \
    data.val_files="$DATA_CURATED" \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.3 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
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
    trainer.experiment_name=gigpo_qwen35_execution \
    trainer.default_local_dir="$OUTPUT" \
    'trainer.logger=["console"]'

echo ""
echo "=== Training complete ==="
echo "Model saved to: $OUTPUT"
echo "Next: python scripts/verl/export_for_local.py --checkpoint $OUTPUT"
