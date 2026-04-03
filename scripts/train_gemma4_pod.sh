#!/bin/bash
# ============================================================
# Gemma4-26B-A4B Topology Policy Training Pipeline
# ============================================================
# Full Gemma4 training: SFT -> GRPO -> Eval -> HF Upload
#
# Model: google/gemma-4-26B-A4B-it (MoE, 25.2B total / 3.8B active)
# Data: v2_final.jsonl (8633 entries, 7063 single-turn + 1570 multi-turn)
# Format: <tool_call> JSON with 2 tools (create_topology, adapt_topology)
# Pod: RunPod 1x H200 SXM 141GB
# Branch: local-GM4
#
# Usage:
#   bash scripts/train_gemma4_pod.sh              # Full pipeline
#   bash scripts/train_gemma4_pod.sh --smoke      # Smoke test
#   bash scripts/train_gemma4_pod.sh --sft-only   # SFT only
#   bash scripts/train_gemma4_pod.sh --skip-sft   # GRPO from SFT checkpoint
#   bash scripts/train_gemma4_pod.sh --eval-only  # Eval only
#
# See: RUNPOD_PLAN_GEMMA4.md
# ============================================================

set -euo pipefail

# ── Parse flags ────────────────────────────────────────────
SMOKE=false
SKIP_SFT=false
SFT_ONLY=false
EVAL_ONLY=false
SKIP_GRPO=false
SKIP_EVAL=false
SKIP_UPLOAD=false

for arg in "$@"; do
    case $arg in
        --smoke)       SMOKE=true ;;
        --skip-sft)    SKIP_SFT=true ;;
        --sft-only)    SFT_ONLY=true ;;
        --eval-only)   EVAL_ONLY=true ;;
        --skip-grpo)   SKIP_GRPO=true ;;
        --skip-eval)   SKIP_EVAL=true ;;
        --skip-upload) SKIP_UPLOAD=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ── Paths ──────────────────────────────────────────────────
REPO_ROOT="/workspace/YGN-SAGE"
SAGE_PYTHON="${REPO_ROOT}/sage-python"
BASE_MODEL="google/gemma-4-26B-A4B-it"
MODEL_CACHE="/home/yann/gemma4_base"
TRAINING_DIR="/home/yann/gemma4_training"
SFT_OUTPUT="${TRAINING_DIR}/sft_checkpoint"
GRPO_OUTPUT="${TRAINING_DIR}/grpo_checkpoint"
MERGED_OUTPUT="/home/yann/gemma4_merged"
SFT_DATA="${SAGE_PYTHON}/data/v2_final.jsonl"
HOLDOUT="${SAGE_PYTHON}/experiments/holdout_50_toolcall.json"
METRICS_DIR="${TRAINING_DIR}/metrics"
LOG_DIR="${TRAINING_DIR}/logs"

# ── Environment ────────────────────────────────────────────
cd "${REPO_ROOT}"
if [ -f .env ]; then
    set -a && source .env && set +a
fi
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export SAGE_VERL_EXEC=0
export SAGE_TRAINING_PHASE=C
export HF_HUB_OFFLINE=0

# ── Setup directories ─────────────────────────────────────
mkdir -p "${TRAINING_DIR}" "${SFT_OUTPUT}" "${GRPO_OUTPUT}" "${METRICS_DIR}" "${LOG_DIR}" "${MODEL_CACHE}"

# ── Timestamp helper ───────────────────────────────────────
ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "============================================================"
echo "Gemma4-26B-A4B Topology Policy Training Pipeline"
echo "============================================================"
echo "Time:        $(ts)"
echo "Base model:  ${BASE_MODEL}"
echo "Model cache: ${MODEL_CACHE}"
echo "SFT data:    ${SFT_DATA}"
echo "Output:      ${TRAINING_DIR}"
echo "Smoke:       ${SMOKE}"
echo ""

# ════════════════════════════════════════════════════════════
# SETUP: Install deps + download model (first run only)
# ════════════════════════════════════════════════════════════
echo "[$(ts)] Setup: checking dependencies..."

pip install --quiet torch transformers peft trl datasets accelerate bitsandbytes 2>&1 | tail -1
cd "${SAGE_PYTHON}" && pip install --quiet -e ".[all,dev]" 2>&1 | tail -1
cd "${REPO_ROOT}"

# Download Gemma4 weights if not cached (~50GB)
if [ ! -f "${MODEL_CACHE}/config.json" ]; then
    echo "[$(ts)] Downloading Gemma4-26B-A4B-it (~50GB, first run only)..."
    python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch, os, shutil

cache_dir = '${MODEL_CACHE}'
model_id = '${BASE_MODEL}'

print('Downloading config + tokenizer...')
config = AutoConfig.from_pretrained(model_id)
config.save_pretrained(cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(cache_dir)

print('Downloading model weights (this takes ~15min on fast connection)...')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir + '/_hf_cache',
)
model.save_pretrained(cache_dir)
del model

# Clean up HF cache to save disk space
hf_cache = os.path.join(cache_dir, '_hf_cache')
if os.path.exists(hf_cache):
    shutil.rmtree(hf_cache)

print(f'Gemma4 saved to {cache_dir}')
print('Download complete.')
"
    echo "[$(ts)] Download complete."
else
    echo "[$(ts)] Gemma4 weights already cached at ${MODEL_CACHE}"
fi

# ── Preflight checks ─────────────────────────────────────
echo "[$(ts)] Preflight checks..."
python3 -c "
import torch, os, json

# GPU check
assert torch.cuda.is_available(), 'No CUDA'
n = torch.cuda.device_count()
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'GPU: {n}x {torch.cuda.get_device_name(0)} ({vram:.0f} GB)')
assert vram >= 120, f'Need >= 120GB VRAM, got {vram:.0f}GB'

# Model check
model_dir = '${MODEL_CACHE}'
assert os.path.exists(os.path.join(model_dir, 'config.json')), f'Model config missing at {model_dir}'
print(f'Model: OK ({model_dir})')

# Data check
sft_data = '${SFT_DATA}'
assert os.path.exists(sft_data), f'Training data missing at {sft_data}'
with open(sft_data) as f:
    n_lines = sum(1 for _ in f)
print(f'Training data: OK ({n_lines} examples)')

# Holdout check
holdout = '${HOLDOUT}'
assert os.path.exists(holdout), f'Holdout missing at {holdout}'
print(f'Holdout: OK')

# Reward function check
from sage.verl.reward import compute_score
print('Reward function: OK')

print('All preflight checks passed.')
"

# ── Smoke mode overrides ──────────────────────────────────
if [ "${SMOKE}" = true ]; then
    SFT_EPOCHS=1
    SFT_MAX_SAMPLES=32
    SFT_LR="2e-5"
    SFT_BATCH_SIZE=2
    SFT_GRAD_ACCUM=2
    SFT_MAX_LENGTH=512
    GRPO_MAX_SAMPLES=16
    GRPO_EPOCHS=1
    GRPO_NUM_GEN=2
    GRPO_MAX_COMP=512
    GRPO_GRAD_ACCUM=2
    GRPO_LR="5e-6"
    echo "[$(ts)] SMOKE MODE: reduced parameters for quick test"
else
    SFT_EPOCHS=2
    SFT_MAX_SAMPLES=0
    SFT_LR="1e-5"
    SFT_BATCH_SIZE=4
    SFT_GRAD_ACCUM=4
    SFT_MAX_LENGTH=1024
    GRPO_MAX_SAMPLES=0
    GRPO_EPOCHS=1
    GRPO_NUM_GEN=4
    GRPO_MAX_COMP=1024
    GRPO_GRAD_ACCUM=8
    GRPO_LR="3e-6"
fi

# ── Data validation ───────────────────────────────────────
echo "[$(ts)] Validating training data..."
python3 -c "
import json, sys

data_path = '${SFT_DATA}'
errors = []
n_valid = 0
n_tool_call = 0

with open(data_path) as f:
    for i, line in enumerate(f, 1):
        try:
            d = json.loads(line)
            if 'messages' not in d:
                errors.append(f'Line {i}: missing messages field')
                continue
            n_valid += 1
            # Check for tool_call format in assistant messages
            for msg in d['messages']:
                if msg.get('role') == 'assistant' and '<tool_call>' in msg.get('content', ''):
                    n_tool_call += 1
                    break
        except json.JSONDecodeError as e:
            errors.append(f'Line {i}: invalid JSON: {e}')

print(f'Data validation: {n_valid} valid entries, {n_tool_call} with <tool_call>')
if errors:
    print(f'WARNING: {len(errors)} errors:')
    for e in errors[:5]:
        print(f'  {e}')
    if len(errors) > 5:
        print(f'  ... and {len(errors)-5} more')

assert n_valid > 0, 'No valid training data!'
assert n_tool_call > 0, 'No <tool_call> examples found!'
print('Data validation passed.')
"

# ════════════════════════════════════════════════════════════
# PHASE 1: SFT
# ════════════════════════════════════════════════════════════
if [ "${SKIP_SFT}" = false ] && [ "${EVAL_ONLY}" = false ]; then
    echo ""
    echo "============================================================"
    echo "[$(ts)] PHASE 1: SFT (Gemma4 learns <tool_call> JSON format)"
    echo "============================================================"
    echo "  Epochs:     ${SFT_EPOCHS}"
    echo "  LR:         ${SFT_LR}"
    echo "  Batch size: ${SFT_BATCH_SIZE} x ${SFT_GRAD_ACCUM} grad accum"
    echo "  Max length: ${SFT_MAX_LENGTH}"
    echo ""

    cd "${SAGE_PYTHON}"

    # CRITICAL: remove_unused_columns=False for mm_token_type_ids
    python3 scripts/train_gemma4_topology.py \
        --model "${MODEL_CACHE}" \
        --sft-data "${SFT_DATA}" \
        --sft-only \
        --sft-epochs ${SFT_EPOCHS} \
        --sft-lr ${SFT_LR} \
        --sft-max-samples ${SFT_MAX_SAMPLES} \
        --sft-batch-size ${SFT_BATCH_SIZE} \
        --sft-grad-accum ${SFT_GRAD_ACCUM} \
        --output "${TRAINING_DIR}" \
        --lora-rank 16 \
        --lora-alpha 32 \
        --lora-targets "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
        --max-completion-length ${SFT_MAX_LENGTH} \
        --bf16 \
        2>&1 | tee "${LOG_DIR}/phase1_sft.log"

    # Copy metrics
    if [ -f "${TRAINING_DIR}/sft_metrics.jsonl" ]; then
        cp "${TRAINING_DIR}/sft_metrics.jsonl" "${METRICS_DIR}/"
    fi

    # Quick N1 sanity check after SFT
    echo ""
    echo "[$(ts)] Phase 1 N1 sanity check..."
    python3 scripts/eval_reward_holdout_gemma4.py \
        --adapter "${SFT_OUTPUT}" \
        --model "${MODEL_CACHE}" \
        --holdout "${HOLDOUT}" \
        2>&1 | tee "${LOG_DIR}/phase1_n1.log"

    echo "[$(ts)] Phase 1 SFT complete."
    cd "${REPO_ROOT}"
else
    echo "[$(ts)] Phase 1: SKIPPED"
    # If skipping SFT, check for existing checkpoint
    if [ ! -d "${SFT_OUTPUT}" ] || [ ! -f "${SFT_OUTPUT}/adapter_config.json" ]; then
        echo "[$(ts)] WARNING: No SFT checkpoint found at ${SFT_OUTPUT}"
        if [ "${EVAL_ONLY}" = false ]; then
            echo "[$(ts)] GRPO will fail without SFT checkpoint. Run SFT first."
            exit 1
        fi
    fi
fi

if [ "${SFT_ONLY}" = true ]; then
    echo ""
    echo "============================================================"
    echo "[$(ts)] SFT-only mode. Exiting."
    echo "============================================================"
    echo "  SFT checkpoint: ${SFT_OUTPUT}"
    echo "  Next step: bash scripts/train_gemma4_pod.sh --skip-sft"
    exit 0
fi

# ════════════════════════════════════════════════════════════
# PHASE 2: GRPO WITH 5-SIGNAL REWARD
# ════════════════════════════════════════════════════════════
if [ "${SKIP_GRPO}" = false ] && [ "${EVAL_ONLY}" = false ]; then
    echo ""
    echo "============================================================"
    echo "[$(ts)] PHASE 2: GRPO (5-signal execution reward)"
    echo "============================================================"
    echo "  K rollouts: ${GRPO_NUM_GEN}"
    echo "  LR:         ${GRPO_LR}"
    echo "  Grad accum: ${GRPO_GRAD_ACCUM}"
    echo "  Max comp:   ${GRPO_MAX_COMP}"
    echo ""

    export SAGE_VERL_EXEC=0
    export SAGE_TRAINING_PHASE=C

    cd "${SAGE_PYTHON}"

    python3 scripts/train_gemma4_topology.py \
        --model "${MODEL_CACHE}" \
        --adapter "${SFT_OUTPUT}" \
        --data "${SFT_DATA}" \
        --output "${GRPO_OUTPUT}" \
        --epochs ${GRPO_EPOCHS} \
        --k ${GRPO_NUM_GEN} \
        --lr ${GRPO_LR} \
        --max-new-tokens ${GRPO_MAX_COMP} \
        --max-samples ${GRPO_MAX_SAMPLES} \
        --grad-accum ${GRPO_GRAD_ACCUM} \
        --beta 0.0 \
        --temperature 1.0 \
        --bf16 \
        --format-drift-interval 50 \
        --log-interval 5 \
        --save-interval 50 \
        2>&1 | tee "${LOG_DIR}/phase2_grpo.log"

    # Copy metrics
    if [ -f "${GRPO_OUTPUT}/grpo_metrics.jsonl" ]; then
        cp "${GRPO_OUTPUT}/grpo_metrics.jsonl" "${METRICS_DIR}/"
    fi

    # N1 check after GRPO
    echo ""
    echo "[$(ts)] Phase 2 N1 evaluation..."
    python3 scripts/eval_reward_holdout_gemma4.py \
        --adapter "${GRPO_OUTPUT}" \
        --model "${MODEL_CACHE}" \
        --holdout "${HOLDOUT}" \
        2>&1 | tee "${LOG_DIR}/phase2_n1.log"

    export SAGE_VERL_EXEC=0

    echo "[$(ts)] Phase 2 GRPO complete."
    cd "${REPO_ROOT}"
else
    echo "[$(ts)] Phase 2: SKIPPED"
    if [ ! -d "${GRPO_OUTPUT}" ] || [ ! -f "${GRPO_OUTPUT}/adapter_config.json" ]; then
        echo "[$(ts)] No GRPO checkpoint found, using SFT checkpoint for eval"
        GRPO_OUTPUT="${SFT_OUTPUT}"
    fi
fi

# ════════════════════════════════════════════════════════════
# PHASE 3: EVALUATION (MASBENCH + BigCodeBench)
# ════════════════════════════════════════════════════════════
if [ "${SKIP_EVAL}" = false ]; then
    echo ""
    echo "============================================================"
    echo "[$(ts)] PHASE 3: EVALUATION"
    echo "============================================================"

    cd "${SAGE_PYTHON}"

    # Determine which checkpoint to evaluate
    EVAL_ADAPTER="${GRPO_OUTPUT}"
    if [ ! -f "${EVAL_ADAPTER}/adapter_config.json" ]; then
        EVAL_ADAPTER="${SFT_OUTPUT}"
    fi
    echo "[$(ts)] Evaluating adapter: ${EVAL_ADAPTER}"

    # ── 3a. N1 Holdout Evaluation ─────────────────────────
    echo ""
    echo "[$(ts)] 3a. N1 Holdout Evaluation (50 prompts, structural reward)..."
    python3 scripts/eval_reward_holdout_gemma4.py \
        --adapter "${EVAL_ADAPTER}" \
        --model "${MODEL_CACHE}" \
        --holdout "${HOLDOUT}" \
        2>&1 | tee "${LOG_DIR}/phase3_n1.log"

    if [ "${SMOKE}" = false ]; then
        # ── 3b. MASBENCH Evaluation ───────────────────────
        echo ""
        echo "[$(ts)] 3b. MASBENCH Evaluation (50 tasks, depth comparison)..."
        SAGE_ENABLE_PATH6=1 \
        SAGE_PATH6_ADAPTER="${EVAL_ADAPTER}" \
        SAGE_PATH6_MODEL="${MODEL_CACHE}" \
        SAGE_PATH6_MODEL_TYPE=gemma4 \
        python3 -m sage.bench --type masbench --limit 50 --timeout 600 \
            2>&1 | tee "${LOG_DIR}/phase3_masbench.log"

        # Extract MASBENCH score
        MASBENCH_SCORE=$(grep -oP 'depth[:\s]+\K[0-9.]+' "${LOG_DIR}/phase3_masbench.log" | tail -1 || echo "0")
        echo "[$(ts)] MASBENCH depth score: ${MASBENCH_SCORE}%"

        # ── 3c. BigCodeBench Hard Instruct (if MASBENCH promising) ──
        if python3 -c "exit(0 if float('${MASBENCH_SCORE:-0}') > 35 else 1)" 2>/dev/null; then
            echo ""
            echo "[$(ts)] 3c. BigCodeBench Hard Instruct (50 tasks)..."
            SAGE_ENABLE_PATH6=1 \
            SAGE_PATH6_ADAPTER="${EVAL_ADAPTER}" \
            SAGE_PATH6_MODEL="${MODEL_CACHE}" \
            SAGE_PATH6_MODEL_TYPE=gemma4 \
            python3 -m sage.bench --type bigcodebench --subset hard --split instruct --limit 50 \
                2>&1 | tee "${LOG_DIR}/phase3_bigcodebench.log"
        else
            echo "[$(ts)] 3c. BigCodeBench: SKIPPED (MASBENCH score ${MASBENCH_SCORE}% < 35% threshold)"
        fi
    fi

    echo "[$(ts)] Phase 3 evaluation complete."
    cd "${REPO_ROOT}"
else
    echo "[$(ts)] Phase 3: SKIPPED"
fi

# ════════════════════════════════════════════════════════════
# PHASE 4: MERGE + UPLOAD TO HUGGINGFACE
# ════════════════════════════════════════════════════════════
if [ "${SKIP_UPLOAD}" = false ] && [ "${SMOKE}" = false ]; then
    echo ""
    echo "============================================================"
    echo "[$(ts)] PHASE 4: MERGE + UPLOAD TO HUGGINGFACE"
    echo "============================================================"

    UPLOAD_ADAPTER="${GRPO_OUTPUT}"
    if [ ! -f "${UPLOAD_ADAPTER}/adapter_config.json" ]; then
        UPLOAD_ADAPTER="${SFT_OUTPUT}"
    fi

    cd "${SAGE_PYTHON}"

    # ── 4a. Merge LoRA into base ──────────────────────────
    echo "[$(ts)] 4a. Merging LoRA into base model..."
    python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print('Loading base model (BF16)...')
base = AutoModelForCausalLM.from_pretrained(
    '${MODEL_CACHE}',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map='cpu',
)
print('Loading tokenizer...')
tok = AutoTokenizer.from_pretrained('${MODEL_CACHE}', trust_remote_code=True)
print('Loading adapter from ${UPLOAD_ADAPTER}...')
model = PeftModel.from_pretrained(base, '${UPLOAD_ADAPTER}')
print('Merging LoRA weights...')
merged = model.merge_and_unload()
merged.save_pretrained('${MERGED_OUTPUT}')
tok.save_pretrained('${MERGED_OUTPUT}')
print('Merged model saved to ${MERGED_OUTPUT}')
"

    # ── 4b. Upload to HuggingFace ─────────────────────────
    echo "[$(ts)] 4b. Uploading to HuggingFace..."
    python3 -c "
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ.get('HF_TOKEN'))
repo_id = 'yannabadie/sage-topology-policy-gemma4'

# Create repo if it doesn't exist
try:
    api.create_repo(repo_id, exist_ok=True, private=False)
except Exception as e:
    print(f'Repo creation note: {e}')

# Upload LoRA adapter
print('Uploading LoRA adapter...')
api.upload_folder(
    folder_path='${UPLOAD_ADAPTER}',
    repo_id=repo_id,
    path_in_repo='grpo_adapter',
    commit_message='feat: Gemma4 GRPO-trained topology adapter (SFT + GRPO 5-signal)',
)

# Upload metrics
metrics_dir = '${METRICS_DIR}'
if os.path.exists(metrics_dir):
    for fname in os.listdir(metrics_dir):
        fpath = os.path.join(metrics_dir, fname)
        if os.path.isfile(fpath):
            print(f'Uploading {fname}...')
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f'metrics/{fname}',
                repo_id=repo_id,
                commit_message=f'metrics: Gemma4 {fname}',
            )

# Upload eval logs
log_dir = '${LOG_DIR}'
if os.path.exists(log_dir):
    for fname in os.listdir(log_dir):
        if fname.startswith('phase3_'):
            fpath = os.path.join(log_dir, fname)
            if os.path.isfile(fpath):
                print(f'Uploading {fname}...')
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=f'eval/{fname}',
                    repo_id=repo_id,
                    commit_message=f'eval: Gemma4 {fname}',
                )

print(f'Done. See: https://huggingface.co/{repo_id}')
"

    echo "[$(ts)] Phase 4 upload complete."
    cd "${REPO_ROOT}"
else
    echo "[$(ts)] Phase 4: SKIPPED"
fi

# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "[$(ts)] GEMMA4 TRAINING PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Model:  google/gemma-4-26B-A4B-it (MoE, 25.2B/3.8B active)"
echo ""
echo "Artifacts:"
echo "  SFT checkpoint:  ${SFT_OUTPUT}"
echo "  GRPO checkpoint: ${GRPO_OUTPUT}"
echo "  Merged model:    ${MERGED_OUTPUT}"
echo "  Metrics:         ${METRICS_DIR}"
echo "  Logs:            ${LOG_DIR}"
echo ""
echo "Comparison baseline (Qwen3-4B Phase C):"
echo "  N1 avg:    0.922"
echo "  MASBENCH:  40% (4/10)"
echo "  Format:    ~95%"
echo ""
echo "Evaluate locally:"
echo "  SAGE_ENABLE_PATH6=1 SAGE_PATH6_MODEL_TYPE=gemma4 python -m sage.bench --type masbench --limit 50"
echo ""
echo "HuggingFace:"
echo "  https://huggingface.co/yannabadie/sage-topology-policy-gemma4"
