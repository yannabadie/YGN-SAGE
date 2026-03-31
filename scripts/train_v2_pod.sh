#!/bin/bash
# ============================================================
# YGN-SAGE V2 Pod Training Pipeline
# ============================================================
# Full V2 training: SFT -> GRPO -> Eval -> HF Upload
#
# Model: Qwen/Qwen3-4B (base) + Phase C adapter
# Data: v2_final.jsonl (8633 entries, 7063 single-turn + 1570 multi-turn)
# Format: <tool_call> JSON with 2 tools (create_topology, adapt_topology)
# Pod: RunPod 1x H200 SXM 141GB
#
# Usage:
#   bash scripts/train_v2_pod.sh              # Full pipeline
#   bash scripts/train_v2_pod.sh --smoke      # Smoke test (CPU OK, <5min)
#   bash scripts/train_v2_pod.sh --skip-sft   # Skip SFT, start from GRPO
#   bash scripts/train_v2_pod.sh --sft-only   # SFT only
#   bash scripts/train_v2_pod.sh --eval-only  # Eval only (needs checkpoint)
#
# See: docs/superpowers/specs/2026-03-31-v2-pod-deployment.md
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
BASE_MODEL="/home/yann/qwen3_4b_base"
PHASE_C_ADAPTER="/home/yann/phase_c_adapter"
TRAINING_DIR="/home/yann/v2_training"
SFT_OUTPUT="${TRAINING_DIR}/sft_checkpoint"
GRPO_OUTPUT="${TRAINING_DIR}/grpo_checkpoint"
MERGED_OUTPUT="/home/yann/v2_merged"
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
mkdir -p "${TRAINING_DIR}" "${SFT_OUTPUT}" "${GRPO_OUTPUT}" "${METRICS_DIR}" "${LOG_DIR}"

# ── Timestamp helper ───────────────────────────────────────
ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "============================================================"
echo "YGN-SAGE V2 Pod Training Pipeline"
echo "============================================================"
echo "Time:        $(ts)"
echo "Base model:  ${BASE_MODEL}"
echo "Adapter:     ${PHASE_C_ADAPTER}"
echo "SFT data:    ${SFT_DATA}"
echo "Output:      ${TRAINING_DIR}"
echo "Smoke:       ${SMOKE}"
echo ""

# ── Preflight checks ─────────────────────────────────────
echo "[$(ts)] Preflight checks..."
python3 -c "
import torch, os, json
assert torch.cuda.is_available(), 'No CUDA'
n = torch.cuda.device_count()
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'GPUs: {n}x {torch.cuda.get_device_name(0)} ({vram:.0f} GB)')
assert os.path.exists('${BASE_MODEL}/config.json'), 'Base model missing at ${BASE_MODEL}'
assert os.path.exists('${SFT_DATA}'), 'Training data missing at ${SFT_DATA}'
assert os.path.exists('${HOLDOUT}'), 'Holdout missing at ${HOLDOUT}'
import sage_core; print(f'sage_core: OK ({len(dir(sage_core))} exports)')
from sage.verl.reward import compute_score; print('Reward function: OK')
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
    echo "[$(ts)] SMOKE MODE: reduced parameters"
else
    SFT_EPOCHS=3
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
    GRPO_LR="5e-6"
fi

# ════════════════════════════════════════════════════════════
# PHASE 1: V2 SFT
# ════════════════════════════════════════════════════════════
if [ "${SKIP_SFT}" = false ] && [ "${EVAL_ONLY}" = false ]; then
    echo ""
    echo "============================================================"
    echo "[$(ts)] PHASE 1: V2 SFT (${SFT_DATA})"
    echo "============================================================"

    cd "${SAGE_PYTHON}"
    python3 scripts/train_local_qwen3_4b.py \
        --model "${BASE_MODEL}" \
        --adapter "${PHASE_C_ADAPTER}" \
        --sft-data "${SFT_DATA}" \
        --sft-only \
        --sft-epochs ${SFT_EPOCHS} \
        --sft-lr ${SFT_LR} \
        --sft-max-samples ${SFT_MAX_SAMPLES} \
        --output "${TRAINING_DIR}" \
        --lora-rank 32 \
        --max-completion-length ${SFT_MAX_LENGTH} \
        2>&1 | tee "${LOG_DIR}/phase1_sft.log"

    # Copy metrics
    if [ -f "${TRAINING_DIR}/sft_metrics.jsonl" ]; then
        cp "${TRAINING_DIR}/sft_metrics.jsonl" "${METRICS_DIR}/"
    fi

    # Quick N1 sanity check after SFT
    echo ""
    echo "[$(ts)] Phase 1 N1 sanity check..."
    python3 scripts/eval_reward_holdout.py \
        --adapter "${SFT_OUTPUT}" \
        --model "${BASE_MODEL}" \
        --holdout "${HOLDOUT}" \
        2>&1 | tee "${LOG_DIR}/phase1_n1.log"

    echo "[$(ts)] Phase 1 SFT complete."
    cd "${REPO_ROOT}"
else
    echo "[$(ts)] Phase 1: SKIPPED"
    # If skipping SFT, GRPO uses the Phase C adapter directly
    if [ ! -d "${SFT_OUTPUT}" ] || [ ! -f "${SFT_OUTPUT}/adapter_config.json" ]; then
        echo "[$(ts)] No SFT checkpoint found, using Phase C adapter for GRPO"
        SFT_OUTPUT="${PHASE_C_ADAPTER}"
    fi
fi

if [ "${SFT_ONLY}" = true ]; then
    echo "[$(ts)] SFT-only mode. Exiting."
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

    # Enable execution reward for GRPO
    export SAGE_VERL_EXEC=1
    export SAGE_TRAINING_PHASE=C

    cd "${SAGE_PYTHON}"

    # Convert SFT data to GRPO prompts (parquet format)
    echo "[$(ts)] Converting SFT data to GRPO prompts..."
    python3 -c "
import json
import pandas as pd

prompts = []
with open('${SFT_DATA}', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        # Extract user prompt (first user turn)
        if 'turns' in entry and isinstance(entry['turns'], list):
            for turn in entry['turns']:
                if turn.get('role') == 'user':
                    prompts.append({
                        'prompt': [
                            {'role': 'system', 'content': entry.get('system_prompt', '')},
                            {'role': 'user', 'content': turn['content']},
                        ],
                        'task_id': entry.get('task_id', ''),
                        'difficulty': entry.get('difficulty', 'moderate'),
                    })
                    break
        elif 'prompt' in entry:
            prompts.append({
                'prompt': [
                    {'role': 'system', 'content': entry.get('system_prompt', '')},
                    {'role': 'user', 'content': entry['prompt']},
                ],
                'task_id': entry.get('task_id', ''),
                'difficulty': entry.get('difficulty', 'moderate'),
            })

max_samples = ${GRPO_MAX_SAMPLES}
if max_samples > 0:
    prompts = prompts[:max_samples]

df = pd.DataFrame(prompts)
out_path = '${TRAINING_DIR}/grpo_prompts.parquet'
df.to_parquet(out_path)
print(f'Saved {len(df)} GRPO prompts to {out_path}')
"

    python3 scripts/train_local_qwen3_4b.py \
        --model "${BASE_MODEL}" \
        --adapter "${SFT_OUTPUT}" \
        --data "${TRAINING_DIR}/grpo_prompts.parquet" \
        --output "${TRAINING_DIR}" \
        --epochs ${GRPO_EPOCHS} \
        --lr ${GRPO_LR} \
        --batch-size 1 \
        --lora-rank 32 \
        --num-generations ${GRPO_NUM_GEN} \
        --max-completion-length ${GRPO_MAX_COMP} \
        --grad-accum ${GRPO_GRAD_ACCUM} \
        --max-samples ${GRPO_MAX_SAMPLES} \
        2>&1 | tee "${LOG_DIR}/phase2_grpo.log"

    # Copy metrics
    if [ -f "${TRAINING_DIR}/grpo_metrics.jsonl" ]; then
        cp "${TRAINING_DIR}/grpo_metrics.jsonl" "${METRICS_DIR}/"
    fi

    # Disable execution mode after GRPO
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
# PHASE 3: EVALUATION
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
    if [ ! -f "${EVAL_ADAPTER}/adapter_config.json" ]; then
        EVAL_ADAPTER="${PHASE_C_ADAPTER}"
    fi
    echo "[$(ts)] Evaluating adapter: ${EVAL_ADAPTER}"

    # ── 3a. N1 Holdout Evaluation ─────────────────────────
    echo ""
    echo "[$(ts)] 3a. N1 Holdout Evaluation (50 prompts, structural reward)..."
    python3 scripts/eval_reward_holdout.py \
        --adapter "${EVAL_ADAPTER}" \
        --model "${BASE_MODEL}" \
        --holdout "${HOLDOUT}" \
        2>&1 | tee "${LOG_DIR}/phase3_n1.log"

    if [ "${SMOKE}" = false ]; then
        # ── 3b. MASBENCH Evaluation ───────────────────────
        echo ""
        echo "[$(ts)] 3b. MASBENCH Evaluation..."
        SAGE_ENABLE_PATH6=1 \
        SAGE_PATH6_ADAPTER="${EVAL_ADAPTER}" \
        SAGE_PATH6_MODEL="${BASE_MODEL}" \
        python3 -m sage.bench --type masbench --timeout 600 \
            2>&1 | tee "${LOG_DIR}/phase3_masbench.log"

        # ── 3c. BigCodeBench Hard Instruct ────────────────
        echo ""
        echo "[$(ts)] 3c. BigCodeBench Hard Instruct (50 tasks)..."
        SAGE_ENABLE_PATH6=1 \
        SAGE_PATH6_ADAPTER="${EVAL_ADAPTER}" \
        SAGE_PATH6_MODEL="${BASE_MODEL}" \
        python3 -m sage.bench --type bigcodebench --subset hard --split instruct --limit 50 \
            2>&1 | tee "${LOG_DIR}/phase3_bigcodebench.log"
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

print('Loading base model...')
base = AutoModelForCausalLM.from_pretrained(
    '${BASE_MODEL}',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map='cpu',
)
print('Loading tokenizer...')
tok = AutoTokenizer.from_pretrained('${UPLOAD_ADAPTER}', trust_remote_code=True)
print('Loading adapter...')
model = PeftModel.from_pretrained(base, '${UPLOAD_ADAPTER}')
print('Merging...')
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

api = HfApi(token=os.environ['HF_TOKEN'])
repo_id = 'yannabadie/sage-topology-policy-local'

# Upload merged model
print('Uploading merged model...')
api.upload_folder(
    folder_path='${MERGED_OUTPUT}',
    repo_id=repo_id,
    path_in_repo='v2_merged',
    commit_message='feat: V2 GRPO-trained model (SFT 8633 + GRPO 5-signal)',
)

# Upload LoRA adapter
print('Uploading LoRA adapter...')
api.upload_folder(
    folder_path='${UPLOAD_ADAPTER}',
    repo_id=repo_id,
    path_in_repo='v2_grpo_adapter',
    commit_message='feat: V2 GRPO LoRA adapter',
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
                path_in_repo=f'v2_metrics/{fname}',
                repo_id=repo_id,
                commit_message=f'metrics: V2 {fname}',
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
                    path_in_repo=f'v2_eval/{fname}',
                    repo_id=repo_id,
                    commit_message=f'eval: V2 {fname}',
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
echo "[$(ts)] V2 TRAINING PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Artifacts:"
echo "  SFT checkpoint:  ${SFT_OUTPUT}"
echo "  GRPO checkpoint: ${GRPO_OUTPUT}"
echo "  Merged model:    ${MERGED_OUTPUT}"
echo "  Metrics:         ${METRICS_DIR}"
echo "  Logs:            ${LOG_DIR}"
echo ""
echo "Evaluate locally:"
echo "  SAGE_ENABLE_PATH6=1 python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 20"
echo ""
echo "HuggingFace:"
echo "  https://huggingface.co/yannabadie/sage-topology-policy-local"
