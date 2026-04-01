# V2 Pod Deployment Spec — RunPod 1x H200 SXM 141GB

**Date:** 2026-03-31
**Branch:** `local`
**Image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
**GPU:** 1x H200 SXM 141GB (single GPU — no FSDP needed)
**Model:** Qwen/Qwen3-4B (base) + local Phase C SFT adapter from `yannabadie/sage-topology-policy-local`
**Goal:** Full V2 training pipeline: SFT on v2_final.jsonl (8633 entries) -> GRPO with 5-signal reward -> N1 + MASBENCH eval -> HuggingFace upload

---

## Section 1: Pod Setup Commands

### 1.1 Pod Selection

RunPod template: **RunPod PyTorch 2.4.0** (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`)
GPU: **1x H200 SXM 141GB** (single GPU — full model fits without quantization if needed)
Disk: **200 GB** container volume (checkpoints), **80 GB** workspace volume (code + data)
Region: EU-RO-1 or US-TX-3 (lowest latency to DeepSeek/OpenAI)
Note: Python 3.11 pre-installed. PyTorch 2.4.0 + CUDA 12.4.1 pre-installed. No FSDP needed (single GPU).

### 1.2 Create User and System Dependencies

```bash
#!/bin/bash
set -euo pipefail

# ── 1. Create user yann with sudo ──────────────────────────
adduser --disabled-password --gecos "" yann
echo "yann ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
echo "yann:$(openssl rand -base64 32)" | chpasswd
su - yann

# ── 2. System dependencies (CUDA is pre-installed on RunPod) ──
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git curl wget tmux htop jq tree \
    build-essential pkg-config libssl-dev \
    sqlite3 libsqlite3-dev \
    python3-dev python3-pip python3-venv

# ── 3. Install Rust ────────────────────────────────────────
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup default stable
rustc --version  # Expect 1.90+

# ── 4. Install Node.js 22 LTS (for Claude Code CLI) ───────
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y -qq nodejs
node --version  # Expect v22.x

# ── 5. Install Claude Code CLI ─────────────────────────────
npm install -g @anthropic-ai/claude-code
claude --version
```

### 1.3 Clone Repo and Install Dependencies

```bash
#!/bin/bash
set -euo pipefail
source "$HOME/.cargo/env"

# ── 6. Clone repository ────────────────────────────────────
cd /workspace
git clone https://github.com/yannabadie/YGN-SAGE.git
cd YGN-SAGE
git checkout local

# ── 7. Create .env with API keys ───────────────────────────
cat > .env << 'ENVEOF'
GOOGLE_API_KEY=<YOUR_GOOGLE_API_KEY>
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
DEEPSEEK_API_KEY=<YOUR_DEEPSEEK_API_KEY>
GROK_API_KEY=<YOUR_GROK_API_KEY>
KIMI_API_KEY=<YOUR_KIMI_API_KEY>
MINIMAX_API_KEY=<YOUR_MINIMAX_API_KEY>
OPEN_ROUTER_API_KEY=<YOUR_OPEN_ROUTER_API_KEY>
HF_TOKEN=<YOUR_HF_TOKEN>
ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
ENVEOF

# Load env
set -a && source .env && set +a

# ── 8. Build sage-core (Rust + PyO3) ──────────────────────
cd /workspace/YGN-SAGE/sage-core
pip install maturin
maturin develop --features smt,onnx,cognitive,tool-executor --release 2>&1 | tail -5

# Verify Rust build
python3 -c "
import sage_core
print(f'sage_core exports: {len(dir(sage_core))}')
from sage_core import TopologyGraph, TopologyNode, TopologyEdge
print('TopologyGraph: OK')
from sage_core import QualityLabeler
print('QualityLabeler: OK')
"

# ── 9. Install sage-python ─────────────────────────────────
cd /workspace/YGN-SAGE
pip install -e "sage-python/.[all,dev]" 2>&1 | tail -5

# Verify Python SDK
python3 -c "
from sage.verl.reward import compute_score
score = compute_score('t', '<tool_call>{\"name\":\"create_topology\",\"arguments\":{\"nodes\":[{\"role\":\"coder\",\"model_tier\":\"fast\",\"prompt\":\"x\"}],\"edges\":[],\"difficulty\":\"simple\",\"reasoning\":\"test\"}}</tool_call>', '', {})
print(f'Reward function test: {score:.3f}')
assert score > 0.5, f'Reward too low: {score}'
print('OK')
"

# ── 10. Install training-specific dependencies ─────────────
pip install trl>=0.17.0 peft>=0.15.0 bitsandbytes>=0.45.0 \
    accelerate>=1.5.0 datasets>=3.4.0 transformers>=4.52.0 \
    huggingface-hub>=0.29.0 wandb 2>&1 | tail -3
```

### 1.4 Download Models and Data from HuggingFace

```bash
#!/bin/bash
set -euo pipefail
cd /workspace/YGN-SAGE

# ── 11. Download Qwen3-4B base model ──────────────────────
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('Qwen/Qwen3-4B', local_dir='/home/yann/qwen3_4b_base')
print(f'Base model at: {path}')
"

# ── 12. Download Phase C SFT adapter from HF ──────────────
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    'yannabadie/sage-topology-policy-local',
    local_dir='/home/yann/phase_c_adapter',
)
print(f'Phase C adapter at: {path}')
"

# ── 13. Verify training data exists ───────────────────────
python3 -c "
import json
data_path = '/workspace/YGN-SAGE/sage-python/data/v2_final.jsonl'
with open(data_path, encoding='utf-8') as f:
    lines = f.readlines()
multi = sum(1 for l in lines if 'turns' in json.loads(l) and isinstance(json.loads(l).get('turns'), list))
single = len(lines) - multi
print(f'v2_final.jsonl: {len(lines)} entries ({multi} multi-turn, {single} single-turn)')
assert len(lines) >= 8600, f'Expected 8633, got {len(lines)}'
print('OK')
"

# ── 14. Copy model to NVMe for fast I/O ──────────────────
# /home/yann/ is NVMe overlay (fast), /workspace/ is FUSE (slow)
# Model already downloaded to /home/yann/ in steps above
ls -lh /home/yann/qwen3_4b_base/
ls -lh /home/yann/phase_c_adapter/
```

### 1.5 Full Verification

```bash
#!/bin/bash
set -euo pipefail
cd /workspace/YGN-SAGE

python3 << 'PYEOF'
import torch
import json
import os

print("=== V2 Pod Verification ===")

# GPU
assert torch.cuda.is_available(), "No CUDA"
n_gpus = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPUs: {n_gpus}x {gpu_name} ({vram_gb:.0f} GB each)")
assert n_gpus >= 1, f"Need 1+ GPU, got {n_gpus}"
assert vram_gb >= 80, f"Need >= 80GB VRAM, got {vram_gb:.0f}"

# sage-core
import sage_core
print(f"sage_core: {len(dir(sage_core))} exports")

# Reward function
from sage.verl.reward import compute_score
score = compute_score('t', '<tool_call>{"name":"create_topology","arguments":{"nodes":[{"role":"coder","model_tier":"fast","prompt":"x"}],"edges":[],"difficulty":"simple","reasoning":"test"}}</tool_call>', '', {})
print(f"Reward function: {score:.3f}")
assert score > 0.5

# Training data
with open('sage-python/data/v2_final.jsonl', encoding='utf-8') as f:
    n_lines = sum(1 for _ in f)
print(f"v2_final.jsonl: {n_lines} entries")
assert n_lines >= 8600

# Base model
assert os.path.exists('/home/yann/qwen3_4b_base/config.json'), "Base model missing"
print("Base model: OK")

# Phase C adapter
adapter_dir = '/home/yann/phase_c_adapter'
assert os.path.exists(adapter_dir), "Phase C adapter missing"
print("Phase C adapter: OK")

# Holdout
holdout_path = 'sage-python/experiments/holdout_50_toolcall.json'
with open(holdout_path) as f:
    holdout = json.load(f)
print(f"Holdout: {len(holdout['prompts'])} prompts")

# API keys
keys = ['GOOGLE_API_KEY', 'OPENAI_API_KEY', 'DEEPSEEK_API_KEY']
for k in keys:
    assert os.environ.get(k), f"Missing {k}"
    print(f"{k}: {'*' * 8}...{os.environ[k][-4:]}")

print("\n=== ALL CHECKS PASSED ===")
PYEOF
```

---

## Section 2: Training Pipeline

### Overview

```
Phase 1: V2 SFT (v2_final.jsonl, 8633 entries)
    |-- Single-turn entries (7063): user prompt -> <tool_call> create_topology
    |-- Multi-turn entries (1570): user prompt -> create_topology -> checkpoint -> adapt_topology
    |-- 3 difficulty levels: simple (4500), moderate (2944), complex (1189)
    |
    v
Phase 2: GRPO with 5-signal execution reward
    |-- Structural (format + density + Rust verifier)
    |-- Execution (real API calls, 7 providers)
    |-- RewardFlow (PageRank per-node credit)
    |-- Resilience (adaptation success bonus)
    |-- Cost efficiency (CARD-style price penalty)
    |
    v
Phase 3: Evaluation (N1 holdout + MASBENCH)
    |-- N1: 50 holdout prompts, avg reward, P(reward > 0.3)
    |-- MASBENCH: multi-agent system benchmark, depth metric
    |-- BigCodeBench Hard Instruct: 50 tasks
    |
    v
Phase 4: Upload to HuggingFace
    |-- Merged model (full weights)
    |-- LoRA adapter
    |-- Training metrics + model card
```

### Phase 1: V2 SFT

**Objective:** Train the model to generate valid `<tool_call>` JSON topologies in both single-turn (create_topology) and multi-turn (create_topology + adapt_topology) formats.

**Base model:** Qwen/Qwen3-4B (4.02B params)
**Starting point:** Phase C adapter from `yannabadie/sage-topology-policy-local` (already SFT-trained, avg 0.922 on N1)
**Data:** `sage-python/data/v2_final.jsonl` (8633 entries)
**Format:** `<tool_call>` JSON with 2 tools (create_topology, adapt_topology)

**Configuration:**
| Parameter | Value | Justification |
|-----------|-------|---------------|
| base_model | /home/yann/qwen3_4b_base | Qwen3-4B on NVMe |
| adapter | /home/yann/phase_c_adapter | Phase C SFT (avg 0.922) |
| learning_rate | 1e-5 | Continued SFT on existing adapter |
| epochs | 3 | Sufficient for 8633 entries |
| batch_size | 8 | Single H200 141GB — plenty of VRAM |
| gradient_accumulation | 2 | Effective batch = 16 |
| max_length | 1024 | Tool-call topologies are 200-600 tokens |
| lora_rank | 32 | Same as local training |
| lr_scheduler | cosine | Standard for SFT |
| warmup_ratio | 0.05 | 5% warmup |
| bf16 | true | H100 native |
| gradient_checkpointing | true | Memory efficiency |

**Expected duration:** ~25 min (8633 / 16 effective batch * 3 epochs * ~0.3s/step)
**Expected loss:** Start ~1.0 (already SFT-trained), end ~0.6-0.8
**Success criteria:**
- Loss decreasing monotonically
- N1 avg reward >= 0.90 (maintain Phase C quality)
- 95%+ outputs are valid `<tool_call>` JSON

### Phase 2: GRPO with 5-Signal Execution Reward

**Objective:** Optimize topology quality via reinforcement learning with real execution feedback from 7 LLM providers.

**Starting point:** Phase 1 SFT checkpoint
**Data:** Same v2_final.jsonl prompts (user turns only, stripped of assistant responses)
**Reward mode:** `SAGE_VERL_EXEC=1`, `SAGE_TRAINING_PHASE=C`

**Reward function (5 signals):**
```
R_total = 0.20 * R_structural     # YAML format + Rust density + verifier
        + 0.35 * R_execution       # Sandbox test: PASSED=1.0, WRONG=0.5, ERROR=0.3
        + 0.20 * R_rewardflow      # PageRank per-node credit from K rollouts
        + 0.15 * R_resilience      # Adaptation success bonus (0.0 / 0.3 / 0.5)
        + 0.10 * R_cost_efficiency # 1.0 - tanh(cost / budget_ref[difficulty])
```

**Configuration:**
| Parameter | Value | Justification |
|-----------|-------|---------------|
| base_model | /home/yann/qwen3_4b_base | Qwen3-4B |
| adapter | Phase 1 SFT output | Fresh from SFT |
| learning_rate | 5e-6 | Lower than SFT for RL stability |
| epochs | 1 | Single pass, 4 rollouts each |
| batch_size | 1 | Per GPU (GRPO is memory-heavy) |
| gradient_accumulation | 8 | Effective batch = 8 |
| num_generations (K) | 4 | GRPO rollouts per prompt |
| max_completion_length | 1024 | Full topology |
| temperature | 1.0 | Exploration (Conductor-proven) |
| beta (KL) | 0.0 | No KL penalty (Conductor) |

**7 providers active for execution reward:**
1. DeepSeek (`deepseek-chat`) -- budget, primary
2. OpenAI (`gpt-5.4`) -- codex tier
3. Google (`gemini-3.1-pro-preview`) -- reasoner
4. Google (`gemini-3.1-flash-lite-preview`) -- fast
5. xAI (`grok-4-1-fast-reasoning`) -- budget-alt
6. Kimi -- optional
7. MiniMax (`minimax-m2.7`) -- optional

**Expected duration:** ~4-8 hours (8633 prompts * 4 rollouts / 8 effective batch * ~2s/step with API calls)
**Expected reward:** Start ~0.3-0.5 (structural), end ~0.6-0.8 (with execution)
**Success criteria:**
- reward/mean > 0.5 by step 200
- reward/mean > 0.6 by end of training
- No mode collapse (entropy > 0.1)
- PASSED rate > 20% on execution tests

### Phase 3: Evaluation

**N1 Evaluation (50 holdout prompts, no API cost):**
```bash
cd /workspace/YGN-SAGE/sage-python
python3 scripts/eval_reward_holdout.py \
    --adapter /home/yann/v2_training/grpo_checkpoint \
    --model /home/yann/qwen3_4b_base \
    --holdout experiments/holdout_50_toolcall.json
```

**Success thresholds:**
| Metric | Phase C baseline | V2 target | Fail |
|--------|-----------------|-----------|------|
| avg reward | 0.922 | >= 0.92 | < 0.85 |
| max reward | 0.99 | >= 0.99 | < 0.95 |
| P(reward > 0.3) | 98% | >= 98% | < 90% |
| clipped ratio | 2% | <= 5% | > 15% |

**MASBENCH (multi-agent depth):**
```bash
cd /workspace/YGN-SAGE/sage-python
SAGE_ENABLE_PATH6=1 \
SAGE_PATH6_ADAPTER=/home/yann/v2_training/grpo_checkpoint \
python3 -m sage.bench --type masbench --timeout 600
```

**BigCodeBench Hard Instruct (50 tasks):**
```bash
cd /workspace/YGN-SAGE/sage-python
SAGE_ENABLE_PATH6=1 \
SAGE_PATH6_ADAPTER=/home/yann/v2_training/grpo_checkpoint \
python3 -m sage.bench --type bigcodebench --subset hard --split instruct --limit 50
```

**Success thresholds:**
| Benchmark | Current | V2 target |
|-----------|---------|-----------|
| MASBENCH depth | 67% | >= 67% |
| BigCodeBench Hard | 37.8% | >= 40% |

### Phase 4: Upload to HuggingFace

```bash
cd /workspace/YGN-SAGE/sage-python

# Merge LoRA into base
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    '/home/yann/qwen3_4b_base',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained('/home/yann/v2_training/grpo_checkpoint', trust_remote_code=True)
model = PeftModel.from_pretrained(base, '/home/yann/v2_training/grpo_checkpoint')
merged = model.merge_and_unload()
merged.save_pretrained('/home/yann/v2_merged')
tok.save_pretrained('/home/yann/v2_merged')
print('Merged model saved to /home/yann/v2_merged')
"

# Push to HuggingFace
python3 -c "
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ['HF_TOKEN'])
repo_id = 'yannabadie/sage-topology-policy-local'

# Upload merged model
api.upload_folder(
    folder_path='/home/yann/v2_merged',
    repo_id=repo_id,
    path_in_repo='v2_merged',
    commit_message='feat: V2 GRPO-trained model (SFT + GRPO 5-signal)',
)

# Upload LoRA adapter
api.upload_folder(
    folder_path='/home/yann/v2_training/grpo_checkpoint',
    repo_id=repo_id,
    path_in_repo='v2_grpo_adapter',
    commit_message='feat: V2 GRPO LoRA adapter',
)

# Upload training metrics
for metrics_file in ['sft_metrics.jsonl', 'grpo_metrics.jsonl']:
    fpath = f'/home/yann/v2_training/{metrics_file}'
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f'v2_metrics/{metrics_file}',
            repo_id=repo_id,
            commit_message=f'metrics: V2 {metrics_file}',
        )

print(f'Uploaded to https://huggingface.co/{repo_id}')
"
```

---

## Section 3: Pod Training Script

This script goes at `scripts/train_v2_pod.sh` in the repo and is the single command to run the entire pipeline.

```bash
#!/bin/bash
# ============================================================
# YGN-SAGE V2 Pod Training Pipeline
# ============================================================
# Full V2 training: SFT -> GRPO -> Eval -> HF Upload
#
# Usage:
#   bash scripts/train_v2_pod.sh              # Full pipeline
#   bash scripts/train_v2_pod.sh --smoke      # Smoke test (CPU OK, <5min)
#   bash scripts/train_v2_pod.sh --skip-sft   # Skip SFT, start from GRPO
#   bash scripts/train_v2_pod.sh --sft-only   # SFT only
#   bash scripts/train_v2_pod.sh --eval-only  # Eval only (needs checkpoint)
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
    SFT_BATCH_SIZE=8
    SFT_GRAD_ACCUM=2
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
```

---

## Section 4: Claude Code Autonomous Prompt

Copy-paste this EXACT prompt into Claude Code on the pod. It contains everything Claude needs to run the full pipeline autonomously.

```
You are operating on a RunPod 1x H200 SXM 141GB pod. Your job is to run the V2 training pipeline for YGN-SAGE and handle any issues that arise.

## Step 0: Verify Environment

Run these checks FIRST, before doing anything else:

```bash
nvidia-smi
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB)')"
ls /home/yann/qwen3_4b_base/config.json
ls /home/yann/phase_c_adapter/adapter_config.json
wc -l /workspace/YGN-SAGE/sage-python/data/v2_final.jsonl
python3 -c "import sage_core; print('sage_core OK')"
python3 -c "from sage.verl.reward import compute_score; print('reward OK')"
env | grep -E "API_KEY|HF_TOKEN" | sed 's/=.*/=***/'
```

If any check fails, fix it before proceeding. Common fixes:
- sage_core missing: `cd /workspace/YGN-SAGE/sage-core && source ~/.cargo/env && maturin develop --features smt,onnx,cognitive,tool-executor --release`
- Base model missing: `python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B', local_dir='/home/yann/qwen3_4b_base')"`
- API keys missing: Edit /workspace/YGN-SAGE/.env and source it

## Step 1: Run Training

```bash
cd /workspace/YGN-SAGE
source .env
bash scripts/train_v2_pod.sh 2>&1 | tee /home/yann/v2_training/logs/full_pipeline.log
```

## Step 2: Monitor During Training

While training runs, monitor these metrics every 50 steps by reading the metrics JSONL:

### SFT Phase (Phase 1)
- Read `/home/yann/v2_training/sft_metrics.jsonl` (last 5 entries)
- HEALTHY: loss decreasing, grad_norm < 5.0
- WORRY: loss increasing after step 50, grad_norm > 10.0
- INTERVENE: loss > 3.0 after 100 steps (model not learning)

### GRPO Phase (Phase 2)
- Read `/home/yann/v2_training/grpo_metrics.jsonl` (last 5 entries)
- HEALTHY: reward increasing, loss negative (policy improving)
- WORRY: reward flat for 100+ steps, clipped_ratio > 0.5
- INTERVENE: reward/mean drops below 0.1 (catastrophic forgetting)

### Intervention Playbook

If reward drops below 0.1 during GRPO:
1. Kill the training process
2. Re-run GRPO with lower lr: add `--lr 1e-6` to the GRPO command
3. If still failing, skip GRPO and evaluate the SFT checkpoint instead

If OOM error:
1. Kill the training process
2. Re-run with reduced batch: edit the script to set GRPO_GRAD_ACCUM=4 instead of 8
3. If still OOM on H100, reduce GRPO_NUM_GEN from 4 to 2

If provider API errors during GRPO (rate limits, timeouts):
1. These are handled automatically -- the reward function falls back to structural scoring
2. Check `/home/yann/v2_training/logs/phase2_grpo.log` for the pattern "EXEC_FALLBACK"
3. If >80% of batches are fallback, the execution signal is too sparse
4. Fix: check API keys, try different providers, or set SAGE_VERL_EXEC=0 for structural-only GRPO

If NaN loss:
1. Kill the training process
2. This means the learning rate is too high
3. Re-run with `--lr 1e-6` (halved)

## Step 3: After Training Completes

1. Verify the eval results in `/home/yann/v2_training/logs/phase3_*.log`
2. Key metrics to report:
   - N1 avg reward (target: >= 0.92)
   - MASBENCH depth (target: >= 67%)
   - BigCodeBench Hard (target: >= 40%)
3. If eval passed, verify the HuggingFace upload succeeded:
   ```bash
   python3 -c "from huggingface_hub import HfApi; api = HfApi(); files = api.list_repo_files('yannabadie/sage-topology-policy-local'); [print(f) for f in files if 'v2' in f]"
   ```
4. Commit the metrics and logs to git:
   ```bash
   cd /workspace/YGN-SAGE
   git add sage-python/models/ sage-python/experiments/
   git commit -m "metrics: V2 pod training results (SFT + GRPO 5-signal)"
   git push origin local
   ```

## Key Facts
- Base model: Qwen/Qwen3-4B at /home/yann/qwen3_4b_base
- Training data: 8633 entries in sage-python/data/v2_final.jsonl (7063 single-turn, 1570 multi-turn)
- Format: <tool_call> JSON with 2 tools (create_topology, adapt_topology)
- Phase C adapter baseline: avg 0.922 on N1
- Reward: compute_score() in sage-python/src/sage/verl/reward.py
- All checkpoints go to /home/yann/ (NVMe, fast) -- NEVER save to /workspace/ (FUSE, slow, corrupt)
- The training script is scripts/train_v2_pod.sh
- The eval script is scripts/eval_reward_holdout.py
```

---

## Section 5: Monitoring

### Metrics to Watch

| Metric | Source | Healthy | Warning | Critical |
|--------|--------|---------|---------|----------|
| **SFT loss** | sft_metrics.jsonl | Decreasing, < 1.5 by end | Flat after step 100 | Increasing after step 50 |
| **SFT grad_norm** | sft_metrics.jsonl | 0.1 - 5.0 | > 10.0 | > 50.0 (exploding) or < 0.001 (vanishing) |
| **GRPO reward/mean** | grpo_metrics.jsonl | > 0.3 by step 50, > 0.5 by step 200 | Flat for 100+ steps | < 0.1 after step 50 (catastrophic) |
| **GRPO loss** | grpo_metrics.jsonl | Negative (policy improving) | Oscillating wildly | Positive and increasing (divergence) |
| **GRPO clipped_ratio** | grpo_metrics.jsonl | < 0.3 | 0.3 - 0.5 | > 0.5 (all outputs truncated) |
| **GPU memory** | `nvidia-smi` | < 85 GB per GPU | > 90 GB | OOM kill |
| **GPU utilization** | `nvidia-smi` | > 60% | < 30% (underutilized) | 0% (hung process) |
| **Exec fallback rate** | phase2_grpo.log | < 30% | 30-60% | > 80% (API broken) |

### Expected Timeline

| Phase | Duration | Steps | Key Milestones |
|-------|----------|-------|----------------|
| Setup + preflight | 5 min | -- | All checks pass |
| Phase 1: SFT | 25-35 min | ~1620 (8633/16 * 3 epochs) | loss < 1.0 by step 500 |
| Phase 1: N1 sanity | 3 min | -- | avg reward >= 0.90 |
| Phase 2: GRPO prompt prep | 2 min | -- | parquet file created |
| Phase 2: GRPO training | 4-8 hours | ~1080 (8633/8 * 1 epoch) | reward > 0.5 by step 200 |
| Phase 3: N1 eval | 3 min | -- | avg reward >= 0.92 |
| Phase 3: MASBENCH | 20-40 min | -- | depth >= 67% |
| Phase 3: BigCodeBench | 30-60 min | -- | >= 40% |
| Phase 4: Merge | 5 min | -- | 8GB merged model |
| Phase 4: Upload | 10 min | -- | HuggingFace updated |
| **TOTAL** | **6-10 hours** | | |

### When to Worry (Decision Tree)

```
SFT loss not decreasing after 100 steps?
    -> Check data loading: grep "Loaded.*SFT examples" in log
    -> If 0 examples loaded: data path wrong
    -> If examples loaded but loss flat: lr too low (try 5e-5)

GRPO reward stuck at 0?
    -> Check format: are outputs starting with <tool_call>?
    -> If <think> output: chat template not applying tool mode
    -> If truncated: increase max_completion_length to 1536
    -> If garbage text: SFT adapter not loaded (check adapter path)

GRPO reward drops from 0.5 to 0.1?
    -> Catastrophic forgetting. LR too high.
    -> Kill, restart with lr=1e-6
    -> If persists at 1e-6: skip GRPO, ship the SFT checkpoint

OOM during GRPO?
    -> Reduce num_generations from 4 to 2
    -> Reduce grad_accum from 8 to 4
    -> Last resort: reduce max_completion_length from 1024 to 768

API providers all timing out?
    -> Check internet: curl -s https://api.deepseek.com/v1/models | head -1
    -> If network down: set SAGE_VERL_EXEC=0, continue with structural-only
    -> If rate limited: training auto-falls-back, but exec signal is sparse
    -> Acceptable: structural-only GRPO still improves topology quality

N1 eval score dropped below 0.85?
    -> V2 training hurt quality. Do NOT upload.
    -> Roll back to Phase C adapter (known good: 0.922)
    -> Investigate: compare v2_final.jsonl distribution vs Phase C data

MASBENCH or BigCodeBench below baseline?
    -> The model may be overfitting to format, not improving reasoning
    -> Check if topologies are diverse (not all sequential 2-node)
    -> If uniform: add temperature=1.2 during GRPO, re-train
```

### Monitoring Commands

```bash
# Watch SFT progress (run in a separate tmux pane)
watch -n 30 'tail -5 /home/yann/v2_training/sft_metrics.jsonl | python3 -c "import sys,json; [print(f\"step={json.loads(l).get(\"step\",\"?\"):>5}  loss={json.loads(l).get(\"loss\",0):.4f}  grad={json.loads(l).get(\"grad_norm\",0):.3f}\") for l in sys.stdin]"'

# Watch GRPO progress (run in a separate tmux pane)
watch -n 30 'tail -5 /home/yann/v2_training/grpo_metrics.jsonl | python3 -c "import sys,json; [print(f\"step={json.loads(l).get(\"step\",\"?\"):>5}  reward={json.loads(l).get(\"reward\",0):.4f}  loss={json.loads(l).get(\"loss\",0):.4f}\") for l in sys.stdin]"'

# GPU utilization
watch -n 10 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader'

# Disk usage (NVMe checkpoints)
watch -n 60 'du -sh /home/yann/v2_training/*'

# Check exec fallback rate during GRPO
grep -c "EXEC_FALLBACK" /home/yann/v2_training/logs/phase2_grpo.log 2>/dev/null; grep -c "EXEC_REAL" /home/yann/v2_training/logs/phase2_grpo.log 2>/dev/null

# Full pipeline log tail
tail -f /home/yann/v2_training/logs/full_pipeline.log
```

### Checkpoint Backup Strategy

Checkpoints are saved to NVMe (`/home/yann/`) which is ephemeral on RunPod. Back up critical checkpoints:

```bash
# After SFT completes (before GRPO)
cd /workspace/YGN-SAGE/sage-python
python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_folder(
    folder_path='/home/yann/v2_training/sft_checkpoint',
    repo_id='yannabadie/sage-topology-policy-local',
    path_in_repo='v2_sft_backup',
    commit_message='backup: V2 SFT checkpoint (pre-GRPO)',
)
print('SFT checkpoint backed up to HuggingFace')
"

# After GRPO completes (before eval)
python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_folder(
    folder_path='/home/yann/v2_training/grpo_checkpoint',
    repo_id='yannabadie/sage-topology-policy-local',
    path_in_repo='v2_grpo_backup',
    commit_message='backup: V2 GRPO checkpoint (pre-eval)',
)
print('GRPO checkpoint backed up to HuggingFace')
"
```

---

## Appendix A: Cost Estimate

| Resource | Cost | Duration | Total |
|----------|------|----------|-------|
| RunPod 1x H200 SXM | ~$4.50/hr | 10 hours | ~$45.00 |
| API calls (GRPO execution) | ~$0.003/topology * 8633 * 4 rollouts | -- | ~$103.60 |
| DeepSeek budget fallback (70% of calls) | ~$0.001/call * 24000 | -- | ~$24.00 |
| **Total estimated** | | | **~$187** |

API cost can be reduced to $0 by running GRPO with `SAGE_VERL_EXEC=0` (structural-only). This loses the execution signal but still improves format and structure quality.

## Appendix B: Rollback Plan

If V2 training produces a worse model than Phase C:

```bash
# 1. Do NOT upload to HuggingFace (skip Phase 4)
# 2. Keep the Phase C adapter as the production model
# 3. Analyze what went wrong:

cd /workspace/YGN-SAGE/sage-python

# Compare Phase C vs V2 on N1
python3 scripts/eval_reward_holdout.py --adapter /home/yann/phase_c_adapter --model /home/yann/qwen3_4b_base
python3 scripts/eval_reward_holdout.py --adapter /home/yann/v2_training/grpo_checkpoint --model /home/yann/qwen3_4b_base

# If V2 SFT is good but GRPO hurt: ship V2 SFT, skip GRPO
# If V2 SFT is also worse: data quality issue in v2_final.jsonl
```

## Appendix C: Differences from Previous RunPod Training

| Aspect | Previous (Nemotron 8B, 2xH100) | V2 (Qwen3-4B, 1x H200) |
|--------|------------------------|----------------|
| Model | nvidia/Nemotron-Orchestrator-8B (8.19B) | Qwen/Qwen3-4B (4.02B) |
| VRAM per model | ~35 GB (bf16) | ~8 GB (bf16), ~2.5 GB (NF4) |
| Framework | verl 0.7.1 (FSDP + vLLM) | TRL (SFTTrainer + GRPOTrainer) |
| Quantization | None (full bf16) | NF4 4-bit (bitsandbytes) |
| LoRA rank | 64 | 32 |
| Training format | YAML (failed) then tool-call JSON | tool-call JSON only |
| Data size | 13K (synthetic) | 8633 (v2_final, curated + expert) |
| Think ban | Required (logit_bias hack) | Not needed (Qwen3-4B no think bias) |
| Checkpoints | FSDP complete (~34 GB each) | LoRA adapter (~130 MB each) |
| Disk strategy | NVMe /home/yann/ for checkpoints | Same: NVMe /home/yann/ |
| Execution reward | Structural only (Phase A) | Full 5-signal (Phase 2 GRPO) |
| Multi-turn | Not reached (Phase C not converged) | In SFT data (1570 multi-turn entries) |
