#!/bin/bash
# ============================================================
# YGN-SAGE Nemotron E2E Training Pipeline
# ============================================================
# THE reference command for the primary training path.
#
# Voie primaire: HF/PEFT/veRL (validated by NVIDIA ToolOrchestra)
# Model: nvidia/Nemotron-Orchestrator-8B (Qwen3 arch, GRPO-trained)
#
# Stages:
#   1. SFT warmup — learn YAML topology format (LoRA)
#   2. Merge SFT — merge LoRA into base model
#   3. GRPO warmup — structural reward, single-turn (Phase A)
#   4. Phase C — multi-step GiGPO with checkpoints (optional)
#   5. Export — merge + push HuggingFace + GGUF quantize
#
# Usage:
#   # Full pipeline on RunPod H100:
#   bash scripts/verl/train_nemotron_e2e.sh
#
#   # Smoke test (no GPU needed, <2min):
#   bash scripts/verl/train_nemotron_e2e.sh --smoke
#
#   # Skip completed stages:
#   bash scripts/verl/train_nemotron_e2e.sh --skip-sft --skip-grpo
#
# Flags:
#   --smoke         Run all stages in smoke mode (2 steps, tiny batch)
#   --skip-sft      Skip SFT warmup (use existing /workspace/sft_merged_model)
#   --skip-grpo     Skip GRPO warmup (use existing checkpoint)
#   --skip-phase-c  Skip Phase C multi-step
#   --skip-export   Skip export/push
# ============================================================

set -euo pipefail

# Parse flags
SMOKE=false
SKIP_SFT=false
SKIP_GRPO=false
SKIP_PHASE_C=false
SKIP_EXPORT=false

for arg in "$@"; do
    case $arg in
        --smoke)       SMOKE=true ;;
        --skip-sft)    SKIP_SFT=true ;;
        --skip-grpo)   SKIP_GRPO=true ;;
        --skip-phase-c) SKIP_PHASE_C=true ;;
        --skip-export) SKIP_EXPORT=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# Paths
MODEL=${SAGE_MODEL:-"/workspace/patched_nemotron_orchestrator"}
SFT_OUTPUT="/workspace/sft_warmup_output"
SFT_MERGED="/workspace/sft_merged_model"
GRPO_OUTPUT="/workspace/topology_verl_output"
PHASE_C_OUTPUT="/workspace/topology_verl_phase_c"
EXPORT_OUTPUT="models/topology_verl_merged"

cd "$(dirname "$0")/../.."  # sage-python/

echo "============================================================"
echo "YGN-SAGE Nemotron E2E Training Pipeline"
echo "============================================================"
echo "Model:     $MODEL"
echo "Smoke:     $SMOKE"
echo "Skip SFT:  $SKIP_SFT"
echo "Skip GRPO: $SKIP_GRPO"
echo "Skip Phase C: $SKIP_PHASE_C"
echo "Skip Export: $SKIP_EXPORT"
echo ""

# ── Stage 1: SFT Warmup ──────────────────────────────────────
if [ "$SKIP_SFT" = false ]; then
    echo "=== Stage 1: SFT Warmup ==="
    SMOKE_FLAG=""
    if [ "$SMOKE" = true ]; then
        SMOKE_FLAG="--smoke"
    fi
    python3 scripts/verl/sft_warmup.py $SMOKE_FLAG --output "$SFT_OUTPUT"
    echo "SFT output: $SFT_OUTPUT"

    # Merge LoRA into base
    if [ "$SMOKE" = false ]; then
        echo "Merging SFT LoRA into base model..."
        python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained('$MODEL', torch_dtype=torch.bfloat16, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained('$SFT_OUTPUT', trust_remote_code=True)
model = PeftModel.from_pretrained(base, '$SFT_OUTPUT')
merged = model.merge_and_unload()
merged.save_pretrained('$SFT_MERGED')
tok.save_pretrained('$SFT_MERGED')
print('Merged model saved to $SFT_MERGED')
"
    else
        echo "Smoke mode: skipping merge (SFT adapter only)"
        SFT_MERGED="$SFT_OUTPUT"
    fi
else
    echo "=== Stage 1: SKIPPED (--skip-sft) ==="
fi

# ── Stage 2: GRPO Warmup (Phase A) ───────────────────────────
if [ "$SKIP_GRPO" = false ]; then
    echo ""
    echo "=== Stage 2: GRPO Warmup (Phase A, single-turn) ==="
    if [ "$SMOKE" = true ]; then
        echo "Smoke mode: skipping GRPO (requires verl + vLLM)"
        echo "In full mode, run: bash scripts/verl/train_topology_v5.sh"
    else
        bash scripts/verl/train_topology_v5.sh
    fi
    echo "GRPO output: $GRPO_OUTPUT"
else
    echo "=== Stage 2: SKIPPED (--skip-grpo) ==="
fi

# ── Stage 3: Phase C Multi-Step (GiGPO) ──────────────────────
if [ "$SKIP_PHASE_C" = false ]; then
    echo ""
    echo "=== Stage 3: Phase C Multi-Step (GiGPO custom) ==="
    PHASE_C_ARGS="--output $PHASE_C_OUTPUT"
    if [ "$SMOKE" = true ]; then
        PHASE_C_ARGS="$PHASE_C_ARGS --smoke"
    fi
    if [ -d "$GRPO_OUTPUT" ]; then
        PHASE_C_ARGS="$PHASE_C_ARGS --checkpoint $GRPO_OUTPUT"
    fi
    python3 scripts/verl/train_phase_c_custom.py \
        --model "$MODEL" \
        --data data/verl_topology_phase_c.parquet \
        $PHASE_C_ARGS
    echo "Phase C output: $PHASE_C_OUTPUT"
else
    echo "=== Stage 3: SKIPPED (--skip-phase-c) ==="
fi

# ── Stage 4: Export ───────────────────────────────────────────
if [ "$SKIP_EXPORT" = false ] && [ "$SMOKE" = false ]; then
    echo ""
    echo "=== Stage 4: Export (merge + push + quantize) ==="
    python3 scripts/verl/post_training_pipeline.py all
    echo "Export output: $EXPORT_OUTPUT"
else
    echo "=== Stage 4: SKIPPED ==="
fi

echo ""
echo "============================================================"
echo "E2E Pipeline Complete"
echo "============================================================"
echo "Next: Enable Path 6 in runtime with SAGE_ENABLE_PATH6=1"
