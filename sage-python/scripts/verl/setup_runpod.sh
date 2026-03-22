#!/bin/bash
# ============================================================
# YGN-SAGE veRL Training Setup for RunPod H100
# ============================================================
# Docker: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Model: Qwen/Qwen3.5-9B (dense 9B, bf16, Apache 2.0)
#
# Usage on RunPod:
#   1. Create pod with RunPod PyTorch 2.4.0 template
#   2. SSH into pod
#   3. git clone https://github.com/yannabadie/YGN-SAGE.git /workspace/YGN-SAGE
#   4. cd /workspace/YGN-SAGE && git checkout VeRLGIGPO
#   5. bash sage-python/scripts/verl/setup_runpod.sh
#   6. bash sage-python/scripts/verl/train_topology.sh
# ============================================================

set -euo pipefail
echo "=== YGN-SAGE veRL Setup for RunPod H100 ==="
echo ""

# Load .env if present (API keys)
if [ -f "/workspace/YGN-SAGE/.env" ]; then
    set -a && source /workspace/YGN-SAGE/.env && set +a
    echo "Loaded API keys from .env"
elif [ -f "../.env" ]; then
    set -a && source ../.env && set +a
    echo "Loaded API keys from ../.env"
fi

# ── 1. Verify GPU ────────────────────────────────────────────
echo "[1/9] Verifying GPU..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'No CUDA!'
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'GPU: {gpu} ({vram:.0f} GB)')
assert vram >= 40, f'Need >= 40GB VRAM, got {vram:.0f}GB'
print('OK')
"

# ── 2. Verify/install vLLM ────────────────────────────────────
echo "[2/9] Checking vLLM..."
python3 -c "
import vllm
print(f'vLLM {vllm.__version__}')
print('OK')
" || {
    echo "vLLM missing. Installing..."
    pip install vllm -q 2>&1 | tail -3
}

# ── 3. Install flash-linear-attention + causal-conv1d ─────────
# CRITICAL: Qwen3.5 uses Gated DeltaNet layers. Without these,
# vLLM falls back to slow PyTorch implementation (3-5x slower).
echo "[3/9] Installing flash-linear-attention (for Qwen3.5 GDN layers)..."
python3 -c "
try:
    import fla
    print(f'flash-linear-attention already installed')
except ImportError:
    import sys; sys.exit(1)
" 2>/dev/null || {
    pip install flash-linear-attention -q 2>&1 | tail -3
    echo "flash-linear-attention installed"
}

python3 -c "
try:
    import causal_conv1d
    print(f'causal-conv1d already installed')
except ImportError:
    import sys; sys.exit(1)
" 2>/dev/null || {
    pip install causal-conv1d -q 2>&1 | tail -3
    echo "causal-conv1d installed"
}

# ── 4. Install verl-agent ─────────────────────────────────────
echo "[4/9] Installing verl-agent..."
python3 -c "
try:
    import verl
    print('verl already installed')
except ImportError:
    import sys; sys.exit(1)
" 2>/dev/null || {
    echo "Installing verl-agent from source..."
    cd /workspace
    [ ! -d verl-agent ] && git clone https://github.com/langfengQ/verl-agent.git
    cd verl-agent && pip3 install -e . -q 2>&1 | tail -5
    cd /workspace/YGN-SAGE
}

# ── 5. Install SAGE Python SDK ────────────────────────────────
echo "[5/9] Installing SAGE Python SDK..."
cd /workspace/YGN-SAGE
pip install -e "sage-python/.[all,dev]" -q 2>&1 | tail -3

# ── 6. Build sage-core (Rust) ─────────────────────────────────
echo "[6/9] Building sage-core (Rust)..."
command -v cargo >/dev/null 2>&1 || {
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
}
cd sage-core
pip install maturin -q
maturin develop --features smt,onnx,cognitive,tool-executor --release 2>&1 | tail -3
cd /workspace/YGN-SAGE

# ── 7. Download model + patch tokenizer ───────────────────────
echo "[7/9] Downloading model and patching tokenizer..."
MODEL=${SAGE_MODEL:-"Qwen/Qwen3.5-9B"}
PATCHED_DIR="/workspace/patched_model"

# Download model weights to HF cache
python3 -c "
from huggingface_hub import snapshot_download
print(f'Downloading {\"$MODEL\"}...')
path = snapshot_download('$MODEL')
print(f'Model cached at: {path}')
"

# Patch tokenizer (removes thinking mode for vLLM rollouts)
cd sage-python
python3 scripts/verl/patch_tokenizer.py --model "$MODEL" --output "$PATCHED_DIR"
bash "$PATCHED_DIR/link_weights.sh"
cd /workspace/YGN-SAGE

echo "Patched model ready at $PATCHED_DIR"

# ── 8. Convert training data ──────────────────────────────────
echo "[8/9] Converting training data to veRL format..."
cd sage-python
python3 scripts/verl/convert_sft_to_verl.py \
    --input data/topology_sft_v2_combined.jsonl \
    --output data/verl_topology_train.parquet
cd /workspace/YGN-SAGE

# ── 9. Final verification ─────────────────────────────────────
echo "[9/9] Verification..."
cd sage-python
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB)')

try:
    import vllm
    print(f'vLLM: {vllm.__version__}')
except ImportError:
    print('vLLM: MISSING !!!')

try:
    import verl
    print('veRL: OK')
except ImportError:
    print('veRL: MISSING !!!')

try:
    import fla
    print('flash-linear-attention: OK')
except ImportError:
    print('flash-linear-attention: MISSING (Qwen3.5 will be SLOW)')

try:
    import sage_core
    print('sage-core: OK')
except ImportError:
    print('sage-core: NOT BUILT')

from sage.verl.reward import compute_score
score = compute_score('t', 'nodes:\n- role: coder', '', {})
print(f'Reward function: {score:.3f}')

import pandas as pd
df = pd.read_parquet('data/verl_topology_train.parquet')
print(f'Training data: {len(df)} entries')

# Verify patched tokenizer
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/workspace/patched_model')
test = tok.apply_chat_template(
    [{'role': 'user', 'content': 'test'}],
    tokenize=False, add_generation_prompt=True
)
assert '<think>' not in test, 'TOKENIZER NOT PATCHED!'
print('Patched tokenizer: OK (no thinking mode)')
"
cd /workspace/YGN-SAGE

echo ""
echo "=== Setup complete ==="
echo ""
echo "To train:"
echo "  cd /workspace/YGN-SAGE/sage-python"
echo "  bash scripts/verl/train_topology.sh"
echo ""
echo "To validate (optional):"
echo "  python3 scripts/verl/validate_setup.py"
