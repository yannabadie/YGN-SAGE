#!/bin/bash
# ============================================================
# YGN-SAGE veRL Training Setup for RunPod H100
# ============================================================
# Docker: verlai/verl:base-v4-cu126-cudnn9.8-torch2.7.1-fa2.8.0-te2.3
# Model: Qwen/Qwen3.5-9B (dense 9B, bf16, Apache 2.0)
# Fallback: Qwen/Qwen2.5-7B-Instruct
#
# Usage on RunPod:
#   1. Create pod with Docker image (see RUNPOD_PLAN.md)
#   2. SSH into pod
#   3. git clone https://github.com/yannabadie/YGN-SAGE.git /workspace/YGN-SAGE
#   4. cd /workspace/YGN-SAGE && git checkout VeRLGIGPO
#   5. bash sage-python/scripts/verl/setup_runpod.sh
#   6. bash sage-python/scripts/verl/train_topology.sh
# ============================================================

set -euo pipefail
echo "=== YGN-SAGE veRL Setup for RunPod H100 ==="
echo ""

# ── 1. Verify GPU ────────────────────────────────────────────
echo "[1/8] Verifying GPU..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'No CUDA!'
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f'GPU: {gpu} ({vram:.0f} GB)')
assert vram >= 40, f'Need >= 40GB VRAM, got {vram:.0f}GB'
print('OK')
"

# ── 2. Install vLLM (>= 0.17 for Qwen3.5-9B) ────────────────
echo "[2/8] Installing vLLM..."
python3 -c "import vllm; v=vllm.__version__; print(f'vLLM {v} already installed')" 2>/dev/null && {
    # Check version is sufficient
    python3 -c "
import vllm
v = tuple(int(x) for x in vllm.__version__.split('.')[:2])
if v < (0, 17):
    print(f'vLLM {vllm.__version__} too old, upgrading...')
    import sys; sys.exit(1)
else:
    print(f'vLLM {vllm.__version__} OK (>= 0.17)')
" 2>/dev/null
} || {
    echo "Installing vLLM >= 0.17..."
    pip install vllm -q 2>&1 | tail -3
}

# ── 3. Install veRL from source ───────────────────────────────
echo "[3/8] Installing veRL..."
python3 -c "import verl; print('veRL already installed')" 2>/dev/null || {
    echo "Installing veRL from source..."
    cd /workspace
    [ ! -d verl ] && git clone https://github.com/volcengine/verl.git
    cd verl && pip install -e ".[vllm]" -q 2>&1 | tail -3
    cd /workspace/YGN-SAGE
}

# ── 4. Install SAGE Python SDK ────────────────────────────────
echo "[4/8] Installing SAGE Python SDK..."
cd /workspace/YGN-SAGE
pip install -e "sage-python/.[all,dev]" -q 2>&1 | tail -3

# ── 5. Build sage-core (Rust) ─────────────────────────────────
echo "[5/8] Building sage-core (Rust)..."
# Install Rust if not present
command -v cargo >/dev/null 2>&1 || {
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
}
cd sage-core
pip install maturin -q
maturin develop --features smt,onnx,cognitive,tool-executor --release 2>&1 | tail -3
cd /workspace/YGN-SAGE

# ── 6. Download model ─────────────────────────────────────────
echo "[6/8] Checking model availability..."
MODEL=${SAGE_MODEL:-"Qwen/Qwen3.5-9B"}
python3 -c "
from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained('$MODEL', trust_remote_code=True)
    arch = cfg.architectures[0] if hasattr(cfg, 'architectures') else 'unknown'
    print(f'Model: $MODEL ({arch}) — OK')
except Exception as e:
    print(f'Model $MODEL not available: {e}')
    print('Will be downloaded at training start by vLLM')
"

# ── 7. Convert training data ──────────────────────────────────
echo "[7/8] Converting training data to veRL format..."
cd sage-python
python3 scripts/verl/convert_sft_to_verl.py \
    --input data/topology_sft_v2_combined.jsonl \
    --output data/verl_topology_train.parquet

# ── 8. Final verification ─────────────────────────────────────
echo "[8/8] Verification..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB)')

try:
    import vllm
    print(f'vLLM: {vllm.__version__}')
except ImportError:
    print('vLLM: MISSING')

try:
    import verl
    print('veRL: OK')
except ImportError:
    print('veRL: MISSING')

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
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To train:"
echo "  cd /workspace/YGN-SAGE/sage-python"
echo "  bash scripts/verl/train_topology.sh"
echo ""
echo "To validate (optional):"
echo "  python3 scripts/verl/validate_setup.py"
