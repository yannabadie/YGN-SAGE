#!/bin/bash
# ============================================================
# YGN-SAGE veRL Training Setup for RunPod H100
# ============================================================
# Docker image: verlai/verl:vllm017.latest
# Model: Qwen/Qwen3.5-9B (9B, bf16)
# Local inference: cyankiwi/Qwen3.5-9B-AWQ-4bit (~5GB)
#
# Usage on RunPod:
#   1. Create pod with Docker image: verlai/verl:vllm017.latest
#   2. SSH into pod
#   3. git clone https://github.com/yannabadie/YGN-SAGE.git /workspace/YGN-SAGE
#   4. cd /workspace/YGN-SAGE && git checkout VeRLGIGPO
#   5. bash sage-python/scripts/verl/setup_runpod.sh
#   6. bash sage-python/scripts/verl/train_topology.sh
# ============================================================

set -euo pipefail
echo "=== YGN-SAGE veRL Setup for RunPod H100 ==="
echo ""

# ── 1. Verify environment ───────────────────────────────────
echo "[1/6] Verifying environment..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'No CUDA!'
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f'GPU: {gpu} ({vram:.0f} GB)')
assert vram >= 40, f'Need >= 40GB VRAM, got {vram:.0f}GB'
print('OK')
"

# ── 2. Install veRL (if not in Docker image) ────────────────
echo "[2/6] Checking veRL..."
python3 -c "import verl; print(f'veRL already installed')" 2>/dev/null || {
    echo "Installing veRL from source..."
    cd /workspace
    [ ! -d verl ] && git clone https://github.com/volcengine/verl.git
    cd verl && pip3 install --no-deps -e . && cd -
}

# ── 3. Install SAGE ─────────────────────────────────────────
echo "[3/6] Installing SAGE Python SDK..."
cd /workspace/YGN-SAGE
pip install -e "sage-python/.[all,dev]" -q

# ── 4. Build sage-core (Rust) ────────────────────────────────
echo "[4/6] Building sage-core..."
cd sage-core
pip install maturin -q
maturin develop --features smt,onnx,cognitive,tool-executor --release 2>&1 | tail -3
cd ..

# ── 5. Convert training data ────────────────────────────────
echo "[5/6] Converting training data to veRL format..."
cd sage-python
python scripts/verl/convert_sft_to_verl.py \
    --input data/topology_sft_v2_combined.jsonl \
    --output data/verl_topology_train.parquet

# ── 6. Final verification ───────────────────────────────────
echo "[6/6] Verification..."
python3 -c "
import torch, vllm
print(f'PyTorch: {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB)')
print(f'vLLM: {vllm.__version__}')

try:
    import verl
    print('veRL: OK')
except ImportError:
    print('veRL: MISSING — run setup again')

try:
    import sage_core
    print('sage-core: OK')
except ImportError:
    print('sage-core: NOT BUILT — check Rust/maturin')

import pandas as pd
df = pd.read_parquet('data/verl_topology_train.parquet')
print(f'Training data: {len(df)} entries')
print(f'Columns: {list(df.columns)}')
print(f'data_source: {df[\"data_source\"].unique()}')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To train:"
echo "  cd /workspace/YGN-SAGE/sage-python"
echo "  bash scripts/verl/train_topology.sh"
echo ""
echo "To monitor:"
echo "  wandb login  # optional, for W&B dashboards"
echo "  tail -f models/topology_verl/logs/*.log"
