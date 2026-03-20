#!/bin/bash
# ============================================================
# YGN-SAGE veRL + GiGPO Training Setup for RunPod H100
# ============================================================
# Usage: After cloning the repo on RunPod, run:
#   cd YGN-SAGE && bash sage-python/scripts/verl/setup_runpod.sh
#
# Prerequisites:
#   - RunPod pod with H100 80GB (or A100 80GB)
#   - CUDA 12.x + Python 3.12+
#   - Git access to yannabadie/YGN-SAGE
# ============================================================

set -euo pipefail
echo "=== YGN-SAGE veRL Setup for RunPod ==="

# ── 1. System packages ──────────────────────────────────────
echo "[1/7] System packages..."
apt-get update -qq && apt-get install -y -qq git curl wget htop nvtop 2>/dev/null || true

# ── 2. Python environment ───────────────────────────────────
echo "[2/7] Python environment..."
pip install --upgrade pip setuptools wheel

# ── 3. veRL + vLLM ──────────────────────────────────────────
echo "[3/7] Installing veRL + vLLM..."
pip install vllm>=0.8.5
pip install flash-attn --no-build-isolation --no-cache-dir 2>/dev/null || echo "flash-attn: using pre-built"

# Install veRL from source (latest)
if [ ! -d "/workspace/verl" ]; then
    git clone https://github.com/volcengine/verl.git /workspace/verl
fi
cd /workspace/verl && pip install -e ".[vllm]" && cd -

# Install verl-agent (GiGPO)
if [ ! -d "/workspace/verl-agent" ]; then
    git clone https://github.com/langfengQ/verl-agent.git /workspace/verl-agent
fi
cd /workspace/verl-agent && pip install -e . && cd -

# ── 4. SAGE dependencies ────────────────────────────────────
echo "[4/7] Installing SAGE..."
cd /workspace/YGN-SAGE
pip install -e "sage-python/.[all,dev]"

# Build sage-core (Rust)
echo "[5/7] Building sage-core..."
cd sage-core && pip install maturin && maturin develop --features smt,onnx,cognitive,tool-executor && cd ..

# ── 5. Convert training data ────────────────────────────────
echo "[6/7] Converting training data to veRL format..."
cd sage-python
python scripts/verl/convert_sft_to_verl.py \
    --input data/topology_sft_v2_combined.jsonl \
    --output data/verl_topology_train.parquet

# ── 6. Verify ───────────────────────────────────────────────
echo "[7/7] Verification..."
python -c "
import torch, vllm, verl
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0)})')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB')
print(f'vLLM: {vllm.__version__}')
print(f'veRL: OK')
try:
    import sage_core
    print(f'sage-core: OK ({sage_core.__version__})')
except:
    print('sage-core: NOT BUILT (run maturin develop in sage-core/)')
import pandas as pd
df = pd.read_parquet('data/verl_topology_train.parquet')
print(f'Training data: {len(df)} entries')
"

echo ""
echo "=== Setup complete ==="
echo "Next: Run training with:"
echo "  bash sage-python/scripts/verl/train_topology.sh"
