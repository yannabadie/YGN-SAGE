#!/bin/bash
# ============================================================
# YGN-SAGE Full Setup — RunPod H100 (March 2026)
# ============================================================
# Installs ALL dependencies, builds Rust sage-core, downloads
# models, and verifies the complete stack.
#
# Usage:
#   bash sage-python/scripts/setup_full.sh
#
# Prerequisites:
#   - RunPod with 2x H100 NVL 94GB
#   - .env file with API keys (see RUNPOD_PLAN.md)
#   - git clone of YGN-SAGE
# ============================================================

set -euo pipefail

SAGE_DIR="${SAGE_DIR:-/workspace/YGN-SAGE}"
cd "$SAGE_DIR"

echo "============================================================"
echo "  YGN-SAGE Full Setup — $(date '+%Y-%m-%d %H:%M UTC')"
echo "============================================================"

# ── 0. Load API keys ──────────────────────────────────────
if [ -f "$SAGE_DIR/.env" ]; then
    set -a && source "$SAGE_DIR/.env" && set +a
    echo "[0] API keys loaded"
else
    echo "[0] WARNING: No .env file found. Create one with API keys."
fi

# ── 1. Python dependencies ────────────────────────────────
echo ""
echo "[1] Installing Python dependencies..."
pip install -r sage-python/requirements-runpod.txt 2>&1 | tail -3
pip install -e "sage-python/.[all]" 2>&1 | tail -3
echo "[1] Python deps: OK"

# ── 2. Rust toolchain ─────────────────────────────────────
echo ""
echo "[2] Installing Rust toolchain..."
if ! command -v rustc &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>&1 | tail -3
    source "$HOME/.cargo/env"
fi
echo "    rustc: $(rustc --version 2>/dev/null || echo 'MISSING')"

# ── 3. Build sage-core (Rust) ─────────────────────────────
echo ""
echo "[3] Building sage-core..."
source "$HOME/.cargo/env" 2>/dev/null
cd sage-core

# Try with all features first, fallback without onnx if pkg-config missing
if maturin develop --features smt,onnx,cognitive,tool-executor 2>/dev/null; then
    echo "[3] sage-core built with ALL features (smt,onnx,cognitive,tool-executor)"
elif maturin develop --features smt,tool-executor 2>/dev/null; then
    echo "[3] sage-core built (smt,tool-executor) — onnx/cognitive need pkg-config/libssl-dev"
else
    # Last resort: build wheel and install
    maturin build --release --features smt,tool-executor 2>&1 | tail -3
    pip install "$SAGE_DIR"/target/wheels/sage_core-*.whl --force-reinstall 2>&1 | tail -3
    echo "[3] sage-core installed from wheel (smt,tool-executor)"
fi
cd "$SAGE_DIR"

# ── 4. Download embedding model ───────────────────────────
echo ""
echo "[4] Downloading embedding model (Snowflake/snowflake-arctic-embed-m)..."
python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('Snowflake/snowflake-arctic-embed-m')
print(f'    Embedding model: OK (dim={m.get_sentence_embedding_dimension()})')
" 2>&1 | grep -v "^$"

# ── 5. Verify complete stack ──────────────────────────────
echo ""
echo "[5] Verifying stack..."
python3 -c "
import sys

checks = []

# Rust core
try:
    import sage_core
    exports = len([a for a in dir(sage_core) if not a.startswith('_')])
    checks.append(('sage_core', f'OK ({exports} exports)'))

    # Key Rust components
    for comp in ['TopologyGraph','TopologyExecutor','TopologyDensity','SmtVerifier','SystemRouter','ContextualBandit','TopologyEngine','RustKnnRouter']:
        if not hasattr(sage_core, comp):
            checks.append((f'  {comp}', 'MISSING (needs feature flag)'))
except ImportError as e:
    checks.append(('sage_core', f'FAIL: {e}'))

# Python modules
modules = {
    'sentence_transformers': 'Embeddings',
    'openai': 'OpenAI/DeepSeek/xAI/OpenRouter provider',
    'google.generativeai': 'Google Gemini (legacy)',
    'torch': 'PyTorch (training)',
    'transformers': 'HuggingFace Transformers',
    'peft': 'LoRA/PEFT',
    'datasets': 'HuggingFace Datasets',
    'bigcodebench': 'BigCodeBench benchmark',
    'z3': 'Z3 SMT solver',
    'yaml': 'YAML parsing',
}
for mod, desc in modules.items():
    try:
        __import__(mod)
        checks.append((desc, 'OK'))
    except ImportError:
        checks.append((desc, 'MISSING'))

# Embedder
try:
    from sage.memory.embedder import Embedder
    e = Embedder()
    checks.append(('Embedder', f'OK (backend={e._backend})'))
except Exception as ex:
    checks.append(('Embedder', f'FAIL: {ex}'))

# kNN Router
try:
    from sage.strategy.knn_router import KnnRouter
    r = KnnRouter()
    if r.is_ready:
        checks.append(('kNN Router (Rust)', f'OK ({r._rust_knn.exemplar_count()} exemplars)' if r._rust_knn else 'OK (Python fallback)'))
    else:
        checks.append(('kNN Router', 'NOT READY (no exemplars or hash embedder)'))
except Exception as ex:
    checks.append(('kNN Router', f'FAIL: {ex}'))

# API keys
import os
keys = ['DEEPSEEK_API_KEY','GOOGLE_API_KEY','OPENAI_API_KEY','GROK_API_KEY','HF_TOKEN']
present = sum(1 for k in keys if os.environ.get(k))
checks.append(('API keys', f'{present}/{len(keys)} configured'))

# Print results
print()
passed = 0
for name, status in checks:
    icon = '✓' if 'OK' in status or 'configured' in status else '✗'
    if 'OK' in status or 'configured' in status: passed += 1
    print(f'    {icon} {name}: {status}')
print(f'\n    {passed}/{len(checks)} checks passed')

if passed < len(checks) - 2:  # Allow 2 failures (optional deps)
    sys.exit(1)
"

echo ""
echo "============================================================"
echo "  Setup complete. Next steps:"
echo "    Training:  bash sage-python/scripts/verl/train_topology_targeted.sh"
echo "    Benchmark: python -m sage.bench --type masbench --limit 10"
echo "    GAIA:      python -m sage.bench --type gaia --level 1"
echo "============================================================"
