---
paths:
  - "**/*.py"
  - "**/*.rs"
  - "**/Cargo.toml"
  - "**/pyproject.toml"
---

# Development Commands & Workflows

## Build
```bash
cd sage-core && maturin develop --features smt,onnx,cognitive,tool-executor  # Full Rust build
cd sage-python && pip install -e ".[all,dev]"                                 # Python SDK
```

## Test
```bash
cd sage-core && cargo test --no-default-features --features smt,tool-executor --lib  # Rust (259 tests)
cd sage-python && python -m pytest tests/ -v                                          # Python (1559 tests)
```

## Benchmarks — USE THESE (not HumanEval+)
```bash
# BigCodeBench (ICLR '25, non-saturated, RELEVANT)
python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 20

# Routing ground truth (50 tasks, instant)
python -m sage.bench --type routing_gt

# Ablation (framework value proof)
python -m sage.bench --type ablation --limit 50
```

## DO NOT USE for proving SAGE value
- HumanEval+ — saturated (99%+ SOTA), measures LLM not framework
- MBPP+ — same issue
- GSM8K — model ceiling, topology has no effect

## Z3 Quality Labels (for training DistilBERT)
```bash
python scripts/collect_quality_labels.py --dataset bigcodebench --subset hard --limit 50
```

## Path 6 (Learned Topology Policy)
```bash
# Enable Path 6 in pipeline (loads 3.8B model on GPU)
export SAGE_ENABLE_PATH6=1
python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 20

# Train topology policy
python scripts/train_topology_grpo.py --mode sft --data data/topology_sft_clean.jsonl --epochs 5
python scripts/upload_hf.py  # publish to yannabadie/sage-topology-policy

# Generate SFT data (requires OPENAI_API_KEY for GPT-5.4)
python scripts/generate_topology_sft.py --dataset bigcodebench --limit 100 --model gpt-5.4
```

## Lint
```bash
cd sage-python && ruff check src/ && mypy src/ --ignore-missing-imports
cd sage-core && cargo clippy --no-default-features
```
