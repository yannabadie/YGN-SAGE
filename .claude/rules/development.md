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
cd sage-core && cargo test --no-default-features --features smt,tool-executor --lib  # Rust (270+ tests)
cd sage-python && python -m pytest tests/ -v                                          # Python (1500+ tests, 68 veRL-specific)
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

## Benchmark Monitoring Protocol

Every bench run MUST produce full observability. No blind runs.

### Before launch
```bash
# Environment: disable dead providers, offline HF, unbuffered output
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MINIMAX_API_KEY="" KIMI_API_KEY="" PYTHONUNBUFFERED=1
# Load keys selectively (not via source .env which leaks dead providers)
export GOOGLE_API_KEY=$(grep GOOGLE_API_KEY .env | cut -d'"' -f2)
export DEEPSEEK_API_KEY=$(grep DEEPSEEK_API_KEY .env | cut -d'"' -f2)
export OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d'"' -f2)
```

### During run — monitor these signals
```bash
# Task progress
grep -cE "^\[.*\]" output.log    # tasks completed
grep -c "PASS" output.log        # tasks passed

# Adaptive bypass
grep -c "BYPASS topology" output.log   # tasks bypassed (single-agent)
grep -c "Assigned models" output.log   # tasks with topology

# Repair / Escalation
grep -c "Repair succeeded" output.log         # reasoner tier repairs
grep -c "Topology escalation" output.log      # bypass→topology escalation

# Provider errors (MUST be 0 after fixes)
grep -c "Error code:" output.log              # API errors
grep "Error code:" output.log | sort | uniq -c | sort -rn  # error breakdown

# Provider reassignment
grep -c "reassigned" output.log   # dead provider → default model
grep -c "FrugalGPT" output.log   # quality cascade triggers
```

### After run — required artifacts
1. **Report JSON**: `docs/benchmarks/{date}-{bench}.json` — pass_rate, routing_breakdown, per_task results
2. **Predictions JSONL**: `docs/benchmarks/{date}-predictions-{bench}.jsonl` — solution + _trace per task
3. **ExecutionTrace** (when wired): per-task structured trace with tokens/cost/latency per node

### Post-run analysis — ALWAYS do before next loop
```python
# Error categorization
errors = [r for r in report['results'] if r.get('error')]
# Pass rate by mode (bypassed vs topology)
# Gained/Lost vs previous run (git show {prev_sha}:report.json)
# Provider error count (MUST decrease each iteration)
# Repair success rate
# Token/cost per task (when ExecutionTrace wired)
```

### What each bench tests (choose accordingly)
| Benchmark | Tests pillars | omega | Proves thesis? |
|-----------|--------------|-------|----------------|
| BigCodeBench Hard | Strategy, repair | 1.3 | NO (atomic tasks) |
| MASBENCH breadth | Topology, decompose | high | YES (p=0.015) |
| SWE-bench | ALL 5 pillars | ~3.4 | YES (target: OpenSAGE 59%) |
| routing_gt | Strategy only | N/A | Routing accuracy |
| ablation | Framework delta | varies | YES (full vs baseline) |

## Z3 Quality Labels (for training DistilBERT)
```bash
python scripts/collect_quality_labels.py --dataset bigcodebench --subset hard --limit 50
```

## Path 6 (Learned Topology Policy)
```bash
# Enable Path 6 in pipeline (loads 3.8B model on GPU)
export SAGE_ENABLE_PATH6=1
python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 20

# Train topology policy (on RunPod H100 — see RUNPOD_PLAN.md)
cd sage-python && bash scripts/verl/train_topology.sh  # GiGPO via verl-agent

# Generate SFT data (requires OPENAI_API_KEY for GPT-5.4)
python scripts/generate_topology_sft.py --dataset bigcodebench --limit 100 --model gpt-5.4
```

## Lint
```bash
cd sage-python && ruff check src/ && mypy src/ --ignore-missing-imports
cd sage-core && cargo clippy --no-default-features
```
