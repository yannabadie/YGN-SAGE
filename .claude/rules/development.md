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
# Load ALL API keys — health check at boot excludes dead providers automatically
set -a && source .env && set +a
# Offline HF (datasets cached), unbuffered output for live monitoring
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
```
DO NOT manually exclude providers via empty env vars. The boot health check
(`ProviderPool.health_check()`) probes every provider and the Rust ModelAssigner
(`exclude_providers()`) removes dead ones from the scoring loop. **Exclusion is
time-bounded** (Apr 18, DEFAULT_EXCLUSION_TTL_SEC=300) — `SWEBenchBench` calls
`ProviderPool.refresh_exclusion_list(assigner)` at batch start so recovered
providers come back automatically. Trust the system.

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

## Path 6 (Learned Topology Policy) — OFF BY DEFAULT
```bash
# Enable Path 6 only if a local checkpoint is set up (3.8B model on GPU)
export SAGE_ENABLE_PATH6=1
python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 20
```

Training code (verl/, scripts/, data/, models/) lives on a dedicated training branch — removed from `main` on 2026-04-15 (commit `b2f59ee`, -4.3 GB). Z3 quality labels, SFT generation and GRPO training live there. Checkpoints remain on HuggingFace (`yannabadie/sage-topology-policy-local`, `yannabadie/sage-topology-policy-v2`).

## Lint
```bash
cd sage-python && ruff check src/ && mypy src/ --ignore-missing-imports
cd sage-core && cargo clippy --no-default-features
```
