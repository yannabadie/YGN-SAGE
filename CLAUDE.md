# YGN-SAGE — Self-Adaptive Generation Engine

Agent Development Kit with 5 cognitive pillars: Topology, Tools, Memory, Evolution, Strategy.
Rust core (sage-core) + Python SDK (sage-python) + Knowledge Pipeline (sage-discover).

## CRITICAL DIRECTIVES

1. **Rust first, Python tolerant** — performance-critical in Rust (sage-core), Python for orchestration only
2. **Minimal heuristics** — routing learned (kNN 92%, bandit Thompson), verification formal (Z3/OxiZ). Adaptation thresholds (THETA_GOOD=0.7, THETA_CRITICAL=0.3, etc.) are calibrated initial values subject to ablation. Safety limits (MAX_RETRIES, MAX_REROUTES, cache bounds) are engineering guards. Replace heuristics with learned alternatives when data permits
3. **No corporate proxy** — this machine has NO proxy. Never add `verify=False`
4. **kNN is primary router** (92% GT) — ComplexityRouter heuristic is DEAD CODE (34% GT)
5. **Evidence before assertions** — run tests + benchmarks before claiming completion
6. **SOTA minimum, AI breakthrough at least** — don't settle for "good enough"

## Architecture (see .claude/rules/architecture.md for details)

```
sage-core/   — Rust (PyO3): TopologyEngine, SystemRouter, ModelAssigner, QualityLabeler, S-MMU, SmtVerifier
sage-python/ — Python SDK: Pipeline (5-stage), AgentLoop, Providers (7), Bench (BigCodeBench, EvalPlus)
sage-discover/ — Knowledge Pipeline (arXiv → ExoCortex)
```

## Pipeline: CLASSIFY → DECOMPOSE → TOPOLOGY → ASSIGN MODELS → EXECUTE → LEARN

## Quick Commands

```bash
# Build
cd sage-core && maturin develop --features smt,onnx,cognitive,tool-executor
cd sage-python && pip install -e ".[all,dev]"

# Test
cd sage-core && cargo test --no-default-features --features smt,tool-executor --lib
cd sage-python && python -m pytest tests/ -v

# Benchmark (USE BigCodeBench, NOT HumanEval+)
python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 20
python -m sage.bench --type routing_gt
python -m sage.bench --type ablation --limit 50

# Training (Nemotron E2E — THE reference command)
pip install -e ".[training]"
bash scripts/verl/train_nemotron_e2e.sh --smoke    # Plumbing test (CPU, <2min)
bash scripts/verl/train_nemotron_e2e.sh             # Full (RunPod H100, ~30h)

# Meta-Harness (harness optimization — arXiv 2603.28052)
# All commands from C:\Code\YGN-SAGE root. Workspace defaults to ~/.sage-meta-harness/
# Use -w .sage-meta-harness to keep workspace inside the project
python -m sage.meta_harness init                    # Create workspace
python -m sage.meta_harness evaluate baseline        # Establish baseline scores
python -m sage.meta_harness propose                  # Generate candidate template
python -m sage.meta_harness evaluate <id>            # Score a candidate
python -m sage.meta_harness status                   # Leaderboard
python -m sage.meta_harness apply                    # Apply best to production
python -m sage.meta_harness.auto_propose -n 10       # Automated search (10 iterations)
```

## Current State (April 1, 2026)

- **Tests**: Python **1951 passed** / Rust **403 passed** / 0 failures
- **Templates**: 11 (sequential, parallel, AVR, selfmoa, hierarchical, hub, debate, brainstorming, robust, horizon_pipeline, parallel_fanout)
- **Routing**: kNN 92% GT, SystemRouter 86%, heuristic 34% (dead code)
- **Providers**: 7 (Google, OpenAI, DeepSeek, xAI, Kimi, MiniMax, OpenRouter), 20 models in cards.toml
- **Benchmarks**: BigCodeBench Hard 37.8%, HumanEval+ 89.6%, MASBENCH +27pp over bare
- **Training data**: `yannabadie/sage-training-data` on HuggingFace
- **Trained models**: `yannabadie/sage-topology-policy-v2` on HuggingFace
- **PyPI**: `pip install ygn-sage` — v0.1.0-alpha

## Detailed rules in .claude/rules/

- `critical-directives.md` — the 5 rules above, expanded
- `environment.md` — LLM models, API keys, SSL, ExoCortex
- `architecture.md` — pillars, pipeline, competitors, benchmarks
- `development.md` — build/test/bench commands, what NOT to benchmark
- `research-decisions.md` — paper-backed decisions, DROPPED items
- `meta-harness.md` — Meta-Harness harness optimization rules (arXiv 2603.28052)
