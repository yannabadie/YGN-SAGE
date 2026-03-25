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
```

## Current State (March 25, 2026)

- **Tests**: Python 1500+ passed / Rust 270+ passed / 0 failures (68 veRL-specific: env, reward, edge credit, execution). veRL training tests: 404 (357 Rust + 47 Python).
- **BigCodeBench Hard Instruct**: 37.8% (budget model) — leaderboard SOTA stale since April 2025
- **HumanEval+ pipeline**: 89.6% (+5.5pp over pre-pipeline 84.1%)
- **Routing GT**: kNN 92%, SystemRouter 86%, heuristic 34%
- **Providers**: 8 functional (Google, OpenAI, DeepSeek, xAI, Kimi, MiniMax, OpenRouter, Codex)
- **Models**: 20 in cards.toml (minimax-m2.7, gpt-5.4-mini/nano, qwen3.5-plus via OpenRouter)
- **Self-Adaptive**: SA-1, SA-3, SA-4 + Path 6 (Phi-4-mini SFT V1, Nemotron-Orchestrator-8B GiGPO V2)
- **GiGPO Training (RunPod H100 NVL 94GB)**:
  - SFT warmup: OK (118 steps, loss 2.87→1.30, YAML valide)
  - Phase A V3: OOM (batch_size=64, 159/167 GB RAM)
  - Phase A V4: 18/1152 steps, reward stalled at 0.02 (97% reward=0)
  - **Root causes**: max_response_length=512 (truncation), lr=5e-5 (drift), no reward shaping
  - **V5 ready**: max_response_length=1024, lr=1e-6, reward shaping (_partial_credit), ~29h H100
  - See `TRAINING_LOG.md` for full post-mortem
- **PyPI**: `pip install ygn-sage` — v0.1.0-alpha
- **HuggingFace**: `yannabadie/sage-topology-policy-v2` — Nemotron-Orchestrator-8B GiGPO (V1 legacy: `yannabadie/sage-topology-policy` Phi-4-mini SFT)

## Detailed rules in .claude/rules/

- `critical-directives.md` — the 5 rules above, expanded
- `environment.md` — LLM models, API keys, SSL, ExoCortex
- `architecture.md` — pillars, pipeline, competitors, benchmarks
- `development.md` — build/test/bench commands, what NOT to benchmark
- `research-decisions.md` — paper-backed decisions, DROPPED items
