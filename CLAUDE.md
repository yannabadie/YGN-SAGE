# YGN-SAGE — Self-Adaptive Generation Engine

Agent Development Kit with 5 cognitive pillars: Topology, Tools, Memory, Evolution, Strategy.
Rust core (sage-core) + Python SDK (sage-python) + Knowledge Pipeline (sage-discover).

## CRITICAL DIRECTIVES

1. **Rust first, Python tolerant** — performance-critical in Rust (sage-core), Python for orchestration only
2. **Zero heuristics** — all decisions formally verified (Z3), learned (ONNX), or research-backed. Never hardcode thresholds
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

## Current State (March 20, 2026)

- **Tests**: Python 1500+ passed / Rust 270+ passed / 0 failures (68 veRL-specific: env, reward, edge credit, execution)
- **BigCodeBench Hard Instruct**: 37.8% (budget model) — leaderboard SOTA stale since April 2025
- **HumanEval+ pipeline**: 89.6% (+5.5pp over pre-pipeline 84.1%)
- **Routing GT**: kNN 92%, SystemRouter 86%, heuristic 34%
- **Providers**: 8 functional (Google, OpenAI, DeepSeek, xAI, Kimi, MiniMax, OpenRouter, Codex)
- **Models**: 20 in cards.toml (minimax-m2.7, gpt-5.4-mini/nano, qwen3.5-plus via OpenRouter)
- **Self-Adaptive**: SA-1, SA-3, SA-4 + Path 6 (Phi-4-mini SFT current, Qwen3.5-9B GiGPO replacing)
- **GiGPO Training**: VeRLGIGPO branch ready for RunPod H100 (multi-step topology env, 8 providers, 1965 entries, GiGPO + Graph-GRPO edge credit)
- **PyPI**: `pip install ygn-sage` — v0.1.0-alpha
- **HuggingFace**: `yannabadie/sage-topology-policy` — SFT Phi-4-mini-instruct (to be replaced by Qwen3.5-9B after veRL training)

## Detailed rules in .claude/rules/

- `critical-directives.md` — the 5 rules above, expanded
- `environment.md` — LLM models, API keys, SSL, ExoCortex
- `architecture.md` — pillars, pipeline, competitors, benchmarks
- `development.md` — build/test/bench commands, what NOT to benchmark
- `research-decisions.md` — paper-backed decisions, DROPPED items
