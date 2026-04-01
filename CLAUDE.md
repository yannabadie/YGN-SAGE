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
```

## Current State (April 1, 2026)

- **Tests**: Python **1951 passed** / Rust **403 passed** / 0 failures
- **MASBENCH 5-axis**: breadth=72%, depth=54%, horizon=16%, parallel=48%, robustness=0% (SAGE vs bare +27pp avg)
- **Rust stack**: 18/18 components operational (54 exports), kNN 92% routing NOW active in pipeline
- **Templates**: **11** (was 8) — added robust, horizon_pipeline, parallel_fanout for MASBENCH axes
- **DAG-driven topology selection**: `select_macro_topology(omega, delta, gamma)` picks template from structural features
- **Adaptive context**: predecessor outputs sized to model context window (was hardcoded 1000 chars)
- **Similarity gate**: S2-MAD Jaccard dedup before inter-node communication (-94% tokens on parallel)
- **Multi-turn debate**: `reset_node()` + `open_gate` action for iterative refinement (max 3 rounds)
- **Bandit → Assigner**: quality posteriors override underperforming model assignments (quality < 0.4)
- **Per-node streaming**: `run_stream()` async generator yields events as nodes complete
- **HITL callback**: `approval_callback` pauses execution at disruptive controller decisions
- **Arithmetic verification**: Z3-free equation checking in TopologyController (depth tasks)
- **13 infrastructure fixes** applied March 29-30 — see TRAINING_LOG.md for full chain
- **BigCodeBench Hard Instruct**: 37.8% (budget model) — leaderboard SOTA stale since April 2025
- **HumanEval+ pipeline**: 89.6% (+5.5pp over pre-pipeline 84.1%)
- **Routing GT**: kNN 92%, SystemRouter 86%, heuristic 34%
- **Providers**: 7 providers (Google, OpenAI, DeepSeek, xAI, Kimi, MiniMax, OpenRouter). Models updated March 29: gpt-5.4, gemini-3.1-pro-preview, deepseek-chat
- **Models**: 20 in cards.toml. Tiers: fast=gemini-3.1-flash-lite, budget=deepseek-chat, reasoner=gemini-3.1-pro-preview, codex=gpt-5.4
- **Self-Adaptive**: SA-1, SA-3, SA-4 + Path 6 (V2: Nemotron-Orchestrator-8B, DAPO training in progress)
- **Training (RunPod 2x H100 NVL 94GB)**:
  - SFT warmup: OK (118 steps, loss 2.87→1.30)
  - Phase A: Done (step 1050, reward 0.225, structural ceiling)
  - **DAPO targeted training**: IN PROGRESS (step 104/1920, reward 0.184, DAPO token-level loss)
  - Phase C: Scripts ready, pending Phase A+B convergence
  - **Key lessons**: never merge LoRA during training, save FSDP complete, DAPO > GRPO, MASBENCH validates topology
  - See `TRAINING_LOG.md` for full history, `RUNPOD_PLAN.md` for updated plan
- **Pipeline fixes (March 29-30)**:
  - DeepSeek fallback (was sending wrong models to Gemini → 404)
  - Per-node timeout 60s (was no timeout → 274s per task)
  - Models updated: gpt-4.1 → gpt-5.4, embedder model name fixed
  - Result: +27pp on MASBENCH depth after fixes
- **PyPI**: `pip install ygn-sage` — v0.1.0-alpha
- **HuggingFace**: `yannabadie/sage-topology-policy-v2` — FSDP checkpoint (34GB) + LoRA + SFT merged (16GB)
- **Research (March 29)**: DAPO, MAS-Orchestra, EvoMAS, GoAgent, Graph-GRPO analyzed. See TRAINING_LOG.md
- **Memory-Evolution (March 30)**: 5 phases merged:
  - Causal memory wired into agent loop (AMA-Bench 2602.22769: memory fails without causality)
  - Episodic→Semantic→Causal consolidation pipeline (MAGMA 2601.03236: +45.5% reasoning)
  - Composite 5-signal write gate (arXiv 2603.15994: 100% vs 13% accuracy) — Rust `RustCompositeWriteGate` + Python fallback
  - Online evolution: `should_evolve()` in Rust + `evolve()` wired in agent loop (SA-3 complete)
  - 12-dimension Agent Stability Index (arXiv 2601.04170) + BehaviorTracker + AdaptiveMutator (ShinkaEvolve)
  - 81 new tests, 0 regressions
- **Runtime Pipeline Hardening (March 31)**: 7-axis improvement — see `docs/superpowers/specs/2026-03-31-runtime-pipeline-hardening-design.md`
  - Fixed `_log` NameError in TopologyRunner code node execution (3 lines)
  - Cross-platform `sys.executable` fix (8 files, was `"python3"` hardcoded — broken on Windows)
  - Memory consolidation + causal wiring + persistence now active in pipeline path (was legacy-only)
  - FrugalGPT cascade now actually upgrades models (exclude_ids + budget escalation, Rust + Python)
  - Bandit + MAP-Elites state periodic persistence every 10 tasks (was atexit-only)
  - OxiZ SAT verification warning flag in PipelineContext for Stage 4 observability
  - **ToolForge**: autonomous tool synthesis — GapDetector + BuildLoop + DualGate (UCT/SMITH pattern, 12 papers)
  - 29 new tests (100 total with existing 71), 0 regressions

## Detailed rules in .claude/rules/

- `critical-directives.md` — the 5 rules above, expanded
- `environment.md` — LLM models, API keys, SSL, ExoCortex
- `architecture.md` — pillars, pipeline, competitors, benchmarks
- `development.md` — build/test/bench commands, what NOT to benchmark
- `research-decisions.md` — paper-backed decisions, DROPPED items
