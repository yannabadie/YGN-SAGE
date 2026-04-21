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
7. **No training-leak model hardcodes** — truth for OpenAI/Gemini/xAI/DeepSeek/Anthropic models in this repo is `sage-core/config/cards.toml`, NOT the agent's training snapshot. Before adding a `"<tag>" in model` check or a quirk branch, verify the tag hits at least one id in cards.toml AND verify the quirk itself via Context7 `/berriai/litellm` or the provider's live docs — cite the source in the code comment. See `docs/patterns/knowledge-cutoff-checks.md`. *2026-04 incident*: hardcoded `o1/o3/o4` for a temperature clamp even though cards.toml only ships `gpt-5.x`.

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

# Training — PARKED on main since 2026-04-15 (commit b2f59ee, -4.3GB)
# verl/, scripts/, data/, models/ and training tests live on a dedicated training branch.
# Trained checkpoints still on HF: yannabadie/sage-topology-policy-local (Phase C, 40% MASBENCH).
# Set SAGE_ENABLE_PATH6=1 to load a local checkpoint at inference time.

# Meta-Harness (harness optimization — arXiv 2603.28052)
# Uses the official framework from stanford-iris-lab/meta-harness
# (cloned to external/meta-harness/). Our in-tree implementation was
# removed on 2026-04-18 — it was a dataclass hyperparameter tuner, not
# the structural-evolution harness search the paper specifies. See ADR-010.
#
# Workflow (for a SAGE reference_example under external/meta-harness/):
cd external/meta-harness/reference_examples/ygn_sage
uv sync
uv run python meta_harness.py --iterations 10 --fresh
```

## Current State (April 20, 2026)

- **Tests**: Python **1958 passed** (49 skipped, 11 pre-existing failures + 5 errors in API-key-dependent test files — `test_e2e_live_providers.py`, `test_provider_pool_wiring.py`, `test_e2e_campaign.py`, `test_pydantic_ai_integration.py` flake; NONE in scope of phase-1 stab) / Rust **480 passed** (+2 net 2026-04-20 phase-1-stab: +8 Task A getters/gate/seed, +1 Task B record_abstain, -7 Task C path-6 regex deletions)
- **Templates**: 12 (sequential, parallel, AVR, selfmoa, hierarchical, hub, debate, brainstorming, robust, horizon_pipeline, parallel_fanout, formal_solver)
- **Routing**: kNN 100% GT (CORAL exact-match override), Rust SystemRouter 88%, heuristic 34% (dead code)
- **Providers**: 7 (Google, OpenAI, DeepSeek, xAI, Kimi, MiniMax, OpenRouter), 20 models in cards.toml. **TTL'd exclusion** (300s re-probe, Apr 18 3148667) — not permanent.
- **Benchmarks**: BigCodeBench Hard **45.9%** / **SWE-bench Lite Docker-graded 10% (v15 1/10 resolved 2026-04-21)** — first real pass-rate after Windows infra fixes (CRLF + UTF-8 in `sage.bench.swebench_ca_patch`). Patch-generation rate **70% average (v5d 4/5, v5e 3/5)** from Apr 18 plumbing fixes is the "how many patches were produced" number, NOT the "how many passed" number. See `docs/benchmarks/2026-04-21-swebench-v15-eval-results.md`.
- **Architecture**: Unified entry point Phases 1-3 MERGED. `system.run()` → `pipeline.run()` single path; topology nodes = real agents via factory.
- **Plumbing Apr 18 (13 commits)**: Revert tool_choice=required ; bench real/sentinel/empty classifier ; sentinel cascade strip ; planner-injection opt-in ; `--offset` CLI ; **telemetry wire-up** (tool_call_count was dead counter) ; **per-model routing** (config.model was ignored) ; quota-aware health_check ; TTL exclusion+reprobe ; provider inference by model_id. See [[ADR-009-Telemetry-And-Routing-Plumbing]] in Obsidian vault.
- **Training**: ⏸ PARKED on main (2026-04-15, b2f59ee). Code on dedicated branch, checkpoints on HF.
- **Trained models**: `yannabadie/sage-topology-policy-local` (Phase C, best), `yannabadie/sage-topology-policy-v2` (Nemotron)
- **PyPI**: `pip install ygn-sage` — v0.1.0-alpha

## Detailed rules in .claude/rules/

- `critical-directives.md` — the 5 rules above, expanded
- `environment.md` — LLM models, API keys, SSL, ExoCortex
- `architecture.md` — pillars, pipeline, competitors, benchmarks
- `development.md` — build/test/bench commands, what NOT to benchmark
- `research-decisions.md` — paper-backed decisions, DROPPED items
- `meta-harness.md` — Meta-Harness harness optimization rules (arXiv 2603.28052)
