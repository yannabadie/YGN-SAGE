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
# Build — since 2026-04-22 ADR-013 §5 flip, sandbox+cranelift+tool-executor
# are in Cargo default features. `maturin develop` with no flags bundles
# the embedded RustPython wasm sandbox (~37 MB) for validate_and_execute.
# Add `--features smt,onnx` when you need the formal QualityLabeler /
# ONNX embedder / tokeniser paths.
cd sage-core && maturin develop --features smt,onnx
cd sage-python && pip install -e ".[all,dev]"

# Build recipe for the embedded RustPython wasm (one-time, cached):
#   rustup target add wasm32-wasip1
#   git clone https://github.com/RustPython/RustPython external/rustpython
#   cd external/rustpython && CARGO_TARGET_DIR=../rustpython-wasm-target \
#     cargo build --release --target wasm32-wasip1 --features freeze-stdlib
# build.rs picks it up from external/rustpython-wasm-target/.../rustpython.wasm
# and include_bytes!s it into sage-core. Without the artifact, the sandbox
# module emits a placeholder and callers fall through to the hard-fail
# path in validate_and_execute.

# Test
cd sage-core && cargo test --features smt --lib
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

## Current State (April 22, 2026)

- **Tests**: Python **1999 passed** (+41 net 2026-04-22 P0.4 B: +40 red-team attacks in `tests/test_wasm_sandbox_redteam.py`, +1 formerly-broken `test_created_tool_executes_in_sandbox` now passing after §5 flip routed `create_python_tool` through `validate_and_execute`. Unchanged pre-existing failures: 11 in API-key-dependent files — `test_e2e_live_providers.py`, `test_provider_pool_wiring.py`, `test_e2e_campaign.py`, `test_pydantic_ai_integration.py`) / Rust **496 passed** (+16 net 2026-04-22 P0.4 B: +8 `sandbox::wasm_python` tests — hello/exit/timeout/args-roundtrip/deny-filesystem/deny-env; +8 structural sandbox tests — double opt-in matrix, env-mutex, wasm-default invariant)
- **Sandbox (2026-04-22, ADR-013)**: `validate_and_execute` runs Python inside embedded RustPython wasm32-wasip1 sandbox **by default** (no env-var opt-in). Deny-by-default WASI-p1 contract: no filesystem, no network, no subprocess, no env inheritance, 256 MiB memory cap, epoch-interrupt timeout. 40 adversarial attacks validated (FS/net/proc/env/clock/mem/introspection/engine). `execute_raw` (which bypasses both AST validation AND the sandbox) still requires `SAGE_UNSAFE_RAW_EXEC=1`. `SAGE_UNSAFE_UNSANDBOXED` gate removed.
- **Dangerous tools (2026-04-23, §5 flip completion)**: `AgentConfig.dangerous_tools` default flipped `True` → `False`. `execute_bash` is no longer registered at boot by default. SWE-bench N=10 paired smoke (2026-04-22) showed typed-only produces 4/10 patches vs bash 3/10 — functional criterion met. `SAGE_DANGEROUS_TOOLS=1` remains as explicit opt-in (escape hatch for bench paths or callers that still need raw shell).
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
