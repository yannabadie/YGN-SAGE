# YGN-SAGE — Self-Adaptive Generation Engine

Agent Development Kit with 5 cognitive pillars: Topology, Tools, Memory, Evolution, Strategy.
Rust core (sage-core) + Python SDK (sage-python) + Knowledge Pipeline (sage-discover).

## CRITICAL DIRECTIVES

1. **Rust first, Python tolerant** — performance-critical in Rust (sage-core), Python for orchestration only
2. **Minimal heuristics** — routing learned (kNN 92%, bandit Thompson), verification formal (Z3/OxiZ). Adaptation thresholds (THETA_GOOD=0.7, THETA_CRITICAL=0.3, etc.) are calibrated initial values subject to ablation. Safety limits (MAX_RETRIES, MAX_REROUTES, cache bounds) are engineering guards. Replace heuristics with learned alternatives when data permits
3. **No corporate proxy** — this machine has NO proxy. Never add `verify=False`
4. **kNN is primary router** (92% GT) — ComplexityRouter heuristic (34% GT) is an **emergency fallback only** (wired at `pipeline.py:477` Priority-3 after Rust SystemRouter + kNN). AUDIT2 2026-04-24 flagged the "DEAD CODE" framing as technically contradicted by the live fallback path; "emergency fallback only" is the accurate framing.
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
# Add `--features otel` (B1.b, 2026-04-25) when you want Rust hot-path
# spans bridged to OpenTelemetry alongside Python — see
# docs/observability/otel-genai-spans.md "Rust spans" section.
cd sage-core && maturin develop --features smt,onnx
# With Rust OTel:
cd sage-core && maturin develop --features otel,smt,onnx
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

# SWE-bench — ALWAYS run with the pre-emission diff-context verifier
# in observe mode (roadmap-A1 2026-04-24: observe is the new default
# for every SWE-bench smoke; only opt out if you have a specific
# reason). Annotates predictions.jsonl with _diff_verifier_mismatches
# for post-hoc analysis (zero cost on clean patches). We need ≥10
# flagged + ≥10 clean before flipping repair-mode as default.
# Gen log goes to <output-stem>-gen.log by default (SAGE_BENCH_LOG_FILE=0 to opt out).
SAGE_DIFF_VERIFIER_MODE=observe \
  python -m sage.bench --type swebench --dataset lite --limit 10 \
    --output docs/benchmarks/{date}-observe.json

# Same smoke with OTel spans piped to stdout (B1, opt-in)
SAGE_OTEL_EXPORTER=console SAGE_DIFF_VERIFIER_MODE=observe \
  python -m sage.bench --type swebench --dataset lite --limit 10 \
    --output docs/benchmarks/{date}-observe.json

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

## Current State (April 26, 2026)

- **Tests**: Python **2501 passing (excluding API-key-dependent files)**, 63 skipped, 8 fail + 2 error in `test_e2e_*` / `test_pydantic_ai_integration.py` (pre-existing, all API-key-gated). Rust **501 passed** with `--features smt`.
- **Static analysis**: **mypy 0 errors** across 183 source files (was 131 errors / 48 files at session start, 2026-04-26). type:ignore ceiling **44/44** unchanged. **ruff clean.**
- **CI baseline**: red since 2026-04-21 AUDIT cycle, brought back to green 2026-04-26 (~13 commits): rust clippy debt, sage-core/tests fmt, E0432 sandbox+cranelift gate, Windows embedded_wasm_available, ruff lint debt across 26 files, mypy 131→0 (a2a-sdk pin <1.0, generated_tools mypy exclude, a2a_server.py runtime bugs, sprint3_evidence.py cascade root, AgentLoop class attrs for toolforge/evolution_memory, StreamingLLMProvider protocol fix, ~30 small structural fixes), maturin build-vs-develop CI recipe.
- **Sandbox (2026-04-22, ADR-013)**: `validate_and_execute` runs Python inside embedded RustPython wasm32-wasip1 sandbox **by default** (no env-var opt-in). Deny-by-default WASI-p1 contract: no filesystem, no network, no subprocess, no env inheritance, 256 MiB memory cap, epoch-interrupt timeout. 40 adversarial attacks validated (FS/net/proc/env/clock/mem/introspection/engine). `execute_raw` (which bypasses both AST validation AND the sandbox) still requires `SAGE_UNSAFE_RAW_EXEC=1`. `SAGE_UNSAFE_UNSANDBOXED` gate removed.
- **Wasm-python JIT cache (2026-04-23, commit `50b4ee8`)**: `WasmPythonExecutor::new()` now caches the compiled Module under `$SAGE_WASM_CACHE_DIR` (or `$HOME/.sage/wasm_python_cache/`) keyed by `Engine::precompile_compatibility_hash` + SHA-256 of the embedded wasm bytes. Cold-start ~30 s → warm ~1 s via `Module::deserialize`. Self-heals on corrupt cache (delete + recompile + atomic-rewrite). Opt-out: `SAGE_WASM_CACHE_DISABLE=1`.
- **Build-time sandbox gate (2026-04-23, commit `cf188df`)**: `SAGE_REQUIRE_WASM=1` at build time turns a missing `rustpython.wasm` into a `panic!` instead of the placeholder-plus-runtime-fail behaviour. Default unchanged for fresh clones; use the flag in release / CI builds that MUST ship a real sandbox artifact.
- **Dangerous tools (2026-04-23, §5 flip completion)**: `AgentConfig.dangerous_tools` default flipped `True` → `False`. `execute_bash` is no longer registered at boot by default. SWE-bench N=10 paired smoke (2026-04-22) showed typed-only produces 4/10 patches vs bash 3/10 — functional criterion met. `SAGE_DANGEROUS_TOOLS=1` remains as explicit opt-in (escape hatch for bench paths or callers that still need raw shell).
- **Pre-emission diff-context verifier (2026-04-23, commits `c05eee0` + `711008a`)**: opt-in observability for SWE-bench emission hygiene. `SAGE_DIFF_VERIFIER_MODE=observe` annotates predictions.jsonl with `_diff_verifier_mismatches` (list per hunk where context/removed lines don't match file bytes at the claimed position). Default `off` (byte-identical to pre-verifier output). First observability smoke (2026-04-23, N=10) caught 2/2 emitted patches as `content_mismatch` with zero false positives — including one headerless-diff false-negative the fix in `711008a` closed. Spec: `docs/superpowers/specs/2026-04-23-diff-context-verifier-design.md`. Repair mode (auto-repair via LLM one-shot) is spec'd but NOT shipped; `SAGE_DIFF_VERIFIER_MODE=repair` downgrades to observe with a warning log.
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
