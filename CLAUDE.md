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
8. **A14 posterior epoch guard is fail-closed** — active in Rust `load_state` and Python `boot_topology.py` since cycle-8 step 2 (`6b2ebcbe` + round-2 closure). Normal state requires `posterior_epoch.json` epoch=1 plus `topology_state_manifest.json` provenance binding over all A14 topology state files. For forensic load-only inspection, set `SAGE_BOOT_BYPASS_EPOCH_GUARD=1` with `SAGE_BOOT_BYPASS_REASON` and `SAGE_OPERATOR_ID`; bypass disables atexit save and `validate_epoch_for_save` / `ensure_clean_epoch_before_save` hard-fail. Reset command: `python -m sage.ops.a14_reset --reason "..."`.
9. **Declared ≠ verified — runtime integrity principle** (cgpro 2026-04-30 architect review, crystallized from 4 cycle traps: cycle-7 SAGE_ORACLE doc/code drift, cycle-8 R6.1c reason raw-leak vs audit policy drift, cycle-8 A14 epoch label vs DB content drift, **cycle-9 A3 timeout-vs-host-suspend drift** added 2026-05-04). Any label that authorizes a side-effect or learning decision MUST be bound to verified content, schema, provenance, or executable proof. Before shipping a new "label gates side-effect" code path: register it in `docs/contracts/runtime-integrity-ledger.md` with all 4 columns (Declared label / Verified content / Side-effect blocked if invalid / Tests), and write the regression test that proves the side-effect is blocked when verification fails. The ledger now has **7 invariants** (event payload schema, oracle evidence, posterior epoch, contaminated backup, RunFrame summary, bandit attribution, **timeout enforcement**). See also `docs/contracts/rust-python-boundary.md` for ownership and `docs/status/current.json` for canonical test counts.

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

## External AI consultation — `cgpro` and `codex` (when stuck or before declaring done)

Two complementary tools. Use deliberately, not reflexively. Different verbs.

### `cgpro` — ChatGPT 5.5 Pro (analytic, slow, second opinion)

- **Invoke (verified 2026-04-28)**: `cgpro ask --json --background --timeout 1800 "$(cat .tmp/<prompt>.md)"` via Bash `run_in_background: true`. **Avoid `--no-stream`** — it buffers until done and loses ALL output if the stream interrupts mid-flight (R1+R2 verify hang incident). `--json` alone gives NDJSON streaming (one event per line, partial captures survive). `--background` keeps Chromium off-screen.
- **Skill `cgpro:cgpro` is doc-only**: it loads cgpro skill instructions but does NOT invoke the CLI. Always call via Bash.
- **Pass the GitHub repo URL** in the prompt — cgpro pulls live source. Verified caught real prod bugs (bandit `restore_arm`, R3 `_remaining_budget_usd` cost_tracker fall-through, R3 default-provider fail-open).
- **Project routing (2026-04-28)**: this cwd is linked to ChatGPT Project `YGN-SAGE` (gizmoId `g-p-69ed9637e63c8191b61c9741b50d1c01`). New conversations auto-route there with project memory pre-pended. `--resume <name|id>` DISABLES project auto-routing — resumed convos stay in original location.
- **Conversation pattern**:
  - **Fresh conv per ticket** (preferred since 2026-04-28): `cgpro ask --new-session --save cgpro_<item>_design ...` creates new conv in YGN-SAGE project, bookmarks for VERIFY follow-up via `--resume cgpro_<item>_design`. Avoids the cgpro_2026_04_26_review thread getting too long.
  - **Cycle-spanning thread**: `cgpro ask --resume cgpro_2026_04_26_review ...` — keeps full context but stays out of the project sidebar.
- **Browser profile lock**: cgpro Chromium uses `~/.cgpro/profile/Default/`. ChatGPT desktop app uses the same profile. **Only ONE process can hold the lock**. Symptom = `browserType.launchPersistentContext: Target page, context or browser has been closed`. Fix: close desktop app, OR `cgpro daemon start` to share warm browser.
- **NEVER `TaskStop` a cgpro BG that may be near completion**. The R1 incident: GPT response landed server-side, my TaskStop killed local CLI before it could read back. If a BG cgpro shows 0 bytes after ~20 min, THEN diagnose. Easiest recovery: ask user to copy-paste the chatgpt.com response into a local `Aᵢ.md` file.
- **Schema-first DESIGN for runtime contracts** (cgpro 2026-04-28 reassess methodology tweak): for tickets touching runner/event semantics, force cgpro to lock the SCHEMA (events, fields, redaction policy, sink failure semantics, public/private API boundary, pre-fix tests, non-goals) BEFORE codex writes any code. This avoids codex turning observability tickets into replay/dashboards/StateCore creep.
- **Use for**: holistic review of a substantial cycle (5+ commits), strategy critique before committing to an approach, finding non-obvious bugs across files, second opinion on stochastic test design, methodology audit, locked-spec DESIGN→VERIFY for the cycle pattern.
- **Don't use for**: quick syntax lookups (Context7), one-line refactors, well-known API explanations.
- **Cycle pattern**: cgpro DESIGN (locked spec) → codex IMPLEMENT (gpt-5.5 xhigh, full-auto direct exec) → claude verify-local (TDD via `git stash --keep-index` for pre-fix evidence) → cgpro VERIFY (debate if needed) → SHIP commit + push. Verified across 4 P0 tickets in cycle 1 (2026-04-28). R3 had a true VERIFY round-trip catching 2 micro-fixes — keep VERIFY for anything touching runner semantics.
- **Prompt hygiene**: lead with repo URL + commit SHA, structured "what shipped / what's stuck / what I'm about to do" summary, then 2-3 specific questions split into "verdict on what I did" / "what should I do next" / "what trap am I missing". For VERIFY: include diff summary + TDD evidence (pre-fix fail / post-fix pass) + 2-3 scrutiny points + commit message draft.
- **Source code reference**: cgpro plugin lives at `C:\Code\CGPro4Code` (also installed as `cgpro` on PATH). Read `src/cli/commands/ask.ts` + `src/core/orchestrator.ts` + `src/cli/commands/project.ts` for behavioral truths.

### `codex` — GPT-5.5 xhigh (action-oriented, second implementation)

- **Invoke**: skill `codex:rescue` or agent `subagent_type=codex:codex-rescue`. The rescue agent is described as "use proactively when stuck, want a second implementation, deeper root-cause investigation, or hand a substantial coding task to Codex through the shared runtime".
- **Use for**: when stuck implementing a specific change, want a second implementation pass on a tricky function, deep root-cause investigation that needs to read many files, hand off a substantial coding task while you work on something else.
- **Don't use for**: holistic review (cgpro is better), strategic decisions (`advisor` is better).

### Pattern: external review after substantial cycles (2026-04-26 evidence)

Before declaring CI green / closing a multi-commit cycle, give cgpro: (1) GitHub repo URL + branch, (2) summary of what shipped + what's stuck, (3) ask "what trap am I missing?". The 2026-04-26 closeout review caught **1 real prod bug** (bandit context persistence) **+ 5 structural improvements** (RNG seam, sort arm_keys, three-layer test split, bandit Pareto contract mismatch, lockfile) — all now tracked as roadmap-A8..A13. Without the review, all 6 would have shipped silently.

`advisor` is the third option — sees this conversation's full transcript automatically. Use for in-flight strategy checks ("am I about to make a mistake?"). Different audience from cgpro (which sees only what you write into the prompt).

## Current State (May 4, 2026 — A3 N=50 ABORTED, evidence-layer rebuild in progress)

- **Tests** (canonical at `docs/status/current.json`): **2934 Python collected** / **549 Rust listed** / **100 sage-discover**. mypy 0 / ruff clean. Delta vs 2907: +24 from A.1-A.4 cycle-9 recovery (event ledger / wall-clock watchdog / Windows keep-awake), +3 from γ.2 host-suspend integration test.
- **A2 v7 (2026-05-03)**: All 60/60 results. `full` 4/10 — gate MET on the surface, but see Fix C correction below.

  | Config | PASS | Passing tasks |
  |--------|------|---------------|
  | full | 4/10 | /19, /34, /37, /92 |
  | baseline | 8/10 | /13, /15, /17, /19, /34, /37, /89, /92 |
  | no-memory | 4/10 | /19, /34, /37, /92 |
  | no-avr | 4/10 | /13, /19, /34, /37 |
  | no-routing | 4/10 | /13, /34, /37, /92 |
  | **no-guardrails** | **7/10** | /13, /17, /19, /34, /37, /89, /92 |

  **CORRECTED FINDING (cgpro 2026-05-04)**: `no-guardrails` ablation flag (`AblationConfig(guardrails=False)`) sets `loop._skip_guardrails=True`, which short-circuits ONLY the `guardrail_pipeline` rule-based checks inside `phases/{act,learn,perceive}.py`. It does **NOT** disable the `TopologyController` (Phase C runtime adaptation: model upgrades, reroutes, debate-gate, prune). The two are orthogonal control surfaces. The previous "Guardrails (adaptive controller: model upgrades…)" framing in this doc and in `project_may03_a2_bench_debug.md` was a methodology error — the v7 4/10 vs 7/10 gap is the effect of `guardrail_pipeline` (rule-based input/output safety), not the TopologyController. Fix C (`a23e196b`) disables the TopologyController for budget tier; that is a different lever and its empirical validation is **still pending**.
- **A3 N=50 ABORTED (2026-05-03 19:58 → 2026-05-04 03:24)**: Windows Modern Standby (S0 DRIPS) suspended PID 34004 overnight despite "Performances élevées" scheme + standby-timeout=0. `asyncio.wait_for(timeout=120)` does not enforce wall-clock under host suspend (asyncio loop frozen alongside the process). BCB/273 reported 20278211ms (5h 38min) of poisoned wall-clock; only 34/300 tasks completed, all in `full` config. Pass rate 11/33 = 33% (BCB/273 excluded), within sampling noise of v7 `full` 4/10 = 40%.
- **Commits shipped (all on GitHub main)**:
  - `2792b44f`: Fix 1 (episodic close) + Fix 3 (OUTPUT REQUIREMENT)
  - `9715ed4e`: cross-provider fix (_is_cross_provider, returns None)
  - `99fd1c31`: 4 regression tests for cross-provider guard
  - `f7a8bc47`: cgpro-validated: topology revert + runner guard + entry_point fix
  - `6e79bf84`: CLAUDE.md + current.json updated
  - `7c6ab507`: architecture.md + sage-discover README updated
  - `a23e196b`: Fix C — disable TopologyController in TopologyRunner when tier=budget (still empirically unvalidated, see correction above)
  - `05e4a7c5`: retry+truncate fallback for episodic.db unlink on Windows PermissionError (10×1s + write_bytes(b"") last resort) — **cgpro flagged truncate as risky for SQLite/WAL; per-run state directories preferred long-term**
- **Cycle-9 recovery plan (cgpro-validated 2026-05-04, conv `cgpro_a3_recovery_20260504`)** — Steps 1-4 + γ SHIPPED, α IN PROGRESS:
  1. **SHIPPED `a56a76e2`** — append-only event ledger (RUN_START / CONFIG_START / TASK_START / TASK_END / TASK_TIMEOUT / TASK_ABORT / CONFIG_END / RUN_END) with fsync per emit, 11/11 tests.
  2. **SHIPPED `0036217b`** — per-task control-surface telemetry (`controller_attached`, `executed_template`, `node_count`, `skip_guardrails`/`skip_avr`/`skip_memory`/`skip_routing`, `llm_tier`, `system_routing`, `was_bypassed`, etc.) wired in `BigCodeBenchBench._capture_control_surface`. Answers "what mechanism caused the v7 gap?".
  3. **SHIPPED `b44156e7` + `0036217b`** — wall-clock watchdog (`HostSuspendDetected` raised when `elapsed_wall > timeout × grace_factor`; default `grace_factor=2.0`); tasks with `host_suspend_or_event_loop_stall=true` excluded from pass/fail aggregation and run marked non-gate-quality. 7/7 watchdog unit tests + 3/3 end-to-end integration tests (γ.2).
  4. **SHIPPED `46c280e3` + `0036217b`** — Windows hardening: `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` via `prevent_os_sleep()` ctx manager (no-op on non-Windows). 6/6 tests, ES_AWAYMODE_REQUIRED anti-pattern guard included. Combined with `powercfg /change standby-timeout-ac 0 /change hibernate-timeout-ac 0` it is **diagnostic-grade only** — gate-quality A3 should still run on cloud VM.
  - **SHIPPED γ** — 7th invariant "Timeout enforcement" registered in `docs/contracts/runtime-integrity-ledger.md` (cycle-9 closure of directive #9 "declared ≠ verified") + end-to-end integration test in `tests/test_bench_host_suspend_integration.py`.
  5. **IN PROGRESS α** — paired diagnostic N=8 (`full` vs `no-guardrails`, same tasks, same order) for **path attribution**. Goal: decide whether Fix C target is right BEFORE paying A3 N=50 cost. CLI flags `--ablation-configs` and `--task-ids` added in α.1+α.2.
  6. **DEFERRED** — A3 N=50 clean rerun on cloud VM, gated on α decision.
- **Budget tier**: `deepseek-v4-flash`. `models.toml` + `llm/router.py` updated. A33 multi-turn safety active.
- **Strategic positioning (cgpro 2026-05-02)**: Cycle-9 = budget tier paired ablation. Premium frontier = Cycle-12+. SWE-bench-Live = Cycle-11. Rust changes only if A2 proves a gap.
- **Directive #9 7th invariant — ADOPTED 2026-05-04 (γ.1)**: `docs/contracts/runtime-integrity-ledger.md` now lists 7 invariants (was 6); **Timeout enforcement** binds the declared per-task `timeout_s` to a wall-clock-verified `elapsed_wall_ms <= timeout_s × grace_factor` check. The 4th adversarial-threats entry documents the cycle-9 A3 N=50 abort as the trap this invariant closes.

## Current State (May 2, 2026 — Cycle-9 A2 smoke running)

- **Tests** (canonical at `docs/status/current.json`): **2902 Python collected** / **549 Rust listed** / **100 sage-discover**. mypy 0 / ruff clean.
- **Cycle-9 closed work** (HEAD `5617440e`): A14b attribution closure (commits `34e42ea5→6f23eea4`, cgpro-APPROVED), T2 memory write paths (YGN-16 tests 9/9 at `886597de`), swebench_patch_repair.py two-stage repair (18/18 tests), deepseek-chat→deepseek-v4-flash migration (3 files, `24f97f3c`), A33 deepseek reasoning_content multi-turn fix (`27770580`).
- **A14 reset 2026-05-02**: pre-A14 bandit/MAP-Elites state moved to `~/.sage/contaminated/pre_a14_20260502`. Clean epoch=1. Audit dump at `.tmp/a14_reset_20260502/`.
- **A2 N=10 smoke RUNNING**: `python -m sage.bench --type ablation --limit 10 --tier budget`. Paired ablation (full/baseline/no-memory/no-avr/no-routing/no-guardrails) × 10 BCB-Hard tasks with deepseek-v4-flash. Gate: ≥4/10 → A3 N=50, 3/10 → diagnostic, ≤2/10 → rollback.
- **Budget tier**: `deepseek-v4-flash` (non-thinking successor to deprecated deepseek-chat). `models.toml` + `llm/router.py` updated. A33 adds `OpenAIModelProfile` for thinking-mode multi-turn safety.
- **Symphony**: YGN-16 T2 tests on main. feat/symphony-dev-orchestration deferred (3 blockers; ops-only PR planned after A2 gate).
- **Strategic positioning locked (cgpro 2026-05-02)**: Cycle-9 = budget tier paired ablation. Premium frontier = Cycle-12+. SWE-bench-Live = Cycle-11. A31 + A32-followup = Tier 2. Rust changes only if A2 proves a gap.

## Current State (April 30, 2026 — post cycle-8)

- **Cycle-8 R6.1c + A14 shipped + closeout in progress**. Stack:
  - `78565578` cycle-8 R6.1c — payload schema versioning + 14 manifests + audit/strict-current modes (3285 LOC, 2-round cgpro VERIFY APPROVED at `49648263`)
  - `6b2ebcbe` cycle-8 step 2 A14 round-1 + `f9521616` round-2 — `topology_state_manifest.json` provenance binding closes "epoch label ≠ DB content" trap (cgpro APPROVED, `_CONTAMINATED.json` written on legacy `~/.sage/contaminated_pre_a14_20260429/` 2026-04-30)
- **Tests** (canonical at `docs/status/current.json`): **2887** Python collected / **544** Rust listed / **100** sage-discover. **+400 vs ancien claim 2484-2501** — cycle-7 + cycle-8 R6.1c (+18 tests) + cycle-8 A14 (+34 tests) + net rebase.
- **Static analysis**: mypy 0 errors / ruff clean (verified on cycle-8 R6.1c + A14 closures).
- **2 contract docs added** (cgpro architect review 2026-04-30): `docs/contracts/runtime-integrity-ledger.md` (5 invariants + module cross-reference) + `docs/contracts/rust-python-boundary.md` (ownership matrix Rust vs Python). No code refactor — documentary only.
- **Cycle-9 ordering reverted from "step 3 A22"** (closed/stale per cgpro architect review) **to closeout + learning attribution loop**. Cycle-9 main = A14b `route_integrated()` Stage-0 repair + minimal T2 memory write paths + A2 N=10 BCB-Hard smoke + decision gate to A3 N=50.

## Current State (April 29, 2026)

- **Cycle-7 default-on flip** (`128e1b89`, 2026-04-29 evening): `SAGE_ORACLE` is now **DEFAULT-ON**. Unset = oracle path active; kill-switch via `SAGE_ORACLE=0|false|off|no|disable|disabled` (case-insensitive; `disable`/`disabled` added in cycle-7 VERIFY round-1, commit `87daf89a`). Centralized predicate in `sage/runtime/oracle/env.py` `oracle_enabled()`. Validated by N=5 unset smoke (5/5 oracle_verdicts emitted, 0 raw leaks) + N=2 kill-switch smoke (0 oracle_verdicts emitted). Bandit / MAP-Elites / online-evolution / training-memory ONLY update when `verdict.trainable=True`. Posterior epoch=1 (post A14 reset 2026-04-29). Pre-flip code commits: `162e82ea` (BCB-Hard N=50 evidence with internal pass@1=30%, official Docker pass@1=32%, 49/50 = 98% per-task agreement) → `f6711385` (closed cgpro PUSH BACK on raw `bench_result["reason"]` leak; reason now SHA-256-hashed into `EvidenceRef.evidence_hash`) → `f9305d74` (post-leak-fix smoke 0 leaks across 106 events) → flip → `a5f916ea` (unset evidence) → `8b4b34b6` (kill-switch evidence). Cycle-7 VERIFY round-1 PUSH BACK closed at `f3a89631`: T4 forced `controller_decision.payload` is now **allowlist-only** (9 keys, no free-form `reason` leak), `reason_code` slug-constrained, `quality_score` clamped — see commit `87daf89a` writer + tests.

## Previous State (April 26, 2026)

- **Tests**: Python **2501 passing (excluding API-key-dependent files)**, 63 skipped, 8 fail + 2 error in `test_e2e_*` / `test_pydantic_ai_integration.py` (pre-existing, all API-key-gated). Rust **501+ passed** with `--features smt,cognitive,sandbox,cranelift` (CI plein vert on commit `50fb8e4f`).
- **Static analysis**: **mypy 0 errors** across 183 source files (was 131 errors / 48 files at start of 2026-04-26 closeout). type:ignore ceiling **45/45** (44→45 for `import yaml  # type: ignore[import-untyped]` — CI Linux runner doesn't get types-PyYAML transitively). **ruff clean.**
- **CI**: **plein vert confirmed on commit `50fb8e4f`** (run 24956390320, 2026-04-26) — first 8/8 GREEN run since the AUDIT-cycle red baseline of 2026-04-21. **25-commit closeout cycle** `87d30837..de640543` covered: rust clippy debt, sage-core/tests fmt, E0432 sandbox+cranelift gate, Windows `embedded_wasm_available` resilience, ruff lint debt across 26 files, mypy 131→0 (a2a-sdk pinned `<1.0`, `tools/generated_tools/` excluded, real a2a_server.py 0.3.x API drift fixed, sprint3_evidence.py dead-code cascade, AgentLoop class attrs, StreamingLLMProvider Protocol async→def fix), maturin `develop`→`build`+`pip install` CI recipe, 5 stochastic test redesigns (3 cma_me + 2 bandit with orthogonal-context training instead of Thompson noise reliance), Linux path-jail backslash normalization, Windows sqlite ORDER BY rowid tie-break, asyncio.sleep timing epsilon, integration-smoke `--limit` removed + REQUESTS_CA_BUNDLE drop, **bandit `restore_arm` persists `context_sum`/`context_count` across save/load** (cgpro find — real prod bug masking contextual learning loss on every restart). Follow-ups roadmap-A8..A13 in `roadmap.md` (CI wasm build, RNG seam, sort arm_keys, three-layer test split, bandit Pareto contract mismatch, lockfile).
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
