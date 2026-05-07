# YGN-SAGE — Self-Adaptive Generation Engine

Agent Development Kit with 5 cognitive pillars: Topology, Tools, Memory, Evolution, Strategy.
Rust core (sage-core) + Python SDK (sage-python) + Knowledge Pipeline (sage-discover).

## CRITICAL DIRECTIVES

1. **Rust first, Python tolerant** — performance-critical in Rust (sage-core), Python for orchestration only
2. **Minimal heuristics** — routing learned (kNN primary, bandit Thompson — `routing.knn_92pct` ≥50/60 LOO-CV + `routing.system_router_88pct` ≥52/60 `delivered` floors in `docs/CLAIMS.yaml`; historical 92%/88% on earlier 50-task GT provenance only), verification formal (Z3/OxiZ). Adaptation thresholds (THETA_GOOD=0.7, THETA_CRITICAL=0.3, etc.) are calibrated initial values subject to ablation. Safety limits (MAX_RETRIES, MAX_REROUTES, cache bounds) are engineering guards. Replace heuristics with learned alternatives when data permits
3. **No corporate proxy** — this machine has NO proxy. Never add `verify=False`
4. **kNN is primary router** (`routing.knn_92pct` `delivered` in `docs/CLAIMS.yaml` at strict-equal floor ≥50/60 LOO-CV on the 60-task GT, S1≥16/20 / S2≥15/20 / S3≥19/20; historical 92% on the earlier 50-task GT subset is provenance only, not recertified by this floor) — ComplexityRouter heuristic is an **emergency fallback only** (wired at `pipeline.py:477` Priority-3 after Rust SystemRouter + kNN). AUDIT2 2026-04-24 flagged the "DEAD CODE" framing as technically contradicted by the live fallback path; "emergency fallback only" is the accurate framing.
5. **Evidence before assertions** — run tests + benchmarks before claiming completion
6. **SOTA minimum, AI breakthrough at least** — don't settle for "good enough"
7. **No training-leak model hardcodes** — truth for OpenAI/Gemini/xAI/DeepSeek/Anthropic models in this repo is `sage-core/config/cards.toml`, NOT the agent's training snapshot. Before adding a `"<tag>" in model` check or a quirk branch, verify the tag hits at least one id in cards.toml AND verify the quirk itself via Context7 `/berriai/litellm` or the provider's live docs — cite the source in the code comment. See `docs/patterns/knowledge-cutoff-checks.md`. *2026-04 incident*: hardcoded `o1/o3/o4` for a temperature clamp even though cards.toml only ships `gpt-5.x`.
8. **A14 posterior epoch guard is fail-closed** — active in Rust `load_state` and Python `boot_topology.py` since cycle-8 step 2 (`6b2ebcbe` + round-2 closure). Normal state requires `posterior_epoch.json` epoch=1 plus `topology_state_manifest.json` provenance binding over all A14 topology state files. For forensic load-only inspection, set `SAGE_BOOT_BYPASS_EPOCH_GUARD=1` with `SAGE_BOOT_BYPASS_REASON` and `SAGE_OPERATOR_ID`; bypass disables atexit save and `validate_epoch_for_save` / `ensure_clean_epoch_before_save` hard-fail. Reset command: `python -m sage.ops.a14_reset --reason "..."`.
9. **Declared ≠ verified — runtime integrity principle** (cgpro 2026-04-30 architect review, crystallized from 4 cycle traps: cycle-7 SAGE_ORACLE doc/code drift, cycle-8 R6.1c reason raw-leak vs audit policy drift, cycle-8 A14 epoch label vs DB content drift, **cycle-9 A3 timeout-vs-host-suspend drift** added 2026-05-04). Any label that authorizes a side-effect or learning decision MUST be bound to verified content, schema, provenance, or executable proof. Before shipping a new "label gates side-effect" code path: register it in `docs/contracts/runtime-integrity-ledger.md` with all 4 columns (Declared label / Verified content / Side-effect blocked if invalid / Tests), and write the regression test that proves the side-effect is blocked when verification fails. The ledger now has **10 invariants** (event payload schema, oracle evidence, posterior epoch, contaminated backup, RunFrame summary, bandit attribution, timeout enforcement, control-surface completeness, CLI protocol versioning added cycle-12 `f647c5ae`, **Tool capability declaration & grant enforcement** added cycle-13 K Phase 1.5 `5a4cfd1e`). See also `docs/contracts/rust-python-boundary.md` for ownership and `docs/status/current.json` for canonical test counts.

## Architecture (see .claude/rules/architecture.md for details)

```
sage-core/   — Rust (PyO3): TopologyEngine, SystemRouter, ModelAssigner, QualityLabeler, S-MMU, SmtVerifier
sage-python/ — Python SDK: CognitiveOrchestrationPipeline, AgentLoop, Providers (7), Bench (BigCodeBench, EvalPlus)
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
# REBUILD AFTER PULLING RUST SOURCE CHANGES (cycle-13 B 2026-05-06):
# the installed `sage_core.cp313-*.pyd` is what Python actually runs;
# pulling new commits in `sage-core/src/` does NOT update the binary.
# Stale binaries cause silent contract violations — most recently the
# 2026-04-30 fix at engine.rs:1031 (write_topology_state_manifest)
# was missed by 4-day-old wheels, leaving ~/.sage/ without a manifest
# and breaking directive #8 fail-closed boot guard. Regression test
# `tests/test_save_state_manifest_contract.py` catches this at the
# Python boundary on local dev. Always re-run `maturin develop` after
# `git pull` if `sage-core/` changed:
cd sage-core && maturin develop --features smt,onnx
# With Rust OTel:
cd sage-core && maturin develop --features otel,smt,onnx
# Workaround if `maturin develop` fails with "--include-debuginfo cannot
# be used with --strip" (pyproject.toml has strip=true for wheel size):
#   cd sage-core && maturin build --release --features smt,onnx --out target/wheels
#   pip install target/wheels/sage_core-0.1.0-cp313-*.whl --force-reinstall --no-deps
cd sage-python && pip install -e ".[all,dev]"

# Build recipe for the embedded RustPython wasm (one-time, cached):
#   rustup target add wasm32-wasip1
#   git clone https://github.com/RustPython/RustPython external/rustpython
#   cd external/rustpython && CARGO_TARGET_DIR=../rustpython-wasm-target \
#     cargo build --release --target wasm32-wasip1 --features freeze-stdlib
# build.rs picks it up from external/rustpython-wasm-target/.../rustpython.wasm
# and include_bytes!s it into sage-core. Without the artifact, the sandbox
# module emits a no-op stub and callers fall through to the hard-fail
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

## Current State (May 6, 2026 — Cycle-13 B chain FULLY CLOSED: stale-binary class wrapped in 4 defense layers)

- **Tests** (canonical at `docs/status/current.json`): **3089 Python collected** / **553 Rust listed** / **100 sage-discover**. Net cycle-13 prelude delta over cycle-12 closure 3021 = +68 (= 21 Pro shape canary + 1 event_log + 1 Yann CSV + 2 manifest contract + 8 a14_reset orphan + 24 sage_core_version + 11 wheel_smoke). mypy 0 / ruff clean.
- **Cycle-13 B chain** (autonomous segment, cgpro `cgpro_pi_mono_pivot_20260505` HITL, ~7 commits 2026-05-06 morning, all pushed `main`):
  - **Empirical bug catch** (cycle-13 E Tier 2.1 smoke runtime, 2026-05-05 evening): `engine.save_state(dir)` returned `Ok(())` but did NOT write `topology_state_manifest.json`. Each successful pipeline run left `~/.sage/` with state files but no manifest → next boot fail-closed per directive #8 → operator forced to run `python -m sage.ops.a14_reset` per cycle. Root cause: `sage_core.cp313-win_amd64.pyd` dated 2026-04-27 — 4 days BEFORE the manifest-write fix at `engine.rs:1031` (commit `f9521616` 2026-04-30). Stale binary class.
  - **L1 source code** (Rust):
    - `bc662d9a` `write_bytes_atomic` closure-wrapped + best-effort `remove_file(.tmp)` on rename failure + 2 Rust unit tests (success-no-leak + rename-failure-cleanup). Pre-fix the `.tmp` files leaked monotonically across retries.
    - `b035973e` build.rs injects `SAGE_CORE_BUILD_COMMIT_SHA` (via `git rev-parse HEAD` OR `SAGE_CORE_COMMIT_SHA_OVERRIDE` env) + `SAGE_CORE_BUILD_TIMESTAMP` (UNIX seconds) + `SAGE_CORE_BUILD_PROFILE` (cargo `PROFILE`); lib.rs exposes 4 PyO3 module attrs `__commit_sha__` / `__build_timestamp__` / `__build_profile__` / `__version__`. cgpro HARD_STOP fixed: `cargo:rerun-if-changed` paths resolved via `git rev-parse --git-path` and emitted only when path exists (handles PyPI sdist + git worktrees).
  - **L2 regression test** (Python boundary): `32d39bdf` `test_save_state_manifest_contract.py` (2 tests). `sage_core.TopologyEngine().save_state(tmp)` MUST write manifest. Per cgpro deep VERIFY 2026-05-06 Q3: every `state_files[]` entry's `sha256` / `size_bytes` MUST equal the file's CURRENT bytes (byte-exact binding). CI builds fresh wheels per commit so this passes on every CI run; local devs with stale `.pyd` get a clear `_REBUILD_HINT` error pointing at `cd sage-core && maturin develop --features smt,onnx`.
  - **L3 boot-time ops**:
    - `b25c28a6` `sage.ops.a14_reset` cleans orphaned `.<name>.<id>.tmp` files (pre-`bc662d9a` artifacts). `_orphan_tmp_files` + `_cleanup_orphaned_tmp_files` helpers source the file names via `_ATOMIC_WRITTEN_NAMES = (POSTERIOR_EPOCH_FILENAME, TOPOLOGY_STATE_MANIFEST_FILENAME, CONTAMINATED_MARKER_FILENAME)` (cgpro HARD_STOP corrected: was hardcoded string literal for the manifest filename). Audit `MANIFEST.json` records `cleaned_orphan_tmp_files: list[str]` always (empty list when none). 8 new tests + 7 surviving from before.
    - `9e426504` `sage.ops.sage_core_version` Python helper consuming the L1 build-info attrs. CLI: `python -m sage.ops.sage_core_version` exits 0 when wheel matches source HEAD, 1 on confirmed stale, 0 on unknown (default) / 1 on unknown with `--strict`. cgpro HARD_STOP fixed: validates cwd's git toplevel is a YGN-SAGE checkout via `git rev-parse --show-toplevel` + sentinel-file check (`sage-core/Cargo.toml` + `sage-python/src/sage/__init__.py`) BEFORE returning HEAD — otherwise running from inside an UNRELATED git repo would falsely flag the YGN-SAGE wheel as stale. 24 new tests.
  - **L4 release pipeline**: `db304bc6` `sage.ops.wheel_smoke` post-install assertion module + CI wiring. 4 phases: `_check_sage_core_imports` / `_check_build_info_attrs` (commit_sha != "unknown" CI smell) / `_check_required_symbols` (canonical 8 pyclasses) / `_check_save_state_manifest_contract` (byte-exact SHA256 binding asserted at runtime, NOT pytest). Wired as a step in `wheels.yml` + `release-test.yml` so the wheel smoke MUST pass on every TestPyPI/PyPI publish OR the publish is blocked. CLI exit 0 on pass / 1 on failure with structured JSON report on stderr. 11 new tests.
- **Cycle-13 F documentation** (`b5fbe064`): runtime-integrity-ledger row added under invariant 3 ("Posterior epoch") listing `tests/test_save_state_manifest_contract.py` as the new Python-boundary verifier. 7th adversarial-threats entry distinguishes the source-correct + binary-stale sub-class from prior 6 declared-vs-verified traps. CLAUDE.md rebuild-after-pull note documents the canonical `cd sage-core && maturin develop --features smt,onnx` recipe + the `--include-debuginfo` vs `strip=true` workaround.
- **Cycle-13 E REAL Pro grader result** (cumulative through `db304bc6`): instance `instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan` — `total_tests=300, PASSED=297, FAILED=3, fail_to_pass_count=3, f2p_resolved=0, p2p_regressed=0`. Cost $0.749 / latency 203.7s / 46 tool calls / 28 turns / 3-node sequential topology / sage/gemini-3.1-pro-preview reasoner-tier. **First end-to-end real Pro grader result in YGN-SAGE history** — agent attempted but didn't resolve, no regressions introduced. Smoke result doc at `sage-python/docs/benchmarks/2026-05-05-cycle13-arm-d-reasoner-n1.md`. Cycle-13 main run cost projection from real data: ~$37.50 / ~3-4h wall-clock for arm D N=50 alone.
- **External reviews invoked across the cycle-13 B chain** (single conv `cgpro_pi_mono_pivot_20260505` resumed): 8 cgpro pre-commit deep VERIFY rounds + 5 post-push `NEXT_BLOCK_ID` reports. **3 HARD_STOP fixes caught real consistency bugs**: cycle-13 I sourcing `TOPOLOGY_STATE_MANIFEST_FILENAME` constant; cycle-13 G Rust hard-coded `../.git/HEAD` path that broke PyPI sdist + worktrees; cycle-13 G Python `get_source_head_sha` running from arbitrary cwd. All fixed + re-VERIFIED before push.
- **Open follow-ups (cycle-13 main + later cycles)**: A. cycle-13 main run wiring (arms A/B/C, 3-5 days, $240-460 budget); C. `sage run --jsonl` v0 protocol gaps (`cli_progress` / `set_budget` / `cancel` / `cli_complete.payload.final_seq` per cgpro DESIGN E trap Q5); D. patch repair budget extension for diff_verifier observe → repair mode; J. ADR-015 Phase C façade rewrite + 6 stub deletion (pipeline.py 1801 → thin facade, pure refactor 0 behavioral changes, 3-5 days).
- **Cycle-13 cgpro/VERIFY transcript continuity**: `.tmp/cgpro_e_design.md` (E DESIGN), `.tmp/cgpro_e_tier_2_1_verify.md` + `.tmp/cgpro_e_tier_2_1_post_push.md` (Tier 2.1 closure), `.tmp/cgpro_b_precommit.md` + `.tmp/cgpro_b_deep_followup.md` (B Q1-Q5 deep VERIFY), `.tmp/cgpro_b_q4_precommit.md` (Q4 Rust .tmp cleanup), `.tmp/cgpro_i_precommit.md` + `.tmp/cgpro_i_precommit_v2.md` (Q4-bis HARD_STOP + fix), `.tmp/cgpro_g_rust_precommit.md` + `.tmp/cgpro_g_rust_precommit_v2.md` (Q1 Rust HARD_STOP + fix), `.tmp/cgpro_g_python_precommit.md` + `.tmp/cgpro_g_python_precommit_v2.md` (Q1 Python HARD_STOP + fix), `.tmp/cgpro_h_precommit.md` + `.tmp/cgpro_h_clarify.md` (Q2 wheel smoke). Read these before resuming cycle-13.

## Current State (May 5, 2026 evening — Cycle-13 E prelude: arm D smoke + event_log integration fix)

- **Tests** (canonical at `docs/status/current.json`): **3043 Python collected** / **553 Rust listed** / **100 sage-discover**. Net cycle-13 prelude delta over cycle-12 closure 3021 = +22 (= 21 Pro patch format adapter + 1 event_log regression). mypy 0 / ruff clean.
- **Cycle-13 prelude commits** (autonomous session segment after Yann "continues en toute autonomie", `6710eb0b..15fc82eb`, all pushed to GitHub `main`, per cgpro DESIGN E `cgpro_pi_mono_pivot_20260505` verdict GO_TIER_1_PLUS_2):
  - `6710eb0b` **Tier 1 scaffolding** — `clients/pi-ygn-sage/` skeleton (npm package pinning `@mariozechner/pi-coding-agent@0.73.0` + `@mariozechner/pi-ai@0.73.0` exact, NOT `@badlogic/pi-mono`; v0.73.0 = commit `dbcb473d6fdb96f60570b9ebe73e7aa6316fa8fb`) + `sage-python/scripts/swebench_pro_fetch.py` (HuggingFace `ScaleAI/SWE-bench_Pro` test split, stratified N=10, idempotent --seed 42) + `docs/benchmarks/2026-05-05-cycle13-arm-wiring.md` (concrete invocation contract per arm A/B/C/D + LF-only JSONL framing rule + env hygiene `PI_OFFLINE=1` `PI_TELEMETRY=0` `PI_SKIP_VERSION_CHECK=1` + Tier 2.1 hard cutoff Docker > 15 min OR API > $5) + `scripts/setup_pi_mono.sh` (idempotent clone of pi-mono v0.73.0 into gitignored `external/pi-mono/`).
  - `cdaa7594` **Tier 2.0 NO-API canary** — `sage-python/scripts/swebench_pro_format_patch.py` + 21 unit tests proving Pro grader's `{instance_id, patch, prefix?}` JSON list shape is produced AND that SWE-bench Lite's `{instance_id, model_name_or_path, model_patch}` shape is REJECTED with diagnostic "unexpected keys" error (the primary trap this whole module exists to prevent — cgpro DESIGN E trap Q5).
  - `d3fc6fe0` **Tier 2.1 + REAL PROD BUG FIX** — `pipeline.py:763` was creating a disabled `RuntimeEventLog(run_id=_new_runtime_run_id())` (no trace_dir kwarg + no `SAGE_TRACE_JSONL_DIR` env → writer.py:162 sets `disabled=True`) and then `install_event_log()` SHADOWED whatever the CLI installed. Result: `sage run --jsonl` emitted ZERO RuntimeEventLog events end-to-end (only `cli_started` + `cli_complete` envelope frames on stdout). Found EMPIRICALLY while building the smoke runner; cgpro DESIGN E flagged some protocol gaps (cli_progress NYI etc.) but did NOT catch the wholesale event-log shadowing. Fix at `pipeline.py:763`: `event_log = current_event_log()` first, fall back to creating a fresh one only if none installed. 80-LOC regression test in `test_pipeline.py::test_pipeline_run_respects_installed_event_log` proves task_started + run_id propagation. Also ships `sage-python/scripts/run_dryrun_arm_d.py` (520 LOC: --mock + real modes, --budget-usd cap, hard cutoff cumulative cost > $5.00, sets `SAGE_BOOT_BYPASS_EPOCH_GUARD=1` per directive #8 to avoid `~/.sage/` pollution across consecutive smokes). End-to-end smoke validated 7 events on `instance_future-architect__vuls-...` Go task (43.7s, $0 cost — agent gave up at budget tier).
  - `15fc82eb` Cycle-13 arm D smoke results doc (168 LOC) at `sage-python/docs/benchmarks/2026-05-05-cycle13-arm-d-smoke-N1.md` — documents Tier 2.0 + Tier 2.1-partial state, what is NYI per cgpro DESIGN E trap Q5 (cli_progress, set_budget, cancel, cli_complete.final_seq), what is Docker-blocked (grader call), real prod bug fixed, grader-ready predictions.json output paths, cycle-13 follow-ups now unblocked.
- **Tier 2.1 acceptance status**: shape-valid grader-ready predictions.json produced + 7 RuntimeEventLog events captured per task. Grader call (Pro `swe_bench_pro_eval.py`) gated by Docker daemon down on host. Once Docker up: `helper_code/gather_patches.py` + `swe_bench_pro_eval.py --use_local_docker` on the smoke output to close Tier 2.1 fully.
- **Real production bug closed**: cycle-12 prelude `sage run --jsonl` (commit `d09bed4d`) was telemetry-blind in CLI mode. Fixed in `d3fc6fe0`. Without this fix, cycle-13 main run (N=50 4-arm) would have produced predictions with no observability, defeating cgpro DESIGN E secondary metrics (oracle.trainable rate, bandit_attribution_mismatch rate, controller_decision distribution). 248/248 wider regression sweep PASS, 0 regression.
- **Open follow-ups (cycle-13 phase 2)**: Tier 2.2 Docker-up grader call + N=10 arm D, post-save manifest gap fix (advisor 2026-05-04 — `engine.save_state` writes manifest in Rust at `engine.rs:1031` but `~/.sage/` empirically lacks it after recent saves; needs investigation), `cli_progress` heartbeat + `set_budget` mid-run + `cancel` cancellation token + `cli_complete.final_seq` per cgpro DESIGN E trap Q5, arms A/B/C wire-up for cycle-13 main run, $240-460 API budget approval, Modal vs local Docker decision for grading.
- **Cycle-13 cgpro/VERIFY transcript continuity**: `.tmp/cgpro_e_design.md` (DESIGN E spec lock + 6 traps), `.tmp/codex_p6a_phase_b_swap.md` (codex IMPLEMENT prompt for cycle-12 P6-A Phase B). Read these before starting cycle-13 phase 2.  <!-- narrative-guard: allow historical-record -->

## Current State (May 5, 2026 — Cycle-12: pi-mono pivot prelude + Phase B 6-stage decomposition + P6-A Phase B + invariant 9 + CI hot-fix)  <!-- narrative-guard: allow historical-record -->

- **Tests** (canonical at `docs/status/current.json`): **3021 Python collected** / **553 Rust listed** / **100 sage-discover**. mypy 0 / ruff clean (type:ignore ceiling 51/51 — bumped 48→51 in `d75a4d71` for cli/run.py `[assignment]` codes with per-ignore justification). Net cycle-12 delta over cycle-11 follow-up baseline 3008 = +13 (= 16 prelude CLI + 16 Phase A wrappers + 3 invariant 9 + 3 factory field-sync − 1 lock retirement − 7 bypass-mutation retirement). Cycle-11 closure baseline 2953 → 3008 was the cycle-11 follow-up sweep (P9 phase 1 + periodic save preflight + cgpro VERIFY suite + Phase A factory + P5 release-test).  <!-- narrative-guard: allow historical-record -->
- **Cycle-12 commits shipped (all pushed to GitHub `main`, `259b2066..7e20372e`, ~50 commits, 13/13 CI green)**:
  - **Prelude** (CI debug 8 + pivot 5): `259b2066..fb617565` cycle-11 closeout CI debug fixes + `d09bed4d` `sage run --jsonl` backend (16 unit tests) + `3ef58aa6` cycle-13 SWE-bench Pro 4-arm ablation plan + `d75a4d71` mypy [assignment] cleanup. Per cgpro `cgpro_pi_mono_pivot_20260505` Option 1 verdict: pi-mono = front-end UX/transport, YGN-SAGE = orchestration backend, communicating via subprocess + JSONL/RPC (NOT MCP — 4-32× cheaper tokens, 100% vs 72% reliability). v0 protocol locked at `docs/contracts/SAGE_CLI_PROTOCOL.md` (18 outbound events / 5 inbound commands / 9 invariants including new invariant 9 "CLI protocol versioning").
  - **Phase A wrappers** (`3a851db3`): 10-module pipeline_v2/ scaffold with 16 byte-identical wrapper tests. Pure additive, no body movement. Acceptance gate: 25 P9 phase 1 tests still byte-identical.  <!-- narrative-guard: allow historical-record -->
  - **Phase B 6-stage decomposition** (`bc2b50c6..fad237be`, ~2050 LOC moved across 6 commits): stage 1 decompose (~24 LOC) → stage 2 classify (~114) → stage 3 assign_models (~84) → stage 4 select_topology (~539, codex IMPLEMENT) → stage 5 learn (~319 async, codex IMPLEMENT) → stage 6 execute (~603, codex IMPLEMENT, bypass mutation block intact). 3-actor recipe locked: cgpro DESIGN → Claude scaffold smallest 2-3 → codex IMPLEMENT 4-N → Claude VERIFY + COMMIT → cgpro VERIFY → SHIP. Recipe ~9 traps caught (async delegators, module globals, `__file__` drift, circular imports, signature drift, helper ownership, P9 test preservation, PipelineContext re-export, factory field sync). Each commit byte-identical green at the 25 P9 acceptance gate. 0 logic changes, 0 regressions.  <!-- narrative-guard: allow historical-record -->
  - **Invariant 9 backport** (`f647c5ae`): runtime-integrity-ledger 8→9 invariants. CLI protocol versioning binds the declared `protocol_version` field to a verified envelope check + payload schema reuse (event_log/payload_schemas.py v0). 3 regression tests in `test_runtime_event_contracts.py`. Closes the cycle-12 prelude SAGE_CLI_PROTOCOL invariant table.
  - **P6-A Phase B** (`9f7783cc` foundation + `7e20372e` swap + `8761f0db` CI hot-fix): replaces ~243-line singleton AgentLoop bypass mutation block in pipeline_v2/execute.py with a per-run `create_bypass_agent_loop()` factory call. Foundation `9f7783cc` propagates 3 implicit fields (`toolforge`, `evolution_memory`, `dangerous_tools`) per cgpro DESIGN trap Q7. Swap `7e20372e` retires the cycle-11 P6-B asyncio.Lock + ContextVar reentry guard + 12-field snapshot/restore (the band-aid). Net diff 622 insertions / 681 deletions (~470 LOC reduction) across 7 files. **Behavioral consequences**: concurrent bypass calls now run truly in parallel (no lock serialization), `sage_recurse` from inside bypass completes (no RuntimeError), H6 drift callback now records_failure against the FINALIZED model_id (post bandit/Rust selection — latent bug closed), gate_source_tier resolution moved to factory-time. Test rename: `test_pipeline_bypass_lock.py` (5 lock-contract tests) → `test_pipeline_bypass_structural_isolation.py` (4 structural-isolation tests proving the stronger invariant: singleton's 12 fields UNCHANGED before/after bypass, N concurrent bypass = N independent loops with singleton.run() never called, recursive bypass no-deadlock). cgpro VERIFY pre-push round returned `GO_PUSH` one-shot (no traps caught). **CI hot-fix `8761f0db`**: 14 follow-on test failures across 3 files (test_pipeline_bypass_restoration.py + test_pipeline_bypass.py + test_pillar_logging.py) all root-caused to the same swap retiring the legacy mutation contract. Per Yann directive 2026-05-05 "ne met pas la poussière sous le tapis": DELETE `test_pipeline_bypass_restoration.py` (3 tests strictly superseded by structural-isolation, file's own docstring even references the B9 refactor that this is), PRUNE 4 obsolete bypass-mutation tests in `test_pipeline_bypass.py` (kept 5 surviving: constructor / fallback / system entry tests), ADD autouse `_spy_loop_passthrough_factory` fixture in `test_pillar_logging.py` so `_SpyAgentLoop` can stand in for the singleton at the factory call site. 151/151 PASS post hot-fix.  <!-- narrative-guard: allow historical-record -->
- **External reviews invoked across the day** (~50 commits, single conv `cgpro_pi_mono_pivot_20260505`): cycle-12 strategic pivot DESIGN (Option 1 verdict), CI debug round (8 root-cause fixes, no sweeping under rug per Yann directive), Phase A wrappers DESIGN, 6× Phase B stage DESIGN+VERIFY, invariant 9 backport, P6-A Phase B DESIGN (Q1-Q7) + VERIFY (GO_PUSH).  <!-- narrative-guard: allow historical-record -->
- **Niche claim locked** (cgpro 2026-05-05): *"The coding agent that can show why it chose a topology, why it trusted or rejected a result, and why it did or did not learn from the run."* Promesse: *"Verified where possible, evidence-gated everywhere."* — the verified adaptive orchestration runtime that a coding agent CLI finally makes usable. NOT another coding agent.
- **Open follow-ups (cycle-13+)**: Cycle-13 npm adapter (`clients/pi-ygn-sage/` TypeScript scaffold consuming `sage run --jsonl`), B4 wheels CI matrix dispatch + TestPyPI dry-run, SWE-bench Pro 4-arm ablation N=10 smoke (validates pivot value-prop), façade rewrite + 6 stub deletion in pipeline.py (Phase C cleanup), P9 phase 2 characterization tests (cancellation / budget exhaustion / oracle gate fail-open), multi-agent fallback attribution sémantique fix.
- **Cycle-12 cgpro/VERIFY transcript continuity**: `.tmp/cgpro_pi_mono_pivot_20260505*.md` (single thread continuity), `.tmp/cgpro_p6a_phase_b_design.md`, `.tmp/cgpro_p6a_phase_b_verify.md`, `.tmp/cgpro_p6a_phase_b_post_push.md`, `.tmp/codex_p6a_phase_b_swap.md` (codex IMPLEMENT prompt). Read these before starting cycle-13.

## Current State (May 4, 2026 — Cycle-11 evening: P6-B + Phase 2 sage-router retire + P4 coupling + A14 isolation)

- **Tests** (canonical at `docs/status/current.json`): **2953 Python collected** / **553 Rust listed** / **100 sage-discover**. mypy 0 / ruff clean. Delta vs cycle-10 closure 2940: +5 P6-B regression suite (`test_pipeline_bypass_lock.py`) + +5 P4 coupling suite (`test_pipeline_topology_skip_guardrails_decoupling.py`) + 0 A14 isolation (autouse fixture, no new tests). Rust unchanged at 553 from cycle-10 closure hot-fix.
- **Cycle-11 commits shipped (all pushed to GitHub `main`, `da5aa636..3bc0a38f`)**:
  - `450786a5` **P6-B AgentLoop bypass lock** — lazy per-event-loop `asyncio.Lock` + `ContextVar` reentry guard wrapping the 12-field bypass mutation block in `pipeline.py:_stage_execute`. Closes the same-event-loop concurrency hazard A0a (2026-04-23) only papered over. Five regression tests cover serialization / exception restore / cancellation restore / sage_recurse fail-fast / non-bypass unaffected. P6-A (per-run AgentLoop factory, structural fix) remains deferred behind ADR-015 characterization tests.
  - `31dc2607` P6-B status bump (current.json `commit_sha 1e19bfae→450786a5`).
  - `3c6df4b2` **Phase 2 — `sage-router/` retired** (1503 LOC across 10 files removed). The standalone routing-only experiment (added 2026-04-25) was never imported by canonical runtime; cycle-10 P2 (97fba93f) made the disposition explicit and cycle-11 closes the decision by deletion. README capability table row flipped `planned` → `retired (cycle-11)`. Test count badge bumped to 2948/553 (and again to 2953/553 in P4 below). Heuristic `quality_estimator.py` deliberately NOT migrated — runtime is Z3/ONNX/abstain per cycle-10 P7 (`SAGE_QUALITY_ONNX` gate).
  - `2e736863` Phase 2 status bump.
  - `147ce18e` **P4 coupling test** — 5-test regression suite that locks the code-level invariant: `_stage_select_topology` and `_is_single_agent_execution` are deterministic functions of `(dag_features, system, domain, ctx.topology shape)` and **independent of `_skip_guardrails`**. Includes a source-inspection guard (`inspect.getsource()`) catching future direct coupling in unexercised branches. cgpro VERIFY round softened the wording per round-5 trap #5: this test settles the narrower question of *deterministic mechanism coupling*, NOT a "definitive proof of v7 gap". A3 N=50 cloud rerun remains useful for tighter intervals on boundary-stochastic tasks (/13, /82, /101).
  - `4087c013` P4 status bump.
  - `3bc0a38f` **A14 cross-test pollution closed** — autouse `_isolate_sage_state_dir` fixture in `sage-python/tests/conftest.py` wipes `<HOME>/.sage/` before each test. Pre-fix, `pipeline.py:3075` periodic `engine.save_state(Path.home() / ".sage")` (every `BANDIT_FLUSH_INTERVAL=10` runs) leaked state files into the per-pid `_PYTEST_HOME` without a matching `posterior_epoch.json`, causing 2 `test_system_run_*` failures when running `test_pipeline.py + test_pipeline_bypass.py + test_pipeline_stages.py` together. Fix is **fixture-driven, NOT env-var bypass** — `SAGE_BOOT_BYPASS_EPOCH_GUARD` stays explicitly forensic-only per directive #8. Result: 62/62 PASS in the previously-failing combination, 119/119 PASS in wider pipeline regression.
- **External reviews invoked tonight (cgpro→work→cgpro dynamic per Yann's directive)**:
  - cgpro round-4 (P6 DESIGN, pre-context-compaction) — drove P6-B implementation with B+ verdict (lock + ContextVar reentry).
  - cgpro round-5 (P6-B closure + Phase 2 priority confirmation, conv `cgpro_kimi_audit_response_20260504`) — confirmed Phase 2 sage-router delete priority, caught README badge + current.json drifts, gave commit-message scope.
  - cgpro VERIFY round (P4 wording + teeth + traps) — softened "definitively settles" framing to "deterministic mechanism coupling", flagged 5th source-inspection test (added), flagged A14 test pollution as next follow-up (closed `3bc0a38f`).
  - advisor consulted 4 times — pre-P4 priority pick, P4 commit prep, A14 fixture design, doc-alignment commit framing.
- **Open follow-ups (not started, queued for fresh-context cycle-11 continuation)**:
  - **P5 finish** — B4 wheels CI multi-OS matrix (`workflow_dispatch`-driven, multi-day runner job).
  - **P9 phase 1 characterization tests** — 5 golden tests per `docs/adr/ADR-015-pipeline-decomposition.md` (run byte-identical / oracle gate / bandit attribution / Fix C / control-surface). cgpro VERIFY explicit: needs *fresh context*, "writing weak characterization tests late is worse than not writing them yet". Acceptance gate before any `pipeline.py` decomposition.
  - **P6-A AgentLoop factory** — structural per-run isolation, blocked on P9 characterization tests.
  - **`pipeline.py:3075` epoch-consistency review** — advisor-flagged `2026-05-04`: mid-pipeline `engine.save_state(Path.home() / ".sage")` does NOT call `ensure_clean_epoch_before_save` like the atexit handler at `boot_topology.py:185` does. Whether mid-pipeline saves should write the manifest the same way is a real production-code question, NOT test hygiene. Touches invariant 3 ("Posterior epoch") in `docs/contracts/runtime-integrity-ledger.md` — should reference the existing invariant, not parallel-track. Separate ticket.
- **Cycle-11 cgpro/VERIFY transcript continuity**: `.tmp/cgpro_p6_complete_review_finaltext.md` (round-4 P6 design), `.tmp/cgpro_p6b_close_phase2_review_finaltext.md` (round-5 P6-B+Phase 2 closure), `.tmp/cgpro_p4_coupling_verify_finaltext.md` (P4 VERIFY round). Read these before starting cycle-11 continuation.

## Current State (May 4, 2026 — Cycle-10 P0-P9 shipped + P4 v7 N=10 closure: v7 gap = sample variance, definitively)

- **Tests** (canonical at `docs/status/current.json`): **2940 Python collected** / **549 Rust listed** / **100 sage-discover**. mypy 0 / ruff clean. Delta vs 2907: +24 from A.1-A.4 cycle-9 recovery (event ledger / wall-clock watchdog / Windows keep-awake), +3 from γ.2 host-suspend integration, +6 from α.1+α.2 targeted-filter tests.
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
  5. **SHIPPED α + cgpro round-2 (2026-05-04 12:08-13:43)** — paired diagnostic N=8 morning (`/13,/19,/34,/82,/89,/92,/93,/101` × full + no-grd, 4/8=4/8 byte-identical) + replay N=8 afternoon (`/13,/17,/19,/34,/37,/82,/89,/101`, full 3/8 vs no-grd 4/8). cgpro round-2 (commit `c136463e`) caught 4 corrections: (a) "v7 = sample variance" verdict was overconfident, (b) "robust → sequential" claim NOT supported by ledger (telemetry bug — `executed_template` was a ULID), (c) `avr_attempted` is BCB repair not internal AVR, (d) recommend full v7 N=10 counterbalanced replay. Closed (a)-(c) by fixing telemetry + adding 8th invariant "Control-surface completeness" to runtime-integrity-ledger. **Replay falsifies cgpro Q3 specific hypothesis**: /17 FAIL in both configs (deterministic), /37 PASS in both (deterministic), /89 TIMEOUT 4/4 — none of these are v7 drivers. Across 32 paired data points (morning 16 + replay 16), **0 deterministic mechanism divergences**. Boundary stochastic tasks (/13, /82, /101) explain v7's 3-task delta via sample variance alone. Topology coupling REPRODUCED (skip_guardrails → 5→3 nodes on /19+/34+/82) but outcomes unchanged. Analysis: `.tmp/paired_diagnostic_n8_analysis.md` + `.tmp/replay_discriminant_analysis.md`.
  6. **CYCLE-10 SHIPPED (P0-P9, 6 commits + P4 measurement)** — disciplined, conservative, no Sage-Lite, runtime-integrity layer intact:
     - `962977ab` **P0 Truth-sync** — README badge/table/sage-python README/.claude/rules aligned 2903→2940, 544→549, commit_sha→ea9f7837. Re-collected pytest at HEAD per advisor "don't replace one stale claim with another".
     - `f2111099` **P1 Rust hardening** — `MutationResult::try_into_graph()`/`expect_valid()` cfg(test), 7× `lock().unwrap()` → `unwrap_or_else(into_inner)`, 3 regression tests. cargo test --features smt --lib green.
     - `97fba93f` **P2 sage-router disposition** — 90-line README "standalone, NOT used by runtime", root pointer. Kimi "empty zombie" framing factually refutable.
     - `eb46002d` **P3 README v0.9** — "Verified Adaptive Orchestration Runtime — Research Preview" tagline + 24-row capability state table (delivered/default-on/opt-in/planned/parked) + "5 pillars" → architecture background.
     - `f22a77a0` **P7 Path 6 / ONNX cleanup** — `SAGE_QUALITY_ONNX=1` opt-in env-var gate (default off → no learned-model silent flip), 3 regression tests, framing aligned.
     - `858b9057` **P9 ADR-015** — pipeline.py decomposition contract (cycle-11/12 implementation), 4 invariants + 5 characterization tests + sequencing dependency on P6 documented. Implementation deferred.
  - **P4 v7 N=10 paired replay closure (2026-05-04 17:54)**: full **4/10** = no-grd **4/10** byte-identical per_task vector `[F,F,F,T,T,T,F,F,T,F]` (PASS: /19, /34, /37, /92). McNemar p=1.0, Cohen's d=0, 0 discordant. **Full reproduces v7 reference (April 2026) EXACTLY** — same 4 tasks PASS. **No-grd does NOT reproduce v7's 7/10**: /13, /17, /89 PASS-on-no-grd were all stochastic outliers (falsified across 4 paired runs = 52 data points). **DEFINITIVE VERDICT: v7 4/10→7/10 gap was sample variance.** Cycle-9 telemetry fixes (`c136463e` + `43726991`) verified populated: `executed_template`, `selected_template`, `dag_omega/delta/gamma`. Topology coupling REVISED — actual coupling is bypass-vs-sequential (single-agent vs 3-node) when `_skip_guardrails` toggles, not the morning-narrative "5→3 robust→sequential" (that was based on empty `executed_template` pre-cycle-9 fix). Latency: full 83.7s vs no-grd 100.2s avg (+19.7%) — replicates morning + replay finding "no-grd lets AVR fire fuller". **A3 N=50 cloud (P8) NO LONGER URGENT** for "settle v7"; would tighten CI on boundary-stochastic tasks but does not change qualitative verdict. Fix C (`a23e196b`) is correctly applied but addresses no deterministic phenomenon. Analysis: `.tmp/p4_v7_counterbalanced_analysis.md`. Output: `docs/benchmarks/2026-05-04-v7-counterbalanced-n10.json` + `.events.jsonl` (46 events, 0 errors).
  - **REMAINING cycle-10 work (multi-day, not for this session)**: P5 B4 wheels PyPI (4-6d CI matrix), P6 AgentLoop B9 per-run factory (2-5d, blocks P9 implementation), P8 A3 N=50 cloud (gated by P4 — now de-prioritized after sample-variance verdict).
- **Budget tier**: `deepseek-v4-flash`. `models.toml` + `llm/router.py` updated. A33 multi-turn safety active.
- **Strategic positioning (cgpro 2026-05-02)**: Cycle-9 = budget tier paired ablation. Premium frontier = Cycle-12+. SWE-bench-Live = Cycle-11. Rust changes only if A2 proves a gap.
- **Directive #9 7th invariant — ADOPTED 2026-05-04 (γ.1)**: `docs/contracts/runtime-integrity-ledger.md` listed 7 invariants at cycle-10 closure (was 6); **Timeout enforcement** binds the declared per-task `timeout_s` to a wall-clock-verified `elapsed_wall_ms <= timeout_s × grace_factor` check. The 4th adversarial-threats entry documents the cycle-9 A3 N=50 abort as the trap this invariant closes. (Cycle-9 α.1 added 8th "Control-surface completeness" + cycle-12 added 9th "CLI protocol versioning" — current ledger total is 9.)

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
- **2 contract docs added** (cgpro architect review 2026-04-30): `docs/contracts/runtime-integrity-ledger.md` (5 invariants at cycle-8 closure; cycle-9 α.1 added 7th + 8th, cycle-12 added 9th — current ledger total is 9) + `docs/contracts/rust-python-boundary.md` (ownership matrix Rust vs Python). No code refactor — documentary only.
- **Cycle-9 ordering reverted from "step 3 A22"** (closed/stale per cgpro architect review) **to closeout + learning attribution loop**. Cycle-9 main = A14b `route_integrated()` Stage-0 repair + minimal T2 memory write paths + A2 N=10 BCB-Hard smoke + decision gate to A3 N=50.

## Current State (April 29, 2026)

- **Cycle-7 default-on flip** (`128e1b89`, 2026-04-29 evening): `SAGE_ORACLE` is now **DEFAULT-ON**. Unset = oracle path active; kill-switch via `SAGE_ORACLE=0|false|off|no|disable|disabled` (case-insensitive; `disable`/`disabled` added in cycle-7 VERIFY round-1, commit `87daf89a`). Centralized predicate in `sage/runtime/oracle/env.py` `oracle_enabled()`. Validated by N=5 unset smoke (5/5 oracle_verdicts emitted, 0 raw leaks) + N=2 kill-switch smoke (0 oracle_verdicts emitted). Bandit / MAP-Elites / online-evolution / training-memory ONLY update when `verdict.trainable=True`. Posterior epoch=1 (post A14 reset 2026-04-29). Pre-flip code commits: `162e82ea` (BCB-Hard N=50 evidence with internal pass@1=30%, official Docker pass@1=32%, 49/50 = 98% per-task agreement) → `f6711385` (closed cgpro PUSH BACK on raw `bench_result["reason"]` leak; reason now SHA-256-hashed into `EvidenceRef.evidence_hash`) → `f9305d74` (post-leak-fix smoke 0 leaks across 106 events) → flip → `a5f916ea` (unset evidence) → `8b4b34b6` (kill-switch evidence). Cycle-7 VERIFY round-1 PUSH BACK closed at `f3a89631`: T4 forced `controller_decision.payload` is now **allowlist-only** (9 keys, no free-form `reason` leak), `reason_code` slug-constrained, `quality_score` clamped — see commit `87daf89a` writer + tests.

## Previous State (April 26, 2026)

- **Tests**: Python **2501 passing (excluding API-key-dependent files)**, 63 skipped, 8 fail + 2 error in `test_e2e_*` / `test_pydantic_ai_integration.py` (pre-existing, all API-key-gated). Rust **501+ passed** with `--features smt,cognitive,sandbox,cranelift` (CI plein vert on commit `50fb8e4f`).
- **Static analysis**: **mypy 0 errors** across 183 source files (was 131 errors / 48 files at start of 2026-04-26 closeout). type:ignore ceiling **45/45** (44→45 for `import yaml  # type: ignore[import-untyped]` — CI Linux runner doesn't get types-PyYAML transitively). **ruff clean.**
- **CI**: **plein vert confirmed on commit `50fb8e4f`** (run 24956390320, 2026-04-26) — first 8/8 GREEN run since the AUDIT-cycle red baseline of 2026-04-21. **25-commit closeout cycle** `87d30837..de640543` covered: rust clippy debt, sage-core/tests fmt, E0432 sandbox+cranelift gate, Windows `embedded_wasm_available` resilience, ruff lint debt across 26 files, mypy 131→0 (a2a-sdk pinned `<1.0`, `tools/generated_tools/` excluded, real a2a_server.py 0.3.x API drift fixed, sprint3_evidence.py dead-code cascade, AgentLoop class attrs, StreamingLLMProvider Protocol async→def fix), maturin `develop`→`build`+`pip install` CI recipe, 5 stochastic test redesigns (3 cma_me + 2 bandit with orthogonal-context training instead of Thompson noise reliance), Linux path-jail backslash normalization, Windows sqlite ORDER BY rowid tie-break, asyncio.sleep timing epsilon, integration-smoke `--limit` removed + REQUESTS_CA_BUNDLE drop, **bandit `restore_arm` persists `context_sum`/`context_count` across save/load** (cgpro find — real prod bug masking contextual learning loss on every restart). Follow-ups roadmap-A8..A13 in `roadmap.md` (CI wasm build, RNG seam, sort arm_keys, three-layer test split, bandit Pareto contract mismatch, lockfile).
- **Sandbox (2026-04-22, ADR-013)**: `validate_and_execute` runs Python inside embedded RustPython wasm32-wasip1 sandbox **by default** (no env-var opt-in). Deny-by-default WASI-p1 contract: no filesystem, no network, no subprocess, no env inheritance, 256 MiB memory cap, epoch-interrupt timeout. 40 adversarial attacks validated (FS/net/proc/env/clock/mem/introspection/engine). `execute_raw` (which bypasses both AST validation AND the sandbox) still requires `SAGE_UNSAFE_RAW_EXEC=1`. `SAGE_UNSAFE_UNSANDBOXED` gate removed.
- **Wasm-python JIT cache (2026-04-23, commit `50b4ee8`)**: `WasmPythonExecutor::new()` now caches the compiled Module under `$SAGE_WASM_CACHE_DIR` (or `$HOME/.sage/wasm_python_cache/`) keyed by `Engine::precompile_compatibility_hash` + SHA-256 of the embedded wasm bytes. Cold-start ~30 s → warm ~1 s via `Module::deserialize`. Self-heals on corrupt cache (delete + recompile + atomic-rewrite). Opt-out: `SAGE_WASM_CACHE_DISABLE=1`.
- **Build-time sandbox gate (2026-04-23, commit `cf188df`)**: `SAGE_REQUIRE_WASM=1` at build time turns a missing `rustpython.wasm` into a `panic!` instead of the stub-plus-runtime-fail behaviour. Default unchanged for fresh clones; use the flag in release / CI builds that MUST ship a real sandbox artifact.
- **Dangerous tools (2026-04-23, §5 flip completion)**: `AgentConfig.dangerous_tools` default flipped `True` → `False`. `execute_bash` is no longer registered at boot by default. SWE-bench N=10 paired smoke (2026-04-22) showed typed-only produces 4/10 patches vs bash 3/10 — functional criterion met. `SAGE_DANGEROUS_TOOLS=1` remains as explicit opt-in (escape hatch for bench paths or callers that still need raw shell).
- **Pre-emission diff-context verifier (2026-04-23, commits `c05eee0` + `711008a`)**: opt-in observability for SWE-bench emission hygiene. `SAGE_DIFF_VERIFIER_MODE=observe` annotates predictions.jsonl with `_diff_verifier_mismatches` (list per hunk where context/removed lines don't match file bytes at the claimed position). Default `off` (byte-identical to pre-verifier output). First observability smoke (2026-04-23, N=10) caught 2/2 emitted patches as `content_mismatch` with zero false positives — including one headerless-diff false-negative the fix in `711008a` closed. Spec: `docs/superpowers/specs/2026-04-23-diff-context-verifier-design.md`. Repair mode (auto-repair via LLM one-shot) is spec'd but NOT shipped; `SAGE_DIFF_VERIFIER_MODE=repair` downgrades to observe with a warning log.
- **Templates**: 12 (sequential, parallel, AVR, selfmoa, hierarchical, hub, debate, brainstorming, robust, horizon_pipeline, parallel_fanout, formal_solver)
- **Routing**: kNN primary (CORAL exact-match override), Rust SystemRouter, heuristic Priority-3 fallback. Accuracy figures cited historically (kNN 100% / 92% GT, SystemRouter 88%, heuristic 34%) were `evidence_pending` in this April 26 archive snapshot; current status: `docs/CLAIMS.yaml` — `routing.knn_92pct` `delivered` floor ≥50/60 LOO-CV, `routing.system_router_88pct` `delivered` floor ≥52/60, ComplexityRouter heuristic `retired`. Historical figures are provenance only.
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
