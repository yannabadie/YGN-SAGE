---
name: May 5 cycle-11 closure + CI debug + cycle-12 prelude+Phase A+Phase B (6 stages moved) + P6-A Phase B
description: HUGE single-session 2026-05-05. ~50 commits pushed across (a) cycle-11 closure (epoch preflight + cgpro VERIFY follow-ups + P6-A Phase A factory + P5 release-test workflow), (b) CI debug 8 root-cause fixes (cargo fmt, MagicMock __int__, time module closure, mypy ceiling, wasm cache probe race, test API drift, clippy hygiene, wasmtime RUSTSEC-2026-0114), (c) cycle-12 prelude pi-mono pivot (SAGE_CLI_PROTOCOL.md + sage run --jsonl backend + 16 unit tests + cycle-13 SWE-bench Pro ablation plan), (d) cycle-12 Phase A additive wrappers + Phase B 6-stage decomposition of pipeline.py (~2050 lines moved into pipeline_v2/), (e) invariant 9 (CLI protocol versioning) backported into runtime-integrity-ledger, (f) **P6-A Phase B SHIPPED** late session: 9f7783cc (factory field-sync per cgpro DESIGN trap Q7) + 7e20372e (~243-line bypass mutation block → ~30-line factory call, ~470 LOC reduction). Test rename `test_pipeline_bypass_lock.py` → `test_pipeline_bypass_structural_isolation.py` (5 lock-contract → 4 structural-isolation tests). cgpro VERIFY pre-push round returned `GO_PUSH` one-shot. Cumulative cycle-12 = ~50 commits, HEAD `7e20372e`. cgpro consultations `cgpro_pi_mono_pivot_20260505` validated Option 1 + GO_ALL_NIGHT for Phase B + GO_COMMIT_PUSH for closeout + GO_PUSH for P6-A Phase B. Recipe proven across 8+ commits: cgpro DESIGN → codex IMPLEMENT (gpt-5.5 xhigh fast) → Claude VERIFY → cgpro VERIFY → SHIP.
type: project
originSessionId: 3b36b883-89b8-4bbd-a2fe-cc9d305be971
---
## Session timeline 2026-05-05

### Cycle-12 Phase A + B execution (8 commits, 3a851db3..f647c5ae)

After CI green on `d75a4d71` confirmed baseline clean, the cycle-12 pipeline.py decomposition executed in 8 commits:

| Commit | Stage | Implementer | Lines moved |
|---|---|---|---|
| `3a851db3` | A.1 — additive wrappers (10 modules + 16 tests) | Claude | scaffold |
| `bc2b50c6` | A.2 stage 1 — `_stage_decompose` body | Claude | ~24 |
| `693366a8` | A.2 stage 2 — `_stage_classify` body | Claude | ~114 |
| `b0bb1e36` | A.2 stage 3 — `_stage_assign_models` body | Claude | ~84 |
| `160e5ca7` | A.2 stage 4 — `_stage_select_topology` body | **codex 0.128 xhigh** | ~539 |
| `84b75a1f` | A.2 stage 5 — `_stage_learn` body | **codex** | ~319 |
| `fad237be` | A.2 stage 6 — `_stage_execute` body (with bypass intact) | **codex** | ~603 |
| `f647c5ae` | B — invariant 9 backport (ledger 8→9, 3 regression tests) | Claude | docs+tests |

**Total: ~2050 lines of pure code motion + 16 phase-A wrapper tests + 3 invariant-9 tests. ZERO logic changes. ZERO test regressions. 25 P9 phase 1 byte-identical green at every commit.**

cgpro VERIFY pre-push round on `cgpro_pi_mono_pivot_20260505` returned `GO_COMMIT_PUSH` for the final 2-commit chain (execute + invariant 9).

### Cycle-13 prelude — autonomous session segment (after Yann's "Continues en toute autonomie")

After cycle-12 closeout (4 commits `9f7783cc..70f3cf4b` 18/18 CI green), Yann set autonomous mode + cgpro picked NEXT_BLOCK_ID=E (SWE-bench Pro 4-arm dry-run). cgpro DESIGN E (`cgpro_pi_mono_pivot_20260505`) returned `GO_TIER_1_PLUS_2` with 6 traps + sub-stage gating (Tier 2.0 NO-API canary BEFORE Tier 2.1 API smoke).

| Commit | Subject | Phase |
|---|---|---|
| `6710eb0b` | Tier 1 scaffolding: `clients/pi-ygn-sage/` + `scripts/swebench_pro_fetch.py` + arm wiring doc + setup_pi_mono.sh | E.T1 |
| `cdaa7594` | Tier 2.0 NO-API canary: `swebench_pro_format_patch.py` + 21 tests rejecting Lite-shape leak | E.T2.0 |
| `d3fc6fe0` | Tier 2.1 + EVENT_LOG REAL PROD BUG FIX: `pipeline.py:763` was creating disabled RuntimeEventLog and shadowing CLI's tee (no events on stdout in `sage run --jsonl`) — fixed via `current_event_log()` first | E.T2.1 |
| `15fc82eb` | Smoke results doc + grader-ready predictions.json output | Doc |

**Real production bug caught**: `sage run --jsonl` (cycle-12 prelude `d09bed4d`) was emitting **ZERO** RuntimeEventLog events end-to-end. cgpro DESIGN E flagged some protocol gaps (cli_progress NYI etc.) but did NOT catch the wholesale event-log shadowing. Found empirically while building the smoke runner. cycle-13 dry-run would have produced predictions with no observability, defeating cgpro DESIGN E secondary metrics. Fix is 5 lines + regression test in `test_pipeline.py::test_pipeline_run_respects_installed_event_log`.

**Pinning corrections per cgpro Q3**: pi-mono npm packages = `@mariozechner/pi-coding-agent@0.73.0` + `@mariozechner/pi-ai@0.73.0` (NOT `@badlogic/pi-mono`). pi-mono v0.73.0 = commit `dbcb473d6fdb96f60570b9ebe73e7aa6316fa8fb`. CLI binary = `pi`.

**Pro patch format trap closed**: `swe_bench_pro_eval.py` expects `{instance_id, patch, prefix?}` JSON list — DIFFERENT from SWE-bench Lite's `{instance_id, model_name_or_path, model_patch}`. Adapter rejects Lite-shape via "unexpected keys" diagnostic before "missing patch" (test `test_validate_record_rejects_lite_shape`).

**Tier 2.1 partial**: smoke produced 7 events end-to-end (`cli_started -> task_started -> routing_decision -> topology_selected -> model_assigned -> final_result -> cli_complete`) on `instance_future-architect__vuls-...` Go task at budget tier. Latency 43.7s, $0 cost (agent gave up). Shape-valid predictions.json grader-ready.

**Tier 2.1 full grader**: gated by Docker daemon being down on host. predictions.json IS ready for grading whenever Docker comes up. To grade: clone scaleapi/SWE-bench_Pro-os + `helper_code/gather_patches.py` + `swe_bench_pro_eval.py --use_local_docker`.

**Post-save manifest gap (advisor 2026-05-04, separate ticket)**: each successful pipeline run leaves `~/.sage/` with state files but no `topology_state_manifest.json`. Subsequent boots fail per directive #8 fail-closed guard. Workaround: runner sets `SAGE_BOOT_BYPASS_EPOCH_GUARD=1` + reason + operator_id (bypass disables atexit save). Real fix needs cycle-13 phase 2 work.

### Cycle-12 P6-A Phase B (2 commits, 9f7783cc + 7e20372e) — late session, after "Continues, ce n'est pas la nuit"

P6-A Phase B was DEFERRED from the Phase B 6-stage chain per cgpro DESIGN (behavior-changing risk separation from pure code motion). Re-tackled later in same session.

| Commit | Subject | Implementer |
|---|---|---|
| `9f7783cc` | Factory field-sync foundation: propagate `toolforge` / `evolution_memory` / `dangerous_tools` from singleton via `create_bypass_agent_loop()` (cgpro DESIGN trap Q7) | Claude |
| `7e20372e` | SWAP: ~243-line singleton AgentLoop bypass mutation block → ~30-line factory call + 2 module-level helpers (`_select_bypass_model`, `_make_bypass_drift_callback`) | **codex 0.128 xhigh** |

**Behavioral closure**: cycle-11 P6-B asyncio.Lock + ContextVar reentry guard + 12-field snapshot/restore RETIRED. Replaced by structural isolation: each bypass run gets a fresh AgentLoop instance, no shared mutable state, no lock to deadlock on. Net diff 622 ins / 681 del / **~470 LOC reduction**. Test rename `test_pipeline_bypass_lock.py` (5 lock-contract tests) → `test_pipeline_bypass_structural_isolation.py` (4 structural-isolation tests proving singleton's 12 fields UNCHANGED before/after bypass + concurrent + recursive).

cgpro DESIGN `cgpro_pi_mono_pivot_20260505` (Q1-Q7) → Claude scaffold + 9f7783cc + verify → codex IMPLEMENT swap → Claude commit 7e20372e → cgpro VERIFY pre-push round returned `GO_PUSH` one-shot (no traps caught) → SHIP.

**Latent bug closed**: legacy code captured `_bypass_model_id` BEFORE bandit/Rust selection took effect; H6 drift callback now records_failure against the FINALIZED model_id post-selection.

**Recipe established (use this template in cycle-13+):**

1. cgpro DESIGN consultation locks order + traps in detail (Q1-Q6 + 8 traps).
2. Claude does the smallest 2-3 stages himself to validate the recipe.
3. codex 0.128 xhigh fast does the larger stages via prompts derived from cgpro template.
4. codex sandbox can't write to .git/ on this Windows setup (`mas_d0z9tb4\codexsandboxoffline` ACL); Claude verifies + commits from parent session each time.
5. After each stage: targeted regression sweep (P9 phase 1 + Phase A wrappers + helpers) — must be byte-identical green.
6. Phase A wrapper tests need direction-inversion when the body moves (was "wrapper delegates TO legacy", becomes "legacy delegates TO wrapper").
7. cgpro VERIFY pre-push round, then push.

### DEFERRED to fresh session

**P6-A Phase B (AgentLoop bypass factory swap)**: cgpro recommended landing it BEFORE execute move. Skipped because:
- Behavior-changing (replaces ~150 lines of singleton mutation with one `create_bypass_agent_loop()` call) — distinct risk category from pure code motion.
- Mixing async + behavior change in same session would multiply trap risks.
- Factory function `create_bypass_agent_loop` already callable + tested via `test_agent_loop_bypass_factory.py` (P6-A Phase A shipped `ae6bc3bf`).
- Execute body at ~603 lines (with bypass intact) vs ~450 lines (post P6-A Phase B). Both byte-identical behavior; only structural simplification deferred.

### Cycle-11 closure pushed earlier in session (12 commits, 213183c1..e2e57ebe)

cycle-11 work bundled with CI debug: epoch preflight (`213183c1`), cgpro VERIFY #1 ctx.topology_id production fix (`96bc9163`), cgpro VERIFY #3 FrugalGPT/error-fallback path tests (`1effcd62`), cgpro VERIFY #4-#5 docs (`fe422475`), P6-A Phase A AgentLoop bypass factory + ADR-016 (`ae6bc3bf`), P5 release-test.yml workflow draft (`5a8cbe99`).

### CI debug 6 root-cause fixes (8 commits)

After push, CI was 6 jobs red. cgpro consultation `cgpro_ci_debug_20260505` triaged. Fixes (in order):

| Surface | Root cause | Commit |
|---|---|---|
| Rust default job (FAKE node_count) | `cargo fmt --check` blocking before clippy/tests | `692aadb8` |
| `test_system_hint.py` 4 failures | MagicMock.__int__ returns 1 by default; mock stubbed `route` not `route_integrated`; `is_training_evidence` kwarg lambda mismatch | `f2c4d6a9` |
| Windows RecursionError | `_make_time_seq` captured `time` MODULE in closure; monkey-patched `time.time` re-entered fake_time → infinite recursion. Fix: capture function reference at module import | `4a125689` |
| mypy_count ceiling 48>45 | 3 cycle-9 keep-awake type:ignores legit (Windows ctypes + ulid fallback) | `9a207b9e` |
| `corrupt_cache_is_self_healing` | Process-global LAST_NEW_USED_CACHE raced sibling tests. Per-call `Arc<AtomicBool>` probe via `CacheOverrides.probe` + tempfile uniqueness pid+thread-id+nonce | `673c27b7` |
| Test API drift | `MutationResult::unwrap()` → `try_into_graph().unwrap()` (cycle-10 P1 rename); `embed`/`embed_batch` got `py: Python` param | `0bcdfd58` |
| Clippy hygiene 13 files | All 7 feature variants (default/onnx/smt/otel/cognitive/sandbox+cranelift/tool-executor) clippy clean | `d646eaa0` |
| RUSTSEC-2026-0114 wasmtime | DoS panic on table allocation. wasmtime 44.0.0→44.0.1 atomic bump 15 wasmtime-* crates | `db1988e1` |

Plus self-inflicted fmt drift (`1b67e9ce`) + ERREURCI.md warning closure (`e2e57ebe`).

**Result**: CI run 25374660067 on `1b67e9ce` = 13/13 GREEN. e2e57ebe maintained.

### cgpro pi-mono pivot consultation `cgpro_pi_mono_pivot_20260505`

Yann strategic question: pivot YGN-SAGE to CLI agent via pi-mono (badlogic/pi-mono, TypeScript). cgpro verdict: **APPROVE_WITH_FOLLOWUPS, Option 1**:

> "YGN-SAGE should not become another coding agent CLI. It should become the verified adaptive orchestration layer that a coding agent CLI finally makes usable."

- pi-mono = front-end UX/transport (TUI, providers abstraction, RPC mode, extensions)
- YGN-SAGE = orchestration backend (verification, topology evolution, evidence-gated learning)
- Subprocess + JSONL/RPC, NOT MCP (4-32x cheaper tokens, 100% vs 72% reliability per web search)
- 4-cycle roadmap: Cycle-12 prelude → Phase B → Cycle-13 npm adapter + ablation → Cycle-14+
- 8 traps identified (pi-mono v0.73 churn risk, philosophy clash, provider duplication, tool-approval protocol-level, wheels first, claims truth, memory ownership, latency mode)
- Niche claim: *"The coding agent that can show why it chose a topology, why it trusted or rejected a result, and why it did or did not learn from the run."*
- Promesse: *"Verified where possible, evidence-gated everywhere."*

### Cycle-12 prelude shipped (5 commits, e2e57ebe..f49fb533)

- `d7613632` `docs/contracts/SAGE_CLI_PROTOCOL.md` v0 — 18 outbound events (14 inherited from RuntimeEventLog + 4 NEW cli_*) + 5 inbound commands (prompt, approve_tool_call, deny_tool_call, cancel, set_budget TIGHTEN-only) + 9 invariants (8 from runtime-integrity-ledger + NEW invariant 9 "CLI protocol versioning")
- `d09bed4d` `sage/cli/run.py` (refactor cli.py → cli/ package) + 16 unit tests in `test_sage_cli_jsonl.py`. Components: `_CliMirrorSinkHandle` tees RuntimeEventLog file → stdout (LF-only, broken-pipe tolerant), `_CliApprovalBridge` bridges TopologyRunner.approval_callback to cli_tool_request sub-protocol with 60s timeout default-deny, `run_jsonl_async` main entry
- `3ef58aa6` `docs/benchmarks/2026-05-05-cli-baseline-plan.md` — Cycle-13 SWE-bench Pro 4-arm ablation: A=Claude Code, B=pi-mono direct, C=YGN-SAGE via pi-mono, D=YGN-SAGE direct. Hypothesis Δpass@1 ≥5pp at ≤2× p50 latency. ~$240-460 budget for first run.
- `fb617565` README capability table + .claude/rules/architecture.md "CLI surface" paragraph + current.json
- `f49fb533` commit_sha PENDING→fb617565 closure

**Verified locally**: 16/16 new CLI tests + 25 P9 phase 1 still byte-identical + 6 test_cli.py + ruff clean. 3008 Python tests collected (was 2992, +16). `sage --help` and `sage run --help` work via both `python -m sage.cli` and console script.

## Active cgpro conversations

| Conv | State | Subject |
|---|---|---|
| `cgpro_pi_mono_pivot_20260505` | **active for cycle-12+** | Strategic pivot, post-push protocol |
| `cgpro_ci_debug_20260505` | closed | CI debug round 1 (resolved) |
| `cgpro_p9_phase1_verify_20260505` | closed | P9 phase 1 review (5 findings shipped) |
| `cgpro_2026_04_26_review` | older | cycle-10 closeout (alias 69ee3d8d) |

Per CGPRO.md protocol: post-push report on `cgpro_pi_mono_pivot_20260505` returns NEXT_BLOCK_ID. Pre-commit review on the same thread before next significant push.

## Open follow-ups (cgpro flagged)

1. **Multi-agent fallback attribution sémantique** — surfaced P9 phase 1 (`1effcd62`): `ctx.executed_template`/`executed_model_ids` not cleared during fallback handler. Bench sees "multi-agent + 2 models" when actual run is single fallback. Dangerous for metrics/traces. Tracked but not blocking.
2. **CI feature-matrix coverage** — clippy currently runs `--no-default-features` only. Nightly job for default+onnx+smt+otel+cognitive+sandbox+cranelift+tool-executor would catch feature-gated regressions.
3. **GH Actions Node.js 20→24 migration** — deadline 2 juin 2026, retrait 16 sept 2026. Future "rouge CI surprise".
4. **9th invariant "CLI protocol versioning"** must be backported into `runtime-integrity-ledger.md` per the maintenance discipline ("when adding a new invariant: append a row to both tables AND wire a regression test").

## Project state at HEAD f49fb533

- 3008 Python tests / 553 Rust tests / 100 sage-discover
- mypy 0 errors, ruff clean, type:ignore ceiling 48/48
- 8 runtime-integrity invariants enforced + 9th in CLI protocol (to be backported)
- ADR-015 (pipeline.py decomposition) + ADR-016 (AgentLoop bypass factory) Proposed
- P9 phase 1 acceptance gate locked (25 byte-identical tests across 5 files)
- P6-A Phase A factory shipped (callable, not yet wired)
- P5 release-test.yml draft (workflow_dispatch only)
- CLI prelude: SAGE_CLI_PROTOCOL.md + sage run --jsonl + 16 component unit tests
- 23 commits pushed today (18 cycle-11 closure + 5 cycle-12 prelude)

## Cycle-12 priority (NEXT_BLOCK_ID per cgpro post-push, TBD)

Per the 4-cycle roadmap, the natural next blocks (cgpro will rank):
- **Phase B**: ADR-015 pipeline.py decomposition (each stage emits contractual events per the JSONL protocol). Multi-week.
- **CLI integration tests**: golden snapshot tests with mocked-LLM pipeline boot. Required before adapter ships.
- **9th invariant backport**: append CLI protocol versioning to `runtime-integrity-ledger.md` + wire regression test.
- **Open-source release prep**: wheels CI matrix (P5 release-test.yml needs first dispatch run), then real PyPI publish workflow.
- **Multi-agent fallback attribution fix**: surface from P9 phase 1, low effort.

## CGPRO.md protocol reminders (Yann opened in IDE 2026-05-05 19:00)

- **Pre-commit review**: short consultation on `cgpro_pi_mono_pivot_20260505` with diff summary + non-changes verified + checks passed. Output format: exactly `GO_COMMIT_PUSH` or `HARD_STOP`.
- **Post-push report**: commit SHA pushed + validation result + main aligned origin/main + return NEXT_BLOCK_ID only.
- **Don't follow blindly**: Codex (Claude Code) re-reads files, verifies diff, runs tests. cgpro decides/challenges.
- **Don't paste**: env values, API keys, tokens, PATs.
- **Wait, don't bypass**: long responses 5-30 min are normal. If `Not signed in` / Cloudflare challenge / Selector broken, escalate to Yann; never auth as him.
