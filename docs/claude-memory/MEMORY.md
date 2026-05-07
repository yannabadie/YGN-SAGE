# YGN-SAGE Project Memory

## User
- [User Profile](user_profile.md) — Yann Abadie, GIE AD BRIVE, Rust+Python dev, French-speaking, RTX 3500 Ada

## Tooling mastery (apply every session)
- [cgpro plugin mastery](feedback_cgpro_mastery.md) — browser profile lock, project linking, NDJSON streaming, response recovery. ChatGPT project YGN-SAGE = `g-p-69ed9637e63c8191b61c9741b50d1c01`, linked 2026-04-28.
- [cgpro YGN-SAGE project centralization](feedback_cgpro_project_centralization.md) — Yann directive 2026-05-06: nouvelles convs cgpro YGN-SAGE doivent être créées SANS `--resume` pour auto-router dans le projet ChatGPT "YGN-SAGE". Garder la même conv vivante pour plusieurs questions sur un même topic est OK et recommandé.
- [status_snapshot.py first, then sync_doc_counters.py](feedback_status_snapshot_first.md) — cgpro Phase 1.5h round-8 lock 2026-05-06: status_snapshot.py is the canonical generator of current.json; sync_doc_counters.py only PROPAGATES from it. Run them in this exact order after any test count change.
- [No sweeping under the rug — fix even pre-existing](feedback_no_sweeping_under_rug.md) — Yann standing directive 2026-05-05: when CI is red, fix every surface at root cause regardless of whether mine or pre-existing. Warnings count. Pair with cgpro for triage, not for licence to skip.
- [codex delegation recipe (proven cycle-12 Phase B)](feedback_codex_delegation_recipe.md) — 3-actor split for repetitive mechanical refactor: cgpro DESIGN → Claude validates smallest 2-3 instances → codex 0.128 xhigh implements 4-N → Claude verifies + commits → cgpro VERIFY → SHIP. ~2050 lines moved 2026-05-05, 0 regressions. codex sandbox can't write to .git on this Windows setup (ACL); Claude commits from parent session.
- [No Co-Authored-By signature in cgpro prompts](feedback_no_signature_in_cgpro.md) — Yann directive 2026-05-07: don't append the Co-Authored-By trailer to cgpro VERIFY/post-push/DESIGN prompts. Keep it for `git commit -m` only.
- [Verify external audits empirically before acting](feedback_audit_verification_first.md) — Yann directive 2026-05-07 ("tu n'as pas avancé sur le fond"): triage audits Mini.md/critique avant cycle dev. `git log --since` + run code actuel + check Cargo default features. Audits peuvent cite état antérieur; not roadmap.

## Latest session
- [routing.* claims pinned delivered + narrative recalibration shipped 2026-05-07](project_may07_routing_claims_evidence.md) — Cycle 12-commit autonomous segment HEAD `0d6f56f8`. cgpro `cgpro_routing_claims_evidence_20260507` (4 rounds: DESIGN + amendment + 2 EDIT_REQUIRED) closed. test_routing_gt_invariant.py ship strict-equal floors empiriques (kNN 50/60 LOO-CV S1=16/20 S2=15/20 S3=19/20 ; SystemRouter 52/60 budget=1.0). Claims `routing.knn_92pct` + `routing.system_router_88pct` flipped `evidence_pending` → `delivered` (2-commit anchor pattern self-referential SHA). Narrative recalibration: 23 surfaces patched (live docs delivered-floor / archive snapshots past-tense + registry pointer per Option C). 4 nouveaux narrative_guard stale-status patterns (knn-status / systemrouter-status / knn-accuracy-pending / routing-currently-pending) + Option C predicate `_is_archive_historic_status_line` (3-token AND: archive context AND past-tense AND registry pointer). claims_audit GREEN 9 delivered ↑7, narrative_guard 28/28 PASS, mypy 0, ruff clean, 3292 Py / 555 Rust / 100 sage-discover. Investigation Phase 5 H-1/H-2/M-1 audits du jour: tous **already-closed** (cognitive default-on depuis 2026-04-22 P3.4) ou misqualified — Yann challenge "fond" provoquée a été fair, j'aurais dû triage avant. Next: A22b bucket-analysis ou stop here per cgpro recommendation.
- [cli_gaps stages A-E shipped 2026-05-07 — sage run --jsonl v0 protocol gaps closed](project_may07_cli_gaps.md) — Cycle-13 K post-Phase-2.2 cli_gaps stage chain CLOSED at HEAD `b4956396`. NEW cgpro conv `cgpro_cli_protocol_gaps_20260507`. Closes 4 NYI v0 gaps documented in `cli/run.py:21-29` at cycle-12 prelude. Stage A `2d557b15` (unify stdout seq + cli_complete.final_seq), Stage B `7bd48c17` (tightening-only set_budget), Stage C `2ce3c877` (cli_progress idle heartbeat with stage labels), Stage D `d0bfea2b` (cooperative v0 cancellation + terminal failure frame), Stage E `b4956396` (protocol doc cleanup + status sync 3241→3290 + final gates). pipeline.py UNTOUCHED across all 5 stages (299 LOC, < 300 HARD GATE). Critical discoveries: runtime failure frame shape is FLAT-redacted on stdout (NOT nested under payload); two-timestamp idle clock prevents heartbeat self-reset; `_StdoutSeqCounter.last` is canonical (mirror tracker is subset). 14 cgpro VERIFY rounds across the chain.
- [Phase 2.2 closure 2026-05-07 — pipeline.py compatibility-method retirement (22 commits, mypy 0 restored at Stage F, ADR-017 closure at E3)](project_may07_phase22_closure.md) — Cycle-13 K Phase 2.2 fully closed at HEAD `1b99271e`. Stage chain A→B (15)→C→D1.a/b/c→D2→D3→D4→E1→E2→F→E3. Final: pipeline.py 293 LOC (<300 HARD GATE), 27 test files migrated to module-function patching, 6 _stage_* + 36 _<helper> retired, 58 deletion contracts PASS, 42/42 P9 byte-identical, 3 ADRs current (015 closed / 016 implemented / 017 NEW). Stage F caught D3 mypy attr-defined regression (179 errors) by class-body annotations. E3 fixed missed D1.c retarget on test_pipeline_fallback_provider.py + raised type:ignore ceiling 51→54 with justification. cgpro `cgpro_phase22_test_rewrite_20260506` closed; tests 3241 Py / 555 Rust / 100 sage-discover.
- [Phase 2.2 progression: 19 commits shipped, Stage A+B+C+D1.a+D1.b complete](project_phase22_design_lock.md) — Autonomous session 2026-05-06, ~4h. cgpro conv `cgpro_phase22_test_rewrite_20260506` (in-project YGN-SAGE, no --resume per Yann directive). Commits: A (`b1f6a09d`) narrative guard + 58 RED/GREEN contracts → B1.1..B2.f (15 commits) all 27 test files rewritten (53 Pattern A + 31 Pattern B sites migrated to module-function patching, 14+ Q6 string monkeypatches preserved, P9 byte-identical 42/42 at every commit) → C (`10e38931`) atomic 6 `_stage_*` deletion + orchestrator retarget (pipeline.py 738→642 LOC, -96/-13%, Q6 + Trap 2 preserved) → D1.a (`4c1cb37c`) memory-gate helper retarget (build_write_gate, record_to_memory, emit_budget_exceeded; 7 production callsites + 13 test sites; _emit stays Q3a) → D1.b (`309136b6`) runtime-events helper retarget (5 helpers across orchestrator + execute + classify + bandit_attribution; 13 production callsites + 4 test sites; signature change reason_code arg[1]→arg[2] documented). 6 stage deletion contracts now PASS, 36 helper deletion contracts still XFAIL (D2 atomic drop scope). Tooling shipped: `scripts/narrative_guard_phase22.py` + `scripts/phase22_inventory.py` (AST-based stage-seam inventory tool, built after the test_pipeline_adaptation B1.2-sync oversight, prevents further missed callsites). Recipe proven: cgpro DESIGN → claude validates+rewrites → cgpro VERIFY (each pre-commit + post-push) → SHIP. **Remaining**: D1.c topology/costing/assign/bandit retarget (5+ files), D2 atomic 36-helper deletion + xfail drop, D3 conditional constructor extraction (only if pipeline.py >= 300 LOC after D2), D4 touched-file cleanup, E full narrative_guard PASS + ADR closure + claims_audit. Conv `cgpro_phase22_test_rewrite_20260506` is hot for resume.
- [May 6 night — Cycle-13 K Phase 2.1 façade rewrite SHIPPED (22 commits, 10 cgpro rounds, pipeline.py 1800→727 LOC)](project_may06_cycle13_k_phase21.md) — Cycle-13 K Phase 2.1 closeout SHIP signed by cgpro round-10 at HEAD `96155232`. Recipe : A1-A4 smallest helpers (Claude single-actor) + B1-B5 volumineux helpers + costing.py NEW + C0 reroute conversion + D0+D1 orchestrator.py NEW (`pipeline_mod` indirection preserves test_run_frame monkeypatches) + E0 PEP 562 lazy `__getattr__` (must land BEFORE E1 to avoid circular) + E1 PipelineContext to context.py with `__module__ = "sage.pipeline"` setattr + F drift sweep + F2-F2.4 5 EDIT_REQUIRED rounds catching predictable docs drifts. **Critical pivot round-4**: empirical grep found 27 test files mock `pipeline._stage_*` as runtime seam — cgpro reclassified delegator deletion to Phase 2.2 OPTION_3 with own DESIGN_LOCK; Phase 2.1 closes as "facade extraction with transition seams retained" inside amended 650-800 LOC target. 14 modules in pipeline_v2/ (3661 LOC). 37/37 P9 byte-identical at every commit. **GO_PHASE_2_2_DESIGN** signed for `cgpro_phase22_test_rewrite_20260506` new conv (no --resume) — scope: rewrite 27 test files + remove 6 _stage_* + remove ~22 _<helper> delegators + reach `pipeline.py < 300 LOC`. **Lesson**: Phase 2.1 5 EDIT_REQUIRED rounds caught drifts a comprehensive pre-push grep would have caught — Phase 2.2 DESIGN_LOCK should include narrative-guard-style mechanical lint.
- [May 6 late evening — Cycle-13 K Phase 1.5 ToolPolicy capability manifest (4 commits, cgpro DESIGN_LOCKED one-shot)](project_may06_cycle13_k_phase15.md) — Phase 1.5 shipped after Phase 0 closure. NEW cgpro conv `cgpro_phase15_toolpolicy_20260506` in-project YGN-SAGE per Yann directive 2026-05-06. DESIGN_LOCKED one-shot, 4 commits `218a695e..6497c427` Claude single-actor (codex not needed for ~600 LOC). 15 tests T1-T15. Ledger 9 → 10 invariants. claims_audit --strict GREEN with 20 claims. Default effective policy = `{pure}` only — every other tier requires explicit grant via `SAGE_TOOL_GRANTS` / TOML / programmatic. No hierarchical implicit closure. Tool.execute is the last-resort gate. AgentTool default = dangerous. ContextVar propagation handles bypass automatically per PEP 567 (T14 attest).
- [May 6 evening — Cycle-13 K Phase 0 + 0.6/0.6b/0.6c/0.6d/0.6e/0.6f closure (ALIRE.md remediation, 12 commits, cgpro 6+1 partial rounds VERIFY, narrative guard 27 docs × 14 patterns, 75 Phase-0 tests)](project_may06_cycle13_k_phase0.md) — Yann ran `cgpro` on new conv `Analyse approfondie de repo` (id `69fb0d11-9bd8-8390-a074-edb6826f8cb6`) producing ALIRE.md. Plan submitted, EDIT_REQUIRED returned 6 corrections. Plan mode → ExitPlanMode approved → applied. **7 commits pushed `7cda0d9f..aadaf6d6`**: Phase 0.1a `6e8ac732` claims registry, 0.2 `18f9c65e` test-counter sync + CI gate, 0.3 `2fdd8ae5` invariant count SoT (8→9), 0.1b `5619b3cf` strict current.json gate (schema v2, codified ≤1-commit grace), 0.4 `763d1fc1` evidence anchor pinning + loose-OR + claims_audit --strict enforcing, 0.5 `ba6078f4` Path 6 default-off non-regression, **0.6 `aadaf6d6` truthfulness coherence patch (cgpro post-Phase-0 EDIT_REQUIRED)**: README/AI-ARCHITECTURE ↔ CLAIMS drift split (capability vs metric rows), Path 6 collision resolution (engine path 6 = TemplateFallback per Rust enum, learned policy sibling-of-6 not 6 itself), Path 6 test docstring rewritten as EXPLICIT NON-PROMISE (runtime path6_usage_count counter deferred to Phase 2.1/2.2), sync_doc_counters docstring corrected (COUNTER-PROPAGATION GATE only, NOT pytest --collect-only), evidence_benchmark prefix gate (must be under docs/benchmarks/ or docs/audits/, +4 tests). 19 claims = 7 delivered + 3 default-on + 4 evidence_pending + 2 opt-in + 2 planned + 1 retired. claims_audit --strict GREEN. 3089→3136 Python tests (+47 Phase-0). HEAD `aadaf6d6`. Awaiting cgpro GO_NEXT_BLOCK Phase 1.5 ToolPolicy on conv 69fb0d11 (BG bkaearbk7).
- [May 6 — Cycle-13 B chain FULLY CLOSED in 4 defense layers (autonomous segment, 7 commits, 8 cgpro deep VERIFY rounds w/ 3 HARD_STOP fixes)](project_may06_cycle13_b_chain.md) — Stale Rust binary class (`sage_core.cp313.pyd` 2026-04-27 predated 2026-04-30 manifest-write fix at engine.rs:1031) wrapped in: L1 source code (`bc662d9a` Rust .tmp cleanup + `b035973e` build-info exposure `__commit_sha__/__build_timestamp__/__build_profile__/__version__`), L2 regression test (`32d39bdf` byte-exact SHA256 binding), L3 boot-time ops (`b25c28a6` a14_reset orphan tmp cleanup + `9e426504` sage_core_version helper w/ sentinel-file guard against unrelated git repos), L4 release pipeline (`db304bc6` wheel_smoke + wheels.yml + release-test.yml wiring). Plus cycle-13 F doc (`b5fbe064` ledger row + adversarial-threats #7 + CLAUDE.md rebuild note). cgpro `cgpro_pi_mono_pivot_20260505` HARD_STOPs caught: TOPOLOGY_STATE_MANIFEST_FILENAME constant sourcing, `git rev-parse --git-path` for PyPI sdist + worktrees, sentinel-file guard against `python -m sage.ops.sage_core_version` running from arbitrary git repo. **20 commits total in autonomous segment (cycle-12 P6-A Phase B 4 + cycle-13 E 8 + Yann's CSV regression 1 + cycle-13 B 7).** HEAD `db304bc6`. 3089 Python tests / 553 Rust / 100 sage-discover. cgpro post-push verdict `E` (stop autonomous segment, defensible end point). Cycle-13 B fully closed; remaining open: A main run wiring / C `sage run --jsonl` v0 protocol gaps / D patch repair budget / J ADR-015 Phase C facade rewrite.
- [May 5 — HUGE session: cycle-11 closure + CI debug + cycle-12 prelude + Phase A + Phase B (6 stages) + invariant 9 + P6-A Phase B](project_may05_cycle12_prelude.md) — ~50 commits today. Cycle-11 closure (epoch preflight + cgpro VERIFY follow-ups + P6-A Phase A factory + P5 release-test), CI debug (8 root-cause fixes), cycle-12 prelude pi-mono pivot (SAGE_CLI_PROTOCOL.md v0 + sage run --jsonl + 16 unit tests + cycle-13 SWE-bench Pro 4-arm ablation plan), **cycle-12 Phase A + Phase B 6-stage decomposition of pipeline.py (~2050 LINES MOVED into pipeline_v2/, 0 logic changes, 0 test regressions, 25 P9 byte-identical at every commit)**, invariant 9 backport (ledger 8→9), **P6-A Phase B SHIPPED LATE SESSION** (`9f7783cc` field-sync + `7e20372e` swap = ~243-line bypass mutation block → ~30-line factory call, ~470 LOC reduction, retires cycle-11 P6-B asyncio.Lock + ContextVar guard + 12-field snapshot/restore via structural isolation). cgpro `cgpro_pi_mono_pivot_20260505` Option 1 + GO_ALL_NIGHT Phase B + GO_COMMIT_PUSH closeout + GO_PUSH P6-A Phase B (one-shot, no traps). **Recipe proven across 8+ commits**: cgpro DESIGN → codex 0.128 xhigh IMPLEMENT → Claude VERIFY → cgpro VERIFY → SHIP. HEAD `7e20372e`. Niche claim locked: "verified adaptive orchestration runtime that a coding agent CLI finally makes usable".
- [May 4 cycle-9 closure: A3 abort recovery + α + cgpro round-2 + replay discriminant — v7 gap likely sample variance](project_may04_a3_recovery_cgpro.md) — A3 N=50 died at 34/300 (Windows Modern Standby S0 DRIPS). 12 commits shipped: event ledger (`a56a76e2`,11) + wall-clock watchdog (`b44156e7`,7) + Windows keep-awake (`46c280e3`,6) + bench wiring (`0036217b`) + 7th invariant + γ.2 (`ae371202`,3) + CLI flags α.1+α.2 (`8e1bf6dd`,6) + α paired diagnostic results (`970367b3`) + DAG features (`43726991`) + cycle-9 doc sync (`e6f66cd2`) + cgpro round-2 telemetry fix + 8th invariant (`c136463e`). 2940 Python tests. **α morning N=8: full=no-grd=4/8 byte-identical**. **Replay N=8 with /17+/37: 3/8 full vs 4/8 no-grd, 1 discordant (/13)**. Across 32 paired datapoints **0 deterministic mechanism divergences**, boundary stochastic on /13+/82+/101 explains v7's 3-task delta via sample variance. cgpro round-2 caught: (a) overconfident verdict, (b) `executed_template` was ULID telemetry bug → fixed, (c) `avr_attempted` is BCB repair not internal AVR, (d) recommend full v7 N=10 counterbalanced replay. Topology coupling REPRODUCED (skip_guardrails → 5→3 nodes on /19+/34+/82) but outcomes unchanged. **Cycle-10 priority**: (1) full v7 N=10 counterbalanced replay, (2) perceive→TaskPlanner coupling test, (3) telemetry split, (4) A3 N=50 cloud only after 1+3.
- [May 2 cycle-9 A2 bench RUNNING + deepseek migration + CLI](project_may02_cycle9_a2_bench.md) — A14 reset, deepseek-v4-flash migration (A33), A2 bench running (4/7 PASS gate MET), CLI dispatcher `c9448a9b`, 2903 tests. A3 authorized.
- [May 1 cycle-9 T2 closeout + test sweep](project_may01_cycle9_t2_closeout.md) — `f01a9cc6` pushed. 3 stale test_pipeline tests fixed (A14b mock wiring). gate_rejected telemetry applied from T2. 50/50 targeted pass. T2 branch integrated. A2 smoke blocked on Symphony infra (YGN-14 NOT READY).
- [April 30 cycle-8 closeout + cgpro architect review + cycle-9 strategy lock](project_april30_cycle8_closeout_architect_review.md) — Closeout shipped `86681ac8`. cgpro architect review (33 KB, conv `cgpro_architect_review`) accepted 6 reproches méthodologiques + Q1/Q2/Q3 cycle-9 locks. **Cycle-9 = budget tier paired ablation, A14b `route_integrated()` repair (γ option), A2 N=10 → decision gate → A3 N=50.** A22/A8/A9/A10/A11/A13 confirmed shipped/stale. SWE-bench-Live = Cycle-11 reproducibility lane. NEW directive #9 "Declared ≠ verified" + 2 contract docs (runtime-integrity-ledger + rust-python-boundary) + scripts/status_snapshot.py (single source of truth: 2887 Py / 544 Rust / 100 sage-discover, +400 vs old claims).
- [April 30 cycle-8 R6.1c payload schema versioning + 2-round VERIFY APPROVED](project_april30_cycle8_r6_1c.md) — Cycle-8 R6.1c closed at `49648263`. 13 event types schemás avec versioning, manifest drift tripwire byte-exact, validator audit/strict-current modes. Empirical Q6 N=50 audit found 56 raw-phrase hits in pre-`f6711385` `oracle_verdict.reason_codes` (cycle-7 historical artifact, post-fix prevented, cgpro chose Option A doc disclosure NOT schema versioning).
- [April 29-30 cycle-7 default-on flip + cgpro 3-round VERIFY APPROVED](project_april29_cycle7_flip.md) — Cycle-7 closed at `4b8af448`. 29 ship + 5 closure commits. T4 forced `controller_decision.payload` allowlist-only (no `reason` leak). Kill-switch `0|false|off|no|disable|disabled`. BCB-Hard N=50 internal 30% / Docker 32% / 49/50 agree. cgpro round-2 trap-fix: stale-phrase lint scoped to current-state docs (golden + README/CLAUDE/roadmap/contracts/Dashboard).
- [April 29 R6.1a EvidenceProducers cycle 6 closure](project_april29_r6_1a_cycle6.md) — 6-cycle runtime arc complete (RuntimeContracts → StateCore → RunFrame → OracleStack → EvidenceProducers). cgpro 3-round VERIFY, APPROVED. Cycle-7 default-on gate locked. Commits `38c0da4e..25e604dd`.
- [April 27 boot loop + test pollution fix](project_april27_boot_loop_fix.md) — 18+3 pre-existing failures resolved (swebench import + asyncio.run cleanup). Commits `0b9c2464` + `20bb93b1`.

## ⭐ Active direction (read this FIRST when fresh session resumes)

**Active cgpro conversation (use --resume for continuity):** `cgpro_pi_mono_pivot_20260505` (cycle-12+ pi-mono pivot strategic thread). Per CGPRO.md protocol: pre-commit review + post-push report on this thread. See [May 5 cycle-12 prelude](project_may05_cycle12_prelude.md) for context. Older thread `cgpro_2026_04_26_review` (alias UUID `69ee3d8d-6154-8392-b79a-3a0202e887d2`) is closed for cycle-10 closeout. See [cgpro is source of truth](feedback_cgpro_source_of_truth.md).

**Cycle COMPLETE 2026-04-27 (9/9 traps shipped, 10 commits pushed to origin/main):**

| Trap | Commit | Subject |
|------|--------|---------|
| Setup | `af1ccb21` | docs(roadmap): set up cgpro-driven trap resolution cycle (A20-A27 + Trap C) |
| A12+Trap F | `5a390c48` | docs(rust): align bandit.rs + tool_executor.rs module docs with code reality |
| A14/roadmap-A20 | `48dc7c3f` | fix(bandit): make Python pipeline learning causal — REAL PROD BUG |
| A15/roadmap-A21 | `761c1797` | fix(packaging): declare sage-core dep + fail-closed dynamic tools |
| A22 | `133b86b5` | feat(bench): diff-verifier reason codes + outcome telemetry |
| A23 | `7f72ad28` | ci(sandbox): build rustpython.wasm + Trap E matrix |
| CI hotfix | `a5688048` | rust-features link error split + Security pip-audit/sbom A21 propagation |
| A25 | `5ef1940f` | feat(bandit,cma): add RNG seam + sort arm_keys |
| A27 | `2c8d2557` | ci(deps): pip-tools constraints + freshness gate + latest-deps |
| A26 | `e57ae680` | test(stochastic): 3-layer split for bandit/CMA tests + empirical workflow |

CI run on `5ef1940f` (last A25 push) confirmed **11/11 jobs GREEN** (incl. Trap E closed: Windows cranelift sandbox 3.12+3.13 GREEN, Build RustPython wasm artifact + cache GREEN, integration-smoke GREEN).

**Methodology that worked (repeat in future cycles):**
- Per-trap state machine: cgpro DESIGN (locked spec) → codex IMPLEMENT (gpt-5.5 xhigh, full-auto direct exec; skip the codex:rescue helper on this setup, `spawn codex ENOENT` issue) → Claude verify-local + capture diff → cgpro VERIFY (debate if needed) → SHIP commit + push.
- One cgpro conversation across the entire cycle (`cgpro_2026_04_26_review`, alias for `69ee3d8d-...`). cgpro keeps repo state cached; resumed turns are fast.
- Token economy: cgpro and codex do the heavy compute on OpenAI side. Claude (~1M context) orchestrates + verifies + composes commits.
- For docs-only changes ≤30 lines: direct Edit is more token-efficient than codex delegation. For ≥200 LOC implementation: always codex.

**Real production bugs caught and fixed by cgpro this cycle:**
1. roadmap-A20 / A14: Python pipeline bandit was off-policy AND its record method was unreachable (PyO3 name mismatch). Two compounded bugs; bandit was learning literally nothing from production traffic. Combined with the 2026-04-26 morning `restore_arm` fix (also cgpro-found), the bandit now actually works. **A14-reset** ticketed as separate follow-up — cgpro recommends reset of accumulated posteriors since old pipeline never recorded model_id/template, so historical telemetry can't prove causality.
2. roadmap-A21 / A15: `pip install ygn-sage` was advertising a one-command install that never worked (no sage-core dep declared). Plus `create_python_tool` silently fell through to Python subprocess sandbox when ToolExecutor was missing. Both fixed; A21 + A18 Gate 1 tested.
3. A23 build-wasm-sandbox closed the P0 sandbox-no-CI-coverage hole; Trap E (Python 3.13 + Windows sandbox+cranelift) closed in same commit.

**Open follow-ups (none blocking):**
- A14-reset — audit/reset accumulated bandit SQLite posteriors
- A14b — repair `route_integrated()` to accept context vector + executed template + cancel pending decision on constraint fallback
- A18b — ToolForge Gate 2 (`_run_tests`) execution isolation when bwrap/isolated_executor unavailable
- A22b — bucket-analysis script that aggregates `_diff_verifier_outcome` first, `_diff_verifier_reasons` second
- A22c — explicit regression test that off-mode JSONL contains NONE of the 3 verifier keys
- A22d — deletion-side `/dev/null` test for `file_creation_or_deletion` (creation case is covered)
- A23-Windows-confirm — was confirmed GREEN on the post-hotfix CI run (Windows cranelift works; no follow-up needed unless future regressions)
- A23-security-followup — investigate `cargo-audit` + `cargo-deny` `continue-on-error: true` license/advisory findings
- A27-followup — regenerate `sage-{python,discover}/constraints.txt` from clean Ubuntu/Python 3.12 baseline (currently Windows-tinted with `pywin32` pins; `python-constraints` job is `continue-on-error: true` until then). Easy: trigger latest-deps workflow's `workflow_dispatch` once and commit its output.
- A27b — per-platform constraints (constraints-linux-py312, constraints-windows-py312, etc.) once Windows-only transient drift becomes recurring
- B4 — publish/bundle sage-core wheels to PyPI; removes the local `--find-links` wheelhouse dependency in A27's generator script
- A3 — paired N=50 observe-vs-repair smoke (API-budget gated, out of cycle scope)

**Codex helper note:** the codex:rescue agent's runtime had `spawn codex ENOENT` on this Windows setup. Bypassed by calling `codex exec --full-auto --skip-git-repo-check` directly via Bash. Direct exec works fine; the helper plugin runtime is broken (separate issue, not blocking).

**Follow-ups recorded for after cycle:**
- A14-reset — audit/reset accumulated bandit posteriors (off-policy garbage). cgpro recommends reset by default — old pipeline never recorded model_id/template so historical telemetry can't prove causality (≥95% threshold unverifiable).
- A14b — repair `route_integrated()` to accept context vector + executed template + cancel pending decision on constraint fallback. Currently Stage 0 still calls legacy `route()`.
- A10 spillover note: when A10 lands (sort `arm_keys`), the new `choose_from_candidates()` helper must ALSO be sorted, not just the original `choose()` / `choose_contextual()` loops. Otherwise A10 is incomplete.
- **Test pollution / order-dependence (NEW 2026-04-26 finding, pre-existing not A14)** — `python -m pytest tests/ -k "not test_e2e and not test_pydantic_ai_integration and not test_live_multiprovider and not test_swebench"` shows **18 failed** (all in test_execution.py / test_sandbox.py / test_sandbox_executor.py / test_sandbox_safety.py with `NotImplementedError`) + **3 errors** (test_provider_pool_wiring). Same 18+3 on HEAD~1 (pre-A14, commit `5a390c48`), so PRE-EXISTING. ALL 67 sandbox/execution tests PASS in isolation (`pytest tests/test_execution.py tests/test_sandbox.py ...`). Test pollution: some test running alphabetically before test_execution.py leaves state that breaks the sandbox path. CI must shard or skip these. Investigate via binary-search of test files. May be related to logfire/asyncio teardown OR module-level singleton leak. Not blocking the cgpro cycle but should be fixed before next CI debt closeout.

**Most recent work (2026-04-26 morning):** **CI debt closeout — CI confirmed plein vert.** **25 commits** from `87d30837` → `de640543`. Run `24956390320` on commit `50fb8e4f` is the first 8/8 GREEN since the AUDIT-cycle red baseline of 2026-04-21 — including Windows pytest, Rust sandbox+ONNX+SMT+tool-executor, OTel Linux+Windows MSVC, and Integration Smoke (real API, main only).

Headline metrics:
- **mypy 131 → 0 errors** across 183 source files. Forensic fixes by root cause, NO silencing.
- **type:ignore ceiling 44 → 45** (only +1 for `import yaml  # type: ignore[import-untyped]` in execution/__init__.py — CI Linux runner doesn't have types-PyYAML transitively while local does).
- **ruff clean**.
- **2501 Python tests passing** (8 fail + 2 error all in API-key-dependent files, pre-existing baseline unchanged).

**Real production bugs found and fixed** (not silenced):
1. `protocols/a2a_server.py` was calling 18 sites of kwargs that don't exist on a2a-sdk 0.3.x: `context.request.message` (gone — use `context.message`), `context.task` (gone — use `context.current_task`), `event_queue.put` (gone — use `enqueue_event`), `AgentEvent(phase=, data=)` (gone — use `type=, step=, timestamp=, meta=`). Tests passed because they only constructed AgentCard + executor, never `execute()`. Always-broken at runtime, never tested.
2. `bench/sprint3_evidence.py:86` — dead-code line `report = bench.run if asyncio.iscoroutinefunction(bench.run) else bench.run` poisoned 12 downstream attribute accesses by typing `report` as `Callable[..., Coroutine[..., BenchReport]]`.
3. `StreamingLLMProvider` Protocol method was `async def ... -> AsyncIterator[str]` so mypy saw `Coroutine[Any, Any, AsyncIterator[str]]` from callers, blowing up `async for chunk in provider.generate_stream(...)`. Canonical pattern is `def` returning AsyncIterator.
4. **`bandit::restore_arm` did not persist `context_sum` / `context_count`** (cgpro find — see "External review pattern" below). Schema only saved 9 fields per arm (Beta/Gamma posteriors); `context_sum: Vec::new(), context_count: 0` on load. **Production deployments lost the contextual cosine-bias channel on every restart while still loading the (less informative) quality posteriors.** Fixed: schema migration adds `context_sum TEXT DEFAULT '[]'` (JSON) + `context_count INTEGER DEFAULT 0`; ALTER TABLE backward-compat for pre-existing DBs; `restore_arm` signature widened (10→11 args); `test_context_bias_survives_save_load` regression test pins the contract.
5. `typed_repo._resolve_within_cwd` — Linux passed `..\..\..\etc\passwd` because backslashes are literal characters there (no traversal). Defense-in-depth: normalize `\\` → `/` before resolution. 47/47 typed-repo redteam tests pass.
6. `episodic.py:list_all` — Windows clock resolution ~15 ms collapses 3 rapid `await store()` calls into the same `created_at` bucket, so SQLite ORDER BY was unstable. Added `, rowid DESC` tie-breaker.
7. `consolidator_single_flight` — Windows `asyncio.sleep(0.1)` returns ~3e-5 s "early" due to timer granularity. Relaxed lower bound to `>= 0.099` (1 ms epsilon).
8. `a2a-sdk` pyproject pin was `>=0.3` with no upper bound. CI was silently resolving to 1.0.2 (released 2026-04-24, real protocol/runtime migration). Pinned `<1.0`. Migration to 1.0 is a separate initiative with smoke tests.

**Stochastic tests redesigned (5 tests, not just budget-bumped)**: 3 cma_me convergence tests (sigma 0.5→1.0, budget 32×20 / 30×16 instead of 4×5 / 8×10) + 2 contextual-bandit tests (orthogonal-context training [1,0] vs [0,1] instead of collinear, threshold 35/50 = 70% with real cosine-similarity differentiation instead of Thompson noise random-walk).

**External review pattern (works — repeat in future cycles)**: When stuck or finishing a substantial cycle, call `cgpro:ask` (ChatGPT 5.5 Pro) with the GitHub repo URL + a concise summary of what you've done. cgpro pulled the persistence.rs from the live repo, traced through `restore_arm`, and surfaced the prod bug in (4) above that I would have missed. Also flagged: RNG seam pattern (`ChaCha8Rng::seed_from_u64` for testability), HashMap iteration order (sort `arm_keys` before Thompson sampling), three-layer test split (deterministic / seeded / statistical-#[ignore]), bandit-Pareto contract mismatch (docs vs code), lockfile for transitive deps. These are tracked as roadmap-A8/A9/A10/A11/A12/A13 — see Gate below.

**CI changes worth knowing for next session**:
- `maturin develop` requires an active venv that GitHub's `setup-python@v6` doesn't create. CI now uses `maturin build --release --features smt,onnx --out target/wheels` + `pip install target/wheels/sage_core-*.whl --force-reinstall --no-deps`. Same recipe in python-sage Linux + windows jobs.
- `[tool.mypy] exclude = ["src/sage/tools/generated_tools/"]` — those are 1-2 line sandbox-eval scripts with injected `json` / `args` globals, not standalone Python.
- `integration-smoke` job: dropped invalid `--limit 5` pytest arg, dropped `REQUESTS_CA_BUNDLE: ""` (violated directive #3), added skip-when-secret-empty guard.
- 5 tests now skip cleanly when `rustpython.wasm` artefact isn't bundled (test_meta_security executes-in-sandbox case, all of test_tool_creation, all of test_swebench_ca_patch via `pytest.importorskip("swebench")`). Sandbox itself has 0 CI regression coverage as a result — see roadmap-A8 below.

**Earlier (2026-04-25):** **roadmap-B1.b Rust OTel bridge shipped.** Independent Rust OTel SDK + W3C traceparent across PyO3 (no PyO3 0.27 upgrade needed). 10-task subagent-driven-development cycle. Spec at `docs/superpowers/specs/2026-04-25-otel-rust-spans-design.md`, plan at `docs/superpowers/plans/2026-04-25-otel-rust-spans.md`. 27 Rust span call sites audited (counts/IDs only, zero raw payloads). 9 Rust unit tests + 1 E2E InMemoryExporter smoke + 5 Python integration tests. CI gates added (rust-features + windows jobs). Sub-items deferred: B1.b.1 (cosmetic span renames), B1.b.7 (logfire-mode Rust export), B1.b.9 (OTLP batch exporter with tokio runtime ownership). B1.b.8 (CI matrix) closed inline.

**Earlier 2026-04-25 work:** **roadmap-B1 OpenTelemetry GenAI spans shipped.** 10-task subagent-driven-development cycle (brainstorm → spec → plan → implement). Spec at `docs/superpowers/specs/2026-04-25-otel-genai-spans-design.md`, plan at `docs/superpowers/plans/2026-04-25-otel-genai-spans.md`. Final reviewer APPROVE; 24/24 obs tests + 2493 full-suite passing; default off (zero overhead via `SAGE_OTEL_EXPORTER=none`); A16 redaction + 4 KiB truncation on payloads + safe-exception path closes auto-stacktrace info-leak.

**Earlier 2026-04-25 work:** A8 Phase 3 — native PydanticAI `OpenAIModelProfile` for Moonshot/Kimi (commit `ec5d0c98`). Replaces manual ThinkingPart hack with native `openai_chat_thinking_field='reasoning_content'` + `openai_chat_send_back_thinking_parts='field'` per Context7 docs.

**Earlier 2026-04-24 work (31 commits):** Full AUDIT{,2,3} remediation + PROMPT.md post-hoc reconstruction + A13/A14 closure. **10/10 security fixes fully wired.** §6.4 advisor verdict = MERGE. Plus roadmap-A1 (observe-mode default) + roadmap-A2 diagnosis (Kimi reasoning_content cascade). See [April 24 AUDIT triple](project_april24_audit_triple.md).

**Gate for next "continue":**
- ✅ **CI confirmed plein vert on commit `50fb8e4f`** (run 24956390320, 2026-04-26). Closeout cycle closed at `de640543`. No open AUDIT rows. roadmap-B1 closed. roadmap-B1.b closed (2026-04-25). roadmap-B8 closed (2026-04-23).
- **Non-gated follow-ups** (use `roadmap-` prefix to disambiguate from AUDIT claim IDs like AUDIT.md §3 A1-A17):
  - **roadmap-A14 — Bandit causality test** (NEW 2026-04-26 cgpro find, REAL PROD BUG **verified**). pipeline.py:461 calls legacy `_rust_router.route()` — bandit not consulted. pipeline.py:1243 calls `bandit.select_with_context()` separately, drops `decision.model_id`/`decision.template`, only stores `decision_id`. pipeline.py:1762 records outcome against orphan ID. Bandit posteriors update for arms whose model never executed. Decision needed: keep accumulated off-policy posteriors vs reset SQLite bandit. See [project_april26_cgpro_review_findings.md](project_april26_cgpro_review_findings.md).
  - **roadmap-A15 — Packaging fail-closed** (NEW 2026-04-26 cgpro find, **verified**). `sage-python/pyproject.toml:18-31` has no `sage_core` dep; README says `pip install ygn-sage`; runtime requires sage_core. PyPI install gets nothing matching CI maturin recipe. Fix: declare sage_core dep + fail closed when ToolExecutor unavailable (no silent fallback to Python subprocess sandbox).
  - **roadmap-A3a — Verifier reason codes** (NEW 2026-04-26 cgpro find, no API budget). Verifier collapses malformed/missing/creation cases all to `[]`. Add reason codes (`clean`/`content_mismatch`/`malformed_hunk_header`/`hunk_body_count_mismatch`/`file_missing`/`file_creation_or_deletion`/`not_unified_diff`/`unsupported_no_opinion`) to turn "zero flags" into interpretable distribution.
  - **roadmap-A8 — Build rustpython.wasm in CI** (task #159, pending). Sandbox-dependent tests currently skip via `embedded_wasm_available()` so the sandbox has 0 CI regression coverage. Options: (a) dedicated `build-wasm-sandbox` job with GitHub Actions cache keyed on RustPython submodule SHA, ~5 min cold / ~30 s warm; (b) download from a release artefact; (c) nightly-only sandbox job. Recipe: `sage-core/src/sandbox/wasm_python.rs` module docstring. **Trap E note**: must add Python 3.13 + Windows-sandbox matrix assertions in same job — 3.13 currently unproven, Windows job builds smt+onnx but NOT sandbox+cranelift.
  - **roadmap-A9 — RNG seam for stochastic algorithms** (cgpro-recommended). `CmaEmitter` and `ContextualBandit` use `rand::rng()` with no seed parameter. Add `&mut impl Rng` overloads (`ask_with_rng`, etc.) and use `ChaCha8Rng::seed_from_u64` (NOT SmallRng/StdRng — those don't promise portable output across platforms per rand docs) in tests. Removes the structural cause of the 5-test flake spree of 2026-04-26.
  - **roadmap-A10 — Sort bandit `arm_keys` before Thompson sampling**. `HashMap` iteration order is arbitrary (Rust stdlib documents this); seeded tests can still see different arm assignments across runs. Sort `arm_keys` by `(model_id, template)` before the Thompson loop. Pairs with A9.
  - **roadmap-A11 — Three-layer test split for stochastic suites**. (1) Deterministic unit tests for mechanics (covariance update, context_mean update, cosine scoring, posterior arithmetic) with no probability thresholds; (2) seeded stochastic tests for realistic flow with fixed seeds and exact expected behavior; (3) `#[ignore]`'d empirical tests for "this usually converges over many seeds" run in scheduled/nightly job. The current cma_me + bandit suite is closer to layer 2/3 without seeds — promote them once A9 lands.
  - **roadmap-A12 — Bandit Pareto contract mismatch**. Top-of-file docs in `bandit.rs` claim a global Pareto front and constraint-aware selection, but `choose()` / `choose_contextual()` only use sampled quality + cosine similarity — cost / latency are sampled for reporting, not selection. Either fix the implementation or fix the docs. Tests can be green while the router lies about what it does.
  - **roadmap-A13 — Lockfile / constraints file for transitive deps**. Today only direct deps are pinned in pyproject.toml (e.g. `a2a-sdk[http-server]>=0.3.25,<1.0`); transitives drift on every CI install. Add either a `requirements.lock` or a `constraints.txt` to make CI reproducible, plus a separate scheduled "latest allowed dependencies" job to catch drift without making every commit hostage to upstream.
  - **roadmap-A14 rollout** — extend `output_schema` to more tools opportunistically.
  - **roadmap-A2 verification** — N=10 post-A8-Phase-3 smoke to confirm Kimi reasoning_content fix holds (budget-gated).
  - **roadmap-A3** — N=50 paired observe vs repair smoke (task #118 pending; needs API budget).
  - **roadmap-B1.c/d/e** — sage-discover / FastAPI auto-instrument / sampler tuning (each independent, days-scope).
  - **roadmap-B1.b.7** — logfire-mode Rust export (Rust spans currently Python-only when SAGE_OTEL_EXPORTER=logfire).
  - **roadmap-B1.b.9** — OTLP batch exporter with explicit tokio runtime ownership (MVP uses `with_simple_exporter`).
  - **roadmap-B2 (durable trace+replay)** — ALIRE B-series, prerequisite likely on B1 schema (now landed).
  - **roadmap-B3 (ToolPolicy capability manifest)** — ALIRE B-series, multi-week.
  - **roadmap-B4 (platform wheels)** — ALIRE B-series, packaging-only.
  - **roadmap-B9 — AgentLoop per-run immutable context** — concurrency-safe refactor; multi-week.
- **Naming disambiguation**: roadmap-Axx / roadmap-Bxx vs AUDIT.md claim Axx. Always use prefix in cross-session communication.

**Pending tasks visible to next session:**
- Task #118 pending — roadmap-A3 N=50 observe vs repair (API-budget gated).
- Task #159 pending — roadmap-A8 wasm-in-CI build.
- Task #160 pending — roadmap-A9 RNG seam.
- Task #161 pending — roadmap-A10 sort bandit arm_keys.
- Task #162 pending — roadmap-A11 three-layer test split (depends on A9).
- Task #163 pending — roadmap-A12 bandit Pareto docs/code mismatch.
- Task #164 pending — roadmap-A13 lockfile / constraints.

**Shipped in 2026-04-26 closeout cycle (25 commits, `87d30837..de640543`):**
- `87d30837` — clippy `-D warnings` debt cleared
- `3b2165b2` — sage-core/tests `cargo fmt` debt
- `98e5d988` — E0432 `wasm_python.rs:75` (sandbox+cranelift gate symmetry)
- `3f2e6678` — Windows `embedded_wasm_available` AttributeError resilience
- `a4421c38..328493a9..42264a86..40fe72e0` — ruff lint debt across 26 sage-python files
- `e5d44df3` — pin a2a-sdk `<1.0` + exclude `tools/generated_tools/` from mypy
- `4de2f59a` — fix a2a_server runtime API drift (real bugs at 0.3.x)
- `c0440bf0` — sprint3_evidence dead-code cascade
- `c8e0eff9` — bench/__main__.py rename gaia/memory_coherence locals
- `e65e2c87` — pipeline.py 8 mypy fixes (no new ignores)
- `2f20c8ec` — memory_coherence + agent_mgmt + topology/runner forensic batch
- `647faafe` — ModelRouter Tier|str + Solver Union + AgentLoop class attrs
- `a95395f3` — Optional defaults + AgentEvent kwargs + Any widening
- `2e3b0041` — StreamingLLMProvider protocol fix (`async def` → `def`) + 11 small
- `056cc4fc` — long mypy tail to 0
- `03f47c30` — CI maturin develop → build + pip install
- `fd1f9c44` — cma_me unflake (8×10 first attempt) + doc sweep
- `5a49d147` — sandbox + swebench-dep skip markers
- `a44be735` — Windows sqlite ordering tie-break + kNN skip
- `924b5996` — Obsidian vault dashboard sync
- `96d12ec0` — yaml import-untyped ignore (CI mypy diff with local; ceiling 44→45)
- `35f20e2d` — cma_me 32×20 + 0.5 threshold (second attempt; 8×10 still flaked)
- `9734a17d` — path-jail backslash normalize + first bandit budget bump
- `d9b0b659` — second bandit budget bump (in-line test in bandit.rs)
- `c7c7fcf3` — integration-smoke `--limit 5` removed + REQUESTS_CA_BUNDLE drop
- `0abee16b` — cma_me test_multiple_generations_converge same budget pattern
- `861a1076` — **bandit context-bias tests rewritten with orthogonal training** (FIRST FULL GREEN RUN)
- `9f251276` — **prod bug fix: bandit restore_arm persists context_sum/context_count** (cgpro find)
- `50fb8e4f` — consolidator timing 0.1→0.099 (Windows asyncio.sleep granularity) — confirmed plein vert
- `de640543` — roadmap.md A8 entry + closeout banner

**Methodological notes for future cycles:**
- When fixing infra (e.g. `maturin develop` → `build`+`pip install`), latent test failures hidden by collection-time ImportError will surface en masse. Budget time for the second-order cleanup.
- Stochastic test flakes are usually one of three patterns: (1) insufficient sample budget, (2) hidden symmetry / mathematical degeneracy in test setup (the bandit collinear-context bug), (3) un-seeded RNG. (2) and (3) need test-design fixes, not budget bumps.
- External review with cgpro:ask + GitHub repo URL works — caught a real prod bug (`bandit::restore_arm` persistence) that I would have missed. Use it after substantial cycles before declaring done.

**Previous session (2026-04-23 — 17 commits):** Track 2+3 closed out, wasm JIT cache shipped, ALIRE quick-wins, **pre-emission diff-context verifier implemented + fixed + validated in observe-mode smoke**. Roadmap written. See [April 23 Track 3 close-out](project_april23_track3_closeout.md).

**Shipped this session:**
- `50b4ee8` — wasm_python JIT cache (~30s cold-start → ~1s warm via Module::serialize / .cwasm)
- `74b92f5` — diff-context verifier design spec
- `c05eee0` — diff-context verifier implementation (observe mode; repair mode stubbed-to-observe pending sample accumulation)
- `711008a` — **critical parser-bug fix** caught by the observe smoke itself (parser required `diff --git` header, missed models' headerless emissions). Post-fix 2/2 patches flag correctly, zero false positives.
- `3c3fc27` — spec correction: match policy narrowed from "ratio ≥ 0.95" (would miss 14182's 0.956 hallucination) to "whitespace-only equivalence"
- `89cfb14` — observe smoke N=10 findings + artefacts
- `5efdd42` — type-ignore hygiene (setattr the gen-log sentinel; ceiling 36→41 for pre-existing drift)
- `be2d3fc`+`d87c4c0`+`cf188df` — ALIRE quick-wins (README reconciliation, subprocess-fallback sweep, SAGE_REQUIRE_WASM gate)
- `4704b51` — doc sync (README/CLAUDE/rules/Obsidian) + roadmap.md
- **ALIRE2 verification follow-up** (plan-mode + superpowers + Context7, per PROMPT.md):
  - `684bb17` — A0c: redact Tool.execute() raw traceback (info-leak fix)
  - `bf220e0` — A0d: caveat DistilBERT ONNX QualityEstimator as not-shipped in 6 docs
  - `9067be5` — A0a: restore all 10 mutated AgentLoop fields in pipeline bypass finally
  - `2bd966c` — A0b: SAGE_STRICT_GOVERNANCE=1 fail-closed mode for write-gate + verification
- **B8 closure (advisor + codex gpt-5.4-high converged on de-scope):**
  - `b9cfb1e` — align 5 pre-existing singleton-bypass tests with A0a restoration semantics (check state DURING run instead of post-run); +A0a mutation-during-run regression test
  - `4a3e0d1` — delete `RustTopologyController::evaluate_and_decide` stub; ADR-012 amendment: "Rust-primary for adaptation state + per-path primitives; orchestration Python-owned"
  - `3583c40` — roadmap B8 → Closed; audit doc records 3 post-advisor blind-spot verifications (A0c completeness, A0a mutation test, A0b emit-vs-raise ordering)
- Plus the 8-commit F3/Track-3 arc listed in the close-out note.

**Next likely directions (after B8 closure):**
1. **Accumulate observe-mode data** across opportunistic SWE-bench smokes (zero-cost, default off but opt-in in CLAUDE.md command examples). Need ≥10 flagged + ≥10 clean before repair-mode flip. See `roadmap.md` A1.
2. **Investigate 20% fast-abort rate** on SWE-bench generation (2/10 tasks aborted in < 60 s in the 2026-04-23 observe smoke). See `roadmap.md` A2.
3. **Larger ALIRE items**: OpenTelemetry GenAI spans (B1), durable trace+replay (B2), ToolPolicy capability manifest (B3), platform wheels (B4). Each is multi-week.
4. **B9 — per-run immutable context for AgentLoop** (full concurrency-safe refactor; A0a is the targeted interim for the serial-reentrant case, B9 closes the concurrent case).

If asked to "continue" / "next": reference `roadmap.md` horizons (A0 = ALIRE2 triage done, B8 = done, A1-A5 still open, B1-B4 + B9 are new high-leverage items); don't assume.

**Audit trail:** `docs/audits/2026-04-23-alire-verification.md` is the authoritative record of which ALIRE/ALIRE2 claims are confirmed / partial / refuted / orphaned on today's main, with one-commit-per-assertion mapping.

**Design specs (prior, completed):** `docs/superpowers/specs/2026-04-20-rust-first-plan-design.md` (2026-04-20), `docs/superpowers/plans/2026-04-21-semantic-quality-plan.md` (Track 2+3 done)
**ADR of the moment:** `docs/adr/ADR-013-wasm-sandbox-default.md` (2026-04-22).
**Methodology:** `docs/audits/bypass-patterns.md` — apply checklist before declaring any architectural fix done.

If asked to "continue" / "next": ask first which of the three directions above; don't assume.

## Current State (April 24, 2026)
- [April 24 — AUDIT triple remediation](project_april24_audit_triple.md) — ⭐ 27 commits, AUDIT.md §3 + §6 + AUDIT2.md §3 + §6 + AUDIT3.md all triaged; 10 fixes shipped (8 wired + 2 library-only product-gated); §6.4 advisor MERGE verdict; A19 middleware wired during §6.4 callout; 3 Codex failures documented.

## Current State (April 23, 2026)
- [April 23 — Track 3 semantic-miss close-out](project_april23_track3_closeout.md) — ⭐ 4 ship-items (F3 JSONL field + prompt hygiene + SR-missing sidecar + gen-log-default); 3.1/3.2 invalidated, 3.5 deferred. 3 distinct failure modes identified; 2 breadcrumbs pinned.

## Current State (April 22, 2026)
- [April 22 — P0.4 B + §5 sandbox flip](project_april22_sandbox_flip.md) — ⭐ 4 commits, 40-attack red-team 40/40 blocked, ADR-013, sandbox-by-default in Cargo. Rust 480→496, Python 1958→1999.

## Current State (April 21, 2026)
- [April 21 — Full session arc v13 → v17](project_april21_session_full.md) — ⭐ 5 commits, 13 tests, 4 fix mechanisms validated per-bucket. Headline stuck at 1/10 because N=10 is noise-dominated (±10pp per task flip). v17 proved Stage 4 healthy-provider fallback recovers 3/5 v13 EMPTYs. Lesson: per-bucket attribution works; total-rate requires N≥30.
- [April 21 — SWE-bench v15 first Docker-graded pass-rate](project_april21_swebench_v15.md) — 1/10 resolved (10%) after 3-fix chain (Directive #3 gating, CRLF, UTF-8) in swebench_ca_patch.py. First real pass-rate; older 70% is gen-rate.

## Current State (April 19-20, 2026)
- [April 20 — Rust-First plan COMPLETE](project_april20_rust_first_complete.md) — ⭐ all 12 items + 1 follow-up in 1 session; 13 commits; RustTopologyController Rust-primary; ADR-011 + ADR-012; 2 new bypasses found + fixed (H9 + H10)
- [April 20 (session 1) — Plan item 1.1 done](project_april20_session1_plan_1_1.md) — b7ced9d singleton max_steps scales 5/10/20 by system tier (H7) [first of 13 session-1 commits]
- [April 20 (AM) — Rust-First Plan written](project_april20_rust_first_plan.md) — plan prepared for autonomous fresh-session execution
- [April 19 (evening) — Bypass-audit sweep](project_april19_bypass_audit_evening.md) — 7 commits, G-series + H1-H6 + bypass-patterns.md methodology catalog

## Current State (April 17, 2026)
- [April 17 (evening) — Smoke v5 remediation](project_april17_evening_smoke_v5.md) — 5 commits reverting tool_choice force, fixing bench classifier, sentinel-strip, planner-injection expé, --offset CLI
- [April 17 (PM) — F7 sequence + ExoCortex repair](project_april17_session_f7_exocortex.md) — 5 commits, F7 sequence done (sink audit caught regression), 3 ExoCortex bugs fixed, rattrapage relaunched
- [April 17 (AM) — Autonomous Sprints 1-6](project_april17_autonomous_sprints.md) — 7 commits, +1200 LOC, 27 new tests, Sprint 5 execution pending
- [April 15 — Training Parked](project_april15_training_parked.md) — commit b2f59ee deletes verl/scripts/data/models (-4.3GB), training now in separate branch
- [April 9-10 Sessions](project_april9_session.md) — Unified entry point Phases 1-3 DONE, 12 commits, -739 LOC, all 9 hazards
- [April 7-8 Sessions](project_april8_session.md) — 35 commits, BCB 45.9%, MASBENCH stats, CORAL, SWE-bench prep
- [April 7 Phase A](project_april7_session.md) — Phase A complete, 2001 tests, A2A+ToolForge+17 fixes
- [April 7 Data](project_april7_session2_data.md) — BCB v1 37.2% / v2 45.2% partial, MASBENCH breadth p=0.015

## Strategic Plan Progress (April 17, 2026)
Plan file: `~/.claude/plans/witty-honking-sky.md`
- **Phase A** (solidify): ✅ COMPLETE
- **Phase B** (prove): B.1 BCB 45.9% ✅ | B.2 MASBENCH stats ✅ | B.3 Script ready | B.4 Done
- **Architecture**: Unified entry point ✅ ALL 3 PHASES COMPLETE (Phase 3 merged Apr 10)
- **Phase C** (train): ⏸ PARKED — moved to separate branch April 15
- **Phase D** (agent): ✅ CODE DONE — D.1 (ToolForge E2E), D.2 (sage_recurse), D.3 (SWE-bench 3 gaps fixed). Sprint 5 ablation execution pending (Docker + budget gate)
- **F7 routing** (advisor sequence Apr 17 PM): ✅ ALL 5 ITEMS DONE — domain-aware floor, sink audit (caught my own regression on `solver`), FrugalGPT wiring, F6+F1 audit clean, ExoCortex bugs repaired
- **ExoCortex rattrapage**: ⏳ RUNNING (b8ny7mz9x, 12 domains since 2026-03-10, ~1-3hr ETA) — runtime `search_exocortex` tool was querying a March-10 store
- **Sprint 6 decision gate**: `scripts/decide_next_phase.py` reads ablation JSON → Gate A (≥35% → v1.0 RC) / B (<20% → training revival) / C (mid → iterate)
- **Next**: SWE-bench smoke with per-fix attribution (gates on rattrapage finish so `search_exocortex` returns fresh papers)

## ⚠ SUPERSEDED
The strategic-plan "Next" above (2026-04-17 witty-honking-sky.md) was overtaken by the 2026-04-19 bypass sweep and the 2026-04-20 Rust-First Plan. The Rust-First plan **completed 2026-04-20**, P0.4 audit remediation **completed 2026-04-22** (ADR-013 sandbox-by-default), Track 2 (search-replace emission) + Track 3 (semantic-miss investigation) **closed out 2026-04-23** (see project_april23_track3_closeout.md). Next direction: see the "Next likely directions" list at the top of this file.

## User Feedback (apply in every session)
- [cgpro source of truth](feedback_cgpro_source_of_truth.md) — defer to cgpro; use --resume <alias>; debate with evidence; verify file:line claims before commit
- [No Heuristic Tuning](feedback_no_heuristic_tuning.md) — research-backed approaches only, no magic numbers
- [Rust First](feedback_rust_first.md) — performance-critical in Rust, Python for orchestration only
- [Monitor Agents](feedback_monitor_agents.md) — check background agents before deciding implementation
- [Training Issues](feedback_training_issues.md) — Windows tqdm, chat template, Unsloth, SSL cert gotchas
- [Improvement Loop](feedback_improvement_loop.md) — LEARN → ANALYZE → PLAN → EVOLVE → LEARN → LOOP
- [No Training-Leak Model Hardcodes](feedback_no_training_leak_hardcodes.md) — cards.toml + Context7 are truth; o1/o3/o4 are NOT wired in this repo
- [No LiteLLM → Use PydanticAI](feedback_no_litellm.md) — LiteLLMProvider is deprecated since 2026-04-18; default provider is sage.providers.pydantic_ai_provider.PydanticAIProvider
- [No TODO-looking strings in Rust source](feedback_no_todos_in_test_strings.md) — test fixtures containing "TODO:" substrings MUST be extracted into named module-level consts with clear intent; inlining raw "TODO: ..." strings in tests reads as unfinished work
- [Prompt anti-affordance = 0 usage](feedback_prompt_anti_affordance.md) — "optional" / "almost never the right tool" framing → model never calls it; measured 0/616 on `search_exocortex` before 2026-04-23 reframe

## Project Context
- [Shadow Traces](project_shadow_trace_findings.md) — 1090 traces: kNN 92% vs Rust 88% vs Python 44%, Rust better calibrated
- [Vision vs Reality Gap](project_vision_gap.md) — 5 pillars ~90% complete; self-adaptive promise partially real; training gap still open
- [LiteLLM Registry Lag](project_litellm_registry_lag.md) — drop_params=True insufficient for gpt-5.4 + kimi-k2.5 (registry stale), keep explicit hand-coded drops
- [OxiZ v0.2.0 — upgraded 2026-04-21](project_oxiz_v020_deferred.md) — bumped in commit 4aa29e7 after audit unparked it; QualityLabeler SMT surface unchanged

## SOTA & Research
- [SOTA March 2026](research_sota_march_2026.md) — BCB SOTA=40.5%, SAGE 37.8% budget / 45.9% tuned, The Conductor primary competitor
- [The Conductor](research_the_conductor.md) — ICLR 2026, Qwen2.5-7B GRPO, 40.0% BCB, recursive self-invocation
- [Cognitive Orchestration](research_cognitive_orchestration_pipeline.md) — 17 papers: model-per-role, dynamic topology
- [Frontier March 2026](research_frontier_march2026.md) — Dr. MAS, Dynamic Reward Weighting, HyEvo, GigaEvo

## Key Design Decisions
- S1/S2/S3 are COGNITIVE SYSTEMS (Kahneman), not pipeline stages
- Template-first topologies: 12 templates → archive → LLM synthesis → MAP-Elites → arbitrary DAGs
- Hybrid formal_solver: Rust exact solving + LLM CoT fallback (don't remove solver — fix it)
- S1 math uses hybrid solver; S1 non-math skips topology (AdaptOrch omega=1)
- kNN primary router (93.3% GT), SystemRouter (88%), heuristic DEAD (34%)
- Default provider: DeepSeek API (NOT Codex CLI — removed April 7)
- A2A uses a2a-sdk 0.3.25 (v1.0 never existed — imports fixed April 7)
- Pipeline uses Rust SystemRouter end-to-end since April 10 (commit 921cb7e)

## Archive
Training-specific memories moved to `archive/` (2026-04-17) — preserved for training branch revival. See `project_april15_training_parked.md` for the list.

## Environment
- Windows 11 Pro, MSYS2 bash, miniforge3 conda
- pip install -e from worktree `local` can override main — always reinstall after switching
- git init.defaultBranch = main (fixed April 7, was master)
- PYTHONIOENCODING=utf-8 for Windows console
- ExoCortex: `fileSearchStores/ygnsageresearch-wii7kwkqozrd`
