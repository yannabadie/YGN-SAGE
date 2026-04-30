---
title: YGN-SAGE Dashboard
type: moc
updated: 2026-04-30
---

# YGN-SAGE — Self-Adaptive Generation Engine

Agent Development Kit qui **apprend** quelle topologie multi-agent utiliser pour chaque tache.
Rust core (sage-core) + Python SDK (sage-python) + Knowledge Pipeline (sage-discover).

## Etat du projet (26 Avril 2026)

| Metrique | Valeur | Notes |
|----------|--------|-------|
| Tests Python | **2501 passing** | 8 fail + 2 error tous dans `test_e2e_*` / `test_pydantic_ai_integration.py` (pré-existants, API-key-gated). |
| Tests Rust | **501/501** | `--features smt`. cma_me::test_small_scale_convergence dé-flake (8×10 fitness evals au lieu de 4×5) le 2026-04-26. |
| **Static analysis** | **mypy 0/183, ruff clean** | mypy passé de 131→0 le 2026-04-26 sans nouveau `# type: ignore`. type:ignore ceiling 44/44 inchangé. |
| kNN Routing GT | **100%** (60/60) | exact-match override (CORAL ff41e53) |
| MASBENCH breadth | **+22pp (p=0.015)** | Seul axe statistiquement significatif |
| MASBENCH depth/horizon | +2/+4pp | Non significatifs (p>0.05) |
| MASBENCH parallel | **-6pp** | Regression — topologie nuit |
| BigCodeBench Hard | **v4 final: 45.9%** (68/148) | bypass + repair reasoner + escalation |
| **SWE-bench Lite smoke** | **v5d 4/5 (80%) v5e 3/5 — moyenne 70%** | (vs v4 1/10=10%, Apr 17). Vrais fixes multi-fichiers avec 31-62 tool_calls/task |
| EvolutionMemory | CORAL **integrated** | SQLite WAL + persistent skills, lazy init au boot |
| Providers | **7 alive + TTL exclusion Apr 18** | Re-probe 300s, recover automatiquement (OpenAI quota/Gemini brown-out) |
| Tools agent | **14** (+sage_recurse) | Tool-calling loop — **62 bash/task confirmé Apr 18** |
| Model selection | Rust affinity + **F7 floor** + **per-model routing réel Apr 18** | LiteLLMProvider honore config.model (était silencieusement ignoré depuis cards.toml) |
| Templates topo | 12 | +`formal_solver` (Apr 17), 6 sink roles maintenus en sync |
| PyPI | `pip install ygn-sage` | v0.1.0-alpha |
| A2A | a2a-sdk 0.3.25 | 6 tests, streaming + cancellation |
| ToolForge | Wire dans agent loop + **E2E** | execute_tool_call ouvre CreationTicket et retry |
| **Telemetry tool_call_count** | **Réel Apr 18** | compteur mort depuis Apr 9, re-wired bypass + topology aggregation |
| **Bench reporter** | **Classifier real/sentinel/empty Apr 18** | header ne peut plus mentir (sentinels étaient comptés comme patches) |
| **Sentinel cascade** | **Résolue Apr 18** (85282e0) | `EMPTY_STEP_SENTINEL` filtré avant d'être transmis en aval |
| Architecture | **Unified entry point Phases 1-3 mergees** | system.run() → pipeline.run() unique path (-944 LOC) |
| Training | **Parked Apr 15** | code retiré (-4.3GB), checkpoints sur HF, branche dédiée |
| ExoCortex | **3 bugs fixes Apr 17** | rattrapage terminé (manifest 132 papers, 2026-04-18) |
| F7 routing | **Apr 17 PM** — domain-aware floor | math/formal S3 → S3, autres S3 → S2 (sink classification audit-protected) |
| PRM gate | **Apr 17 PM** — domain-symetrique | Z3 PRM uniquement sur math/formal, voir [[ADR-008-PRM-Gate-Domain]] |
| tool_choice="required" | **Reverted Apr 18** (e69cb7f) | causait sentinels cascades ; diagnostic basé sur compteur mort |

## Navigation

### Architecture
- [[00-Architecture-MOC|Architecture]] — 5 piliers cognitifs, pipeline 6 etapes
- [[Pipeline|Pipeline CLASSIFY-LEARN]] — Le flux complet d'une tache
- [[Provider-Architecture|Providers]] — 7 providers, circuit breaker, failover

### Recherche
- [[00-Papers-MOC|Papers]] — 25+ papers de recherche backing le projet
- [[00-Training-MOC|Training]] — Qwen3-4B local + Nemotron-8B pod

### Resultats
- [[00-Benchmarks-MOC|Benchmarks]] — MASBENCH, BigCodeBench, Routing GT
- [[00-Issues-MOC|Issues connues]] — Problemes ouverts et fixes
- [[00-Decisions-MOC|Decisions]] — ADRs et choix architecturaux

## Verites inconfortables

> [!warning] Ce qui ne marche pas ou est incomplet
> - **MASBENCH 4/5 axes non significatifs** : seul breadth (p=0.015) prouve topology > model
> - **MASBENCH parallel -6pp** : la topologie regresse sur les taches paralleles
> - **MASBENCH robustness 0%** : 0% bare ET SAGE — a debugger
> - **BigCodeBench omega=1.3** : topologie n'est PAS le levier (ADR-006). Gains viennent du repair
> - **MiniMax** : ACTIF dans le pipeline (Apr 17, contrairement à ce qui était documenté). Bug 400 orphan-tool fixe `591d3c4`
> - **Path 6 (learned policy)** : opt-in (`SAGE_ENABLE_PATH6=1`), pas dans le pipeline par defaut
> - **sage-discover ExoCortex** : 3 bugs latents fixes Apr 17 (if/else, upload timeout, manifest persistence). Rattrapage v3 active
> - **Memory consolidation** : design documente, implementation incomplete
> - **Sandbox meta-tools** : ~~durci (regex) + structured argv allowlist~~ **résolu Apr 22** — embedded RustPython wasm sandbox est le chemin par defaut (ADR-013) ; 40/40 red-team attaques bloquées ; deny-by-default FS/net/proc/env/mem. `SAGE_UNSAFE_UNSANDBOXED` supprimé.
> - **Benchmarks** : leaderboard BigCodeBench gele avril 2025, comparaison biaisee
> - **`_tool_call_count=0` interprétation mensongère (résolu Apr 18)** : le compteur était mort (champ `PipelineContext` déclaré jamais incrémenté). v3/v4/v5a conclusions "modèles refusent outils" fausses. Avec telemetry réel (988aa99 + 0677376) : 31-62 bash/task, vraie investigation multi-fichiers
> - **Z3 PRM** : applicable uniquement à math/formal (voir [[ADR-008-PRM-Gate-Domain]]). Sur code l'AVR niveau 2 suffit
> - **Budget steps S3=20** : encore trop tight quand planner consomme 20+ tool_calls → sentinel. Prochaine direction : dynamic scaling par plateau detection

> [!info] Architecture vs Realite
> ~90% de l'architecture documentee est implementee et integree (Apr 18).
> Les 10% restants : evolution (opt-in), consolidation memoire, preuves formelles sandbox, learned prompt registry (discuté, pas implémenté), dynamic step budget.

> [!success] Shipped (Apr 28-29, **5-cycle RuntimeContracts → StateCore → RunFrame → OracleStack arc** — 18 commits + 2 doc fixes, +90 tests)
> - **Cycle 1 R0-R4 — RuntimeContracts P0** (4 commits `b66b07b..42234c4`) : (1) controller single-commit removed duplicate `evaluate_and_decide` call in `_execute_node_via_agent_loop` that mutated Rust state 2× per node ; (2) `_run_core` async-gen unifies `run/run_traced/run_stream` (was 3 divergent paths — `run_traced` ran parallel batches sequentially, `run_stream` skipped controller in parallel branch entirely) ; (3) capability-aware fallback via `ModelAssigner` replaces `get_available_providers()[0]` anti-pattern (kimi-k2.5/k2.6 incident class fixed) ; (4) sandbox fail-closed in `_execute_code_node` when `BWRAP_AVAILABLE=False` (silent fail-open on every non-Linux platform). 21 tests + 8 prod bugs surfaced and fixed via cgpro VERIFY round-trips.
> - **Cycle 2 R5 RuntimeEventLog v0** (`86fded6`, 1771+/-97) : NEW `sage/runtime/event_log/` module (6 files). 11 typed events (TaskStarted/RoutingDecision/TopologySelected/ModelAssigned/NodeStarted/NodeCompleted/ControllerDecision/Failure/Budget/StateApplied/FinalResult) with full SHA-256 envelope hash + canonical JSON. Per-run JSONL files at `<SAGE_TRACE_JSONL_DIR>/<run_id>.jsonl`. Disable-on-first-failure preserves contiguous JSONL prefix. Default OFF unless env set. ULID run_id at task entry (Crockford Base32, 26 chars). 16 acceptance tests + 1 cgpro round-trip ULID-canonical regression.
> - **R5.1 edge-binding contract** (`4742295e`, +99) : pins `graph.get_edges()` as Python-visible canonical edge-typing API ; `edges_of_type()` confirmed Rust-only (4 contract tests). Prevents R6 design from drifting toward unavailable PyO3 helper.
> - **Cycle 3 R6 StateCore v0** (`bc481588`, 1340+/-65) : NEW `sage/runtime/state/` module (StateFrame, StateDelta, StateApplyResult, EvidenceRef, StateConflict, apply_delta, apply_deltas). Behind `SAGE_STATECORE=1` (default OFF byte-identical). `_partition_incoming_edges` via `get_edges()` — legacy mode: control feeds both control+message channels ; strict mode: control-only + unknown raises ValueError + emits failure event. State delta sibling merge atomic via `apply_deltas` (sort by source_node_id, all-or-nothing). cgpro trap fixes : `_maybe_planner_injection` channel-aware in strict mode + sibling-state-frame isolation. 13 tests + cgpro round-trip NodeStarted ON/OFF asymmetry.
> - **R6.0.1 contract snapshots** (`4f38a51c`+`af26d75c`, +523) : machine-readable contract matrix at `docs/contracts/runtime-event-log.md` + golden JSONL fixtures at `tests/golden/runtime_events/`. Catches the R6-class bug where a field type-checks fine but is forbidden in legacy mode (only golden-fixture validation surfaces this). 5 contract tests.
> - **Cycle 4 R7 RunFrame v0** (`8082b203`, 1826+/-133) : NEW `sage/runtime/run_frame/` module (frame, builder, __init__). Public `RunFrame`/`NodeRunRecord`/`TopologyRunRef` + `RUN_FRAME_SCHEMA_VERSION="0"` + `_RunFrameBuilder` private hot accumulator. `node_run_id = f"{topology_epoch}:{node_id}:{attempt}"` prevents overwrite across retries/open_gate/upgrade_model/reroute. Allowlisted env capture (8 keys, no wildcard). NEW `pipeline.run_with_frame()` returns `tuple[str, RunFrame]`. Trailing `run_frame_summary` event behind `SAGE_RUN_FRAME=1` ; `parent_event_id == final_result.seq` ; best-effort emission. 19 tests + cgpro round-trip ON/OFF asymmetry.
> - **R7.0.1+R7.0.2** (`f1557a1f`, +55) : doc fix (11→12 event types) + `run_with_frame()` signature parity with `run()` (budget_usd + system_hint kwargs).
> - **Cycle 5 R9 OracleStack v0** (`276cc7d4`, 1528+/-123, 30 files) : NEW `sage/runtime/oracle/` module (errors, verdict, config, _oracles, stack, __init__ with lazy import to break circular). Hard invariant: NO bandit/MAP-Elites/online-evolution/training-memory promotion update unless `OracleVerdict.trainable=True`. Hierarchy: Exact > Tool > Formal > Spec > LLMJudge (stubbed) > Abstain. CRITICAL pipeline reorder when `SAGE_ORACLE=1`: execute → emit_final_result → oracle.evaluate → emit_oracle_verdict → record_to_memory(is_training_evidence=...) → stage_learn (gated). ALL training sinks gated, not just bandit. 17 tests including cgpro round-trip mention-vs-reassertion guard (14 negation phrases) preventing lexical-fallback failure mode.
> - **R9.0.1 evidence-starved Abstain pin** (`84b77f7e`, +65) : explicit test documenting v0 hierarchy fallthrough state (Tool/Formal None placeholders → Abstain) + roadmap "Current operational gates" section locking SAGE_ORACLE default-off until R6.1a + smoke validation.
> - **Cycle 6 R6.1a EvidenceProducers v0** (`38c0da4e`, 2990+/-23, 54 files) : NEW `sage/runtime/evidence/` module — public `RuntimeDelta` frozen+slots dataclass with `__post_init__` enforcement of (producer, delta_kind) pairs, per-kind polarity rules, payload schema validation against `PAYLOAD_ALLOWED_KEYS`, deep-freeze via `MappingProxyType`, stable `evidence_hash = sha256(canonical_json({schema_version, producer, delta_kind, polarity, source_id, payload}))` excluding `run_id`/`node_run_id`/`event_seq`/`confidence`. 6 producers: tool / test_parser (regex on pytest summary line) / diff (extends swebench verifier outcomes) / formal (obligation_proved/refuted/unknown — refuses raw SAT/UNSAT, requires obligation_id+verifier_id+encoding) / code_node (structured return validation) / planner (assumption_invalidated). Tool/Formal/Spec oracles promoted from v0 None placeholders to v1 structured-evidence consumption: Tool partial = passed/total fraction (Q5.a deterministic counts only), Formal Q5.b obligation semantics, Spec dual-source (state_frames + assumption_invalidated deltas). 3 LIVE emission points (agent_loop_execution, swebench_diff_verifier, swebench_patch_repair, all `SAGE_ORACLE=1`-gated) + 3 scaffolded for cycle-7. NEW `RunFrame.runtime_deltas: tuple[RuntimeDelta, ...]` field (Q4.a name lock). 37 acceptance tests + 22 round-trip JSON fixture pairs covering all 6 producer × delta-kind matrices.
> - **Methodology validated** : every cycle followed cgpro DESIGN (locked spec) → codex IMPLEMENT (gpt-5.5 xhigh full-auto) → claude verify-local (TDD via `git stash --keep-index`) → cgpro VERIFY → SHIP. 5 cgpro VERIFY round-trips caught 5 contract leaks before SHIP (R3 `_remaining_budget_usd` cost_tracker fall-through, R5 ULID fallback canonicality, R6 NodeStarted ON/OFF schema asymmetry, R7 builder ownership concurrency-safety + 11→12 doc fix, R9 spec oracle mention-vs-reassertion).
> - **Project totals (post-cycle-6)**: 19 commits + 2 doc fixes + ADR-014..019 pushed, **~12300 LOC** (cycle 6 +2990), **2484 regression tests** (+37 R6.1a vs cycle-5 2447), **mypy 0 errors / 204 source files**, ruff clean, **522 Rust tests** unchanged (R6.1a pure Python).
> - **Strategic feature flags** :
>   - `SAGE_STATECORE=1` (R6 channel separation) — default OFF
>   - `SAGE_RUN_FRAME=1` (R7 typed run frame + trailing diagnostic) — default OFF
>   - **`SAGE_ORACLE`** (R9 training gate) — **DEFAULT-ON since cycle-7 flip 2026-04-29 (`128e1b89`)**. Unset = ON; kill-switch via `SAGE_ORACLE=0|false|off|no|disable|disabled` (case-insensitive; `disable`/`disabled` added in cycle-7 VERIFY round-1, commit `87daf89a`). Validated N=5 unset (5/5 oracle_verdicts emitted) + N=2 kill-switch (0 oracle_verdicts) — commits `a5f916ea` + `8b4b34b6`. T4 forced `controller_decision.payload` is allowlist-only since round-1 (commit `87daf89a` writer + `f3a89631` docs).
>   - `SAGE_TRACE_JSONL_DIR=<path>` (R5 durable JSONL sink) — Off when unset.
>   - `SAGE_BENCH_ORACLE_SEAM=1` (Path E synchronous-eval bench feedback) — default OFF.
>   - `SAGE_BOOT_BYPASS_EPOCH_GUARD=1` (A14 forensic load-only bypass) — normal boot/load requires `posterior_epoch.json` epoch=1 + `topology_state_manifest.json` SHA-256/size binding for all A14 topology state files; save hard-fails under bypass.

> [!success] Shipped (Apr 29 evening, **cycle 7 default-on flip + 6 commits**)
> - **`162e82ea` Phase 2 BCB-Hard N=50 + Phase 2bis Docker re-grade** : 30%/32% calibrated (internal/official), 49/50 = 98% per-task agreement, all 9 cycle-7 pass criteria green, validator PASS.
> - **`f6711385` _exact_oracle raw reason leak fix (cgpro PUSH BACK closure)** : raw `bench_result["reason"]` was leaking into `oracle_verdict.reason_codes` (unittest tracebacks). Now hashed to SHA-256 and stored in `EvidenceRef.evidence_hash`; reason_codes carry only structured tags. Validator extended to scan full event document + phrase-scan for Traceback/AssertionError/etc. BCB seam evaluator emits `reason_code` enum + `reason_sha256` instead of raw `reason` string. 3 regression tests pin the contract.
> - **`f9305d74` post-leak-fix N=5 smoke** : 0 raw leaks across 106 events. Closure of cgpro PUSH BACK validated end-to-end.
> - **`128e1b89` cycle-7 default-on flip** : centralized predicate `sage/runtime/oracle/env.py` `oracle_enabled()` (avoids 8-site hand-edit drift). All 8 call sites refactored. Reason-code regex sanitization added (`exact_reason_code_rejected` sentinel for non-conforming bench inputs). 13 env-predicate regression tests + 2 reason-code sanitization tests + module-wide `SAGE_ORACLE=0` autouse fixture for legacy `test_pipeline.py` tests.
> - **`a5f916ea` N=5 SAGE_ORACLE UNSET smoke** : 5/5 oracle_verdicts emitted with no SAGE_ORACLE env var ⇒ default-on works.
> - **`8b4b34b6` N=2 SAGE_ORACLE=0 kill-switch smoke** : 0 oracle_verdicts + 0 evidence_deltas + run_frame_summary `oracle_verdict=None` ⇒ kill-switch silences oracle.
> - **A14 reset operation** (DONE earlier, commit `8c8a1c27`): 14 bandit_arms + 5 MAP-Elites entries → contaminated_pre_a14 archive; epoch=1 marker; SHA-256 audit dump.
> - **A14 guard round-2 closure** (cycle-8 step 2, `6b2ebcbe` + closure changeset): `topology_state_manifest.json` binds active topology state bytes to epoch=1; DB-only restore over a valid epoch marker now fails closed.
> - **Operational implication** : all future runs default to OracleStack training gate. Bandit / MAP-Elites / online-evolution / training-memory ONLY update on `verdict.trainable=True`. Posterior epoch=1 (post A14 reset). Operator escape hatch: `SAGE_ORACLE=0|false|off|no|disable|disabled` (case-insensitive).

> [!success] Shipped (Apr 26, **CI debt closeout — 14 commits**, vert après une semaine rouge)
> - **mypy 131→0** sans nouveau `# type: ignore` — fix forensique par root-cause: `protocols/a2a_server.py` API drift réel à 0.3.x (`context.message` vs `context.request.message`, `event_queue.enqueue_event` vs `.put`, `AgentEvent(type=, step=, timestamp=, meta=)` vs kwargs inexistants — bugs runtime que les tests ne révélaient pas) ; `bench/sprint3_evidence.py:86` dead-code qui poisonnait 12 attribute accesses ; `StreamingLLMProvider` protocol method un-`async`'d (close `Coroutine has no __aiter__` cascade) ; AgentLoop class attrs `toolforge`/`evolution_memory` ; ~30 small structural fixes (Optional defaults, Solver Union annotation, AgentEvent constructor).
> - **a2a-sdk pinned `>=0.3.25,<1.0`** — pyproject n'avait pas d'upper bound, CI résolvait silencieusement vers 1.0.2 (1.0 est une vraie migration protocol/runtime, pas du type drift). Validé par GPT-5.5 Pro + advisor avant action.
> - **CI maturin recipe** — `develop` requiert un venv que setup-python@v6 ne crée pas → bascule vers `maturin build --release --out target/wheels` + `pip install` du wheel. Ajouté à python-sage Linux + windows jobs.
> - **`tools/generated_tools/` exclu de mypy** — sandbox-eval templates 1-2 lignes avec globals `json`/`args` injectés au runtime, pas du Python standalone.
> - **Tests pré-existants exposés par le fix maturin** : sandbox-dépendants skip si pas de `rustpython.wasm` bundle (test_meta_security, test_tool_creation), swebench-dépendants skip si package absent (test_swebench_ca_patch), `episodic.py:list_all` order tie-break par rowid pour résoudre l'instabilité Windows clock 15ms, kNN `assess_complexity` skip win32 (OOD threshold sensible à l'embedder fixture).
> - **Doc sweep** : README badge 2339→2501, CLAUDE.md Current State refresh 2026-04-26, .claude/rules/architecture.md + development.md mypy/ruff status, roadmap.md banner closeout, MEMORY.md gate update, ce dashboard.

> [!success] Shipped (Apr 23, **Track 3 close-out + verifier + wasm JIT cache + ALIRE quick-wins** — 17 commits)
> - **Track 2+3 close-out** — F3 JSONL field (`cb03773`), F2 diagnosis note (`ed4bf0e`+`60241c1`), prompt hygiene (`29987bc`), SR-missing sidecar (`2793a74`), gen-log-by-default (`9ec3dfd`), Track 3 close-out (`0b94877`+`d4d9c01`). Invalidations : sub-tasks 3.1 & 3.2 (tracers DO read test files ; prompt ne peut pas fixer les 3 modes orthogonaux de semantic-miss).
> - **Wasm-python JIT cache** (`50b4ee8`) — cold-start ~30s → ~1s via `Module::serialize` / `.cwasm`. Cache key = SHA256(wasmtime-version || RUSTPYTHON_WASM). Self-healing sur corruption. Opt-out `SAGE_WASM_CACHE_DISABLE=1`.
> - **Pre-emission diff-context verifier** (`c05eee0` + fix `711008a`) — observe mode opt-in via `SAGE_DIFF_VERIFIER_MODE=observe`. Annote `_diff_verifier_mismatches` dans predictions.jsonl. Smoke Apr 23 N=10 : 2/2 patches flaggés content_mismatch post-fix, zéro faux positif. Repair mode spec'd mais pas shipped — downgrade warn→observe.
> - **ALIRE audit quick-wins** — README reconcilié (`be2d3fc`), subprocess-fallback sweep post-ADR-013 (`d87c4c0`), `SAGE_REQUIRE_WASM=1` build-time gate (`cf188df`).
> - **Type-ignore hygiene** (`5efdd42`) — setattr sur le sentinel gen-log, ceiling bumpé 36→41 pour le drift pré-existant.

> [!success] Recemment fixe (Apr 22-23, **6 commits P0.4 B + §5 flip complet** — voir [[ADR-013-Wasm-Sandbox-Default]])
> - **P0.4 A+C+D** (511ac87) — spec reframe + `test_double_opt_in_structural_invariants` + red-team plan
> - **P0.4 B embedded wasm** (fe142e2) — RustPython 0.5.0 wasm32-wasip1 + freeze-stdlib (37 MB), wasmtime 43 + cranelift JIT, câblé dans `execute_raw`. deny-by-default WASI-p1 : no fs/net/proc/env/stdio inheritance
> - **Red-team corpus** (cf12ea4) — 40/40 attaques bloquées en 138s ; 0 SENTINEL leak ; 0 panic wasmtime. Bugs révélés + fix : epoch deadline monotonique (AtomicU64), StoreLimits memory cap (256 MiB)
> - **§5 flip sandbox-par-défaut** (c2113d8) — `validate_and_execute` tourne sandboxé par défaut ; `SAGE_UNSAFE_UNSANDBOXED` supprimé ; `sandbox`+`cranelift`+`tool-executor` dans les features Cargo par défaut ; `create_python_tool` passe à `validate_and_execute` (fixe régression P0.3) ; ADR-013 publiée
> - **SWE-bench parity smoke N=10** (81acc2e) — bash 3/10 vs typed-only 4/10 patches sur le même slice ; critère fonctionnel §5 met. Gap 10pp dans le bruit (variance ±10pp/tâche).
> - **`dangerous_tools=False` default** (Apr 23) — `execute_bash` plus registered au boot par défaut ; `SAGE_DANGEROUS_TOOLS=1` reste escape-hatch pour bench paths qui veulent shell brut.
> - **Follow-up** : Docker-eval sur les N=10 déjà générés pour vérifier que les patches typed-only passent les tests autant que les bash (non bloquant pour le flip).

> [!success] Recemment fixe (Apr 18, **13 commits plumbing** + 5 commits Apr 17)
> - **Bench reporter classifier** (4a33faa) — real/sentinel/empty, le header cesse de mentir
> - **Sentinel cascade strip** (85282e0) — `EMPTY_STEP_SENTINEL` filtré en predecessor context
> - **Telemetry re-wiring** (988aa99 bypass + 0677376 topology) — `_tool_call_count` cesse d'être mort
> - **Per-model routing réel** (c9ff902) — LiteLLMProvider honore `config.model` (était ignoré)
> - **Provider inference intelligente** (4a2c038 + f754535) — "unknown" → inférence par model_id ; qwen/* → openrouter
> - **Quota-aware + TTL health_check** (fe66d52 + 3148667) — 429 insufficient_quota = DEAD ; **re-probe 300s pour recovery auto** (pas permanent)
> - **`--offset` CLI** (97fc64f) — mesure hygiénique hors tâches mémorisées
> - **Planner injection opt-in** (ea09dd6) — `SAGE_PLANNER_INJECTION=1`, MASS arXiv 2502.02533
> - **tool_choice required reverted** (e69cb7f) — diagnostic basé sur compteur mort, causait sentinels
> - **Résultats v5d/v5e** : 4/5 puis 3/5 real patches (moyenne 70%), 0 sentinels, 62 bash/task (vs v4: 1/10=10%)

> [!success] Recemment fixe (Apr 9 → 17, 50+ commits, voir [[Changelog-Apr9-17]])
> - **Unified entry point Phases 1-3 mergees** (Apr 9-10) — single execution path, -944 LOC
> - **Training pipeline retire** (Apr 15, b2f59ee) — code sur branche dediee, checkpoints HF
> - **Sprints 1-6 autonomes** (Apr 17 AM) — Sprint 5 ablation scaffolding, Sprint 6 decision gate
> - **CORAL integration** (cherry-picks Apr 17) — kNN exact-match (100% GT), TopologyController, S2+sequential bypass removed
> - **F7 routing** (Apr 17 PM) — role-aware tier promotion, sink audit (-formal_solver regression), FrugalGPT cascade wiring
> - **PRM domain gate** (Apr 17 PM) — Z3 PRM uniquement sur math/formal (smoke v3 : 0 → 17 CEGAR, 2/3 patches debloques)
> - **ExoCortex bugs** (Apr 17 PM) — 3 bugs reels + 1 perf, rattrapage terminé (manifest 132 papers)
