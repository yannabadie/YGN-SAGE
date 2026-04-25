# YGN-SAGE Project Memory

## User
- [User Profile](user_profile.md) — Yann Abadie, GIE AD BRIVE, Rust+Python dev, French-speaking, RTX 3500 Ada

## ⭐ Active direction (read this FIRST when fresh session resumes)

**Most recent work (2026-04-24 — 27 commits):** Full AUDIT{,2,3} remediation + PROMPT.md post-hoc reconstruction. 10 security/safety tickets shipped (8/10 fully wired, 2/10 library-only awaiting product decisions). §6.4 advisor verdict = MERGE-AVEC-RÉSERVES (defensible). A19 upgraded ⚠️→✅ during §6.4 after advisor callout. See [April 24 AUDIT triple](project_april24_audit_triple.md).

**Gate for next "continue":**
- If asked to continue A13 (prompt-injection) or A14 (ToolResult validation) wiring: these are **product-decision gated** (A13: log-vs-refuse; A14: which tools get schemas). Ask user before implementing.
- If asked to continue roadmap work: A1 (observe-mode accumulation), A2 (fast-abort investigation), A3 (N=50 paired smoke, task #118 in_progress) are all non-product-gated.

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
