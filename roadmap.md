# YGN-SAGE roadmap

**Last updated:** 2026-05-02
**Scope:** forward-looking work. Living backlog grouped by expected
time horizon; priorities inside each horizon ordered by
impact-over-effort.

## ⭐ Current operational gates (2026-05-02 — Cycle-9 A2 smoke RUNNING)

**Cycle-8 closed at `86681ac8`** (closeout commit). Stack: `78565578` R6.1c → `9944674e` R6.1c round-1 fixes → `49648263` R6.1c round-2 disclosure → `6b2ebcbe` A14 round-1 → `f9521616` A14 round-2 fixes → `86681ac8` closeout (status JSON + 2 contract docs + directive #9 + ALIRE3 advisory + cgpro architect review locks).

**Live test counts** (canonical at `docs/status/current.json`): **2902 Python collected** / **549 Rust listed** / **100 sage-discover**. mypy 0 / ruff clean.

**Cycle-9 completed work (2026-05-02 HEAD `24f97f3c`):**
- A14b Stage-0 attribution closure: `route_integrated()` + `record_outcome_checked()` + cancel-on-skip/refuse (commits `34e42ea5` → `6f23eea4`). cgpro-APPROVED.
- T2 memory write paths: `create_node_agent_loop()` now injects episodic/semantic/memory_agent/causal backends. `gate_rejected` telemetry added. YGN-16 tests (9/9) integrated at `886597de`.
- swebench_patch_repair.py: two-stage repair pipeline (programmatic + LLM). 18/18 tests pass.
- deepseek-chat → deepseek-v4-flash migration: models.toml + llm/router.py + masbench.py (sunset 2026-07-24).
- **A14 reset (2026-05-02):** pre-A14 bandit/MAP-Elites state moved to `~/.sage/contaminated/pre_a14_20260502`. Clean epoch=1 for causal learning. Audit dump at `.tmp/a14_reset_20260502/`.

**Cycle-9 GATE — A2 smoke running:**
`python -m sage.bench --type ablation --limit 10 --tier budget --output docs/benchmarks/2026-05-02-a2-ablation-bcb-hard-n10-deepseek-v4-flash.json`
Gate: **≥4/10 (40%)** → proceed to A3 N=50 / **3/10 (30%)** → diagnostic / **≤2/10 (<25%)** → rollback/stop. Verify: oracle_verdict.trainable=true, bandit.record_outcome not cancelled, memory.archive.grow, evolution.should_evolve logged.

**Strategic positioning locked (cgpro 2026-05-02)**: Cycle-9 = **budget tier paired ablation** (deepseek-v4-flash). Premium frontier = Cycle-12+. SWE-bench-Live = Cycle-11 reproducibility lane. NO model hardcoding — baseline = "frontier current at eval date". A31 (S-MMU cold-start) + A32-followup (AdaptiveMutator) = Tier 2. Symphony infra (feat/symphony-dev-orchestration) deferred to ops-only PR after A2 gate.

**Earlier gates still locked**: R0..R7 + R9 + R6.1a (~12000+ LOC), Gate D PASS, Path E step 3 PASS, A14 reset 2026-04-29 (`~/.sage/contaminated_pre_a14_20260429/`), **A14 reset 2026-05-02** (pre-A14 state without `topology_state_manifest.json` moved to `~/.sage/contaminated/pre_a14_20260502/`, clean epoch=1 for A2 causal measurement, commit `24f97f3c`). **A14 fail-closed guard active** since cycle-8 step 2 (`6b2ebcbe + f9521616`) — Rust + Python defense-in-depth, `topology_state_manifest.json` provenance binding (SHA-256 over A14 state files).

Strategic runtime flags after cycle 7:

- **`SAGE_ORACLE`** (R9 + R6.1a) — OracleStack training gate. **DEFAULT-ON since cycle-7 flip 2026-04-29** (commit `128e1b89`). Unset = ON. Kill-switch (operator escape hatch): `SAGE_ORACLE=0|false|off|no|disable|disabled` (case-insensitive; `disable`/`disabled` added in cycle-7 VERIFY round-1, commit `87daf89a`). Centralized predicate in `sage/runtime/oracle/env.py` `oracle_enabled()`. Validated by N=5 unset smoke (5/5 oracle_verdicts emitted) + N=2 kill-switch smoke (0 oracle_verdicts). T4 forced `controller_decision.payload` is allowlist-only since round-1 (no free-form `reason` leak).
- **`SAGE_STATECORE=1`** (R6) — opt-in. Control/Message/State edge-channel separation.
- **`SAGE_RUN_FRAME=1`** (R7) — opt-in. Typed RunFrame trailing diagnostic.
- **`SAGE_TRACE_JSONL_DIR=<path>`** (R5) — opt-in durable JSONL sink.
- **A14 posterior epoch guard** (cycle-8 step 2, `6b2ebcbe` + round-2 closure) — boot/load fail-closed unless A14 topology state files match both `posterior_epoch.json` epoch=1 and `topology_state_manifest.json` SHA-256/size provenance binding. `_CONTAMINATED.json` remains a poison pill. `SAGE_BOOT_BYPASS_EPOCH_GUARD=1` is load-only forensic bypass; save hard-fails under bypass.

Plus the new bench seam (R6.1a Path E):
- **`SAGE_BENCH_ORACLE_SEAM=1`** — BCB/synchronous-eval benches feed `bench_result["passed"]` to the OracleStack BEFORE final_result/oracle_verdict/learn so Exact verdicts fire on the live trace.

Cycle sequencing post-Path-E-step3 (cgpro 2026-04-29 lock):

```
Cycle 6 = R6.1a deterministic delta producers (DONE 38c0da4e..426dfb6f)
Path E  = bench-result feedback seam (DONE c1a45213) + step 3 BCB-N10 PASS (e74289fd)
A14     = reset to empty, epoch=1 (DONE 2026-04-29) + fail-closed manifest binding
          (cycle-8 step 2 round-2 closure, 2026-04-30, commit pending after verify)
Cycle 7 = T1-T5 diagnostic tickets (below) + tonight's BCB-Hard N=50 evidence run +
          official Docker re-grade + SAGE_ORACLE default-on flip + post-flip smoke
Cycle 8+ = R6.1b (pytest parser anchoring + planner producer live + ToolOracle
           incidental-fatal cleanup), R6.1c (per-(producer, delta_kind) payload schema),
           T2-T5 architecture activation work
```

## Cycle 7 diagnostic tickets (cgpro 2026-04-29 Path E postmortem)

Path E step 3 BCB-Hard N=10 trace analysis revealed 4 architecture-activation gaps + 1 provider-skew observation. cgpro confirmed these are NOT training-safety blockers (default-on can ship from clean A14-reset epoch with oracle gate intact) — they are cycle-7 follow-up tickets.

- **roadmap-T1 topology engine-first diagnostic flags** (codex BG `bj9duspc0`): `_stage_select_topology` short-circuits to `dag_template` path before `DynamicTopologyEngine.generate()` ever runs ⇒ monotonic `sequential` template on BCB cold-start. Add `SAGE_TOPOLOGY_SKIP_DAG_TEMPLATE=1`, `SAGE_TOPOLOGY_FORCE_ENGINE=1`, `SAGE_TOPOLOGY_LOG_ALL_CANDIDATES=1` env flags. Default OFF, byte-identical legacy.
- **roadmap-T2 memory backend wiring + telemetry split**: `content_too_short` skip reason mislabels — root cause is `create_node_agent_loop()` not injecting `episodic_memory`/`semantic_memory`/`memory_agent` into node loops. Telemetry split: `memory_backend_unwired` vs `gate_rejected` vs `content_too_short`. Wire backends BEFORE tuning thresholds.
- **roadmap-T3 BCB tools wiring + validate-before-final instruction**: BCB prompt doesn't tell agents to call validation tool before final answer. `synthesizer` role memory-only, no `sandbox_manager`/`tool_executor` injection. Add code-domain instruction + sandbox wiring + emit `RuntimeDelta` (test_parser/tool_execution) from validation step.
- **roadmap-T4 unredact controller_decision payload safe fields** (codex BG): expose `quality_score, quality_source, threshold_band, action, reason_code` in event payload. Hashes/scores only, no raw output. Already in cgpro spec for the diagnostic round.
- **roadmap-T5 model assigner top-3 candidate logging** (codex BG): `SAGE_ASSIGNER_LOG_TOP3=1` logs per-node top-3 candidates with score components (affinity/domain/cost_norm/hint_bonus/diversity_penalty). Diagnose 75% Google skew without hard-overriding Stage 4.
- **roadmap-T6 SAGE_BENCH_DISABLE_REPAIR flag** (codex BG): bypass repair/escalation branch for clean first-attempt measurement on tonight's BCB N=50 evidence run.

The cycle-7 default-on gate post-A14-reset was: T1+T6 shipped (for Phase 2 evidence run) + the BCB-Hard N=50 evidence run with `SAGE_BENCH_DISABLE_REPAIR=1` + `SAGE_RUN_FRAME=1` + `SAGE_BENCH_ORACLE_SEAM=1` (and `SAGE_ORACLE` **unset** for the default-on path; the cycle-7 flip itself is what enabled the oracle path). Run produced ≥1 trainable Exact verdict path + ≥1 trainable formal/tool path + 0 raw leaks + official Docker re-grade per-task agreement (49/50). Headline 30% internal / 32% Docker pass@1 — see `docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50-validation.md`.



**2026-04-26 CI debt closeout** (~24 commits, including a real prod-bug fix
flagged by an external review pass via `cgpro`: `bandit::restore_arm` was
silently dropping `context_sum` / `context_count` on every save/load, so
the contextual cosine-bias channel reset to zero on every restart in
production. SQLite schema extended with `context_sum TEXT DEFAULT '[]'` /
`context_count INTEGER DEFAULT 0` + ALTER-TABLE migration + serde_json
serialization; new `test_context_bias_survives_save_load` regression
test pins the contract).

Follow-up not in this cycle:
- **roadmap-A8 — Build rustpython.wasm in CI**. The embedded sandbox
  artefact isn't built on GitHub-hosted runners; sandbox-dependent
  tests (`test_meta_security::test_created_tool_executes_in_sandbox`,
  `tests/integration/test_tool_creation.py`, parts of
  `test_swebench_ca_patch.py`) skip via `embedded_wasm_available()`.
  Means the sandbox has zero regression coverage in CI today. Options:
  (a) dedicated `build-wasm-sandbox` job with GitHub Actions cache
  keyed on RustPython submodule SHA (~5 min cold / ~30s warm); (b)
  download from a release artefact; (c) nightly-only sandbox job.
  Recipe: see `sage-core/src/sandbox/wasm_python.rs` module docstring.

**Earlier closeout work** (~13 commits): brought CI back to green
after the 2026-04-21 AUDIT cycle had left it red on every commit.
Distinct fix waves: clippy `-D warnings`, sage-core/tests `cargo fmt`,
E0432 in `wasm_python.rs:75` (sandbox+cranelift gate), Windows
`embedded_wasm_available` runtime-attribute resilience, ruff lint debt
across 26 files, **mypy 131→0 errors** (a2a-sdk pinned `<1.0`,
`tools/generated_tools/` excluded, real a2a_server.py runtime-API
bugs at 0.3.x fixed, sprint3_evidence.py async cascade root removed,
AgentLoop class attrs for `toolforge`/`evolution_memory`,
StreamingLLMProvider protocol method un-`async`'d, ~30 small
structural fixes), maturin `develop`→`build`+`pip install` CI recipe
(setup-python@v6 has no venv). Type:ignore ceiling 44/44 unchanged.
ruff clean. 2501 Python tests passing.

Reference frames:
- `ALIRE.md` — external audit of commit `44a157c` (2026-04-22). Several of
  its critical items landed this week (dangerous_tools flip, subprocess-
  fallback docs sweep, ADR-013); the remaining items are in this roadmap.
- `docs/benchmarks/2026-04-23-track3-closeout.md` — the Track 3 breadcrumbs
  pinned there map to items below.
- `docs/superpowers/specs/2026-04-23-diff-context-verifier-design.md` —
  spec that drove the 2026-04-23 observe-mode ship; repair-mode
  implementation is in the short-term section below.

---

## Horizon A — short-term (next 1-2 weeks)

### A0 — ALIRE2 triage remediations (2026-04-23)

The 2026-04-23 ALIRE2 verification (`docs/audits/2026-04-23-alire-verification.md`)
confirmed 4 high-severity live gaps on main. All 4 were shipped the
same session as small, reviewable commits — kept here so the audit
trail is explicit.

| Item | Shipped commit | Note |
|---|---|---|
| **A0a** Restore all 10 mutated AgentLoop fields in the pipeline bypass `finally`. Targeted fix; full refactor = B9. | `9067be5` | `pipeline.py` snapshots `_orig_bypass_state` dict pre-mutation; finally restores all 10 (happy + exception). +2 regression tests. |
| **A0b** `SAGE_STRICT_GOVERNANCE=1` turns the write-gate-init-failure and verification-failed paths into hard aborts (emit `EXECUTE_HALTED_UNVERIFIED`). Default off; byte-identical. | `2bd966c` | `pipeline.py` + `_is_strict_governance()` helper. +16 regression tests. |
| **A0c** Redact raw traceback from `Tool.execute()` model-visible output (info-leak fix, ALIRE2 §6). | `684bb17` | `tools/base.py` returns type:message; `log.exception` keeps operator-side visibility. +3 regression tests. |
| **A0d** Caveat the "DistilBERT ONNX QualityEstimator" claim in 6 docs — active backend is Z3 QualityLabeler + abstention; ONNX artefact NOT shipped. | `bf220e0` | README.md, pillars.md, results.md, methodology.md, Obsidian Papers/CascadeRouting.md + Architecture/Pipeline.md. |

### A1. Accumulate observe-mode data across opportunistic SWE-bench smokes
**Why:** repair-mode flip needs ≥10 flagged + ≥10 clean observations to
discriminate signal from false positives at a meaningful SE. Dedicated
smokes cost $5-10 apiece; instead, **every** future SWE-bench run opts
into observe mode passively.

**Progress (2026-04-24, N=20 cumulative):**

| Smoke | N | PATCH | EMPTY | Flagged | Verifier-missed fails |
|---|---|---|---|---|---|
| 2026-04-23 | 10 | 2 | 8 | 2/2 | 0 |
| 2026-04-24 | 10 | 2 | 8 | 1/2 | 1 (malformed hunk header) |
| **Total** | **20** | **4** | **16** | **3** | **1** |

**New finding (2026-04-24):** astropy-6938 emitted a patch that Docker
rejected with `malformed patch at line 15` because the hunk header's
old/new line counts don't match the diff body. The current verifier
walks context lines INSIDE well-formed hunks — it can't run the content
check if `patch` aborts at header parse. Adding hunk-header arithmetic
(old-line-count vs number of ` `/`-` lines; new-line-count vs ` `/`+`)
as a pre-check would close this blind spot with ~20 LOC.

**Concrete action:** update smoke invocations in docs + scripts (already
done in `.claude/rules/development.md` + `CLAUDE.md`) so new runs include
`SAGE_DIFF_VERIFIER_MODE=observe`. Two more passive smokes → N=40 with
~8 PATCHes. Consider verifier extension for the second failure class.

**Done when:** ≥10 PATCH entries observed AND at least one Docker-clean
PATCH exists in the sample. **Currently stuck at 0 clean PATCHes after
N=20** — the coder-role model emits unapplicable diffs deterministically.
This is an orthogonal concern to repair-mode coverage and may need its
own sub-item (see A6 below).

### A2. Fast-abort root cause — ✅ DIAGNOSED 2026-04-24 (Kimi k2.5 tool-call)
**Why:** astropy-14182 (~58 s) and astropy-7746 (~72 s) both aborted
with 0-ish tool calls in the 2026-04-23 AND 2026-04-24 smokes —
deterministic (N=2/2). Investigation via gen-log dive:

**Root cause chain (both tasks, identical signature):**
1. Rust `ModelAssigner` filters `needs_tools && !card.supports_tools`
   (`model_assigner.rs:289`). `needs_tools` comes from
   `node.required_capabilities.contains("tools")`.
2. In `sage-core/src/topology/templates.rs`, only the `coder` node
   declares `"tools"`; `planner`, `worker_N`, `synthesizer`, `source`,
   `mixer`, `dispatcher` etc. use `["text_processing"]` or
   `["reasoning"]` without `"tools"`.
3. Rust assigner picks kimi-k2.5 for a non-coder node (scoring +
   diversity penalty + affinity).
4. Python `AgentLoop` factory grants tools to EVERY node regardless
   of declared capabilities at execution time.
5. Kimi's Moonshot API rejects the 4th+ assistant message carrying
   tool_calls with `HTTP 400 "thinking is enabled but
   reasoning_content is missing in assistant tool call message at
   index 3"`. `cards.toml:598-609` already documents this (F9 audit
   fix marked `supports_tools=false`) — but the filter only fires
   when `needs_tools=true` from the node caps.
6. Stage 4 multi-agent fails → single-agent fallback to
   gemini-3.1-flash-lite-preview returns empty content → task
   aborts as EMPTY.

**Fix ticketed as A7** (next entry); the diagnosis itself is done.

**Impact:** closes the 20% silent-budget-leak once A7 lands. All
2026-04-23 and 2026-04-24 smokes' per-task signal was masked by this
bug.

### A7. Template capability hygiene (fix for A2) — ✅ SHIPPED 2026-04-24
**Why:** A2's Option A. Prevent kimi-k2.5 from being assigned to
tool-using nodes by declaring `"tools"` in every non-sink template
role's `required_capabilities`.

**Action taken:**
- Added `"tools"` to 23 nodes across all 12 templates in
  `sage-core/src/topology/templates.rs`. Each edit carries an
  inline comment pointing to A7's rationale and the model_assigner
  filter site.
- Sink-prompted roles (SINK_NODE_PROMPT) deliberately kept
  tool-free:
  - `synthesizer` (sequential, brainstorming)
  - `aggregator` (parallel, horizon_pipeline, parallel_fanout)
  - `mixer` (selfmoa)
  - `judge` (debate)
  - `verifier` in `robust` only (AVR's `verifier` gets `"tools"` —
    it's not sink-prompted)
  - `solver` (formal_solver — deterministic Rust, not LLM)
- New regression test `test_a7_tool_capability_hygiene_all_templates`
  in `templates.rs`. Iterates every template; asserts non-sink
  nodes declare `"tools"` AND sink nodes do NOT. Fires with a
  pointer to A7's rationale if a future template adds a role
  without `"tools"` or regresses an existing one.

**Validation:** Rust `cargo test --features smt --lib topology::templates`
→ 20/20 PASS (+1 new A7 test). The Rust-level filter at
`model_assigner.rs:289` (`needs_tools && !card.supports_tools`)
now fires on every template-built non-sink node, excluding kimi-k2.5
from the candidate pool. No Python-side change needed — the existing
`cards.toml:610 supports_tools = false` on kimi is the other half
of the contract.

**Option B (correct long-term):** make AgentLoop honour the node's
`required_capabilities` — tool-free roles get a tool-free agent
variant. Matches F9's original intent. Wider refactor
(agent_loop_factory + phases/act.py). Do at B9 (AgentLoop
concurrency refactor) — until then, declaring `"tools"` universally
(except SINK_NODE_PROMPT roles) is the pragmatic close-out.

**Expected impact:** closes A2's 20% silent-budget-leak on
template-built multi-agent topologies. Next opportunistic observe
smoke should see astropy-14182 / astropy-7746 no longer fast-abort.
A1's patch-rate gate becomes achievable since every task gets real
tool-call attempts rather than HTTP 400s on turn 4.

**Empirical verification (2026-04-24, N=6 gen-only smoke on the same
task IDs as the pre-A7 smoke).** See
`docs/benchmarks/2026-04-24-a7-verification/findings.md`.

- **Zero kimi-k2.5 HTTP 400 occurrences** in the 82 KB gen log (vs 2
  in yesterday's pre-A7 smoke).
- **astropy-7746** (canonical fast-abort): `EMPTY 72 s` → `PATCH 634 chars` ✅
- **astropy-12907**: `EMPTY 266 s` → `PATCH 4879 chars` ✅
- **astropy-14995**: `EMPTY 174 s` → `PATCH 474 chars` ✅
- **astropy-14182**: still 0 chars, but now `Generation timed out`,
  not fast-abort. kimi-400 path closed; residual failure is a
  different class (agent-loop convergence / gen-timeout). Separate
  investigation — NOT an A7 regression.
- **Patch rate: 4/6 = 67%** (vs 20% pre-A7 on the same N=10 superset).

Secondary side effect (documented, not a regression): running the
smoke in parallel with the rustpython wasm source build triggered
`WinError 1455` (Windows paging file exhaustion) on astropy-7746's
`git apply --check`. The patch was still written to predictions.jsonl
(634 chars survived); only the repair-validator phase was
short-circuited. Future large parallel runs should stagger wasm
builds vs benches.

### A8. Migrate kimi-k2.5 → kimi-k2.6 in cards.toml — ✅ SHIPPED 2026-04-24 (Phase 1: model id + specs)
**Why:** user flagged that Moonshot's latest is **kimi-k2.6**, not
k2.5. cards.toml's `kimi-k2.5` entry carries `supports_tools = false`
from the F9 audit — the documented reason is Moonshot's
`reasoning_content` API requirement on the 4th assistant tool-call
turn with `thinking` enabled (the exact symptom that caused A7 to
exist). If k2.6 resolved that contract, we could flip
`supports_tools` back to `true` after migration and regain kimi as a
tool-capable routing option. A7 stays load-bearing either way (it's a
belt-and-suspenders capability hygiene fix) but the kimi-k2.6 move is
a potential net win on routing diversity + cost.

**Action taken (Phase 1):**

1. Fetched https://platform.kimi.ai/docs/guide/kimi-k2-6-quickstart
   (Directive #6 — live docs, not training data).
2. Verified via same doc: k2.6 inherits k2.5's `reasoning_content`
   requirement on multi-turn tool calls when thinking is enabled
   (which is the default). Same HTTP 400 class as before.
3. Updated `sage-core/config/cards.toml` + mirror in
   `sage-python/config/cards.toml` (no longer a symlink on Windows —
   must sync manually):
   - id: `kimi-k2.5` → `kimi-k2.6`
   - family: `kimi-k2.5` → `kimi-k2`
   - context_window: `128000` → `262144` (256K per docs; old was
     stale pre-k2.5 carryover)
   - affinity scores bumped +0.02 (code/reasoning/tools/math)
     reflecting "Kimi's latest and most intelligent" positioning —
     subject to ablation
4. `supports_tools = false` STAYS until the reasoning_content
   plumbing is fixed. Path forward (still open, ticketed below):
   - (a) plumb `thinking: {type: "disabled"}` in our PydanticAI
     wrapper whenever the request carries tools. k2.6 supports the
     disable toggle per its quickstart.
   - (b) OR fix our wrapper to preserve reasoning_content across
     turns. PydanticAI has a `MoonshotAIProvider` class
     (https://pydantic.dev/docs/ai/api/pydantic-ai/providers/) but
     the docs don't confirm reasoning_content passthrough — needs
     a direct code inspection.
5. Updated substring quirks in `openai_compat.py`: temperature+top_p
   hard-strip now matches `k2.6` / `k2-6` / `k2-thinking` variants
   alongside existing k2.5 patterns (same quirk, documented by the
   k2.6 quickstart "temperature and top_p fixed when thinking
   enabled").
6. Updated 6 call-sites with hardcoded `kimi-k2.5`: connector.py
   default_model, 2 test fixtures, test_provider_quirks cases
   (renamed `test_kimi_k25_strips_temperature` →
   `test_kimi_k26_strips_temperature_and_top_p` + kept a
   back-compat test for callers pinning legacy id), models.toml
   comment, test_pydantic_ai_integration
   (`test_kimi_k2_5_supports_tools_is_false` →
   `test_kimi_k2_6_supports_tools_is_false`).

**Validation:** 30/30 Python quirks + provider-integration tests
PASS; 20/20 Rust template tests PASS.

**Phase 2 (pending, tracked as sub-items):**
- Investigate PydanticAI's `MoonshotAIProvider` source to determine
  if reasoning_content is preserved across turns. If yes, SAGE's
  wrapper is the bug; swap wrapper → flip `supports_tools = true`.
- If no, plumb `thinking: {type: "disabled"}` in our provider layer
  for kimi-k2.6 whenever tools are present. Ablation smoke required
  to confirm disabling thinking doesn't regress reasoning quality.

**Note on user's concern ("PydanticAI should handle this"):**
The user's intuition is sound — PydanticAI has a dedicated Moonshot
provider. The observed failure was through our SAGE `PydanticAIProvider`
wrapper. The A8 Phase 1 change is compatible with either future
resolution (wrapper fix OR thinking-disable plumbing); the cards.toml
flag gates tool use until Phase 2 lands.

### A10. search_repo size-cap against MemoryError — ✅ SHIPPED 2026-04-24
**Why:** astropy-14182's post-A7 `Generation timed out` failure
(investigated 2026-04-24) was caused by `search_repo`'s Python
fallback calling `p.read_text(encoding="utf-8", errors="ignore")` on
every file in the repo — including a big data file that exhausted
process memory with `MemoryError`. The exception propagated up to
the agent loop as a tool error; the agent retried; D8 stall cap
fired; task timed out with 0 chars. Orthogonal to A7 (kimi fast-
abort) — this is a large-file-in-repo class.

**Action taken:**
- `sage-python/src/sage/tools/typed_repo.py:_build_search_repo_tool`
  Python fallback path now: (1) `_MAX_FILE_BYTES = 1 MiB` size gate
  via `p.stat().st_size` before reading, (2) `MemoryError` added to
  the per-file `except` clause so a single pathological file can't
  abort the whole scan.
- Regression test `test_search_repo_skips_large_files` — creates a
  1.5 MiB file containing the search term + a small file containing
  the same term, asserts small file IS found AND large file is
  skipped.

**Validation:** 53/53 typed_repo tool tests PASS.

**Not touched:** the ripgrep fast-path (the primary happy path for
any machine that has rg installed) doesn't hit this bug — it uses
memory-mapped search and streams lines. The fix is specifically for
the Python fallback.

### A8 Phase 2. PydanticAI reasoning_content passthrough — ✅ SHIPPED 2026-04-24
**Why:** A8 Phase 1 (commit `c17ffd68`) migrated kimi-k2.5 → kimi-k2.6
but kept `supports_tools=false` because the underlying bug was
unverified. User's correct intuition: "on utilise pydanticAI je suis
surpris que l'on ait des problemes avec kimi." Deep-dive confirmed
SAGE's PydanticAI wrapper was dropping `reasoning_content` on both
sides of the translation — PydanticAI's MoonshotAIProvider had the
right profile (`openai_chat_send_back_thinking_parts='field'`), but
nothing to serialize because SAGE's `Message` dataclass had no
`thinking` field and `_our_messages_to_pydantic` never emitted
`ThinkingPart`.

**Action taken (commit `df150a2a`):**
- `sage/llm/base.py`: added `thinking: str = ""` to `Message` and
  `LLMResponse` with contract doc pointing to this roadmap entry.
- `sage/providers/pydantic_ai_provider.py`:
  - `_pydantic_response_to_ours` captures incoming `ThinkingPart`
    content into `LLMResponse.thinking` (previously explicitly
    dropped with a TODO comment).
  - `_our_messages_to_pydantic` prepends `ThinkingPart(msg.thinking)`
    before `TextPart`/`ToolCallPart` on assistant messages.
    Ordering is required by Moonshot's API spec —
    `reasoning_content` precedes `content` in streaming.
  - `_HAS_THINKING_PART` guard so older pydantic-ai versions
    degrade gracefully (non-thinking models keep working).
- `sage/phases/act.py` + `sage/agent.py`: four call sites now
  propagate `response.thinking` onto the assistant `Message` before
  appending to history — CGRS brake, no-tool-calls final, tool-calls
  continuation in act.py, and the equivalents in agent.py.

**Tests (2 new):**
- `test_thinking_roundtrips_both_directions` — ThinkingPart →
  LLMResponse.thinking AND Message.thinking → ThinkingPart, with
  strict ordering assertion.
- `test_thinking_empty_string_does_not_emit_thinking_part` —
  non-thinking models don't emit spurious reasoning_content on
  requests.

**Validation:** 9/9 test_pydantic_ai_integration tests PASS.

**NOT DONE (explicit — Directive #5):** kimi-k2.6 `supports_tools`
stays **false** in cards.toml until a live smoke exercises
kimi-k2.6 on a multi-turn tool-call path with zero HTTP 400s
observed. Evidence before assertions — the code fix is necessary
but not sufficient proof. A3 Phase 3 tracks this.

**Phase 3 (ticketed below as A3 validation smoke):** run a kimi-
forced smoke (e.g. bandit override pinning kimi-k2.6 to a coder
node for N=5-10 SWE-bench tasks). Confirm zero 400s in the log.
If green: flip `supports_tools=true` on kimi-k2.6 in cards.toml
AND deepseek-v4-pro (same fix class). Deepseek-v4-pro is already
provisionally `true` — the flip there is a no-op that just
upgrades the confidence level.

### A12. Paired N=50 observe vs repair smoke (NEW 2026-04-24, ticketed)
**Why:** spec § "Validation plan" — measure A3 repair-mode impact.
Same 50 lite tasks, two passes (observe-only vs observe+repair),
compare Docker pass-rate + repair-stage distribution + per-bucket
analysis.

**Cost:** ~$25-30, ~4-8 hours wall-clock (gen + Docker grading for
2 × 50 tasks). Not runnable in a single interactive session.

**Prerequisite:** A3 validation smoke (below) confirms the
repair-mode wiring produces non-empty `verifier_repair` stages in
live traffic before committing to the paired-N=50 budget. Current
status: N=20 repair-mode smoke launched 2026-04-24 to validate
wiring end-to-end.

**Concrete action when ready:** two sequential bench runs with
identical `--limit 50 --dataset lite`, only `SAGE_DIFF_VERIFIER_MODE`
differs. Script-templatable; could use `/schedule` to run during
off-hours to avoid interactive-session overhead.

### A11. DeepSeek v4 migration — ✅ SHIPPED 2026-04-24 (Phase 1: cards + default_model)
**Why:** https://api-docs.deepseek.com/ (fetched 2026-04-24) announces
**deprecation of `deepseek-chat` and `deepseek-reasoner` on
2026-07-24** (3 months from today). They become the non-thinking and
thinking modes of `deepseek-v4-flash`. Also ships
`deepseek-v4-pro` as the high-accuracy thinking variant.

**Action taken:**
- `sage-core/config/cards.toml` + `sage-python/config/cards.toml`
  (sync'd manually — Windows symlink broken):
  - **NEW** `deepseek-v4-flash`: non-thinking, 1M context, $0.14
    cache-miss / $0.28 output per 1M. `supports_tools=true`.
  - **NEW** `deepseek-v4-pro`: thinking-mode, 1M context, $1.74
    cache-miss / $3.48 output per 1M. `supports_tools=true`
    provisionally (thinking-mode reasoning_content quirk is the
    same class as kimi-k2.5/k2.6; if a production smoke hits 400s,
    flip to false like we did for kimi — same template filter
    handles it via A7).
  - **LEGACY** `deepseek-chat` + `deepseek-reasoner`: kept until
    sunset (2026-07-24) so bandit history + pinned fixtures keep
    working. Block comments mark them LEGACY.
- `connector.py::default_model` for deepseek: `deepseek-chat` →
  `deepseek-v4-flash`.
- `test_live_multiprovider.py` matrix: added v4-flash + v4-pro
  entries, kept legacy rows with LEGACY comment.
- Pricing source: https://api-docs.deepseek.com/quick_start/pricing
  (fetched 2026-04-24). Affinities seeded from the matching legacy
  entry + research-backed bumps; ablation-subject.

**Validation:** 30/30 provider + quirks + assigner tests PASS.
cards load cleanly via PyO3 ModelRegistry.from_toml_file; 7 models
spot-checked (cost + context + tools flags match live docs).

**Phase 2 (pending, tracked here):**
- **v4-pro thinking quirk first-run triage.** First production
  smoke that assigns v4-pro to a tool-using node will reveal
  whether PydanticAI's MoonshotAIProvider-style
  `send_back_thinking_parts='field'` contract covers DeepSeek too,
  or if we need the same plumbing we're ticketing for kimi
  (A8 Phase 2). If 400s appear: flip `supports_tools=false` on
  v4-pro, A7 filter excludes it from tool-using nodes, plumbing
  work becomes the unblocker.
- **Pre-sunset flip.** Before 2026-07-24, swap test fixtures that
  pin `deepseek-chat` → `deepseek-v4-flash`, remove LEGACY blocks
  from cards.toml, and kill the deepseek-chat / deepseek-reasoner
  entries from live-provider matrix. Could schedule a /schedule
  agent to open that cleanup PR around 2026-07-01.

### A9. Investigate gpt-5.5 — ✅ SHIPPED 2026-04-24 (Phase 1: cards)
**Why:** user flagged OpenAI released gpt-5.5 as a new model. Our
cards.toml ships gpt-5.4, gpt-5.4-pro, gpt-5.4-mini, gpt-5.4-nano,
gpt-5.3-codex. If gpt-5.5 is a measurable improvement over 5.4 on
reasoning/code, adding it (or replacing 5.4 as the default reasoner
tier) could materially improve SWE-bench and BCB results.

**Action taken (Phase 1):**
1. WebSearch-verified against live OpenAI sources (2026-04-24):
   - https://openai.com/index/introducing-gpt-5-5/ (announcement)
   - https://developers.openai.com/api/docs/models/gpt-5.5 (API ref)
   - https://developers.openai.com/api/docs/pricing (pricing)
   - Wikipedia gpt-5 page also lists 5.5 as a successor, confirming.
2. Added to cards.toml (both sage-core + sage-python mirrors):
   - **gpt-5.5**: $5/$30 per 1M tokens, 1M context, supports tools +
     vision + json + responses API. Affinities +0.01 above gpt-5.4
     on reasoning/code — conservative, ablation-subject.
   - **gpt-5.5-pro**: $30/$180 per 1M, 1M context, pro-tier
     affinities. S2 affinity lowered to 0.60 to prevent pro-rate
     assignment on mid-tier tasks.
   - gpt-5.4 + gpt-5.4-pro kept intact; bandit-managed over time.
3. `test_live_multiprovider.py` matrix: gpt-5.5 + gpt-5.5-pro rows
   added before gpt-5.4 entries.

**Validation:** Cards load cleanly via PyO3 ModelRegistry
(cost_in=$5.0, ctx=1M for gpt-5.5; $30.0/1M for pro). 30/30 provider
+ quirks tests PASS.

**Phase 2 (tracked here, not shipped):**
- Live-call smoke against gpt-5.5 once OPENAI_API_KEY holder has
  access (model availability at API level not verified in-session
  — only via docs).
- Ablation smoke (N=10) comparing gpt-5.5 vs gpt-5.4 on BigCodeBench
  Hard / SWE-bench to validate affinity seed values.
- Bandit calibration sweep (N=50 routing_gt) to confirm bandit
  doesn't over-select 5.5 on tasks where 5.4 remains adequate.

### A3. Repair-mode implementation — ✅ SHIPPED 2026-04-24

### A3. Repair-mode implementation — ✅ SHIPPED 2026-04-24
**Why:** spec § "Validation plan" — repair mode feeds the diff-verifier
mismatch diagnostic to an LLM one-shot repair. The 2026-04-23 observe
smoke confirmed the mismatch signal is clean (zero false positives on
two patches); 2026-04-24 A7 verification accumulated more observe data
(4 PATCH with verifier-flagged failures + 0 clean).

**Action taken:**

- `sage-python/src/sage/bench/swebench_diff_verifier.py`: new
  `repair_with_verifier_feedback(llm, problem_statement, broken_patch,
  mismatches, instance_id, timeout) -> (new_patch, stage)` async
  function. Builds a structured repair prompt that shows per-hunk
  EXPECTED-vs-ACTUAL line text (truncated to 20 lines × 200 chars
  per section to keep the prompt bounded). Returns stages
  `"verifier_repair"`, `"verifier_repair_empty"`, or
  `"verifier_repair_skipped"`.
- `_get_diff_verifier_mode`: "repair" no longer downgrades. Invalid
  modes still fall back to "off" with a WARN.
- `generate_patches`: when `verifier_mode == "repair"` AND
  `mismatches` is non-empty AND an LLM handle exists, the verifier-
  repair call runs BEFORE `try_repair_patch`. If it succeeds
  (`stage == "verifier_repair"`), the corrected diff replaces the
  original patch; `try_repair_patch` then sees the corrected input
  (and may still apply CRLF normalization / counts-fix / git-apply
  LLM-repair on top). The combined repair_stage is serialised as
  `"verifier_repair+<downstream>"` so operators can see the full
  chain.
- Observe-mode behaviour unchanged — annotations are populated in
  both observe AND repair modes, so the existing bucket-analysis
  scripts work on repair-mode predictions without modification.

**Tests (new):**
- `test_diff_verifier_repair_calls_llm_with_mismatch_feedback`
  in `test_swebench_emission_wiring.py` — end-to-end wire test:
  builds a real git repo, serves a canned agent emission with
  wrong context lines, installs a spy LLM that returns a corrected
  diff. Asserts: (a) observe-like annotation, (b) LLM called
  exactly once with a prompt naming the mismatched file AND the
  actual file contents, (c) the corrected patch propagates to the
  prediction dict.

**Test suite:** 48/48 verifier + repair + emission-wiring tests
PASS. The legacy `test_diff_verifier_repair_warns_and_downgrades_to_observe`
was replaced by the new wire test (same contract, updated
behaviour).

**Next (tracked here, not yet done):**
- **Paired N=50 smoke** (observe-only vs observe+repair) per the
  spec § "Validation plan". Measures whether repair-mode actually
  improves Docker pass-rate on flagged patches, or just generates
  new kinds of invalid diffs. Cost: 2 × 30-minute N=50 runs ≈
  $15-20 total. Gated on: stable A1 observe data (currently at
  N=20 cumulative, 4 PATCH — enough to run the pair if user wants
  signal quickly, but larger-N paired runs give more statistical
  power).
- **Repair-mode coverage of the malformed-header class (A6).**
  CRLF-normalized diffs with correct content but wrong counts are
  handled by `try_repair_patch`'s programmatic-counts stage.
  Content-mismatch is handled by A3's verifier_repair. Both
  classes are now covered; no residual is known.

### A6. CRLF normalization in patch emission — ✅ SHIPPED 2026-04-24
**Why (initial framing):** astropy-6938 in the 2026-04-24 smoke was
rejected by Docker `patch` at line 18. Surface reading suggested a
malformed hunk header (`@@ -1541,10 +1541,4 @@`) where the header
counts didn't match the body.

**Actual root cause (byte-level inspection):** the patch file had
`\r\n` line endings throughout. `git apply` and GNU `patch` both
reject `\r` bytes in diff bodies as "corrupt patch". The repair
pipeline (`swebench_patch_repair.try_repair_patch`) DID run on this
patch — both stages failed with the same CRLF-corruption error. The
existing `_fix_hunk_header_counts` works on whole lines and preserves
the embedded `\r`, and the LLM repair stage also returned a CRLF
patch (the prompt carried the CRLF in the example body). So the
two-stage repair was insufficient to fix a CRLF-only bug.

Source of the CRLF: Windows `open(path, "w")` text-mode translates
every emitted `\n` into `\r\n`. The chain is:
1. Agent emits LF patch → `_extract_patch_from_response` normalizes
   CRLF → LF at line 216 ✓
2. `write_predictions` writes with default `open(...,"w")` → CRLF
   line terminator between JSON records, inside each record the
   patch stays JSON-escaped as `\\n` ✗ (the fault)
3. SWE-bench harness reads predictions.jsonl, extracts
   `model_patch` string (LF only in memory)
4. Harness writes `patch.diff` inside Docker context via text-mode
   open → CRLF again ✗
5. Docker invokes `patch` which aborts

**Shipped fix:**
- `swebench_patch_repair.py`: new `_normalize_line_endings()` helper;
  `try_repair_patch` gains a Stage 0 CRLF normalization before
  validation. New `"crlf_normalized"` stage value for telemetry.
  Belt-and-suspenders — even if upstream writes CRLF, the repair
  pipeline now starts from LF-only bytes.
- `swebench_bench.write_predictions`: now uses
  `open(path, "w", encoding="utf-8", newline="")` so line terminators
  are bare LF on all platforms. JSON-escaped patch body stays LF by
  construction.

**Tests (4 new, all passing):**
- `test_normalize_crlf_to_lf`
- `test_normalize_bare_cr_to_lf`
- `test_normalize_noop_on_lf_only`
- `test_repair_crlf_normalization_resolves` (end-to-end through
  `try_repair_patch` with a real git repo)
- `test_write_predictions_writes_lf_only_line_endings` (byte-level
  assertion on emitted JSONL)

**Impact:** closes the astropy-6938 failure class. Future observe
smokes should see repair-stage breakdown including
`crlf_normalized` for Windows-emitted patches. A3 repair-mode
implementation no longer needs a separate malformed-header class
— the existing `content_mismatch` covers all true hallucination
cases once CRLF noise is stripped.

**NB:** the original A6 framing ("extend the verifier to detect
malformed hunk headers") was wrong. The verifier's docstring
explicitly defers structural issues to `swebench_patch_repair`;
that deferral was correct. The real gap was upstream — a platform-
emission hygiene bug, not a verifier coverage gap.

### A4. Public-claims reconciliation (README ↔ PyPI)
**Why:** ALIRE flagged a three-way divergence (README/commits/PyPI) on
test counts. README has been refreshed (commit `be2d3fc`), but the
published PyPI `ygn-sage 0.1.0` still carries the March description.
Publishing a new PyPI patch release (`v0.1.1`) would sync, at the cost
of a minor release cycle.

**Concrete action:** bump patch version, refresh classifiers, push to
PyPI via the existing release workflow if one exists (grep for it)
or add a brief `docs/RELEASE.md`.

### A5. Fix the `ss.F.` anomaly captured in `tasks/bsc5mma67.output`
**Why:** the test-ignore ceiling drifted below tests and landed (commit
`5efdd42`). The original failing run wasn't labelled and ceased to exist
in the task runtime — attribute the output to a session context for
next time, or wrap long-running pytest invocations in a `TaskCreate` so
the task ID survives.

**Cost:** trivial, mostly a workflow hygiene task.

### A13. Prompt-injection filter design + implementation (NEW 2026-04-24, AUDIT3 #10)

**Why:** AUDIT3 §3 claim #10 confirmed (✅): no explicit prompt-injection
filtering exists. Blast radius HIGH — every LLM call reaches
classifier, model, tool args. Severity HIGH per
`docs/audits/2026-04-24-audit3-triage/phase3-severity-sota.md`.

**Deferred from Phase-5 fix batch:** security-architectural, not a
200-LOC patch. PROMPT.md principle #1 ("le code non modifié est plus
sûr que modifié") — a weak regex filter would give false security
while breaking legitimate prompts.

**SOTA landscape 2026:**
1. Classifier-based (PromptGuard-2 Meta 2024, Lakera Guard, Azure
   Prompt Shields GA Nov 2024)
2. Instruction-hierarchy training (GPT-4o priv-separation, OpenAI
   research Nov 2024)
3. Structured-output / spotlighting (Microsoft arXiv 2403.14720,
   constitutional filtering)

**Concrete action when ready:** research spike (1 week) → design doc
(which inputs, which method, which action) → impl spike (1-2 weeks)
→ red-team against OWASP LLM01 corpus. Total ~2-3 weeks.

**Prerequisite:** none.
**Cost:** multi-week.

### A14. Tool-output Pydantic contracts (ToolResult v2) (NEW 2026-04-24, AUDIT3 #17)

**Why:** AUDIT3 §3 claim #17 (⚠️ partial): `ToolDef.parameters` JSON
schema exists on input side; `ToolResult.output` is a free-form string
on output side. Downstream LLM input can be corrupted by malformed
tool output. Severity MEDIUM.

**Deferred from Phase-5 fix batch:** scope touches `ToolResult`
contract — a core data structure across `sage.tools`, `sage.agent_loop`,
`sage.topology.runner`, and every tool implementation (18+ tools).
Not a ≤ 10-file, ≤ 200-LOC fix.

**SOTA pattern:** MCP (Model Context Protocol) spec 2025 uses JSON
Schema for both input AND output. LangChain
`StructuredTool.return_schema`. Our input side matches MCP; output
side doesn't.

**Concrete action when ready:** (a) design per-tool output contracts,
(b) Pydantic validator at agent-loop boundary, (c) gradual rollout —
existing tools keep free-form output until contract added.

**Prerequisite:** none.
**Cost:** 1-2 weeks.

### A16. Centralized redaction layer for logs/events/memory (NEW 2026-04-24, AUDIT/AUDIT2 §6)

**Why:** AUDIT.md + AUDIT2.md independently flag secret leakage into
event bus payloads + episodic memory content. No redaction layer
exists. Severity **HIGH**: blast = system-wide logs/traces;
exploit = any API key, user prompt, or tool output passing through
the bus; frequency = every event.

**SOTA:** centralized secret-scanner (regex for API keys + JWTs +
AWS/GCP keys) at trace emission boundary; encrypted memory tier for
sensitive context; opt-in logging for user data.

**Action when ready:** (a) spec redaction policy, (b) integrate into
`events/bus.py` + `memory/episodic.py` + `memory/working.py` emit
paths, (c) add regression tests with known secret patterns.

**Cost:** ~1 week (design + impl + tests).

### A17. Supply-chain security CI gates (NEW 2026-04-24, AUDIT/AUDIT2 §6)

**Why:** AUDIT.md + AUDIT2.md both flag missing dependency audit
infrastructure. Python `requirements.txt` not fully pinned; no
`pip-audit`, `cargo-audit`, `cargo-deny`, Semgrep, Dependabot;
GitHub Actions are version-tag pinned (not SHA-pinned); PyPI not
using Trusted Publishing. Severity **HIGH**: blast = RCE via
compromised upstream; exploit = any public-registry attacker;
frequency = every build.

**SOTA:** `cargo-audit` + `cargo-deny` in CI workflow; `pip-audit`
+ `safety` for Python; SHA-pinned Actions; Dependabot weekly
updates; PyPI Trusted Publishing.

**Action when ready:** incrementally add each gate with failure-is-
informational first week, enforce in week 2.

**Cost:** ~3-5 days.

### A18. Dynamic tool validation fail-closed (NEW 2026-04-24, AUDIT2 §6)

**Why:** AUDIT2 §6 identifies `forge.py:352-354` falls back to
`ast.parse()` alone when Rust validator errors — **fails open**, not
fail-closed. Severity **HIGH**: blast = unsafe tool registered;
exploit = Rust validator DLL-unavailable; frequency = cold-start or
build-pipeline mismatch.

**SOTA:** fail-closed pattern mirrors `SAGE_STRICT_GOVERNANCE` from
A0b — on Rust validator error, raise instead of downgrading. Add env
`SAGE_TOOLFORGE_STRICT=1` (default on in production) to flip the
behaviour.

**Action when ready:** change `_validate_python_code` to
re-raise on Rust error when env flag set; default flag to `1` after
empirical testing.

**Cost:** ~1 day.

### A19. MCP/A2A gateway authentication (NEW 2026-04-24, AUDIT.md §6)

**Why:** AUDIT.md §6 flags `sage-discover/mcp_gateway.py` +
`sage-python/src/sage/protocols/serve.py` may expose tools/resources
without authenticated HTTP transports. Severity **MEDIUM**: impact
depends on whether user actually exposes gateway publicly.

**SOTA:** MCP 2025-11-25 spec mandates authorization for HTTP
transports (not stdio). OAuth2 bearer tokens + capability-scoped
resources.

**Action when ready:** (a) localhost-only default, (b) require bearer
token for remote, (c) audit log every resource/tool access.

**Cost:** ~1 week.

### A20. Bandit causality test — Python pipeline off-policy (cgpro 2026-04-26) — ✅ SHIPPED 2026-04-26 (commit `48dc7c3f`)

**Aka `cg-A14` in MEMORY.md cross-references.** Source: cgpro 2026-04-26 post-closeout review (conversation `cgpro_2026_04_26_review`).

**Why (REAL PROD BUG, verified):** `pipeline.py:461` (Stage 0 classify) calls `self._rust_router.route(ctx.task, ctx.budget)` — the **legacy API** at `system_router.rs:199-200` (`pub fn route()` doc-comment "legacy API"). Bandit not consulted there. `pipeline.py:1232-1244` (Stage 4 execute) calls `self.bandit.select_with_context(0.1, task_context)` separately, returns `BanditDecision { decision_id, model_id, template, ... }`, **but only `decision.decision_id` is stored** as `ctx.bandit_decision_id`. Lines 1244, 1250, 1545, 1551 all drop `decision.model_id` and `decision.template`. Stage 5 `pipeline.py:1762` records outcome against the orphan decision_id. **The bandit posteriors update for an arm whose model never executed.** Off-policy learning silently corrupting all bandit data in production.

Combined with the 2026-04-26 morning `restore_arm` fix (commit `9f251276`, persists `context_sum`/`context_count`), the bandit was simultaneously:
1. Learning the wrong attribution forever (this item)
2. Losing context bias on every restart (fixed today)

**Decision needed before fixing:**
- (a) keep accumulated posteriors and fix only forward attribution
- (b) reset SQLite bandit state (lose history, clean restart)
- (c) audit posteriors first — sample N decisions, check if `(model_id, template)` selected by `select_with_context()` correlates with the executed model. If random: reset is the only honest path.

**cgpro proposal:** before `record_outcome()` fires, assert `decision.model_id` is what executed. Likely shape: replace pipeline.py:461 `route()` with `route_integrated()` (which DOES use the bandit) so Stage 0's decision IS the bandit's selection. Then drop the duplicate `bandit.select_with_context()` at pipeline.py:1232-1252 — Stage 0 already did the work. Or alternative: store full `BanditDecision` (not just decision_id) at Stage 4 and re-score `_last_routing_decision.model_id` against `ctx.bandit_decision.model_id` before allowing record_outcome.

**Acceptance test (TDD):** new test at `sage-python/tests/test_bandit_causality.py` — force two distinguishable arms (different model_ids), run one pipeline task, assert (a) the executed model matches the bandit-selected arm AND (b) record_outcome is called with quality of THAT model's execution. Green only after the fix.

**Cycle-9 A14b closure (2026-04-30):** closed forward attribution at `6f23eea4` after cgpro VERIFY round-2. The pipeline no longer records via a standalone bandit recorder: Stage 0 issues the bandit decision through Rust `route_integrated()`, Stage 5 records through `SystemRouter.record_outcome_checked()`, and every invalid or skipped attribution path consumes/cancels the pending `decision_id`. `bandit_attribution_mismatch.v1` records blocked cases; per-node attribution for parallel/debate/selfmoa remains deferred to cycle-10+.

**Cost:** shipped.

### A21. Packaging fail-closed — `pip install ygn-sage` doesn't get sage_core (cgpro 2026-04-26) — ✅ SHIPPED 2026-04-26 (commit `761c1797`)

**Aka `cg-A15` in MEMORY.md cross-references.**

**Why (verified):** `sage-python/pyproject.toml:18-31` lists deps httpx/pydantic/rich/anyio/aiosqlite/numpy/truststore/pydantic-ai. **No `sage_core` dependency.** README announces `pip install ygn-sage`. `.claude/rules/architecture.md` says "sage_core is required at runtime — ImportError raised at TopologyController.__init__ if absent." CI compensates with separate `maturin build` + `pip install --force-reinstall --no-deps` recipe; PyPI users get nothing matching that. When `sage_core.ToolExecutor` is missing, `create_python_tool()` falls back to Python subprocess sandbox (timeout-only, no seccomp/namespaces/cgroups/FS/network isolation — opposite of ADR-013 contract).

**cgpro proposal:** two coordinated fixes
1. Declare `sage_core` as a dep in `sage-python/pyproject.toml` (or document the wheel ships sage_core as binary extension; currently neither is true).
2. When Rust ToolExecutor unavailable, **fail closed** — refuse to register dynamic tools instead of silently falling through to Python subprocess sandbox. Add explicit env-var opt-in (e.g. `SAGE_UNSAFE_SUBPROCESS=1`) for the legacy fallback.

**Pairs with A18** (toolforge fail-closed on Rust validator error). Reconcile in design — same failure mode, different layer.

**Cost:** 2-3 days (packaging story + fail-closed wiring + smoke test).

### A22. Diff-context verifier reason codes (cgpro 2026-04-26) — ✅ SHIPPED 2026-04-26 (commit `133b86b5`; A22b/c/d follow-ups closed 2026-04-27)

**Aka `cg-A3a` in MEMORY.md cross-references.**

**Why:** verifier currently collapses malformed-input / header-drift / missing-files / creation+deletion all to `[] = "no opinion"`. A1 already records a real missed failure (astropy-6938 malformed hunk header arithmetic, Docker rejected before content verifier could help). Zero-flag observations could mean "patch is clean" OR "verifier had no opinion" — currently indistinguishable.

**cgpro proposal:** emit reason codes per hunk + a top-level outcome:
- `clean`
- `content_mismatch`
- `file_missing`
- `malformed_hunk_header` (closes A1's astropy-6938 class)
- `hunk_body_count_mismatch`
- `file_creation_or_deletion`
- `not_unified_diff`
- `unsupported_no_opinion`

Annotate `predictions.jsonl` with new field `_diff_verifier_reasons: list[str]` alongside existing `_diff_verifier_mismatches`. Bucket-analysis script aggregates reasons across runs.

**Follow-up closure (2026-04-27):** A22b/c/d added the no-op/off-mode
annotation regressions, deletion-side hunk-body count coverage, and
`scripts/analyze_diff_verifier_buckets.py` for outcome/reason bucket
aggregation across archived predictions.

**Cost:** ~1 day. **No API budget needed** — local code change exercised by existing test fixtures.

### A23. Build rustpython.wasm in CI + Python 3.13 + Windows sandbox matrix (cgpro 2026-04-26 + Trap E) — ✅ SHIPPED 2026-04-26 (commits `7f72ad28` + `a5688048` hotfix)

**Aka `cg-A8` in MEMORY.md cross-references** (renamed from earlier "roadmap-A8" wasm-ci entry at top of this file). Bundles **Trap E** (CI matrix gaps) and **Trap F** (`tool_executor.rs` stale subprocess-fallback doc) since they're all "CI matrix coherence + sandbox truthfulness" class.

**Why:** sandbox-dependent tests skip via `embedded_wasm_available()` so the sandbox has 0 CI regression coverage today. Verified gaps:
- `.github/workflows/ci.yml` uses Python 3.12 at every site; pyproject advertises 3.13 but it's never tested (Trap E).
- `windows-pytest` job builds Rust wheel `--features smt,onnx` only — NOT `sandbox,cranelift`. Embedded wasm sandbox never exercised on Windows in CI (Trap E).
- `tool_executor.rs:1-7` module doc claims subprocess fallback "always available" — `validate_and_execute()` hard-fails when wasm absent post ADR-013 (Trap F).

**cgpro proposal:**
1. New `build-wasm-sandbox` job: clone `external/rustpython` (submodule), `cargo build --release --target wasm32-wasip1 --features freeze-stdlib`, upload as artefact. Cache key = submodule SHA + wasm32-wasip1 toolchain version.
2. Downstream sage-core builds with `SAGE_REQUIRE_WASM=1` + `--features smt,onnx,sandbox,cranelift,tool-executor`. Missing wasm = build fail.
3. Add Python 3.13 to linux-pytest matrix.
4. Add `sandbox,cranelift` features to windows-pytest wheel build.
5. New CI assertion: `python -c "import sage_core; assert sage_core.embedded_wasm_available()"` runs in linux-pytest, integration-smoke, AND windows-pytest.

**Cost:** ~½ day (mostly CI YAML; cache wiring is the time sink).

### A24. Bandit Pareto contract docs vs code reconciliation (cgpro 2026-04-26) — ✅ SHIPPED 2026-04-26 Path A docs (commit `5a390c48`)

**Aka `cg-A12` in MEMORY.md cross-references.** Path A (docs-only) shipped 2026-04-26 alongside Trap F bundled in same commit. Path B (real Pareto multi-obj routing) still tracked here as future work — see Trap C (GammaPosterior semantics) below for the prerequisite posterior-arithmetic work.

**Why:** `sage-core/src/routing/bandit.rs:1-7` module-doc claims "Builds a global Pareto front at decision time and selects based on runtime constraints." Actual `choose()` (lines 377-461) and `choose_contextual()` (lines 474+): pure Thompson sample on `arm.quality.sample()` (with cosine bonus for contextual). Cost/latency are sampled for telemetry on the returned `BanditDecision`, **not** used for selection. Docs lie about behavior.

**Two paths:**
- **Path A — fix docs (~30 min):** rewrite module-doc to describe what code does today; annotate the future Pareto-multi-objective fix path. **Bundles with Trap F (`tool_executor.rs:1-7`)** — same docs/code mismatch class. Pilot run for the cgpro-driven trap protocol (`docs/superpowers/plans/2026-04-26-cgpro-driven-trap-resolution.md`).
- **Path B — fix code (multi-day):** real Pareto multi-objective routing. Decision needed: define posterior semantics first (Trap C below — `GammaPosterior::update(value)` increments `rate` by observed value, so `mean = shape/rate` decreases with larger observed cost/latency — counterintuitive if cost/latency are minimands).

**Decision:** start with Path A (this cycle, pilot run). Path B becomes a separate item when prioritized.

### A25. RNG seam + sort `arm_keys` for stochastic determinism (cgpro 2026-04-26) — ✅ SHIPPED 2026-04-26 (commit `5ef1940f`)

**Aka `cg-A9 + cg-A10` in MEMORY.md cross-references.** Paired — A9 alone is insufficient because `HashMap` iteration order is undefined (Rust stdlib documents this).

**Why:** `CmaEmitter::ask()` and `ContextualBandit::choose()/choose_contextual()` use `rand::rng()` with no seed parameter (verified `bandit.rs:389`). Tests can't pin behavior. The 5-test stochastic flake spree of the 2026-04-26 closeout was budget-bumped (sigma 0.5→1.0, budget 32×20) — works but not the structural fix.

**cgpro proposal:**
1. Add `&mut impl Rng` overloads to `choose()` / `choose_contextual()` / `CmaEmitter::ask()`. New PyO3 wrappers `*_with_rng` for tests.
2. Use `ChaCha8Rng::seed_from_u64(seed)` in tests (NOT `SmallRng`/`StdRng` — those don't promise portable output across platforms per rand docs).
3. Sort `arm_keys` by `(model_id, template)` before Thompson sampling: `let mut arm_keys: Vec<_> = self.arms.keys().collect(); arm_keys.sort_by(|a, b| (a.model_id.as_str(), a.template.as_str()).cmp(&(b.model_id.as_str(), b.template.as_str())));`. Pairs with #1 — seeded RNG alone insufficient.

**Cost:** 1-2 days.

### A26. Three-layer test split for stochastic suites (cgpro 2026-04-26) — ✅ SHIPPED 2026-04-27 (commit `e57ae680`)

**Aka `cg-A11` in MEMORY.md cross-references.** **Depends on A25.**

**Why:** current cma_me + bandit suite mixes deterministic mechanics with seeded stochastic tests with empirical convergence — all run on every commit. Flakes from layer 3 break commits that touched layer 1.

**cgpro proposal:**
- Layer 1 — deterministic unit tests for mechanics (covariance update, context_mean update, cosine scoring, posterior arithmetic). No probability thresholds.
- Layer 2 — seeded stochastic tests for realistic flow with fixed `ChaCha8Rng::seed_from_u64`. Exact expected behavior.
- Layer 3 — `#[ignore]`'d empirical tests for "this usually converges over many seeds". Run in scheduled/nightly job, not per-commit.

**Cost:** 1-2 days. Gated on A25 RNG seam.

### A27. Lockfile / constraints for transitive deps (cgpro 2026-04-26) — ✅ SHIPPED 2026-04-26 (commit `2c8d2557`; A27-followup closed 2026-04-27)

**Aka `cg-A13` in MEMORY.md cross-references.**

**Why:** `pyproject.toml` pins direct deps (e.g., `a2a-sdk[http-server]>=0.3.25,<1.0`); transitives drift on every CI install. The a2a-sdk drift to 1.0.2 in the 2026-04-26 closeout was caught by chance (CI started failing, mypy traced the API drift). Without lockfile, similar drift could ship silently.

**cgpro proposal:**
1. Generate `sage-python/constraints.txt` from a clean `pip install -e .[all,dev]` resolution.
2. CI installs with `pip install -c constraints.txt -e .[all,dev]`.
3. Separate weekly-scheduled `latest-deps` CI job re-resolves without constraints to catch drift without making every commit hostage to upstream.

**Follow-up closure (2026-04-27):** constraints were regenerated from
the Linux/Python 3.12 baseline, Windows-only transients were removed,
`typer` was capped to `<0.22` to match `docling` and keep the layered
`sage-python` + `sage-discover` install satisfiable, and the Python
constraints/discovery CI jobs were restored as hard gates.

**Cost:** ~½ day.

### Trap C — `GammaPosterior` cost/latency semantics (cgpro 2026-04-26, latent in A24 fix-code path)

**Why:** `bandit.rs::GammaPosterior::update(value)` increments `rate` by observed value. `mean = shape / rate`. **Larger observed cost/latency → lower posterior mean.** Counterintuitive if `expected_cost` / `expected_latency` are presented as "minimands" (the documented intent on `BanditDecision`).

**Today inert:** `choose_contextual()` only uses sampled quality + cosine; cost/latency sampled for telemetry. **Tomorrow's bomb:** if A24 takes Path B (fix code, real Pareto multi-objective), this orientation needs explicit handling — flip convention OR document "lower posterior mean = higher expected cost".

**Action:** add a deterministic posterior-arithmetic test before any A24-Path-B work. Today: docs-only annotation in `GammaPosterior::update` saying "convention: larger value → lower mean (cost/latency are minimands)".

**Cost:** 1 hour for docs annotation; 1 day for the regression test (deferred to A24-Path-B).

### A28. Engine extras persistence — CMA-ME emitter + mutation Thompson posteriors (sans-pitié audit 2026-04-27) — ✅ SHIPPED 2026-04-27 (commit `160e57e0`)

**Why:** `TopologyEngine::save_state` only persisted bandit + MAP-Elites archive. It silently dropped `cma_emitter` (mean, sigma, cov_diag, generation — full CMA-ES state for continuous parameter optimisation of max_cost_usd / max_wall_time_s / edge_weight) AND `mutation_stats` (Thompson Beta posteriors per mutation operator: alphas[7], betas[7]). Same defect class as `bandit::restore_arm` dropping `context_sum`/`context_count` (A26-cycle find).

The pre-existing round-trip test `test_engine_save_load_round_trip` was test theatre — it only asserted bandit_arm count and archive_cell count, so neither extras field was inspected. Identical pattern to a2a_server runtime drift hidden by tests that only constructed AgentCard.

**Action shipped (commit `160e57e0`):** serialised both as JSON in a third file `engine_extras.json` alongside `bandit_state.db` / `archive_state.db`. `MutationStats` already had `Serialize/Deserialize`; added `Serialize/Deserialize` to `CmaEmitter` + `DimensionBounds`. New `TopologyEngine::cma_generation()` accessor exposes the round-tripped value.

Two new tests pin the contract:
- `test_engine_extras_survives_save_load` — drives `evolve()` once, asserts CMA generation round-trips byte-identical AND every `(alphas[i], betas[i], attempts[i], successes[i])` round-trips byte-identical for all 7 operators.
- `test_engine_load_pre_extras_checkpoint` — covers cold-start path (missing extras file → keep TopologyEngine::new defaults; bandit/archive still load).

`cargo test --features smt --lib`: 522 passed (was 501 — +21 incl. these). Python `test_engine_persistence.py` 6/6 still green.

**Cost:** ½ day. Closed in same session.

### A29. Boot pipeline health-check loop preservation (sans-pitié audit 2026-04-27) — ✅ SHIPPED 2026-04-27 (commit `160e57e0`)

**Why:** Earlier in the session we fixed `_discover_models()` calling `asyncio.run()` and stripping the caller's loop on cleanup, breaking subsequent grpc.aio.Channel construction (xAI / Google providers) on Python 3.12+. The same anti-pattern lives at `boot_pipeline.py:200` `_run_health()` — every fresh boot from sync code calls `asyncio.run(provider_pool.health_check())`, whose exit calls `events.set_event_loop(None)`. Today there are no later sync grpc.aio call sites between health-check and end-of-init, so the bug is dormant — but adding any new grpc client construction post-health-check would silently break.

**Action shipped:** snapshot caller's loop, restore (or create fresh) after `_run_health()`. Same 12-line pattern as the `_discover_models` fix.

**Cost:** 10 minutes (defence in depth).

### A30. Dormant persistence-round-trip gaps (sans-pitié audit 2026-04-27) — ✅ ALL RESOLVED 2026-04-27 (commit `352d7687`)

cgpro 2026-04-27 verdict (resumed conversation `cgpro_2026_04_26_review`) split the 3 items into delete/persist with clear rationale:

**A30a — `SemanticMemory._entity_owner`: ✅ DELETED.** cgpro: "private, unwired multi-tenant feature whose failure mode is worse than its current value: if `agent_id` is ever set, loaded entities lose owner data and `get_context_for()` filters them all out. EpisodicMemory already owns the agent-scoped persisted memory concept." Removed `agent_id` constructor param, `_agent_id` + `_entity_owner` instance vars, and the multi-tenant filter in `get_context_for`. No production caller passed `agent_id`; no tests asserted on `_entity_owner`.

**A30b — CausalMemory entity + edge metadata: ✅ PERSISTED.** cgpro: "metadata is already part of the public API shape (`add_entity(metadata=...)`, `CausalEdge.metadata`, `add_causal_edge(**metadata)`), and tests already exercise passing it. The current save/load path drops it because the SQLite tables store only name/sort_order and source/target/cause_type; that is a genuine persistence-contract bug, not merely dead private state. Add JSON columns for entity metadata and edge metadata, keep backward-compatible load for old DBs with missing columns, and add round-trip tests."

Schema upgrade: `metadata_json TEXT NOT NULL DEFAULT '{}'` added to `causal_entities` + `causal_edges`. Additive ALTER TABLE migration with `duplicate column name` catch (steady-state post-v2). `load()` detects column presence per-table via PRAGMA table_info and falls back to the legacy SELECT when absent. Two new tests (`test_causal_memory_metadata_survives_save_load` + `test_causal_memory_load_legacy_schema_no_metadata_column`) pin the contract — including a synthetic v1 DB construction that asserts post-load schema upgrade. Existing `causal.db` on disk (74 entities + 122 edges) preserved through the upgrade.

**A30c — Qwen3 thinking-mode quirk: ✅ DELETED.** User confirmed Qwen3 won't be used. Branch was gated on `provider_name == "qwen"` but cards.toml ships qwen3.5 under `provider = "openrouter"` — never reachable anyway, so deletion is also a dead-branch cleanup not just a feature drop.

### A32. AdaptiveMutator persistence + offline-only wiring stance (cgpro 2026-04-27) — ✅ SHIPPED 2026-04-27 (commit `352d7687`)

cgpro verdict: "AdaptiveMutator: keep, wire, and persist. Do not delete it. The class is honest that it is not currently invoked, but its design maps directly to the ShinkaEvolve thesis: LLM-driven program evolution benefits from bandit-based LLM ensemble/model selection. The live class already has the right posterior shape (`_successes`, `_failures`, `_total_selections`) but no durable state; add a persistence interface and then wire it into the offline evolution path, not the runtime pipeline by default."

Action shipped:
- `state_dict()` / `load_state_dict(state)` for in-process serialisation (mirrors the PyTorch convention).
- `save(db_path)` / `load(db_path)` via SQLite (`adaptive_mutator_state(tier, successes, failures, total_selections)`). Snapshot semantics with `INSERT OR REPLACE` so widening the tier list later doesn't lose history.
- 4 round-trip tests covering state_dict, sqlite, missing-file cold start, and tier-list-widening migration.
- `record()` comment updated to reflect cgpro's "offline only" guidance — runtime per-task `AgentLoop` calls keep their fixed-tier behaviour. The runtime pipeline default-path is unchanged.

The "wire into offline evolution loop" call site remains a roadmap follow-up — adding the persistence + tests now closes the persistence-loss bug class without changing runtime behaviour.

**Open follow-up — A32-wire:** integrate `AdaptiveMutator` into `EvolutionEngine.evolve()` or the standalone evolution training scripts (offline only). Keep the per-task `AgentLoop` runtime path on fixed-tier. ~½ day, gated on a real benchmark to validate that bandit-driven tier selection beats fixed-tier mutation in practice.

### A33. deepseek-v4-flash reasoning_content multi-turn fix — ✅ SHIPPED 2026-05-02 (commit `27770580`)

**Why:** deepseek-v4-flash returns `reasoning_content` in thinking-mode responses during multi-agent topology turns. Without an `OpenAIModelProfile`, pydantic-ai omits it in subsequent requests → DeepSeek API HTTP 400 "reasoning_content must be passed back". Manifested as `Stage 4 multi-agent execution failed → falling back to single-agent` in A2 smoke. Same failure class as Kimi k2.5 (roadmap-A8 Phase 3, commit `ec5d0c98`).

**Fix:** New `deepseek_openai` provider kind with `OpenAIModelProfile(supports_thinking=True, openai_chat_thinking_field='reasoning_content', openai_chat_send_back_thinking_parts='field')`. `_PROVIDER_MAP["deepseek"]` now uses this kind. `model_profiles.toml` updated: `deepseek-chat` → `deepseek-v4-flash` with corrected pricing from cards.toml A11 values.

**Note:** A2 bench was running with the old code — fallbacks occurred on 2/N tasks before fix was deployed. A3 N=50 will use the fixed code.

### A31. S-MMU cold-start gap (sans-pitié audit 2026-04-27) — architectural follow-up

**Why:** `MultiViewMMU` (`sage-core/src/memory/smmu.rs`) has no `save`/`load` — chunks are wholly in-memory. Path 1 of the 6-path topology generation (S-MMU similarity hit, similarity > 0.7 AND quality > 0.5) is therefore guaranteed to miss on every cold start. Falls through to Path 2 (archive, which IS persisted) so the loss is bounded — but the costly retrieval work S-MMU was doing is reset with each restart.

**Decision needed:** is this by design (S-MMU = recency-cache only, durable signal lives in archive) or a real gap (S-MMU was meant to be durable)? The 4-tier memory diagram in `architecture.md` lists S-MMU as the working tier with episodic/semantic/causal/exocortex below, suggesting recency-cache is the intent.

**Action:** if recency-cache is intent, add a one-line ADR to that effect. If durability is intent, add Arrow IPC dump/restore (S-MMU is already Arrow-backed) keyed off the same state directory as `engine.save_state`.

**Cost:** 10 minutes for the ADR; ½ day for Arrow IPC persistence if needed.

### A15. Single-flight consolidation + graceful shutdown (NEW 2026-04-24, AUDIT3 #23)

**Why:** AUDIT3 §4 claim #23 (⚠️ partial, post-Phase-3-inspection):
`agent_loop.py:313-330` `_maybe_run_consolidation` awaits
`consolidator.consolidate()` — blocks the agent loop for one batch
pass every 10 steps. Not fire-and-forget. Severity LOW (bounded
impact, non-default multi-producer only).

**Deferred from Phase-5 fix batch:** naïve
`asyncio.create_task(...)` tempting but risky:
- concurrent consolidation passes could race on SQLite writes
- agent-loop termination could orphan an in-flight task
- "consolidated" metadata marker needs atomic-by-key semantics to
  avoid double-processing

**Concrete action when ready:** single-flight queue + graceful
shutdown hook + idempotent `consolidated=True` marker.

**Prerequisite:** none.
**Cost:** 2-3 days.

---

## Horizon B — medium-term (next 1-2 months)

### B1. OpenTelemetry GenAI spans — Closed (2026-04-25)

Spec: `docs/superpowers/specs/2026-04-25-otel-genai-spans-design.md`
Plan: `docs/superpowers/plans/2026-04-25-otel-genai-spans.md`
Implementation commits: b92836a7 → 3ca4c5cb (10 tasks via subagent-driven-development)

Sub-items:
- **B1.b** ✅ CLOSED 2026-04-25 — sage-core Rust spans via `tracing-opentelemetry`
  bridge. Approach A (independent Rust OTel SDK + W3C traceparent across
  PyO3, no PyO3 0.27 upgrade). 27 span call sites audited (counts/IDs only,
  zero raw payloads). 9 unit tests + 1 E2E smoke + 5 Python integration
  tests, all green. CI gates added (`rust-features` + `windows` jobs).
  Spec: `docs/superpowers/specs/2026-04-25-otel-rust-spans-design.md`.
  Plan: `docs/superpowers/plans/2026-04-25-otel-rust-spans.md`.
  Implementation: c2d4969b → 3c4e812d (~13 commits via subagent-driven-development).
- **B1.c** — sage-discover MCP server retrieval spans. 0.5–1 day.
- **B1.d** — ui/FastAPI auto-instrumentation. 0.5 day.
- **B1.e** — sampler tuning once production volume data lands. TBD.

B1.b deferred follow-ups (lower priority; days-scope each):
- **B1.b.1** — rename Rust span names to `sage.<crate>.<op>` form (cosmetic alignment).
- **B1.b.7** — logfire-mode Rust export (auth header + endpoint contract underspecified
  in public docs; Python logfire path covers the primary surface today).
- **B1.b.8** ✅ closed inline (CI gates landed in 3c4e812d).
- **B1.b.9** — OTLP batch exporter with explicit tokio runtime ownership (MVP
  ships with `with_simple_exporter` to avoid the "no live tokio runtime at PyO3
  boundary" panic class).

### B2. Durable trace + deterministic replay harness
**Why:** ALIRE high-severity item, and it's the prerequisite for any
serious formal-verification work (ALIRE's "runtime assurance" direction).
Every LLM call + tool call + topology decision + memory write gets a
trace ID and payload hashed; replay reconstructs the exact decision
sequence under mocked providers/tools.

**Concrete action:** start with an in-memory JSONL trace schema (one
line per event, typed). Ship the writer first; deserialisation + replay
later. Target: SWE-bench bench can reproduce a prior run's decisions
when replayed on mocks.

**Cost:** significant (~2-3 weeks). Defer if A1-A3 uncover a smaller
lever.

### B3. ToolPolicy capability manifest
**Why:** ALIRE critical item. Current `ToolRegistry` exposes
register/list/describe but no capability labels, side-effect contracts,
or approval requirements. That's a prompt-injection exposure.

**Concrete action:** new `ToolSpec` field `capabilities:
list[Capability]` + `data_access: DataScope` + `approval_required: bool`.
Policy check inserted in `execute_tool_call` before the tool runs.
Migrate all builtin tools (`execute_bash`, `read_file`, `search_repo`,
`apply_patch`, etc.) to declare capabilities. Default-deny on new tools
without a manifest.

**Dependencies:** none. Can ship incrementally (typed tools first, then
generated ToolForge tools).

### B4. Platform wheels with Rust core on PyPI
**Why:** ALIRE high severity. Current PyPI wheel is pure-Python;
`pip install ygn-sage` doesn't get the Rust extension. Either fix the
install story or clearly scope the PyPI package as "Python bindings —
build Rust from source".

**Concrete action:** extend the release workflow with a `maturin build
--release` matrix (Linux/macOS/Windows) and upload via `twine`. Include
the `rustpython.wasm` artefact (or build it in CI) so `SAGE_REQUIRE_WASM=1`
passes. Decide whether the wasm artefact goes in the wheel (37 MB bloat)
or a separate package (`ygn-sage-sandbox`).

**Cost:** 1 week. Mostly CI work.

### B5. CI job that builds wasm + enforces `SAGE_REQUIRE_WASM=1`
**Why:** the 2026-04-23 `SAGE_REQUIRE_WASM` gate (commit `cf188df`)
turns missing `rustpython.wasm` into a build error, but no CI job
exercises that path. A release pipeline that doesn't verify the wasm
artefact is the same failure mode in a nicer suit.

**Concrete action:** add a `.github/workflows/sandbox-build.yml` job
that clones RustPython, builds `wasm32-wasip1`, caches it, then runs
`SAGE_REQUIRE_WASM=1 cargo build --features sandbox`. ~20 min added to
CI (cacheable via `actions/cache`).

### B6. Fast-abort root cause fix (depends on A2 findings)
**Why:** whatever A2 surfaces, fixing the 20% fast-abort rate has
proportional payoff on every SWE-bench smoke.

### B8. Finish or de-scope the Rust controller orchestration entry — ✅ CLOSED 2026-04-23 (de-scope)
**Why:** ALIRE2 verification (A0a companion, 2026-04-23) confirmed
`RustTopologyController::evaluate_and_decide` returned `None` (scaffold
stub). Per-path methods (`check_empty_error_reroute`,
`check_quality_cascade`, `check_parallel_inconsistency`,
`check_importance_prune`, `is_in_gate_band`,
`should_trigger_emergent_spawn`) ARE populated and ARE called from
Python's `TopologyController`. "Rust-primary since 2026-04-20" (ADR-012)
was accurate for each decision path, but misleading if read as
"complete Rust control plane".

**Decision (2026-04-23):** de-scope + delete. Advisor (built-in
reviewer) and codex `gpt-5.4` at `model_reasoning_effort=high` both
converged independently:

- The orchestration cascade depends on Python-resident subsystems
  (embedder, SmtVerifier access, topology-graph `predecessors`, gate
  management, upgrade-model resolution). A Rust top-level wrapper
  would be a thin PyO3 shim back into Python — no measurable win.
- No current or near-term scenario (2026 horizon) shows the
  Python↔Rust crossing as a hot path. Benchmarks back this up.
- Renaming the stub preserves a misleading artifact; deletion makes
  the Rust/Python boundary explicit and matches current behaviour.

**Action taken:**

- **Deleted** `RustTopologyController::evaluate_and_decide` from
  `sage-core/src/topology/controller.rs`.
- **Updated** the module doc comment to state "Rust primitives that
  back the Python `TopologyController` … Python orchestrates which
  primitive to invoke".
- **Updated** ADR-012 with an amendment replacing "Rust-primary
  controller" with: `RustTopologyController` is Rust-primary for
  adaptation state and per-path decision primitives; orchestration
  remains Python-owned where decisions depend on Python-resident
  subsystems (embedder, SMT feedback access, topology graph, gate
  management, upgrade-model resolution).
- **Test contract**: `test_rust_topology_controller_exposes_per_path_primitives`
  (replacing `test_rust_topology_controller_evaluate_scaffold_returns_none`)
  asserts per-path primitives exist AND `evaluate_and_decide` is
  deliberately absent — regression guard against future resurrection.

**Effort:** ~2h (estimate matched).

### B9. Per-run immutable execution context (AgentLoop refactor)
**Why:** A0a (shipped `9067be5`) fixes incomplete restoration of the
10 mutated fields in the bypass path's `finally`. That closes
state-bleed under serial nested calls, but NOT under concurrent ones —
two `run()` coroutines on the same singleton AgentLoop would still
race on the shared fields between the snapshot and the `finally`.

The architectural fix is stop mutating a shared object entirely: build
a fresh, immutable execution context per run. References ALIRE2 §4.

**Estimated effort:** 1-2 weeks. Don't start until A1/A3 (observe +
repair mode) stabilise; this is a big refactor and interleaving it
with Track 3 work multiplies risk.

### B7. Test count drift gate automation
**Why:** the `test_mypy_count.py` ceiling drifted by 5 between commits
over weeks (caught in this session). The ceiling-bump flow is manual
and error-prone.

**Concrete action:** add a pre-commit hook or CI step that, on ceiling
drift, auto-generates a commit that lists the new ignores and their
source commits for review. Or: replace the ceiling test with a
per-file baseline (count per module is allowed; new modules must
declare).

---

## Horizon C — longer-term (3-6 months)

### C1. Semantic LLM verifier (option C from the diff-context spec)
**Gated on:** diff-verifier repair mode shipping (A3) and NOT moving
resolved rate ≥ 4 pp at N=50.

**Why:** the 2026-04-23 Track 3.1b analysis found 3 orthogonal
semantic-miss failure modes (context hallucination, over-aggressive
scope, wrong-layer coercion). The content-verifier addresses mode 1.
Modes 2 and 3 would need an LLM pass asking "does this diff actually
fix the problem". Expensive (+ $0.02-0.10/task) but the only
prompt-level remediation for mode 3.

### C2. Domain-knowledge injection (Track 3 breadcrumb #2)
**Why:** django-10924's `FilePathField.formfield` wrong-layer fix
requires knowing about Django's migration autodetector semantics. Not
a library-API contract (so `lookup_library_docs` doesn't help),
not a research-paper question (so `search_exocortex` doesn't help).
Could be:
- domain-specific knowledge bases seeded at boot (Django, Flask,
  numpy, etc.)
- or a pre-emission "is there framework-specific convention I'm
  missing?" LLM meta-question

Open research question. Keep a backlog note; implement only if observed
more than once in tracers.

### C3. External benchmark harness (GAIA, AgentBench, τ-bench, SWE-bench-Live)
**Why:** ALIRE high severity. Internal benchmark docs are candid but
don't substitute for external baselines. GAIA, AgentBench, and
τ-bench are the credible frames for agent evaluation; SWE-bench-Live
is the contamination-resistant next generation of SWE-bench-Verified.

**Cost:** multi-week. Likely need a dedicated harness module + artefact
store.

### C4. Runtime assurance layer (ALIRE "highest-leverage change" tier)
**Why:** pre/post-conditions on every tool call, data-flow constraints,
policy enforcement before side effects. Builds on B3 (ToolPolicy) and
B2 (durable trace).

### C5. Model-checked orchestration specs (TLA+/Alloy)
**Why:** ALIRE "world-class" tier. Multi-agent coordination,
cancellation, retry semantics captured in a formal spec. Tests verify
the runtime traces match the spec.

**Cost:** very high. Consider only once B2 + B3 land; speculative
otherwise.

---

## Open questions (breadcrumbs)

These are design ambiguities without a decisive experiment yet. Kept
here so they don't get lost.

1. **Is 0.95 really the right fuzzy-threshold for the diff-context
   verifier?** Spec correction `3c3fc27` narrowed the fuzzy branch to
   whitespace-only equivalence, with 0.95 retained as an observability
   hint. If the observe bucket accumulates fuzzy-below-threshold cases
   that correlate with whitespace drift we didn't anticipate, the
   threshold may need tuning or the branch may need another narrowing.

2. **Is `search_exocortex` actually useful in SWE-bench?** Prompt
   hygiene commit `29987bc` reframed it from anti-affordance to
   positive use case, but a follow-up audit hasn't measured whether
   usage went from 0 calls to non-zero. Worth a one-line grep on the
   next SWE-bench gen log: does `tool.call name=search_exocortex`
   appear anywhere?

3. **Module::serialize cache vs wasmtime built-in cache.** The
   2026-04-23 JIT cache (commit `50b4ee8`) hand-rolls a single-file
   .cwasm artefact. wasmtime's `Config::cache_config_load_default()`
   is the official built-in alternative. If we find ourselves adding
   cache-management complexity (invalidation rules, concurrent-writer
   safety beyond the current tempfile+rename, multi-module caches),
   consider migrating to the built-in.

4. **Why 8/10 EMPTY is the norm on SWE-bench Lite smokes.** Covered
   by A2 above but worth repeating as a meta-question: is this a
   model/capability ceiling, an infra issue (circuit breaker), or a
   prompt issue? The answer shapes which roadmap items are
   prioritised next.

5. **Do we need a dedicated prompt-injection red-team corpus?** The
   2026-04-22 wasm sandbox red-team (40/40 passing) validated the
   capability layer; nothing equivalent exists for prompt injection
   via the agent's tool-call path. ALIRE flagged this as high; hasn't
   surfaced as a concrete incident yet.

---

## Dropped / superseded directions

* **SWE-bench parity smoke at N=50** — the `±2 pp` statistical gate was
  below the N=50 SE ceiling (noise floor ≈ 10 pp/task; combined arm-gap
  SE ≈ 2 pp at N=50). Replaced by the functional criterion (typed-only
  produces patches) which landed 2026-04-22.
* **Track 3.2 "read test files first" prompt addition** — invalidated
  by the Track 3.1 finding that agents DO read test files on all three
  tracers.
* **Track 3.5 dedicated N=50 paired smoke** — deferred at the 2026-04-23
  close-out; no prompt-level lever had a lift hypothesis worth the $30-50
  spend.
* **Noise-floor calibration via paired identical-config runs** —
  advisor correctly flagged this as "measurement theater"; noise floor
  can be computed post-hoc from any future N=50+ smoke via resampling.

* **ALIRE2 §6 sandbox claims E/F/G (subprocess-based execution paths)**
  — accurate-as-described in code but **orphaned** post ADR-013 §5
  flip (`c2113d8`, 2026-04-22). `sandbox/manager.py` +
  `sandbox/isolated_executor.py` remain in the tree as legacy fallback
  that is unreachable on the default path (`allow_local=False`, Wasm
  first, hard-fail if Wasm absent). `sage-core/src/sandbox/subprocess.rs`
  gated behind `SAGE_UNSAFE_RAW_EXEC=1`. Deletion of the Python-side
  orphan files is tidy-but-not-urgent — ticketed as a separate B-tier
  cleanup; see `docs/audits/2026-04-23-alire-verification.md` for the
  divergence note.

* **ALIRE2 §3 "heuristic extraction in QualityEstimator" (claim I)**
  — refuted by code inspection. No regex / length / keyword signals in
  `quality_estimator.py`. Path is Z3 + (optional ONNX) + None
  abstention. The ONNX-not-shipped half of the claim IS true (handled
  by A0d docs sweep, `bf220e0`).

---

## Horizon pacing

Roughly: two Horizon A items per week; one Horizon B item per month;
Horizon C items tracked but not worked on until a B item forces the
issue. Do not lump A and B in the same commit; ship them separately.
