# astropy-14995 Decision-Path Audit

**Date:** 2026-04-18
**Anchor task:** `astropy__astropy-14995` (SWE-bench Lite, instance 1/5 of baseline_v2)
**Source log:** `C:\Users\yann.abadie\AppData\Local\Temp\baseline_v2.log` (lines 94–163)
**Scope:** DECISION-path audit (not codebase audit). Every claim cites `file:line`.

---

## 0. Success criteria (set before running)

| # | Criterion | Status |
|---|-----------|--------|
| a | 3-sentence explanation of 51 chars / 20 tool calls with `file:line` | ✅ §1 |
| b | Touched-vs-skipped handles table: edges, controller, memory, evolution, quality | ✅ §2 |
| c | Top 5 decisions anchored on *this* sentinel (not abstract findings) | ✅ §3 |

---

## 1. Three-sentence sentinel explanation

1. **The 20-step cap is wrong for every node in sequential.** `agent_loop_factory.py:109–114` assigns `node_max_steps = 20` when `system_level >= 3`, but the factory receives the *outer task system* (S3) from `pipeline_stages.py`, **not** the per-node system declared in `templates.rs:36,44,54` (sequential = `1/2/1`); all three nodes therefore run a 20-step S3 budget they were never designed for, and the SINK synthesizer (`templates.rs:60` — `SINK_NODE_PROMPT` says "Output ONLY the final answer") ends up tool-exploring instead of synthesizing.

2. **The 51-char sentinel is built in `phases/learn.py:108,116`** — `EMPTY_STEP_SENTINEL.format(step_count=20)` → exactly `"[sage: agent exited after 20 steps with no content]"` (51 chars + newline in patch = 52 chars in log line 163). The loop in `agent_loop.py:323` (`while self.step_count < self.config.max_steps:`) exits when the LLM emits 20 consecutive tool-call turns without a final-answer content payload, and `phases/learn.py:108` returns the sentinel instead of raising.

3. **The cascade failed because the synthesizer has no Plan B when its only predecessor is a sentinel.** `runner.py:193–199` correctly strips the coder's sentinel, but `runner.py:225` then returns an **empty string** to the synthesizer; `runner.py:243–276` only runs the `_maybe_planner_injection` fallback when `SAGE_PLANNER_INJECTION=1` (default **OFF**, line 243), so the synthesizer runs with task prompt alone, burns 20 steps of `gemini-2.5-flash` tool calls (log 141–161, 21 HTTP POSTs), and emits the same sentinel.

---

## 2. Touched-vs-skipped handles on astropy-14995

### 2.1 Classification & Decomposition (Stage 0–2)

| Handle | Fired? | Evidence |
|---|---|---|
| Rust `SystemRouter` (88% GT) | ✅ | log L95: `Stage 0: Rust routing → S3 model=deepseek-reasoner (conf=0.93)` |
| Rust `KnnRouter` (92% GT) | ❓ | Not in log — either ran silently or bypassed in favour of SystemRouter |
| `TaskPlanner` decomposition | ✅ | log L98: `omega=1 delta=4 gamma=0.50` |
| `select_macro_topology` | ✅ | log L98: `template=sequential` |
| `TopologyEngine.generate` (6 paths) | ❌ | Template path won; Path 1 (S-MMU archive), Path 3 (LLM synthesis), MCTS never fired — see `engine.rs` priority |

### 2.2 Model Assignment (Stage 3)

| Handle | Fired? | Evidence |
|---|---|---|
| `ModelAssigner` per-node scoring | ✅ | log L99: `Assigned models to 3 nodes (budget=10.00, task_system=3)` |
| **`task_system=3` overrides per-node `system_level=1/2/1`** | 🚨 | `templates.rs:36,44,54` declares 1/2/1 but factory sees S3 → 20-step cap for all |
| `ContextualBandit.thompson_sample` override | ❓ | Not logged — unclear if bandit steered `deepseek-reasoner` or default |
| `exclude_providers` (TTL 300s) | ✅ | log L93: `still dead=['openai']` |

### 2.3 Edges (Rust `TopologyEdge`)

| Field | Populated? | Consumed? | Evidence |
|---|---|---|---|
| `edge_type = "control"` | ✅ | ⚠️ | `templates.rs:66,67` — only control edges; runner treats all edges as control implicitly |
| `field_mapping: Option<HashMap>` | ❌ | ❌ | Sequential never calls `TopologyEdge::message(Some(mapping))` — only hub/parallel do (`templates.rs:124,127,347`) |
| `gate: "open"/"closed"` | ⚠️ default `"open"` | ❌ | `topology_graph.rs:315-361` exposes it but sequential never flips it; only `debate`/`AVR` templates use it (runner.py:1053-1062 `open_gate` path) |
| `condition: Option<String>` | ❌ | ❌ | Set only in hub (`templates.rs:347`) — `condition: "task_type == 'type_N'"` — not evaluated anywhere in the runner's sequential path |
| `weight: f32` | ✅ default `1.0` | ❌ | Never read by sequential traversal |
| `typed_gate()` call | ❌ | ❌ | Method exists (`topology_graph.rs:372-375`) but not invoked on this run |

**⇒ Sequential template uses TopologyEdge as *pure metadata*. 4 out of 5 edge fields are ignored.**

### 2.4 TopologyController (Phase C runtime adaptation)

| Handle | Fired? | Evidence |
|---|---|---|
| Controller instantiated | ✅ | log L82: `TopologyController initialized (Phase C)` |
| `upgrade_model` action | ❌ | Not in log; would need quality < 0.7 per-node signal but QE didn't run per-node |
| `spawn_subagent` | ❌ | `topology_controller.py:53` MAX_SPAWNS=3; zero fires on this task |
| `reroute_topology` | ❌ | `topology_controller.py:51` MAX_REROUTES=1; zero fires |
| `open_gate` (multi-turn) | ❌ | Sequential has no back-edge to open; only AVR/debate wire this |
| `prune_node` | ❌ | 3-node DAG — pruning would break linearity; controller never invoked |
| **DriftMonitor `SWITCH_MODEL`** | ✅×2 | log L127: `score=0.472` (coder), L146: `score=0.552` (synthesizer) — but this is `monitoring/drift.py:99`, **NOT** TopologyController. It fires inside `agent_loop.py:196` per-step, not on the DAG level |

**⇒ TopologyController existed but never decided. DriftMonitor switched models mid-node but couldn't rescue a 20-step empty run because the switch happened too late (after step 15+ based on `cost=0.0452`).**

### 2.5 Memory (4-tier) — writes happen but write-gate is bypassed

*Corrected 2026-04-18 per Codex gpt-5.4 xhigh second-opinion. Original log-grep missed the real write strings.*

| Tier | Writes? | Evidence |
|---|---|---|
| Rust Arrow STM | ✅ | `pipeline.py:145-165` (working-memory events fed in) |
| SQLite Episodic | ✅ | `pipeline.py:168-189` + `phases/act.py:206-213` (step storage) |
| Entity Semantic (causal) | ❓ unverified | `phases/act.py:217-223` — only activates when content length > 50; sentinel outputs (51 chars but cascade-empty) may NOT trigger it |
| ExoCortex RAG | ❌ | `tools/exocortex_tools.py:33-39` is query-only; no writes issued on this task |
| `RustCompositeWriteGate` 5-signal | ❌ bypassed | `memory/write_gate.py:66-93,162-247` exists but runtime writes call storage APIs *directly* — the salience gate is architecturally dead weight |

**⇒ Memory IS being written (STM+Episodic), but the 5-signal write-gate that exists specifically to suppress low-value writes (sentinel outputs, zero-content turns, drift-flagged events) is never consulted. The RustCompositeWriteGate is architecturally present but bypassed. Consolidator promotes to semantic only when content > 50 chars; sentinel == 51 chars may promote garbage.**

### 2.6 Evolution

| Handle | Fired? | Evidence |
|---|---|---|
| `AdaptiveMutator` | ❌ | Boot log only (L16 `Online evolution enabled`, L18 `EvolutionMemory wired`) |
| MAP-Elites insertion | ❌ | No `archive.insert` log; evolution is LEARN-stage driven and task failed before learn |
| CMA-ME emitter | ❌ | — |
| LLM mutator (Codex) | ❌ | — |

### 2.7 Quality

| Handle | Fired? | Evidence |
|---|---|---|
| **Outer FrugalGPT quality gate** | ✅×1 | log L338: `Stage 4: quality=0.00 < 0.3, triggering FrugalGPT cascade retry` (this is for a LATER task, not astropy-14995) |
| `RustQualityEstimator` (5-signal) | ❌ | Not per-node on astropy-14995 |
| `QualityLabeler` (Z3 formal) | ❌ | Needs `smt` + `tool-executor` features; never shows label output |
| Per-node Z3 labeling | ❌ | Phase C design calls for it; not wired on sequential |

**⇒ Only the outer pipeline's FrugalGPT gate runs; per-node quality scoring (the whole point of Phase C) is not wired into sequential traversal.**

### 2.8 Summary matrix

| Stage | Touched | Skipped |
|---|---|---|
| Classify | SystemRouter | KnnRouter (silent) |
| Decompose | TaskPlanner | — |
| Topology | Template path | 5 other generation paths |
| Assign | ModelAssigner, TTL exclusion | Bandit override (unlogged), edge `field_mapping` |
| Execute | DriftMonitor SWITCH_MODEL | TopologyController (all 5 actions), sentinel strip had nothing to replace with |
| Learn | — | MAP-Elites, EvolutionMemory, 4-tier Memory, Z3 QualityLabeler |

---

## 3. Top 5 decisions anchored on this sentinel

Ranked by **would-this-prevent-51-char-sentinel-on-astropy**.

### D1. Fix per-node `system_level` propagation into `agent_loop_factory` 🚨 BLOCKER
- **File:** `sage-python/src/sage/agent_loop_factory.py:53,95-114`
- **Bug:** The function receives `system_level` from the outer task classifier (`task_system=3`), not from `TopologyNode.system` (1/2/1 for sequential).
- **Fix:** Read `node.system` from the TopologyNode Rust struct exposed via pyo3 (`topology_graph.rs:TopologyNode`), pass it as `system_level` to `build_agent_loop_for_node`.
- **Impact:** Synthesizer (S1) gets `max_steps=5`, not 20 → fails *fast*, outer pipeline can reroute. Coder (S2) gets 10, not 20 → cuts waste in half when stuck. Planner (S1) gets 5.
- **Verification:** Re-run astropy-14995, confirm synthesizer tool_calls ≤ 5.

### D2. Add mandatory sentinel-fallback replacement in `_gather_predecessor_context`
- **File:** `sage-python/src/sage/topology/runner.py:183-225`
- **Bug:** When all predecessors are sentinels, `_gather_predecessor_context` returns `""` and the synthesizer runs with task prompt alone.
- **Fix:** If `parts_with_roles` is empty AFTER strip and predecessors existed, emit a SHORT explicit note: `"[system]: upstream nodes failed to produce content. Task context only below:\n{task}"`. Better than silent empty — lets synthesizer know it must fall back to direct solving.
- **Impact:** Synthesizer knows it's a cold-start, can short-circuit to emit "no patch" instead of burning 20 steps exploring.
- **Verification:** Mock predecessor_outputs = [sentinel, sentinel]; assert context contains `"upstream nodes failed"`.

### D3. Wire TopologyController into sequential traversal — not just debate/AVR
- **File:** `sage-python/src/sage/topology/runner.py:1020-1230` (controller invocation sites)
- **Observation:** Grep shows controller decision points at lines 1032, 1104, 1162, 1223 — all gated behind multi-turn/reroute conditions. On sequential, node execution completes → next node executes; controller never runs `decide()` after a sentinel.
- **Fix:** After each node completion in `_execute_node_via_agent_loop` (runner.py:379), if `_is_sentinel(output)` and `self.controller`, call `controller.decide(...)` with action space `{upgrade_model, spawn_subagent}`. On sequential a reroute doesn't help but an upgrade or spawn does.
- **Impact:** Sentinel on coder → controller triggers `upgrade_model` to S3 reasoner → rerun coder → likely produces content.
- **Verification:** Add integration test: sequential with stub coder that emits sentinel once → controller fires upgrade_model → second attempt succeeds.

### D4. Move sentinel detection INSIDE `agent_loop` — stop returning silent sentinels
- **File:** `sage-python/src/sage/phases/learn.py:108,116` + `agent_loop.py:323`
- **Bug:** Returning `EMPTY_STEP_SENTINEL` as a normal-looking string masks the failure; callers have to string-match `_SENTINEL_PREFIX`.
- **Fix:** Raise a typed exception `AgentLoopBudgetExhausted` when `step_count == max_steps` and no final content was emitted. Runner catches it, records structured failure metadata (step_count, last_tool_call, last_response), passes to controller.
- **Impact:** Structural cleanup. Sentinel string becomes a fallback only, not the primary signal. Controller gets real evidence instead of substring parsing.
- **Verification:** `pytest tests/agents/test_agent_loop.py::test_budget_exhausted_raises`.

### D5. Enable `SAGE_PLANNER_INJECTION=1` by default for sequential topologies
- **File:** `sage-python/src/sage/topology/runner.py:243` (`os.environ.get("SAGE_PLANNER_INJECTION") != "1"`)
- **Observation:** Planner produced 1678 chars successfully (log L117). That work was *thrown away* when the coder's sentinel replaced it downstream — synthesizer never saw the plan.
- **Fix:** Flip default to ON for `sequential` and `hierarchical` templates (MASS paper backing per runner.py:231). Keep off for parallel/debate where upstream context is already managed.
- **Impact:** Synthesizer gets the 1678-char plan even when coder fails → can emit a best-effort patch from plan structure alone.
- **Verification:** Run astropy-14995 with flag on; confirm synthesizer output > 51 chars.

### D6. Make DriftMonitor `SWITCH_MODEL` actionable (currently log-only)
- **File:** `sage-python/src/sage/agent_loop.py:193-205` + `monitoring/drift.py:96-111`
- **Bug (Codex gpt-5.4 review confirms):** DriftMonitor classifies drift as `SWITCH_MODEL`/`RESET_AGENT`/`CONTINUE` and `agent_loop.py:196` just emits `log.warning(...)` + a DRIFT event. **Nothing listens to those events in this code path.** `SWITCH_MODEL` in the log is a classification label, not an imperative — the model is NOT switched.
- **Evidence:** Log L127 `score=0.472 action=SWITCH_MODEL` fired mid-coder-run; the coder then made 10 more `openrouter.ai` POSTs with the same model anyway.
- **Fix:** Subscribe `ProviderPool.fallback_to_next()` to DRIFT events with `action="SWITCH_MODEL"`. On `RESET_AGENT`, reset working memory + clear tool call history for the current node.
- **Impact:** The first drift event in coder-node would re-route to a cheaper/faster model before burning 20 steps.

### D7. Stop SWE-bench extraction from treating sentinel text as a patch
- **File:** `sage-python/src/sage/bench/swebench_bench.py:228-283, 576-580, 872-875`
- **Bug (Codex):** The final emitted PATCH for astropy-14995 was 52 chars = sentinel + newline. Docker eval would fail because it's not a valid diff, but the JSONL output records it as a patch attempt. Metric = `pass_rate / len(instances)` counts this as a failed attempt, not a refused attempt.
- **Fix:** In `swebench_bench.py` extractor, if result matches `_SENTINEL_PREFIX`, emit empty-string patch + mark `_trace.structured_failure = "step_budget_exhausted"`. Prevents bench metric pollution.

### D8. Add early "no-content after N tool turns" breaker in agent_loop
- **File:** `sage-python/src/sage/agent_loop.py:323-389` + `phases/learn.py:91-108`
- **Bug (Codex):** The loop only stops at `max_steps`. An S3 budget of 20 steps with zero final-content emission between step 5 and 20 is guaranteed failure. No soft cap.
- **Fix:** After 5 consecutive tool-calling steps with no final content, raise a recoverable `AgentLoopStalled` that the runner/controller can catch and react to (upgrade_model, abort node, etc.). Must co-land with D3 (controller on sequential).

### Lower-priority / research follow-ups

- **D9 (audit-only):** Sequential template uses `TopologyEdge::control()` exclusively. `field_mapping`/`gate`/`condition` are pure metadata. AgentConductor/MASS use structured message edges. **Defer** until D1–D8 land — edge semantics don't help if budget + drift + sentinel paths aren't fixed.
- **D10 (log hygiene):** Add explicit log line `ModelAssigner: node={role} → model={model_id} (system={level}, score={s})` so per-node assignment is visible. Currently only the aggregate `Assigned models to 3 nodes` appears (log L99).
- **D11 (memory):** `RustCompositeWriteGate` is bypassed — all runtime writes call storage APIs directly. Wire it in as a mandatory predicate. Also write STRUCTURED FAILURES (tool invocations + sentinel context) for bandit priors, not just successes.

---

## 4. Apply order — ALL 8 DECISIONS APPLIED (2026-04-18)

| # | Fix | LOC | Files | Status |
|---|---|---|---|---|
| D5 | Planner injection default ON | 3 | `runner.py:243` | ✅ |
| D2 | Sentinel-fallback cold-start note | 20 | `runner.py:218-240, 296-306` | ✅ |
| D1 | Per-node `system_level` override | 15 | `runner.py:475-501` | ✅ |
| D7 | SWE-bench sentinel patch filter | 30 | `bench/swebench_bench.py:78-114, 228-238, 540-570, 846-848` | ✅ |
| D4 | Typed `AgentLoopBudgetExhausted` exception | 80 | `agent_loop.py:66-130, 215-225`, `phases/learn.py:91-140`, `agent.py:15-32` | ✅ |
| D8 | Soft-cap stall breaker | 55 | `agent_loop.py:460-505`, `agent_loop_factory.py:109-126` | ✅ |
| D3 | Controller detects sentinel output | 12 | `topology_controller.py:39-50, 289-303` | ✅ |
| D6 | DriftMonitor → ProviderPool.fallback | 55 | `agent_loop.py:244-253, 270-284`, `agent_loop_factory.py:55, 141-146`, `runner.py:493-528` | ✅ |

**Total: ~270 LOC of behavior changes + 12 new regression tests, 150/150 existing tests still pass.**

Regression tests added:
- `tests/test_runner_agent_loop.py::test_factory_receives_per_node_system_level` (D1)
- `tests/test_runner_agent_loop.py::test_factory_system_level_unset_when_node_system_zero` (D1)
- `tests/test_runner_agent_loop.py::test_factory_receives_on_drift_when_provider_pool_set` (D6)
- `tests/test_topology_runner.py::test_all_sentinels_produces_cold_start_note` (D2)
- `tests/test_topology_runner.py::test_planner_injection_on_by_default` (D5)
- `tests/test_topology_runner.py::test_planner_injection_opt_out_via_env` (D5)
- `tests/test_topology_controller.py::test_is_empty_or_error_detects_sentinel_string` (D3)
- `tests/test_topology_controller.py::test_sentinel_triggers_reroute_not_continue` (D3)
- `tests/test_topology_controller.py::test_sentinel_respects_max_reroute_budget` (D3)
- `tests/test_topology_controller.py::test_agent_loop_exhaustion_dataclass_exposes_fields` (D4)
- `tests/test_topology_controller.py::test_agent_loop_budget_exhausted_wraps_detail` (D4)
- `tests/test_swebench_bench.py::test_extract_patch_returns_empty_for_sentinel` (D7)
- `tests/test_swebench_bench.py::test_classify_prediction_dict_form_reads_structured_failure` (D7)

### Interaction between fixes

- D1 + D8 combine: S1 nodes now have `max_steps=5` and `stall_after_tool_steps=2`, so a thrashing synthesizer breaks after 2 consecutive tool steps instead of 20.
- D4 + D3: when D4 populates `loop.last_exhaustion`, D3's controller detects the sentinel string output and triggers reroute on first occurrence, continue on second.
- D2 + D5: if the coder produces a sentinel but the planner succeeded, D2's cold-start note + D5's planner injection ensure the synthesizer still sees structured upstream context.
- D6 + circuit breaker: drift score > 0.4 triggers `SWITCH_MODEL`, which now calls `ProviderPool.record_failure` → after 3 drift events in a row, the circuit breaker opens and the next node resolves to a different provider.
- D7 is bench-side only — zero impact on runtime behavior, affects reporting only.

### Smoke v1 (initial caps too aggressive)

First smoke after D1-D8: `stall_cap = max_steps // 2`. Result: **0/5 real patches**
vs baseline **3/5**. Regression because:
- S1 planner (max=5, cap=2) bailed at 2 tool steps before producing plans.
- S2 coder (max=10, cap=5) bailed at 5 before completing grep+read+edit cycle.
- S1 synthesizer (max=5, cap=2) bailed at 2 without emitting patches.

**Lesson**: D8 was catching normal SWE-bench exploration, not pathological thrash.
SWE-Lite coders need 8-15 tool calls before content emission.

### Smoke v2 (revised caps) — RESULTS

Revised heuristic (`agent_loop_factory.py:109-128`):
- S1 (max=5): `stall_cap=0` — D8 DISABLED (budget too tight for a stall window).
- S2 (max=10): `stall_cap=7` — 7 consecutive tool calls without content, then bail.
- S3 (max=20): `stall_cap=17` — catches 20-for-20 thrash, preserves exploration budget.

Log: `docs/benchmarks/2026-04-18-swebench-smoke-v2.log`

| Metric | Baseline_v2 (honest) | Smoke v2 | Delta |
|---|---|---|---|
| Real patches | 3/5 (incl. 1 sentinel-as-fake-patch = actually 2 real) | 1/5 | -1 (or −2 vs reported) |
| Sentinels (honest) | 3 (1 hidden as fake PATCH) | 3 | — |
| Empty (honest) | 0 | 1 | +1 |
| Timeouts | 1 | 0 | **−1 (D8 wins)** |
| Time/task avg | ~180s | ~55s | **3× faster** |

Tasks overlap with baseline: astropy-14995 and astropy-6938.
- `astropy-14995`: baseline = sentinel reported as 52-char fake PATCH → v2 = EMPTY (honest).
- `astropy-6938`: baseline = real 474-char patch → v2 = EMPTY (REGRESSION).

### Honest verdict

- **Visibility win**: the 1 fake PATCH that baseline emitted is now correctly classified EMPTY. D7 eliminates the metric pollution that masked failure as success.
- **Throughput win**: total wall time dropped ~3× because D8 breaks thrash early (preserving budget where content was emerging, killing where stuck).
- **No timeout**: the 1 baseline timeout (django-10914) is gone — D8 bails before 300s hits.
- **Raw real-patch regression**: 2 baseline reals → 1 v2 real. astropy-6938 coder now bails at stall-cap instead of completing its 474-char patch.

### Follow-ups (new tasks)

- **F1**: Relax S2 stall_cap from 7 → 9 (leave 1 step for final content). Hypothesis: astropy-6938 recovers if coder gets 2 extra tool calls.
- **F2**: Investigate why 3 of 5 v2 tasks still produce sentinel (not empty) even with D2 cold-start + D5 planner injection. Smoke v2 log has the details.
- **F3**: Docker-eval the 1 v2 patch (astropy-14365, 647 chars) to verify it's semantically correct, not just syntactically a diff.
- **F4**: Run smoke v3 on the SAME 5 tasks as baseline_v2 (offset adjustment) for apples-to-apples, since smoke v2 landed on different tasks.

---

## 5. What this audit did NOT cover

- Per-model routing accuracy on astropy-14995 (config.model was wired Apr 18; audit assumes it works).
- Whether kNN actually fired silently (no log line — needs `log.info` added at `routing/knn.rs`).
- Whether `bench/swebench_bench.py` Docker eval would have validated a real patch.
- Meta-Harness on top of a fixed harness (deferred per user decision Apr 18).

---

## References

- Log: `/tmp/baseline_v2.log` (mirror) / `C:\Users\yann.abadie\AppData\Local\Temp\baseline_v2.log`
- `sage-core/src/topology/templates.rs:27–70` (sequential factory)
- `sage-core/src/topology/topology_graph.rs:141–410` (TopologyNode.system + TopologyEdge)
- `sage-python/src/sage/agent_loop_factory.py:53–136` (max_steps by system_level)
- `sage-python/src/sage/pipeline.py:961–987` (partial binding system_level=ctx.system)
- `sage-python/src/sage/topology/runner.py:183–240, 278–310, 474–495, 1020–1230`
- `sage-python/src/sage/phases/learn.py:94–116` (EMPTY_STEP_SENTINEL)
- `sage-python/src/sage/topology_controller.py:40–75` (5 actions + thresholds)
- `sage-python/src/sage/monitoring/drift.py:49–111` (SWITCH_MODEL classification)
- `sage-python/src/sage/agent_loop.py:191–207` (drift event — log-only)

## Cross-review

- Advisor (Claude Opus) via `advisor()` tool: flagged D1 as unverified in initial draft; caller at `pipeline.py:969` then confirmed.
- Codex gpt-5.4 xhigh via `consult:consult-agent` (~10min, 8.2M input tokens, file:line citations required): confirmed A+B+C sections, corrected memory-writes claim, surfaced D6/D7/D8 (drift actionable, bench sentinel filter, early content breaker).
