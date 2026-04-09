# Unified Entry Point — Design Spec

**Date:** 2026-04-09
**Status:** Approved
**Goal:** One execution path through SAGE. `system.run()` = `pipeline.run()`. Every node is a real agent with tools.

## Problem

4 execution paths exist with different capabilities:

| Path | Routing | Topology | Tools | Validation | Guardrails |
|------|---------|----------|-------|------------|------------|
| Pipeline | Rust SystemRouter | Yes | Yes (since apr 8) | No | No |
| Agent loop | None | No | Yes | Yes (S2/S3) | Yes |
| Legacy | Rust/Python | Partial | Yes | Yes | Yes |
| Mock | None | No | No | No | No |

No path has everything. The pipeline lacks validation/guardrails. The agent_loop lacks routing/topology. The legacy path is 400 lines of duplication.

## Solution

Pipeline orchestrates (routing, topology, model assignment). Agent_loop executes (tools, validation, guardrails, memory). One path.

### Architecture

```
system.run(task) → pipeline.run(task, budget)
  Stage 0: CLASSIFY        → Rust SystemRouter (kNN + bandit + domain + budget)
  Stage 1: DECOMPOSE       → TaskPlanner LLM → DAG features (omega, delta, gamma)
  Stage 2: SELECT TOPOLOGY  → Rust TopologyEngine 6-path → topology graph
  Stage 3: ASSIGN MODELS    → Rust ModelAssigner (cards.toml affinity scoring)
  Stage 4: EXECUTE
    bypass (topology=None):  → agent_loop.run(task) — full agent
    multi-node:              → TopologyRunner → agent_loop.run(node_prompt) per node
  Stage 5: LEARN            → bandit + archive + evolution memory
```

### Each node = real agent

Every topology node executes via `agent_loop.run()` which provides:
- Tool-calling loop (execute_bash, create_python_tool, memory tools)
- S2 validation (AVR: act-verify-refine with test feedback)
- S3 validation (Z3 PRM formal verification)
- Guardrail pipeline (input/output/runtime checks)
- Memory operations (working memory read/write, episodic store)

Tools are filtered by node role:
- **actor**: all tools (execute_bash, create_python_tool, memory)
- **verifier**: execute_bash (can run tests), memory
- **output_formatter**: memory only (no code execution)

### Shared state between nodes

1. **Context**: TopologyRunner passes predecessor_output as context in node prompt
2. **Budget**: pipeline splits budget proportionally by model cost estimate per node
3. **Memory**: all nodes share the same working_memory (Rust Arrow STM). Episodic scoped by node_id.

### What gets removed

- `boot.py` legacy path (lines 129-280): routing + topology + agent_loop setup → ~150 lines
- `agent_loop._run_legacy()` + supporting functions → ~400 lines
- `agent_loop_execution.py` legacy functions (legacy_think_step, run_legacy_s3, run_legacy_avr) → ~200 lines
- `SAGE_AGENT_LOOP_LEGACY` env var
- The `if self.pipeline ... else legacy` branch in `system.run()`
- Mock bypass (mock goes through pipeline too)

### What stays

- `agent_loop.run()` (non-legacy version, lines 228-260) — the execution engine for each node
- S2/S3 validation in agent_loop — used by every node
- Guardrails in agent_loop — used by every node  
- Tool-calling in agent_loop — used by every node
- TopologyRunner — orchestrates multi-node DAG execution order
- All Rust components (SystemRouter, TopologyEngine, ModelAssigner, kNN, bandit)

## Implementation Phases

### Phase 1: Pipeline calls agent_loop for bypass (small change, big impact)
- Stage 4 bypass: `agent_loop.run(task)` instead of `provider.generate()`
- Remove the `if pipeline ... else legacy` in system.run()
- system.run() = pipeline.run(). Always.
- Mock mode: pipeline with mock provider (no special case)

### Phase 2: TopologyRunner nodes = agent_loop (medium change)
- TopologyRunner: each node calls `agent_loop.run(node_prompt)` 
- Tools filtered by node role
- Budget split per node
- Shared working memory

### Phase 3: Delete legacy code (cleanup)
- Delete _run_legacy() + legacy_think_step + run_legacy_s3 + run_legacy_avr
- Delete SAGE_AGENT_LOOP_LEGACY env var
- Audit tests for legacy path dependencies

## Identified Integration Hazards (from codebase study)

### H1: Double routing
`perceive()` in agent_loop calls `metacognition.assess_complexity_async()` (Python router).
The pipeline already routed via Rust SystemRouter. Two different routing results.
**Fix:** when pipeline calls agent_loop.run(), set `loop._skip_routing = True` and inject
the Rust routing decision (system, model_id) into the loop context.

### H2: State reset per-node
agent_loop.run() resets `total_cost_usd = 0.0` and `step_count = 0` (line 247).
In multi-node mode, cost tracking across nodes is lost.
**Fix:** create a fresh agent_loop config per node with `max_cost_usd = budget_remaining / nodes_left`.
Or pass a shared cost accumulator.

### H3: Double tool-calling loop
Pipeline Stage 4 has a tool-calling loop (max 30 turns, lines 870-910).
agent_loop.run() has its own THINK→ACT loop (max_steps).
If pipeline calls agent_loop.run(), the pipeline loop is redundant.
**Fix:** in bypass mode, pipeline calls agent_loop.run() directly (no pipeline tool loop).
DELETE the pipeline tool-calling loop entirely — agent_loop handles it.

### H4: Triple topology execution
think() (line 54) checks `step_count == 1` and calls `loop._run_topology(task)`.
The pipeline already executed the topology in Stage 2.
**Fix:** when pipeline calls agent_loop.run(), set `loop._current_topology = None` so
think() skips topology execution. The pipeline owns topology, not the agent_loop.

### H5: Mock mode test dependency
2001 tests use `use_mock_llm=True` which bypasses the pipeline.
Making mock go through pipeline could break test expectations.
**Fix:** keep mock provider working within LiteLLMProvider (mock responses).
The pipeline path should work with mock providers — validate with full test suite.

### H6: Recursive validation (Codex finding)
A "verifier" node running through agent_loop gets S2 AVR validation applied to IT.
The verifier validates the actor's code, then AVR validates the verifier's output,
triggering another verification loop. Recursive.
**Fix:** verifier nodes should run with `validation_level=0` (no AVR/Z3). Only the
"actor" node gets full validation. Set per-node via agent_loop config.

### H7: Predecessor context injection (Codex finding)
agent_loop.run(task) takes a string task. It has no mechanism to inject
predecessor_output from the previous topology node.
**Fix:** TopologyRunner builds the node prompt as `f"{node_system_prompt}\n\n## Previous agent output:\n{predecessor_output}\n\n## Task:\n{task}"`.
This is already done for provider.generate() — same pattern for agent_loop.run().

### H8: Async concurrency (Codex finding)
TopologyRunner uses asyncio.gather() for independent nodes (same DAG depth).
Multiple agent_loop.run() calls concurrently on the same event loop could conflict
on shared state (working_memory, drift_monitor, circuit breakers).
**Fix:** per-node AgentLoop factory creates INDEPENDENT instances. No shared mutable
state. Working memory shared READ-ONLY; each node writes to its own episodic scope.

### H9: Mock mode test dependency (Codex finding)
2001 tests use `use_mock_llm=True` which currently bypasses the pipeline entirely.
Forcing mock through pipeline could break test expectations (different output format,
different event sequence, different cost tracking).
**Fix:** keep mock bypass as a TESTED EXCEPTION in system.run():
`if mock: return agent_loop.run(task)` — one if, not a full legacy path.
Migrate tests incrementally to pipeline-aware mocks in Phase 3.

## Codex Migration Strategy (8 ordered steps)
1. Keep mock bypass (don't break 2001 tests)
2. Add _skip_routing flag before structural changes
3. Create AgentLoop factory for per-node instances (fresh state)
4. Clear _current_topology = None before each node call
5. Pass predecessor_output in node prompt
6. Budget split via max_cost_usd per-node config
7. Delete pipeline tool-calling loop (agent_loop handles it)
8. Delete legacy code AFTER all fixes verified against full test suite

## Risk

- 2001 existing tests may depend on legacy path behavior
- Phase 1 must be validated against full test suite before Phase 2
- Agent_loop.run() per node is slower than provider.generate() per node
  (but produces REAL agents, not fake prompt chains — OpenSAGE proves this works at 59% SWE-bench)
- Google ADK pattern (SequentialAgent + LlmAgent per node) validates the architecture
- Codex found 10 issues (5 BLOCKING), all with identified fixes

## Success Criteria

- `system.run()` has ONE code path (no if/else)
- Every topology node can call tools (execute_bash at minimum)
- S2/S3 validation works for all nodes (not just legacy path)
- 2001+ tests pass with 0 regressions
- SWE-bench produces patches with real code exploration (not blind generation)
- No double routing, no double tool loop, no triple topology
