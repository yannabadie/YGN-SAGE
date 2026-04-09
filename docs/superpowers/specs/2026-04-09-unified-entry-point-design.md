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

## Risk

- 2001 existing tests may depend on legacy path behavior
- Phase 1 must be validated against full test suite before Phase 2
- Agent_loop.run() per node is slower than provider.generate() per node
  (but produces REAL agents, not fake prompt chains — OpenSAGE proves this works at 59% SWE-bench)

## Success Criteria

- `system.run()` has ONE code path (no if/else)
- Every topology node can call tools (execute_bash at minimum)
- S2/S3 validation works for all nodes (not just legacy path)
- 2001+ tests pass with 0 regressions
- SWE-bench produces patches with real code exploration (not blind generation)
