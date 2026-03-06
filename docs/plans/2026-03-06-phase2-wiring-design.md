# Phase 2: Wiring + HumanEval + ExoCortex Auto — Design Document

**Date**: 2026-03-06
**Author**: Yann Abadie + Claude Opus 4.6
**Status**: Approved
**Prerequisite**: v2 Convergence (Tasks 0-7) completed

---

## Goal

Wire the remaining unconnected components into the agent loop, make ExoCortex
auto-configured, and deliver a real HumanEval benchmark. After this phase,
every component in the system is connected and exercised.

## Changes

### 1. ExoCortex Auto-Configuration

The store name is hardcoded as default. No env var needed for normal operation.

```python
# remote_rag.py
DEFAULT_STORE = "fileSearchStores/ygnsageresearch-wii7kwkqozrd"

class ExoCortex:
    def __init__(self, store_name=None):
        self.store_name = store_name or os.environ.get("SAGE_EXOCORTEX_STORE") or DEFAULT_STORE
```

Resolution order: explicit param > env var > hardcoded default.

### 2. Guardrails Wired into Agent Loop

Three insertion points in agent_loop.py:

- **Input (after PERCEIVE, before THINK loop)**: run InputGuardrails on task text
- **Runtime (before sandbox execute in S2 AVR)**: run RuntimeGuardrails on code
- **Output (before return)**: run OutputGuardrails on result (cost, schema)

GuardrailPipeline injected via boot.py. Events emitted: GUARDRAIL_CHECK, GUARDRAIL_BLOCK.

### 3. SemanticMemory + MemoryAgent Wired

- **Before loop**: inject SemanticMemory.get_context_for(task) as system message
- **LEARN phase**: call MemoryAgent.extract(response) -> SemanticMemory.add_extraction()
- Both are best-effort (try/except, no crash)

### 4. HumanEval Benchmark

New sage/bench/humaneval.py:
- Load 164 problems from bundled JSON (no external dependency)
- Submit each to AgentSystem.run()
- Extract code from response
- Execute tests in subprocess sandbox
- Measure pass@1, latency, cost, routing breakdown
- Emit BENCH_* events on EventBus

## Tasks

| # | Description | Files |
|---|-------------|-------|
| 8 | ExoCortex auto-config + is_available fix | remote_rag.py, boot.py |
| 9 | Wire guardrails into agent_loop | agent_loop.py, boot.py |
| 10 | Wire semantic memory + memory agent | agent_loop.py, boot.py |
| 11 | HumanEval benchmark | sage/bench/humaneval.py, __main__.py |
| 12 | Integration tests for all wiring | tests/ |

Tasks 8-10 are parallelizable. Task 11 depends on 8-10. Task 12 validates all.
