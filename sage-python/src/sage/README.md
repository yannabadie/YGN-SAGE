# sage

Root package for the YGN-SAGE Python SDK. Exports the top-level API: `Agent`, `AgentConfig`, `LLMConfig`, `Tool`, `ToolRegistry`, `ToolResult`.

## Core Modules

- **`boot.py`** -- `boot_agent_system()` and `AgentSystem` dataclass. Wires all five pillars (Topology, Tools, Memory, Evolution, Strategy), EventBus, and GuardrailPipeline into a single control plane. Entry point for production usage.
- **`agent.py`** -- `Agent` and `AgentConfig`. Core LLM-Tool loop runtime with working memory, sandbox, and process reward model. Tracks AIO (Agentic Infrastructure Overhead) stats.
- **`agent_loop.py`** -- `AgentLoop` with structured perceive-think-act-learn cycle. Integrates S2 AVR (empirical validation), Z3 S3 (formal verification), guardrails (input/runtime/output), S-MMU semantic context injection, and CircuitBreaker fault isolation. Emits `AgentEvent` to the EventBus.
- **`agent_pool.py`** -- `SubAgentSpec` and `AgentPool`. Dynamic sub-agent lifecycle management: register, deregister, list, mark running, store results, collect for ensemble.
- **`orchestrator.py`** -- `CognitiveOrchestrator`. Legacy capability-based multi-provider routing using `ModelRegistry`.
- **`resilience.py`** -- `CircuitBreaker`. Per-subsystem failure tracking (default `max_failures=3`). After 3 consecutive failures, the circuit opens and calls are skipped with a WARNING log. `record_success()` resets the counter.
- **`evidence.py`** -- `EvidenceLevel` enum (HEURISTIC through EMPIRICALLY_VALIDATED) and `EvidenceRecord` dataclass. Tracks proof strength, coverage, and assumptions for every claim.

## Subpackages

| Package | Purpose |
|---------|---------|
| `agents/` | Composition patterns (sequential, parallel, loop, handoff) |
| `contracts/` | TaskNode IR, DAG, Z3 verification, policy, executor, planner, repair |
| `memory/` | 4-tier memory system (working, episodic, semantic, ExoCortex) |
| `llm/` | LLM provider abstraction and model routing |
| `providers/` | Provider discovery, capability matrix, OpenAI-compat |
| `strategy/` | ComplexityRouter (S1/S2/S3), resource allocation |
| `topology/` | MAP-Elites topology evolution, KG-RLVR process reward |
| `evolution/` | Evolutionary engine, LLM mutation, population management |
| `tools/` | Tool registry, built-in tools, memory tools, ExoCortex tools |
| `events/` | EventBus (emit/subscribe/stream/query) |
| `guardrails/` | GuardrailPipeline, CostGuardrail, SchemaGuardrail |
| `bench/` | BenchmarkRunner, HumanEval, routing accuracy |
| `sandbox/` | SandboxManager (host execution disabled by default) |
| `routing/` | ShadowRouter (dual Rust/Python traces) |
| `analytics/` | Scaling analysis utilities |
| `monitoring/` | Drift detection |
