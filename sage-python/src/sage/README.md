# sage

Root package for the YGN-SAGE Python SDK. Exports the top-level API: `Agent`, `AgentConfig`, `LLMConfig`, `Tool`, `ToolRegistry`, `ToolResult`.

## Core Modules

- **`boot.py`** -- `boot_agent_system()` and `AgentSystem` dataclass. Wires all five pillars (Topology, Tools, Memory, Evolution, Strategy), EventBus, and GuardrailPipeline into a single control plane. Entry point for production usage.
- **`agent.py`** -- `Agent` and `AgentConfig`. Core LLM-Tool loop runtime with working memory, sandbox, and process reward model. Tracks AIO (Agentic Infrastructure Overhead) stats.
- **`agent_loop.py`** -- `AgentLoop` with structured perceive-think-act-learn cycle. Integrates S2 AVR (empirical validation), Z3 S3 (formal verification), guardrails (input/runtime/output), S-MMU semantic context injection, and CircuitBreaker fault isolation. Emits `AgentEvent` to the EventBus.
- **`agent_pool.py`** -- `SubAgentSpec` and `AgentPool`. Dynamic sub-agent lifecycle management: register, deregister, list, mark running, store results, collect for ensemble.
- **`orchestrator.py`** -- `CognitiveOrchestrator`. Legacy capability-based multi-provider routing using `ModelRegistry`.
- **`resilience.py`** -- `CircuitBreaker`. Per-subsystem failure tracking (default `max_failures=3`). After 3 consecutive failures, the circuit opens and calls are skipped with a WARNING log. `record_success()` resets the counter.
- **`pipeline.py`** -- `CognitiveOrchestrationPipeline`: 5-stage orchestration (Classify → Decompose → Select Topology → Assign Models → Execute). Driven by ModelCards. EventBus observability at each stage.
- **`pipeline_stages.py`** -- Pure stage functions: `_infer_domain`, `compute_dag_features` (ω,δ,γ from AdaptOrch), `select_macro_topology`.
- **`topology_controller.py`** -- `TopologyController`: runtime adaptation for Pipeline Stage 4. 4 actions: model upgrade, agent pruning, topology re-route, sub-agent spawn.
- **`consistency.py`** -- `ConsistencyScore`: mean pairwise cosine similarity for parallel output comparison. Rust SIMD `batch_cosine_similarity` when available.

## Subpackages

| Package | Purpose |
|---------|---------|
| `agents/` | Composition patterns (sequential, parallel, loop, handoff) |
| `contracts/` | TaskNode IR, DAG, Z3 verification, policy, executor, planner, repair |
| `memory/` | 4-tier memory system (working, episodic, semantic, ExoCortex) |
| `llm/` | LLM provider abstraction and model routing |
| `providers/` | Provider discovery, capability matrix, OpenAI-compat |
| `strategy/` | AdaptiveRouter (4-stage learned routing), KnnRouter (92% accuracy), ComplexityRouter (heuristic fallback) |
| `topology/` | MAP-Elites topology evolution, KG-RLVR process reward |
| `evolution/` | Evolutionary engine, LLM mutation, population management |
| `tools/` | Tool registry, built-in tools, memory tools, ExoCortex tools |
| `events/` | EventBus (emit/subscribe/stream/query) |
| `guardrails/` | GuardrailPipeline, CostGuardrail, SchemaGuardrail |
| `bench/` | EvalPlus HumanEval+/MBPP+, routing accuracy/quality, ablation, evaluation protocol |
| `sandbox/` | SandboxManager (host execution disabled by default) |
| `routing/` | ShadowRouter (dual Rust/Python traces) |
| `monitoring/` | Drift detection |
