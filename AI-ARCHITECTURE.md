# AI-ARCHITECTURE.md — YGN-SAGE System Reference

> **Audience**: LLMs (Claude, GPT, Gemini, etc.) consuming this file as sole context for code reasoning, bug diagnosis, and refactoring proposals.
> **Generated**: 2026-03-23 from VeRLGIGPO branch (eb57aa6).

---

## Table of Contents

- [Mental Model](#mental-model)
- [Architecture Diagram](#architecture-diagram)
- [Component Registry](#component-registry)
- [Data Flow Narratives](#data-flow-narratives)
- [Key Modules Reference](#key-modules-reference)
- [Research References](#research-references)
- [LLM Quick-Reference Cheatsheet](#llm-quick-reference-cheatsheet)

---

## Mental Model

YGN-SAGE is an **Agent Development Kit (ADK)** that orchestrates multi-agent LLM systems via learned topology graphs. Given an arbitrary task, it classifies cognitive difficulty (S1/S2/S3 Kahneman systems), generates a directed multi-agent graph (topology), assigns heterogeneous LLM models to each node, executes the graph, and feeds outcomes back into evolutionary and bandit learning loops. The core thesis: **topology structure matters more than model capability** on hard tasks (AdaptOrch 2602.16873: Var_tau/Var_M >= 20).

**Stack**: Rust crate `sage-core` (PyO3 bindings) handles performance-critical paths: topology engine (petgraph), kNN/SystemRouter routing, S-MMU memory (Arrow columnar), SMT verification (OxiZ), MCTS/MAP-Elites/CMA-ME evolution. Python package `sage-python` handles orchestration: 5-stage pipeline, agent loop, 8 LLM providers, benchmarks, veRL GiGPO training. `sage-discover` is a knowledge pipeline (arXiv to ExoCortex RAG).

**Invariants**: (1) Rust first for performance, Python for orchestration only. (2) Zero heuristics -- every threshold is formally verified (OxiZ SAT/UNSAT), learned (ONNX), or research-backed. (3) kNN router is primary (92% ground truth accuracy); the old ComplexityRouter heuristic (34% GT) is dead code. (4) TopologyGraph is the unified IR -- petgraph DiGraph with three-flow edges (Control, Message, State). (5) All quality labels come from QualityLabeler (Rust, OxiZ formal proofs), not heuristic scoring.

**Known debts**: Python shadow modules not yet deleted (need <5% divergence gate from 1090 traces, currently 49.6%). Evolution lacks quantitative evidence (need N>=10 Wilcoxon). Path 6 (learned topology policy) is 70% YAML valid. RustEntityGraph reserved but not wired. DeBERTa quality estimator superseded by ModernBERT (backlog).

---

## Architecture Diagram

```mermaid
flowchart TD
    subgraph sage-core["sage-core (Rust + PyO3)"]
        TE[TopologyEngine<br/>7-path generation]
        TG[TopologyGraph<br/>petgraph DiGraph IR]
        TX[TopologyExecutor<br/>static/dynamic scheduler]
        ME[MAP-Elites Archive<br/>quality-diversity]
        CMA[CMA-ME Emitter<br/>continuous params]
        MCTS_R[MCTS Searcher<br/>UCB1 topology search]
        MUT[Mutations<br/>7 operators]
        TMPL[Templates<br/>8 built-in patterns]
        LLM_SYN[LLM Synthesis<br/>3-stage parse+build]
        HV[HybridVerifier<br/>O(V+E) structural+semantic]
        LTL[LtlVerifier<br/>temporal model checking]
        SMT[SmtVerifier<br/>OxiZ QF_LIA]
        QL[QualityLabeler<br/>formal code scoring]
        SR[SystemRouter<br/>S1/S2/S3 + domain scoring]
        KNN_R[RustKnnRouter<br/>cosine kNN + OOD reject]
        MA[ModelAssigner<br/>affinity+domain+cost]
        MR[ModelRegistry<br/>cards.toml 20 models]
        CB[ContextualBandit<br/>Thompson Beta/Gamma]
        SMMU[S-MMU<br/>4-view memory graph]
        WM[WorkingMemory<br/>Arrow STM + S-MMU]
        EMB[RustEmbedder<br/>arctic-embed-m ONNX]
        EG[RustEntityGraph<br/>semantic+causal]
        RG[RustRelevanceGate<br/>keyword overlap]
        RC[RagCache<br/>LRU embedding cache]
        TE_X[ToolExecutor<br/>Wasm WASI + subprocess]
        TD_R[TopologyDensity<br/>S_complex score]
        TR_R[TopologyReward<br/>multi-signal reward]
    end

    subgraph sage-python["sage-python (Python SDK)"]
        PIPE[CognitiveOrchestrationPipeline<br/>5-stage: classify->execute->learn]
        BOOT[boot.py<br/>AgentSystem factory]
        AL[AgentLoop<br/>perceive->think->act->learn]
        AR[AdaptiveRouter<br/>4-stage cascade]
        KNN_P[KnnRouter<br/>Python wrapper + Rust hot-path]
        PP[ProviderPool<br/>model_id -> provider]
        TR[TopologyRunner<br/>execute graph as agents]
        TC[TopologyController<br/>runtime adaptation]
        LC[LlmCaller<br/>topology synthesis prompts]
        TA[TopologyArchive<br/>QD archive Python]
        PRM[ProcessRewardModel<br/>KG-RLVR Z3-backed]
        EVO[EvolutionEngine<br/>MAP-Elites + SAMPO]
        EP[EpisodicMemory<br/>SQLite cross-session]
        EXO[ExoCortex<br/>Google GenAI RAG]
        MC[MemoryCompressor<br/>LLM summarization]
        DM[DriftMonitor<br/>sliding-window drift]
        SR_P[ShadowRouter<br/>Rust/Python dual traces]
        QE[QualityEstimator<br/>Python heuristic fallback]
    end

    subgraph verl["sage-python/verl (GiGPO Training)"]
        ENV[SageTopologyEnv<br/>4-state gym env]
        REW[reward.py<br/>format+structure+exec]
        EC[edge_credit.py<br/>Graph-GRPO advantages]
        RF[RewardFlowPropagator<br/>PageRank per-node credit]
        SRV[StepRewardVector<br/>per-step GiGPO rewards]
        TM[TrainingMemory<br/>SQLite episodic]
    end

    subgraph sage-discover["sage-discover"]
        DISC[Discovery Pipeline<br/>arXiv -> ExoCortex]
        CUR[Curator<br/>paper relevance filter]
        ING[Ingestion<br/>PDF -> chunks]
    end

    %% Main pipeline flow
    PIPE -->|"Stage 0: classify"| AR
    AR -->|"Stage 0.5: kNN"| KNN_P
    KNN_P -->|"Rust hot-path"| KNN_R
    AR -->|"fallback"| SR
    PIPE -->|"Stage 2: topology"| TE
    TE --> TG
    TE --> ME
    TE --> MCTS_R
    TE --> CMA
    TE --> MUT
    TE --> TMPL
    TE --> LLM_SYN
    LLM_SYN -.->|"Python callback"| LC
    TG --> HV
    HV --> LTL
    PIPE -->|"Stage 3: assign"| MA
    MA --> MR
    PIPE -->|"Stage 4: execute"| TR
    TR --> TX
    TR --> PP
    PIPE -->|"Stage 5: learn"| CB
    PIPE -->|"Stage 5: quality"| QL
    QL --> SMT
    QL --> TE_X

    %% Memory flow
    AL --> WM
    WM --> SMMU
    SMMU --> EMB
    AL --> EP
    AL --> EXO

    %% Training flow
    ENV -->|"step rewards"| SRV
    ENV -->|"episode traces"| REW
    REW --> EC
    REW --> RF
    ENV --> TM

    %% Discover flow
    DISC --> CUR
    CUR --> ING
    ING --> EXO

    %% Reward signals
    TR_R --> REW
    TD_R --> TR_R
```

---

## Component Registry

### sage-core (Rust)

| Component | Type | Responsibility | Internal Deps | Exposes (PyO3) |
|-----------|------|---------------|---------------|-----------------|
| `topology/engine.rs` | Struct `DynamicTopologyEngine` | 7-path topology generation orchestrator (S-MMU hit, archive hit, LLM synthesis, mutation, MCTS, Path 6, template fallback) | `smmu`, `map_elites`, `mcts`, `cma_me`, `mutations`, `templates`, `llm_synthesis`, `verifier`, `bandit` | `PyTopologyEngine` |
| `topology/topology_graph.rs` | Struct `TopologyGraph` | Unified IR: petgraph DiGraph with TopologyNode/TopologyEdge, three-flow model (Control/Message/State), gate-based edge blocking | `petgraph`, `ulid` | `TopologyGraph`, `TopologyNode`, `TopologyEdge` |
| `topology/executor.rs` | Struct `TopologyExecutor` | Dual-mode scheduler: Static (Kahn's toposort O(V+E)) for DAGs, Dynamic (gate-based readiness polling) for cyclic topologies | `topology_graph` | `PyTopologyExecutor` |
| `topology/map_elites.rs` | Struct `MapElitesArchive` | 4D behavior-descriptor grid (108 cells), Pareto dominance insertion, HybridVerifier gate | `topology_graph`, `verifier` | -- (via engine) |
| `topology/cma_me.rs` | Struct `CmaEmitter` | Diagonal-covariance CMA for 3D continuous params (cost, wall time, edge weight), Box-Muller sampling | -- | -- (via engine) |
| `topology/mcts.rs` | Struct `MctsSearcher` | UCB1 tree search over mutation space, heuristic rollout scoring, no LLM calls | `mutations`, `topology_graph`, `verifier` | -- (via engine) |
| `topology/mutations.rs` | Fn `apply_random_mutation` | 7 topology mutation operators, each validated by HybridVerifier before return | `topology_graph`, `verifier` | -- (via engine) |
| `topology/templates.rs` | Struct `PyTemplateStore` | 8 factory functions: Sequential, Parallel, AVR, SelfMoA, Hierarchical, Hub, Debate, Brainstorming | `topology_graph` | `PyTemplateStore` |
| `topology/llm_synthesis.rs` | Struct `TopologySynthesizer` | 3-stage parse pipeline (role JSON -> structure JSON -> graph build + verify), rate limiting | `topology_graph`, `verifier` | -- (Python callback) |
| `topology/verifier.rs` | Struct `HybridVerifier` | O(V+E) structural + semantic checks: DAG cycle detection, orphan nodes, role/switch/loop/reachability, LTL temporal | `topology_graph`, `ltl` | `PyHybridVerifier`, `VerificationResult` |
| `topology/density.rs` | Struct `TopologyDensity` | S_complex from AgentConductor: S_node + S_edge + S_depth, difficulty-aware N_max bounds (S1=4, S2=7, S3=10) | `topology_graph` | `TopologyDensity`, `DensityScore` |
| `topology/reward.rs` | Struct `TopologyReward` | Multi-signal reward: execution (pass@1) + structural (verifier) + density (S_complex) + temporal (LTL) + resilience + cost efficiency | `density`, `verifier`, `ltl` | `TopologyReward`, `RewardScore` |
| `routing/system_router.rs` | Struct `SystemRouter` | S1/S2/S3 classification from StructuralFeatures + formal keyword detection, domain-aware model selection from ModelRegistry, budget constraint, telemetry history | `features`, `model_card`, `model_registry`, `bandit` | `SystemRouter`, `RoutingDecision`, `RoutingConstraints` |
| `routing/knn.rs` | Struct `RustKnnRouter` | L2-normalize + cosine dot product, top-k partial sort, distance-weighted majority vote, OOD rejection | -- | `RustKnnRouter` |
| `routing/model_assigner.rs` | Struct `ModelAssigner` | Per-node model assignment: score = 0.4*affinity + 0.4*domain + 0.2*cost, budget-aware allocation | `model_registry`, `topology_graph` | `ModelAssigner` |
| `routing/model_card.rs` | Struct `ModelCard` | Single model description: id, provider, CognitiveSystem, domain scores, cost, capabilities, context window | -- | `ModelCard`, `CognitiveSystem` |
| `routing/model_registry.rs` | Struct `ModelRegistry` | cards.toml parser, model lookup by id/system/capabilities | `model_card` | `ModelRegistry` |
| `routing/bandit.rs` | Struct `ContextualBandit` | Per-arm Beta posteriors (quality) + Gamma posteriors (cost/latency), Thompson sampling, Pareto front selection | -- | `ContextualBandit`, `BanditDecision` |
| `routing/quality.rs` | Struct `RustQualityEstimator` | 5-signal lexical quality (non-empty, length, code presence, no error patterns, AVR convergence) | -- | `RustQualityEstimator` |
| `routing/features.rs` | Struct `StructuralFeatures` | Keyword-based complexity/uncertainty scoring (Stage 0 structural analysis) | -- | `StructuralFeatures` |
| `verification/smt.rs` | Struct `SmtVerifier` | OxiZ QF_LIA: memory safety, loop bounds, arithmetic verification, invariant pre/post-condition, provider assignment SAT, CEGAR feedback loop | `oxiz` | `SmtVerifier`, `SmtVerificationResult` |
| `verification/quality_labeler.rs` | Struct `QualityLabeler` | Formal code quality scoring: tree-sitter syntax validation + SmtVerifier proofs, zero heuristics | `smt`, `sandbox/validator` | `QualityLabeler`, `QualityLabel` |
| `verification/ltl.rs` | Struct `LtlVerifier` | 4 temporal properties on TopologyGraph: reachability, safety (no info flow high->low), liveness, bounded liveness; petgraph BFS/DFS | `topology_graph` | `LtlVerifier`, `LtlResult` |
| `memory/smmu.rs` | Struct `MultiViewMMU` | 4 orthogonal views in single DiGraph: Temporal (chronological), Semantic (cosine sim), Causal (parent-child), Entity (Jaccard keywords), ULID chunk IDs | `petgraph`, `ulid` | `PyMultiViewMMU` |
| `memory/mod.rs` (WorkingMemory) | Struct `WorkingMemory` | Per-agent memory: active buffer (Vec) + Arrow compacted chunks + S-MMU graph, compress/retrieve/evict | `smmu`, `arrow_tier`, `paging` | `WorkingMemory` |
| `memory/arrow_tier.rs` | Fn `compact_buffer_to_arrow` | Compact MemoryEvents into Arrow RecordBatch, register chunk in S-MMU | `smmu`, `arrow` | -- (via WorkingMemory) |
| `memory/embedder.rs` | Struct `RustEmbedder` | ONNX Runtime loader for arctic-embed-m (768-dim), L2-normalized output, DLL auto-discovery | `ort`, `tokenizers` | `RustEmbedder` (behind `onnx` feature) |
| `memory/entity_graph.rs` | Struct `RustEntityGraph` | Unified semantic + causal + temporal entity graph (petgraph), BFS neighborhood queries | `petgraph` | `RustEntityGraph` (reserved, not wired) |
| `memory/relevance_gate.rs` | Struct `RustRelevanceGate` | CRAG-style keyword overlap gate, stop word filtering, threshold=0.3 | -- | `RustRelevanceGate` |
| `memory/rag_cache.rs` | Struct `RagCache` | LRU cache for embedding query results | -- | `RagCache` |
| `sandbox/tool_executor.rs` | Struct `ToolExecutor` | Wasm WASI sandbox (wasmtime 36 LTS) + subprocess fallback, tree-sitter validation | `validator`, `subprocess`, `wasm` | `ToolExecutor` |
| `sandbox/validator.rs` | Fn `validate_python_code` | tree-sitter Python syntax validation, returns ValidationResult | `tree-sitter-python` | `ValidationResult` |
| `agent.rs` | Struct `AgentConfig` | Agent configuration (id, role, capabilities) | -- | `AgentConfig` |
| `pool.rs` | Struct `AgentPool` | Concurrent agent registry with DashMap | `agent` | `AgentPool` |
| `types.rs` | Enums | TopologyRole, MemoryScope, AgentStatus, ToolSpec | -- | all exported |

### sage-python

| Component | Type | Responsibility | Internal Deps | Exposes |
|-----------|------|---------------|---------------|---------|
| `pipeline.py` | Class `CognitiveOrchestrationPipeline` | 5-stage orchestration: Classify -> Decompose -> Select Topology -> Assign Models -> Execute, with LEARN feedback | `pipeline_stages`, `contracts.z3_verify` | `pipeline.run(task, budget)` |
| `pipeline_stages.py` | Functions | Pure functions: `_infer_domain()` (regex), `compute_dag_features()` (AdaptOrch omega/delta/gamma), `select_macro_topology()` | -- | `DAGFeatures` |
| `boot.py` | Function `boot()` | Full system factory: loads cards.toml, initializes Rust/Python routers, builds AgentSystem dataclass | all SDK modules | `AgentSystem` |
| `agent_loop.py` | Class `AgentLoop` | Per-agent runtime: perceive -> think -> act -> learn, tool calling, memory compression, cost tracking, stagnation detection | `llm.base`, `tools.registry`, `memory.working`, `memory.compressor`, `resilience` | `AgentLoop.run()` |
| `strategy/adaptive_router.py` | Class `AdaptiveRouter` | 4-stage cascade: structural features -> kNN embeddings -> BERT ONNX -> entropy probe, duck-type compat with ComplexityRouter | `strategy.metacognition`, `strategy.structural_features` | `AdaptiveRouter.route()` |
| `strategy/knn_router.py` | Class `KnnRouter` | Python kNN wrapper: loads routing_exemplars.npz, embeds task, delegates to RustKnnRouter for hot-path, OOD fallback | `sage_core.RustKnnRouter`, `memory.embedder` | `KnnRouter.route()` |
| `strategy/metacognition.py` | Classes | `ComplexityRouter` (legacy heuristic, dead code 34% GT), `CognitiveProfile`, `RoutingDecision` | -- | `RoutingDecision(system=1/2/3)` |
| `routing/shadow.py` | Class `ShadowRouter` | Dual Rust/Python routing with JSONL trace logging for divergence analysis | `sage_core.SystemRouter` | `ShadowRouter.route()` |
| `topology/runner.py` | Class `TopologyRunner` | Execute TopologyGraph as real multi-agent system: aggregate predecessor outputs -> build prompt -> LLM call -> store output, per-node provider resolution | `llm.base`, `sage_core.TopologyExecutor` | `TopologyRunner.run()` |
| `topology/llm_caller.py` | Functions | Build role assignment + structure design prompts for LLM topology synthesis (Path 3) | -- | `build_role_prompt()`, `build_structure_prompt()` |
| `topology/engine.py` | Class `Topology` | Legacy Python topology (Vertical/Horizontal/Mesh), largely superseded by Rust TopologyGraph | -- | -- |
| `topology/topology_archive.py` | Class `TopologyArchive` | Python QD archive: per-task-type best topology records | `topology_verifier` | `TopologyArchive.recommend()` |
| `topology/evo_topology.py` | Classes | `TopologyEvolver`, `TopologyPopulation` for evolutionary topology search (Python side) | -- | -- |
| `topology/kg_rlvr.py` | Class `ProcessRewardModel` | KG-RLVR: Z3-backed process reward for reasoning paths (<think> tags), uses Rust OxiZ or Python z3 | `sage_core.SmtVerifier` | `ProcessRewardModel.score()` |
| `topology/ltl_bridge.py` | Module | Python bridge to Rust LtlVerifier | `sage_core.LtlVerifier` | -- |
| `llm/base.py` | Classes | `LLMProvider` (abstract), `LLMConfig`, `Message`, `Role` | -- | base protocol |
| `llm/provider_pool.py` | Class `ProviderPool` | Resolve model_id -> (LLMProvider, LLMConfig) with caching + circuit breaker fallback | `llm.base`, `resilience` | `ProviderPool.resolve()` |
| `llm/model_card.py` | Class `ModelCard` | Python ModelCard (mirrors Rust, loaded from cards.toml) | -- | `ModelCard`, `CognitiveSystem` |
| `llm/model_registry.py` | Class `ModelCardCatalog` | Python model registry, cards.toml parser | `model_card` | `ModelCardCatalog` |
| `llm/router.py` | Class `ModelRouter` | Maps CognitiveSystem -> model_id via ModelCards | `model_card` | `ModelRouter.select()` |
| `llm/config_loader.py` | Module | Load provider configs from env/files | -- | -- |
| `llm/google.py` | Class | Google Gemini provider (google-genai SDK) | `llm.base` | -- |
| `llm/codex.py` | Class | OpenAI Codex CLI provider | `llm.base` | -- |
| `providers/registry.py` | Class `ModelRegistry` | Runtime discovery: 8 providers (Google, OpenAI, DeepSeek, xAI, Kimi, MiniMax, OpenRouter, Codex) | `providers.connector` | -- |
| `providers/openai_compat.py` | Class | OpenAI-compatible provider (DeepSeek, xAI, Kimi, MiniMax) | `llm.base` | -- |
| `memory/working.py` | Class `WorkingMemory` | Python working memory (list-based, legacy) | -- | -- |
| `memory/episodic.py` | Class `EpisodicMemory` | SQLite or in-memory cross-session episodic store | -- | `EpisodicMemory` |
| `memory/semantic.py` | Module | Entity-relation semantic memory (Python) | -- | -- |
| `memory/causal.py` | Module | Causal memory (Python) | -- | -- |
| `memory/compressor.py` | Class `MemoryCompressor` | LLM-based memory summarization | `llm.base` | `MemoryCompressor.compress()` |
| `memory/smmu_context.py` | Module | Python S-MMU context manager, bridges to Rust S-MMU | `sage_core.PyMultiViewMMU` | -- |
| `memory/embedder.py` | Class `Embedder` | Python embedder: delegates to RustEmbedder (arctic-embed-m ONNX) or sentence-transformers fallback | `sage_core.RustEmbedder` | `Embedder.embed()` |
| `memory/relevance_gate.py` | Class `RelevanceGate` | Python wrapper for RustRelevanceGate | `sage_core.RustRelevanceGate` | -- |
| `memory/remote_rag.py` | Class `ExoCortex` | Google GenAI File Search API persistent RAG | -- | `ExoCortex.query()` |
| `evolution/engine.py` | Class `EvolutionEngine` | MAP-Elites + SAMPO 5 strategic actions, LLM mutation, evaluation cascade | `evolution.population`, `evolution.mutator`, `evolution.evaluator`, `strategy.solvers` | `EvolutionEngine.run()` |
| `evolution/llm_mutator.py` | Class `LLMMutator` | LLM-as-mutation-operator (AlphaEvolve-style) | `llm.base` | -- |
| `quality_estimator.py` | Class `QualityEstimator` | Python heuristic quality (5-signal, legacy fallback for when Rust QualityLabeler unavailable) | -- | -- |
| `topology_controller.py` | Class `TopologyController` | Runtime adaptation: upgrade_model, spawn_subagent, reroute, prune decisions after each node | -- | `TopologyController.evaluate_and_decide()` |
| `contracts/z3_verify.py` | Function `verify_provider_assignment` | OxiZ SAT verification for provider-model assignment constraints | `sage_core.SmtVerifier` | -- |
| `contracts/dag.py` | Module | TaskDAG contract IR | -- | `TaskDAG` |
| `contracts/planner.py` | Class `TaskPlanner` | Decompose task into TaskDAG | -- | -- |
| `bench/bigcodebench_bench.py` | Module | BigCodeBench Hard Instruct adapter | -- | -- |
| `bench/routing_ground_truth.py` | Module | 50-task routing GT evaluation | -- | -- |
| `bench/eval_protocol.py` | Module | Real-condition benchmark with error logging | -- | -- |
| `tools/agent_tool.py` | Class `AgentTool` | `AgentTool.from_agent()`: wraps any agent as a Tool | `tools.base` | -- |
| `tools/registry.py` | Class `ToolRegistry` | Tool registration and lookup | -- | -- |
| `sandbox/manager.py` | Class `SandboxManager` | Sandbox lifecycle management | -- | -- |
| `events/bus.py` | Class `EventBus` | Pub/sub event bus for pipeline observability | -- | -- |
| `monitoring/drift.py` | Class `DriftMonitor` | Sliding-window output drift detection | -- | -- |
| `resilience.py` | Class `CircuitBreaker` | Per-provider failure circuit breaker | -- | -- |

### sage-python/verl (GiGPO Training)

| Component | Type | Responsibility | Internal Deps | Exposes |
|-----------|------|---------------|---------------|---------|
| `topology_env.py` | Class `SageTopologyEnv` | Gym-style 4-state env: AWAITING_YAML -> EXECUTING -> AWAITING_DECISION -> TERMINAL, verl-agent compatible | `step_reward` | `reset()`, `step()`, `get_step_rewards()` |
| `reward.py` | Function `compute_score` | veRL reward: format scoring (YAML validity [-2,+1]) + structure scoring ([0,1]) + execution scoring, edge credit integration | `edge_credit` | `compute_score(data_source, solution_str, ground_truth, extra_info)` |
| `edge_credit.py` | Class `EdgeStats` + function `compute_edge_advantages` | Graph-GRPO (arXiv 2603.02701): per-edge success rates across K topologies, normalized advantages | -- | `compute_edge_advantages(topologies)` |
| `rewardflow.py` | Class `RewardFlowPropagator` | Per-node credit via state-graph Personalized PageRank backward propagation (RewardFlow arXiv 2603.18859) | -- | `compute(rollouts) -> list[dict[int, float]]` |
| `step_reward.py` | Class `StepRewardVector` | Per-step reward decomposition for GiGPO anchor-based advantage normalization | -- | `to_verl_format()` |
| `training_memory.py` | Class `TrainingMemory` | SQLite episodic memory across training epochs | -- | `store_episode()`, `find_similar()` |
| `env_register.py` | Module | Registers SageTopologyEnv with verl-agent env registry | -- | -- |

### sage-discover

| Component | Type | Responsibility | Internal Deps | Exposes |
|-----------|------|---------------|---------------|---------|
| `discovery.py` | Module | arXiv paper discovery and filtering | -- | -- |
| `curator.py` | Module | Paper relevance scoring and curation | -- | -- |
| `ingestion.py` | Module | PDF -> text chunks for RAG | -- | -- |
| `pipeline.py` | Module | End-to-end discovery pipeline | `discovery`, `curator`, `ingestion` | -- |
| `knowledge.py` | Module | Knowledge graph building from papers | -- | -- |
| `researcher.py` | Module | Automated research assistant | -- | -- |
| `model_watcher.py` | Module | Monitor new model releases | -- | -- |

---

## Data Flow Narratives

### Scenario 1: Simple Task (S1 Routing -> Template Topology -> Single Provider)

```
INPUT: task = "Write a Python function to reverse a string"

1. CLASSIFY (AdaptiveRouter.route)
   -> StructuralFeatures.extract(task) -> complexity=0.15, uncertainty=0.08
   -> KnnRouter.route(task) -> embed via arctic-embed-m -> cosine kNN k=5
      -> distance-weighted vote -> S1 (confidence=0.92)
   -> result: RoutingDecision(system=1, model_id="deepseek-chat")

2. DECOMPOSE (TaskPlanner)
   -> Single-node TaskDAG (no decomposition for S1)
   -> DAGFeatures(omega=1, delta=1, gamma=0.0)

3. SELECT TOPOLOGY (TopologyEngine.generate)
   -> S1 -> template fallback -> templates::sequential("deepseek-chat")
   -> TopologyGraph: 3 nodes (input_processor -> worker -> output_formatter)
   -> HybridVerifier.verify() -> valid=True
   -> source=TemplateFallback, confidence=0.8

4. ASSIGN MODELS (ModelAssigner)
   -> For each node: score = 0.4*affinity(S1) + 0.4*domain("code") + 0.2*cost
   -> All nodes -> "deepseek-chat" (budget model, best cost/quality for S1)
   -> Budget check: 3 nodes * ~$0.001 < $5.00 budget

5. EXECUTE (TopologyRunner)
   -> TopologyExecutor(mode=Static) -> Kahn's toposort -> [0, 1, 2]
   -> Node 0: ProviderPool.resolve("deepseek-chat") -> DeepSeek provider
      -> LLM call -> "Here is a function..."
   -> Node 1: context = node_0 output -> LLM call -> actual implementation
   -> Node 2: context = all outputs -> LLM call -> formatted final answer

6. LEARN (feedback)
   -> QualityLabeler.label(response) -> tree-sitter valid + SmtVerifier proofs -> 0.85
   -> ContextualBandit.record_outcome(arm=("deepseek-chat","sequential"), quality=0.85)
   -> MAP-Elites: BehaviorDescriptor(agents=3, depth=3, cost=$0.003, diversity=0.0)
      -> insert if Pareto-dominant

OUTPUT: formatted code response, cost ~$0.003, latency ~2s
```

### Scenario 2: Complex Task (S3 Routing -> Learned Topology -> Multi-Provider -> Adaptation)

```
INPUT: task = "Prove that merge sort is O(n log n) using induction, then implement and verify"

1. CLASSIFY
   -> StructuralFeatures: complexity=0.78, uncertainty=0.65
   -> KnnRouter: embed -> kNN vote -> S3 (formal keywords: "prove", "induction")
   -> SystemRouter: has_formal_keywords("prove","induction") -> force S3
   -> result: RoutingDecision(system=3)

2. DECOMPOSE
   -> TaskPlanner -> 3-subtask DAG: [proof, implementation, verification]
   -> DAGFeatures(omega=2, delta=2, gamma=0.33)

3. SELECT TOPOLOGY (TopologyEngine.generate, 7-path priority)
   -> Path 1: S-MMU.retrieve(task_embedding) -> similarity=0.45 < 0.7 -> skip
   -> Path 2: BehaviorDescriptor -> archive lookup -> found, quality=0.6 -> use
   -> HybridVerifier -> valid=True
   -> If Path 6 enabled (SAGE_ENABLE_PATH6=1):
      -> Load Qwen3.5-9B policy -> generate YAML topology -> parse -> verify
   -> source=ArchiveHit (or Path6), confidence=0.7

4. ASSIGN MODELS
   -> Node "prover": S3 + domain="formal" -> "gemini-3.1-pro-preview" (affinity 0.9)
   -> Node "coder": S2 + domain="code" -> "gpt-5.3-codex" (affinity 0.95)
   -> Node "verifier": S3 + domain="formal" -> "gemini-3.1-pro-preview"
   -> Budget: $2.00 + $1.50 + $2.00 = $5.50 (exceeds $5.00 -> downgrade verifier)
   -> Verifier downgraded to "deepseek-chat" ($0.28)

5. EXECUTE (TopologyRunner with TopologyController)
   -> TopologyExecutor(mode=Dynamic) -> gate-based readiness
   -> Node "prover": Google Gemini -> formal proof text
   -> TopologyController.evaluate_and_decide(node="prover", quality=0.7)
      -> quality < 0.8 threshold -> action: upgrade_model
      -> Reroute to "gpt-5.3-codex" for retry -> quality=0.9 -> continue
   -> Node "coder": OpenAI Codex -> implementation (parallel with verifier prep)
   -> Node "verifier": DeepSeek -> verification of proof + code

6. LEARN
   -> QualityLabeler: tree-sitter valid, SmtVerifier proves assertions -> 0.92
   -> ContextualBandit: update posteriors for all 3 arms
   -> MAP-Elites: topology with adaptation history archived
   -> S-MMU: store topology + task embedding for future retrieval

OUTPUT: proof + implementation + verification, cost ~$4.28, latency ~15s
```

### Scenario 3: GiGPO Training (veRL -> Topology Env -> Reward -> Advantage)

```
INPUT: Qwen3.5-9B model + 1965 training entries (BigCodeBench + CodeContests)

1. ENV RESET (SageTopologyEnv.reset)
   -> Pick prompt from training data
   -> State machine -> AWAITING_YAML
   -> Return observation: {"prompt": task, "anchor": "topology:S2:hash123"}

2. STEP 0: MODEL GENERATES TOPOLOGY (SageTopologyEnv.step)
   -> Model outputs YAML: {nodes: [{role: planner, ...}, {role: coder, ...}], edges: [...]}
   -> reward.py::_score_format(yaml) -> +1.0 (valid YAML with nodes key)
   -> reward.py::_score_structure(yaml) -> 0.8 (has edges, all roles, reasoning)
   -> State -> EXECUTING
   -> Anchor: "topology:S2:hash123" (for GiGPO step grouping)

3. EXECUTION STEPS: REAL LLM CALLS
   -> For each node in topology:
     -> ProviderPool resolves model_id -> 8 available providers
     -> LLM call with node prompt + predecessor context
     -> StepResult(reward, anchor_key="coder:adequate:hash456")
   -> TopologyController: checkpoint decisions (continue/upgrade/reroute)

4. TERMINAL: SANDBOX EXECUTION
   -> Final code extracted from last node output
   -> ToolExecutor: tree-sitter validate + subprocess execute
   -> Test cases from ground truth -> pass/fail
   -> Execution reward: 1.0 (pass@1) or 0.0 (fail)

5. REWARD COMPUTATION
   -> Base: 0.3*format + 0.3*structure + 0.4*execution = e.g. 0.7
   -> Edge credit (Graph-GRPO):
     -> For K=8 topologies on same prompt, compute per-edge success rates
     -> EdgeStats.from_topologies -> S_ij = P(success | edge(i,j))
     -> Normalize: A_ij = (S_ij - mean) / (std + eps)
     -> edge_credit = mean(advantages for edges in this topology)
     -> final = base + 0.2 * edge_credit
   -> RewardFlow (optional):
     -> Build state graph from K rollouts (state = role:quality_bucket)
     -> Personalized PageRank backward from terminal rewards
     -> Per-node credit assignment

6. GiGPO ADVANTAGE ESTIMATION
   -> StepRewardVector: [r_0=0.8, r_1=0.6, r_2=0.9] with anchor_keys
   -> GiGPO groups steps by anchor key across batch
   -> Per-anchor normalization: A_t = (r_t - mean(anchor_group)) / std(anchor_group)
   -> Step-level policy gradient (not episode-level like GRPO)
   -> TrainingMemory stores episode for cross-epoch learning

LOOP: Repeat for all 1965 entries per epoch, ~3-5 epochs on RunPod H100
TARGET: Qwen3.5-9B learns to generate task-adaptive topologies
```

---

## Key Modules Reference

### sage-core/src/topology/engine.rs — DynamicTopologyEngine

- **Role**: Central topology generation orchestrator. Given a task description and cognitive system level, produces a verified TopologyGraph.
- **Mechanism**: 7-path priority cascade: (1) S-MMU similarity hit (cosine > 0.7 AND quality > 0.5), (2) MAP-Elites archive lookup via BehaviorDescriptor, (3) LLM synthesis (Python callback), (4) Mutation of best archive entry, (5) MCTS structural search, (6) Path 6 learned policy (Qwen3.5-9B, opt-in SAGE_ENABLE_PATH6=1), (7) Template fallback (S1->sequential, S2->AVR, S3->debate).
- **Interface**: `generate(task, system, context) -> GenerateResult { topology, source, confidence }`, `evolve(outcome) -> ()`, `record_outcome(topology_id, quality, cost) -> ()`
- **Calls**: `MultiViewMMU`, `MapElitesArchive`, `MctsSearcher`, `CmaEmitter`, `apply_random_mutation`, `HybridVerifier`, `ContextualBandit`, `TopologySynthesizer`
- **Called by**: `CognitiveOrchestrationPipeline` (Stage 2), `boot.py` (Phase 6 init)
- **Paper**: AgentConductor (arXiv 2602.17100) for RL topology evolution paradigm
- **Debts**: Path 6 is 70% YAML valid. LLM synthesis path delegates to Python (no pure Rust LLM call).

### sage-core/src/topology/topology_graph.rs — TopologyGraph

- **Role**: Unified intermediate representation for all multi-agent topologies.
- **Mechanism**: petgraph DiGraph with typed nodes (TopologyNode: role, model_id, system 1/2/3, capabilities, max_cost, max_wall_time) and three-flow edges (TopologyEdge: type=Control|Message|State, gate=Open|Closed, condition). 8 TopologyTemplate variants + Custom(Ulid). ULID-based topology_id.
- **Interface**: `try_new(template) -> TopologyGraph`, `add_node(TopologyNode) -> usize`, `try_add_edge(from, to, TopologyEdge) -> Result`, `get_node(idx) -> TopologyNode`, `node_count() -> usize`, `is_acyclic() -> bool`, `toposort() -> Vec<usize>`
- **Calls**: `petgraph`, `ulid`
- **Called by**: Every topology-related module (engine, executor, templates, mutations, verifier, density, reward, map_elites, mcts, model_assigner)
- **Paper**: MASFactory (arXiv 2603.06007) three-flow edge model
- **Debts**: `get_edges()` not exposed to Python (PyO3 inventory issue on Windows). Python uses execution-order tracking as workaround.

### sage-core/src/topology/executor.rs — TopologyExecutor

- **Role**: Schedule node execution in a TopologyGraph.
- **Mechanism**: Two modes. Static: Kahn's topological sort O(V+E), deterministic for acyclic DAGs (Sequential, Parallel, Hierarchical, Brainstorming). Dynamic: gate-based readiness polling with loop support for cyclic topologies (AVR, Hub, Debate, SelfMoA). NodeStatus FSM: Pending -> Ready -> Running -> Completed/Skipped. Safety limit: 1000 max iterations.
- **Interface**: `new(graph, mode) -> TopologyExecutor`, `next_ready(graph) -> Vec<usize>`, `mark_completed(idx)`, `is_done() -> bool`
- **Called by**: `TopologyRunner` (Python), `SageTopologyEnv` (veRL)
- **Paper**: Dual-mode concept from MASFactory (2603.06007)

### sage-core/src/routing/system_router.rs — SystemRouter

- **Role**: Classify tasks into S1/S2/S3 cognitive systems and select the best model.
- **Mechanism**: (1) StructuralFeatures.extract(task) -> complexity/uncertainty scores. (2) Formal keyword detection ("prove", "theorem", etc.) -> force S3. (3) Feature-based classification with configurable thresholds. (4) Model selection from ModelRegistry with domain scoring. (5) Budget constraint downgrade. (6) Telemetry history (VecDeque) for drift detection. (7) `route_integrated()` consults ContextualBandit for exploration.
- **Interface**: `route(task, constraints) -> RoutingDecision { system, model_id, confidence }`, `route_integrated(task, constraints) -> RoutingDecision`, `record_outcome(decision_id, quality, cost)`
- **Calls**: `StructuralFeatures`, `ModelRegistry`, `ContextualBandit`
- **Called by**: `AdaptiveRouter` (Python), `ShadowRouter`, pipeline Stage 0
- **Accuracy**: 86% on 50 GT tasks (below kNN's 92%, used as secondary/fallback)

### sage-core/src/routing/knn.rs — RustKnnRouter

- **Role**: Fast kNN routing on pre-computed exemplar embeddings.
- **Mechanism**: L2-normalize query, dot product against all exemplars (cosine similarity), partial top-k sort, distance-weighted majority vote for S1/S2/S3 classification, OOD rejection when max similarity < distance_threshold (default 0.3).
- **Interface**: `load_exemplars(embeddings: Vec<f32>, labels: Vec<i32>, dim)`, `predict(query_embedding) -> (label, confidence)`, `predict_with_distances(query) -> Vec<(label, distance)>`
- **Calls**: none (self-contained)
- **Called by**: `KnnRouter` (Python wrapper in `strategy/knn_router.py`)
- **Paper**: arXiv 2505.12601 — kNN on embeddings outperforms MLP/GNN/attention routers
- **Data**: `config/routing_exemplars.npz` (50 tasks x 768-dim arctic-embed-m, auto-built from ground truth)
- **Accuracy**: 92% GT, LOO-CV 80%

### sage-core/src/routing/bandit.rs — ContextualBandit

- **Role**: Online learning of (model, topology) arm quality via Thompson sampling.
- **Mechanism**: Per-arm Beta posteriors for quality (bounded [0,1], conjugate to Bernoulli), Gamma posteriors for cost/latency (non-negative). Thompson sampling draws at decision time. Global Pareto front construction: filter dominated arms, select based on exploration_budget constraint. Arms = (model_id, template_name).
- **Interface**: `register_arm(model_id, template)`, `select(constraints) -> BanditDecision { arm, quality_sample, cost_sample }`, `record_outcome(decision_id, quality, cost, latency)`
- **Called by**: `DynamicTopologyEngine`, `SystemRouter.route_integrated()`, pipeline Stage 5

### sage-core/src/routing/model_assigner.rs — ModelAssigner

- **Role**: Assign optimal LLM model to each TopologyGraph node.
- **Mechanism**: For each node: filter by required_capabilities (tools, json), filter by budget, score = 0.4*system_affinity + 0.4*domain_match + 0.2*(1 - cost/max_cost). Budget-aware: remaining budget tracked per-node, stops when exhausted.
- **Interface**: `assign_models(graph, task_domain, budget_usd) -> usize (assigned count)`
- **Calls**: `ModelRegistry.all_models()`, `TopologyGraph.get_node/set_model_id`
- **Called by**: Pipeline Stage 3
- **Source of truth**: `sage-core/config/cards.toml` (20 models, 8 providers)

### sage-core/src/verification/smt.rs — SmtVerifier

- **Role**: Formal verification via OxiZ (pure Rust SMT solver, zero C++ deps).
- **Mechanism**: QF_LIA (quantifier-free linear integer arithmetic). Expression AST with parser for comparison/arithmetic/boolean operations. Methods: `verify_memory_safety(bounds)`, `verify_loop_bound(n, max)`, `verify_arithmetic(expr, expected)`, `verify_invariant(pre, post)`, `verify_invariant_with_feedback(pre, post, max_rounds)` (CEGAR loop, max 5 rounds), `verify_provider_assignment(model_ids, constraints)`.
- **Interface**: 10 PyO3 methods, all return `SmtVerificationResult { valid, message, time_ms }`
- **Called by**: `QualityLabeler`, `ProcessRewardModel`, `z3_verify.py`, pipeline Stage 5
- **Performance**: Sub-0.1ms (0.024ms PRM, 0.060ms mutation validation)

### sage-core/src/verification/quality_labeler.rs — QualityLabeler

- **Role**: Formal quality scoring for LLM-generated code, zero heuristics.
- **Mechanism**: (1) Extract ```python code blocks from markdown. (2) tree-sitter syntax validation via `validate_python_code()`. (3) Structural completeness: def/return presence check. (4) Extract arithmetic assertions -> SmtVerifier proof. (5) Combined score from formal signals only: syntax valid + structural complete + assertions proven.
- **Interface**: `label(response_text, task_text) -> QualityLabel { score, signals }`
- **Calls**: `SmtVerifier`, `sandbox::validator`
- **Called by**: Pipeline Stage 5, veRL reward computation
- **Feature gate**: `#[cfg(all(feature = "smt", feature = "tool-executor"))]`

### sage-core/src/verification/ltl.rs — LtlVerifier

- **Role**: Linear Temporal Logic model checking on TopologyGraph.
- **Mechanism**: 4 checks using petgraph BFS/DFS (all O(V+E), no SMT): (1) Reachability: can node A reach node B? (2) Safety: no high-to-low security label information flow. (3) Liveness: every entry node reaches at least one exit. (4) Bounded liveness: all entry-exit paths within depth limit.
- **Interface**: `check_reachability(graph, from, to) -> LtlResult`, `check_safety(graph) -> LtlResult`, `check_liveness(graph) -> LtlResult`, `check_bounded_liveness(graph, max_depth) -> LtlResult`
- **Called by**: `HybridVerifier`, `TopologyReward` (temporal signal)

### sage-core/src/memory/smmu.rs — MultiViewMMU (S-MMU)

- **Role**: 4-orthogonal-view semantic memory management unit.
- **Mechanism**: Single petgraph DiGraph with edges labeled by EdgeKind (Temporal, Semantic, Causal, Entity). Temporal: chronological links weighted by time proximity. Semantic: cosine similarity on embeddings (bounded to MAX_SEMANTIC_NEIGHBORS=128 recent chunks). Causal: parent-child agent links. Entity: Jaccard similarity on keyword sets. Retrieval: BFS up to max_hops with per-view weight factors `[temporal, semantic, causal, entity]`. ULID chunk IDs.
- **Interface**: `register_chunk(metadata) -> chunk_id`, `retrieve_relevant(active_chunk_id, max_hops, weights) -> Vec<(chunk_id, score)>`
- **Called by**: `WorkingMemory`, `DynamicTopologyEngine` (Path 1)
- **Paper**: CoALA cognitive architecture

### sage-core/src/memory/embedder.rs — RustEmbedder

- **Role**: ONNX Runtime inference for arctic-embed-m (109M params, 768-dim).
- **Mechanism**: `load-dynamic` strategy loads onnxruntime DLL at runtime (avoids Windows MSVC static linking issues). Auto-discovers DLL via model sibling, sys.prefix, VIRTUAL_ENV, user site-packages. L2-normalized output embeddings.
- **Interface**: `new(model_path, tokenizer_path) -> RustEmbedder`, `embed(texts) -> Vec<Vec<f32>>`
- **Called by**: `Embedder` (Python), `KnnRouter`, `S-MMU` semantic edge creation
- **Feature gate**: `#[cfg(feature = "onnx")]`

### sage-python/src/sage/pipeline.py — CognitiveOrchestrationPipeline

- **Role**: 5-stage orchestration pipeline replacing inline routing+topology+execution.
- **Mechanism**: Stage 0: router.route(task) -> system. Stage 1: TaskPlanner.decompose(task) -> TaskDAG. Stage 2: engine.generate(task, system) -> TopologyGraph. Stage 3: assigner.assign_models(graph, domain, budget). Stage 4: TopologyRunner.run(graph) -> result. Stage 5: QualityLabeler/Estimator -> score, bandit.record_outcome(), MAP-Elites insert.
- **Interface**: `async run(task, budget_usd=5.0) -> str`
- **Calls**: `AdaptiveRouter`, `DynamicTopologyEngine`, `ModelAssigner`, `TopologyRunner`, `ProviderPool`, `ContextualBandit`, `QualityLabeler`, `EventBus`

### sage-python/src/sage/topology/runner.py — TopologyRunner

- **Role**: Execute a TopologyGraph as a real multi-agent system with LLM calls.
- **Mechanism**: Uses TopologyExecutor for scheduling. Per-node: aggregate completed predecessor outputs -> build system/user prompt from node role/capabilities -> resolve provider via ProviderPool -> LLM call -> store output. TopologyController (if provided) evaluates quality after each node and triggers adaptation actions (upgrade_model, spawn_subagent, reroute, prune).
- **Interface**: `async run(task, context="") -> str`
- **Calls**: `TopologyExecutor`, `ProviderPool`, `LLMProvider`, `TopologyController`
- **Paper**: MASFactory (2603.06007) node lifecycle

### sage-python/src/sage/verl/topology_env.py — SageTopologyEnv

- **Role**: Gym-style multi-step environment for GiGPO topology policy training.
- **Mechanism**: 4-state machine: AWAITING_YAML (model generates topology) -> EXECUTING (nodes run with real LLM calls) -> AWAITING_DECISION (model decides continue/upgrade/reroute at checkpoints) -> TERMINAL (sandbox execution + test). Anchor keys for GiGPO step-level grouping: `"{role}:{difficulty}:{context_hash}"`. Real LLM calls via 8 providers. verl-agent compatible interface.
- **Interface**: `reset(prompt, task_id) -> observation`, `step(model_response) -> (obs, reward, done, info)`, `get_step_rewards() -> StepRewardVector`
- **Calls**: `reward.py`, `step_reward.py`, `ProviderPool`, `ToolExecutor`
- **Paper**: GiGPO (arXiv 2505.10978) for step-level advantage, verl-agent env_manager

### sage-python/src/sage/verl/reward.py — compute_score

- **Role**: veRL-compatible reward function for topology training.
- **Mechanism**: Three components: (1) Format: YAML validity [-2.0, +1.0], strips markdown fences. (2) Structure: node count, edges, role completeness, reasoning field [0.0, 1.0]. (3) Execution: pass@1 from sandbox {0.0, 1.0}. Combined: 0.3*format + 0.3*structure + 0.4*execution. Optional: edge credit from Graph-GRPO added with weight 0.2.
- **Interface**: `compute_score(data_source, solution_str, ground_truth, extra_info) -> float`
- **Paper**: Graph-GRPO (arXiv 2603.02701)

### sage-python/src/sage/verl/edge_credit.py — compute_edge_advantages

- **Role**: Per-edge credit assignment for Graph-GRPO training.
- **Mechanism**: For K topologies on same prompt: (1) Compute per-edge success rate S_ij = P(success | edge(i,j) in G). (2) Normalize: A_ij = (S_ij - mean(S)) / (std(S) + eps). Provides finer-grained credit than per-topology reward.
- **Interface**: `compute_edge_advantages(topologies: list[dict]) -> dict[tuple[int,int], float]`
- **Paper**: Graph-GRPO (arXiv 2603.02701) Eq. 4-5

### sage-python/src/sage/verl/rewardflow.py — RewardFlowPropagator

- **Role**: Per-node credit assignment via state-graph PageRank.
- **Mechanism**: (1) Build state graph from K rollouts: state = (role, quality_bucket). Edges = transition counts between consecutive states. (2) Terminal states receive execution reward. (3) Personalized PageRank backward propagation (damping=0.85, max_iters=20). (4) Map state-level credits back to per-node rewards per rollout.
- **Interface**: `compute(rollouts) -> list[dict[int, float]]`
- **Paper**: RewardFlow (arXiv 2603.18859, AAMAS 2026)

### sage-python/src/sage/strategy/knn_router.py — KnnRouter

- **Role**: Python wrapper for kNN S1/S2/S3 routing with Rust hot-path.
- **Mechanism**: Loads `config/routing_exemplars.npz` (pre-computed arctic-embed-m embeddings for 50 GT tasks). At routing time: embed input -> delegate to RustKnnRouter for cosine kNN (k=5, distance_threshold=0.3). Falls back to None if exemplars missing, embedder non-semantic, or all neighbors below threshold.
- **Calls**: `sage_core.RustKnnRouter`, `memory.embedder.Embedder`
- **Paper**: arXiv 2505.12601

### sage-python/src/sage/strategy/adaptive_router.py — AdaptiveRouter

- **Role**: 4-stage learned routing cascade, primary router for the pipeline.
- **Mechanism**: Stage 0: StructuralFeatures (keyword complexity). Stage 0.5: KnnRouter (embedding kNN, 92% GT). Stage 1: ONNX BERT classifier (Rust only). Stage 2: Entropy probe (logprobs/token diversity). Stage 3: Reserved. Each stage has confidence threshold; if exceeded, returns early.
- **Interface**: `route(task) -> RoutingDecision`, duck-type compatible with ComplexityRouter

### sage-python/src/sage/topology/kg_rlvr.py — ProcessRewardModel

- **Role**: Process reward model for internal reasoning paths using SMT verification.
- **Mechanism**: Parses <think> tags from LLM output, extracts arithmetic/logical assertions, verifies via OxiZ SmtVerifier (Rust) or z3 (Python fallback). Returns per-step process rewards rather than outcome-only rewards.
- **Calls**: `sage_core.SmtVerifier`
- **Paper**: FoVer (Z3 auto-labels), MASPRM (arXiv 2510.24803)

### sage-python/src/sage/memory/remote_rag.py — ExoCortex

- **Role**: Persistent RAG via Google GenAI File Search API.
- **Mechanism**: Uses File Search stores for automatic chunking/embedding of research papers. Free storage. Default store: `fileSearchStores/ygnsageresearch-wii7kwkqozrd`. Default query model: `gemini-3.1-flash-lite-preview`. Conforms to KnowledgeStore protocol for backend swapability.
- **Interface**: `query(question) -> list[str]`, `upload(file_path) -> str`

### sage-python/src/sage/evolution/engine.py — EvolutionEngine

- **Role**: Evolutionary optimization with quality-diversity.
- **Mechanism**: MAP-Elites population (N-dim grid), SAMPO 5 strategic actions (optimize perf, fix correctness, explore novel, tighten constraints, simplify), LLM-as-mutation-operator (AlphaEvolve-style). Evaluation cascade with async provider calls.
- **Interface**: `async run(seed_population, n_generations) -> Population`
- **Calls**: `Population`, `Mutator`/`LLMMutator`, `Evaluator`, `SAMPOSolver`
- **Debts**: No quantitative evidence of improvement yet (need N>=10 Wilcoxon, Cohen's d)

### sage-python/src/sage/llm/provider_pool.py — ProviderPool

- **Role**: Resolve model_id to live LLM provider at execution time.
- **Mechanism**: LRU cache of (model_id -> (LLMProvider, LLMConfig)). Looks up model in providers.registry, matches provider name to pre-built provider instances. Per-provider CircuitBreaker (3 failures -> open). Falls back to default_provider when model unknown or provider circuit open.
- **Interface**: `resolve(model_id) -> (LLMProvider, LLMConfig)`, `record_failure(provider_name, error)`
- **Providers**: Google (Gemini), OpenAI, DeepSeek, xAI (Grok), Kimi, MiniMax, OpenRouter, Codex

---

## Research References

| Tag | Reference | arXiv | Used In | Implementation Status |
|-----|-----------|-------|---------|----------------------|
| GiGPO | Group-in-Group Policy Optimization | 2505.10978 | `verl/topology_env.py`, `verl/step_reward.py` | Step-level advantages with anchor keys, 4-state env. GiGPO = GRPO for single-action; multi-step is the differentiator. |
| Graph-GRPO | Graph-level GRPO with edge credit | 2603.02701 | `verl/edge_credit.py`, `verl/reward.py` | Per-edge success rates, normalized advantages (Eq. 4-5). Integrated into reward function. |
| RewardFlow | State-graph PageRank credit | 2603.18859 | `verl/rewardflow.py` | Personalized PageRank backward propagation, per-node credit. AAMAS 2026. |
| CARD | Causal Reward Decomposition | 2603.01089 | Referenced in design | Not implemented. Related to RewardFlow approach. |
| AgentConductor | RL topology evolution | 2602.17100 | `topology/engine.rs`, `topology/density.rs` | S_complex density function (S_node + S_edge + S_depth), N_max bounds. 97.5% HumanEval with 3B model. |
| The Conductor | GRPO + 6 providers | 2512.04388 | Competitor analysis | PRIMARY competitor. Sakana AI ICLR 2026. Qwen2.5-7B GRPO. BigCodeBench 40.0%. |
| AdaptOrch | Topology > model capability | 2602.16873 | `pipeline_stages.py` | DAGFeatures (omega, delta, gamma). Key finding: Var_tau/Var_M >= 20 on hard tasks. |
| OpenSage | AI picks model per sub-agent | 2602.16891 | Architecture reference | ICML 2026. 59% SWE-Bench Pro. Inspired runtime model selection. |
| kNN Routing | kNN on embeddings | 2505.12601 | `strategy/knn_router.py`, `routing/knn.rs` | PRIMARY router (92% GT). Pre-computed arctic-embed-m exemplars. Outperforms MLP/GNN/attention. |
| TopoCurate | Topology-aware data curation | 2603.01714 | Training data design | Informed SFT data curation approach. |
| MASPRM | Multi-agent system PRM | 2510.24803 | `topology/kg_rlvr.py` | Process reward model with Z3 backing. |
| MAPPA | Multi-agent Planning with Prediction-Aware | 2601.23228 | Architecture reference | Referenced in planning design. |
| AgentDropout | Runtime agent pruning | 2503.18891 | Architecture reference | ACL 2025. -21.6% tokens. Influenced TopologyController prune action. |
| OFA-MAS | MoE graph generative | 2601.12996 | Architecture reference | WWW 2026. Per-node LLM_i formalization. |
| ARG-Designer | Autoregressive graph generation | 2507.18224 | Architecture reference | AAAI 2026 Oral. Influenced Path 6 design. |
| SYMPHONY | UCB scheduling heterogeneous LLM pool | 2601.22623 | Architecture reference | NeurIPS 2025. Influenced ContextualBandit design. |
| Cascade Routing | Quality estimators > routing algorithms | 2410.10347 | `routing/quality.rs` | ETH-SRI ICML 2025. Key insight: quality estimation is the bottleneck. |
| Budget-Aware Routing | Budget-aware LLM routing | 2602.21227 | `routing/system_router.rs` | Informed budget constraint in SystemRouter. |
| Router-R1 | LLM-as-router reasoning | 2506.09033 | Architecture reference | NeurIPS 2025. Multi-round routing reasoning. |
| MASFactory | Vibe Graphing LLM -> graph | 2603.06007 | `topology/runner.py`, `topology/topology_graph.rs` | Three-flow edge model (Control/Message/State). Node lifecycle pattern. |
| CoALA | Cognitive Architecture for Language Agents | -- | `memory/smmu.rs`, `memory/episodic.py` | 4-tier memory (working + episodic + semantic + procedural). Inspired S-MMU design. |
| FoVer | Z3 auto-labels for PRM | -- | `verification/quality_labeler.rs`, `topology/kg_rlvr.py` | Z3/OxiZ formal labeling for training data. SAGE has Z3 but incomplete FoVer pipeline. |
| LLMRouterBench | Embedding backbone impact limited | 2601.07206 | Routing design decisions | Many routing methods converge; embedding choice has limited impact. |
| PILOT | Contextual bandit LLM routing with budget | 2508.21141 | `routing/bandit.rs` | Inspired ContextualBandit with budget constraints. |
| AlphaEvolve | LLM as mutation operator | -- | `evolution/llm_mutator.py` | DeepMind 2025. LLM-guided mutations in evolution engine. |
| Survey 6 Paradigms | 6 routing paradigms survey | 2603.04445 | Architecture validation | Validated SAGE as SOTA architecture combining multiple paradigms. |

---

## LLM Quick-Reference Cheatsheet

1. **Entry point**: `sage.boot.boot()` -> `AgentSystem` -> `pipeline.run(task)`. All 5 stages execute in order. Rust `sage_core` is imported via PyO3; if unavailable, Python fallbacks activate.
2. **Critical invariant**: TopologyGraph (petgraph DiGraph) is the universal IR. Every topology passes through HybridVerifier before use. Every quality score comes from QualityLabeler (OxiZ formal) or is explicitly None.
3. **Config source of truth**: `sage-core/config/cards.toml` defines all 20 models and 8 providers. `sage-python/config/cards.toml` is a symlink. Routing exemplars: `config/routing_exemplars.npz`.
4. **Anti-patterns to avoid**: (a) Never hardcode thresholds -- use Z3/ONNX/paper citation. (b) Never use ComplexityRouter (34% GT, dead code). (c) Never add `verify=False`. (d) Never use HumanEval+ for framework value proofs (saturated). (e) Never use hash embeddings for routing (OK for S-MMU dedup only).
5. **Active branch**: `VeRLGiGPO` -- GiGPO training for Qwen3.5-9B topology policy on RunPod H100. Training data: 1965 entries (BigCodeBench + CodeContests). Target: replace Phi-4-mini SFT (70% YAML valid) with RL-trained policy.
