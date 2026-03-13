# The Five Cognitive Pillars

YGN-SAGE is built on five pillars that together form a self-adaptive agent system. Each pillar is implemented across both the Rust core and the Python SDK.

---

## Topology

The Topology pillar treats multi-agent structure as a first-class optimization target. Rather than using fixed pipelines, SAGE evolves agent graph topologies that adapt to the task at hand.

### Topology Graph IR

The `TopologyGraph` is a unified intermediate representation wrapping `petgraph::DiGraph` with:

- **Typed nodes**: roles, capabilities, budgets
- **Three-flow edges**: Control (execution order), Message (field-level data routing), State (parent-child shared state)

This three-flow edge model comes from the MASFactory research (arXiv 2603.06007), which demonstrated that distinguishing control, message, and state flows enables 84.76% HumanEval accuracy with 97% code reduction.

### 8 Built-in Templates

The `PyTemplateStore` provides battle-tested starting points:

| Template | Description |
|----------|-------------|
| Sequential | Chain agents in series |
| Parallel | Fan-out/fan-in with aggregation |
| AVR | Act-Verify-Refine self-correction loop |
| SelfMoA | Self-Mixture of Agents |
| Hierarchical | Manager delegates to workers |
| Hub | Central coordinator with spokes |
| Debate | Adversarial argumentation |
| Brainstorming | Divergent idea generation |

### 6-Path Generation Strategy

The `DynamicTopologyEngine` uses a prioritized 6-path strategy to generate topologies:

1. **S-MMU recall** -- retrieve a similar topology from prior successful runs
2. **Archive lookup** -- query the MAP-Elites archive for a quality-diversity match
3. **LLM synthesis** -- 3-stage pipeline (Role Assignment -> Structure Design -> Validation) via `TopologySynthesizer`
4. **Mutation** -- apply one of 7 mutation operators to an existing topology
5. **MCTS** -- Monte Carlo Tree Search (UCB1 selection, 50 simulations or 100ms budget)
6. **Template fallback** -- use one of 8 built-in templates

### Evolutionary Search

**MAP-Elites** (`MapElitesArchive`): Quality-diversity archive with a 4-dimensional BehaviorDescriptor spanning 108 cells. Uses Pareto dominance for replacement. SQLite persistence for cross-session continuity.

**CMA-ME** (`CmaEmitter`): Covariance Matrix Adaptation MAP-Elites for directional search on continuous topology parameters. Diagonal covariance with elite-weighted mean/variance updates. Integrated into `evolve()` with 50% random / 50% CMA-sampled mutations.

**7 Mutation Operators**: `add_node`, `remove_node`, `swap_model`, `rewire_edge`, `split_node`, `merge_nodes`, `mutate_prompt`. All mutations are validated by the HybridVerifier before acceptance.

### Verification

The `HybridVerifier` runs 6 structural + 4 semantic checks plus LTL temporal property integration, all in O(V+E) time:

- **Structural**: connectivity, acyclicity, degree bounds, etc.
- **Semantic**: role coverage, switch consistency, loop termination, reachability
- **LTL**: safety (no HIGH->LOW information flow), liveness (all entries reach exits), bounded liveness (depth <= K)

### Execution

The `TopologyExecutor` supports two scheduling modes:

- **Static mode**: Kahn's topological sort for acyclic DAGs
- **Dynamic mode**: Gate-based readiness scheduling for cyclic topologies with loops and conditional switches

---

## Strategy

The Strategy pillar classifies tasks into cognitive systems and selects appropriate models, drawing on Kahneman's dual-process theory extended to three tiers.

### Cognitive Systems

| System | Character | Example Tasks |
|--------|-----------|---------------|
| **S1** | Fast, intuitive | "What is 2+2?", greetings, simple lookups |
| **S2** | Analytical, deliberate | Code generation, multi-step reasoning, essay writing |
| **S3** | Formal verification | Prove memory safety, verify invariants, SAT solving |

### 5-Stage Adaptive Routing

The `AdaptiveRouter` implements a 5-stage learned routing pipeline:

**Stage 0 -- Structural Features** (Rust, `StructuralFeatures`): Zero-cost keyword and structural feature extraction. Always compiled, no model dependencies.

**Stage 0.5 -- kNN Embeddings** (`KnnRouter`): kNN classification using pre-computed exemplar embeddings from snowflake-arctic-embed-m. Achieves 92% accuracy on 50 human-labeled ground truth tasks versus 52% for keyword heuristics (arXiv 2505.12601). Distance-weighted majority vote. Auto-builds from ground truth at boot if `.npz` is missing. Refuses hash embeddings.

**Stage 1 -- BERT ONNX** (Rust `AdaptiveRouter`): DistilBERT classifier running in ONNX Runtime via the Rust `ort` crate. Dynamic input discovery, 512-token truncation, binary/multi-class support.

**Stage 2 -- Entropy Probe**: Confidence calibration check on the classifier output.

**Stage 3 -- Cascade Fallback**: Falls back to the heuristic `ComplexityRouter` if confidence remains low.

### Contextual Bandit

The `ContextualBandit` uses per-arm Beta/Gamma posteriors with Thompson sampling and Pareto front selection. It learns which models work best for which task types over time. SQLite persistence for cross-session learning.

### Model Selection

The `SystemRouter` combines multiple signals in priority order:

1. **Hard constraints** -- capability requirements, budget limits
2. **Structural scoring** -- keyword-based pre-filtering
3. **Domain hint** -- task domain matches model domain scores from `cards.toml`
4. **Bandit/budget selection** -- Thompson sampling with cost-quality Pareto optimization

### Quality Estimation

The `QualityEstimator` uses 5 signals to score outputs: non-empty, length adequacy, code presence, error absence, and AVR convergence. A DistilBERT ONNX model (0.9 MB, trained on 600 triples) achieves +34.4pp Pearson correlation improvement over the baseline heuristic.

---

## Memory

The Memory pillar implements a 4-tier hierarchy inspired by computer architecture memory hierarchies, from fast volatile storage to persistent knowledge bases.

### Tier 0 -- Working Memory (STM)

Rust Arrow-backed buffer with SIMD/AVX-512 acceleration. The MEM1 compressor writes internal state at every agent step with pressure-triggered compression. Uses ULID-based chunk IDs that are globally unique and cross-session stable.

**S-MMU (Structured Memory Management Unit)**:

- **Write path**: Compressor calls `compact_to_arrow_with_meta()` with keywords, embeddings (via `Embedder`), and dynamic summaries. `register_chunk()` uses bounded recency scan (last 128 chunks).
- **Read path**: `retrieve_smmu_context()` queries the multi-view graph during THINK phase and injects top-k chunk summaries as a SYSTEM message.

**Embedder (3-tier fallback)**:

1. RustEmbedder -- ONNX via `ort` crate, native SIMD, snowflake-arctic-embed-m (768-dim, 109M params)
2. sentence-transformers -- Python fallback
3. SHA-256 hash -- emergency fallback (forbidden for routing, OK for S-MMU dedup)

### Tier 1 -- Episodic Memory

SQLite-backed (`~/.sage/episodic.db`) with cross-session persistence. CRUD operations plus keyword search. WAL mode for concurrent access.

### Tier 2 -- Semantic Memory

In-memory entity-relation graph with SQLite persistence (`~/.sage/semantic.db`). The `MemoryAgent` autonomously extracts entities during the LEARN phase using either heuristic or LLM-based extraction.

Additional graph types:

- **CausalMemory**: Directed causal edges with temporal ordering
- **WriteGate**: Confidence-based write gating with abstention tracking

### Tier 3 -- ExoCortex (Persistent RAG)

Google GenAI File Search API with 500+ research sources. Auto-configured with a default store. Available as the `search_exocortex` agent tool. Passive grounding was removed after Sprint 3 evidence showed it adds latency without benefit for code tasks.

### CRAG Relevance Gate

The `RelevanceGate` implements CRAG-style keyword overlap scoring (threshold=0.3) to block irrelevant memory injection. This prevents context pollution from unrelated prior interactions.

### 9 Agent Tools

- 3 STM tools (read/write/search working memory)
- 4 LTM tools (episodic CRUD + keyword search)
- `search_exocortex` (query the ExoCortex knowledge base)
- `refresh_knowledge` (trigger knowledge pipeline refresh)

---

## Tools

The Tools pillar provides secure, sandboxed code execution with multiple isolation layers.

### ToolExecutor Security Pipeline

When `sage_core` is compiled with `tool-executor` + `sandbox` features, code execution follows this pipeline:

**Layer 1 -- tree-sitter AST Validation**: Static analysis blocks 23 dangerous modules (`os`, `sys`, `subprocess`, etc.) and 11 dangerous calls (`exec`, `eval`, `open`, etc.). Error-tolerant -- handles partial parse trees from broken code.

**Layer 2 -- Wasm WASI Sandbox**: wasmtime v36 LTS with deny-by-default policy:

- NO filesystem access
- NO environment variable access
- NO network access
- NO subprocess spawning
- Only stdout/stderr inherited

**Layer 3 -- Subprocess Fallback**: tokio-based execution with timeout and kill-on-drop isolation.

The execution priority is: Wasm WASI -> bare Wasm -> subprocess. Python's `create_python_tool` tries the Rust ToolExecutor first and falls back to `sandbox_executor.py`.

### 6 Security Vectors Blocked

1. Filesystem read/write
2. Environment variable access
3. Network access
4. Subprocess spawning
5. Dangerous module imports
6. Dangerous function calls (eval, exec, etc.)

### SnapBPF

Rust Copy-on-Write memory snapshots via eBPF (`solana_rbpf`) for mutation rollback. Provides efficient state checkpointing during evolutionary search.

### Agent Tools

The `AgentTool` class wraps any agent (with `async run(task) -> str`) as a Tool:

```python
from sage.tools.agent_tool import AgentTool

tool = AgentTool.from_agent(agent, name="researcher", description="Deep research")
```

---

## Evolution

The Evolution pillar enables agents to self-modify their topology, hyperparameters, and behavior through directed evolutionary search.

### SAMPO Strategic Actions

The DGM (Directed Generative Model) context uses a SAMPO solver to choose 1 of 5 strategic actions injected into the LLM mutation prompt:

1. **Explore** -- seek novel topology regions
2. **Modify mutations** -- adjust `mutations_per_generation`
3. **Modify clipping** -- adjust `clip_epsilon`
4. **Modify filtering** -- adjust `filter_threshold`
5. **Exploit** -- refine the current best topology

Actions 2, 3, and 4 enable self-modification of the evolution engine's own hyperparameters.

### MAP-Elites Quality-Diversity Search

The `MapElitesArchive` maintains a 4-dimensional behavioral descriptor space with 108 cells. New topologies are placed into their behavioral niche, replacing existing entries only via Pareto dominance. This ensures diversity of solutions rather than convergence to a single optimum.

### CMA-ME Directional Search

The `CmaEmitter` provides Covariance Matrix Adaptation for continuous topology parameter optimization. It is integrated into the evolution loop with a 50/50 split between random mutations and CMA-sampled mutations.

### MCTS Exploration

The `MctsSearcher` uses Monte Carlo Tree Search with UCB1 selection and random mutation expansion. The rollout heuristic is based on the HybridVerifier's structural/semantic score. Budget: 50 simulations or 100ms.

### Formal Verification of Mutations

Every mutation is validated by the `HybridVerifier` before acceptance. The `SmtVerifier` can additionally validate mutations via `validate_mutation()`, and the `LtlVerifier` checks temporal properties (safety, liveness) on the resulting topology graph.

### CEGAR Invariant Synthesis

The `synthesize_invariant()` method implements Counterexample-Guided Abstraction Refinement: it iteratively weakens or strengthens post-conditions over up to 5 rounds. `verify_invariant_with_feedback()` returns clause-level diagnostic violations that feed into S3 escalation prompts.
