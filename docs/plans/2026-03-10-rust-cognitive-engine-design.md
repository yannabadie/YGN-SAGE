# YGN-SAGE: Full Rust Cognitive Engine — Design Document

**Date:** 2026-03-10
**Status:** Approved
**Author:** Yann Abadie + Claude Opus 4.6

## 1. Objective

Migrate the entire cognitive decision pipeline (routing S1/S2/S3, topology generation, model selection, online learning) from Python into `sage-core` (Rust). All rewritten Python modules are deleted — no dual maintenance.

### Non-Negotiable Requirements
- **Full Rust-First**: all decisional logic in sage-core via PyO3
- **Multi-objective Pareto reward**: quality × cost × latency, no fixed weights
- **Arbitrary DAG topologies**: generated/mutated via MAP-Elites evolution
- **Triple-layer persistence**: Arrow (hot) + SQLite (cross-session) + S-MMU (semantic retrieval)
- **Z3 dual verification**: constraints in MAP-Elites archive + runtime re-verification
- **Incremental deploy**: 3 phases, each independently testable and mergeable

## 2. Research Context

### State-of-the-Art (March 2026)
| System | Technique | Limitation vs YGN-SAGE |
|--------|-----------|----------------------|
| DyTopo (arXiv 2602.06039) | Semantic matching for per-round topology wiring | No Rust, no Z3, no cost-awareness, no memory learning |
| GTD (arXiv 2510.07799) | Diffusion-guided topology generation + proxy reward | No formal verification, no bandit, no ModelCards |
| Google A2A Protocol (v0.3) | Agent Cards for capability discovery + handoff | Agent-level (not model-level), no evolutionary topology |
| LLMRank / LLMRouterBench | Neural ranking for prompt-aware LLM routing | No safety guardrails, no memory-aware routing |
| MAP-Elites / GAME | Quality-diversity optimization for agent architectures | Not applied to runtime topology + model co-selection |
| Bandit (UCB/Thompson) | Online model selection with regret bounds | No topology awareness, no formal verification |

**YGN-SAGE's novel contribution**: First system combining ModelCard-driven routing + evolutionary topology generation + multi-objective Pareto bandit + Z3 dual verification + Rust-native data-plane with triple-layer persistence.

## 3. Architecture

```
                         ┌─────────────────────────────────┐
                         │        Python (boot.py)          │
                         │  AgentLoop ← PyO3 binding only   │
                         └──────────────┬──────────────────┘
                                        │ route(query, context)
                         ┌──────────────▼──────────────────┐
                         │     sage_core::SystemRouter      │
                         │  ┌───────────────────────────┐   │
                         │  │ 1. Structural analysis     │   │
                         │  │ 2. ModelCard semantic match │   │
                         │  │    → decides S1 / S2 / S3  │   │
                         │  │ 3. Bandit Pareto selection  │   │
                         │  │ 4. Z3 runtime verification  │   │
                         │  └───────────────────────────┘   │
                         │        ↓ RoutingDecision          │
                         │   (system, model, topology)       │
                         └──────────────┬──────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
          ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
          │TopologyEngine│    │ BanditState   │    │ModelRegistry  │
          │ MAP-Elites   │    │ Pareto fronts │    │ 24+ cards     │
          │ Z3 constrain │    │ Arrow+SQLite  │    │ cards.toml    │
          │ Mutation ops │    │ +S-MMU edges  │    │ CRUD+score    │
          └─────────────┘    └──────────────┘    └──────────────┘
```

## 4. Phase 1: ModelCard + SystemRouter

### 4.1 ModelCard (`sage-core/src/routing/model_card.rs`)

```rust
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelCard {
    #[pyo3(get)] pub id: String,
    #[pyo3(get)] pub provider: String,
    #[pyo3(get)] pub family: String,
    // Capability scores [0.0, 1.0]
    #[pyo3(get)] pub code_score: f32,
    #[pyo3(get)] pub reasoning_score: f32,
    #[pyo3(get)] pub tool_use_score: f32,
    #[pyo3(get)] pub math_score: f32,
    #[pyo3(get)] pub formal_z3_strength: f32,
    // Cost & latency
    #[pyo3(get)] pub cost_input_per_m: f32,
    #[pyo3(get)] pub cost_output_per_m: f32,
    #[pyo3(get)] pub latency_ttft_ms: f32,
    #[pyo3(get)] pub tokens_per_sec: f32,
    // System affinity [0.0, 1.0]
    #[pyo3(get)] pub s1_affinity: f32,
    #[pyo3(get)] pub s2_affinity: f32,
    #[pyo3(get)] pub s3_affinity: f32,
    // Topology preferences
    #[pyo3(get)] pub recommended_topologies: Vec<String>,
    // Capability flags
    #[pyo3(get)] pub supports_tools: bool,
    #[pyo3(get)] pub supports_json_mode: bool,
    #[pyo3(get)] pub supports_vision: bool,
    #[pyo3(get)] pub context_window: u32,
}
```

Inspired by Google A2A Agent Cards but specialized for LLM model selection (not agent discovery). Each card declares what a model is good at and how it fits S1/S2/S3 cognitive systems.

### 4.2 ModelRegistry (`sage-core/src/routing/model_registry.rs`)

- `from_toml(path) -> ModelRegistry` — loads cards from `sage-core/config/cards.toml`
- `register(card)` / `unregister(id)` — dynamic CRUD at runtime
- `select_for_system(system, constraints) -> Vec<ModelCard>` — filtered by system affinity + capabilities
- `get(id) -> Option<ModelCard>`
- Pre-computes embedding per card at load time via `RustEmbedder`

### 4.3 SystemRouter (`sage-core/src/routing/system_router.rs`)

The SystemRouter does NOT implement a sequential pipeline. It decides which **cognitive system** (S1/S2/S3) to activate — like the brain choosing between intuition, deliberation, or formal reasoning. Each System is a complete mode of thought, not a stage in a chain.

**Cognitive Systems (Kahneman-inspired):**
- **System 1 (Fast/Intuitive)** — Direct response, cheap model, no verification loop. For obvious tasks.
- **System 2 (Deliberate/Tools)** — Code execution, AVR loop (Act→Verify→Repair), sandbox. For tool-use and coding.
- **System 3 (Formal/Reasoning)** — Deep reasoning, Z3 bounds checking, formal proofs. For mathematical/logical tasks.

**Decision process** (all Rust, <2ms total):

1. **Structural analysis** (existing StructuralFeatures, <0.1ms): keyword/complexity/uncertainty extraction. If unambiguous (confidence > 0.85), directly selects the System.

2. **ModelCard semantic matching** (new, <1ms):
   - `RustEmbedder.embed(query)` → 384-dim vector
   - Cosine similarity vs pre-computed card embeddings
   - System score: `Σ(card.sN_affinity × similarity × capability_match)` for each System
   - Selects the System with highest aggregate score

3. **Bandit Pareto selection** (new, <0.5ms):
   - Given the chosen System, Thompson sampling on Pareto front per (model × topology) combo
   - Returns the non-dominated combo matching caller's runtime preference

4. **Z3 runtime verification** (<1ms):
   - Budget check: remaining budget ≥ estimated cost
   - Capability check: provider supports required features
   - Topology check: DAG is valid (no cycles, fan-in/fan-out within limits)
   - If fails → next Pareto-dominant combo (max 3 fallback attempts)

**Output:**
```rust
#[pyclass]
pub struct RoutingDecision {
    pub system: CognitiveSystem,    // S1 | S2 | S3
    pub model_id: String,
    pub topology: TopologyGraph,
    pub confidence: f32,
    pub pareto_rank: u32,
    pub estimated_cost: f32,
    pub estimated_latency_ms: f32,
}
```

### 4.4 Python files deleted (Phase 1)
- `sage-python/src/sage/strategy/metacognition.py` (ComplexityRouter)
- `sage-python/src/sage/strategy/adaptive_router.py` (AdaptiveRouter)
- `sage-python/src/sage/strategy/training.py` (BERT retraining export)

### 4.5 Config: `sage-core/config/cards.toml`

Migrated from `sage-python/config/model_profiles.toml` with added fields:

```toml
[[models]]
id = "gemini-2.5-flash"
provider = "google"
family = "gemini-2.5"
code_score = 0.85
reasoning_score = 0.80
tool_use_score = 0.90
math_score = 0.75
formal_z3_strength = 0.60
cost_input_per_m = 0.075
cost_output_per_m = 0.30
latency_ttft_ms = 200.0
tokens_per_sec = 200.0
s1_affinity = 0.70
s2_affinity = 0.85
s3_affinity = 0.40
recommended_topologies = ["sequential", "avr", "self-moa"]
supports_tools = true
supports_json_mode = true
supports_vision = true
context_window = 1048576
```

## 5. Phase 2: DynamicTopologyEngine + MAP-Elites

### 5.1 TopologyGraph (`sage-core/src/topology/topology_graph.rs`)

```rust
#[pyclass]
pub struct TopologyGraph {
    graph: petgraph::DiGraph<TopologyNode, TopologyEdge>,
    id: Ulid,
    fitness: ParetoPoint<3>,  // [quality, 1/cost, 1/latency]
}

#[pyclass]
#[derive(Clone)]
pub struct TopologyNode {
    pub role: String,              // "coder", "reviewer", "reasoner"
    pub model_id: String,          // "gemini-2.5-flash"
    pub system: CognitiveSystem,   // S1 | S2 | S3
    pub prompt_template: String,
}

#[derive(Clone)]
pub struct TopologyEdge {
    pub weight: f32,
    pub transform: EdgeTransform,  // PassThrough | Summarize | Filter
}
```

Built on petgraph DiGraph (already a dependency). Serializable to JSON for Python interop.

### 5.2 MAP-Elites Archive (`sage-core/src/topology/map_elites.rs`)

- N-dimensional grid with behavior descriptors: `(agent_count, max_depth, cost_bucket, model_diversity)`
- Each cell holds the Pareto-dominant `TopologyGraph` for that behavior region
- **Z3 constraint gate**: topology must pass Z3 validation before archive insertion
- Fitness: multi-objective Pareto (quality, 1/cost, 1/latency)

### 5.3 Mutation Operators (`sage-core/src/topology/mutations.rs`)

| Operator | Description |
|----------|-------------|
| `add_node` | Insert new agent node with random/selected model |
| `remove_node` | Remove lowest-fitness node, rewire edges |
| `swap_model` | Change a node's model_id (from ModelRegistry) |
| `rewire_edge` | Add/remove/redirect an edge |
| `split_node` | Split one node into two specialized nodes |
| `merge_nodes` | Merge two nodes into one generalist |
| `mutate_prompt` | LLM-guided prompt mutation (via PyO3 callback) |

### 5.4 DynamicTopologyEngine (`sage-core/src/topology/engine.rs`)

```rust
#[pyclass]
pub struct DynamicTopologyEngine {
    archive: MapElitesArchive,
    registry: Arc<ModelRegistry>,
    memory: Arc<WorkingMemory>,
    z3_validator: Z3TopologyValidator,
}

#[pymethods]
impl DynamicTopologyEngine {
    /// Select best topology for query from archive, mutate if needed
    fn generate(&self, query: &str, budget: f32) -> TopologyGraph;

    /// Run offline evolution cycle (async, background)
    fn evolve(&mut self, population_size: usize, generations: usize);

    /// Record outcome for bandit + archive update
    fn record_outcome(&mut self, topology_id: &str, quality: f32, cost: f32, latency: f32);
}
```

### 5.5 Python files deleted (Phase 2)
- `sage-python/src/sage/topology/evo_topology.py`
- `sage-python/src/sage/topology/engine.py`
- `sage-python/src/sage/topology/patterns.py`
- `sage-python/src/sage/topology/topology_archive.py`
- `sage-python/src/sage/topology/topology_verifier.py`
- `sage-python/src/sage/topology/planner.py`

## 6. Phase 3: Bandit Pareto + Persistence + Z3 Dual

### 6.1 Pareto Front (`sage-core/src/routing/bandit.rs`)

```rust
/// A point in N-objective space
pub struct ParetoPoint<const N: usize> {
    pub objectives: [f32; N],  // [quality, 1/cost, 1/latency]
    pub combo: ComboKey,
    pub timestamp: chrono::DateTime<Utc>,
}

/// Multi-objective Pareto front
pub struct ParetoFront<const N: usize> {
    points: Vec<ParetoPoint<N>>,
}

impl<const N: usize> ParetoFront<N> {
    /// Insert point, prune dominated
    fn insert(&mut self, point: ParetoPoint<N>);
    /// Thompson sampling: sample from posterior per objective, return non-dominated
    fn sample_thompson(&self, rng: &mut impl Rng) -> &ParetoPoint<N>;
}
```

### 6.2 BanditState (`sage-core/src/routing/bandit.rs`)

```rust
#[pyclass]
pub struct BanditState {
    fronts: HashMap<ComboKey, ParetoFront<3>>,
    // ComboKey = (CognitiveSystem, ModelId, TopologyId)
}

#[pymethods]
impl BanditState {
    fn select(&self, system: CognitiveSystem) -> RoutingDecision;
    fn record(&mut self, combo: ComboKey, quality: f32, cost: f32, latency: f32);
    fn save_to_sqlite(&self, path: &str) -> PyResult<()>;
    fn load_from_sqlite(path: &str) -> PyResult<Self>;
}
```

### 6.3 Triple-Layer Persistence

| Layer | Data | Storage | Access Pattern |
|-------|------|---------|----------------|
| Arrow (Tier 0) | BanditState hot, active MAP-Elites archive | `WorkingMemory` Rust | Every request, <0.1ms |
| SQLite (Tier 1) | Full archive + Pareto history | `~/.sage/topology.db` | Boot load + periodic flush (every N requests) |
| S-MMU (Tier 2) | Topology chunks with task embeddings | Multi-view graph | Semantic retrieval: "similar task → best topology" |

**Boot sequence:** SQLite → Arrow (restore hot state)
**Runtime:** Arrow for all reads. S-MMU `register_chunk` on each `record_outcome`.
**Periodic flush:** Arrow → SQLite every 50 requests or on graceful shutdown.

### 6.4 Z3 Dual Verification

**In MAP-Elites (evolution-time):**
- Budget feasibility: estimated cost ≤ max budget
- Capability coverage: all required capabilities satisfied by assigned models
- Topology validity: DAG (no cycles), fan-in ≤ max, fan-out ≤ max
- Security labels: info-flow lattice respected (HIGH → LOW blocked)

**In SystemRouter (runtime):**
- Re-verify with CURRENT context: remaining budget, live provider status, task-specific constraints
- If verification fails → next Pareto-dominant combo
- Max 3 fallback attempts before returning default S1 topology

### 6.5 Python files deleted (Phase 3)
- `sage-python/src/sage/strategy/solvers.py` (SAMPOSolver)
- `sage-python/src/sage/evolution/engine.py`
- `sage-python/src/sage/evolution/llm_mutator.py`
- `sage-python/src/sage/evolution/population.py`
- `sage-python/src/sage/evolution/evaluator.py`
- `sage-python/src/sage/evolution/mutator.py`
- `sage-python/src/sage/evolution/self_improve.py`
- `sage-python/src/sage/evolution/ebpf_evaluator.py`

## 7. Python Integration (boot.py + agent_loop.py)

### boot.py changes
```python
# Phase 1
from sage_core import SystemRouter, ModelRegistry
registry = ModelRegistry.from_toml("config/cards.toml")
router = SystemRouter(registry=registry, embedder=embedder, memory=working_memory)

# Phase 2
from sage_core import DynamicTopologyEngine
topo_engine = DynamicTopologyEngine(registry=registry, memory=working_memory)
router.set_topology_engine(topo_engine)

# Phase 3
router.enable_bandit(persistence_path=str(Path.home() / ".sage" / "topology.db"))
```

### agent_loop.py changes
```python
# ROUTING: single Rust call replaces all Python routing
decision = system.router.route(task, context=system.working_memory)
# decision.system → S1 | S2 | S3
# decision.model_id → "gemini-2.5-flash"
# decision.topology → TopologyGraph (full DAG)

# LEARN: feedback to bandit + MAP-Elites
system.router.record_outcome(decision.id, quality, cost, latency)
```

## 8. Cargo.toml Changes

```toml
[features]
default = []
sandbox = ["wasmtime", "dep:wasmtime-wasi"]
cranelift = ["wasmtime/cranelift"]
onnx = ["dep:ort", "dep:tokenizers", "dep:ndarray"]
tool-executor = ["dep:tree-sitter", "dep:tree-sitter-python", "dep:process-wrap"]
topology = []  # NEW: always-on (no optional deps beyond petgraph which is already included)

[dependencies]
# Existing deps used by new modules:
# petgraph = "0.6.4"      (TopologyGraph DAG)
# dashmap = "6"            (concurrent BanditState)
# chrono = "0.4"           (timestamps)
# serde/serde_json         (serialization)
# ulid                     (topology IDs)
# arrow/pyo3-arrow         (persistence)

# New dependency:
toml = "0.8"               # cards.toml parsing
rand = "0.9"               # Thompson sampling
rusqlite = { version = "0.33", features = ["bundled"] }  # SQLite persistence
```

## 9. New Rust File Structure

```
sage-core/src/
├── routing/
│   ├── mod.rs              (existing, updated)
│   ├── features.rs         (existing, unchanged)
│   ├── router.rs           (existing Stage 1 ONNX, kept behind onnx feature)
│   ├── model_card.rs       (NEW — Phase 1)
│   ├── model_registry.rs   (NEW — Phase 1)
│   ├── system_router.rs    (NEW — Phase 1)
│   └── bandit.rs           (NEW — Phase 3)
├── topology/
│   ├── mod.rs              (NEW — Phase 2)
│   ├── topology_graph.rs   (NEW — Phase 2)
│   ├── map_elites.rs       (NEW — Phase 2)
│   ├── mutations.rs        (NEW — Phase 2)
│   └── engine.rs           (NEW — Phase 2)
├── config/
│   └── cards.toml          (NEW — Phase 1)
└── tests/
    ├── test_model_card.rs    (NEW — Phase 1)
    ├── test_system_router.rs (NEW — Phase 1)
    ├── test_topology.rs      (NEW — Phase 2)
    ├── test_map_elites.rs    (NEW — Phase 2)
    └── test_bandit.rs        (NEW — Phase 3)
```

## 10. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Routing latency | < 2ms (full decision) | Benchmark with 1000 queries |
| Routing accuracy | ≥ 100% on 30 GT tasks | Existing routing benchmark |
| Topology generation | < 5ms per new topology | Benchmark MAP-Elites generate() |
| Pareto convergence | Front stabilizes within 100 queries | Track front size over time |
| Z3 validation | < 1ms per check | Benchmark Z3 runtime verify |
| Cross-session restore | < 100ms boot from SQLite | Measure boot time delta |
| Python LOC deleted | ~3000 LOC | Count deleted files |
| Rust LOC added | ~2600 LOC | Count new files |
| All existing tests pass | 1036+ Python, 66+ Rust | CI green |

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Z3 Rust bindings don't exist | Use z3-sys crate or keep Z3 checks in Python via PyO3 callback |
| Thompson sampling convergence slow with Pareto | Start with UCB fallback, switch to Thompson after 50 observations |
| LLM mutation callback (Rust→Python→LLM→Python→Rust) latency | Mutation is async/offline, not on hot path |
| Windows ONNX DLL issues | Already solved via OnceLock + auto-discovery pattern |
| rusqlite bundled increases binary size | Acceptable trade-off for zero system dependency |

## 12. A2A Protocol Integration (Future)

YGN-SAGE's ModelCard is analogous to A2A Agent Cards but specialized for LLM model selection. Future integration:
- Export ModelCards as A2A-compatible `/.well-known/agent.json` for external discovery
- Import external A2A Agent Cards into ModelRegistry for cross-system routing
- Use A2A task lifecycle (send/receive/stream) for multi-system topology execution

## 13. References

- DyTopo: arXiv 2602.06039 (Feb 2026)
- GTD: arXiv 2510.07799 (Oct 2025)
- Google A2A Protocol v0.3 (Jul 2025)
- LLMRouterBench: arXiv 2601.07206 (Jan 2026)
- MAP-Elites: Mouret & Clune (2015), GAME: arXiv 2505.06617
- Multi-Armed Bandits + LLM: arXiv 2505.13355
- Contextual Bandits for LLM Selection: arXiv 2506.01767
