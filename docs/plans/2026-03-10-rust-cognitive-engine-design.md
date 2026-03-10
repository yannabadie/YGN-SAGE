# YGN-SAGE: Full Rust Cognitive Engine — Design Document

**Date:** 2026-03-10
**Status:** Approved (Revised after expert reviews #1, #2, and MASFactory analysis)
**Author:** Yann Abadie + Claude Opus 4.6
**Expert Review #1:** Applied — latency targets relaxed, OxiZ replaces z3-sys, opaque topology IDs, decay factor, tracing added
**Expert Review #2:** Applied — shadow mode before deletion, contextual bandit replaces flat Thompson, topology unified with Contract IR, template-first approach, hard constraints before semantic, A2A pinned to v1.0, embedding upgraded to arctic-embed-m
**MASFactory Analysis (arXiv 2603.06007):** Applied — LLM-synthesized topology generation as 5th path in DynamicTopologyEngine, three-flow edge model (control/message/state), gate-based dual-mode executor, expanded template catalogue (8 templates), semantic diagnosis layer in verifier

## 1. Objective

Migrate the entire cognitive decision pipeline (routing S1/S2/S3, topology generation, model selection, online learning) from Python into `sage-core` (Rust). Python modules kept as frozen oracles in shadow mode during transition, deleted only after Rust parity proven on traces.

### Non-Negotiable Requirements
- **Full Rust-First**: all decisional logic in sage-core via PyO3
- **Evidence-first**: no module deleted without shadow-mode parity proof on real traces
- **Multi-objective Pareto reward**: quality × cost × latency via contextual bandit
- **Template-first topologies**: typed catalogue → archive retrieval → LLM synthesis → MAP-Elites offline → arbitrary DAGs
- **LLM-synthesized topology generation**: 5th path in DynamicTopologyEngine — role decomposition + structure design + validation (inspired by MASFactory Vibe Graphing, arXiv 2603.06007)
- **Unified IR**: TopologyGraph extends existing Contract IR (TaskDAG), not parallel structure
- **Three-flow edge model**: control flow (scheduling) + message flow (field-level data routing) + state flow (parent↔child sync)
- **Triple-layer persistence**: DashMap/Rust native (hot) + SQLite (durable) + Arrow (snapshot/analytics) + S-MMU (semantic retrieval)
- **Hybrid verifier + semantic diagnosis**: graph algos + bitsets + arithmetic + SMT for symbolic + role/switch/loop/connectivity validation
- **Dual-mode executor**: Kahn's topological sort (static DAGs) + gate-based readiness polling (dynamic topologies with loops/switches)
- **Incremental deploy**: 5 phases, evidence gates between each

## 2. Research Context

### State-of-the-Art (March 2026)
| System | Technique | Limitation vs YGN-SAGE |
|--------|-----------|----------------------|
| DyTopo (arXiv 2602.06039) | Semantic matching for per-round topology wiring | No Rust, no cost-awareness, no memory learning |
| AdaptOrch (arXiv 2602.16873) | Task-adaptive topology selection (seq/par/hier/hybrid) | No formal verification, no bandit, no model-level cards |
| GTD (arXiv 2510.07799) | Diffusion-guided topology generation + proxy reward | No formal verification, no bandit, no ModelCards |
| Google A2A Protocol (v1.0) | Agent Cards for capability discovery + handoff | Agent-level (not model-level), no evolutionary topology |
| LLMRouterBench (arXiv 2601.07206) | 400K-instance unified eval of 10 routing baselines | Shows embedding backbone impact limited, many methods converge |
| PILOT (arXiv 2508.21141) | Contextual bandit LLM routing with budget awareness | No topology, no formal verification |
| Contextual Bandits (arXiv 2506.17670) | LinUCB multi-LLM selection, budget+position-aware | No topology, no memory |
| MAP-Elites / GAME | Quality-diversity optimization for agent architectures | Not applied to runtime topology + model co-selection |
| MASFactory (arXiv 2603.06007) | Vibe Graphing: LLM compiles NL intent → executable graph (3-stage: role→structure→parameterize) | No Rust, no bandit, no formal verification, no learning/evolution, no cost-awareness |
| OFA-MAS (arXiv 2601.12996) | MoE graph generative model for universal MAS topology | One-shot generation, no runtime adaptation, no verification |
| Topology Structure Learning (arXiv 2505.22467) | 3-stage framework: agent selection → structure profiling → topology synthesis. Up to 10% performance gap between topologies | Survey/position paper, no unified system |

**YGN-SAGE's research hypothesis** (to be validated empirically): Combining ModelCard-driven routing + template-to-evolved topology generation + contextual multi-objective bandit + hybrid verification + Rust-native data-plane with triple-layer persistence can outperform fixed-model and fixed-topology baselines on cost-quality Pareto frontier.

**Key insight from LLMRouterBench**: Many routing methods converge under unified evaluation. Embedding backbone has limited impact on routing quality. This means: **hard constraints first, then structured/telemetry scoring, then semantic matching as tie-breaker only**. ModelCard scores must be treated as calibrable priors, not fixed scientific constants.

## 3. Architecture

```
                         ┌─────────────────────────────────┐
                         │        Python (boot.py)          │
                         │  AgentLoop ← PyO3 binding only   │
                         │  Python routers kept as shadow    │
                         │  oracles until Rust parity proven │
                         └──────────────┬──────────────────┘
                                        │ route(query, constraints)
                         ┌──────────────▼──────────────────┐
                         │     sage_core::SystemRouter      │
                         │  ┌───────────────────────────┐   │
                         │  │ 1. Hard constraints filter │   │
                         │  │ 2. Structural analysis     │   │
                         │  │ 3. Telemetry-calibrated    │   │
                         │  │    scoring → S1 / S2 / S3  │   │
                         │  │ 4. Contextual bandit       │   │
                         │  │    selection                │   │
                         │  │ 5. Graph validation         │   │
                         │  │ (semantic match = tiebreak) │   │
                         │  └───────────────────────────┘   │
                         │        ↓ RoutingDecision          │
                         │   (system, model, topology_id)    │
                         └──────────────┬──────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
          ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
          │TemplateStore │    │ContextBandit  │    │ModelRegistry  │
          │ Typed catalog│    │ Per-arm posts  │    │ 24+ cards     │
          │ MAP-Elites   │    │ Global front   │    │ cards.toml    │
          │ (offline)    │    │ DashMap+SQLite │    │ Calibrable    │
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

Inspired by Google A2A Agent Cards (v1.0) but specialized for LLM model selection (not agent discovery). Each card declares what a model is good at and how it fits S1/S2/S3 cognitive systems.

**IMPORTANT:** All scores are **calibrable priors**, not fixed scientific constants. Initial values from provider documentation; refined via telemetry feedback loop. `supports_tools`, `supports_json_mode`, cost, latency must reflect **actual adapter behavior** (documented quirks, not brochure claims).

### 4.2 ModelRegistry (`sage-core/src/routing/model_registry.rs`)

- `from_toml(path) -> ModelRegistry` — loads cards from `sage-core/config/cards.toml`
- `register(card)` / `unregister(id)` — dynamic CRUD at runtime
- `select_for_system(system, constraints) -> Vec<ModelCard>` — filtered by system affinity + capabilities
- `get(id) -> Option<ModelCard>`
- Pre-computes embedding per card at load time via `RustEmbedder` (arctic-embed-m, 768-dim)

### 4.3 SystemRouter (`sage-core/src/routing/system_router.rs`)

The SystemRouter does NOT implement a sequential pipeline. It decides which **cognitive system** (S1/S2/S3) to activate — like the brain choosing between intuition, deliberation, or formal reasoning. Each System is a complete mode of thought, not a stage in a chain.

**Cognitive Systems (Kahneman-inspired):**
- **System 1 (Fast/Intuitive)** — Direct response, cheap model, no verification loop. For obvious tasks.
- **System 2 (Deliberate/Tools)** — Code execution, AVR loop (Act→Verify→Repair), sandbox. For tool-use and coding.
- **System 3 (Formal/Reasoning)** — Deep reasoning, Z3 bounds checking, formal proofs. For mathematical/logical tasks.

**Decision process** (all Rust, **< 20ms P99 total** — invisible vs 200ms+ LLM TTFT):

1. **Hard constraints filter** (< 0.01ms): eliminates models that violate caller constraints:
   - `max_cost`, `max_latency`, `min_quality` (runtime contract — explicit, not implicit)
   - `required_capabilities` (tools, json_mode, vision, min_context_window)
   - `security_labels` (info-flow policy from Contract IR)
   - If no model survives → return error, don't silently degrade

2. **Structural analysis** (existing StructuralFeatures, < 0.1ms): keyword/complexity/uncertainty extraction + formal keyword detection. If unambiguous (confidence > 0.85), directly selects the System.

3. **Telemetry-calibrated scoring** (< 0.1ms):
   - System score from structural features + telemetry-adjusted card affinities
   - `Σ(card.sN_affinity × capability_match × telemetry_weight)` for each System
   - Card scores are **priors calibrated by observed outcomes** (not static TOML values)
   - Selects the System with highest aggregate score

4. **Contextual bandit selection** (< 0.5ms):
   - Given the chosen System, **contextual bandit** (LinUCB-based, per arXiv 2506.17670):
     - Per-arm posteriors (mean + variance) for each objective (quality, cost, latency)
     - **Global** Pareto front built at decision time from current posteriors (not stored per-combo)
     - Context features: task structural fingerprint + system + budget tier
   - Runtime contract selects point from front: lexicographic `min_quality → min_cost → min_latency` or explicit preference vector
   - **Warm-start:** Bayesian priors from `sN_affinity` in cards.toml (avoids cold-start catastrophe)
   - **Decay:** `decay_factor = 0.995` per observation (non-stationary environment adaptation)

5. **Semantic matching** (optional tie-breaker, < 15ms P99 when needed):
   - Only invoked when top-2 candidates score within 5% of each other
   - `RustEmbedder.embed(query)` → 768-dim vector (arctic-embed-m ONNX INT8, ~12-15ms CPU)
   - Cosine similarity vs pre-computed card embeddings (< 0.1ms)
   - **CRITICAL:** If ONNX unavailable, **hard-fail to structural-only scoring** — hash fallback FORBIDDEN for routing decisions (hash is not semantically meaningful)

6. **Graph validation** (< 0.01ms for S1/S2, async for S3):
   - S1/S2: `petgraph::algo::is_cyclic_directed()` + fan-in/out check — O(V+E), pure Rust, < 10μs
   - S3: OxiZ SMT verification (async/background thread) for formal proofs
   - Budget + capability: bitset intersection + arithmetic (fast, no SMT needed)
   - If fails → next Pareto-dominant combo (max 3 fallback attempts)

**Output:**
```rust
#[pyclass]
pub struct RoutingDecision {
    pub decision_id: Ulid,          // Unique ID for this decision (used in record_outcome)
    pub system: CognitiveSystem,    // S1 | S2 | S3
    pub model_id: String,           // Primary model — for multi-model topologies, this is the "lead" model
    pub topology_id: Ulid,          // Opaque ID — lazy-load via engine.get_topology(id)
    pub confidence: f32,
    pub pareto_rank: u32,
    pub estimated_cost: f32,
    pub estimated_latency_ms: f32,
}
// IMPORTANT: Never return full TopologyGraph across PyO3 FFI boundary.
// PyO3 serialization of nested petgraph DiGraph = massive overhead.
// Use opaque Ulid + lazy-loading via Arrow shared memory or Rust-side get().
//
// Note: model_id is the "lead" model for the chosen system. Multi-model topologies
// assign per-node models internally — access via get_topology(topology_id).
```

### 4.4 Python files — shadow mode (Phase 1)

**NOT deleted.** Python routers kept as frozen reference oracles:
- `sage-python/src/sage/strategy/metacognition.py` (ComplexityRouter) — shadow oracle
- `sage-python/src/sage/strategy/adaptive_router.py` (AdaptiveRouter) — shadow oracle
- `sage-python/src/sage/strategy/training.py` (BERT retraining export) — shadow oracle

Boot.py runs both Rust and Python routers, logs divergences. Deletion only after Rust parity proven on 1000+ real traces with < 5% divergence rate.

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

## 5. Phase 2: Hybrid Verifier + Runtime Contracts + Template Topologies

### 5.1 Unified IR: TopologyGraph extends Contract IR

**Key design change (from Critique2 review):** The existing Python `contracts/` module already provides a production-ready TaskDAG IR with: TaskNode (typed I/O, capabilities, security labels, budgets, failure policies), TaskDAG (Kahn's topo sort, cycle detection), DAGExecutor, PolicyVerifier, RepairLoop, CostTracker. Creating a parallel `TopologyGraph` would be **IR duplication**.

Instead, `TopologyGraph` in Rust **extends** the Contract IR concepts:

```rust
#[pyclass]
pub struct TopologyGraph {
    graph: petgraph::DiGraph<TopologyNode, TopologyEdge>,
    id: Ulid,
    fitness: ParetoPoint<3>,  // [quality, 1/cost, 1/latency]
    template_type: TopologyTemplate,  // typed template origin
}

/// Typed topology templates — start here, evolve later
/// Expanded from 5 to 8 templates after MASFactory analysis (arXiv 2603.06007)
#[derive(Clone, Copy)]
pub enum TopologyTemplate {
    Sequential,    // A → B → C (existing SequentialAgent)
    Parallel,      // A + B + C → Aggregator (existing ParallelAgent)
    AVR,           // Act → Verify → Repair loop (existing S2 pattern)
    SelfMoA,       // Multiple agents + mixture-of-agents aggregation
    Hierarchical,  // Parent delegates, children report back
    Hub,           // Central coordinator + spoke delegation via handoff (MASFactory HubGraph)
    Debate,        // Two agents argue opposing approaches, third judges (adversarial reasoning)
    Brainstorming, // N agents diverge in parallel, then converge via aggregator
    Custom(Ulid),  // MAP-Elites evolved or LLM-synthesized (Phase 4 only)
}

#[pyclass]
#[derive(Clone)]
pub struct TopologyNode {
    pub node_id: String,
    pub role: String,              // "coder", "reviewer", "reasoner"
    pub model_id: String,          // "gemini-2.5-flash"
    pub system: CognitiveSystem,   // S1 | S2 | S3
    // Contract IR fields (from existing TaskNode)
    pub required_capabilities: Vec<String>,
    pub security_label: u8,        // SecurityLabel enum
    pub max_cost_usd: f32,
    pub max_wall_time_s: f32,
}

/// Three-flow edge model (inspired by MASFactory, arXiv 2603.06007)
/// Separates control (scheduling), message (data routing), and state (parent↔child sync)
#[derive(Clone)]
pub struct TopologyEdge {
    pub edge_type: EdgeType,        // Control | Message | State
    pub field_mapping: Option<HashMap<String, String>>,  // field-level routing: "code" → "review_input"
    pub gate: Gate,                 // Open | Closed — for dynamic execution
    pub condition: Option<String>,  // for switch edges (evaluated at runtime)
    pub weight: f32,                // fitness/priority scoring
}

#[derive(Clone, Copy, PartialEq)]
pub enum EdgeType {
    Control,   // Scheduling constraint: A must finish before B starts
    Message,   // Data routing: specific fields from A's output → B's input
    State,     // Hierarchical sync: parent graph ↔ child subgraph state
}

#[derive(Clone, Copy, PartialEq)]
pub enum Gate {
    Open,      // Edge participates in current execution
    Closed,    // Edge excluded (unused branch, switch not taken)
}
```

### 5.2 Template Catalogue (Phase 2a — before MAP-Elites)

**Progression** (per AdaptOrch arXiv 2602.16873: "orchestration topology dominates over model capability"):
1. **(a) Template catalogue** — 8 typed templates (Sequential, Parallel, AVR, SelfMoA, Hierarchical, Hub, Debate, Brainstorming)
2. **(b) S-MMU retrieval** — "similar task → best topology" semantic lookup (see §7)
3. **(c) LLM synthesis** (Phase 4) — role decomposition → structure design → validation (see §7.5)
4. **(d) MAP-Elites offline** (Phase 4) — evolve + mutate templates into custom topologies (see §7)
5. **(e) Task-adaptive custom DAGs** (Phase 4+) — generated on-the-fly from archive + mutation + LLM (see §7.4)

**Important:** Steps (c) and (d) are the ambitious core of YGN-SAGE's topology innovation. They are deferred to Phase 4 for **sequencing reasons** (templates must exist before you can evolve them), NOT because they are cut. The full MAP-Elites + mutation + S-MMU retrieval pipeline is described in §7.

### 5.3 Hybrid Verifier (`sage-core/src/topology/verifier.rs`)

Replaces the "all-Z3" approach with a layered verification strategy:

| Check | Method | Latency |
|-------|--------|---------|
| DAG validity (acyclic) | `petgraph::algo::is_cyclic_directed()` | < 10μs |
| Capability coverage | Bitset intersection | < 1μs |
| Budget feasibility | Arithmetic comparison | < 1μs |
| Fan-in/fan-out limits | Graph degree check | < 1μs |
| Security labels (info-flow) | Lattice comparison | < 1μs |
| Symbolic constraints (rare) | OxiZ SMT (async) | < 50ms |

**Semantic Diagnosis Layer** (inspired by MASFactory diagnose_node, arXiv 2603.06007):

On top of the structural/formal checks, the verifier runs semantic validation:

| Check | Method | Purpose |
|-------|--------|---------|
| Role coherence | String match vs task keywords | Each node's role makes sense for the task |
| Switch condition completeness | Non-empty condition on all switch edges | Prevents undefined routing |
| Loop termination guarantee | Controller node exists + max_iterations set | Prevents infinite loops |
| Entry/exit reachability | BFS from entry, reverse-BFS from exit | All nodes participate in execution |
| Field mapping consistency | Input keys ⊆ predecessor output keys | No dangling references in message flow |
| Capability aggregation | Union of node capabilities ⊇ task requirements | Topology can fulfill the task |

This layer catches semantic errors that structural checks miss (e.g., a "coder" node with no code capability, a switch with missing conditions, a loop without termination).

**Total for S1/S2: < 15μs.** SMT only for genuinely symbolic/policy constraints.

### 5.4 Runtime Contract

```rust
/// Explicit runtime constraints — passed to every route() call
#[pyclass]
pub struct RoutingConstraints {
    pub max_cost_usd: f32,         // Hard budget cap
    pub max_latency_ms: f32,       // Hard latency cap
    pub min_quality: f32,          // Minimum acceptable quality (0.0-1.0)
    pub required_capabilities: Vec<String>,  // ["tools", "json_mode", "vision"]
    pub security_label: u8,        // Maximum allowed security level
    pub exploration_budget: f32,   // 0.0 = pure exploit, 1.0 = full explore
}
```

This addresses the Critique2 point: "Pareto gives an excellent dashboard but not a policy." The runtime contract provides the **explicit selection policy** for choosing a point from the Pareto front.

### 5.5 Python files — shadow mode (Phase 2)

**NOT deleted.** Kept as frozen oracles:
- `sage-python/src/sage/topology/evo_topology.py` — shadow
- `sage-python/src/sage/topology/engine.py` — shadow
- `sage-python/src/sage/topology/patterns.py` — shadow

**Unified:** `sage-python/src/sage/contracts/` (TaskDAG, TaskNode, PolicyVerifier) — KEPT and extended, not replaced.

## 6. Phase 3: Contextual Bandit + Persistence

### 6.1 Contextual Bandit (`sage-core/src/routing/bandit.rs`)

**Redesigned per Critique2 + arXiv 2506.17670 (PILOT framework):**

The previous design stored `HashMap<ComboKey, ParetoFront<3>>` — a per-combo Pareto front. This is conceptually wrong: a Pareto front compares **arms against each other**, not historical points within one arm. The correct design:

```rust
/// Per-arm posterior estimates for each objective
pub struct ArmPosterior {
    pub arm: ArmKey,                    // (ModelId, TopologyTemplate)
    pub quality: BetaPosterior,          // Beta(alpha, beta) for quality
    pub cost: GammaPosterior,            // Gamma(shape, rate) for cost
    pub latency: GammaPosterior,         // Gamma(shape, rate) for latency
    pub observation_count: u32,
    pub last_updated: chrono::DateTime<Utc>,
}

/// Contextual bandit state — builds global front at decision time
#[pyclass]
pub struct ContextualBandit {
    arms: HashMap<ArmKey, ArmPosterior>,
    // ArmKey = (ModelId, TopologyTemplate) — NOT per-system
    // System is part of the context, not the arm
    context_weights: Vec<f32>,      // LinUCB feature weights
    decay_factor: f32,              // 0.995 — temporal discounting
    exploration_bonus: f32,         // UCB exploration term
}

#[pymethods]
impl ContextualBandit {
    /// Select best arm given context + constraints
    /// Builds global Pareto front from current posteriors, then selects
    /// per runtime contract (lexicographic or preference vector)
    fn select(
        &self,
        system: CognitiveSystem,
        constraints: &RoutingConstraints,
        context_features: &[f32],   // structural features + system + budget tier
    ) -> RoutingDecision;

    /// Record outcome — updates posterior for the chosen arm
    fn record(
        &mut self,
        decision_id: Ulid,
        quality: f32,
        cost: f32,
        latency: f32,
    );

    fn save_to_sqlite(&self, path: &str) -> PyResult<()>;
    fn load_from_sqlite(path: &str) -> PyResult<Self>;
}
```

**Note on `Σ(affinity × similarity × capability_match)`:** This scoring in SystemRouter step 3 is explicitly acknowledged as a **heuristic initial scoring function** that will be calibrated by the bandit's feedback loop. It is not "weight-free" — it's a warm-start prior.

### 6.2 Triple-Layer Persistence (clarified roles)

| Layer | Data | Storage | Role | Access Pattern |
|-------|------|---------|------|----------------|
| DashMap/Rust native | ContextualBandit hot state, template catalogue | In-process | **Primary mutable state** | Every request, <0.1ms |
| SQLite (Tier 1) | Full bandit posteriors + archive history | `~/.sage/topology.db` | **Durable truth** | Boot load + periodic flush |
| Arrow | Snapshot/export of bandit state for analytics | `WorkingMemory` | **Analytics/export** (not primary mutable) | On-demand snapshot |
| S-MMU (Tier 2) | Template chunks with task embeddings | Multi-view graph | **Semantic retrieval** | "similar task → best template" |

**Boot sequence:** SQLite → DashMap (restore hot state)
**Runtime:** DashMap for all reads/writes. S-MMU `register_chunk` on each `record()`.
**Periodic flush:** DashMap → SQLite every 50 requests or on graceful shutdown.
**SQLite config:** `PRAGMA journal_mode=WAL;` — treat `-wal` file as persistent state, not disposable artifact.
**Async writes:** MPSC channel (tokio) for non-blocking SQLite flushes — avoids lag spikes on the Nth request.
**Long tasks:** `py.allow_threads()` for multi-ms Rust-only work (evolve, flush, compaction).

### 6.3 SMT Verification (OxiZ — pure Rust, no C++ deps)

**Solver:** OxiZ v0.1.3+ (crates.io/crates/oxiz) — pure Rust CDCL(T) SMT solver.
Feature flag: `smt = ["dep:oxiz"]`. System compiles and works WITHOUT SMT — hybrid verifier handles all common cases.

**Note on z3/z3-sys:** These crates DO exist (docs.rs/z3, docs.rs/z3-sys) and are maintained. However OxiZ is preferred for YGN-SAGE because: (1) pure Rust = no C++ toolchain/cross-compile issues, (2) z3-sys requires locating Z3 shared library at build time, (3) OxiZ is WASM-ready for future sandbox use.

**When is SMT actually needed:**
- Only for genuinely symbolic constraints (info-flow lattice proofs, complex policy interactions)
- All common checks (budget, capability, DAG validity, fan-in/out) use the hybrid verifier (bitsets + arithmetic + petgraph) — ~2000x faster than SMT as documented in CLAUDE.md

### 6.4 Python files — shadow mode (Phase 3)

**NOT deleted until Phase 5.** Kept as frozen oracles:
- `sage-python/src/sage/strategy/solvers.py` — shadow
- `sage-python/src/sage/evolution/` (all files) — shadow

## 7. Phase 4: S-MMU Topology Retrieval + MAP-Elites Evolution + Custom DAGs

This is the ambitious core of YGN-SAGE's topology system. Phase 2 provides the typed template foundation; Phase 4 makes topologies **adaptive, evolutionary, and task-specific**.

### 7.1 S-MMU Topology Retrieval (`sage-core/src/topology/smmu_bridge.rs`)

The S-MMU (Semantic Memory Management Unit) already exists in sage-core as a multi-view graph with embedding-based semantic edges. Phase 4 extends it to store and retrieve **topology outcomes**:

**Write path** (on every `record_outcome`):
```rust
// After task completes, store topology performance as S-MMU chunk
smmu.register_chunk(TopologyChunk {
    task_embedding: embedder.embed(task),    // 768-dim arctic-embed-m
    topology_id: decision.topology_id,
    template_type: topology.template_type,
    quality: measured_quality,
    cost: measured_cost,
    latency: measured_latency,
    task_fingerprint: structural_features,   // keyword complexity, uncertainty, etc.
});
```

**Read path** (on every `route()`):
```rust
// Before bandit selection, query S-MMU for similar past tasks
let similar = smmu.query_similar(task_embedding, top_k=5);
// Returns: [(topology_id, quality, similarity_score), ...]
// Inject as prior into contextual bandit: boost arms that worked for similar tasks
bandit.inject_similarity_prior(similar);
```

This creates a **feedback loop**: good topologies for similar tasks get boosted, enabling the system to learn task→topology mappings over time without explicit rules.

### 7.2 MAP-Elites Archive (`sage-core/src/topology/map_elites.rs`)

N-dimensional quality-diversity archive. Each cell holds the **best topology** for a specific behavioral region.

```rust
#[pyclass]
pub struct MapElitesArchive {
    grid: HashMap<BehaviorDescriptor, TopologyGraph>,
    // BehaviorDescriptor = (agent_count_bucket, max_depth_bucket, cost_bucket, model_diversity_bucket)
    verifier: HybridVerifier,
}

#[pymethods]
impl MapElitesArchive {
    /// Insert a topology — must pass hybrid verifier before acceptance
    fn insert(&mut self, topology: TopologyGraph) -> bool;

    /// Retrieve best topology for a given behavior region
    fn get(&self, descriptor: &BehaviorDescriptor) -> Option<TopologyGraph>;

    /// Return all non-empty cells (the full archive)
    fn archive_size(&self) -> usize;

    /// Export archive to SQLite for persistence
    fn save(&self, path: &str) -> PyResult<()>;
    fn load(path: &str) -> PyResult<Self>;
}
```

**Z3/OxiZ gate**: Before a topology enters the archive, it must pass the hybrid verifier:
- DAG validity (petgraph) — mandatory
- Capability coverage (bitset) — mandatory
- Budget feasibility (arithmetic) — mandatory
- Security labels (OxiZ, if `smt` feature enabled) — optional, for formal proofs

### 7.3 Mutation Operators (`sage-core/src/topology/mutations.rs`)

7 operators that transform topologies. These create **novel topologies** that didn't exist in the template catalogue — the mechanism by which YGN-SAGE discovers task-optimal architectures.

| Operator | Description | Example |
|----------|-------------|---------|
| `add_node` | Insert new agent node with selected model | Add a "reviewer" node to a Sequential topology |
| `remove_node` | Remove lowest-fitness node, rewire edges | Prune an unnecessary "summarizer" from a heavy topology |
| `swap_model` | Change a node's model_id (from ModelRegistry) | Replace expensive GPT-5.4 with cheaper Gemini Flash |
| `rewire_edge` | Add/remove/redirect an edge between nodes | Convert Sequential to Parallel by removing dependency |
| `split_node` | Split one node into two specialized nodes | Split "coder" into "coder" + "tester" |
| `merge_nodes` | Merge two nodes into one generalist | Merge "coder" + "reviewer" into "coder-reviewer" |
| `mutate_prompt` | LLM-guided prompt mutation (via PyO3 callback) | Refine a node's system prompt for better performance |

**Mutation validation**: Every mutated topology MUST pass the hybrid verifier before it can be used or archived. Invalid mutations are discarded (no retry — generate a new one).

### 7.4 DynamicTopologyEngine (`sage-core/src/topology/engine.rs`)

The engine that ties everything together: archive retrieval, S-MMU similarity, mutation, and evolution.

```rust
#[pyclass]
pub struct DynamicTopologyEngine {
    archive: MapElitesArchive,
    registry: Arc<ModelRegistry>,
    smmu: Arc<WorkingMemory>,        // S-MMU for semantic topology retrieval
    verifier: HybridVerifier,
    embedder: Option<Arc<RustEmbedder>>,  // arctic-embed-m for task embedding
}

#[pymethods]
impl DynamicTopologyEngine {
    /// Select best topology for a task.
    ///
    /// Strategy — 5 paths (in order):
    /// 1. Query S-MMU for similar past tasks → retrieve their winning topology
    /// 2. If S-MMU hit with quality > threshold → return directly (cache hit)
    /// 3. Look up archive for best topology matching task's behavior descriptor
    /// 4. If archive hit → optionally mutate (exploration_budget > 0) → validate → return
    /// 5. ★ LLM synthesis → role decomposition → structure design → validate → return ★
    ///    (inspired by MASFactory Vibe Graphing, arXiv 2603.06007)
    ///    Only invoked when S-MMU miss + archive miss + exploration_budget > 0.3
    ///    Result cached in archive for future reuse.
    /// 6. Fallback: return default template for the chosen CognitiveSystem
    fn generate(
        &self,
        task: &str,
        system: CognitiveSystem,
        constraints: &RoutingConstraints,
    ) -> TopologyGraph;

    /// Run offline evolution cycle (async, background thread).
    /// Spawns N mutations from existing archive entries, evaluates via surrogate
    /// model or S-MMU replay, inserts survivors into archive.
    ///
    /// This is the "explore" mechanism — discovers novel topologies.
    /// Called periodically (every 100 requests) or on explicit trigger.
    fn evolve(&mut self, population_size: usize, generations: usize);

    /// Record outcome — feeds S-MMU + archive + bandit
    fn record_outcome(
        &mut self,
        topology_id: Ulid,
        task_embedding: Vec<f32>,
        quality: f32,
        cost: f32,
        latency: f32,
    );

    /// Get a topology by ID (for lazy-loading from RoutingDecision.topology_id)
    fn get_topology(&self, id: Ulid) -> Option<TopologyGraph>;
}
```

**Offline evolution flow** (`evolve()`):
1. Sample N topologies from archive (diverse cells)
2. Apply 1-3 random mutations to each
3. Validate via hybrid verifier (reject invalid)
4. Estimate fitness via **surrogate model** (lightweight quality predictor trained on S-MMU history) OR **S-MMU replay** (replay similar past tasks' outcomes)
5. Insert survivors into archive if they dominate existing cell occupant
6. Log via `tracing`: mutation type, validation result, fitness delta

**Task-adaptive custom DAGs**: When `generate()` mutates an archive topology (step 4), the result is a **custom DAG** tailored to the specific task. Over time, as the archive fills with evolved variants, the system discovers topologies that no human designed — e.g., a 3-node "coder→tester→fixer" loop that outperforms the 2-node AVR template for debugging tasks.

### 7.5 LLM-Synthesized Topology Generation (`sage-core/src/topology/llm_synthesis.rs`)

**Inspired by MASFactory Vibe Graphing (arXiv 2603.06007).** MASFactory proves that LLM-generated topologies are competitive: 84.76% HumanEval with per-task topology synthesis, at $0.26-$0.59 per workflow (10x cheaper than Vibe Coding).

**Key insight:** Mutations are *local* transformations (add/remove/rewire). LLM synthesis can make *non-local structural changes* — e.g., transforming a pipeline into a hub-and-spoke in one operation. Both are complementary: LLM synthesis explores broadly, mutations explore locally.

**3-stage synthesis pipeline** (adapted from MASFactory, but autonomous — no HITL):

```rust
/// LLM-synthesized topology — called via PyO3 callback to Python LLM provider
pub struct TopologySynthesizer {
    verifier: HybridVerifier,
    registry: Arc<ModelRegistry>,
}

impl TopologySynthesizer {
    /// Generate a topology from task description via LLM.
    ///
    /// Stage 1: Role Assignment — LLM decomposes task into agent roles
    ///   Input: task description + available models (from ModelRegistry)
    ///   Output: Vec<(role_name, required_capabilities)>
    ///
    /// Stage 2: Structure Design — LLM generates topology from roles
    ///   Input: roles + template examples + constraints
    ///   Output: adjacency matrix + node specs (JSON)
    ///
    /// Stage 3: Validation + Instantiation
    ///   Parse JSON → TopologyGraph
    ///   Validate via HybridVerifier + semantic diagnosis
    ///   If invalid → discard (no retry, fallback to template)
    ///
    /// The LLM call happens via PyO3 callback (py.allow_threads() released).
    /// Cost amortized: result cached in MAP-Elites archive + S-MMU for reuse.
    pub fn synthesize(
        &self,
        task: &str,
        system: CognitiveSystem,
        constraints: &RoutingConstraints,
        llm_callback: &PyObject,  // Python callable: fn(prompt) -> str
    ) -> Option<TopologyGraph>;
}
```

**When invoked:** Only when S-MMU miss + archive miss + `exploration_budget > 0.3`. This ensures:
- Most requests use cached/retrieved topologies (fast, < 1ms)
- LLM synthesis only fires for genuinely novel task types (~5-10% of requests)
- Result enters archive immediately → future similar tasks hit cache

**Adjacency matrix output format** (inspired by MASFactory AdjacencyMatrixGraph):
LLMs generate matrices more reliably than edge lists. The synthesis prompt asks for:
```json
{
  "nodes": [{"role": "coder", "model": "gemini-2.5-flash", "capabilities": ["code", "tools"]}],
  "adjacency": [[null, {"fields": {"code": "review_input"}}], [null, null]],
  "template_hint": "sequential"
}
```

**Cost guard:** LLM synthesis invocation capped at 1 per minute to prevent runaway API spend. Budget deducted from task's `RoutingConstraints.max_cost_usd`.

### 7.6 Dual-Mode Topology Executor (`sage-core/src/topology/executor.rs`)

**Problem:** Static topologies (Sequential, Parallel) work well with Kahn's topological sort. But dynamic topologies (AVR loops, Hub handoffs, Debate, conditional switches) need gate-based readiness polling — nodes execute when their input gates are satisfied, loops reopen gates, switches close unused branches.

**Solution:** Dual-mode executor leveraging petgraph's `Topo` iterator for static mode and custom gate polling for dynamic mode.

```rust
pub struct TopologyExecutor {
    mode: ExecutionMode,
}

#[derive(Clone, Copy)]
pub enum ExecutionMode {
    /// Kahn's topological sort via petgraph::visit::Topo — deterministic, O(V+E)
    /// Used for: Sequential, Parallel, Hierarchical, Brainstorming
    Static,
    /// Gate-based readiness polling — nodes fire when all input gates are Open
    /// Used for: AVR (loop), Hub (handoff), Debate (adversarial), Custom (any)
    Dynamic { max_iterations: u32 },  // safety limit (default: 1000)
}

impl TopologyExecutor {
    /// Determine execution mode from template type
    pub fn mode_for(template: TopologyTemplate) -> ExecutionMode {
        match template {
            TopologyTemplate::Sequential
            | TopologyTemplate::Parallel
            | TopologyTemplate::Hierarchical
            | TopologyTemplate::Brainstorming => ExecutionMode::Static,

            TopologyTemplate::AVR
            | TopologyTemplate::Hub
            | TopologyTemplate::Debate
            | TopologyTemplate::SelfMoA
            | TopologyTemplate::Custom(_) => ExecutionMode::Dynamic { max_iterations: 1000 },
        }
    }

    /// Execute topology — returns ordered list of (node_id, should_execute)
    /// Static: returns full topological order upfront
    /// Dynamic: returns next ready node(s), caller must loop until exit gate opens
    pub fn next_ready(&mut self, graph: &TopologyGraph) -> Vec<String>;
}
```

**Why both modes:**
- **Static** (petgraph `Topo`): Deterministic, predictable, zero overhead. Perfect for 70% of topologies.
- **Dynamic** (gate polling): Handles loops (controller reopens gates), switches (closes non-taken branches), and handoffs (activates/deactivates spokes). Essential for advanced topologies.

**Note:** The executor is Rust-side only. Python `AgentLoop` calls `executor.next_ready()` to get the next node(s), then executes them. This keeps the scheduling logic in Rust while keeping LLM calls in Python.

## 8. Python Integration (boot.py + agent_loop.py)

### boot.py changes
```python
# Phase 1: Rust router + Python shadow oracle
from sage_core import SystemRouter, ModelRegistry, RoutingConstraints
registry = ModelRegistry.from_toml("config/cards.toml")
rust_router = SystemRouter(registry=registry)

# Shadow mode: run both, log divergences
python_router = AdaptiveRouter(...)  # KEPT as reference oracle
# Log: if rust_decision.system != python_decision.system → warning

# Phase 2: add hybrid verifier + template store
from sage_core import TemplateStore, HybridVerifier
template_store = TemplateStore.from_defaults()  # 5 typed templates
rust_router.set_template_store(template_store)

# Phase 3: add contextual bandit
rust_router.enable_bandit(persistence_path=str(Path.home() / ".sage" / "topology.db"))

# Phase 4: add S-MMU topology retrieval + MAP-Elites evolution engine
from sage_core import DynamicTopologyEngine
topo_engine = DynamicTopologyEngine(
    archive=MapElitesArchive.load_or_create(str(Path.home() / ".sage" / "topology.db")),
    registry=registry,
    smmu=working_memory,       # S-MMU for "similar task → best topology"
    embedder=rust_embedder,    # arctic-embed-m for task embeddings
)
rust_router.set_topology_engine(topo_engine)
# Now route() queries S-MMU, archive, and can mutate topologies
# evolve() runs in background every 100 requests
```

### agent_loop.py changes
```python
# ROUTING: single Rust call with explicit constraints
constraints = RoutingConstraints(
    max_cost_usd=budget, max_latency_ms=5000.0,
    min_quality=0.0, required_capabilities=[],
    security_label=0, exploration_budget=0.1,
)
decision = system.rust_router.route(task, constraints)
# decision.decision_id → Ulid (for record_outcome)
# decision.system → S1 | S2 | S3
# decision.model_id → "gemini-2.5-flash"
# decision.topology_id → Ulid (opaque, lazy-load)

# LEARN: feedback to bandit
system.rust_router.record_outcome(decision.decision_id, quality, cost, latency)
```

## 9. Cargo.toml Changes

```toml
[features]
default = []
sandbox = ["wasmtime", "dep:wasmtime-wasi"]
cranelift = ["wasmtime/cranelift"]
onnx = ["dep:ort", "dep:tokenizers", "dep:ndarray"]
tool-executor = ["dep:tree-sitter", "dep:tree-sitter-python", "dep:process-wrap"]
cognitive = ["dep:rusqlite"]      # Phase 1: ModelCard/SystemRouter + optional SQLite
smt = ["dep:oxiz"]                # Phase 3: OxiZ pure-Rust SMT solver (no C++ deps)

[dependencies]
# Existing deps used by new modules:
# petgraph = "0.6.4"      (TopologyGraph DAG + is_cyclic_directed validation)
# dashmap = "6"            (concurrent BanditState)
# chrono = "0.4"           (timestamps)
# serde/serde_json         (serialization)
# ulid                     (topology IDs)
# arrow/pyo3-arrow         (persistence)

# New dependencies (Phase 1):
toml = "0.8"               # cards.toml parsing
rand = "0.9"               # Thompson sampling
rusqlite = { version = "0.33", features = ["bundled"], optional = true }  # SQLite persistence
tracing = "0.1"            # Structured observability (routing decisions, Pareto samples)
tracing-subscriber = { version = "0.3", features = ["env-filter"], optional = true }

# Phase 3:
oxiz = { version = "0.1", optional = true }  # Pure Rust SMT solver (replaces z3-sys)
```

## 10. New Rust File Structure

```
sage-core/src/
├── routing/
│   ├── mod.rs              (existing, updated)
│   ├── features.rs         (existing, unchanged)
│   ├── router.rs           (existing Stage 1 ONNX, kept behind onnx feature)
│   ├── model_card.rs       (DONE — Phase 1)
│   ├── model_registry.rs   (DONE — Phase 1)
│   ├── system_router.rs    (DONE — Phase 1, extends in Phase 2)
│   └── bandit.rs           (NEW — Phase 3: contextual bandit)
├── topology/
│   ├── mod.rs              (NEW — Phase 2)
│   ├── topology_graph.rs   (NEW — Phase 2: unified with Contract IR)
│   ├── templates.rs        (NEW — Phase 2: 8 typed templates)
│   ├── verifier.rs         (NEW — Phase 2: hybrid verifier + semantic diagnosis)
│   ├── smmu_bridge.rs      (NEW — Phase 4: S-MMU topology read/write)
│   ├── map_elites.rs       (NEW — Phase 4: quality-diversity archive)
│   ├── mutations.rs        (NEW — Phase 4: 7 topology mutation operators)
│   ├── llm_synthesis.rs    (NEW — Phase 4: LLM-synthesized topology generation)
│   ├── executor.rs         (NEW — Phase 4: dual-mode topology executor)
│   └── engine.rs           (NEW — Phase 4: DynamicTopologyEngine)
├── config/
│   └── cards.toml          (DONE — Phase 1)
└── tests/
    ├── test_model_card.rs    (DONE — Phase 1)
    ├── test_system_router.rs (DONE — Phase 1)
    ├── test_topology.rs      (NEW — Phase 2)
    ├── test_verifier.rs      (NEW — Phase 2)
    ├── test_bandit.rs        (NEW — Phase 3)
    └── test_map_elites.rs    (NEW — Phase 4)
```

## 11. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Routing latency (P99) | **< 20ms** (full decision incl. ONNX embed) | criterion benchmark with 1000 queries |
| Routing latency (no embed) | < 0.5ms (structural features only) | Phase 1 current path |
| Routing accuracy | ≥ 80% on 45 GT tasks | routing_quality benchmark (NOT self-consistency) |
| Shadow mode divergence | < 5% Rust vs Python on 1000+ traces | Before any Python deletion |
| Template selection | < 1ms per template lookup (8 templates) | Benchmark template store |
| S-MMU topology retrieval | < 5ms for top-5 similar tasks | Benchmark smmu_bridge query |
| Topology generation (with mutation) | < 10ms per new topology | Benchmark engine.generate() |
| MAP-Elites archive coverage | ≥ 80% of behavior space covered after 1000 evolve cycles | Track non-empty cells |
| LLM synthesis success rate | ≥ 60% of synthesized topologies pass verifier | Track synthesis attempts vs accepts |
| LLM synthesis invocation rate | < 10% of total route() calls | Monitor synthesis trigger frequency |
| Dual-mode executor correctness | 100% on topology execution test suite (static + dynamic) | Integration tests |
| Bandit convergence | Front stabilizes within 200 queries (warm-start) | Track front size + decay factor |
| Graph validation (S1/S2) | < 0.015ms (hybrid verifier) | Benchmark is_cyclic + bitsets |
| OxiZ SMT validation (S3) | < 50ms P99 (async, background) | Benchmark offline |
| Cross-session restore | < 100ms boot from SQLite | Measure boot time delta |
| Downstream quality | Rust router ON ≥ best-fixed-model baseline | A/B paired comparison |
| All existing tests pass | 1036+ Python, ~94+ Rust | CI green |

## 12. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Router doesn't beat "always best model" baseline | **Phase 0.5 shadow mode**: measure before deleting. Downstream quality tracking mandatory. |
| ~~Z3 Rust bindings don't exist~~ | **Resolved**: z3/z3-sys DO exist (docs.rs/z3), but OxiZ preferred (pure Rust, no C++ deps). Behind `smt` feature flag. |
| Contextual bandit convergence slow | Warm-start from sN_affinity priors + UCB exploration bonus first 50 obs + decay_factor=0.995 |
| ONNX embedding latency dominates routing | Embedding is tie-breaker only (invoked < 20% of decisions). Phase 1 = structural-only. |
| Hash fallback produces meaningless vectors | **Hard-fail** to structural-only scoring. Hash fallback FORBIDDEN for routing (still OK for S-MMU dedup). |
| PyO3 FFI overhead for TopologyGraph | Return opaque Ulid, lazy-load via Rust-side get() |
| TopologyGraph duplicates Contract IR | **Unified IR**: TopologyNode extends TaskNode fields. Single graph abstraction. |
| LLM mutation callback latency | Mutation is offline (Phase 4+), not on hot path |
| LLM synthesis generates invalid topologies | HybridVerifier + semantic diagnosis rejects invalid. No retry — fallback to template. ≥ 60% success rate target. |
| LLM synthesis API cost runaway | Rate-limited (1/min), budget deducted from RoutingConstraints, only invoked on cache miss + high exploration_budget |
| Gate-based executor infinite loop | `max_iterations` safety limit (default 1000). Timeout at executor level. |
| Non-stationary LLM environment | Exponential decay factor (0.995) in ContextualBandit + arm-level volatility tracking |
| Routing is a black box | **tracing** crate + structured spans: `routing_decision`, `bandit_select`, `graph_validate` |
| Documentation drift (test counts, capabilities) | Single `BenchmarkManifest` as CI source of truth. Auto-generated from test runner output. |

## 13. A2A Protocol Integration (Future — pinned to v1.0)

**Version pin:** A2A v1.0 (breaking changes from v0.3: `protocolVersion` moved to per-interface, `url` removed from top-level, Part types unified, enums SCREAMING_SNAKE_CASE, JWS signature verification for Agent Cards).

YGN-SAGE's ModelCard is analogous to A2A Agent Cards but specialized for LLM model selection. Future integration:
- Export ModelCards as A2A v1.0 compatible `/.well-known/agent-card.json` (NOT `agent.json` — changed in v1.0)
- Import external A2A Agent Cards into ModelRegistry for cross-system routing
- Use A2A task lifecycle (send/receive/stream) for multi-system topology execution
- **Do NOT couple ModelCard internal schema to A2A wire format** — use an explicit projection layer

## 14. Phased Execution (revised from 3 phases to 5)

| Phase | Name | Key Deliverable | Evidence Gate |
|-------|------|-----------------|---------------|
| **1** (DONE) | ModelCard + SystemRouter | Rust routing with Python fallback | ≥ 80% on 45 GT tasks |
| **1.5** | Shadow Mode + Evidence | Dual routing (Rust + Python), divergence logging | < 5% divergence on 1000+ traces |
| **2** | Hybrid Verifier + Templates | Runtime contracts, 8 typed templates, unified IR, 3-flow edges, semantic diagnosis | Template selection < 1ms, downstream ≥ baseline |
| **3** | Contextual Bandit | LinUCB-based selection, SQLite persistence | Front stabilizes in 200 queries |
| **4** | S-MMU + MAP-Elites + LLM Synthesis + Custom DAGs | S-MMU feedback loop, 7 mutations, LLM synthesis (3-stage), DynamicTopologyEngine, dual-mode executor, offline evolution | Archive 80%+, LLM synthesis ≥ 60% valid, custom topologies beat templates on 20%+ tasks |
| **5** | Python Cleanup | Delete frozen Python oracles | All evidence gates passed |

## 15. Embedding Model

**Model:** Snowflake/snowflake-arctic-embed-m (108.9M params, 768-dim, BERT architecture)
**Format:** ONNX INT8 (~110MB)
**Latency:** ~12-15ms on modern CPU (2026), within < 20ms P99 budget
**Usage:** S-MMU semantic edges (primary), routing tie-breaker (secondary, invoked < 20% of decisions)
**Fallback:** Structural-only scoring (NO hash fallback for routing decisions)

**Note (LLMRouterBench insight):** Embedding backbone has limited impact on routing quality specifically. But for S-MMU semantic memory retrieval, higher quality embeddings (MTEB 64.3 vs 58.8) yield measurably better recall.

## 16. References

- DyTopo: arXiv 2602.06039 (Feb 2026) — sparse dynamic topology
- AdaptOrch: arXiv 2602.16873 (Feb 2026) — topology > model capability
- GTD: arXiv 2510.07799 (Oct 2025) — diffusion-guided topology
- Google A2A Protocol v1.0 (2025-2026) — Agent Card discovery + handoff
- LLMRouterBench: arXiv 2601.07206 (Jan 2026) — 400K unified routing eval
- PILOT: arXiv 2508.21141 (Aug 2025) — contextual bandit LLM routing with budget
- Contextual Bandits for Multi-LLM: arXiv 2506.17670 (Jun 2025) — LinUCB + budget-aware
- LLM Bandit: arXiv 2502.02743 (Feb 2025) — preference-conditioned MAB
- Pareto Regret in MOMAB: arXiv 2212.00884 (Dec 2022) — multi-objective theory
- MAP-Elites: Mouret & Clune (2015), GAME: arXiv 2505.06617
- Multi-Armed Bandits + LLM: arXiv 2505.13355
- RouterArena: arXiv 2510.00202 (Sep 2025) — open LLM router comparison platform
- MASFactory: arXiv 2603.06007 (Mar 2026) — Vibe Graphing: LLM compiles NL → executable graph, 3-flow edge model, 11 composed templates, 84.76% HumanEval
- OFA-MAS: arXiv 2601.12996 (Jan 2026) — MoE graph generative model for universal MAS topology (WWW '26)
- Topology Structure Learning: arXiv 2505.22467 (May 2025) — 3-stage framework, up to 10% performance gap between topologies
