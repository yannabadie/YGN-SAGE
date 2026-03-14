# Cognitive Orchestration Pipeline — Design Spec

**Date:** 2026-03-14
**Author:** Yann Abadie + Claude Opus 4.6
**Status:** Draft
**Scope:** Phase B (static pipeline) — Phase C (runtime adaptation) deferred

## Problem Statement

YGN-SAGE has 17 ModelCards across 6 providers (22 fields, 7 domain_scores each) in `cards.toml`, but TopologyRunner executes every node with the **same LLM**. The `TopologyNode.model_id` field exists in Rust but is never read by Python at execution time. The ModelCard system — the core differentiator for multiprovider+multiagent orchestration — is architecturally present but functionally dead.

Additionally, the codebase has accumulated incoherence:
- 178 LOC dead code (`DynamicRouter`)
- ~80 LOC dead code (`TopologyPlanner/StochasticDTS`)
- Two classes named `ModelRegistry` in different modules
- 3 Rust modules marked `#[deprecated]` but actively used in production

## Research Basis

17 papers analyzed across ExoCortex, local PDFs, and web/arXiv (full details in `memory/research_cognitive_orchestration_pipeline.md`):

| Paper | Venue | Key Technique Adopted |
|-------|-------|----------------------|
| OFA-MAS (2601.12996) | WWW 2026 | Per-node `LLM_i` formalization |
| AdaptOrch (2602.16873) | arXiv 2026 | DAG features (ω,δ,γ) for topology routing |
| OpenSage (2602.16891) | ICML | AI-driven model-per-sub-agent assignment |
| SYMPHONY (2601.22623) | NeurIPS 2025 | UCB on heterogeneous model pool |
| Cascade Routing (2410.10347) | ICML 2025 | Quality estimation > routing algorithm |
| AgentDropout (2503.18891) | ACL 2025 | Runtime agent pruning (Phase C hook) |

**Key finding:** AdaptOrch proves `Var_topology / Var_model >= 20` for code tasks — topology choice has 20x more impact than model choice. But within a chosen topology, per-node model assignment using capability profiles yields measurable gains (OpenSage: Gemini 3 Pro planning + GPT-5 Mini execution matches GPT-5 at lower cost).

## Design Decision: Approach B — Unified Rust Core

**Assignment logic in Rust, orchestration in Python.**

Rationale:
- Rust `ModelRegistry` + `ModelCard` already implement `calibrated_affinity()`, `domain_score()`, `estimate_cost()` — no duplication
- Sub-ms assignment matters when evaluating multiple topology candidates (10+ nodes × multiple topologies in MAP-Elites)
- Python fallback guarantees progressive enhancement
- No new Rust modules for orchestration — only a focused `ModelAssigner` that composes existing Rust primitives

Rejected alternatives:
- **Approach A (Python-first):** Would duplicate ModelCard scoring logic. Rust ModelRegistry telemetry calibration (P95 ring buffer, Bayesian blending) would be reimplemented poorly.
- **Approach C (Hybrid):** Acceptable but less principled. Assignment is a hot path in topology evolution — Rust gives measurable speedup.

## Architecture Overview

```
TASK INPUT
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 0: CLASSIFY                                      │
│  AdaptiveRouter.assess_complexity() → system + domain   │
│  Reuses: ComplexityRouter, KnnRouter (92% accuracy)     │
│  New: _infer_domain() heuristic (~20 LOC)               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Stage 1: DECOMPOSE (S2/S3 only, skip for S1)          │
│  TaskPlanner.plan_auto() → TaskDAG                      │
│  New: compute_dag_features(dag) → ω, δ, γ (~30 LOC)    │
│  Reuses: ContractPlanner (existing, never wired)        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Stage 2: SELECT TOPOLOGY                               │
│  New Path 0: select_macro_topology(ω,δ,γ) → hint       │
│  DynamicTopologyEngine.generate(task, system, hint)     │
│  Reuses: 6-path engine (S-MMU→archive→LLM→mut→MCTS→tpl)│
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Stage 3: ASSIGN MODELS (Rust ModelAssigner)            │
│  For each node: filter by caps → score by affinity +    │
│  domain + cost → assign best model_id in-place          │
│  Reuses: ModelRegistry.calibrated_affinity(),           │
│  ModelCard.domain_score(), ModelCard.estimate_cost()     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Stage 4: EXECUTE                                       │
│  TopologyRunner with ProviderPool                       │
│  Each node resolves model_id → provider at execution    │
│  Reuses: TopologyExecutor scheduling (Kahn/gate-based)  │
│  Hook: _check_adaptation() → None (Phase C placeholder) │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Stage 5: LEARN                                         │
│  ContextualBandit.record_outcome()                      │
│  ModelRegistry.record_telemetry_full()                   │
│  MAP-Elites archive topology+assignment                  │
│  EventBus emit PIPELINE events                          │
│  Reuses: all existing learning infrastructure            │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Cleanup (prerequisite)

**Delete:**
- `sage-python/src/sage/routing/dynamic.py` — DynamicRouter (178 LOC, never instantiated in production, superseded by CognitiveOrchestrator)
- `sage-python/src/sage/topology/planner.py` — TopologyPlanner/StochasticDTS (~80 LOC, superseded by Rust DynamicTopologyEngine)
- Associated dead test imports

**Rename:**
- `sage.llm.model_registry.ModelRegistry` → `ModelCardCatalog`
- Update all imports in boot.py, tests, and any other callers

**Fix deprecated tags:**
- **Remove** `#[deprecated]` from `SystemRouter` (system_router.rs) and `TopologyEngine` (engine.rs) — both actively used in production
- **Keep** `#[deprecated]` on Rust `AdaptiveRouter` (router.rs) and `TopologyBridge` (smmu_bridge.rs) — genuinely unused
- Update CLAUDE.md deprecated section

**Add setter:**
- `TopologyNode.model_id`: change from `#[pyo3(get)]` to `#[pyo3(get, set)]`

### 2. Rust ModelAssigner (`sage-core/src/routing/model_assigner.rs`)

New PyO3 class. ~150 LOC. Behind no feature flag (uses only ModelRegistry + TopologyGraph, both always compiled).

```rust
#[pyclass]
pub struct ModelAssigner {
    registry: ModelRegistry,
}

#[pymethods]
impl ModelAssigner {
    #[new]
    fn new(registry: &ModelRegistry) -> Self;

    /// Assign model_id to every node in the topology graph.
    /// Modifies graph in-place. Returns number of nodes assigned.
    fn assign_models(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
    ) -> PyResult<usize>;

    /// Assign a single node (for Phase C runtime re-assignment).
    fn assign_single_node(
        &self,
        graph: &mut TopologyGraph,
        node_idx: usize,
        task_domain: &str,
        budget_usd: f32,
    ) -> PyResult<String>;
}
```

**Assignment algorithm per node:**

```
For each node in TopologyGraph (topological order):
  1. Extract: role, system (S1/S2/S3), required_capabilities, max_cost_usd
  2. Filter ModelCards by:
     - supports_tools if "tools" in required_capabilities
     - supports_json_mode if "json" in required_capabilities
     - estimate_cost(1000, 500) <= min(node.max_cost_usd, remaining_budget)
  3. Score candidates:
     score = WEIGHT_AFFINITY * calibrated_affinity(model_id, node.system)
           + WEIGHT_DOMAIN   * domain_score(task_domain)
           + WEIGHT_COST     * (1.0 - cost_normalized)
     Constants: WEIGHT_AFFINITY=0.4, WEIGHT_DOMAIN=0.4, WEIGHT_COST=0.2
  4. Select highest scorer → set node.model_id via graph.node_weight_mut()
  5. Deduct estimated cost from remaining_budget
```

`calibrated_affinity` blends card prior with telemetry observations: `w = min(count/50, 0.8); (1-w)*card_affinity + w*observed_quality`. Already implemented in `model_registry.rs`.

**Edge case:** If no candidate passes filters (all too expensive or missing capabilities), keep the node's existing model_id unchanged and log a warning.

### 3. Python ModelAssigner fallback (`sage-python/src/sage/llm/model_assigner.py`)

~60 LOC. Same algorithm using `ModelCardCatalog` (the renamed Python ModelRegistry). Used when `sage_core` is not compiled. Field-for-field compatible with Rust version.

### 4. ProviderPool (`sage-python/src/sage/llm/provider_pool.py`)

~80 LOC. Resolves `model_id` → `(LLMProvider, LLMConfig)` at execution time.

```python
class ProviderPool:
    def __init__(self, default_provider: LLMProvider, registry: ModelProfileRegistry):
        self._default = default_provider
        self._registry = registry  # sage.providers.registry (runtime discovery)
        self._cache: dict[str, tuple[LLMProvider, LLMConfig]] = {}

    def resolve(self, model_id: str) -> tuple[LLMProvider, LLMConfig]:
        """Resolve model_id to instantiated provider + config.
        Falls back to default_provider if model_id unknown or unavailable."""
```

Uses `sage.providers.registry.ModelRegistry` (the runtime discovery registry that knows which providers have valid API keys). Does not create new providers — reuses those discovered at boot via `registry.refresh()`.

Cache keyed by model_id to avoid repeated lookups. Cache invalidation: none needed (provider availability doesn't change within a session).

### 5. TopologyRunner modification

~15 LOC changed in `sage-python/src/sage/topology/runner.py`.

```python
# __init__ gains optional provider_pool parameter:
def __init__(self, graph, executor, llm_provider, llm_config=None, provider_pool=None):
    ...
    self._provider_pool = provider_pool

# _execute_node resolves per-node provider:
node_model_id = getattr(node, "model_id", "")
if node_model_id and self._provider_pool:
    provider, config = self._provider_pool.resolve(node_model_id)
else:
    provider, config = self._llm, self._config
response = await provider.generate(messages=messages, config=config)
```

Backward compatible: without `provider_pool`, behavior is identical to today.

### 6. Pipeline stages (`sage-python/src/sage/pipeline_stages.py`)

~150 LOC. Pure functions, one per stage.

**`_infer_domain(task, profile)`** (~20 LOC): Maps CognitiveProfile signals to ModelCard domain names. Uses keyword presence (code patterns → "code", math symbols → "math", etc.). No LLM call.

**`compute_dag_features(dag)`** (~30 LOC): Computes AdaptOrch's 3 DAG structural metrics:
- ω (parallelism width): maximum antichain size
- δ (critical path depth): longest weighted path
- γ (coupling density): average edge weight / max possible

**`select_macro_topology(features)`** (~25 LOC): AdaptOrch routing heuristic with thresholds θ_ω=0.5, θ_γ=0.6, θ_δ=5. Returns topology template hint (sequential/parallel/hierarchical/hybrid).

### 7. CognitiveOrchestrationPipeline (`sage-python/src/sage/pipeline.py`)

~250 LOC. Chains the 5 stages.

```python
@dataclass
class PipelineContext:
    task: str
    budget: float = 5.0
    domain: str = ""
    system: int = 0
    task_dag: Any = None
    dag_features: DAGFeatures | None = None
    topology: Any = None
    assignments: dict[int, str] = field(default_factory=dict)
    result: str = ""

class CognitiveOrchestrationPipeline:
    def __init__(self, router, engine, assigner, provider_pool,
                 bandit, quality_estimator, event_bus):
        ...

    async def run(self, task: str, budget_usd: float = 5.0) -> str:
        ctx = PipelineContext(task=task, budget=budget_usd)
        ctx = stage_classify(ctx, self.router)
        ctx = stage_decompose(ctx)
        ctx = stage_select_topology(ctx, self.engine)
        ctx = stage_assign_models(ctx, self.assigner)
        ctx = await stage_execute(ctx, self._make_runner)
        self._record_outcome(ctx)
        return ctx.result
```

### 8. Boot wiring (`sage-python/src/sage/boot.py`)

~50 LOC modified. After existing instantiations:

```python
# ModelAssigner (Rust primary, Python fallback)
try:
    from sage_core import ModelAssigner as RustModelAssigner
    model_assigner = RustModelAssigner(rust_registry)
except ImportError:
    from sage.llm.model_assigner import ModelAssigner as PyModelAssigner
    model_assigner = PyModelAssigner(py_model_card_catalog)

# ProviderPool
provider_pool = ProviderPool(default_provider=llm_provider, registry=model_profile_registry)

# Pipeline
pipeline = CognitiveOrchestrationPipeline(
    router=metacognition, engine=topology_engine, assigner=model_assigner,
    provider_pool=provider_pool, bandit=bandit,
    quality_estimator=quality_est, event_bus=event_bus,
)
```

`AgentSystem.run()` delegates to pipeline when available, falls back to `_run_legacy()` (current inline code, preserved unchanged) when pipeline is None.

## Phase C Hooks (not implemented, structurally prepared)

Four adaptation points in `stage_execute`, after each node completion:

| Action | Trigger | Method |
|--------|---------|--------|
| Model upgrade | QualityEstimator score < θ_min | `assigner.assign_single_node()` + retry |
| Agent pruning | Importance score < threshold | `executor.mark_completed()` (skip) |
| Topology re-route | ConsistencyScore < θ (parallel outputs) | `engine.generate()` with tighter constraints |
| Sub-agent spawn | Emergent sub-task detected | `agent_pool.create()` |

`_check_adaptation()` returns `None` in Phase B. The `assign_single_node()` Rust method is included in Phase B to avoid a breaking Rust API change in Phase C.

## Files Changed

| Action | File | LOC |
|--------|------|-----|
| CREATE | `sage-core/src/routing/model_assigner.rs` | ~150 |
| CREATE | `sage-python/src/sage/pipeline.py` | ~250 |
| CREATE | `sage-python/src/sage/pipeline_stages.py` | ~150 |
| CREATE | `sage-python/src/sage/llm/provider_pool.py` | ~80 |
| CREATE | `sage-python/src/sage/llm/model_assigner.py` | ~60 |
| MODIFY | `sage-core/src/topology/topology_graph.rs` | ~5 |
| MODIFY | `sage-core/src/lib.rs` | ~3 |
| MODIFY | `sage-core/src/routing/system_router.rs` | ~1 (remove deprecated) |
| MODIFY | `sage-core/src/topology/engine.rs` | ~1 (remove deprecated) |
| MODIFY | `sage-python/src/sage/topology/runner.py` | ~15 |
| MODIFY | `sage-python/src/sage/boot.py` | ~50 |
| MODIFY | `sage-python/src/sage/llm/model_registry.py` | ~10 (rename) |
| MODIFY | `CLAUDE.md` | ~30 |
| DELETE | `sage-python/src/sage/routing/dynamic.py` | -178 |
| DELETE | `sage-python/src/sage/topology/planner.py` | -80 |
| TESTS | 7 test files | ~400 |
| **Net** | | **~+950, -258** |

## Testing Strategy

| Test File | Scope | Type |
|-----------|-------|------|
| `tests/test_model_assigner.py` | Rust/Python assigner: domain scoring, budget, capabilities | Unit, mock registry |
| `tests/test_provider_pool.py` | model_id → provider resolution, fallback, cache | Unit |
| `tests/test_pipeline.py` | Full 5-stage pipeline with mocks | Integration |
| `tests/test_pipeline_stages.py` | Each stage function in isolation | Unit |
| `tests/test_dag_features.py` | ω, δ, γ computation on known DAGs | Unit, deterministic |
| `tests/test_cleanup.py` | ModelCardCatalog rename, DynamicRouter removed | Regression |
| `sage-core: model_assigner tests` | Rust assignment: filtering, scoring, budget | cargo test |

No real LLM calls in tests. Existing E2E proof (`tests/e2e_proof.py`) validates full integration.

## Success Criteria

1. `TopologyRunner` executes nodes with **different models** based on ModelCard scores
2. `ModelAssigner` assigns optimal model_id per node in < 1ms for 10-node topology
3. Pipeline emits observable events on EventBus at each stage transition
4. All existing tests pass (zero regressions)
5. Progressive enhancement: full pipeline with sage_core, graceful Python fallback without
6. Dead code removed, naming ambiguity resolved, deprecated tags honest
