# Phase C: Runtime Adaptation — Design Spec

**Date:** 2026-03-14
**Author:** Yann Abadie + Claude Opus 4.6
**Status:** Approved

**Note:** Fixes latency_ms units bug in Phase B `_stage_learn` (was dividing by 1000, param expects ms).
**Prerequisite:** Phase B (Cognitive Orchestration Pipeline) — complete on `dev`

## Problem Statement

Phase B delivers a 5-stage pipeline (Classify → Decompose → Select Topology → Assign Models → Execute) that makes optimal decisions **upfront** but cannot adapt during execution. If a node produces low-quality output, the pipeline has no recourse — it continues with a bad result.

Additionally, OxiZ (sub-0.1ms Rust SMT solver) with 10 PyO3 methods is wired but dormant — locked behind `validation_level >= 3` which triggers on only ~10% of tasks. Four z3_verify.py functions are dead code (never imported). The contracts modules (executor.py, repair.py, policy.py) are annotated RESERVED but unconnected.

## Research Basis

| Paper | Venue | Technique Adopted |
|-------|-------|-------------------|
| AgentDropout (2503.18891) | ACL 2025 | Runtime agent pruning (-21.6% tokens) |
| AdaptOrch (2602.16873) | arXiv 2026 | Consistency scoring + topology re-routing |
| OpenSage (2602.16891) | ICML | Runtime sub-agent spawn |
| Self-Regulation (2502.04576) | arXiv | PRM-guided model escalation |
| Cascade Routing (2410.10347) | ICML 2025 | Quality estimation as bottleneck |
| SYMPHONY (2601.22623) | NeurIPS 2025 | UCB-driven heterogeneous model scheduling |

## Design Overview

Two layers: (1) wire existing OxiZ/contracts into the pipeline, (2) build the TopologyController that decides when and how to adapt.

```
Stage 3 (ASSIGN):
  ModelAssigner.assign_models()           ← scoring heuristique (Phase B)
  + verify_provider_assignment() via OxiZ ← preuve formelle (Phase C)
  + PolicyVerifier.verify_all()           ← security/budget gates (Phase C)

Stage 4 (EXECUTE) per-node loop:
  execute_node(i) → result
  TopologyController.evaluate_and_decide():
    quality = QualityEstimator (80%) + PRM lightweight (20%)
    IF quality >= 0.7          → CONTINUE
    IF quality < 0.3, retry<2  → UPGRADE_MODEL (assign_single_node)
    IF parallel inconsistency  → REROUTE_TOPOLOGY (AdaptOrch γ+0.2)
    IF importance < 0.2        → PRUNE_NODE (AgentDropout)
    IF emergent sub-task       → SPAWN_SUBAGENT (OpenSage)
    ELSE                       → CONTINUE (accept imperfect)

Stage 5 (LEARN):
  QualityEstimator.estimate()             ← 5 signals (Phase B)
  + PRM.calculate_r_path() lightweight    ← 6th formal signal (Phase C)
```

## Component Details

### 1. OxiZ Wiring (5 connections, no new Rust modules)

**1.1 verify_provider_assignment() in Stage 3**

Currently dead code in `contracts/z3_verify.py`. Wire into `pipeline.py:_stage_assign_models()` after ModelAssigner completes. Uses OxiZ SAT solver (exactly-one boolean encoding) to prove assignment validity. Falls back gracefully if OxiZ unavailable.

**IR impedance mismatch:** `verify_provider_assignment()` takes `TaskDAG` + `list[ProviderSpec]`, but the pipeline operates on `TopologyGraph`. Solution: create a lightweight adapter function `topology_to_assignment_spec(topology) -> (nodes, providers)` that extracts the fields z3_verify needs (node capabilities, model_ids, security labels) without converting the entire IR. ~20 LOC in `pipeline.py`. The adapter does NOT create a TaskDAG — it builds the minimal dicts that `verify_provider_assignment` actually inspects.

If verification fails: log warning + emit EventBus PIPELINE:ASSIGN_VERIFICATION_FAILED, but do NOT block execution (assignment heuristic is still valid, just not formally proven).

**1.2 PolicyVerifier in Stage 4 (pre-node)**

Add `verify_node(node, predecessors, budget_remaining)` as a `@staticmethod` on `PolicyVerifier` in `contracts/policy.py` (RESERVED file). This is node-scoped — does NOT require the TaskDAG-dependent constructor. Checks:
- Info-flow: node.security_label >= max(pred.security_label for pred in predecessors)
- Budget: node.max_cost_usd <= budget_remaining
- Fan-in: len(predecessors) <= max_fan_in (default 5)

Uses duck-typing on node attributes (works with both TopologyNode and TaskNode).

Called before each node in TopologyRunner. Violation → skip node + warning.

**1.3 PRM lightweight in Stage 5**

In `_stage_learn()`, if result contains structured reasoning (`<think>` blocks, code assertions, math expressions): run `ProcessRewardModel.calculate_r_path()` for a formal quality signal. Blend with heuristic quality at 80/20 weight.

**Guard:** Detect structured content BEFORE calling PRM: check for `<think>` tags, `assert` statements, or math expressions. Do NOT call `calculate_r_path()` on plain text — it returns -1.0 ("penalty for not reasoning") which would drag quality negative. PRM exceptions are caught and logged (not silent `pass`), falling back to heuristic-only quality.

**1.4 Topology pre-validation**

Before executing a multi-node topology, verify budget feasibility: sum of all node.max_cost_usd <= ctx.budget. Uses OxiZ `verify_arithmetic_expr()` for formal proof, falls back to Python arithmetic.

**1.5 verify_invariant_with_feedback() in adaptation**

When TopologyController decides UPGRADE_MODEL on an S3 node, use clause-level feedback from `verify_invariant_with_feedback()` to guide the retry prompt: "Your response failed on clause: {violated_clause}. Fix this specific issue."

### 2. Rust: Batch cosine similarity on RustEmbedder

New method on existing `sage-core/src/memory/embedder.rs`:

```rust
/// Compute pairwise cosine similarity for a batch of texts.
/// Returns flattened upper-triangle similarity matrix.
/// NOTE: This is a #[pymethods] method — requires `py: Python<'_>` token.
/// Delegates to self.embed_batch() internally (which holds GIL for ONNX inference).
#[pyo3(name = "batch_cosine_similarity")]
pub fn py_batch_cosine_similarity(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<f32>> {
    let embeddings = self.embed_batch(py, texts)?;
    // embeddings: Vec<Vec<f32>> (768-dim each)
    let n = embeddings.len();
    let mut sims = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i+1)..n {
            sims.push(cosine_sim(&embeddings[i], &embeddings[j]));
        }
    }
    Ok(sims)
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 { return 0.0; }
    dot / (norm_a * norm_b)
}
```

Behind `onnx` feature flag (same as RustEmbedder). Python fallback: `sentence-transformers` embeddings + numpy cosine (NOT hash embeddings — hash cosine is meaningless). If only hash embeddings available, skip consistency scoring entirely (return 1.0 to avoid spurious reroutes).

### 3. TopologyController (`sage-python/src/sage/topology_controller.py`)

~200 LOC. Pure Python — orchestration logic with async operations.

```python
@dataclass
class AdaptationDecision:
    action: str  # "continue", "upgrade_model", "prune_node", "reroute_topology", "spawn_subagent"
    target_node: int | None = None
    reason: str = ""
    new_model_id: str | None = None
    invariant_feedback: str | None = None  # clause-level from OxiZ

class TopologyController:
    """Runtime adaptation controller for Pipeline Stage 4.

    Evaluates node output quality after each execution step and decides
    whether to continue, upgrade, prune, reroute, or spawn.
    """

    # Named thresholds (calibrated on TopologyBench results)
    THETA_GOOD = 0.7       # quality above this → continue
    THETA_CRITICAL = 0.3   # quality below this → upgrade model
    THETA_CONSISTENCY = 0.5 # cosine sim below this → reroute topology
    THETA_PRUNE = 0.2      # importance below this → prune node
    MAX_RETRIES = 2         # max upgrade retries per node
    MAX_SPAWNS = 3          # max sub-agent spawns per topology

    def __init__(self, assigner, quality_estimator, prm=None,
                 policy_verifier=None, embedder=None):
        ...

    def evaluate_and_decide(self, node_idx, result, task, topology, ctx) -> AdaptationDecision:
        """Core decision logic — called after each node execution."""
        ...

    def compute_consistency_score(self, outputs: list[str]) -> float:
        """Cosine similarity between parallel node outputs via RustEmbedder."""
        ...

    def compute_importance_score(self, node_idx, result, all_outputs) -> float:
        """Ratio of unique content contributed by this node."""
        ...
```

### 4. Adaptation actions in TopologyRunner

Modify `topology/runner.py` to accept `controller: TopologyController | None` and call `evaluate_and_decide()` after each node.

**MODEL UPGRADE:** `assigner.assign_single_node()` → resolve via ProviderPool → retry node with new provider. If S3 node + OxiZ available: include `invariant_feedback` in retry prompt.

**AGENT PRUNING:** `executor.mark_completed(node_idx)` without executing. Log importance score.

**TOPOLOGY RE-ROUTE:** Regenerate topology via engine with tighter constraints (coupling γ += 0.2). Re-assign models. Restart execution from scratch on new topology. Max 1 reroute per pipeline run. Counter tracked on `TopologyController._reroute_count` (instance variable, reset per `evaluate_and_decide` session). When limit hit → force CONTINUE with current results.

**SUB-AGENT SPAWN:** Extract emergent sub-task from result. Run via direct LLM call (`llm_provider.generate()`) — NOT `agent_pool.run_single()` (does not exist). Alternatively use `AgentTool.from_agent()` from `tools/agent_tool.py` for richer agent wrapping. Inject sub-result into node outputs. Max 3 spawns per topology, tracked on `TopologyController._spawn_count`.

### 5. ConsistencyScore (`sage-python/src/sage/consistency.py`)

~50 LOC. Python wrapper around Rust `batch_cosine_similarity()` with numpy fallback.

```python
def consistency_score(texts: list[str], embedder=None) -> float:
    """Mean pairwise cosine similarity of text embeddings.

    Uses Rust SIMD embedder if available, numpy fallback otherwise.
    Returns 0.0-1.0 (1.0 = identical outputs, 0.0 = orthogonal).
    """
```

### 6. Boot.py wiring

```python
# After pipeline creation
controller = None
if quality_est and model_assigner:
    from sage.topology_controller import TopologyController
    controller = TopologyController(
        assigner=model_assigner,
        quality_estimator=quality_est,
        prm=prm,
        policy_verifier=PolicyVerifier() if model_assigner else None,
        embedder=embedder,  # RustEmbedder or Python fallback
    )
# Pass to pipeline
pipeline.controller = controller
```

## Files Changed

| Action | File | LOC |
|--------|------|-----|
| CREATE | `sage-python/src/sage/topology_controller.py` | ~200 |
| CREATE | `sage-python/src/sage/consistency.py` | ~50 |
| MODIFY | `sage-core/src/memory/embedder.rs` (batch_cosine_similarity) | ~30 |
| MODIFY | `sage-python/src/sage/pipeline.py` (OxiZ wiring + controller integration) | ~80 |
| MODIFY | `sage-python/src/sage/topology/runner.py` (adaptation loop) | ~60 |
| MODIFY | `sage-python/src/sage/contracts/z3_verify.py` (wire into pipeline) | ~15 |
| MODIFY | `sage-python/src/sage/contracts/policy.py` (add verify_node) | ~30 |
| MODIFY | `sage-python/src/sage/boot.py` (controller wiring) | ~20 |
| MODIFY | `CLAUDE.md` | ~20 |
| TESTS | 4 test files | ~350 |
| **Total net** | | **~+855** |

## Testing Strategy

| Test File | Scope | Type |
|-----------|-------|------|
| `tests/test_topology_controller.py` | 4 actions + thresholds + quality blending + edge cases | Unit, mocks |
| `tests/test_pipeline_adaptation.py` | Pipeline with node failure → upgrade → success | Integration |
| `tests/test_oxiz_pipeline.py` | verify_provider_assignment + PolicyVerifier + PRM in learn | Unit |
| `tests/test_consistency_score.py` | Cosine similarity: identical=1.0, orthogonal≈0.0, Rust/Python parity | Unit |

## Success Criteria

1. Adaptation hook in `TopologyRunner.run()` loop — called after each `_execute_node()`. NOTE: the Phase B spec mentioned `_check_adaptation()` as a stub but it was never implemented. Phase C creates this logic directly in the runner loop, not as a named method.
2. Model upgrade triggered and verified: node quality < 0.3 → new model assigned → retry succeeds
3. Agent pruning measurable: low-importance node skipped, token count reduced
4. Topology re-route triggered: parallel inconsistency → hierarchical → consistent output
5. verify_provider_assignment called after every assign_models (sub-0.1ms overhead)
6. PRM scoring enriches Stage 5 LEARN when structured content detected
7. ConsistencyScore uses Rust SIMD when available, numpy fallback otherwise
8. Zero regressions — controller=None → Phase B behavior unchanged
