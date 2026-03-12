# YGN-SAGE: SOTA Breakout Design Spec

**Date**: 2026-03-12
**Status**: Approved (brainstorming)
**Author**: Claude Opus 4.6 + Yann Abadie
**Objective**: Push YGN-SAGE beyond state-of-the-art while preserving all 5 cognitive pillars

## Context

### Q&A Synthesis

YGN-SAGE is an Agent Development Kit built on 5 cognitive pillars (Topology, Tools, Memory, Evolution, Strategy). A strategic Q&A identified:

1. **Proven differentiator**: S1/S2/S3 cognitive routing with kNN (92% accuracy) + 7-provider FrugalGPT cascade — no competitor has this
2. **Unique capabilities**: topology evolution (MAP-Elites + CMA-ME + MCTS), formal verification (OxiZ/Z3), 4-tier memory, Wasm WASI sandbox — none exist in competing frameworks
3. **Unproven at scale**: topology impact and memory contribution measured on only 20 tasks — not statistically robust
4. **DX gap**: 36K LOC, Rust + Python + ONNX required, ~1 hour setup vs competitors' 5 minutes
5. **Strategic position**: research prototype, not product — needs benchmarks for credibility, DX for adoption

### SOTA Competitive Analysis (Context7, March 2026)

| Capability | SAGE | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|
| Cost-aware routing S1/S2/S3 | kNN 92%, 3 tiers | LLM structured output (1 expensive call per decision) | None | None |
| Multi-provider cascade | 7 providers, FrugalGPT | Single model per graph | Single model per agent | Single model per agent |
| Topology evolution | MAP-Elites + CMA-ME + MCTS | Static StateGraph | Sequential/Hierarchical fixed | Static TypeSubscription |
| Formal verification | OxiZ/Z3, 10 SMT methods | None | None | None |
| Memory | 4 tiers (STM→Episodic→Semantic→ExoCortex) | State checkpointing | Scoped memory per-agent | Message history |
| Sandbox | Wasm WASI + tree-sitter + eBPF | None native | None native | Basic code executor |
| Guardrails | 3-layer (input/runtime/output) | None native | Per-task + retry | Intervention handler |
| Durable execution | None | Yes (persistence, resume) | No | No |
| Human-in-the-loop | None | Yes (interrupt() native) | No | No |
| Per-agent memory scoping | No (global tiers) | No | Yes (memory.scope()) | No |
| Developer experience | Complex (Rust + Python + ONNX) | Simple (pip install) | Simple (pip install) | Moderate |

### Key Research Findings

- **ETH-SRI (ICLR 2025)**: "Quality estimators are the bottleneck, not routing algorithms." → QualityEstimator is the key lever
- **arXiv 2505.12601**: kNN on embeddings beats MLP/GNN/attention-based routers — validated in SAGE (92%)
- **AdaptOrch (2602.16873)**: "topology > model capability" — justifies the Topology pillar
- **NVIDIA DeBERTa-v3-base**: 98.1% multi-head task classification — target for Stage 1 ONNX
- **RouteLLM (ICLR 2025, 2406.18665)**: BERT 0.3B on Chatbot Arena preference data — alternative Stage 1
- **Survey (arXiv 2603.04445)**: 6 routing paradigms — SAGE's AdaptiveRouter validated as SOTA architecture

## Design

### Phase 1: Routing Dominance (~2 sprints)

Triple investment in the proven differentiator.

#### 1.1 Learned QualityEstimator

**Problem**: The 5-signal heuristic QualityEstimator cannot distinguish a correct short answer from an incorrect verbose one. ETH-SRI identifies quality estimation as the cascade routing bottleneck.

**Solution**: Fine-tune a small BERT model on `(task, response, quality_score)` triples.

**Training data**:
- Source: SAGE benchmark runs — HumanEval+ (164) + MBPP+ (378) produce (task, response) pairs with pass/fail labels
- Augmentation: GPT generates task variations → target 2000+ triples
- Label: continuous [0.0, 1.0] derived from pass/fail + partial credit signals
- **Overfitting mitigation** (2000 triples is small for 66M params):
  - 5-fold stratified cross-validation on all training runs
  - Early stopping on validation loss (patience=3 epochs)
  - Aggressive dropout (0.3) + weight decay (0.01)
  - Freeze lower transformer layers (fine-tune only top 2 + classification head ≈ 3M trainable params)
  - Data augmentation: paraphrase tasks via back-translation, swap response ordering
  - **Honest scope**: With 2000 triples and partial fine-tuning, expect modest but measurable improvement over the 5-signal heuristic. The "+5-15%" claim from ETH-SRI is for models trained on 10K+ data points — at our data scale, realistic expectation is +3-8% on held-out quality correlation. Larger gains require community contribution of benchmark results.

**Model selection**:
- Primary: DistilBERT (66M params, ~3M trainable with layer freezing) — good accuracy/size tradeoff
- Alternative: TinyBERT (14.5M) if latency budget is tight (<5ms) — also reduces overfitting risk
- Both exportable to ONNX for Rust inference via `ort`

**Integration**:
- New class: `LearnedQualityEstimator` in `sage-python/src/sage/quality_estimator.py`
- Replaces `QualityEstimator.estimate()` when ONNX model available
- Bandit contextual receives finer quality signals → better cascade decisions
- Fallback: existing 5-signal heuristic when ONNX unavailable

**Training pipeline**:
- New module: `sage-python/src/sage/training/quality_trainer.py`
- Reads benchmark results from `docs/benchmarks/*.json`
- Exports ONNX model to `sage-core/models/quality_estimator.onnx`
- Retraining triggered manually or on benchmark completion

**Expected impact**: +3-8% quality estimation correlation at 2000 triples (honest scope); +5-15% achievable at 10K+ triples (ETH-SRI cascade paper scale).

**Metrics**: measure on held-out HumanEval+ subset — quality score correlation with actual pass/fail.

#### 1.2 DeBERTa-v3-base Task Classifier (Stage 1 ONNX)

**Problem**: kNN at 92% is good but not SOTA. NVIDIA's DeBERTa-v3-base achieves 98.1% on multi-head task classification.

**Solution** (2-step):

1. **Zero-shot evaluation**: Download NVIDIA `prompt-task-and-complexity-classifier`, convert to ONNX, measure accuracy on SAGE's 50 GT tasks. **Statistical note**: 50 samples gives ±6% CI at 95% confidence (binomial). Ship criterion: bootstrap 95% CI lower bound ≥ 90% (not point estimate ≥ 95%). If the lower bound meets the threshold → ship. If the point estimate is >95% but CI is too wide → expand test set to 200 tasks (from Phase 2.3) before deciding.

2. **Fine-tune** (if needed): Use NVIDIA model as initialization, fine-tune on SAGE GT tasks augmented to 500+ via synthetic generation. Evaluate on the 50 original GT tasks (held-out during fine-tuning) + the expanded set from Phase 2.3.

**Architecture**:
- DeBERTa-v3-base (86M params)
- Multi-head output:
  - `complexity`: S1/S2/S3 (maps to cognitive systems)
  - `task_type`: code/math/reasoning/creative/tool_use/factual
- Input: task text, 512-token truncation (same as current Rust router)

**Integration**:
- `AdaptiveRouter` Stage 1 (after structural features Stage 0)
- kNN kept as Stage 0.5 (instantaneous, no model load latency)
- DeBERTa as Stage 1 (more accurate, ~10ms inference)
- Shadow comparison: both run in parallel during transition, JSONL traces (same pattern as existing ShadowRouter)

**Files**:
- Rust: `sage-core/src/routing/router.rs` — add DeBERTa model loading + inference (extends existing ONNX Stage 1 placeholder)
- Python: `sage-python/src/sage/strategy/adaptive_router.py` — wire DeBERTa results into routing pipeline
- Config: `sage-core/models/deberta_router.onnx` — the exported model

**Target**: 96-98% routing accuracy (vs 92% kNN current).

#### 1.3 `sage-router` Standalone Library

**Problem**: Using SAGE routing requires installing the full 36K LOC framework. Zero adoption path.

**Solution**: Extract the routing pipeline as a standalone pip-installable library.

```
pip install sage-router
pip install sage-router[onnx]  # with DeBERTa + kNN ONNX acceleration
```

**API**:
```python
from sage_router import CognitiveRouter, RouteDecision

router = CognitiveRouter()  # auto-loads kNN + ONNX if available
decision: RouteDecision = router.route("Implement a distributed sort algorithm")
# → RouteDecision(system=2, confidence=0.87, method="knn")

# Multi-provider cost optimization
best_model = router.select_model(
    decision, budget_usd=0.01, providers=["google", "openai"]
)
```

**Contents** (all already migrated to Python in the audit sprint):
- `structural_features.py` — Stage 0 keyword extraction
- `knn_router.py` — Stage 0.5 kNN on arctic-embed-m
- `model_card.py` + `model_registry.py` — model catalog with telemetry
- `adaptive_router.py` — 5-stage pipeline orchestrator (simplified: kNN → DeBERTa, no entropy probe or cascade stages)
- DeBERTa ONNX model (optional, in `[onnx]` extra)

**Dependency extraction** (sage-router must be self-contained):
- `CognitiveProfile` + `RoutingDecision` dataclasses: **copy into sage-router** as `sage_router.types` (12 LOC each, no external deps). SAGE itself imports from sage-router (not the other way around).
- `Embedder`: sage-router bundles a minimal embedder wrapper (`sage_router.embedder`) supporting `sentence-transformers` only (no Rust ONNX). The `[onnx]` extra adds `onnxruntime` for direct ONNX inference.
- `ComplexityRouter`/`metacognition.py`: NOT included. sage-router exposes `CognitiveRouter` (new class) that wraps the pipeline without the CGRS self-braking or speculative zone logic (those are SAGE-specific orchestration concerns).

**Zero Rust dependency**. Pure Python with optional ONNX acceleration.

**Packaging**: separate `pyproject.toml` in `sage-router/`, published to PyPI independently. SAGE itself depends on `sage-router` (dogfooding).

**Strategy**: even if people don't use full SAGE, the router alone has standalone value. This is the adoption wedge.

---

### Phase 2: TopologyBench + Scale Proof (~2 sprints)

Prove the Topology pillar's value with evidence no competitor can produce.

#### 2.1 TopologyBench — First Topology Impact Benchmark

**Problem**: AdaptOrch claims "topology > model capability" but no public benchmark measures this. SAGE has 8 templates + MAP-Elites + MCTS + HybridVerifier — and zero proof it matters.

**Solution**: Create TopologyBench, measuring `(task × topology) → performance`.

**Task suite** (200 tasks, 5 categories, graduated difficulty):

| Category | Count | Source | Difficulty Range |
|---|---|---|---|
| Code | 60 | HumanEval+ subset (easy/medium/hard) + multi-file tasks | S1-S3 |
| Math | 40 | Arithmetic → formal proofs | S1-S3 |
| Reasoning | 40 | Logic, analogies, planning | S1-S3 |
| Tool-use | 30 | File I/O, API calls, search | S2-S3 |
| Creative | 30 | Writing, summarization, translation | S1-S2 |

**Topologies tested** (10 configurations):

| # | Topology | Type |
|---|---|---|
| 1 | Sequential | Template (baseline) |
| 2 | Parallel | Template |
| 3 | AVR (Act-Verify-Refine) | Template |
| 4 | SelfMoA (Mixture of Agents) | Template |
| 5 | Hierarchical | Template |
| 6 | Hub | Template |
| 7 | Debate | Template |
| 8 | Brainstorming | Template |
| 9 | Evolved (MAP-Elites/MCTS) | Dynamic |
| 10 | Oracle (human-selected per category) | Upper bound |

**Metrics per (task, topology)**:
- `pass_rate` — correctness (binary or partial credit)
- `latency_ms` — P50, P95
- `cost_usd` — tokens × pricing
- `token_efficiency` — tokens used / minimum estimated tokens

**Output**: 200×10 performance matrix + heatmap. Answers:
1. Does topology impact pass_rate? By how much?
2. Which topology dominates per task category?
3. Does evolved topology (MAP-Elites) beat fixed templates?
4. Is the topology overhead (latency, tokens) justified?

**Implementation**:
- New module: `sage-python/src/sage/bench/topology_bench.py`
- Uses existing `TopologyExecutor` + `BenchmarkRunner` infrastructure
- Results: `docs/benchmarks/YYYY-MM-DD-topology-bench.json`

**Cost estimate** (corrected — multi-agent topologies are NOT single-call):

| Topology | Avg calls/task | Notes |
|---|---|---|
| Sequential | 1 | Single agent |
| Parallel | 2-3 | Concurrent agents |
| AVR | 3 | Act + Verify + Refine |
| SelfMoA | 3-5 | Multiple experts + aggregator |
| Hierarchical | 2-4 | Manager + workers |
| Hub | 2-3 | Hub + spokes |
| Debate | 4-6 | Proposer + opponent + judge (multi-round) |
| Brainstorming | 3-5 | Multiple generators + synthesizer |
| Evolved | 3-5 | Varies by evolved structure |
| Oracle | 1-6 | Depends on category |

**Total**: 200 tasks × ~4.5 avg calls/topology × 10 topologies = **~9000 LLM calls** (not 2000).
Budget at $0.001/call (Gemini Flash): ~$9. At $0.01/call (Gemini Pro for S3): ~$90.
**Mitigation**: Run S1 tasks on Flash only, S3 on Pro. Estimated blended cost: ~$30-50.

**Publication potential**: First multi-agent topology benchmark. Publishable to arXiv + workshop (ICML/NeurIPS agent workshops).

#### 2.2 Large-Scale Ablation (500+ tasks)

**Problem**: Current ablation (6 configs × 20 tasks = 120 data points). Not statistically significant.

**Solution**: Scale to 6 configs × 100+ tasks = 600+ data points.

**Configs** (unchanged from existing):
1. `full` — all pillars active
2. `baseline` — raw LLM, zero SAGE
3. `no-memory` — disable all 4 memory tiers
4. `no-avr` — disable Act-Verify-Refine
5. `no-routing` — force S2 for everything
6. `no-guardrails` — disable all 3 guardrail layers

**Task pool**: EvalPlus HumanEval+ (164) + MBPP+ subset (200) + reasoning tasks (136 from TopologyBench) = 500 tasks.

**Statistical analysis**:
- 95% confidence intervals (bootstrap, 10K resamples)
- Effect sizes (Cohen's d) for each pillar
- Paired McNemar test for pass/fail significance
- Target: p < 0.01 for each pillar vs baseline

**Expected output**:

| Pillar Disabled | Δ pass_rate | Effect Size | Significance |
|---|---|---|---|
| Memory | -X pp | d=Y | p < 0.01 |
| AVR | -X pp | d=Y | p < 0.01 |
| Routing | -X pp | d=Y | p < 0.01 |
| Guardrails | -X pp | d=Y | p < 0.05 |

**Cost estimate** (corrected — configs with AVR/memory/routing add calls):

| Config | Avg calls/task | Notes |
|---|---|---|
| full | 2.5 | Routing + AVR retries |
| baseline | 1 | Raw LLM, single call |
| no-memory | 2 | Routing + AVR but no memory injection |
| no-avr | 1.5 | Routing but no verify/refine loop |
| no-routing | 2 | Force S2 + AVR |
| no-guardrails | 2.5 | All pillars except guardrails |

**Total**: 500 tasks × 6 configs × ~2.0 avg calls = **~6000 LLM calls** (not 3000).
Budget at blended $0.003/call: ~$18. Realistic total with retries/failures: **~$25-40**.

**Implementation**: extend `sage-python/src/sage/bench/ablation.py` with bootstrap CI and McNemar tests. New `--scale full` flag for 500-task runs.

#### 2.3 kNN Exemplar Expansion

**Problem**: 50 GT tasks for kNN = fragile. LOO-CV drops from 92% to 80%.

**Solution**: Use TopologyBench results to label 200+ tasks empirically.

- Each TopologyBench task receives a **proposed** S1/S2/S3 label based on observed performance (empirical: if it passes with Sequential/S1 alone → S1, needs AVR → S2, needs formal verification → S3)
- **Human review gate** (anti-circularity): Auto-labels are proposals, not ground truth. The labeling pipeline outputs a review CSV (`docs/benchmarks/proposed_labels.csv`) with columns: `task_id, task_text, auto_label, confidence, evidence`. Labels with confidence < 0.8 or ambiguous evidence are flagged for manual review. Only human-approved labels enter the exemplar set.
- **Circularity risk**: If we label based on which topology succeeds, we're measuring system behavior, not task intrinsic difficulty. Mitigation: the 50 original human-labeled GT tasks serve as anchor — new auto-labels must not contradict them. Any task where auto-label disagrees with kNN prediction on the original 50 GT set is flagged for review.
- 200 labeled tasks (after human review) → rebuild `config/routing_exemplars.npz`
- kNN accuracy expected: 95%+ (vs 92% on 50) — evaluated on original 50 GT held-out
- Also feeds DeBERTa fine-tuning data (Phase 1.2)

---

### Phase 3: DX Parity (~1.5 sprints)

Selective adoption of competitor features that fit SAGE's architecture.

#### 3.1 Per-Agent Memory Scoping

**Inspiration**: CrewAI `memory.scope("/agent/researcher")`.

**Integration** (additive — zero breaking changes):

| Tier | Current | After |
|---|---|---|
| T0 — Working Memory | Already per-agent (each AgentLoop has its own Arrow buffer) | Unchanged |
| T1 — Episodic | Global SQLite table | Add `agent_id` column, optional filter on queries |
| T2 — Semantic | Global entity-relation graph | Add ownership on edges, scoped queries |
| T3 — ExoCortex | Global shared RAG | Stays global — knowledge, not context |

**API**:
```python
# Explicit scope (new)
agent_memory = memory.scope(agent_id="researcher")
results = agent_memory.search("neural architecture")  # T1+T2 filtered

# Shared access (default, backward-compatible)
results = memory.search("neural architecture")  # everything, as today
```

**Files**:
- `sage-python/src/sage/memory/episodic.py` — add `agent_id` parameter to `store()` and `search()`
- `sage-python/src/sage/memory/semantic.py` — add `agent_id` on entity extraction, filter on `get_context_for()`
- `sage-python/src/sage/memory/scoped.py` — new `ScopedMemory` wrapper class

**Impact**: zero breaking change. Scope is optional. Existing agents use global memory as before.

#### 3.2 Durable Execution (checkpoint/resume)

**Inspiration**: LangGraph persistence — automatic resume after crash on long-running workflows.

**Problem**: If an AgentLoop crashes mid-task (S3, multi-step), everything is lost.

**Solution**: Checkpoint at phase boundaries.

**Checkpoint points**: After each phase in the PERCEIVE → THINK → ACT → LEARN cycle.

**Storage**: SQLite (already have aiosqlite for episodic/semantic).

```sql
CREATE TABLE checkpoints (
    task_id   TEXT PRIMARY KEY,
    phase     TEXT,     -- 'perceive', 'think', 'act', 'learn'
    state     BLOB,     -- msgpack serialized state dict
    timestamp REAL,
    agent_id  TEXT
);
```

**Resume API**:
```python
system = await boot_agent_system()
result = await system.run("complex task", resume=True)
# Auto-detects incomplete task_id, resumes from last checkpoint
```

**Checkpoint state boundaries** (what is serialized vs reconstructed):

| State | Serialized? | Format | Notes |
|---|---|---|---|
| Task text + task_id | Yes | msgpack string | Immutable input |
| Current phase (P/T/A/L) | Yes | msgpack enum | Resume point |
| LLM response(s) so far | Yes | msgpack string list | THINK/ACT outputs |
| AVR iteration count | Yes | msgpack int | Resume mid-AVR |
| Routing decision | Yes | msgpack dict | S1/S2/S3 + model_id |
| Memory injections | Yes | msgpack string list | Avoid re-querying S-MMU |
| Guardrail results | Yes | msgpack dict | Skip re-evaluation |
| Working Memory (Arrow) | **No — reconstructed** | — | Re-read from S-MMU on resume |
| LLM provider handles | **No — reconstructed** | — | Re-initialized at boot |
| EventBus subscriptions | **No — reconstructed** | — | Stateless, re-subscribe |
| Sandbox state | **No — not preserved** | — | Tools re-execute from scratch |

**Design choice**: Only serialize what's needed to skip completed phases. Anything that involves live connections or Rust FFI handles is reconstructed from boot.

**Scope limitation**: Local single-process only. NOT distributed durable execution (LangGraph Cloud). The checkpoint is a safety net, not an orchestration feature.

**Files**:
- `sage-python/src/sage/checkpointing.py` — new module: `Checkpointer` class with `save()`, `load()`, `cleanup()`
- `sage-python/src/sage/agent_loop.py` — checkpoint calls at phase boundaries
- Uses `msgpack` for serialization (faster than pickle, safer)

#### 3.3 Python-Only Mode

**Problem**: Full SAGE requires Rust compilation. Barrier to entry.

**Solution**: SAGE starts and functions without `sage_core`, with graceful degradation.

**Already working** (from audit sprint):
- `StructuralFeatures` → Python ✓
- `ModelCard` / `CognitiveSystem` → Python ✓
- `ModelRegistry` → Python ✓
- Working Memory → Python mock with warning ✓
- Embedder → sentence-transformers or hash fallback ✓
- `SmtVerifier` → z3-solver Python fallback ✓

**Remaining gaps to cover** (1823 LOC Rust → Python):

| Rust Module | LOC | Python Port | Effort | Complexity |
|---|---|---|---|---|
| `TopologyGraph` (topology_graph.rs) | 687 | `networkx.DiGraph` wrapper with typed nodes/edges + three-flow edge model | 1 sprint | Medium — petgraph API → networkx, typed nodes/edges |
| `HybridVerifier` (verifier.rs) | 660 | 6 structural checks (Python graph traversal) + 4 semantic checks (string matching) | 0.5 sprint | Low — checks are simple graph algorithms, no SIMD |
| `TopologyExecutor` (executor.rs) | 476 | Kahn's toposort (stdlib `graphlib.TopologicalSorter`) + gate-based readiness (Python dict + deque) | 0.5 sprint | Medium — dual-mode scheduling logic |
| Templates (templates.rs) | 661 | 8 template constructors returning networkx graphs | 0.5 sprint | Low-Medium — pure graph construction, but 8 templates × validation |

**Total**: 2484 LOC Rust → ~1000-1200 LOC Python (Rust is more verbose for graph code). **Revised effort: 2 sprints** (accounting for test parity, LTL integration, three-flow edge model, and edge cases in dynamic executor mode). This is the largest single item in Phase 3.

**What stays Rust-only**:
- Wasm WASI sandbox (wasmtime) — Python-only mode uses subprocess with timeout
- eBPF executor (solana_rbpf) — no Python equivalent needed
- SIMD/AVX-512 memory operations — Python uses PyArrow (slower but functional)
- ONNX embedding via `ort` — Python uses `onnxruntime` pip package directly

**Installation**:
```bash
pip install sage-python              # Python-only, works immediately
pip install sage-python[rust]        # with sage_core for performance
pip install sage-python[rust,onnx]   # full performance + ONNX models
```

**Performance tradeoff**: ~3-5x slower on embeddings, ~10x slower on memory operations, no Wasm sandbox. Functional for development and evaluation; Rust recommended for production.

---

## Phase Summary

| Phase | Components | Effort | Key Metric | SOTA Impact |
|---|---|---|---|---|
| **1 — Routing Dominance** | Learned QualityEstimator, DeBERTa Stage 1, sage-router lib | ~2 sprints | Routing 92% → 96-98% | Surpasses RouteLLM, approaches NVIDIA 98.1% |
| **2 — TopologyBench** | 200-task benchmark, 500-task ablation, kNN expansion | ~2 sprints | First topology benchmark published | Novel contribution, no competitor equivalent |
| **3 — DX Parity** | Memory scoping, durable execution, Python-only mode | ~2 sprints | `pip install` → working in 5 min | Closes gap with LangGraph/CrewAI |
| **Total** | 9 components | ~6 sprints | | |

## Constraints

- **No solution removal**: All existing technical solutions are preserved. New components ADD or REPLACE with strictly better alternatives.
- **Backward compatibility**: All new APIs are additive. Existing code continues to work unchanged.
- **Evidence-first**: No claim without benchmark proof. TopologyBench and large-scale ablation provide the evidence.
- **Dual-import shadow period**: New ONNX models (QualityEstimator, DeBERTa) run in shadow mode alongside existing heuristics until parity is proven.

## Dependencies

- Phase 1.2 (DeBERTa) depends on NVIDIA model availability and ONNX export compatibility
- Phase 2.1 (TopologyBench) requires API credits for ~9000 LLM calls (multi-agent topologies, see cost estimate above), ~$30-50
- Phase 2.2 (large ablation) requires API credits for ~6000 LLM calls (multi-step configs), ~$25-40
- Phase 1.3 (sage-router) depends on Phases 1.1 + 1.2 for maximum value but can ship with kNN-only first

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| DeBERTa zero-shot <95% on SAGE GT | Fine-tune with augmented data; kNN stays as fallback |
| TopologyBench shows topology doesn't matter | Publish negative result (equally valuable); focus on routing |
| QualityEstimator training data too small | Synthetic augmentation; collect from community benchmark runs |
| sage-router adoption low | Library still used internally; no wasted effort |
| Python-only mode too slow for production | Clear docs: "development mode"; Rust recommended for production |
| Ablation/TopologyBench task overlap | Run ablation independently — no parameter tuning between runs |

## Implementation Notes

- **New dependency**: `msgpack` must be added to `pyproject.toml` for checkpoint serialization (Phase 3.2)
- **sage-router import migration**: Use transitional re-export (`sage.strategy.metacognition` re-exports from `sage_router.types`) so downstream code does not break during extraction
