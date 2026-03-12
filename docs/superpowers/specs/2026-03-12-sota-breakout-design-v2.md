# YGN-SAGE: SOTA Breakout Design Spec — v2 (Post-Audit)

**Date**: 2026-03-12
**Status**: Draft (post-audit revision)
**Author**: Claude Opus 4.6 + Yann Abadie
**Objective**: Fix three critical gaps exposed by independent audit, then push SAGE beyond SOTA

## Audit Findings — Honest Assessment

An independent audit rated the v1 spec 6/10 and exposed three false claims.
Every finding was **confirmed by deep code review**:

### Finding 1: Topology Execution is Prompt Decoration

**Claim (v1)**: "Topology evolution: MAP-Elites + CMA-ME + MCTS"
**Reality**: `agent_loop.py:442-452` converts topology schedule into a TEXT STRING:
```python
topology_hint = (
    f"[Topology: {len(topology_schedule)} agents — {roles_desc}. "
    f"You are acting as: {topology_schedule[0]['role']}]"
)
messages.insert(0, Message(role=Role.SYSTEM, content=topology_hint))
```
Then line 461: **single `self._llm.generate()` call**. No multi-agent dispatch.

**What exists but isn't connected**:
- `TopologyGraph` (Rust) — three-flow edges (Control, Message, State), typed nodes
- `TopologyExecutor` (Rust) — readiness-based scheduling (dynamic mode)
- `agents/sequential.py`, `agents/parallel.py`, `agents/loop_agent.py` — actual multi-agent primitives
- `AgentTool` — wraps any agent as a tool
- `CognitiveOrchestrator:299-347` — DOES real multi-agent for S3 (SequentialAgent/ParallelAgent)

**Verdict**: All pieces exist in isolation. The wiring is missing. MASFactory (2603.06007) provides the exact execution model already matching SAGE's IR.

### Finding 2: FrugalGPT is Exception Retry, Not Quality Cascade

**Claim (v1)**: "7 providers, FrugalGPT"
**Reality**: `orchestrator.py:96-113` ModelAgent catches `Exception`, tries next model. No quality check.
`QualityEstimator` is NEVER called during routing — only for S-MMU feedback (`boot.py:381`) and benchmark reporting (`eval_protocol.py:786`).

**What exists but isn't connected**:
- `QualityEstimator` — 5-signal scoring (non-empty, length, code presence, error absence, AVR convergence)
- `ModelRegistry.select()` — weighted model selection with telemetry
- `DriftMonitor` — latency/error/cost drift detection

**Verdict**: Real FrugalGPT requires: cheap model → quality check → escalate if quality < threshold. One function to write.

### Finding 3: S3 Gating is Soft (Accept After Retries)

**Claim (v1)**: "Formal verification (OxiZ/Z3, 10 SMT methods)"
**Reality**: `agent_loop.py:503-524` — PRM validation retries twice, then:
```python
# Max retries reached -- accept response as-is
log.warning("S3 retry limit reached, accepting response without <think> tags.")
```
The verification IS wired (OxiZ → `kg_rlvr.py` → `verify_step()`), but the gating is soft.

**Verdict**: The machinery works. The gate needs to be hard: reject + CEGAR repair instead of accept-anyway.

### Dead Code Confirmed

- `_last_avr_iterations` — READ via `getattr(..., 0)` in `boot.py:385`, **NEVER SET**. QualityEstimator Signal 5 always returns 0.
- `AdaptiveRouter` Stage 3 — "Reserved for cascade/online learning" — **ZERO code** implements it.

---

## Corrected Competitive Analysis

| Capability | SAGE (actual, today) | SAGE (after Phase 0) | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|---|
| Cost-aware routing | kNN 92%, 3 tiers | kNN 92%+ quality-gated cascade | LLM structured output | None | None |
| Multi-provider fallback | 7 providers, **exception-retry** | 7 providers, **quality-cascade** | Single model | Single model | Single model |
| Topology execution | **Prompt hint only** (single LLM call) | **Real multi-agent dispatch** (TopologyExecutor → agents) | StateGraph (real) | Sequential/Hierarchical (real) | TypeSubscription (real) |
| Topology evolution | MAP-Elites + CMA-ME + MCTS (generates graph, **not executed**) | MAP-Elites + CMA-ME + MCTS (generates **and executes**) | None | None | None |
| Formal verification | OxiZ/Z3, 10 methods, **soft gating** | OxiZ/Z3, 10 methods, **hard gating + CEGAR repair** | None | None | None |
| Memory | 4 tiers (all wired) | 4 tiers (unchanged) | State checkpointing | Scoped memory | Message history |
| Sandbox | Wasm WASI + tree-sitter + eBPF | Unchanged | None native | None native | Basic executor |
| Guardrails | 3-layer (input/runtime/output) | Unchanged | None native | Per-task | Intervention |
| Durable execution | None | None (Phase 3) | Yes | No | No |

---

## Target Persona (unchanged)

**Primary: "L'Architecte Agent" — Senior AI/ML Engineer, 5-10 ans XP**

> *"Je construis des systèmes multi-agents en production. J'ai besoin de contrôler les coûts, de prouver la fiabilité, et de ne pas être verrouillé sur un seul provider."*

| Dimension | Profil |
|---|---|
| **Rôle** | Senior ML Engineer / AI Platform Engineer dans une scale-up ou un labo R&D industriel |
| **Douleur #1 — Coût** | Facture LLM >$5K/mois. Routing intelligent = économie directe |
| **Douleur #2 — Fiabilité** | Hallucinations en prod. Vérification formelle = Sev2 évités |
| **Douleur #3 — Lock-in** | Prisonnier d'un provider. Multi-provider avec failover = liberté |
| **Douleur #4 — Topologie figée** | Sequential partout. Topologie adaptative = meilleure perf/tâche |
| **Ce qui le convainc** | Benchmarks chiffrés, papiers cités, résultats reproductibles |
| **Ce qui le bloque** | Setup >30 min, docs manquante, breaking changes |
| **Parcours** | `sage-router` (pip, 5 min) → gain coût → SAGE complet |

**Secondaire**: "Le Chercheur MAS" — PhD/Post-doc, publie sur les architectures multi-agents.
**Tertiaire**: "L'Intégrateur DevTools" — Tech Lead, veut juste le routing standalone.

**Implication design**: Chaque feature répond à ≥1 douleur de l'Architecte Agent. Sinon = scope creep.

---

## Revised Design

### Phase 0: Wire What Exists (~1.5 sprints)

**Priority: Connect the pieces that are already built but disconnected.**
This phase transforms SAGE from "impressive components, no integration" to "working multi-agent system."

Every item addresses a specific audit finding with code already in the codebase.

#### 0.1 Real Topology Execution (Audit Finding 1)

**Problem**: `agent_loop.py:442-452` converts TopologyGraph to a text string. Single LLM call.

**Solution**: Replace prompt-hint code with actual multi-agent dispatch.

**Architecture** (MASFactory-aligned — 2603.06007):

```
TopologyGraph (Rust IR, already exists)
    ↓
TopologyExecutor.next_ready() (Rust readiness scheduler, already exists)
    ↓
TopologyRunner (NEW Python class)
    ↓ per ready node
AgentLoop / ModelAgent (existing execution primitives)
    ↓ outputs
Message passing via node state dict (NEW, simple dict per node)
    ↓
TopologyExecutor.mark_completed() (already exists)
    ↓ loop until is_done()
Final aggregation → return result
```

**Key design decisions**:

1. **One `AgentLoop` per topology node** (not one LLM call). Each node gets its own system prompt derived from `node.role` + `node.capabilities`. The node's `model_id` selects the LLM tier.

2. **Message passing = dict per node**. When TopologyExecutor yields a ready node, the runner collects outputs from predecessor nodes (following Message-flow edges) and injects them into the node's prompt. This implements MASFactory's "aggregate inputs → execute → dispatch outputs" lifecycle without a framework rewrite.

3. **Concurrency follows edge types**:
   - **Control edges**: sequential execution order (TopologyExecutor already handles this)
   - **Message edges**: content passes between nodes
   - **State edges**: shared mutable state (Python dict, thread-safe via asyncio — single-threaded event loop)

4. **Fallback**: If TopologyExecutor unavailable or topology has 1 node → current behavior (single LLM call). Zero breaking change.

**Changes**:
- **New file**: `sage-python/src/sage/topology/runner.py` (~200 LOC)
  - `TopologyRunner` class: takes TopologyGraph + config, yields results
  - `async run(task: str) -> str`: main execution loop
  - Per-node: creates temporary `AgentLoop` or `ModelAgent` with node-specific config
  - Aggregation: last node's output (Sequential) or synthesized (Parallel/Hub)
- **Modify**: `sage-python/src/sage/agent_loop.py:442-452`
  - Replace prompt-hint injection with `TopologyRunner.run()` delegation
  - When topology has >1 node: delegate to `TopologyRunner`, return its result
  - When topology has 1 node: current single-LLM behavior (unchanged)
- **Test**: `sage-python/tests/test_topology_runner.py`
  - Test with 2-node Sequential topology
  - Test with 3-node Parallel topology (2 workers + aggregator)
  - Test with AVR topology (act → verify → refine)
  - Test fallback to single-LLM when no topology

**What this unlocks**: TopologyBench (Phase 2) becomes meaningful — it measures actual multi-agent execution, not prompt decoration.

**Persona impact**: Douleur #4 (topologie figée) — SAGE becomes the ONLY framework where topology is auto-generated AND executed.

#### 0.2 Real Quality-Gated Cascade (Audit Finding 2)

**Problem**: ModelAgent retry is exception-only. QualityEstimator disconnected from routing.

**Solution**: Wire QualityEstimator into the cascade loop.

**Current flow** (exception-only):
```
ModelAgent.run(task) → try LLM call → Exception? → try next model → Exception? → try next → give up
```

**New flow** (quality-gated):
```
ModelAgent.run(task) → try cheap model → QualityEstimator.estimate() → quality < threshold?
    → yes: try better model → QualityEstimator.estimate() → quality < threshold?
        → yes: try best model → return
    → no: return (cheap model was good enough!)
```

**Implementation**:
- **Modify**: `sage-python/src/sage/orchestrator.py` — `ModelAgent` class
  - Add `quality_threshold: float = 0.6` parameter
  - Add `quality_estimator: QualityEstimator | None = None` parameter
  - After successful LLM call (no exception), run `QualityEstimator.estimate(task, result)`
  - If quality < threshold AND cascade attempts remain → try next-better model
  - If quality >= threshold → return immediately (cost saved!)
  - Keep exception-based fallback as the inner try/except (defense in depth)
- **Modify**: `sage-python/src/sage/orchestrator.py` — `CognitiveOrchestrator`
  - Pass `QualityEstimator` instance to `ModelAgent` constructor
  - S1 tasks: `quality_threshold=0.4` (low bar, cheap is usually fine)
  - S2 tasks: `quality_threshold=0.6` (moderate quality required)
  - S3 tasks: `quality_threshold=0.8` (high quality, willing to pay)
- **Fix dead code**: `boot.py:385` — wire `_last_avr_iterations` properly
  - In `agent_loop.py`, after AVR loop completes, SET `self._last_avr_iterations = self._s2_avr_retries`
  - This makes QualityEstimator Signal 5 (AVR convergence) actually fire

**Cost impact for L'Architecte Agent**: Real FrugalGPT. S1 tasks stop at cheapest model when quality is sufficient. Measurable $/task reduction.

**Test**:
- `test_quality_cascade.py` — mock 3 models: model_A returns "short bad answer" (quality 0.3), model_B returns "good answer" (quality 0.7). Verify cascade stops at model_B.
- `test_quality_cascade_cheap_sufficient.py` — model_A returns good answer (quality 0.7). Verify NO cascade (cost saved).

#### 0.3 Hard S3 Gating with CEGAR Repair (Audit Finding 3)

**Problem**: `agent_loop.py:523-524` accepts response after max retries even when PRM fails.

**Solution**: Replace "accept anyway" with CEGAR repair loop.

**Current flow**:
```
PRM.calculate_r_path(content) → r_path < 0? → retry (max 2) → ACCEPT ANYWAY
```

**New flow**:
```
PRM.calculate_r_path(content) → r_path < 0?
    → retry with feedback (max 2)
    → still failing? → CEGAR repair:
        1. Extract counterexample from PRM details
        2. Inject counterexample + clause-level feedback into prompt
        3. Request targeted fix (not full regeneration)
        4. Re-verify
    → still failing after CEGAR? → DEGRADE to S2 (not accept bad S3)
```

**Implementation**:
- **Modify**: `sage-python/src/sage/agent_loop.py:503-526`
  - Replace `log.warning("S3 retry limit reached, accepting response...")` with CEGAR repair
  - Use existing `verify_invariant_with_feedback()` (already returns clause-level diagnostics)
  - After CEGAR: if still failing → degrade to S2 AVR (retry without formal requirements)
  - Emit `GUARDRAIL_BLOCK` event when S3 degrades (observability for L'Architecte)
- **New method**: `agent_loop._cegar_repair(content, details) -> str`
  - Extract failed clauses from PRM details
  - Build targeted repair prompt: "Your formal assertions failed: {clauses}. Fix specifically: {feedback}"
  - Single LLM call with repair context
  - Re-run PRM on repaired content
  - Return repaired content or None (triggers S2 degradation)

**Persona impact**: Douleur #2 (fiabilité) — S3 tasks either pass formal verification or explicitly degrade with logging. No silent acceptance of unverified output.

**Test**:
- `test_s3_hard_gate.py` — mock PRM that always fails → verify degradation to S2, not acceptance
- `test_s3_cegar_repair.py` — mock PRM that fails first, passes after repair → verify repair path works

#### 0.4 Dead Code Cleanup

Quick fixes, no architecture change:

| Item | Fix | LOC |
|---|---|---|
| `_last_avr_iterations` never set | Set in `agent_loop.py` after AVR loop | 1 line |
| `AdaptiveRouter` Stage 3 "Reserved" | Remove placeholder comment, document as "future: online learning" in docstring | 3 lines |
| `boot.py:385` getattr for avr_iterations | Now works because 0.4 fixes the setter | 0 lines |

---

### Phase 1: Routing Dominance (~2 sprints)

**Prerequisite**: Phase 0 complete (quality cascade wired, topology executing).

#### 1.1 Learned QualityEstimator (unchanged from v1)

Fine-tune DistilBERT on `(task, response, quality_score)` triples from benchmark runs.
- 2000+ triples from HumanEval+ (164) + MBPP+ (378) + augmentation
- Honest scope: +3-8% quality correlation at 2000 triples (not +5-15%)
- ONNX export for Rust inference via `ort`
- Replaces heuristic `QualityEstimator.estimate()` when ONNX model available
- **New importance after Phase 0**: This model powers the quality-gated cascade. Better QualityEstimator = better cascade routing = more cost savings for L'Architecte.

#### 1.2 DeBERTa-v3-base Task Classifier (unchanged from v1)

- Zero-shot evaluation of NVIDIA `prompt-task-and-complexity-classifier` on SAGE 50 GT
- Ship criterion: bootstrap 95% CI lower bound >= 90%
- Fine-tune if needed with 500+ augmented GT tasks
- Wire as AdaptiveRouter Stage 1 (after kNN Stage 0.5)
- Shadow comparison: both run in parallel, JSONL traces

#### 1.3 `sage-router` Standalone Library (unchanged from v1)

Extract routing pipeline as standalone pip-installable library.
- `pip install sage-router` — zero Rust dependency
- `pip install sage-router[onnx]` — with DeBERTa + kNN ONNX acceleration
- Adoption wedge for L'Intégrateur DevTools persona
- SAGE itself depends on `sage-router` (dogfooding)

**Auditor validated**: "Best idea in the entire project."

---

### Phase 2: TopologyBench + Scale Proof (~2 sprints)

**Prerequisite**: Phase 0.1 complete (topology execution is real).

**Critical change from v1**: TopologyBench is now meaningful because it measures actual multi-agent execution through TopologyRunner, not prompt decoration.

#### 2.1 TopologyBench (unchanged structure, corrected semantics)

200 tasks × 10 topologies. Each topology now runs through `TopologyRunner` with actual per-node LLM calls.

**What changed**: "Evolved (MAP-Elites/MCTS)" topologies are no longer decorative graphs — they produce actual agent ensembles via `TopologyRunner`.

Cost estimate unchanged: ~9000 LLM calls, ~$30-50 blended.

#### 2.2 Large-Scale Ablation (unchanged from v1)

6 configs × 500+ tasks. Bootstrap CI, McNemar test, Cohen's d.

#### 2.3 kNN Exemplar Expansion (unchanged from v1)

50 GT → 200+ tasks with human review gate. Anti-circularity via anchor set.

---

### Phase 3: DX Parity (~1.5 sprints)

Unchanged from v1:
- **3.1**: Per-agent memory scoping (add `agent_id` to episodic/semantic)
- **3.2**: Durable execution (checkpoint at phase boundaries, local SQLite)
- **3.3**: Python-only mode (topology ports to networkx, already partially done)

---

## Phase Summary (revised)

| Phase | Focus | Effort | Key Deliverable | Audit Finding Addressed |
|---|---|---|---|---|
| **0 — Wire What Exists** | Connect disconnected components | ~1.5 sprints | Real multi-agent execution, quality cascade, hard S3 | ALL THREE findings |
| **1 — Routing Dominance** | Improve routing accuracy + extract sage-router | ~2 sprints | 92% → 96-98% routing, sage-router on PyPI | Strengthens existing strength |
| **2 — TopologyBench** | Prove topology value with evidence | ~2 sprints | First topology benchmark (now meaningful) | Validates Phase 0 |
| **3 — DX Parity** | Reduce setup friction | ~1.5 sprints | `pip install` → 5 min working | Adoption for L'Architecte |
| **Total** | | ~7 sprints | | |

---

## Key Research References

| Paper | Relevance | Usage in SAGE |
|---|---|---|
| **MASFactory (2603.06007)** | Three-flow execution model, node lifecycle, readiness scheduling | Phase 0.1: TopologyRunner architecture matches MASFactory exactly |
| **ETH-SRI Cascade (ICLR 2025)** | Quality estimators are the bottleneck | Phase 0.2 + 1.1: QualityEstimator powers cascade routing |
| **arXiv 2505.12601** | kNN beats MLP/GNN/attention routers | Validated: SAGE kNN at 92% |
| **AdaptOrch (2602.16873)** | Topology > model capability | Phase 2.1: TopologyBench tests this claim |
| **NVIDIA DeBERTa** | 98.1% multi-head classification | Phase 1.2: target for Stage 1 ONNX |
| **AlphaEvolve (2506.13131)** | Evolution + evaluator feedback | Evolution engine already implemented |
| **PSRO (1711.00832)** | Multi-agent population evolution | MAP-Elites + CMA-ME already implemented |

---

## Constraints

- **No solution removal**: All existing code preserved. Phase 0 ADDS wiring, not rewrites.
- **Backward compatibility**: All new APIs additive. Topology hint fallback for 1-node topologies.
- **Evidence-first**: No claim without benchmark proof.
- **Audit-honest**: Competitive analysis shows actual state, not aspirational state.
- **Persona-driven**: Every feature addresses ≥1 douleur of L'Architecte Agent.

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| TopologyRunner adds latency (multi-LLM calls) | S1 tasks: force 1-node topology (no overhead). Multi-node only for S2/S3 |
| Quality cascade over-escalates (costs more) | Tune thresholds on benchmark data. Start conservative (threshold=0.6) |
| CEGAR repair fails to fix (infinite loop) | Max 1 CEGAR attempt → degrade to S2. Bounded cost |
| TopologyBench shows topology doesn't matter | Publish negative result (equally valuable). Focus on routing |
| DeBERTa zero-shot <90% CI lower bound | Fine-tune with augmented data; kNN stays as fallback |

## Dependencies

- Phase 0: No new dependencies. Uses only existing SAGE components.
- Phase 1.1: `transformers` + `torch` for training (dev dependency only)
- Phase 1.3: New `sage-router/` package with separate `pyproject.toml`
- Phase 2: API credits ~$30-50 for TopologyBench, ~$25-40 for ablation
- Phase 3.2: `msgpack` for checkpoint serialization

## Revision History

- **v1** (2026-03-12): Initial spec — rated 6/10 by independent audit
- **v2** (2026-03-12): Post-audit revision
  - [Critical] Added Phase 0 "Wire What Exists" to fix all three audit findings
  - [Critical] Corrected competitive analysis — honest about current gaps
  - [Critical] TopologyBench moved after Phase 0 (topology must be real first)
  - [Important] Added MASFactory as architectural reference for topology execution
  - [Important] Removed "FrugalGPT" claim — replaced with honest "exception-retry → quality-cascade"
  - [Important] Fixed dead code (_last_avr_iterations, Stage 3 placeholder)
  - Phase structure: 3 phases → 4 phases (0-1-2-3)
  - Effort: ~6 sprints → ~7 sprints (Phase 0 adds 1.5)
