# Research Foundations

YGN-SAGE draws on and extends research from multi-agent systems, LLM routing, evolutionary computation, and formal verification. This page maps key papers to their SAGE implementations.

---

## Routing and Model Selection

### kNN Embedding-Based Routing
**arXiv 2505.12601** -- Demonstrates that kNN on embeddings outperforms MLP, GNN, and attention-based routers for LLM task classification.

**SAGE implementation**: `sage.strategy.knn_router.KnnRouter` uses pre-computed exemplar embeddings from snowflake-arctic-embed-m (768-dim). Distance-weighted majority vote achieves **92% accuracy** on 50 human-labeled ground truth tasks versus 52% for keyword heuristics. Integrated as Stage 0.5 in the `AdaptiveRouter` pipeline.

---

### LLMRouterBench
**arXiv 2601.07206** -- Comprehensive benchmark of LLM routing methods. Key finding: embedding backbone impact is limited, and many routing methods converge to similar performance.

**SAGE takeaway**: Validated that the choice of routing *algorithm* matters less than having a robust multi-stage fallback pipeline. SAGE's 4-stage pipeline with cascade fallback ensures graceful degradation.

---

### PILOT: Contextual Bandit LLM Routing
**arXiv 2508.21141** -- Contextual bandit approach to LLM routing with budget constraints.

**SAGE implementation**: `sage_core.ContextualBandit` uses per-arm Beta/Gamma posteriors with Thompson sampling and Pareto front selection. Budget-aware routing integrated into `SystemRouter.record_outcome()`.

---

### LinUCB Multi-LLM Selection
**arXiv 2506.17670** -- LinUCB-based approach to multi-LLM selection.

**SAGE implementation**: The contextual bandit in `sage_core` uses a similar LinUCB-inspired approach but extends it with per-arm posteriors rather than global linear models, and adds Pareto front optimization for cost-quality tradeoffs.

---

### RouteLLM
**ICLR 2025, arXiv 2406.18665** -- BERT 0.3B router trained on Chatbot Arena preference data.

**SAGE context**: Alternative approach to the BERT ONNX classifier in Stage 1 of the AdaptiveRouter. RouteLLM demonstrates that preference data can train effective routers, but SAGE's kNN approach achieves 92% accuracy without requiring preference data collection.

---

### Cascade Routing (ETH-SRI)
**ICLR 2025** -- Key finding: quality estimators are the bottleneck in cascade routing, not routing algorithms.

**SAGE response**: Led to the development of the DistilBERT QualityEstimator (ONNX, 0.9 MB), which achieves +34.4pp Pearson correlation improvement over the heuristic baseline. Validates the research finding that investing in quality estimation pays off more than routing algorithm sophistication.

---

### LLM Routing Survey
**arXiv 2603.04445** -- Comprehensive survey identifying 6 routing paradigms.

**SAGE validation**: SAGE's AdaptiveRouter architecture (multi-stage cascade with learned components at each stage) is confirmed as a SOTA architecture pattern by this survey.

---

### NVIDIA DeBERTa Classifier
**HuggingFace: nvidia/prompt-task-and-complexity-classifier** -- DeBERTa-v3-base multi-head classification with 98.1% accuracy on NVIDIA's dataset.

**SAGE evaluation**: Zero-shot evaluation on SAGE's 50 GT tasks yields only 52% accuracy with S3=0%. Fine-tuning required for SAGE's cognitive system taxonomy. This is a future Stage 1 candidate.

---

## Topology and Multi-Agent Systems

### AdaptOrch
**arXiv 2602.16873** -- Demonstrates that topology structure impacts agent performance more than model capability.

**SAGE implementation**: The entire Topology pillar is built on this hypothesis. `DynamicTopologyEngine` with 6-path generation, `MapElitesArchive` for quality-diversity search, and `TopologyBench` for empirical validation. TopologyBench results show a 4.3pp spread across 9 topologies (92.1% -- 96.3%), though confidence intervals overlap at 164 tasks.

---

### MASFactory
**arXiv 2603.06007** -- Vibe Graphing approach: LLM-to-graph conversion with three-flow edges, achieving 84.76% HumanEval accuracy with 97% code reduction.

**SAGE implementation**: The `TopologyGraph` IR uses the three-flow edge model (Control, Message, State) directly from MASFactory. `TopologySynthesizer` implements the 3-stage LLM pipeline (Role Assignment -> Structure Design -> Validation). This is Path 3 in the `DynamicTopologyEngine`.

---

### OFA-MAS
**arXiv 2601.12996** (WWW '26) -- MoE graph generative model for universal multi-agent system topology.

**SAGE context**: Informs the design of the topology generation pipeline, particularly the use of graph-based representations for agent topologies.

---

### Topology Structure Learning
**arXiv 2505.22467** -- Demonstrates up to 10% performance gap between different agent topologies.

**SAGE validation**: TopologyBench confirms a 4.3pp spread across topologies, consistent with this paper's findings at a smaller scale.

---

## Evolutionary Computation

### MAP-Elites Quality-Diversity

MAP-Elites is a quality-diversity algorithm that maintains a grid of solutions indexed by behavioral descriptors, ensuring diversity rather than convergence to a single optimum.

**SAGE implementation**: `sage_core.MapElitesArchive` with 4-dimensional BehaviorDescriptor spanning 108 cells. Pareto dominance for replacement. SQLite persistence. Integrated into `DynamicTopologyEngine.evolve()`.

---

### CMA-ME (Covariance Matrix Adaptation MAP-Elites)

CMA-ME extends MAP-Elites with directional search using covariance matrix adaptation on continuous parameters.

**SAGE implementation**: `sage_core.CmaEmitter` with diagonal covariance and elite-weighted mean/variance updates. 50% random / 50% CMA-sampled mutations in the evolution loop.

---

### Monte Carlo Tree Search (MCTS)

UCB1-based tree search applied to topology space exploration.

**SAGE implementation**: `sage_core.MctsSearcher` with random mutation expansion, HybridVerifier-based rollout heuristic. Budget: 50 simulations or 100ms. 6th path in `DynamicTopologyEngine.generate()`.

---

## Formal Verification

### OxiZ (Pure-Rust SMT Solver)

Pure-Rust SMT solver preferred over z3-sys for zero C++ dependencies.

**SAGE implementation**: `sage_core.SmtVerifier` with QF_LIA integer solving, recursive descent expression parser, and 10 PyO3 methods. ALL Python callers fully wired to Rust -- zero Z3-only code paths remain. Sub-0.1ms verification latency (0.024ms PRM, 0.060ms mutation validation).

---

### CEGAR (Counterexample-Guided Abstraction Refinement)

Iterative refinement of invariants using counterexamples.

**SAGE implementation**: `SmtVerifier.synthesize_invariant()` iteratively weakens/strengthens post-conditions over up to 5 rounds. `verify_invariant_with_feedback()` returns clause-level diagnostic violations wired into S3 escalation prompts.

---

### LTL Model Checking

Linear Temporal Logic verification on graph structures.

**SAGE implementation**: `sage_core.LtlVerifier` with 4 checks on TopologyGraph: reachability (BFS), safety (no HIGH->LOW paths), liveness (all entries reach exits), bounded liveness (depth <= K). Wired into HybridVerifier (safety produces errors, liveness produces warnings).

---

## Original Contributions

SAGE introduces several approaches not found in the referenced literature:

1. **6-path topology generation**: Combining S-MMU recall, archive lookup, LLM synthesis, mutation, MCTS, and template fallback in a single prioritized pipeline

2. **4-stage adaptive routing**: Structural features -> kNN embeddings -> BERT ONNX -> entropy probe (cascade fallback; stage 3 online learning reserved) -- with each stage able to short-circuit

3. **Three-flow topology IR**: Extending MASFactory's concept with a petgraph-backed unified IR that serves both as topology graph and contract verification target

4. **Dual-mode topology execution**: Static (Kahn's toposort) for acyclic and dynamic (gate-based readiness) for cyclic topologies in a single executor

5. **S-MMU for routing**: Storing routing decisions as S-MMU chunks and retrieving similar prior decisions for context injection

6. **CRAG-gated memory injection**: Using keyword overlap scoring to prevent irrelevant context pollution from the memory system

7. **CircuitBreaker resilience**: Per-subsystem failure isolation ensuring graceful degradation rather than cascading failures
