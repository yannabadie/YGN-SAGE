# YGN-SAGE Architecture Decisions

Key design rationales not fully captured in CLAUDE.md or code comments. These record the *why* behind decisions — useful when revisiting trade-offs or onboarding. Consolidated from 45 planning documents. Last updated: 2026-03-12.

---

## Cognitive Systems (S1/S2/S3)

**Decision:** S1/S2/S3 are cognitive systems (Kahneman dual-process theory), NOT pipeline stages.

**Why:** Early design treated them as sequential stages (task enters S1, escalates to S2, then S3). Two expert reviews corrected this — they're parallel processing modes selected by the router. A simple question goes to S1 directly; a formal proof goes to S3 directly. Escalation (S2→S3) happens only when S2 exhausts its AVR budget.

**Implication:** Router decides upfront which system handles the task. No mandatory progression through S1→S2→S3.

---

## Evidence-First Philosophy

**Decision:** Every architectural claim must be backed by empirical evidence before being stated as fact.

**Why:** Three independent audits (March 5-9, 2026) found confabulated claims in README and docs — "SOTA Mandate", "ASI Phase", performance numbers without benchmarks, circular routing tests. The project adopted a strict policy: prove before claiming.

**Implication:**
- Routing benchmark relabeled from "accuracy" to "self-consistency" (circular — labels came from the heuristic itself). Fixed with 50 human-labeled ground truth tasks.
- ExoCortex passive grounding removed after Sprint 3 evidence showed it adds latency without benefit for code tasks. Active `search_exocortex` tool only.
- Evolution engine kept despite marginal measured value — MAP-Elites + CMA-ME provide quality-diversity exploration that may compound over time.

---

## Routing: Hard Constraints First

**Decision:** Routing follows a strict priority: hard constraints → structural scoring → contextual bandit → semantic match (tie-breaker only).

**Why:** Early design let the bandit freely choose models, leading to budget violations and capability mismatches. Hard constraints (budget cap, required capabilities, safety labels) must be non-negotiable — the bandit only optimizes within the feasible set.

**Implication:** `SystemRouter.route_constrained()` filters models by capabilities and budget BEFORE bandit scoring. Semantic matching (domain affinity from ModelCard `domain_scores`) is the last tie-breaker, not a primary signal.

---

## Template-First Topologies

**Decision:** Topology generation follows 6 paths: S-MMU retrieval → MAP-Elites archive → LLM synthesis → mutation → MCTS → template fallback.

**Why:** MASFactory (arXiv 2603.06007) showed LLM-synthesized topologies achieve 84.76% HumanEval, but LLM synthesis is expensive and unreliable. Templates provide a fast, verified baseline. The archive and S-MMU provide learned alternatives. LLM synthesis is path 3 (not path 1) because it requires rate-limited API calls.

**Implication:** 8 templates always available as fallback. MAP-Elites archive grows over time. LLM synthesis reserved for novel task types not in archive.

---

## Three-Flow Edge Model

**Decision:** TopologyGraph edges carry 3 flow types: Control, Message, State.

**Why:** MASFactory analysis showed single-type edges conflate control flow ("A runs before B"), data flow ("A sends field X to B"), and state flow ("parent shares context with child"). This prevents proper verification — you can't check deadlocks without distinguishing control from data edges.

**Implication:** `EdgeType::Control` for sequencing, `EdgeType::Message` for field-level routing, `EdgeType::State` for parent↔child context sharing. HybridVerifier checks each flow type independently.

---

## Hash Embedding Forbidden for Routing

**Decision:** SHA-256 hash embeddings are acceptable for S-MMU deduplication but FORBIDDEN for routing decisions.

**Why:** Hash embeddings produce random vectors with no semantic content. Using them for kNN routing would return random neighbors, making routing decisions meaningless. The kNN router (arXiv 2505.12601) explicitly refuses hash embeddings at both build and route time.

**Implication:** `KnnRouter.build_from_ground_truth()` and `KnnRouter.route()` both check `embedder.is_hash_fallback` and return None/False if true. The Embedder 3-tier fallback (Rust ONNX > sentence-transformers > hash) means routing degrades gracefully — falls back to structural features when no semantic embedder is available.

---

## Python Shadow Oracles

**Decision:** Python routing/topology modules are frozen as shadow oracles. They are deleted ONLY after Rust parity is proven on 1000+ traces with <5% divergence.

**Why:** Premature deletion of Python modules caused regressions in earlier sprints. The shadow approach lets Rust code run in production while Python code validates correctness in parallel. Divergence traces are logged as JSONL for offline analysis.

**Implication:** `ShadowRouter` runs both paths and logs divergences. Phase 5 gate: soft (500 traces, <10%) enables Rust-primary; hard (1000 traces, <5%) enables Python deletion. Gate not yet reached — needs operational trace collection.

---

## Self-MoA Finding

**Decision:** Single strong model with retries is preferred over multi-model ensembles for code tasks.

**Why:** Multi-provider design research (March 8, 2026) found that ensemble routing across providers added latency and cost without improving quality on HumanEval. A single capable model (e.g., Codex gpt-5.3) with AVR retry loops outperformed scatter-gather across 3+ models.

**Implication:** FrugalGPT cascade is for *fallback* (provider down), not for *quality* (ensemble voting). The cascade tries the primary model, then falls back to alternatives only on failure — not for aggregating multiple responses.

---

## OxiZ over z3-sys

**Decision:** Use OxiZ (pure Rust SMT solver) as primary formal verification backend, with Python z3-solver as fallback.

**Why:** z3-sys requires compiling Z3 C++ library (~15 min build, 500MB), causing CI timeouts and Windows build failures. OxiZ is pure Rust with zero C++ dependencies, sub-0.1ms verification times, and seamless PyO3 integration. Trade-off: OxiZ only supports QF_LIA (quantifier-free linear integer arithmetic), while Z3 supports full SMT-LIB.

**Implication:** All 10 SMT verification methods route through Rust OxiZ first. Python z3-solver is imported lazily as fallback for features OxiZ can't handle. `solver.set_logic("QF_LIA")` is required for OxiZ integer solving.

---

## Dual-Mode Topology Executor

**Decision:** TopologyExecutor supports two scheduling modes — Static (Kahn's toposort) for acyclic graphs, Dynamic (gate-based readiness) for cyclic topologies.

**Why:** Most topologies are DAGs and benefit from simple topological sort. But some advanced topologies (AVR loops, debate cycles) contain cycles. A single executor must handle both without requiring separate code paths.

**Implication:** The executor detects cycles at init time and selects the appropriate scheduler. Dynamic mode uses gate counters per node — a node fires when all incoming gates are open.

---

## arctic-embed-m for Embeddings

**Decision:** Use Snowflake arctic-embed-m (109M params, 768-dim) as the embedding model.

**Why:** User explicitly approved this model. It balances quality (MTEB top-tier) with size (109M fits in memory on dev machine). The 768-dim output is standard for downstream tasks (kNN routing, S-MMU similarity). Previous model (all-MiniLM-L6-v2, 384-dim) was replaced when kNN routing required higher-quality embeddings.

**Implication:** Embedding dimension is 768 everywhere (Rust embedder, Python fallback, .npz exemplar files, S-MMU chunks). Changing the model requires rebuilding routing_exemplars.npz and restarting.

---

## A2A Pinned to v1.0

**Decision:** A2A (Agent2Agent) protocol pinned to v1.0 SDK.

**Why:** Breaking changes confirmed between v0.3 and v1.0 (different AgentCard schema, TaskState enum, auth flow). The v1.0 API stabilized the AgentExecutor pattern and skill-based routing.

**Implication:** `a2a-sdk >= 1.0` in pyproject.toml. Do not upgrade to 2.x without verifying AgentCard compatibility.

---

## ExoCortex: Active Tool Only

**Decision:** ExoCortex (Google File Search) is exposed only as an active `search_exocortex` agent tool, not as passive grounding in every _think() call.

**Why:** Sprint 3 evidence benchmarks showed passive ExoCortex grounding adds 2-5s latency per think cycle without improving code task quality. The retrieval was often irrelevant (research papers don't help with "write a sort function"). Active tool use lets the agent decide when external knowledge is needed.

**Implication:** `search_exocortex` tool is always available. Passive grounding code was removed. Re-enable only if evidence shows benefit for specific task types (e.g., research synthesis, novel algorithm implementation).

---

## CircuitBreaker Pattern

**Decision:** 6 independent circuit breakers protect non-critical subsystems in agent_loop.

**Why:** Memory, evolution, and guardrail subsystems can fail without killing the core perceive→think→act loop. Rather than wrapping everything in try/except (silent failures), CircuitBreaker tracks consecutive failures per subsystem and opens (skips calls with WARNING) after 3 failures.

**Implication:** Breakers: semantic_memory, smmu_context, runtime_guardrails, episodic_store, entity_extraction, evolution_stats. A broken S-MMU doesn't prevent task completion — it just degrades context quality. `record_success()` resets the counter.

---

## Routing Benchmark: Non-Circular Ground Truth

**Decision:** Routing accuracy must be measured against human-labeled ground truth, not labels derived from the router itself.

**Why:** The original 30-task benchmark scored 100% because the labels were reverse-engineered from the heuristic's own outputs — a circular test that proves nothing. The 50-task ground truth dataset uses domain expertise to label tasks as S1/S2/S3 independently.

**Implication:** `config/routing_ground_truth.json` contains 50 tasks (10 S1, 20 S2, 20 S3) labeled by complexity analysis, not by running the router. kNN routing scores 92% on this dataset (vs 52% heuristic).
