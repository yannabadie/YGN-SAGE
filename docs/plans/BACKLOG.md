# YGN-SAGE Backlog

Consolidated from 45 planning documents (2026-03-02 to 2026-03-10). Items already implemented are omitted — see CLAUDE.md for current state. Last updated: 2026-03-12.

---

## P0 — Operational Gates (blocking further progress)

### Shadow Trace Evidence Gate (Phase 5)
- Run 1000+ dual Rust/Python routing traces via ShadowRouter
- Soft gate: 500 traces, <10% divergence → enable Rust-primary routing
- Hard gate: 1000 traces, <5% divergence → delete Python routing modules
- Source: `sage-adaptive-router.md`, `rust-cognitive-engine-design.md`

### BigCodeBench Benchmark (1140 tasks)
- Adapter for BigCodeBench real-world coding tasks
- Validates framework value beyond HumanEval+/MBPP+
- Source: `official-benchmarks.md` Task 5

### Full Memory Ablation Study
- 4 configs: none, tier0-only, tier0+episodic, full 4-tier
- Proves each memory tier's marginal contribution
- Source: `audit-response-v2.md` Sprint 3

### Full Evolution Ablation Study
- 4-arm: full pipeline vs random mutation vs MAP-Elites only vs seed-only
- Quantifies evolution engine value vs random search
- Source: `audit-response-v2.md` Sprint 3

---

## P1 — High Value, Designed but Unbuilt

### Speculative S1+S2 Parallel Execution
- When routing confidence is in indecisive zone (0.35-0.55), launch both S1 and S2 in parallel
- Return S1 result if good enough (quality > threshold), else use S2
- Saves latency on borderline tasks at cost of extra compute
- Source: `wire-smmu-plan.md` Task 9

### Drift Monitor
- Track agent behavioral degradation: weighted score = 40% latency + 40% errors + 20% cost
- Alert when 7-day rolling average degrades beyond 1σ
- Dashboard visualization of drift trends
- Source: `phase3-domination.md` Task 19

### SelfImprovementLoop
- Autonomous cycle: benchmark → diagnose weak spots → evolve (mutation/retraining) → re-benchmark
- Requires BigCodeBench + evolution engine + reliable quality estimation
- Source: `phase3-domination.md` Task 24

### Dynamic Agent Factory
- Self-programming: LLM generates agent blueprint (role, tools, topology) from task description
- Agent registry for reuse of discovered specializations
- Source: `phase3-domination.md` Task 18

### A2A Agent Card Export/Import
- Export SAGE ModelCards as A2A Agent Cards (external discovery)
- Import remote A2A Agent Cards into ModelRegistry (federated agent mesh)
- Map A2A skill tags as routing input in SystemRouter
- A2A TaskState → perceive/think/act/learn phases
- Auth: extend to OAuth2/mTLS for production
- Source: `a2a-protocol-research.md`

### Provider Smoke Tests at Boot
- Validate each discovered provider can: generate text, call tools, respond within latency baseline
- Disable providers that fail smoke test (currently relies on cascade fallback)
- Source: `multi-provider-design.md`

### S-MMU ANN Index (usearch)
- Replace O(n²) brute-force similarity in S-MMU with approximate nearest neighbor
- usearch or faiss for sub-ms retrieval at scale
- Source: `audit-response-v2.md` Sprint 2

---

## P2 — Medium Value, Research-Backed

### BERT/DeBERTa Classifier for Stage 1 Routing
- NVIDIA prompt-task-and-complexity-classifier (DeBERTa-v3-base): 98.1% multi-head classification
- RouteLLM (ICLR 2025, 2406.18665): BERT 0.3B on Chatbot Arena preference data
- Would replace/augment kNN Stage 0.5 with learned classifier
- ONNX infrastructure ready (ort 2.0, tokenizers), needs training data
- Source: `sage-adaptive-router.md`, memory notes

### Full PSRO/DCH Strategy Engine
- Replace contextual bandit with game-theoretic meta-solvers
- Regret Matching, PRD, Softmax, SHOR-PSRO approaches researched
- VAD-CFR: volatility-adaptive CFR with EWMA + dynamic annealing
- Needs: sufficient routing data + compute for equilibrium computation
- Source: `opensage-surpass-implementation.md`, `notebooklm_research_synthesis.md`

### vqsort-rs Integration
- Replace manual AVX-512 sorts with portable vqsort
- Target: h96_quicksort, h96_quicksort_zerocopy, vectorized_partition_h96
- New h96_argsort() for indexed sort in MCTS UCB
- Source: `z3-simd-smmu-implementation.md`

### Coordination Performance Model
- Data-driven analysis: when does topology > model capability?
- AdaptOrch finding: topology structure explains up to 10% perf gap
- Would inform when to invest in topology search vs model selection
- Source: `phase3-domination.md` Task 23

### ModelWatcher (New Model Detection)
- Monitor provider APIs for new model releases
- Auto-add to ModelRegistry with default capability estimates
- Re-run smoke tests + routing benchmark on new models
- Source: `phase3-domination.md` Task 22

### CPython WASI Component (Tool Executor Phase 2-3)
- Full CPython interpreter running inside Wasm WASI sandbox
- componentize-py for building .wasm components
- Pre-compiled .cwasm for Windows
- Would make Wasm primary execution path (subprocess = fallback only)
- Source: `rust-tool-executor.md` Phase 2-3

---

## P3 — Future / Research Directions

### Hypothesis H7: YGN-SAGE 2.0 Architecture
- Arrow-native memory with SIMD vectorization (AVX-512/ARM NEON)
- ULID-based memory IDs (replace UUID strings, 6 bytes vs 36)
- GPU-Direct memory access (VRAM-direct token injection)
- CUDA/Triton fused kernels for game-theoretic meta-solvers
- eBPF sandboxing (replace Docker/Wasm for Linux, 100,000x faster claim)
- Firecracker microVMs (<5ms boot, alternative to Docker)
- JIT-compiled agent topologies (Rust→Wasm hot-swap)
- Liquid Neural Routing (dynamic topology rewiring per token)
- Source: `ygn_sage_future_evaluation.md`

### Neo4j/Qdrant Persistent Backends
- Replace SQLite episodic/semantic with Neo4j graph DB + Qdrant vector DB
- Cross-session semantic recall with native graph traversal
- Source: `ygn-sage-architecture-design.md`

### Graph RAG Architecture
- Hierarchical: working memory → long-term Neo4j → RAG sync
- Zettelkasten mapping: Notes=Arrow chunks, Links=multi-graph edges
- Source: `notebooklm_research_synthesis.md`, `z3-simd-smmu-integration-design.md`

### SWE-bench Lite
- Real-world software engineering benchmark
- Deferred until TaskNode IR proven + reliable code generation
- Source: `v2-evidence-first-design.md`

### Agents-as-Tools Pattern
- Expose any agent as a callable tool for other agents
- Recursive composition without explicit topology wiring
- Source: `v2-convergence-design.md`

### gRPC IPC Layer
- Replace HTTP/WebSocket with gRPC for inter-agent communication
- Enables full mesh topology with real-time streaming
- Source: `ygn-sage-implementation.md`

### Docker Multi-Stage Orchestration
- Production-grade container deployment
- Multi-tier sandboxing (Wasm inside Docker)
- Source: `ygn-sage-implementation.md`

---

## Commercialization Ideas (Non-Technical)

### Go-to-Market Strategy
- Phase 1: GitHub stars + HN front page (eBPF sandbox benchmarks, dashboard demo)
- Phase 2: Thought leadership (LinkedIn/Twitter series: Wrapper AI vs Algorithmic ASI)
- Phase 3: Monetization — Licensing ($100k/yr), Consulting (Fractional CAIO), SaaS (cloud sandbox)
- Source: `marketing-strategy.md`

---

## Deferred Cleanup

### Dead Code Removal (Low Priority)
- Verify and remove: anthropic.py, openai.py (if still present as legacy providers)
- Archive labs/ experiments → Researches/experimental_hft/
- PyO3 0.26+ migration (py.allow_threads → py.detach when available)
- Wasmtime upgrade to next LTS when v36 EOL (Aug 2027)
- Source: `codebase-audit.md`, `codebase-cleanup.md`
