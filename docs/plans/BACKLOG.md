# YGN-SAGE Backlog

Consolidated from 45 planning documents (2026-03-02 to 2026-03-10). Items already implemented are omitted — see CLAUDE.md for current state. Last updated: 2026-03-12.

Research verdicts added 2026-03-12 via Context7 + ExoCortex + web search.

---

## P0 — Operational Gates (blocking further progress)

### Shadow Trace Evidence Gate (Phase 5)
- Run 1000+ dual Rust/Python routing traces via ShadowRouter
- Soft gate: 500 traces, <10% divergence → enable Rust-primary routing
- Hard gate: 1000 traces, <5% divergence → delete Python routing modules
- **Status**: Batch runner script ready (`scripts/collect_shadow_traces.py`). Run `python scripts/collect_shadow_traces.py --rounds 20` for 1000 traces.
- Source: `sage-adaptive-router.md`, `rust-cognitive-engine-design.md`

### BigCodeBench + LiveCodeBench Benchmarks
- **BigCodeBench** (ICLR '25, 1140 tasks): real-world coding, unsaturated (best ~62%), standard eval harness
- **LiveCodeBench** (rolling, contamination-free): monthly new problems from competitive programming
- **Verdict**: KEEP BOTH — HumanEval is saturated (99%+), these are the SOTA validation targets
- SWE-bench Lite replaced by SWE-bench Pro (Feb 2026) — skip both until TaskNode IR proven
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

### ModernBERT Classifier for Stage 1 Routing
- **ModernBERT** (Dec 2024): 2-4x faster than DeBERTa, 8192-token context, ONNX-ready
- Replaces NVIDIA DeBERTa-v3-base (P2 item) — strictly superior for our use case
- ONNX infrastructure ready (ort 2.0, tokenizers), needs labeled training data
- **Verdict**: MODIFY — use ModernBERT instead of DeBERTa. Collect 500+ labeled traces first.
- Source: `sage-adaptive-router.md`, RouteLLM (ICLR 2025, 2406.18665)

### Dynamic Agent Factory (via TopologyEngine)
- **Verdict**: MODIFY — don't build a separate factory. Use existing `DynamicTopologyEngine.generate()` with LLM synthesis (Path 3). Already supports role assignment → structure → validation pipeline.
- Add an agent registry for caching discovered specializations (HashMap<task_type, topology_id>)
- Source: `phase3-domination.md` Task 18, MASFactory (2603.06007)

### SelfImprovementLoop
- Autonomous cycle: benchmark → diagnose weak spots → evolve (mutation/retraining) → re-benchmark
- Requires BigCodeBench + evolution engine + reliable quality estimation
- **Verdict**: KEEP — blocked on BigCodeBench adapter. Quality estimation is the bottleneck (ETH-SRI, ICLR 2025).
- Source: `phase3-domination.md` Task 24

### A2A Agent Card Export/Import
- Export SAGE ModelCards as A2A Agent Cards (external discovery)
- Import remote A2A Agent Cards into ModelRegistry (federated agent mesh)
- **Verdict**: KEEP — A2A v1.0 SDK stable, straightforward mapping
- Source: `a2a-protocol-research.md`

### Coordination Performance Model
- Data-driven analysis: when does topology > model capability?
- AdaptOrch finding: topology structure explains up to 10% perf gap
- **Verdict**: KEEP — collect data from TopologySmmuBridge outcomes, then fit model
- Source: `phase3-domination.md` Task 23, AdaptOrch (2602.16873)

### ModelWatcher (New Model Detection)
- Monitor provider APIs for new model releases
- Auto-add to ModelRegistry with default capability estimates
- **Verdict**: KEEP — low-effort, high operational value
- Source: `phase3-domination.md` Task 22

---

## P2 — Medium Value, Deferred

### CPython WASI Component (Tool Executor Phase 2-3)
- Full CPython interpreter running inside Wasm WASI sandbox
- **Verdict**: DEFER to WASIp3 (mid-2026) — componentize-py still experimental, WASI p2 lacks stable socket/filesystem APIs needed for real Python code execution. Current subprocess fallback is adequate.
- Source: `rust-tool-executor.md` Phase 2-3

### S-MMU ANN Index
- Replace O(K) brute-force scan with approximate nearest neighbor
- **Verdict**: DEFER — at K=128 (MAX_SEMANTIC_NEIGHBORS), brute force is optimal. ANN overhead (index build, memory) only pays off at ~10K+ chunks. Revisit when S-MMU exceeds 1000 chunks in production.
- Source: `audit-response-v2.md` Sprint 2

---

## Dropped Items (with rationale)

### ~~Speculative S1+S2 Parallel Execution~~
- **DROP**: No SOTA backing. ETH-SRI (ICLR 2025) shows quality estimation is the bottleneck, not routing speed. Speculative execution adds complexity for marginal latency gain on a narrow confidence band (0.35-0.55).

### ~~Full PSRO/DCH Strategy Engine~~
- **DROP**: Wrong abstraction for LLM routing. PSRO/Nash equilibria assume adversarial agents with fixed strategy sets — our problem is cooperative model selection with non-stationary rewards. Contextual bandit (LinUCB) is the correct framework per PILOT (2508.21141) and arXiv 2506.17670.

### ~~vqsort-rs Integration~~
- **DROP**: The `vqsort-rs` crate is dead (last update 2023, no SIMD backend). Rust stdlib sort is already O(n log n) with pattern-defeating quicksort. Manual AVX-512 sorts in sage-core are only for the 50-element kNN distance array — not a bottleneck.

### ~~Provider Smoke Tests at Boot~~
- **DROP**: FrugalGPT cascade already handles provider failures transparently. Adding boot-time smoke tests would increase startup latency and create false negatives (transient API errors during boot ≠ permanent provider failure).

### ~~Neo4j/Qdrant Persistent Backends~~
- **DROP**: sqlite-vec (by Alex Garcia, 2024) provides vector search inside SQLite — no external DB needed. Current S-MMU is in-memory with SQLite persistence for episodic/semantic. Adding Neo4j/Qdrant is overengineering for current scale (<1000 chunks).

### ~~SWE-bench Lite~~
- **DROP**: Replaced by SWE-bench Pro (Feb 2026). Both require reliable code generation + tool use, which depends on TaskNode IR maturity. Defer to P3.

### ~~BERT/DeBERTa Classifier~~ (superseded)
- **Superseded by ModernBERT** (P1 above). ModernBERT is 2-4x faster with 8192-token context vs DeBERTa's 512.

---

## P3 — Future / Research Directions

### Hypothesis H7: YGN-SAGE 2.0 Architecture
- Arrow-native memory with SIMD vectorization (AVX-512/ARM NEON)
- GPU-Direct memory access (VRAM-direct token injection)
- CUDA/Triton fused kernels for game-theoretic meta-solvers
- eBPF sandboxing (replace Docker/Wasm for Linux)
- Firecracker microVMs (<5ms boot, alternative to Docker)
- JIT-compiled agent topologies (Rust→Wasm hot-swap)
- Liquid Neural Routing (dynamic topology rewiring per token)
- Source: `ygn_sage_future_evaluation.md`

### Graph RAG Architecture
- Hierarchical: working memory → long-term graph → RAG sync
- Zettelkasten mapping: Notes=Arrow chunks, Links=multi-graph edges
- Source: `notebooklm_research_synthesis.md`

### gRPC IPC Layer
- Replace HTTP/WebSocket with gRPC for inter-agent communication
- Enables full mesh topology with real-time streaming
- Source: `ygn-sage-implementation.md`

### Docker Multi-Stage Orchestration
- Production-grade container deployment
- Multi-tier sandboxing (Wasm inside Docker)
- Source: `ygn-sage-implementation.md`

---

## Recently Completed (2026-03-12)

### ~~ULID Memory IDs~~
- S-MMU chunk_id migrated from `usize` to ULID strings (Rust `ulid` crate)
- Cross-session stable, globally unique, lexicographically sortable
- All 6 Rust files updated, 249 tests passing

### ~~Agents-as-Tools Pattern~~
- `sage.tools.agent_tool.AgentTool`: wraps any `async run(task)->str` as a Tool
- Universal pattern (OpenAI Agents SDK, Google ADK, CrewAI)
- 4 tests passing

### ~~DriftMonitor Wiring~~
- Wired into `agent_loop.py._emit()` with sliding window analysis
- Emits DRIFT events on EventBus when action != CONTINUE
- 3-signal composite: latency (40%), errors (40%), cost (20%)

### ~~Shadow Traces Batch Runner~~
- `scripts/collect_shadow_traces.py --rounds N`
- Runs GT tasks through ShadowRouter, reports Phase 5 gate status

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
