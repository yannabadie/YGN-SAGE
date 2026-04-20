---
paths:
  - "sage-core/**"
  - "sage-python/**"
---

# Architecture Quick Reference

## Project Structure
- `sage-core/` — Rust orchestrator (PyO3). 478 tests (+32 for RustTopologyController 2026-04-20).
- `sage-python/` — Python SDK. 1939 tests (45 skipped; 5 asyncio-fixture errors pre-existing).
- `sage-discover/` — Knowledge pipeline (arXiv → ExoCortex). 52 tests.
- `ui/` — Dashboard (FastAPI + WebSocket).
- `Researches/` — 25+ research papers backing architecture decisions.

## 5 Cognitive Pillars
1. **Topology** — Rust TopologyEngine: 6-path generation (S-MMU → archive → LLM → mutation → MCTS → template). 11 templates (sequential, parallel, AVR, selfmoa, hierarchical, hub, debate, brainstorming, robust, horizon_pipeline, parallel_fanout). DAG-driven selection via `select_macro_topology(omega, delta, gamma)`. MAP-Elites + CMA-ME evolution. Multi-turn debate loop (reset_node + open_gate, max 3 rounds). Per-node streaming via `run_stream()`. HITL approval callback. Path 6 learned policy opt-in via `SAGE_ENABLE_PATH6=1`.
2. **Tools** — AgentTool.from_agent(), 3-layer sandbox (tree-sitter → Wasm WASI → subprocess).
3. **Memory** — 4-tier: Rust Arrow STM → SQLite Episodic → Entity Semantic + Causal → ExoCortex RAG. S-MMU paging with ULID chunks. Inter-tier consolidation (Episodic→Semantic→Causal, MAGMA). Composite 5-signal write gate (Rust `RustCompositeWriteGate`). Causal edges from entity extraction + tool calls.
4. **Evolution** — MAP-Elites quality-diversity + CMA-ME + MCTS topology search. DGM/SAMPO 5 strategic actions. Online evolution: Rust `should_evolve()` gates `evolve()` in agent loop (SA-3 complete). AdaptiveMutator (Thompson sampling, ShinkaEvolve). Statistical validation via Wilcoxon signed-rank + Cohen's d.
5. **Strategy** — S1/S2/S3 cognitive routing (Kahneman). kNN primary (92%), Rust SystemRouter (88%). ContextualBandit Thompson sampling. Runtime adaptation (`TopologyController`): **Rust-primary since 2026-04-20** (ADR-012) — decision paths 1 (empty/error reroute), 2 (quality cascade), 3 (debate-gate threshold), 4 (parallel inconsistency), 5 (importance prune), 6 (emergent spawn) + state machine all live in Rust `RustTopologyController`. Python wraps for embedder/SmtVerifier/topology-graph access (scoring + enrichment). Legacy Python path preserved as `_evaluate_and_decide_legacy` for sage_core-less environments.

## Pipeline (5-stage)
```
CLASSIFY (kNN/SystemRouter) → DECOMPOSE (TaskPlanner → DAGFeatures omega/delta/gamma)
→ SELECT TOPOLOGY (DAG-driven select_macro_topology OR TopologyEngine 6-path)
→ ASSIGN MODELS (Rust ModelAssigner + bandit quality override for underperforming models)
→ EXECUTE (TopologyRunner: adaptive context, similarity gate, multi-turn, streaming, HITL)
→ LEARN (QualityEstimator Z3 → Bandit + MAP-Elites archive, persisted to SQLite)
```

## Self-Adaptive Engine
- **SA-1**: Runtime Agent Factory — custom TopologyNode prompts, LLM-generated agent specs. Done.
- **SA-3**: Online Evolution — _auto_evolve=True, pipeline records outcomes to archive. Done.
- **SA-4**: Z3 Quality Pipeline — QualityLabeler (Rust formal), zero heuristics. Done.
- **Path 6**: Learned topology policy. V1 (legacy): Phi-4-mini-instruct SFT, 70% YAML valid. V2: Nemotron-Orchestrator-8B GiGPO. Opt-in via `SAGE_ENABLE_PATH6=1`. Auto-downloads from `yannabadie/sage-topology-policy-v2` on HuggingFace. Lazy-loaded on first call (no boot impact). Falls back to templates on invalid output.
- **GiGPO veRL** (VeRLGIGPO branch): Multi-step topology env (SageTopologyEnv) with per-node rewards + anchor states. Edge-level credit (Graph-GRPO arXiv 2603.02701). QualityLabeler (OxiZ) for per-node scoring. ModelAssigner + 8 providers. nvidia/Nemotron-Orchestrator-8B (NVIDIA Open Model License, Qwen3 architecture, GRPO-trained orchestrator — arXiv 2511.21689) on RunPod H100. 12 audit fixes applied: bandit learning loop, predecessor context, upgrade model resolution (set_node_model_id), fresh executor on reroute, real embeddings, trivial topology penalty, replay buffer, S-MMU utility eviction (auto-trigger 10K), archive descriptor enriched, quality cache eviction.

## Key Competitors
- **OpenSage** (ICML '26): AI-created agents+tools+memory at runtime. 59% SWE-Bench Pro.
- **AgentConductor** (arXiv 2602.17100): RL topology evolution, 97.5% HumanEval with 3B model.

## Benchmarks
- BigCodeBench Hard Instruct: SAGE 37.8% (budget model) vs leaderboard 33.1% (o3-mini, stale)
- Leaderboard is frozen since April 2025. Frontier 2026 models (GPT-5.4, Opus 4.6) not submitted.
- The VALUE of SAGE is the framework delta, not absolute score.
