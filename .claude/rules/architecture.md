---
paths:
  - "sage-core/**"
  - "sage-python/**"
---

# Architecture Quick Reference

## Project Structure
- `sage-core/` — Rust orchestrator (PyO3). 259 tests.
- `sage-python/` — Python SDK. 1500+ tests (68 veRL/GiGPO-specific).
- `sage-discover/` — Knowledge pipeline (arXiv → ExoCortex). 52 tests.
- `ui/` — Dashboard (FastAPI + WebSocket).
- `Researches/` — 25+ research papers backing architecture decisions.

## 5 Cognitive Pillars
1. **Topology** — Rust TopologyEngine: 7-path generation (S-MMU → archive → LLM → mutation → MCTS → **Path 6: learned policy** → template). MAP-Elites + CMA-ME evolution. Online evolution enabled (_auto_evolve=True). Path 6 uses SFT Phi-4-mini-instruct (70% YAML valid, opt-in via `SAGE_ENABLE_PATH6=1`).
2. **Tools** — AgentTool.from_agent(), 3-layer sandbox (tree-sitter → Wasm WASI → subprocess).
3. **Memory** — 4-tier: Rust Arrow STM → SQLite Episodic → Entity Semantic → ExoCortex RAG. S-MMU paging with ULID chunks.
4. **Evolution** — MAP-Elites quality-diversity + CMA-ME + MCTS topology search. DGM/SAMPO 5 strategic actions. Online evolution wired in pipeline Stage 5.
5. **Strategy** — S1/S2/S3 cognitive routing (Kahneman). kNN primary (92%), Rust SystemRouter (88%). ContextualBandit Thompson sampling.

## Pipeline (5-stage)
```
CLASSIFY (kNN/SystemRouter) → DECOMPOSE (TaskPlanner) → SELECT TOPOLOGY (TopologyEngine)
→ ASSIGN MODELS (Rust ModelAssigner: affinity 0.4 + domain 0.4 + cost 0.2)
→ EXECUTE (TopologyRunner with per-node ProviderPool resolution)
→ LEARN (QualityEstimator Z3 → Bandit + MAP-Elites archive)
```

## Self-Adaptive Engine
- **SA-1**: Runtime Agent Factory — custom TopologyNode prompts, LLM-generated agent specs. Done.
- **SA-3**: Online Evolution — _auto_evolve=True, pipeline records outcomes to archive. Done.
- **SA-4**: Z3 Quality Pipeline — QualityLabeler (Rust formal), zero heuristics. Done.
- **Path 6**: Learned topology policy (Phi-4-mini-instruct SFT, 70% YAML valid). Opt-in via `SAGE_ENABLE_PATH6=1`. Auto-downloads from `yannabadie/sage-topology-policy` on HuggingFace. Lazy-loaded on first call (no boot impact). Falls back to templates on invalid output.
- **GiGPO veRL** (VeRLGIGPO branch): Multi-step topology env (SageTopologyEnv) with per-node rewards + anchor states. Edge-level credit (Graph-GRPO arXiv 2603.02701). QualityLabeler (OxiZ) for per-node scoring. ModelAssigner + 8 providers. Qwen3.5-9B on RunPod H100.

## Key Competitors
- **OpenSage** (ICML '26): AI-created agents+tools+memory at runtime. 59% SWE-Bench Pro.
- **AgentConductor** (arXiv 2602.17100): RL topology evolution, 97.5% HumanEval with 3B model.

## Benchmarks
- BigCodeBench Hard Instruct: SAGE 37.8% (budget model) vs leaderboard 33.1% (o3-mini, stale)
- Leaderboard is frozen since April 2025. Frontier 2026 models (GPT-5.4, Opus 4.6) not submitted.
- The VALUE of SAGE is the framework delta, not absolute score.
