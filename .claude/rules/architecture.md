---
paths:
  - "sage-core/**"
  - "sage-python/**"
---

# Architecture Quick Reference

## Project Structure
- `sage-core/` — Rust orchestrator (PyO3). **584 tests** with `--features smt` (canonical: `docs/status/current.json`).
- `sage-python/` — Python SDK. **3631 collected** (canonical: `docs/status/current.json`). 2026-05-10 provider/model-catalog ticket: targeted provider/routing pytest slice green, **mypy 0 errors / 263 source files**, ruff clean. Full-suite recertification remains a separate gate.
- `sage-discover/` — Knowledge pipeline (arXiv → ExoCortex). **100 tests** (canonical: `docs/status/current.json`).
- `ui/` — Dashboard (FastAPI + WebSocket).
- `Researches/` — 25+ research papers backing architecture decisions.

## 5 Cognitive Pillars
1. **Topology** — Rust TopologyEngine: 6-path generation (S-MMU → archive → LLM → mutation → MCTS → template fallback). 11 templates (sequential, parallel, AVR, selfmoa, hierarchical, hub, debate, brainstorming, robust, horizon_pipeline, parallel_fanout). DAG-driven selection via `select_macro_topology(omega, delta, gamma)`. MAP-Elites + CMA-ME evolution. Multi-turn debate loop (reset_node + open_gate, max 3 rounds). Per-node streaming via `run_stream()`. HITL approval callback. Optional learned-policy path (legacy env-var name `SAGE_ENABLE_PATH6=1`, sibling-of-6, NOT engine path 6).
2. **Tools** — AgentTool.from_agent(), 2-layer sandbox since 2026-04-22 §5 flip (ADR-013): tree-sitter AST validator + embedded RustPython wasm32-wasip1 runtime (deny-by-default WASI). `validate_and_execute` runs sandboxed by default (no opt-in), subprocess fallback removed. `execute_raw` bypasses both layers and stays gated by `SAGE_UNSAFE_RAW_EXEC=1`. Wasm-python JIT cache (2026-04-23, commit `50b4ee8`) amortises the ~30 s cranelift cold-start to ~1 s via `Module::serialize` + `$HOME/.sage/wasm_python_cache/`. SWE-bench bench layer carries a pre-emission **diff-context verifier** (observe mode via `SAGE_DIFF_VERIFIER_MODE=observe`) that flags hunks whose context/removed lines don't match file bytes at the claimed position — designed to catch the context-hallucination class (astropy-14182 Arm B precedent).
3. **Memory** — 4-tier: Rust Arrow STM → SQLite Episodic → Entity Semantic + Causal → ExoCortex RAG. S-MMU paging with ULID chunks. Inter-tier consolidation (Episodic→Semantic→Causal, MAGMA). Composite 5-signal write gate (Rust `RustCompositeWriteGate`). Causal edges from entity extraction + tool calls.
4. **Evolution** — MAP-Elites quality-diversity + CMA-ME + MCTS topology search. DGM/SAMPO 5 strategic actions. Online evolution: Rust `should_evolve()` gates `evolve()` in agent loop (SA-3 complete). AdaptiveMutator (Thompson sampling, ShinkaEvolve). Statistical validation via Wilcoxon signed-rank + Cohen's d.
5. **Strategy** — S1/S2/S3 cognitive routing (Kahneman). kNN primary, Rust SystemRouter (`routing.knn_92pct` ≥50/60 LOO-CV + `routing.system_router_88pct` ≥52/60 `delivered` in `docs/CLAIMS.yaml`; historical 92%/88% on earlier 50-task GT provenance only). ContextualBandit Thompson sampling. Runtime adaptation (`TopologyController`): **Rust-primary since 2026-04-20** (ADR-012) — decision paths 1 (empty/error reroute), 2 (quality cascade), 3 (debate-gate threshold), 4 (parallel inconsistency), 5 (importance prune) + state machine all live in Rust `RustTopologyController`. Python wraps for embedder/SmtVerifier/topology-graph access (scoring + enrichment). Emergent subtasks routed via `sage_recurse` tool with Rust-side `should_trigger_emergent_spawn` budget gate (ADR-012 follow-up, 2026-04-20 phase-1 stab). `sage_core` is required at runtime — `ImportError` raised at `TopologyController.__init__` if absent.

## Pipeline
```
CLASSIFY (kNN/SystemRouter) → DECOMPOSE (TaskPlanner → DAGFeatures omega/delta/gamma)
→ SELECT TOPOLOGY (DAG-driven select_macro_topology OR TopologyEngine 6-path)
→ ASSIGN MODELS (Rust ModelAssigner + bandit quality override for underperforming models)
→ EXECUTE (TopologyRunner: adaptive context, similarity gate, multi-turn, streaming, HITL)
→ LEARN (QualityEstimator Z3 → Bandit + MAP-Elites archive, persisted to SQLite)
```

## CLI surface (cycle-12 prelude, 2026-05-05)

`sage run --jsonl <task>` — machine-readable backend for pi-mono / TUI / IDE
front-ends. Implements the v0 protocol at `docs/contracts/SAGE_CLI_PROTOCOL.md`:
14 inherited `RuntimeEventLog` events tee'd to stdout + 4 CLI-shell envelope
events (`cli_started`, `cli_progress`, `cli_tool_request`, `cli_complete`) +
5 inbound commands (`prompt`, `approve_tool_call`, `deny_tool_call`, `cancel`,
`set_budget`). Strict JSONL with LF-only delimiters (NOT Node `readline`-
compatible — pi-mono RPC spec). The CLI is the runtime contract's public
surface; cycle-13 wraps it in a TypeScript `clients/pi-ygn-sage/` npm package
running pi-mono as the TUI shell. Frontend MUST NOT decide model / topology
/ learning gate — those stay in YGN-SAGE backend per cgpro pivot review
2026-05-05 ("YGN-SAGE should not become another coding agent CLI; it should
become the verified adaptive orchestration layer that a coding agent CLI
finally makes usable").

## Self-Adaptive Engine
- **SA-1**: Runtime Agent Factory — custom TopologyNode prompts, LLM-generated agent specs. Done.
- **SA-3**: Online Evolution — _auto_evolve=True, pipeline records outcomes to archive. Done.
- **SA-4**: Z3 Quality Pipeline — QualityLabeler (Rust formal), zero heuristics. Done.
- **Optional learned-policy path** (legacy env-var name `SAGE_ENABLE_PATH6`; sibling-of-6, NOT engine path 6 per Rust `TopologySource` enum which reserves engine path 6 for `TemplateFallback`). V1 (legacy): Phi-4-mini-instruct SFT, 70% YAML valid. V2: Nemotron-Orchestrator-8B GiGPO. **Inference-only on `main`** (training code parked since 2026-04-15 commit `b2f59ee`, lives on dedicated `training` branch). **Opt-in via `SAGE_ENABLE_PATH6=1`** — flag is read at boot and the learned-policy path is off otherwise. When opted in, the runtime auto-downloads the inference checkpoint from `yannabadie/sage-topology-policy-v2` (or `-local` for the Phase C V1 best, 0.922 structural / 40% MASBENCH). Lazy-loaded on first call (no boot impact). Falls back to templates on invalid output. **Capability state: opt-in / inference-only on main** (see root README "Capability State Table" + `docs/claims/topology.yaml` `topology.path6_learned`).
- **GiGPO veRL** (VeRLGIGPO branch): Multi-step topology env (SageTopologyEnv) with per-node rewards + anchor states. Edge-level credit (Graph-GRPO arXiv 2603.02701). QualityLabeler (OxiZ) for per-node scoring. ModelAssigner + 8 providers. nvidia/Nemotron-Orchestrator-8B (NVIDIA Open Model License, Qwen3 architecture, GRPO-trained orchestrator — arXiv 2511.21689) on RunPod H100. 12 audit fixes applied: bandit learning loop, predecessor context, upgrade model resolution (set_node_model_id), fresh executor on reroute, real embeddings, trivial topology penalty, replay buffer, S-MMU utility eviction (auto-trigger 10K), archive descriptor enriched, quality cache eviction.

## Key Competitors
- **OpenSage** (ICML '26): AI-created agents+tools+memory at runtime. 59% SWE-Bench Pro.
- **AgentConductor** (arXiv 2602.17100): RL topology evolution, 97.5% HumanEval with 3B model.

## Benchmarks
- BigCodeBench Hard Instruct (full pipeline, fast tier): SAGE 45.9% (2026-04-26, internal eval).
- BCB-Hard N=50 official Docker (2026-04-29): internal 30% / Docker 32% / 49/50 per-task agreement (budget tier, oracle path).
- A2 ablation v7 (budget tier, N=10): **4/10 PASS on `full` config — GATE MET** (2026-05-03). Fixes: episodic.db close, OUTPUT REQUIREMENT, cross-provider guard, runner guard, entry_point. A3 N=50 RUNNING.
- SWE-bench Lite Docker-graded: 10% (1/10, 2026-04-21). Patch-generation rate 70%.
- Leaderboard is frozen since April 2025. Frontier 2026 models not submitted.
- The VALUE of SAGE is the framework delta (ablation study), not absolute score vs frontier.
