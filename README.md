<p align="center">
  <img src="assets/logo.svg" alt="YGN-SAGE" width="128" height="128">
</p>

<h1 align="center">YGN-SAGE</h1>

<p align="center">
  <strong>Self-Adaptive Generation Engine — Agent Development Kit</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/ygn-sage/"><img src="https://img.shields.io/pypi/v/ygn-sage?style=flat-square" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/tests-1518%20passed-brightgreen?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/python-3.12+-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/rust-1.90+-orange?style=flat-square" alt="Rust">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
</p>

---

**YGN-SAGE learns which multi-agent topology to use for each task, adding +5.5pp over bare LLM calls on coding benchmarks.**

```bash
pip install ygn-sage
```

YGN-SAGE is a Self-Adaptive Agent Development Kit built on **5 cognitive pillars**: Topology, Tools, Memory, Evolution, Strategy. Rust core (sage-core) + Python SDK (sage-python) + Knowledge Pipeline (sage-discover).

The system designs, executes, and improves its own multi-agent architectures autonomously.

## Architecture

```
                    ┌─── PIPELINE 5-STAGE ───┐
                    │                         │
CLASSIFY ──> DECOMPOSE ──> TOPOLOGY ──> ASSIGN ──> EXECUTE ──> LEARN
   │            │            │            │          │           │
   v            v            v            v          v           v
SystemRouter TaskPlanner TopologyEngine ModelAssigner TopologyRunner Bandit+
  (Rust)      (Python)     (Rust)        (Rust)      (Python)    MAP-Elites
   │                         │                         │
   ├─ kNN (92% GT)           ├─ S-MMU retrieval        ├─ Wasm WASI sandbox
   ├─ StructuralFeatures     ├─ MAP-Elites archive     ├─ tree-sitter validator
   └─ ContextualBandit       ├─ CMA-ME mutation        ├─ subprocess fallback
                             ├─ MCTS search            └─ ProviderPool (8 providers)
                             ├─ LLM synthesis               ├─ CircuitBreaker
                             ├─ Path 6: learned policy (Qwen3-8B GiGPO V2)
                             └─ Template fallback (x8)       └─ FrugalGPT cascade
```

## 5 Cognitive Pillars

### 1. Topology (Rust)
Dynamic multi-agent architecture generation via 6-path DynamicTopologyEngine:
- **S-MMU hit**: retrieve similar past topology (similarity > 0.7)
- **MAP-Elites**: 108-cell quality-diversity archive (4D behavior descriptor)
- **CMA-ME**: covariance matrix adaptation for continuous topology parameters
- **MCTS**: Monte Carlo tree search over mutation space (UCB1)
- **LLM synthesis**: 3-stage pipeline (roles, structure, validation)
- **Templates**: 8 pre-wired patterns (sequential, parallel, AVR, debate, hub, hierarchical, selfmoa, brainstorming)

Topology graph: petgraph DiGraph with 3-flow edges (Control, Message, State).

### 2. Tools (Rust)
3-layer sandbox with defense-in-depth:
- **tree-sitter**: AST validation (23 blocked modules, 21 blocked calls)
- **Wasm WASI**: wasmtime v36 LTS Component Model (deny-by-default)
- **subprocess**: kill-on-drop timeout fallback

### 3. Memory (Rust + Python)
4-tier hierarchical memory:
- **Tier 0**: Working memory (Rust Arrow, zero-copy, SIMD)
- **Tier 1**: Episodic (SQLite, cross-session)
- **Tier 2**: Semantic (entity-relation graph)
- **Tier 3**: ExoCortex (Google File Search RAG, 500+ research sources)

S-MMU (Semantic Memory Management Unit): 4-view graph (temporal, semantic, causal, entity) with ULID chunk IDs.

### 4. Evolution (Rust)
- **MAP-Elites + CMA-ME**: quality-diversity search over topology space
- **Online evolution**: `_auto_evolve=True`, pipeline Stage 5 records outcomes to archive
- **7 mutation operators**: add/remove node, swap model, rewire edge, split/merge, mutate prompt
- **Path 6: Learned topology policy** — V1 (legacy): SFT Phi-4-mini-instruct on 2624 GPT-5.4 topologies, 70% YAML validity. V2: Qwen3-8B GiGPO via veRL on RunPod H100. Auto-downloads from [yannabadie/sage-topology-policy-v2](https://huggingface.co/yannabadie/sage-topology-policy-v2). Enable with `SAGE_ENABLE_PATH6=1`.
- **RLVR-Topology**: GiGPO with TopologyReward Rust (execution-based, not format-only) + Graph-GRPO edge credit

### 5. Strategy (Rust)
- **S1/S2/S3 cognitive routing** (Kahneman dual-process): kNN (92% GT), SystemRouter (86% GT)
- **ContextualBandit**: per-arm Thompson sampling with Pareto front
- **ModelAssigner**: per-node model selection (affinity 0.4 + domain 0.4 + cost 0.2)
- **CircuitBreaker**: per-provider fault tolerance (auto-skip after 3 failures)

## Formal Verification (Rust)

- **OxiZ SmtVerifier**: sub-0.1ms formal proofs (memory safety, loop bounds, arithmetic, invariants, CEGAR synthesis)
- **HybridVerifier**: 6 structural + 4 semantic checks on TopologyGraph, O(V+E)
- **LtlVerifier**: 4 temporal property checks (reachability, safety, liveness, bounded liveness)
- **QualityLabeler**: Z3-based quality scoring (zero heuristics)
- **TopologyDensity**: S_complex cost metric (AgentConductor Theorem 1) + N_max bounds
- **TopologyReward**: multi-signal dense reward for RL topology training

## Self-Adaptive Engine (In Progress)

| Component | Description | Status |
|-----------|-------------|--------|
| **SA-1** | Runtime Agent Factory: custom TopologyNode prompts, LLM-generated agent specs | Phase 1+2 done |
| **SA-3** | Online Evolution: `_auto_evolve=True`, pipeline records to MAP-Elites | Done |
| **SA-4** | Z3 Quality Pipeline: QualityLabeler replaces heuristic, zero false signals | Done |
| **Path 6** | Learned topology policy: V1 Phi-4-mini SFT (legacy), V2 Qwen3-8B GiGPO | Done (opt-in) |
| **RLVR-Topology** | GiGPO with TopologyReward Rust (execution-based reward) + Graph-GRPO edge credit | Training |

## Benchmark Results

| Benchmark | Score | Notes |
|-----------|-------|-------|
| **HumanEval+ pipeline** | **89.6%** (147/164) | +5.5pp over pre-pipeline (84.1%). Budget model. |
| **BigCodeBench Hard Instruct** | **37.8%** (56/148) | Budget model. Leaderboard SOTA 33.1% (stale, Apr 2025). |
| **kNN routing GT** | **92%** (46/50) | arXiv 2505.12601, arctic-embed-m |
| **Rust SystemRouter GT** | **86%** (43/50) | Domain scoring from cards.toml |

### Tests

| Suite | Result |
|-------|--------|
| Python | **1500+ passed**, 0 failures (68 veRL/GiGPO-specific) |
| Rust | **270+ passed** (with smt, tool-executor, onnx features) |
| veRL training | **404 passed** (357 Rust + 47 Python), 0 failures |
| CI | 5 jobs (Rust, Rust features, Python, Discover, Windows) |

## Quick Start

```bash
# Build
cd sage-core && maturin develop --features smt,onnx,cognitive,tool-executor
cd sage-python && pip install -e ".[all,dev]"

# Test
cd sage-core && cargo test --no-default-features --features smt,tool-executor --lib
cd sage-python && python -m pytest tests/ -v

# Benchmark (USE BigCodeBench, NOT HumanEval+)
python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 20
python -m sage.bench --type routing_gt
python -m sage.bench --type ablation --limit 50

# APPS / LiveCodeBench (competition-level code)
python -m sage.bench --type apps --limit 20
python -m sage.bench --type livecodebench --limit 20
```

## Use as a Library

```python
import asyncio
from sage.boot import boot_agent_system

async def main():
    system = boot_agent_system(llm_tier="fast")
    result = await system.run("Write a function to check if a number is prime.")
    print(result)

asyncio.run(main())
```

## Project Structure

```
YGN-SAGE/
|-- sage-core/           # Rust core (PyO3): TopologyEngine, SystemRouter, ModelAssigner,
|   |                    #   QualityLabeler, S-MMU, SmtVerifier, HybridVerifier, LtlVerifier,
|   |                    #   WasmSandbox, ToolExecutor, ContextualBandit, MAP-Elites, CMA-ME,
|   |                    #   MCTS, TopologyDensity, TopologyReward, RustEmbedder, RustKnnRouter
|   +-- config/cards.toml  # Single source of truth for 20 model cards (8 providers)
|-- sage-python/         # Python SDK
|   +-- src/sage/
|       |-- agents/      # Sequential, Parallel, Loop, Handoff composition
|       |-- bench/       # BigCodeBench, APPS, LiveCodeBench, EvalPlus, ablation, routing GT
|       |-- events/      # EventBus (central nervous system)
|       |-- guardrails/  # Cost, Output, Schema guardrails
|       |-- llm/         # Google, OpenAI, DeepSeek, xAI, Kimi, MiniMax, Codex providers
|       |-- memory/      # 4-tier: STM (Arrow), Episodic (SQLite), Semantic, ExoCortex
|       |-- strategy/    # S1/S2/S3 routing, kNN, AdaptiveRouter
|       |-- topology/    # TopologyRunner, LLM caller, TopologyController
|       |-- pipeline.py  # 5-stage CognitiveOrchestrationPipeline (ONLY execution path)
|       +-- boot.py      # System bootstrap
|-- sage-discover/       # Knowledge pipeline (arXiv -> ExoCortex)
|-- Researches/          # 25+ research papers backing architecture decisions
+-- docs/                # Specs, plans, benchmarks
```

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google AI API key (Gemini models) |
| `OPENAI_API_KEY` | No | OpenAI API key (GPT-5.x, Codex) |
| `DEEPSEEK_API_KEY` | No | DeepSeek API key |
| `GROK_API_KEY` | No | xAI Grok API key |
| `KIMI_API_KEY` | No | Moonshot Kimi API key |
| `MINIMAX_API_KEY` | No | MiniMax API key |

Provider discovery cached 24h at `~/.sage/discovery_cache/`. Model cards at `sage-core/config/cards.toml` (single source of truth).

## Research References

- **kNN routing**: arXiv 2505.12601 (92% accuracy)
- **AgentConductor**: arXiv 2602.17100 (RL topology evolution, +29pp from RL)
- **Graph-GRPO**: arXiv 2603.02701 (edge-level credit assignment)
- **AdaptOrch**: arXiv 2602.16873 (topology > model capability)
- **OpenSage**: arXiv 2602.16891 (AI-created agents at runtime, ICML '26)
- **AlphaEvolve**: arXiv 2506.13131 (LLM as mutation operator)
- **Cascade Routing**: arXiv 2410.10347 (quality estimator > routing algorithm)
- **CoALA**: cognitive architecture foundation
- **FoVer**: Z3 auto-labels for PRM training

## License

MIT License. (c) 2026 Yann Abadie. See [LICENSE](LICENSE).
