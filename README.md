<p align="center">
  <img src="assets/logo.svg" alt="YGN-SAGE" width="128" height="128">
</p>

<h1 align="center">YGN-SAGE</h1>

<p align="center">
  <strong>Self-Adaptive Generation Engine — Agent Development Kit</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/ygn-sage/"><img src="https://img.shields.io/pypi/v/ygn-sage?style=flat-square" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/tests-1980%20passed-brightgreen?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/python-3.12+-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/rust-1.90+-orange?style=flat-square" alt="Rust">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
</p>

---

YGN-SAGE is an open-source Agent Development Kit that **learns** which multi-agent topology to use for each task. Inspired by [OpenSAGE](https://arxiv.org/abs/2602.16891) (UC Berkeley), SAGE extends it with formal verification, evolutionary topology search, and a Rust performance core.

Unlike frameworks that use fixed agent pipelines, SAGE designs, executes, and improves its own multi-agent architectures autonomously — adding **+27pp over bare LLM calls** on multi-agent benchmarks (MASBENCH).

## How a task flows through SAGE

```
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌────────┐     ┌─────────┐     ┌───────┐
│ CLASSIFY │────>│DECOMPOSE │────>│ TOPOLOGY │────>│ ASSIGN │────>│ EXECUTE │────>│ LEARN │
└─────────┘     └──────────┘     └──────────┘     └────────┘     └─────────┘     └───────┘
     │                │                │                │              │              │
  kNN router     TaskPlanner     TopologyEngine    ModelAssigner  TopologyRunner   Bandit +
  (92% GT)       → TaskDAG       6 paths:          per-node       node-by-node    MAP-Elites +
  S1/S2/S3                       S-MMU hit         model from     with provider   Memory +
  routing                        MAP-Elites        cards.toml     resolution      Evolution
                                 LLM synthesis     (7 providers)  + code nodes
                                 Mutation/MCTS                    + adaptation
                                 Path 6: learned
                                 Templates (x8)
```

**Stage 1 — Classify:** The kNN router (92% accuracy on ground truth, [arXiv 2505.12601](https://arxiv.org/abs/2505.12601)) classifies the task into cognitive system S1 (simple), S2 (moderate), or S3 (complex) using arctic-embed-m embeddings.

**Stage 2 — Decompose:** TaskPlanner breaks the task into a dependency DAG with features (complexity, branching, domain).

**Stage 3 — Select Topology:** The DynamicTopologyEngine tries 6 paths in order:
1. **S-MMU retrieval** — find a similar past topology from memory (semantic similarity > 0.7)
2. **MAP-Elites archive** — pick the best elite from the quality-diversity archive
3. **LLM synthesis** — generate a new topology via structured LLM prompt
4. **Mutation** — mutate an existing elite (7 operators: add/remove node, swap model, rewire edge, split/merge, mutate prompt)
5. **MCTS** — Monte Carlo tree search over the mutation space
6. **Path 6: Learned policy** — a trained model (Qwen3-4B local, Nemotron-8B pod) generates the topology via tool-call. Enable with `SAGE_ENABLE_PATH6=1`.
7. **Templates** — 8 pre-wired fallback patterns (sequential, parallel, AVR, debate, hub, hierarchical, selfmoa, brainstorming)

A contextual bandit (Thompson sampling) modulates exploration vs exploitation.

**Stage 4 — Assign Models:** The Rust ModelAssigner scores each candidate model for each node: `0.4 * affinity + 0.4 * domain + 0.2 * (1 - cost)`. Provider hints from the topology can bias selection (+0.15 bonus). 7 providers available: DeepSeek, Google, OpenAI, xAI, Kimi, MiniMax, OpenRouter — each node can use a different provider.

**Stage 5 — Execute:** TopologyRunner executes nodes one by one, respecting the DAG order (predecessors first). After each node:
- **QualityEstimator** evaluates the output (OxiZ formal verification for code, DistilBERT ONNX for text)
- **TopologyController** decides: `continue` (quality > 0.7), `upgrade_model` (quality < 0.3, retry with better model), `prune_node` (skip useless node), `reroute_topology` (rebuild from scratch), or `spawn_subagent`
- **Code nodes** (HyEvo-inspired) are executed in a sandbox instead of LLM calls — offloading deterministic work (validation, parsing, computation) from expensive inference

If a provider fails, the circuit breaker opens and the runner falls back to the next available provider.

**Stage 6 — Learn:** The outcome feeds back into 5 systems:
- **Bandit** updates which topology template/model combination works best
- **MAP-Elites archive** stores the topology if it's a new elite in its behavioral niche
- **Episodic memory** records the full execution trace (SQLite, cross-session)
- **Consolidation** transforms episodic memories into semantic knowledge (every 10 steps)
- **Online evolution** (`should_evolve()` in Rust) triggers topology mutation when enough outcomes accumulate

## 5 Cognitive Pillars

### 1. Topology (Rust + Python)

The core insight: **which agents work together matters more than which model you use** ([AdaptOrch, arXiv 2602.16873](https://arxiv.org/abs/2602.16873): Var_topology / Var_model >= 20).

- **DynamicTopologyEngine** (Rust): 6-path generation with bandit-modulated exploration
- **MAP-Elites** (Rust): 4D quality-diversity archive (agent_count, max_depth, cost, model_diversity) with Pareto insertion
- **CMA-ME** (Rust): covariance matrix adaptation for continuous topology parameters, sigma decay, warm_start
- **MCTS** (Rust): UCB1 tree search over mutation space
- **TopologyGraph** (Rust, petgraph): DAG with 3-flow edges (Control, Message, State) from [MASFactory](https://arxiv.org/abs/2603.06007)
- **TopologySchema** (Python): shared contract between training and runtime — nodes, edges, node_type (llm/code), model_tier, provider_hint, adaptation metadata
- **HyEvo hybrid nodes** ([arXiv 2603.19639](https://arxiv.org/abs/2603.19639)): `node_type="code"` for deterministic sandbox execution (13-19x cost reduction on MBPP)
- **Cascaded evaluation**: 4-stage filtering (schema → security → smoke → full) — 26-87% eval cost savings
- **Reflect-then-generate**: diagnose execution traces → structured recommendations → improved topology candidate
- **Path 6: Learned policy** — Qwen3-4B (local, N1=0.922) or Nemotron-Orchestrator-8B (pod) trained via SFT → GRPO to generate topology via `<tool_call>` JSON

### 2. Tools (Rust + Python)

3-layer defense-in-depth sandbox:
- **tree-sitter** (Rust): AST validation — 23 blocked modules, 21 blocked calls, 14 blocked dunders
- **Wasm WASI** (Rust, wasmtime v43): deny-by-default component model isolation
- **subprocess** (Rust/Python): kill-on-drop timeout fallback, bwrap on Linux

Plus:
- **AgentTool**: `AgentTool.from_agent()` wraps any agent as a callable tool
- **ToolForge**: runtime tool generation — GapDetector identifies missing capabilities, BuildLoop synthesizes new tools
- **Dynamic sub-agent creation** (`agent_mgmt.py`): OpenSAGE-like self-programming

### 3. Memory (Rust + Python)

4-tier hierarchical memory inspired by CoALA (cognitive architecture):
- **Tier 0 — Working Memory** (Rust Arrow): zero-copy columnar STM, SIMD operations, ULID chunk IDs
- **Tier 1 — Episodic Memory** (SQLite): cross-session event log with temporal queries
- **Tier 2 — Semantic Memory**: entity-relation graph with embeddings (arctic-embed-m 768-dim)
- **Tier 3 — ExoCortex** (Google File Search RAG): 500+ research papers, arXiv discovery pipeline

**S-MMU** (Selective Memory Management Unit, Rust): 4-view graph (temporal, semantic, causal, entity) with:
- Utility-based eviction (recency × access_count)
- Auto-GC at 10K chunks
- Composite write gate (5 signals: salience 25%, novelty 30%, reliability 20%, recency 10%, relevance 15%) — [arXiv 2603.15994](https://arxiv.org/abs/2603.15994): 100% precision vs 13% without gate

**Causal wiring**: agent_loop traces tool call → result → decision chains. The causal graph is injected into LLM prompts for informed reasoning. Based on [AMA-Bench 2602.22769](https://arxiv.org/abs/2602.22769): memory fails without causality.

**Consolidation pipeline**: every 10 steps, episodic memories are transformed into semantic relations — the agent equivalent of sleep. Based on [MAGMA 2601.03236](https://arxiv.org/abs/2601.03236): +45.5% reasoning with consolidation.

### 4. Evolution (Rust + Python)

- **MAP-Elites + CMA-ME** (Rust): quality-diversity search over topology space with configurable behavior descriptors
- **Online evolution**: `should_evolve()` in Rust triggers mutation when >5 outcomes accumulate. Topologies improve between tasks, not just during training.
- **7 mutation operators**: add/remove node, swap model, rewire edge, split/merge, mutate prompt
- **LLM-as-mutator**: GPT-based intelligent mutations with AdaptiveMutator (Thompson sampling on mutation operators). Based on [ShinkaEvolve, ICLR 2026](https://arxiv.org/abs/2601.04170).
- **Statistical validation**: Wilcoxon signed-rank test + Cohen's d for evolution significance
- **Drift monitor 12D**: 9 signals (semantic, behavioral, thematic) detect performance degradation. Agent Stability Index from [arXiv 2601.04170](https://arxiv.org/abs/2601.04170).

### 5. Strategy — Cognitive Routing (Rust)

- **S1/S2/S3** cognitive routing (Kahneman dual-process): kNN primary (92% GT), SystemRouter (88% GT)
- **ContextualBandit** (Rust): per-arm Thompson sampling with Beta/Gamma posteriors, Pareto front selection. Based on [PILOT, EMNLP 2025](https://arxiv.org/abs/2508.21141).
- **ModelAssigner** (Rust): per-node model selection with configurable weights + provider hints (+0.15 bonus)
- **ProviderPool**: 7 API providers + Codex, per-node resolution, circuit breaker with auto-failover
- **4-stage cascade routing**: validated by [ETH-SRI ICML 2025](https://arxiv.org/abs/2410.10347) and [routing survey 2603.04445](https://arxiv.org/abs/2603.04445)

## Formal Verification (Rust)

Unique among agent frameworks — no competitor has this:
- **OxiZ SmtVerifier**: sub-0.1ms formal proofs (memory safety, loop bounds, arithmetic, invariants, CEGAR synthesis with 5 refinement rounds)
- **HybridVerifier**: 6 structural + 4 semantic checks on TopologyGraph, O(V+E)
- **LtlVerifier**: 4 temporal property checks (reachability, safety, liveness, bounded liveness)
- **QualityLabeler**: tree-sitter AST + Z3 SMT formal quality scoring
- **TopologyDensity**: S_complex cost metric from [AgentConductor Theorem 1](https://arxiv.org/abs/2602.17100) + N_max bounds
- **DistilBERT QualityEstimator** (ONNX): learned quality prediction for text outputs where formal verification can't apply. Trained on 600 (task, result, quality) triples.

## Training Pipeline

Two parallel training tracks:

### Local (RTX 3500 Ada 12 GB) — Qwen3-4B
```bash
pip install -e ".[training]"
python scripts/train_local_qwen3_4b.py --sft-data data/topology_sft_v2_combined.jsonl
```
- **Phase A SFT**: loss 1.59→0.225, N1=0.865 (tool-call JSON format)
- **Phase C SFT Adaptive**: N1=0.922 (+6.6%, with checkpoints + adapt_topology decisions)
- Stack: bitsandbytes NF4 + PEFT LoRA + TRL (SFTTrainer + GRPOTrainer)
- Model on HF: [yannabadie/sage-topology-policy-local](https://huggingface.co/yannabadie/sage-topology-policy-local)

### Pod (RunPod 2xH100 NVL) — Nemotron-Orchestrator-8B
```bash
bash scripts/verl/train_nemotron_e2e.sh --smoke  # plumbing test
bash scripts/verl/train_nemotron_e2e.sh           # full (~30h)
```
- **SFT warmup**: loss 2.87→1.30
- **Phase A GRPO**: step 1050, reward 0.225 (structural ceiling)
- **DAPO targeted**: in progress
- Stack: verl 0.7.1 + vLLM + Ray + FSDP
- Model on HF: [yannabadie/sage-topology-policy-v2](https://huggingface.co/yannabadie/sage-topology-policy-v2)

Phase A/B = GRPO warm-up (single-turn, learn YAML format). Phase C = GiGPO multi-step with 4-state machine, checkpoints, and micro-decisions (continue/upgrade/reroute).

## Multi-Provider Architecture

All provider configurations are in `sage-python/src/sage/providers/connector.py` (single source of truth):

| Provider | API URL | Default Model | Status |
|----------|---------|---------------|--------|
| DeepSeek | api.deepseek.com/v1 | deepseek-chat | Primary (cheapest, no rate limits) |
| Google | native SDK | gemini-3.1-flash-lite | Reliable fallback |
| OpenAI | api.openai.com/v1 | gpt-5.4 | Best quality |
| xAI | api.x.ai/v1 | grok-4-1-fast-reasoning | Fast reasoning |
| Kimi | api.moonshot.ai/v1 | kimi-k2.5 | Vision + reasoning |
| MiniMax | api.minimax.io/v1 | minimax-m2.7 | 4M token context |
| OpenRouter | openrouter.ai/api/v1 | qwen/qwen3.5-plus | Access to 200+ models |

Each topology node can use a different provider. The policy model can express `provider_hint` to bias selection. The circuit breaker auto-fails over when a provider is down.

## Benchmark Results

| Benchmark | Score | Notes |
|-----------|-------|-------|
| **MASBENCH** | **67%** (vs 40% bare = +27pp) | First empirical proof topology helps |
| **HumanEval+ pipeline** | **89.6%** (147/164) | +5.5pp over pre-pipeline (84.1%) |
| **BigCodeBench Hard Instruct** | **37.8%** (56/148) | Budget model. SOTA 40.0% (The Conductor) |
| **kNN routing GT** | **92%** (46/50) | [arXiv 2505.12601](https://arxiv.org/abs/2505.12601) |
| **TopologyBench** | **94.0%** mean (9/9) | 4.3pp spread across topologies |

### Tests

| Suite | Result |
|-------|--------|
| Python | **1980 passed**, 0 failures |
| Rust | **311 passed** (base), **395+** with smt, tool-executor, onnx, cognitive features |
| Discovery | 95 tests |
| CI | 5 jobs (Rust, Rust features, Python, Discover, Windows) |

## sage-discover — Knowledge Discovery Engine

Autonomous arXiv→ExoCortex pipeline:
- **Discovery**: fetch latest papers by topic, citation graph exploration
- **Extraction**: claims, entities, key phrases from papers
- **Curation**: adaptive ranking with kNN + LinUCB bandit
- **Ingestion**: Qdrant vector DB with SPECTER2 + SPLADE hybrid search
- **ClaimGraph**: OxiZ SMT verification of research claims
- **MCP server**: 5 tools exposing the knowledge base to agents
- **Model watcher**: monitors HuggingFace for new model releases

## Protocols

- **A2A v1.0** (Google Agent-to-Agent): `a2a_server.py` exposes SAGE as an AgentCard with 3 skills (general, code, research). Any A2A client (Google ADK, LangGraph, CrewAI) can delegate tasks.
- **MCP** (Model Context Protocol): `mcp_server.py` exposes SAGE's tool registry to MCP-compatible clients.

## Dashboard (ui/)

FastAPI + WebSocket real-time dashboard:
- Topology DAG visualization (Cytoscape.js + ELK layout, 3-flow edge model)
- Chat with SSE streaming + markdown rendering
- Provider health monitoring (circuit breaker status, latency, cost)
- Event stream (agent lifecycle events)
- Routing pipeline visualization (S1/S2/S3 + kNN scores)

## Quick Start

```bash
# Build
cd sage-core && maturin develop --features smt,onnx,cognitive,tool-executor
cd sage-python && pip install -e ".[all,dev]"

# Test
cd sage-core && cargo test --no-default-features --features smt,tool-executor --lib
cd sage-python && python -m pytest tests/ -v

# Benchmark
python -m sage.bench --type routing_gt
python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 20
python -m sage.bench --type masbench --limit 5

# Training (Nemotron E2E — THE reference command)
pip install -e ".[training]"
bash scripts/verl/train_nemotron_e2e.sh --smoke    # Plumbing test (CPU, <2min)
bash scripts/verl/train_nemotron_e2e.sh             # Full (RunPod H100, ~30h)
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
|-- sage-core/           # Rust core (PyO3): 50 modules, 20K+ lines
|   |-- src/topology/    #   TopologyEngine (6 paths), MAP-Elites, CMA-ME, MCTS
|   |-- src/routing/     #   SystemRouter, kNN, Bandit, ModelAssigner, ModelRegistry
|   |-- src/memory/      #   S-MMU (4-view graph, Arrow, paging, write gate)
|   |-- src/verification/#   OxiZ SMT, LTL, QualityLabeler
|   |-- src/sandbox/     #   Wasm WASI (v43), tree-sitter, subprocess
|   +-- config/cards.toml#   20 model cards, 7 providers
|
|-- sage-python/         # Python SDK: 175 modules, 31K+ lines
|   +-- src/sage/
|       |-- agents/      #   Sequential, Parallel, Loop, Handoff composition
|       |-- bench/       #   15+ benchmarks (BigCodeBench, MASBENCH, GAIA, HumanEval+, etc.)
|       |-- contracts/   #   Z3 verification, TaskDAG, auto-repair
|       |-- events/      #   EventBus (central nervous system)
|       |-- evolution/   #   MAP-Elites population, LLM mutator, self-improve, drift monitor 12D
|       |-- guardrails/  #   Cost, Output, Schema guardrails
|       |-- llm/         #   7 providers (Google, OpenAI, DeepSeek, xAI, Kimi, MiniMax, OpenRouter) + Codex
|       |-- memory/      #   4-tier: STM (Arrow), Episodic (SQLite), Semantic, ExoCortex + consolidator + causal
|       |-- monitoring/  #   Drift detection (12 signals), Agent Stability Index
|       |-- phases/      #   PERCEIVE → THINK → ACT → LEARN (OODA loop)
|       |-- protocols/   #   A2A v1.0, MCP server, unified serve
|       |-- providers/   #   connector.py (single source of truth), OpenAI-compat wrapper
|       |-- strategy/    #   S1/S2/S3 routing, kNN (92%), AdaptiveRouter
|       |-- tools/       #   ToolForge, AgentTool, agent_mgmt, sandbox_executor, gap_detector
|       |-- topology/    #   TopologyRunner (code node dispatch), LLM caller (Path 6 V1/V2), controller
|       |-- verl/        #   Training: topology_env (4-state), reward (5-signal), manifest, cascaded_eval,
|       |                #     reflection, edge_credit, rewardflow, topology_schema (shared contract)
|       |-- pipeline.py  #   5-stage CognitiveOrchestrationPipeline (primary path, legacy fallback exists)
|       +-- boot.py      #   System bootstrap (7 providers auto-detected from .env)
|
|-- sage-discover/       # Knowledge pipeline: 17 modules — arXiv → ExoCortex
|-- sage-router/         # Standalone routing module
|-- ui/                  # FastAPI + WebSocket dashboard (Cytoscape.js topology viz)
|-- docs/                # 60+ specs, 15 audits, 20+ benchmark results
+-- Researches/          # 25+ backing research papers
```

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPSEEK_API_KEY` | At least one | DeepSeek API (cheapest, primary) |
| `GOOGLE_API_KEY` | At least one | Google Gemini |
| `OPENAI_API_KEY` | Optional | OpenAI GPT-5.4 |
| `GROK_API_KEY` | Optional | xAI Grok |
| `KIMI_API_KEY` | Optional | Moonshot Kimi K2.5 |
| `MINIMAX_API_KEY` | Optional | MiniMax |
| `OPEN_ROUTER_API_KEY` | Optional | OpenRouter (200+ models) |
| `SAGE_ENABLE_PATH6` | Optional | Enable learned topology policy |
| `SAGE_VERL_EXEC` | Optional | `1` = execution reward (real API calls) |
| `SAGE_TRAINING_PHASE` | Optional | `A` = simple reward, `C` = enriched reward |

## Research References

| Feature | Paper | Venue |
|---------|-------|-------|
| Architecture inspiration | [OpenSAGE](https://arxiv.org/abs/2602.16891) | ICML 2026 |
| Topology > model | [AdaptOrch](https://arxiv.org/abs/2602.16873) | arXiv 2026 |
| kNN routing | [arXiv 2505.12601](https://arxiv.org/abs/2505.12601) | 2025 |
| Cascade routing | [ETH-SRI](https://arxiv.org/abs/2410.10347) | ICML 2025 |
| Bandit routing | [PILOT](https://arxiv.org/abs/2508.21141) | EMNLP 2025 |
| Routing survey | [6 paradigms](https://arxiv.org/abs/2603.04445) | 2026 |
| 3-flow edges | [MASFactory](https://arxiv.org/abs/2603.06007) | 2026 |
| Density metric | [AgentConductor](https://arxiv.org/abs/2602.17100) | 2026 |
| Conditional topology | [CARD](https://arxiv.org/abs/2603.01089) | ICLR 2026 |
| Hybrid LLM+code nodes | [HyEvo](https://arxiv.org/abs/2603.19639) | 2026 |
| Edge-level credit | [Graph-GRPO](https://arxiv.org/abs/2603.02701) | 2026 |
| GiGPO multi-step | [arXiv 2505.10978](https://arxiv.org/abs/2505.10978) | NeurIPS 2025 |
| Memory consolidation | [MAGMA](https://arxiv.org/abs/2601.03236) | 2026 |
| Write gate | [arXiv 2603.15994](https://arxiv.org/abs/2603.15994) | 2026 |
| Agent stability | [arXiv 2601.04170](https://arxiv.org/abs/2601.04170) | ICLR 2026 |
| Competitor | [The Conductor](https://arxiv.org/abs/2512.04388) | ICLR 2026 |

## License

MIT — see [LICENSE](LICENSE).
