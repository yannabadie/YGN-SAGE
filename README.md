<p align="center">
  <img src="assets/logo.svg" alt="YGN-SAGE" width="128" height="128">
</p>

<h1 align="center">YGN-SAGE</h1>

<p align="center">
  <strong>Self-Adaptive Generation Engine — Agent Development Kit</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/ygn-sage/"><img src="https://img.shields.io/pypi/v/ygn-sage?style=flat-square" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/tests-1999%20Py%20%2B%20496%20Rust-brightgreen?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/python-3.12+-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/rust-1.90+-orange?style=flat-square" alt="Rust">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
</p>

---

## Try it now

```python
pip install ygn-sage
```

```python
import asyncio
from sage.boot import boot_agent_system

system = boot_agent_system()
print(asyncio.run(system.run("Write a Python function that checks if a number is prime")))
```

SAGE automatically routes to the right cognitive system (S1/S2/S3), builds a multi-agent topology, assigns models from 7 providers, executes with formal verification, and learns from every run.

---

YGN-SAGE is an open-source Agent Development Kit that **learns** which multi-agent topology to use for each task. Inspired by [OpenSAGE](https://arxiv.org/abs/2602.16891) (UC Berkeley), SAGE extends it with formal verification, evolutionary topology search, and a Rust performance core.

Unlike frameworks that use fixed agent pipelines, SAGE designs, executes, and improves its own multi-agent architectures autonomously. On our internal multi-agent suite (`sage-mas-bench`, inspired by but NOT the published MAS-Bench of arXiv 2509.06477 or the AAMAS 2021 crowd-simulation benchmark — distinct from both) a statistically-significant +22pp gain on the *breadth* axis (p=0.015, N=50) is reported; other axes show p>0.05, not significant. Earlier "+27pp" / "+30pp" headlines have been retracted (see [2026-04-22 audit verification](docs/audits/2026-04-22-audit-verification-master.md) and commit 796af27).

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
7. **Templates** — 11 pre-wired fallback patterns (sequential, parallel, AVR, debate, hub, hierarchical, selfmoa, brainstorming, robust, horizon_pipeline, parallel_fanout)

Before the 6-path engine, `select_macro_topology()` uses DAG structural features (omega=parallelism, delta=depth, gamma=coupling) to select specialized templates: deep chains → `horizon_pipeline`, wide parallel → `parallel_fanout`, coupled parallel → `robust` (majority voting).

A contextual bandit (Thompson sampling) modulates exploration vs exploitation.

**Stage 4 — Assign Models:** The Rust ModelAssigner scores each candidate model for each node: `0.4 * affinity + 0.4 * domain + 0.2 * (1 - cost)`. Provider hints from the topology can bias selection (+0.15 bonus). 7 providers available: DeepSeek, Google, OpenAI, xAI, Kimi, MiniMax, OpenRouter — each node can use a different provider. The bandit's learned quality priors override assignments for underperforming models (quality < 0.4).

**Stage 5 — Execute:** TopologyRunner executes nodes in DAG order with adaptive context — each node receives predecessor outputs sized to the model's context window (no fixed truncation). Near-identical outputs from parallel workers are deduplicated via Jaccard similarity gate (S2-MAD, -94% tokens). After each node:
- **QualityEstimator** evaluates the output (OxiZ formal verification for code, DistilBERT ONNX for text)
- **TopologyController** decides: `continue` (quality > 0.7), `upgrade_model` (quality < 0.3, retry with better model), `prune_node` (skip useless node), `reroute_topology` (rebuild from scratch), `spawn_subagent`, or `open_gate` (re-execute a node for multi-turn refinement, max 3 rounds)
- **Code nodes** (HyEvo-inspired) are executed in a sandbox instead of LLM calls
- **Arithmetic verification** catches calculation errors and triggers model upgrade
- **HITL callback** (optional) pauses for human approval on disruptive actions
- **Per-node streaming** via `run_stream()` yields events as each node completes

If a provider fails, the circuit breaker opens and the runner falls back to the next available provider.

**Stage 6 — Learn:** The outcome feeds back into 5 systems:
- **Bandit** updates which topology template/model combination works best (persisted to SQLite, feeds back into Stage 4)
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

**S-MMU** (Structured Memory Multi-view graph, Rust — not an OS MMU; naming clarified per 2026-04-22 audit): 4-view graph (temporal, semantic, causal, entity) with:
- Utility-based eviction (recency × access_count)
- Auto-GC at 10K chunks
- Composite write gate (5 signals: salience 25%, novelty 30%, reliability 20%, recency 10%, relevance 15%) — [arXiv 2603.15994](https://arxiv.org/abs/2603.15994): 100% precision vs 13% without gate

**Causal wiring**: agent_loop traces tool call → result → decision chains. The causal graph is injected into LLM prompts for informed reasoning. Based on [AMA-Bench 2602.22769](https://arxiv.org/abs/2602.22769): memory fails without causality.

**Consolidation pipeline**: every 10 steps, episodic memories are transformed into semantic relations — the agent equivalent of sleep. Based on [MAGMA 2601.03236](https://arxiv.org/abs/2601.03236): +45.5% reasoning with consolidation.

### 4. Evolution (Rust + Python)

- **MAP-Elites + CMA-ME** (Rust): quality-diversity search over topology space with configurable behavior descriptors
- **Online evolution**: `should_evolve()` in Rust triggers mutation when >5 outcomes accumulate. Topologies improve between tasks, not just during training.
- **7 mutation operators**: add/remove node, swap model, rewire edge, split/merge, mutate prompt
- **LLM-as-mutator**: GPT-based intelligent mutations with AdaptiveMutator (Thompson sampling on mutation operators). Based on [ShinkaEvolve, arXiv 2509.19349](https://arxiv.org/abs/2509.19349).
- **Statistical validation**: Wilcoxon signed-rank test + Cohen's d for evolution significance
- **Drift monitor 12D**: 9 signals (semantic, behavioral, thematic) detect performance degradation. Agent Stability Index from [Agent Drift, arXiv 2601.04170](https://arxiv.org/abs/2601.04170).

### 5. Strategy — Cognitive Routing (Rust)

- **S1/S2/S3** cognitive routing (Kahneman dual-process): kNN primary (92% GT on our internal 50-task set; the [backing research arXiv 2505.12601](https://arxiv.org/abs/2505.12601) validates kNN as a viable router class, reporting 52-77% AUC on public RouterBench/AlpacaEval), SystemRouter (88% GT on the same set)
- **ContextualBandit** (Rust): per-arm Thompson sampling with Beta/Gamma posteriors, Pareto front selection. General principle — bandit must learn from actual quality signal — is from the [Cascade Routing ETH-SRI ICLR 2025](https://arxiv.org/abs/2410.10347) line of work.
- **ModelAssigner** (Rust): per-node model selection with configurable weights + provider hints (+0.15 bonus)
- **ProviderPool**: 7 API providers + Codex, per-node resolution, circuit breaker with auto-failover
- **4-stage cascade routing**: validated by [ETH-SRI ICLR 2025](https://arxiv.org/abs/2410.10347) and [routing survey 2603.04445](https://arxiv.org/abs/2603.04445)

## Formal Verification (Rust)

Verification components (scope and framing corrected 2026-04-22 per audit — some competitors have structural workflow verification, though SMT-grounded component verification for agent runtimes is not standard):
- **OxiZ SmtVerifier**: QF_LIA SMT solving for verifiable fragments — memory safety (bounds with literal integers), loop bounds (concrete caps), arithmetic (constants + linear expressions), invariants (pre → post implication). Solve time is sub-millisecond on the fragments we verify; claim is scoped, not a general-case performance promise.
- **Invariant synthesis**: candidate enumeration with simple syntactic weakening (`>` → `>=`). This is **not** Counter-Example Guided Abstraction Refinement; the previous "CEGAR" label was inaccurate and is removed.
- **HybridVerifier**: 6 structural + 4 semantic checks on TopologyGraph, O(V+E)
- **LtlVerifier** (graph-property checker): reachability (BFS source→target), safety (no HIGH→LOW edges), liveness (entries reach exits by BFS), bounded liveness (DFS with depth cap). This is not LTL model checking in the Büchi-automaton sense — no formula parser, no temporal operators. Name retained for API compat; the 2026-04-22 audit recommends renaming to `GraphPropertyChecker` in a future breaking change.
- **QualityLabeler** (hybrid, **not zero-heuristics**): tree-sitter AST formal parsing + OxiZ SMT for arithmetic assertions, combined with heuristic checks for structural completeness, code-block extraction, and answer-pattern matching. The audit-verified ratio is ~80% heuristic / ~20% formal; the earlier "zero heuristics" claim was incorrect.
- **TopologyDensity**: S_complex cost metric from [AgentConductor Theorem 1](https://arxiv.org/abs/2602.17100) + N_max bounds
- **DistilBERT QualityEstimator** (ONNX) — *planned, not shipped*: this learned quality predictor was planned to complement the Rust formal path but the ONNX artifact is not in this repo or on a release. Current runtime uses the Rust QualityLabeler + abstention; 600-triple training is a future item.

## Training Pipeline

Training code (veRL integration, SFT/GRPO scripts, datasets, checkpoints) was moved out of `main` on 2026-04-15 (commit `b2f59ee`, ~4.3 GB removed). Focus shifted to orchestration correctness and agentic benchmarks. The code lives in a dedicated training branch; revive it there when needed.

Trained checkpoints remain available on HuggingFace:
- [yannabadie/sage-topology-policy-local](https://huggingface.co/yannabadie/sage-topology-policy-local) — Qwen3-4B (Phase C: 0.922 structural, 40% on our internal sage-mas-bench depth — **best local model**)
- [yannabadie/sage-topology-policy-v2](https://huggingface.co/yannabadie/sage-topology-policy-v2) — Nemotron-Orchestrator-8B (veRL pod checkpoints)

Path 6 (learned topology policy) is currently off by default. To use a trained model:
- Enable via `SAGE_ENABLE_PATH6=1`, point to your local checkpoint, and install the training extras in a dedicated environment.
- See the training branch for SFT/GRPO scripts and the `V2 GRPO lessons` note (avoid `environment_factory`, use plain `reward_funcs`).

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
| **BigCodeBench Hard Instruct (tuned)** | **45.9%** (68/148) | 2026-04-07 v4: pre-filter + reasoner repair + escalation. Above SOTA 40.0% (The Conductor). |
| **BigCodeBench Hard Instruct (budget)** | **37.8%** (56/148) | Budget model baseline (2026-03). |
| **sage-mas-bench breadth** (our internal suite, NOT the published MAS-Bench arXiv 2509.06477) | **+22pp** p=0.015 | Only statistically significant axis (N=50). Other axes p>0.05. |
| **HumanEval+ pipeline** | **84.1%** (138/164) | The "89.6%" figure previously cited here was an aspirational projection of 84.1% + 5.5pp; it was never actually measured. Saturated benchmark — prefer BCB for framework delta. |
| **kNN routing GT** | **92%** (46/50) | [arXiv 2505.12601](https://arxiv.org/abs/2505.12601). Rust SystemRouter 88%. |
| **sage-topo-bench** (our internal topology sweep, NOT the UCL TopologyBench 2024 optical-network dataset) | **94.0%** mean (9/9) | 4.3pp spread across topologies. Distinct from both the optical-network TopologyBench and the TopoBench (arXiv 2603.12133) LLM puzzle benchmark. |
| **SWE-bench Lite** | 10% (1/10) resolved, 40% (4/10) patch-generated | 2026-04-21 v15 Docker-graded smoke after 3-fix chain (Directive #3 gating, CRLF, UTF-8). Gen-rate prior to Docker grading was 70% (patch-produce rate), not pass-rate. |

### Tests

| Suite | Result |
|-------|--------|
| Python | **1999 passed** (2026-04-22 P0.4 B +41: 40 red-team attacks + 1 fixed regression; 11 pre-existing failures are API-key-dependent) |
| Rust | **496 passed** with `smt` feature (2026-04-22 P0.4 B +16: 8 `wasm_python` + 8 structural sandbox tests). `sandbox`, `cranelift`, `tool-executor`, `cognitive` are now Cargo default features (ADR-013 §5 flip). |
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
```

Training commands live on the training branch — see the [Training Pipeline](#training-pipeline) section.

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
|       |-- bench/       #   15+ benchmarks (BigCodeBench, sage-mas-bench internal, GAIA, HumanEval+, etc.)
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
| Architecture inspiration | [OpenSAGE](https://arxiv.org/abs/2602.16891) | arXiv 2026 (ICML 2026 submission; notification Apr 30 2026) |
| Topology > model | [AdaptOrch](https://arxiv.org/abs/2602.16873) | arXiv 2026 |
| kNN routing | [arXiv 2505.12601](https://arxiv.org/abs/2505.12601) | 2025 |
| Cascade routing | [ETH-SRI](https://arxiv.org/abs/2410.10347) | ICLR 2025 |
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
