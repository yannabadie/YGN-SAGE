<p align="center">
  <img src="assets/logo.svg" alt="YGN-SAGE" width="128" height="128">
</p>

<h1 align="center">YGN-SAGE</h1>

<p align="center">
  <strong>Verified Adaptive Orchestration Runtime</strong><br>
  <em>Research Preview — multi-agent topology runtime with evidence-gated learning</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/ygn-sage/"><img src="https://img.shields.io/pypi/v/ygn-sage?style=flat-square" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/tests-2940%20Py%20%2B%20549%20Rust-brightgreen?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/status-research%20preview-yellow?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/python-3.12+-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/rust-1.90+-orange?style=flat-square" alt="Rust">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
</p>

---

## What this is (Research Preview)

YGN-SAGE is a **verified adaptive orchestration runtime**: it routes tasks
through cognitive systems (S1/S2/S3), constructs and executes multi-agent
DAG topologies, and adapts at runtime. Crucially, **bandit / MAP-Elites /
online-evolution / training-memory updates are gated by verified
evidence**: a trainable verdict from the OracleStack (default-on since
cycle 7) — never raw output. This is the runtime-integrity layer cycles
5–9 built; it is the differentiating feature, not the marketing tagline.

This is a **Research Preview**, not a production-ready SDK. See the
[Capability State Table](#capability-state-table) below — every claim is
labeled `delivered` / `default-on` / `opt-in` / `planned` / `parked`. If
a capability is `planned`, the runtime path either abstains or falls back;
nothing is silently fabricated.

For full architectural framing see [`AI-ARCHITECTURE.md`](AI-ARCHITECTURE.md)
and the contract docs at `docs/contracts/runtime-integrity-ledger.md`
(8 invariants binding declared labels to verified content).

## Install from source

YGN-SAGE currently requires the Rust `sage_core` extension at runtime.
Until B4 publishes platform wheels, install from a source checkout:

```bash
git clone https://github.com/yannabadie/YGN-SAGE.git
cd YGN-SAGE

python -m pip install maturin

cd sage-core
maturin build --release --features smt,onnx --out target/wheels
python -m pip install target/wheels/sage_core-*.whl --force-reinstall --no-deps

cd ../sage-python
python -m pip install -e ".[all]"
```

One-command pip install ygn-sage is intentionally not advertised until the
sage_core runtime wheel is published or bundled.

## Try it

```python
import asyncio
from sage.boot import boot_agent_system

system = boot_agent_system()
print(asyncio.run(system.run("Write a Python function that checks if a number is prime")))
```

SAGE automatically routes to the right cognitive system (S1/S2/S3), builds a multi-agent topology, assigns models from 7 providers, executes with formal verification of the verifiable fragments (OxiZ SMT for bounded integer arithmetic), and learns from each run **only when the OracleStack emits a `trainable=True` verdict** — runs without verified evidence do not update bandit / MAP-Elites / online-evolution / training-memory.

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
- **QualityEstimator** evaluates the output (OxiZ formal verification for code is the active backend; the DistilBERT ONNX path for text is wired but the model artefact is not shipped — see "DistilBERT QualityEstimator" note below)
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

## 5 Cognitive Pillars (architecture background)

> The five-pillar framing is an **architectural decomposition**, not a
> capability advertisement. Per-capability current state (delivered /
> default-on / opt-in / planned / parked) is in the
> [Capability State Table](#capability-state-table). This section
> describes the design surface; the table tells you what is actually
> wired in the runtime today.

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

## Runtime Architecture — typed execution spine (v0, 2026-04-29)

A 7-cycle arc shipped a typed runtime layer underneath the orchestration pipeline. The OracleStack training-gate is **default-on since cycle 7** (2026-04-29, commit `128e1b89`); the other strategic flags remain opt-in (byte-identical to legacy when unset).

| Layer | Module | Flag | Purpose |
|-------|--------|------|---------|
| RuntimeContracts (cycle 1) | `sage/topology/runner.py` | always-on | Controller single-commit, unified `_run_core`, capability-aware fallback, sandbox fail-closed |
| RuntimeEventLog (cycle 2) | `sage/runtime/event_log/` | `SAGE_TRACE_JSONL_DIR=<path>` (opt-in) | 13 typed events, ULID `run_id`, full SHA-256 envelope hashes, redaction-on by default |
| StateCore (cycle 3) | `sage/runtime/state/` | `SAGE_STATECORE=1` (opt-in) | Control / Message / State edge-channel partitioning, atomic delta reducer |
| RunFrame (cycle 4) | `sage/runtime/run_frame/` | `SAGE_RUN_FRAME=1` (opt-in) | Private builder + public frozen snapshot, allowlisted env capture (8 keys, no wildcard) |
| OracleStack (cycle 5) | `sage/runtime/oracle/` | **default-on (cycle 7); kill-switch `SAGE_ORACLE=0\|false\|off\|no\|disable\|disabled`** | Hierarchical quality verdicts (Exact > Tool > Formal > Spec > LLMJudge > Abstain) — Stage 6 learning ONLY consumes `trainable=True` evidence |
| EvidenceProducers (cycle 6, R6.1a) | `sage/runtime/evidence/` | gated by oracle (default-on) | 6 deterministic producers (tool / test / diff / formal / code-node / planner) emit typed `RuntimeDelta` records consumed by Tool/Formal/Spec v1 oracles |
| Cycle-7 default-on flip | `sage/runtime/oracle/env.py` `oracle_enabled()` | predicate (default-on) | Centralised `SAGE_ORACLE` predicate; replaces 8 scattered `os.environ.get == "1"` checks. cgpro 2026-04-30 VERIFY round-1: forced `controller_decision.payload` is **allowlist-only** (no free-form `reason` leak), operator-friendly kill-switch values (`disable`/`disabled`). |

Hard invariant under default-on: bandit / MAP-Elites / online-evolution / training-memory **never** update from unverified outputs (`verdict.trainable=False` blocks the learning gate). Cycle-7 default-on flip evidence: BCB-Hard N=50 internal pass@1 30% / official Docker 32% / 49/50 = 98% per-task agreement (commit `01b0bb24`); kill-switch smoke (commit `8b4b34b6`) confirms operator escape hatch silences oracle path end-to-end. A14 reset paired with the flip (Posterior epoch=1, old off-policy bandit posteriors discarded).

Detail: [ADR-014..ADR-019](YGN-SAGE/Decisions/) (Obsidian vault), [docs/contracts/runtime-event-log.md](docs/contracts/runtime-event-log.md) (mode-aware contract matrix + golden fixtures).

## Capability State Table

Every notable capability has one of five states. This table is the
single source of truth — section text below cross-references it. Last
updated 2026-05-04 (cycle-10 P3, post cycle-9 closure at HEAD `97fba93f`).

| Capability | State | Evidence / Notes |
|---|---|---|
| **Source install (`maturin develop`)** | `delivered` | Requires Rust toolchain. See [Install from source](#install-from-source). |
| **`pip install ygn-sage` (one command)** | `planned` | Cycle-10 P5 (B4 wheels). Until then, source-only. |
| **Cognitive routing S1/S2/S3 (kNN primary, 92% GT)** | `delivered` | [arXiv 2505.12601](https://arxiv.org/abs/2505.12601), 60-task internal GT. Rust SystemRouter 88% GT secondary. |
| **Topology engine 6-path (S-MMU/archive/LLM/mutation/MCTS/templates)** | `delivered` | Rust `TopologyEngine`, 11 templates fallback. |
| **Multi-provider runtime (7 providers + Codex)** | `delivered` | TTL'd circuit breaker + per-node provider resolution. |
| **OracleStack trainable-evidence gate** | `default-on (cycle 7)` | Commit `128e1b89`. Kill-switch `SAGE_ORACLE=0\|false\|off\|no\|disable\|disabled`. |
| **Runtime integrity ledger (8 invariants)** | `delivered` | `docs/contracts/runtime-integrity-ledger.md`. Every label binds to verified content/schema/provenance/proof. |
| **A14 epoch guard + `topology_state_manifest.json`** | `delivered` | Cycle-8 step 2 (`6b2ebcbe + f9521616`). Fail-closed boot if epoch ≠ DB SHA-256. |
| **Wasm sandbox (RustPython wasm32-wasip1)** | `default-on (cycle 8)` | ADR-013 §5 flip 2026-04-22. `validate_and_execute` deny-by-default. `execute_raw` gated by `SAGE_UNSAFE_RAW_EXEC=1`. |
| **OpenTelemetry GenAI spans (Python + Rust bridge)** | `delivered` | B1 (2026-04-25) + B1.b. `SAGE_OTEL_EXPORTER={none,console,otlp_http,logfire}`. |
| **Cycle-9 bench infrastructure (event ledger + watchdog + keep-awake)** | `delivered` | NDJSON fsync per emit; `HostSuspendDetected` on wall-clock > timeout × grace; Windows `SetThreadExecutionState`. |
| **`SAGE_DANGEROUS_TOOLS` (`execute_bash` register)** | `opt-in (default off, cycle 8)` | Flipped 2026-04-23. Set `SAGE_DANGEROUS_TOOLS=1` only for SWE-bench/research. |
| **Path 6 — learned topology policy** | `opt-in / inference-only on main` | `SAGE_ENABLE_PATH6=1`. Lazy-loads HF checkpoint (`yannabadie/sage-topology-policy-local` 0.922 structural). Training code parked. |
| **ONNX QualityEstimator (`quality_estimator_v2.onnx`)** | `planned, not shipped` | Artifact absent. `_try_load_onnx()` returns None → runtime falls back to OxiZ Z3 labeler or abstains. No invented score. |
| **GiGPO / veRL training (Nemotron-Orchestrator-8B)** | `parked on main (since 2026-04-15)` | Code on dedicated `training` branch (`b2f59ee`, -4.3 GB). Inference-only on main via Path 6. |
| **A3 N=50 ablation (cycle-10 cloud rerun)** | `pending` | A3 morning aborted 2026-05-04 (Modern Standby S0 DRIPS). Recovery infra (event ledger + watchdog + keep-awake) shipped cycle-9. Cycle-10 P8 cloud rerun planned. |
| **`pip install sage-router` standalone** | `planned (decision pending)` | Lib exists at `sage-router/` (1374 LOC), NOT used by canonical runtime. Cycle-10/11 fork: B4 PyPI publish OR fold back into `sage-python/strategy/`. See [`sage-router/README.md`](sage-router/README.md). |
| **A2A v1.0 (Google Agent-to-Agent)** | `delivered` | `a2a_server.py` with `a2a-sdk 0.3.x`. Cancellation TODO. |
| **MCP server** | `delivered` | `mcp_server.py` exposes tool registry. |
| **Dashboard (FastAPI + WebSocket)** | `delivered (preview)` | `ui/app.py` 876 LOC. Not in CI. |
| **BCB Hard Instruct pass rate (full pipe, internal-tuned)** | `measured: 45.9% (68/148)` | NOT a leaderboard submission. See [Benchmark Results](#benchmark-results). |
| **BCB Hard pass rate (budget tier, official Docker)** | `measured: 32% N=50` | Docker-graded, cycle-7 evidence at commit `01b0bb24`. |
| **SWE-bench Lite Docker-graded** | `measured: 10% (1/10) resolved, 70% patch-emit rate` | 2026-04-21. Diff verifier observe-mode opt-in. |
| **`pipeline.py` decomposition** | `planned (cycle 11/12)` | 2983 lines, 44 `Any`. Cycle-10 P9 ships ADR + characterization tests only; refactor deferred. |

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
| DeepSeek | api.deepseek.com/v1 | deepseek-v4-flash | Primary (cheapest, no rate limits) |
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
| **BigCodeBench Hard Instruct (tuned)** | **45.9%** (68/148) | 2026-04-07 v4: pre-filter + reasoner repair + escalation. Above our internal reference SOTA 40.0% (The Conductor). **Not an official BigCodeBench leaderboard submission** — this is an internal tuned run (pre-filter + repair + escalation), not a calibrated Pass@1 following the [leaderboard protocol](https://bigcode-bench.github.io/). AUDIT2 2026-04-24 flagged "above SOTA" framing as unsupported for leaderboard-style claims. |
| **BigCodeBench Hard Instruct (budget)** | **37.8%** (56/148) | Budget model baseline (2026-03). |
| **sage-mas-bench breadth** (our internal suite, NOT the published MAS-Bench arXiv 2509.06477) | **+22pp** p=0.015 | Only statistically significant axis (N=50). Other axes p>0.05. |
| **HumanEval+ pipeline** | **84.1%** (138/164) | The "89.6%" figure previously cited here was an aspirational projection of 84.1% + 5.5pp; it was never actually measured. Saturated benchmark — prefer BCB for framework delta. |
| **kNN routing GT** | **92%** (~46/50 accuracy on 60-task stratified set) | [arXiv 2505.12601](https://arxiv.org/abs/2505.12601). Ground-truth set lives at `sage-python/config/routing_ground_truth.json` with 60 stratified S1/S2/S3 tasks (human-labeled 2026-03-11, criteria in the JSON). The 50-task figure in earlier docs was a stale subset reference — AUDIT2 2026-04-24 confirmed dataset has 60 tasks. Rust SystemRouter 88%. |
| **sage-topo-bench** (our internal topology sweep, NOT the UCL TopologyBench 2024 optical-network dataset) | **94.0%** mean (9/9) | 4.3pp spread across topologies. Distinct from both the optical-network TopologyBench and the TopoBench (arXiv 2603.12133) LLM puzzle benchmark. |
| **SWE-bench Lite** | 10% (1/10) resolved, 40% (4/10) patch-generated | 2026-04-21 v15 Docker-graded smoke after 3-fix chain (Directive #3 gating, CRLF, UTF-8). Gen-rate prior to Docker grading was 70% (patch-produce rate), not pass-rate. |
| **Pre-emission diff-context verifier** | observe-mode instrumentation, opt-in | 2026-04-23: `SAGE_DIFF_VERIFIER_MODE=observe` annotates predictions.jsonl with `_diff_verifier_mismatches` when an emitted hunk's context/removed lines don't match file bytes at the hunk position. First observability smoke caught a parser false-negative (commit `711008a`) — both emitted patches in the N=10 slice flagged `content_mismatch` post-fix, zero false positives. Repair mode deferred until ≥10 clean + ≥10 flagged observations accumulate. See `docs/superpowers/specs/2026-04-23-diff-context-verifier-design.md`. |

### Tests

Live counts canonicalized at `docs/status/current.json` (regenerated via `python scripts/status_snapshot.py`). The numbers below reflect the snapshot at last update; consult the JSON for current-run truth.

| Suite | Result |
|-------|--------|
| Python (`sage-python`) | **2940 collected** (cycle-9 closure + cgpro round-2 telemetry fix, source of truth = `docs/status/current.json`). Pre-existing 8 fail + 2 error in `test_e2e_*` / `test_pydantic_ai_integration.py` — API-key-gated fixtures. |
| Rust (`sage-core`) | **549 listed** with `--features smt,cognitive,sandbox,cranelift,tool-executor`. `sandbox`/`cranelift`/`tool-executor`/`cognitive` are Cargo default features (ADR-013 §5 flip). |
| Discovery (`sage-discover`) | **100 collected** |
| CI | `.github/workflows/ci.yml` runs 10 jobs (`build-wasm-sandbox`, `python-constraints`, `rust`, `otel-bridge`, `otel-bridge-windows`, `rust-features`, `python-sage`, `python-discover`, `windows-pytest`, `integration-smoke`) + 3 supporting workflows (`security.yml` pip-audit/SBOM, `latest-deps.yml` weekly drift, `stochastic-empirical.yml` scheduled). |
| Static analysis | mypy 0 errors; ruff clean (per cycle-8 R6.1c + A14 closure verification) |

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
|       |-- runtime/     #   2026-04-29 typed runtime spine (event_log, state, run_frame, oracle)
|       |-- strategy/    #   S1/S2/S3 routing, kNN (92%), AdaptiveRouter
|       |-- tools/       #   ToolForge, AgentTool, agent_mgmt, sandbox_executor, gap_detector
|       |-- topology/    #   TopologyRunner (code node dispatch), LLM caller (Path 6 V1/V2), controller
|       |-- verl/        #   Training: topology_env (4-state), reward (5-signal), manifest, cascaded_eval,
|       |                #     reflection, edge_credit, rewardflow, topology_schema (shared contract)
|       |-- pipeline.py  #   5-stage CognitiveOrchestrationPipeline (primary path, legacy fallback exists)
|       +-- boot.py      #   System bootstrap (7 providers auto-detected from .env)
|
|-- sage-discover/       # Knowledge pipeline: 17 modules — arXiv → ExoCortex (adjunct, depends on ygn-sage)
|-- sage-router/         # Standalone routing-only package (1374 LOC, NOT used by sage-python runtime — see sage-router/README.md)
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

## Observability

YGN-SAGE emits OpenTelemetry GenAI spans on the orchestration path.
Default off; opt in with `SAGE_OTEL_EXPORTER=console` for stdout
debug or `otlp_http` to ship to a collector. Full doc:
`docs/observability/otel-genai-spans.md`.

## License

MIT — see [LICENSE](LICENSE).
