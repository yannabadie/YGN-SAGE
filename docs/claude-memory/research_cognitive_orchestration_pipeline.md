---
name: Cognitive Orchestration Pipeline Research
description: Research findings on model-per-role assignment, dynamic topology adaptation, and capability-aware routing for the end-to-end cognitive pipeline
type: project
---

## Cognitive Orchestration Pipeline — Research Synthesis (2026-03-14)

**Why:** The user identified that ModelCards (17 models × 22 fields × 7 domain_scores) are not wired into per-node model assignment in topologies. This is the #1 credibility gap for SAGE as a multiprovider+multiagent ADK.

**How to apply:** Use these findings to design and implement the 5-stage Cognitive Orchestration Pipeline (Classify → Decompose → Select Topology → Assign Models Per Node → Execute+Adapt → Learn).

### Key Research References

| Paper | Venue | Key Technique for SAGE |
|-------|-------|----------------------|
| **OpenSage** (2602.16891) | ICML | AI chooses model_name per sub-agent at runtime. Vertical+horizontal topology switching. 59% SWE-Bench Pro |
| **AdaptOrch** (2602.16873) | arXiv 2026 | Var_topology/Var_model ≥ 20. DAG features (ω,δ,γ) for topology routing. Adaptive re-routing on consistency failure |
| **OFA-MAS** (2601.12996) | WWW 2026 | Per-node LLM_i formalization. MoE graph generation. Universal role pool. 93% across 6 benchmarks |
| **SYMPHONY** (2601.22623) | NeurIPS 2025 | UCB scheduling on heterogeneous LLM pool at each MCTS node |
| **AgentDropout** (2503.18891) | ACL 2025 | Runtime pruning of agents/edges per round. -21.6% prompt tokens, +1.14 perf |
| **ARG-Designer** (2507.18224) | AAAI 2026 Oral | Autoregressive graph generation (node+edge generators). Best on 6 benchmarks |
| **Router-R1** (2506.09033) | NeurIPS 2025 | LLM-as-router with multi-round reasoning about which models to invoke |
| **DiSRouter** (2510.19208) | arXiv | Decentralized self-routing: each LLM self-assesses competence |
| **xRouter** (2510.08439) | Salesforce | RL-trained router for 20+ LLMs with cost-aware reward |
| **Cascade Routing** (2410.10347) | ICML 2025 ETH | Quality estimators > routing algorithms. Validates DistilBERT QE |
| **OrchMAS** (2603.03005) | arXiv 2026 | Two-tier orchestration: orchestration model + execution model per step |
| **TopologyStructureLearning** (2505.22467) | arXiv | Three-stage framework. Cost-aware: M* = argmax R(x,M) - λ·C(M) |
| **Foundation Agents** (2504.01990v2) | Survey | DyLAN agent importance scoring. PIANO slow/fast cognitive layers |
| **DyLAN** (2310.02170) | arXiv | Agent Importance Score + team reformation during execution |
| **DRTAG** (Frontiers 2025) | Journal | Real-time agent creation/integration during execution |
| **GTD** (2510.07799) | arXiv | Graph diffusion for topology synthesis with proxy reward model |
| **MASEval** (2603.08835) | arXiv 2026 | Framework choice matters as much as model choice |

### Architecture Consensus

1. **Per-node model assignment is mainstream** (OpenSage, OrchMAS, SYMPHONY, OFA-MAS, xRouter)
2. **Runtime topology adaptation is validated** (AgentDropout, DyLAN, DRTAG, AdaptOrch, SYMPHONY)
3. **Quality estimation > routing algorithm** (ETH-SRI ICML 2025)
4. **Topology > model selection** when models converge (AdaptOrch: Var_tau/Var_M ≥ 20)
5. **Three-stage pattern converges independently**: Agent Selection → Structure Design → Configuration (OFA-MAS, MASFactory, TopologyStructureLearning)

### Proposed 5-Stage Pipeline

1. **CLASSIFY** — ComplexityRouter S1/S2/S3 + domain hint (kNN 92%)
2. **DECOMPOSE** — ContractPlanner → TaskDAG + DAG features (ω,δ,γ from AdaptOrch)
3. **SELECT TOPOLOGY** — DynamicTopologyEngine + new Path 0 f(ω,δ,γ)→macro-type
4. **ASSIGN MODELS PER NODE** — ModelRegistry.select_best_for_domain() + ContextualBandit + CostTracker
5. **EXECUTE+ADAPT** — TopologyExecutor with feedback controller: model upgrade, agent pruning, topology re-route, sub-agent spawn

### 4 Runtime Adaptation Actions

| Action | Trigger | Source Paper |
|--------|---------|-------------|
| Model upgrade per node | QualityEstimator < θ | Self-Regulation (2502.04576) |
| Agent pruning | Importance score < seuil | AgentDropout (ACL 2025) |
| Topology re-route | ConsistencyScore < θ | AdaptOrch (2602.16873) |
| Sub-agent spawn | Emergent sub-task detected | OpenSage (ICML), DRTAG |
