# SOTA Research: Multi-Tier Agent Memory & Evolutionary Topology Optimization

**Date:** March 30, 2026
**Scope:** Papers from 2025-2026 directly applicable to YGN-SAGE memory-evolution system

---

## 1. Multi-Tier Agent Memory (Working/Episodic/Semantic/Causal)

### 1.1 Hierarchical Memory Architectures

**MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents**
- arXiv: 2601.03236 | Jan 6, 2026
- Authors: Dongming Jiang, Yi Li, Guanpeng Li, Bingzhe Li
- **Key contribution:** Represents each memory item across 4 orthogonal graphs: semantic, temporal, causal, and entity. Retrieval is policy-guided traversal over relational views, enabling query-adaptive selection. Decouples memory representation from retrieval logic.
- **Results:** 45.5% higher reasoning accuracy on long-context benchmarks, 95% token reduction, 40% faster query latency.
- **SAGE relevance:** CRITICAL. MAGMA's 4-graph architecture maps directly to S-MMU's 4-tier design (working/episodic/semantic/causal). The policy-guided traversal approach could replace or augment our current retrieval.

**MemoryOS: Memory OS of AI Agent**
- arXiv: 2506.06326 | May 30, 2025 | EMNLP 2025 Oral
- Authors: Jiazheng Kang, Mingming Ji, Zhe Zhao, Ting Bai
- **Key contribution:** OS-inspired 3-tier hierarchy (STM/MTM/LTM). Segmented page organization with heat-based eviction. STM-to-MTM follows dialogue-chain FIFO, MTM-to-LTM uses segmented paging.
- **Results:** +49.11% F1, +46.18% BLEU-1 over baselines on LoCoMo.
- **SAGE relevance:** HIGH. Heat-based eviction maps to S-MMU paging. Segmented page strategy is directly implementable in our Arrow-tier storage.

**EverMemOS: Self-Organizing Memory Operating System**
- arXiv: 2601.02163 | Jan 5, 2026
- Authors: Chuanrui Hu et al.
- **Key contribution:** Engram-inspired lifecycle: (1) Episodic Trace Formation (dialogue -> MemCells with episodic traces, atomic facts, foresight signals), (2) Semantic Consolidation (MemCells -> thematic MemScenes), (3) Reconstructive Recollection (MemScene-guided retrieval).
- **SAGE relevance:** HIGH. The 3-phase lifecycle (episodic -> semantic consolidation -> recollection) is exactly the consolidation pipeline we need for S-MMU write gates.

**TiMem: Temporal-Hierarchical Memory Consolidation**
- arXiv: 2601.02845 | Jan 6, 2026
- Authors: TiMEM-AI team
- **Key contribution:** Temporal Memory Tree (TMT) organizing conversations through systematic consolidation from raw observations to progressively abstracted persona representations. Complexity-aware memory recall.
- **Results:** 75.30% on LoCoMo, 76.88% on LongMemEval-S, 52.20% memory length reduction.
- **SAGE relevance:** MEDIUM. The TMT structure could inform our episodic-to-semantic consolidation pathway.

**Continuum Memory Architectures for Long-Horizon LLM Agents**
- arXiv: 2601.09913 | Jan 14, 2026
- Authors: Joe Logan et al.
- **Key contribution:** Formalizes CMA as an architectural class with lifecycle: ingest -> activation -> retrieval -> consolidation. Multi-resolution graphs where coarse clusters handle most queries, fine-grained nodes activate on demand.
- **SAGE relevance:** MEDIUM. The multi-resolution graph approach maps to our hierarchical S-MMU tiers.

**Hindsight: Agent Memory that Retains, Recalls, and Reflects**
- arXiv: 2512.12818 | Dec 14, 2025
- Authors: Chris Latimer et al.
- **Key contribution:** 4 logical networks: world facts, agent experiences, synthesized entity summaries, and evolving beliefs. Three core operations: retain, recall, reflect. Temporal + entity-aware memory layer.
- **Results:** 83.6% accuracy (from 39% baseline) on LongMemEval; 91.4% with scaled backbone.
- **SAGE relevance:** HIGH. The 4-network architecture is a close match to our 4-tier model. The retain/recall/reflect operations map to our write/read/consolidation pipeline.

**H-MEM: Hierarchical Memory for High-Efficiency Long-Term Reasoning**
- arXiv: 2507.22925 | Jul 23, 2025
- Authors: Haoran Sun, Shaoning Zeng
- **Key contribution:** Multi-level memory organized by degree of semantic abstraction. Each memory vector has positional index encoding pointing to sub-memories in next layer. Index-based routing for efficient layer-by-layer retrieval.
- **SAGE relevance:** MEDIUM. The positional index encoding concept could improve our S-MMU retrieval efficiency.

### 1.2 Causal Memory & Reasoning

**AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications**
- arXiv: 2602.22769 | Feb 26, 2026
- Authors: Yujie Zhao et al. (Meta/UCSD)
- **Key contribution:** Shows existing memory systems fail because they lack causality and objective information. Proposes AMA-Agent with causality graph + tool-augmented retrieval. Real-world agentic trajectory evaluation.
- **Results:** 57.22% average accuracy, +11.16% over strongest baseline.
- **SAGE relevance:** HIGH. Validates our decision to include a causal memory tier. The causality graph approach should inform our causal.py implementation.

**ActMem: Bridging Memory Retrieval and Reasoning in LLM Agents**
- arXiv: 2603.00026 | Feb 4, 2026
- Authors: Xiaohui Zhang et al.
- **Key contribution:** Transforms unstructured dialogue into structured causal + semantic graph. Uses counterfactual reasoning and commonsense completion for implicit constraint deduction and conflict resolution.
- **SAGE relevance:** HIGH. The counterfactual reasoning approach for conflict detection is directly applicable to our causal memory tier.

**Language Agents Meet Causality**
- arXiv: 2410.19923 | Oct 25, 2024
- Authors: John Gkountouras et al.
- **Key contribution:** Integrates causal representation learning with LLMs. Causal world model acts as simulator that LLM can query. Variables linked to natural language expressions.
- **SAGE relevance:** MEDIUM. Foundational work for causal memory implementation.

### 1.3 Write Gates & Memory Consolidation

**Selective Memory: Write-Time Gating with Hierarchical Archiving**
- arXiv: 2603.15994 | Mar 16, 2026
- Authors: Oliver Zahn, Simran Chana
- **Key contribution:** Filters incoming knowledge using composite salience scores (source reputation, novelty, reliability). Objects below threshold go to cold storage, not deleted. Version chains preserve prior states.
- **Results:** 100% accuracy vs 13% ungated at 8:1 distractor ratio. Self-RAG collapses to 0% at same ratio. 1/9th query-time cost.
- **SAGE relevance:** CRITICAL. This is exactly the write-gate mechanism we need. The 3-factor salience scoring (reputation, novelty, reliability) maps directly to our write_gate.py. Cold storage archiving aligns with our Arrow-tier.

**CraniMem: Cranial Inspired Gated and Bounded Memory**
- arXiv: 2603.15642 | Mar 3, 2026
- Authors: Pearl Mody et al.
- **Key contribution:** Goal-conditioned gating + utility tagging. Bounded episodic buffer for near-term continuity + structured long-term knowledge graph for semantic recall. Scheduled consolidation loop replays high-utility traces into graph while pruning low-utility items.
- **SAGE relevance:** HIGH. Goal-conditioned gating and utility tagging are directly applicable to our write gate. The bounded buffer concept maps to our working memory capacity limits.

**D-MEM: Dopamine-Gated Agentic Memory via Reward Prediction Error Routing**
- arXiv: 2603.14597 | Mar 15, 2026
- Authors: Yuru Song, Qi Xin
- **Key contribution:** Biologically-inspired RPE (Reward Prediction Error) gating. Lightweight Critic Router assesses Surprise and Utility. Low-RPE inputs bypass to O(1) fast buffer. High-RPE inputs trigger full O(N) memory evolution pipeline. Reduces token consumption 80%.
- **SAGE relevance:** HIGH. The RPE-based gating (surprise + utility) could enhance our write gate. The fast/slow dual-path routing is directly applicable to S-MMU.

**FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory**
- arXiv: 2601.18642 | Jan 26, 2026
- Authors: Lei Wei et al.
- **Key contribution:** Differential decay rates across dual-layer hierarchy. Adaptive exponential decay modulated by semantic relevance, access frequency, and temporal patterns. LLM-guided conflict resolution + memory fusion.
- **Results:** 45% storage reduction with superior multi-hop reasoning.
- **SAGE relevance:** MEDIUM. The decay function (relevance x frequency x recency) could inform our set_decay_factor() implementation.

**D-Mem: A Dual-Process Memory System**
- arXiv: 2603.18631 | Mar 19, 2026
- Authors: Zhixing You, Jiachen Yuan, Jason Cai
- **Key contribution:** Fast vector retrieval for routine queries + full deliberation for complex requests. Multi-dimensional Quality Gating policy decides which process to use.
- **Results:** F1 53.5 on LoCoMo, recovering 96.7% of full deliberation performance.
- **SAGE relevance:** MEDIUM. The dual-process gating concept aligns with our working vs episodic memory routing.

**MemMA: Coordinating the Memory Cycle through Multi-Agent Reasoning**
- arXiv: 2603.18718 | Mar 19, 2026
- Authors: Minhua Lin et al.
- **Key contribution:** Meta-Thinker produces structured guidance for memory construction/retrieval. In-situ self-evolving memory: synthesizes probe QA pairs, verifies current memory, converts failures into repair actions.
- **SAGE relevance:** MEDIUM. The self-evolving memory verification loop could inform quality assurance of stored memories.

**AgeMem: Agentic Memory - Unified Long-Term and Short-Term Management**
- arXiv: 2601.01885 | Jan 5, 2026
- Authors: Yi Yu et al.
- **Key contribution:** Exposes memory operations (store, retrieve, update, summarize, discard) as tool-based actions. 3-stage progressive RL training with step-wise GRPO for sparse rewards.
- **SAGE relevance:** MEDIUM. The tool-based memory action approach and GRPO training are relevant to our training pipeline.

**MEM1: Learning to Synergize Memory and Reasoning**
- arXiv: 2506.15841 | Jun 18, 2025
- Authors: Zijian Zhou et al. (MIT)
- **Key contribution:** End-to-end RL framework for constant-memory operation. Compact shared internal state for joint memory consolidation and reasoning. 3.5x performance improvement, 3.7x memory reduction vs Qwen2.5-14B.
- **SAGE relevance:** MEDIUM. The reasoning-driven consolidation approach could inform our semantic memory layer.

### 1.4 Surveys & Benchmarks

**The AI Hippocampus: How Far are We From Human Memory?**
- arXiv: 2601.09113 | Jan 14, 2026
- **Key contribution:** Comprehensive taxonomy: implicit, explicit, and agentic memory paradigms. Covers cross-modal memory integration.

**Anatomy of Agentic Memory: Taxonomy and Empirical Analysis**
- arXiv: 2602.19320 | Feb 22, 2026
- **Key contribution:** 4 memory structures taxonomy. Identifies benchmark saturation, backbone-dependent accuracy, and latency overhead as key pain points.

**MemoryAgentBench: Evaluating Memory in LLM Agents**
- arXiv: 2507.05257 | Jul 7, 2025
- **Key contribution:** 4 core competencies: accurate retrieval, test-time learning, long-range understanding, conflict resolution.

**SEEM: Structured Episodic Event Memory**
- arXiv: 2601.06411 | Jan 10, 2026
- **Key contribution:** Graph memory layer (relational facts) + dynamic episodic memory layer (narrative progression). Episodic Event Frames with provenance pointers. Reverse Provenance Expansion for context reconstruction.
- **SAGE relevance:** HIGH. The EEF concept with provenance pointers is directly applicable to our episodic memory implementation.

---

## 2. MAP-Elites & Quality-Diversity for LLM/Agent Optimization

### 2.1 Core Frameworks

**GigaEvo: Open Source Optimization Framework (LLMs + Evolution)**
- arXiv: 2511.17592 | Nov 17, 2025
- Authors: Valentin Khrulkov et al. (AIRI Institute)
- **Key contribution:** Open-source AlphaEvolve reproduction. Modular MAP-Elites QD algorithms, async DAG-based evaluation, LLM-driven mutation with insight generation + bidirectional lineage tracking, multi-island strategies.
- **Results:** Reproduces AlphaEvolve results on Heilbronn triangles, circle packing, kissing numbers.
- **SAGE relevance:** CRITICAL. GigaEvo's architecture is the reference implementation for our evolution engine. Multi-island MAP-Elites + LLM mutation + lineage tracking maps directly to our design. Already referenced in project memory.

**ShinkaEvolve: Open-Ended And Sample-Efficient Program Evolution**
- arXiv: 2509.19349 | Sep 17, 2025 | ICLR 2026
- Authors: Robert Tjarko Lange, Yuki Imajuku, Edoardo Cetin (Sakana AI)
- **Key contribution:** 3 innovations: (1) Novel parent sampling balancing exploration/exploitation, (2) Code novelty rejection-sampling for efficient search, (3) Bandit-based adaptive LLM ensemble selection. Outperforms AlphaEvolve on circle packing.
- **SAGE relevance:** HIGH. The bandit-based LLM ensemble selection is directly applicable to our mutator model selection. Rejection-sampling for novelty maps to our diversity checks.

**LoongFlow: Directed Evolutionary Search via Plan-Execute-Summarize**
- arXiv: 2512.24077 | Dec 30, 2025
- Authors: Chunhui Wan et al.
- **Key contribution:** Cognitive PES paradigm for structured mutations. Hybrid evolutionary memory: Multi-Island + MAP-Elites + adaptive Boltzmann selection. Outperforms OpenEvolve, ShinkaEvolve by up to 60%.
- **SAGE relevance:** HIGH. The PES paradigm for mutation operators and hybrid memory system are directly applicable. The Multi-Island + MAP-Elites combination is our planned architecture.

**CycleQD: Agent Skill Acquisition via Quality Diversity**
- arXiv: 2410.14735 | ICLR 2025
- Authors: (multiple)
- **Key contribution:** Cyclic adaptation of MAP-Elites for multi-task LLM optimization. Each skill's metric optimized in isolation, others as behavioral characteristics. SVD-based mutation + model merging crossover.
- **Results:** LLAMA3-8B surpasses traditional fine-tuning, matches GPT-3.5-TURBO.
- **SAGE relevance:** HIGH. The cyclic QD approach for multi-skill optimization maps to our topology evolution where different topology features serve as behavioral dimensions.

**Diverse Prompts: Illuminating Prompt Space with MAP-Elites**
- arXiv: 2504.14367 | Apr 19, 2025
- Authors: Gabriel Machado Santos et al.
- **Key contribution:** CFG + MAP-Elites for prompt space exploration. Diversity dimensions: number of shots + reasoning depth. Evaluated on BigBench Lite tasks.
- **SAGE relevance:** MEDIUM. Demonstrates MAP-Elites applicability to LLM optimization with behavioral dimensions.

### 2.2 Topology Optimization

**Graph-GRPO: Stabilizing Multi-Agent Topology Learning**
- arXiv: 2603.02701 | Mar 3, 2026
- Authors: Yueyang Cang et al.
- **Key contribution:** Group Relative Policy Optimization for topology. Samples group of diverse communication graphs per query, computes edge-level advantage based on relative performance within group. Normalizes rewards across sampled group.
- **SAGE relevance:** CRITICAL. Already referenced in project (Graph-GRPO). The edge-level credit assignment maps directly to our topology evolution.

**AgentConductor: Topology Evolution for Competition-Level Code Generation**
- arXiv: 2602.17100 | Feb 19, 2026
- Authors: Siyu Wang et al.
- **Key contribution:** RL-optimized MAS with LLM orchestrator. Task-adapted density-aware layered DAG topology. Novel topological density function for communication-aware characterization. Difficulty interval partitioning.
- **Results:** +14.6% pass@1 accuracy, -13% density, -68% token cost.
- **SAGE relevance:** HIGH. Already in project memory. The density function and difficulty-aware topology generation are directly applicable.

**TopoDIM: One-shot Topology Generation with Diverse Interaction Modes**
- arXiv: 2601.10120 | Jan 15, 2026
- Authors: Rui Sun et al.
- **Key contribution:** Decentralized one-shot topology generation. Agents autonomously construct heterogeneous communication without iterative coordination. -46.41% tokens, +1.50% average performance.
- **SAGE relevance:** MEDIUM. The one-shot generation approach could augment our template-first topology strategy.

**GoAgent: Group-of-Agents Communication Topology Generation**
- arXiv: 2603.19677 | Mar 20, 2026
- Authors: Hongjiang Chen et al.
- **Key contribution:** Groups as atomic units. Autoregressive group selection + connection. Conditional Information Bottleneck (CIB) objective compresses inter-group communication.
- **Results:** 93.84% accuracy, -17% token consumption.
- **SAGE relevance:** MEDIUM. The group-as-unit concept could inform hierarchical topology generation.

**MASS: Multi-Agent System Search**
- arXiv: 2502.02533 | Feb 4, 2025
- Authors: Han Zhou et al. (Google)
- **Key contribution:** 3-stage optimization: block-level prompt optimization -> workflow topology optimization -> workflow-level prompt optimization. Each stage conditioned on prior iterations.
- **SAGE relevance:** MEDIUM. The staged optimization approach (prompts then topology then global prompts) could inform our evolution pipeline ordering.

---

## 3. Self-Improving Agents / Evolution Engines

### 3.1 Self-Evolving Agent Frameworks

**Group-Evolving Agents (GEA): Open-Ended Self-Improvement**
- arXiv: 2602.04837 | Feb 4, 2026
- Authors: Zhaotian Weng et al.
- **Key contribution:** Group as fundamental evolutionary unit (vs tree-structured). Explicit experience sharing within group. Converts early exploratory diversity into sustained long-term progress.
- **Results:** 71.0% SWE-bench Verified (vs 56.7% prior SOTA), 88.3% Polyglot. Fixes framework bugs in 1.4 iterations (vs 5 for tree-based).
- **SAGE relevance:** HIGH. Group-level evolution with experience sharing maps to our population-based approach. The efficiency gains over tree-based evolution validate our design.

**HyEvo: Self-Evolving Hybrid Agentic Workflows**
- arXiv: 2603.19639 | Mar 20, 2026
- Authors: Beibei Xu et al.
- **Key contribution:** LLM-driven multi-island evolutionary strategy with reflect-then-generate. Hybrid nodes: probabilistic LLM nodes + deterministic code nodes. Iteratively refines both topology and node logic via execution feedback.
- **Results:** Up to 19x cost reduction, 16x latency reduction vs SOTA.
- **SAGE relevance:** CRITICAL. Already referenced in project. The hybrid LLM/code node concept maps directly to our TopologyGraph. Multi-island evolution + reflect-then-generate is our planned approach.

**EvoFSM: Controllable Self-Evolution with Finite State Machines**
- arXiv: 2601.09465 | Jan 14, 2026
- Authors: Shuo Zhang et al.
- **Key contribution:** Evolves explicit FSM instead of free-form rewriting. Decouples into macro Flow (state transitions) + micro Skill (state behaviors). Self-evolving memory distills successful trajectories as priors and failures as constraints.
- **Results:** 58.0% on DeepSearch benchmark.
- **SAGE relevance:** HIGH. The FSM-based evolution with Flow/Skill decomposition could inform our topology mutation operators. The self-evolving memory is directly relevant.

**AVO: Agentic Variation Operators for Autonomous Evolutionary Search**
- arXiv: 2603.24517 | Mar 25, 2026
- Authors: Terry Chen et al. (23 researchers)
- **Key contribution:** Replaces traditional mutation/crossover with autonomous coding agents. Self-directed agent loop: consult lineage -> propose -> repair -> critique -> verify. Domain-specific knowledge base integration.
- **Results:** Outperforms cuDNN by 3.5%, FlashAttention-4 by 10.5% on multi-head attention kernels.
- **SAGE relevance:** HIGH. The agentic variation operator concept (propose-repair-critique-verify loop) maps to our llm_mutator.py. Lineage tracking validates our bidirectional tracking approach.

**CoMAS: Co-Evolving Multi-Agent Systems via Interaction Rewards**
- arXiv: 2510.08529 | Oct 9, 2025
- Authors: Xiangyuan Xue et al.
- **Key contribution:** Agents improve through mutual discussion/collaboration. Intrinsic rewards from discussion dynamics. LLM-as-judge for reward formulation. Decentralized scalable co-evolution.
- **SAGE relevance:** MEDIUM. The interaction-based reward generation could inform our evolution reward shaping.

**AgentEvolver: Efficient Self-Evolving Agent System**
- arXiv: 2511.10395 | Nov 13, 2025
- Authors: Yunpeng Zhai et al.
- **Key contribution:** 3 mechanisms: self-questioning (curiosity-driven task generation), self-navigating (experience reuse + hybrid policy), self-attributing (differentiated rewards per state/action).
- **SAGE relevance:** MEDIUM. The differentiated reward attribution is relevant to our evolution reward shaping.

**AgentFactory: Self-Evolving via Executable Subagent Accumulation**
- arXiv: 2603.18000 | Mar 18, 2026
- Authors: Zhang Zhang et al.
- **Key contribution:** Preserves successful task solutions as executable subagent code (not textual experience). Subagents continuously refined via execution feedback. Pure Python code with standardized documentation.
- **SAGE relevance:** MEDIUM. The executable code preservation concept could augment our topology archive.

**Towards AGI: Pragmatic Self-Evolving Agent**
- arXiv: 2601.11658 | Jan 15, 2026
- Authors: Indrajit Kar et al.
- **Key contribution:** Hierarchical multi-agent framework. Compares Curriculum Learning, RL, and GA evolution. CL: fast recovery; RL: hard tasks; GA: high diversity.
- **SAGE relevance:** LOW-MEDIUM. The comparison of evolution paradigms informs our algorithm selection.

### 3.2 Surveys

**A Survey of Self-Evolving Agents: On Path to ASI**
- arXiv: 2507.21046 | Jul 28, 2025
- Authors: Huan-ang Gao et al.
- **Key contribution:** First comprehensive survey. Three dimensions: what/when/how to evolve. Covers model, memory, tools, architecture evolution. Intra-test-time vs inter-test-time adaptation.
- **SAGE relevance:** HIGH. Reference survey for our evolution engine design decisions.

**A Survey on Self-Evolution of LLMs**
- arXiv: 2404.14387 | Apr 22, 2024
- Authors: Zhengwei Tao et al.
- **Key contribution:** Iterative cycle: experience acquisition -> refinement -> updating -> evaluation.
- **SAGE relevance:** MEDIUM. Foundational framework for self-evolution lifecycle.

---

## 4. Drift Monitoring for Adaptive AI Systems

### 4.1 LLM-Specific Drift Detection

**Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems**
- arXiv: 2601.04170 | Jan 7, 2026
- Author: Abhishek Rath
- **Key contribution:** Defines "agent drift" as progressive degradation of behavior, decision quality, and inter-agent coherence. Three manifestations: semantic drift, coordination drift, behavioral drift. Proposes Agent Stability Index (ASI) across 12 dimensions (response consistency, tool usage patterns, reasoning pathway stability, inter-agent agreement rates). Mitigation: episodic memory consolidation, drift-aware routing, adaptive behavioral anchoring.
- **SAGE relevance:** CRITICAL. The ASI metric and 3-type drift taxonomy map directly to our drift.py monitoring. The mitigation strategies (memory consolidation, drift-aware routing) are exactly what we're implementing.

**Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces**
- arXiv: 2601.09001 | Jan 13, 2026
- Authors: Pedro Memoli Buffa, Luciano Del Corro
- **Key contribution:** Output-entropy profiles from final-layer logprobs. 11 statistics summarize entropy trace. Lightweight classifier predicts instance correctness -> domain-level accuracy estimate. Evaluated on 9 LLMs (3B-20B), 10 STEM benchmarks.
- **SAGE relevance:** HIGH. The entropy-based monitoring could be integrated into our quality estimation pipeline for real-time drift detection.

**Entropy-Based Measurement of Value Drift and Alignment in LLMs**
- arXiv: 2512.03047 | Nov 19, 2025
- Author: Samih Fadli
- **Key contribution:** "Second Law of Intelligence" framework. Ethical entropy S(t) as measurable state variable. 5-category behavioral taxonomy. Instruction-tuned models reduce drift 80%. Monitoring pipeline alerts when entropy exceeds stability threshold.
- **SAGE relevance:** MEDIUM. The entropy-based drift monitoring with threshold alerts could inform our drift detection strategy.

**When Agents Disagree With Themselves: Measuring Behavioral Consistency**
- arXiv: 2602.11619 | Feb 12, 2026
- Author: Aman Mehta
- **Key contribution:** ReAct agents produce 2.0-4.2 distinct action sequences per 10 runs. Consistent behavior (<=2 paths) achieves 80-92% accuracy; highly inconsistent (>=6 paths) achieves 25-60%. 69% of divergence occurs at step 2 (first search query).
- **SAGE relevance:** HIGH. Behavioral consistency as a predictive signal for failure is directly usable in our drift monitor. The step-2 divergence finding informs where to focus monitoring.

### 4.2 Production Monitoring Systems

**Failure Modes in LLM Systems: A System-Level Taxonomy**
- arXiv: 2511.19933 | Nov 25, 2025
- Author: Vaishali Vinay
- **Key contribution:** 15 hidden failure modes including multi-step reasoning drift, latent inconsistency, context-boundary degradation, version drift, cost-driven performance collapse. Frames LLM reliability as system-engineering problem.
- **SAGE relevance:** HIGH. The failure mode taxonomy informs our drift monitoring categories.

**OAKS: Online Adaptation to Continual Knowledge Streams**
- arXiv: 2603.07392 | Mar 8, 2026
- Authors: Jiyeon Kim et al. (Adobe)
- **Key contribution:** Benchmark for online adaptation over streaming, continually updating knowledge. Facts change dynamically across time intervals. Even SOTA models and agentic memory systems fail to adapt robustly.
- **SAGE relevance:** MEDIUM. Validates the need for our adaptive memory system's ability to handle knowledge evolution.

**Accurate Failure Prediction Does Not Imply Effective Prevention**
- arXiv: 2602.03338 | Feb 3, 2026
- Authors: Rakshith Vasudev et al.
- **Key contribution:** LLM critic with AUROC 0.94 can cause 26pp performance collapse. Disruption-recovery tradeoff: interventions may disrupt succeeding trajectories. 50-task pilot test to estimate intervention safety.
- **SAGE relevance:** HIGH. Critical finding for our drift response strategy: detection alone is insufficient; intervention safety must be validated.

### 4.3 Concept Drift Theory

**The Window Dilemma: Why Concept Drift Detection is Ill-Posed**
- arXiv: 2602.06456 | Feb 6, 2026
- Authors: Brandon Gower-Winter et al.
- **Key contribution:** Perceived drift is a product of windowing, not necessarily the underlying process. Traditional batch learning often outperforms drift-aware counterparts. Drift detection verification is implausible in practice.
- **SAGE relevance:** MEDIUM. Important cautionary finding: our drift detection must account for windowing artifacts.

---

## Summary: Key Takeaways for SAGE Implementation

### Memory System (Priority: CRITICAL)
1. **MAGMA's 4-graph architecture** (semantic/temporal/causal/entity) validates our 4-tier S-MMU design
2. **Write-time gating** (arXiv 2603.15994) achieves 100% vs 13% accuracy -- implement salience scoring (reputation, novelty, reliability) in write_gate.py
3. **RPE-based routing** (D-MEM) for fast/slow memory paths reduces token consumption 80%
4. **EverMemOS engram lifecycle** (episodic trace -> semantic consolidation -> recollection) is the consolidation pipeline to implement
5. **CraniMem's bounded episodic buffer + scheduled consolidation loop** maps to our working memory capacity management

### Evolution Engine (Priority: HIGH)
1. **GigaEvo** remains the reference implementation (MAP-Elites + LLM mutation + lineage tracking)
2. **ShinkaEvolve's bandit-based LLM ensemble selection** for mutation operator selection
3. **LoongFlow's PES paradigm** (Plan-Execute-Summarize) for structured mutations, 60% efficiency gain over GigaEvo
4. **HyEvo's hybrid LLM/code nodes** with multi-island evolution validates our design
5. **AVO's agentic variation operators** (propose-repair-critique-verify) for sophisticated mutation
6. **GEA's group-level evolution** with experience sharing outperforms tree-based by 14.3pp on SWE-bench

### Topology Optimization (Priority: HIGH)
1. **Graph-GRPO edge-level credit assignment** via group sampling
2. **AgentConductor's density-aware DAG generation** with difficulty partitioning
3. **CycleQD's cyclic MAP-Elites** for multi-skill optimization

### Drift Monitoring (Priority: MEDIUM)
1. **Agent Drift's ASI metric** across 12 dimensions with 3-type taxonomy
2. **Entropy Sentinel's logprob-based monitoring** for real-time accuracy estimation
3. **Behavioral consistency** (arXiv 2602.11619): variance in action sequences predicts failure
4. **CAUTION**: Intervention without validation can cause 26pp regression (arXiv 2602.03338)
5. **CAUTION**: Windowing artifacts can create false drift signals (arXiv 2602.06456)
