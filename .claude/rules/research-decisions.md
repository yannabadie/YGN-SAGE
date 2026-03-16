---
paths:
  - "**/*.py"
  - "**/*.rs"
  - "docs/**"
---

# Research-Backed Decisions

## Routing
- **kNN on arctic-embed-m** (arXiv 2505.12601): 92% GT accuracy. PRIMARY router.
- **Cascade routing > upfront routing** on hard code tasks (ETH-SRI ICLR '25, Cascade Routing 2410.10347).
- **Quality estimation is the bottleneck**, not routing algorithm (ETH-SRI).
- **DeBERTa zero-shot: 52%** — needs fine-tuning, not zero-shot (arXiv eval).

## Topology
- **Multi-agent 2x single-model** on SWE-bench (55% vs 27.3%).
- **Topology effect small on saturated benchmarks** (HumanEval+ 3.1pp spread, 0/6 significant).
- **Topology matters on hard tasks** where base accuracy is 60-80% (AdaptOrch 2602.16873).
- **AgentConductor** (2602.17100): RL topology evolution, 97.5% HumanEval with 3B model.

## Evolution
- **Online evolution > offline** (Live-SWE-agent: 77.4% SWE-bench by self-evolving).
- **AlphaEvolve**: LLM as intelligent mutation operator (DeepMind 2025).
- **Evolution -1pp on simple tasks** (our data, NOT -10pp as previously documented).

## Memory
- **CoALA cognitive architecture**: working memory + episodic/semantic/procedural LTM.
- **FoVer**: Z3 auto-labels for PRM training data — SAGE has Z3 but doesn't yet use it for training.
- **MEM1**: RL-trained compression > pressure threshold.

## Quality
- **QualityEstimator**: Z3 formal labeler (Rust) > ONNX learned model > None (abstain).
- **Zero heuristics**: 5-signal heuristic REMOVED. Proved mediocre (r=0.34 Pearson).

## DROPPED Ideas (with evidence)
- Speculative S1+S2 parallel: no SOTA backing
- Full PSRO/DCH: wrong abstraction for LLM routing (contextual bandit is correct)
- vqsort-rs: dead crate
- Provider smoke tests at boot: FrugalGPT cascade handles failures
- Neo4j/Qdrant: sqlite-vec sufficient at current scale
- SWE-bench Lite: replaced by Pro
- DeBERTa: superseded by ModernBERT (P1 backlog)
