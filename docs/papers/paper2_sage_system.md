# YGN-SAGE: A Self-Adaptive Agent Development Kit with Five Cognitive Pillars

## Abstract

We present YGN-SAGE, an agent development kit built on five cognitive pillars: Topology, Tools, Memory, Evolution, and Strategy. The system achieves 84.1% pass@1 on HumanEval+ (164 problems, 80x harder tests) using a budget-tier Gemini 2.5 Flash model — competitive with flagship models (GPT-4o: 87%, O1: 89%). A 6-configuration ablation study shows the full framework adds +15pp over the bare LLM baseline, with each pillar contributing measurably: AVR self-refinement (+10pp), memory (+10pp), routing (+5pp), guardrails (+5pp). The system implements a Rust+Python hybrid architecture with formal verification (OxiZ SMT), 4-tier memory (Arrow working memory → episodic → semantic → ExoCortex RAG), and evolutionary topology search (MAP-Elites + CMA-ME + MCTS).

## Key Results

| Benchmark | Score | Notes |
|-----------|-------|-------|
| HumanEval+ pass@1 | **84.1%** (138/164) | 80x harder tests, budget model |
| MBPP+ pass@1 | **75.1%** (284/378) | 35x harder tests |
| Ablation: full vs bare | **+15pp** (100% vs 85%) | Paired, same model, 20 tasks |
| Routing GT (kNN) | **92%** (46/50) | vs 52% heuristic baseline |
| Routing GT (SystemRouter) | **88%** (44/50) | Rust, domain-aware |
| TopologyBench (20 tasks) | **90-100%** (4 topologies) | Disjoint failures, Jaccard=0.00 |

## Architecture

### Five Pillars
1. **Topology**: 8 templates (Sequential, Parallel, AVR, SelfMoA, Hierarchical, Hub, Debate, Brainstorming), MAP-Elites evolutionary search, CMA-ME refinement, MCTS exploration, LLM synthesis
2. **Tools**: AgentTool composition, Wasm WASI sandbox (deny-by-default), tree-sitter AST validation
3. **Memory**: 4-tier system — Rust Arrow STM → SQLite episodic → entity-relation semantic → ExoCortex RAG (500+ sources)
4. **Evolution**: DGM context injection, SAMPO solver (5 strategic actions), self-modifying hyperparameters
5. **Strategy**: S1/S2/S3 cognitive routing (Kahneman), kNN + structural + ONNX BERT cascade, contextual bandit model selection

### Formal Verification
- OxiZ (pure Rust SMT, QF_LIA) for invariant verification, memory safety proofs
- LTL model checking on topology graphs (reachability, safety, liveness, bounded liveness)
- CEGAR invariant synthesis with clause-level diagnostic feedback

## SOTA Comparison (HumanEval+ pass@1)

| System | Score | Model | Cost Tier |
|--------|-------|-------|-----------|
| O1 | ~89% | Flagship reasoning | $$$ |
| GPT-4o | ~87% | Flagship | $$ |
| Qwen2.5-Coder-32B | ~87% | Specialized coding | $$ |
| **YGN-SAGE** | **84.1%** | Budget (Gemini 2.5 Flash) | $ |
| Claude Sonnet 3.5 | ~82% | Mid-tier | $$ |

## Reproducibility

```bash
cd sage-python
pip install -e ".[all,dev]"
python -m sage.bench --type evalplus --dataset humaneval  # HumanEval+ (164)
python -m sage.bench --type ablation --limit 20           # 6-config ablation
python tests/e2e_proof.py                                  # 25/25 E2E tests
```

## Topology Significance Results

TopologyBench on HumanEval+ with real multi-agent execution (20-task pilot):

| Topology | pass@1 | Failures |
|----------|--------|----------|
| debate | **100.0%** (20/20) | 0 |
| sequential | 95.0% (19/20) | HumanEval/14 |
| parallel | 95.0% (19/20) | HumanEval/10 |
| brainstorming | 90.0% (18/20) | HumanEval/0, /9 |

**Key finding**: Zero failure overlap (Jaccard=0.00) across all 6 topology pairs. Each topology has unique blind spots, suggesting topology-aware routing can exceed any single topology's performance.

## Evolution Statistical Proof

5-run paired experiment (10 HumanEval+ tasks per run, budget model):

| Config | Mean pass@1 | Runs |
|--------|-------------|------|
| No-evolution | **98.0%** | [100%, 100%, 100%, 90%, 100%] |
| Full (evo ON) | 88.0% | [90%, 90%, 90%, 90%, 80%] |

**Honest negative result**: Evolution hurts (-10pp) on simple tasks with a budget model. Cohen's d = -1.41 (large negative), Bootstrap 95% CI: [-16pp, -4pp]. The TopologyEngine adds multi-node overhead that introduces failure points without benefit when the base system already achieves ~98%.

**Interpretation**: Evolution is designed for multi-model heterogeneous routing across complex tasks. On simple HumanEval+ tasks with a single budget model, the simpler orchestrator path outperforms. This is consistent with the "model ceiling" finding — when base accuracy is high, added complexity hurts.

## Limitations

- TopologyBench HumanEval+ pilot is N=20 tasks; full 164-task confirmation in progress
- GSM8K topology significance: null result (model ceiling too high at 96%)
- Evolution: negative result on budget model — may perform differently with multi-model routing
- SWE-Bench evaluation: adapter created, not yet run (requires Docker)
- kNN routing ground truth: 50 tasks (small, needs expansion)
