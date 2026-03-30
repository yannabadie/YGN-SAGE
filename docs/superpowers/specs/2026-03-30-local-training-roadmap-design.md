# Local Training Roadmap — Autoresearch Loop for YGN-SAGE Path 6

**Date**: 2026-03-30
**Branch**: `local`
**Hardware**: RTX 3500 Ada 12GB (Windows, TRL+PEFT+bitsandbytes)
**Model**: Qwen3-4B NF4 + LoRA rank 32
**Goal**: Train a topology policy model that orchestrates the full SAGE Rust+Python pipeline

## Context

### Validated Results (March 29-30)
- SFT warmup: 1880 examples, loss 2.74 -> 0.92 (56 min)
- GRPO post-SFT: avg reward 0.40, max 0.93 on Phase A structural reward
- Baseline without SFT: avg reward 0.13 (GRPO cold-start problem)
- MASBENCH (main branch): SAGE 67% vs bare model 40% (+27pp with full Rust stack)
- Pod training: DAPO > GRPO (token-level loss, better convergence)

### The Key Insight
Phase A structural reward (YAML validity) is bootstrapping only. The real objective is
a model that generates topologies which **execute successfully in the SAGE pipeline**:
TopologyRunner -> ProviderPool -> per-node LLM calls -> sandbox code testing -> QualityLabeler.

## Architecture: Autoresearch Loop

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
fixed-budget experiments, structured journal, autonomous iteration.

```
HYPOTHESIZE -> TRAIN -> EVALUATE -> RECORD -> repeat
     ^                                  |
     +----------------------------------+
     (reads journal of previous experiments)
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| Orchestrator | `scripts/autoresearch_loop.py` | Reads journal, proposes hypothesis, runs experiment |
| Training script | `scripts/train_local_qwen3_4b.py` | SFT/GRPO/DAPO, accepts JSON config |
| Evaluator N1 | `scripts/eval_reward_holdout.py` | Reward on 50 holdout prompts (0 cost, 2 min) |
| Evaluator N2 | `scripts/eval_masbench_local.py` | MASBENCH depth 20 tasks (low API cost, 10 min) |
| Evaluator N3 | `scripts/eval_bigcodebench.py` | BigCodeBench Hard Instruct 20 tasks (API cost, 30 min) |
| Journal | `experiments/journal.jsonl` | Structured experiment log |
| Holdout set | `experiments/holdout_50.json` | 50 prompts never seen in training |
| Config registry | `experiments/configs/` | JSON configs for reproducibility |

### Evaluation Cascade

Every experiment runs N1. N2 runs if N1 improves. N3 runs if N2 improves.

| Level | What | Cost | Duration | When |
|-------|------|------|----------|------|
| N1 | Reward avg/max on 50 holdout | 0 | 2 min | Every experiment |
| N2 | MASBENCH depth 20 tasks | ~$0.50 | 10 min | N1 improves over best |
| N3 | BigCodeBench Hard 20 tasks | ~$2.00 | 30 min | N2 improves over best |

### Journal Entry Format
```json
{
  "id": "exp-042",
  "timestamp": "2026-04-02T14:30:00Z",
  "phase": "2-execution",
  "hypothesis": "Increasing temperature from 0.7 to 1.0 improves topology diversity",
  "config_delta": {"temperature": 1.0},
  "base_checkpoint": "sft_v3_epoch2",
  "train_budget_min": 10,
  "train_steps": 40,
  "metrics": {
    "n1_reward_avg": 0.52,
    "n1_reward_max": 0.87,
    "n1_clipped_ratio": 0.15,
    "n2_masbench_depth": null,
    "n3_bigcodebench_hard": null
  },
  "conclusion": "Temperature 1.0 improves diversity but reward variance increases. Keep for execution phase.",
  "duration_min": 12.3
}
```

## Roadmap

### Phase 0 — Foundations (1-2 days)

**Objective**: Build the autoresearch infrastructure.

**Deliverables**:
1. `autoresearch_loop.py` — orchestrator script
2. `experiments/holdout_50.json` — stratified holdout (15 simple, 20 moderate, 15 complex)
3. `eval_reward_holdout.py` — standalone N1 evaluator
4. Refactor `train_local_qwen3_4b.py` to accept JSON config for full reproducibility
5. Journal format + first baseline entry (current SFT+GRPO results)

**Exit criteria**: Can run `python scripts/autoresearch_loop.py --budget 10` and get a journal entry.

### Phase 1 — SFT Ablation (3-5 days)

**Objective**: Find the optimal SFT checkpoint as RL starting point.

**Variables** (one per experiment):
| Variable | Values | Hypothesis |
|----------|--------|------------|
| SFT epochs | 1, 2, 3, 5 | More epochs = better YAML but risk overfitting |
| Learning rate | 1e-5, 2e-5, 5e-5 | H100 used 2e-5, local 4-bit may need different |
| Data mix | balanced / code-heavy / math-heavy | BigCodeBench tasks need code-heavy? |
| LoRA rank | 16, 32, 64 | Higher rank = more capacity but more VRAM |
| Max seq length | 512, 768, 1024 | Longer = more complete topologies but slower |

~15-20 experiments, 10 min each = 3-4 hours GPU total.

**Evaluation**: N1 (reward holdout). Best SFT checkpoint selected by highest P(reward > 0.3).

**Exit criteria**: SFT checkpoint with P(reward > 0.3) > 60% on holdout.

### Phase 2 — Execution Reward Training (1-2 weeks)

**Objective**: The model learns which topologies actually produce working code.

**Key change**: `SAGE_VERL_EXEC=1` — reward becomes 30% structural + 70% execution.
The TopologyRunner executes each generated topology with real API calls:
- Parses YAML -> TopologyGraph (Rust TopologyEngine)
- Assigns models via ModelAssigner (Rust, cards.toml)
- Routes through kNN (92%) + SystemRouter (88%)
- Executes per-node via ProviderPool (7 providers)
- Tests output code in sandbox (tree-sitter -> Wasm WASI)
- QualityLabeler (Rust Z3/OxiZ) scores the result

**Reward graduation** (from reward.py):
| Outcome | Reward |
|---------|--------|
| PASSED (tests pass) | 1.5 |
| WRONG_ANSWER | 1.0 |
| RUNTIME_ERROR | 0.7 |
| TIMEOUT | 0.5 |
| COMPILATION_ERROR | 0.3 |
| INVALID_TOPOLOGY | structural only (~0.1-0.4) |

**Algorithm**: DAPO if implementable locally, GRPO otherwise.
DAPO advantages (arXiv 2503.14476):
- Token-level loss: denser gradient for partially correct YAML
- Asymmetric clipping: prevents over-conservatism on good trajectories
- Dynamic sampling: focuses on prompts where the model has most to learn

**Training budget**: 60 min per experiment (execution is slower due to API calls).
API cost: ~$0.10/experiment (20 prompts x 2-3 node topologies x budget models).

**Variables**:
| Variable | Values | Hypothesis |
|----------|--------|------------|
| Algorithm | GRPO vs DAPO | DAPO converges faster (pod evidence) |
| Exec timeout | 30s, 60s, 120s | Shorter = faster iteration, longer = more complex topologies pass |
| Structural weight | 0.3, 0.5, 0.7 | Higher = safer (fall back to structure when exec fails) |
| Temperature | 0.7, 1.0, 1.2 | Exploration vs exploitation |
| K rollouts | 2, 4 | More = better advantage estimation but slower |

**Evaluation**: N1 + N2 (MASBENCH). The model must improve MASBENCH depth score.

**Exit criteria**:
- N1 reward avg > 0.5 (execution-weighted)
- N2 MASBENCH depth > 50% improvement over baseline
- P(PASSED) > 20% on holdout (model produces working code)

### Phase 3 — Phase C: Adaptive Orchestration (1-2 weeks)

**Objective**: The model learns micro-decisions within the SAGE pipeline.

**Key change**: `SAGE_TRAINING_PHASE=C` — bonuses for:
- Correct `model_tier` per node (+0.1 x tier_ratio)
- Adaptation checkpoints (+0.1 when placed correctly)
- Provider hints (+0.05 when model knows which provider is best)
- Hybrid LLM+code topologies (+0.1 for code nodes, HyEvo-style)

**What the model learns**:
1. "For a simple math problem, 2 nodes with budget tier is optimal"
2. "For complex code generation, use coder(codex) -> reviewer(reasoner) -> synthesizer(fast)"
3. "Place a checkpoint after the coder node so the system can upgrade if quality is low"
4. "Use DeepSeek for budget, Google for fast, OpenAI for codex"

**Curriculum learning**: Sort prompts by difficulty (simple -> moderate -> complex).
Justified: Bengio et al. 2009 (curriculum), The Conductor uses difficulty-aware training.

**Advanced features to activate**:
- Edge-level credit (Graph-GRPO, arXiv 2603.02701) — per-edge success rates
- RewardFlow (arXiv 2603.18859) — PageRank-based per-node credit
- Cost efficiency scoring (CARD, arXiv 2603.01089) — budget penalty

**Evaluation**: N1 + N2 + N3. This is where BigCodeBench enters.

**Exit criteria**:
- N1 reward avg > 0.7 (Phase C max is 1.35)
- N2 MASBENCH depth > main branch (67%)
- N3 BigCodeBench Hard > 37.8% (current SAGE score)

### Phase 4 — Deployment + Final Proof (3-5 days)

**Objective**: Ship the model and prove it helps.

**Deliverables**:
1. Merge LoRA or keep as adapter (benchmark both for inference speed)
2. Push to HuggingFace: `yannabadie/sage-topology-policy-local`
3. Wire into TopologyEngine as Path 6 (`SAGE_ENABLE_PATH6=1`)
4. **Final benchmark**: BigCodeBench Hard Instruct (full 50 tasks)
   - Without Path 6 (baseline)
   - With Path 6 local (Qwen3-4B)
   - With Path 6 pod (Nemotron-8B) if available
5. Write the experiment report (journal summary, convergence curves, ablation tables)
6. Commit everything to `local` branch, PR to `main`

**Exit criteria**:
- BigCodeBench Hard delta > 0pp with Path 6 (any improvement = success)
- Reproducible: anyone can clone, run setup, get same results
- All experiments in journal with full configs

## Constraints

### Hardware
- RTX 3500 Ada 12GB VRAM (WDDM, Windows)
- `nvidia-smi -lgc 3105` required (GPU clock lock)
- `powercfg //setactive 8c5e7fda` (High Performance power plan)
- No vLLM (Linux-only) — native generation only

### VRAM Budget
| Phase | VRAM | Bottleneck |
|-------|------|-----------|
| SFT | ~6.7 GB | Forward+backward with gradient checkpointing |
| GRPO generation | ~5.7 GB | Autoregressive decoding (no KV cache during GC) |
| GRPO backward | ~12 GB | Full activations without GC |
| Execution eval | ~5.7 GB + API | Network I/O, not GPU |

### API Cost (Phase 2+)
- Execution reward uses budget models (DeepSeek ~$0.003/topology)
- MASBENCH: ~$0.50/run (20 tasks x budget)
- BigCodeBench: ~$2.00/run (20 tasks x budget)
- Budget total Phase 2: ~$10-20. Phase 3: ~$20-50.

### Time per Experiment
| Type | Duration |
|------|----------|
| SFT ablation | 10 min |
| GRPO structural | 30 min |
| GRPO execution | 60 min |
| MASBENCH eval | 10 min |
| BigCodeBench eval | 30 min |

## Scientific Rigor

### Reproducibility
- Every experiment has a JSON config in `experiments/configs/`
- Every result in `experiments/journal.jsonl`
- Random seed fixed (42) for all experiments
- Holdout set fixed (50 prompts, stratified by difficulty)

### Statistical Validation
- Report mean +/- std across K rollouts
- Wilcoxon signed-rank test for paired comparisons (Phase A vs C, GRPO vs DAPO)
- Cohen's d for effect size (already in evolution/evaluator.py)
- Minimum 3 runs per critical comparison

### Publications-Ready Outputs
- Convergence curves (reward vs step) per phase
- Ablation tables (one variable at a time)
- MASBENCH depth comparison (with/without Path 6)
- BigCodeBench Hard delta (framework value proof)

## References

- InstructGPT (Ouyang et al. 2022): SFT -> RL pipeline justification
- GRPO (Shao et al. 2024, arXiv 2402.03300): Group Relative Policy Optimization
- DAPO (arXiv 2503.14476): Token-level loss, asymmetric clip, dynamic sampling
- The Conductor (arXiv 2512.04388): SFT -> GRPO for topology, KL=0, binary reward converges in 200 iters
- AgentConductor (arXiv 2602.17100): RL-optimized layered DAG orchestrator, 97.5% HumanEval
- Graph-GRPO (arXiv 2603.02701): Edge-level credit assignment
- RewardFlow (arXiv 2603.18859): PageRank per-node credit propagation
- CARD (arXiv 2603.01089): Cost-aware reward conditioning
- Curriculum Learning (Bengio et al. 2009): Easy-to-hard training order
- karpathy/autoresearch: Fixed-budget autonomous experiment loop
