# YGN-SAGE V2 — Learned Adaptive Multi-Agent Topologies with Runtime Recovery

**Date:** 2026-03-20
**Branch:** VeRLGIGPO
**Status:** Design approved, ready for implementation

## Objective

Train a model (Qwen3.5-4B local validation → Qwen3.5-9B pod) that generates multi-agent topologies capable of self-correction during execution. The model learns three skills simultaneously:
1. **Topology generation** — DAG structure with roles, prompts, model tiers
2. **Runtime revision** — when to upgrade, reroute, or continue after checkpoint failures
3. **Experience exploitation** — use episodic memory of past successes/failures

No existing system combines learned topology + runtime adaptation + multi-provider + episodic memory. SAGE V2 is the first.

## Research Foundation

| Paper | ID | Contribution to SAGE |
|-------|-----|---------------------|
| RewardFlow | 2603.18859 (AAMAS 2026) | PageRank reward propagation → per-node credit |
| MASPRM | 2510.24803 | Multi-agent PRM via MCTS, zero human annotations |
| MAPPA | 2601.23228 | Per-action per-agent rewards from AI feedback |
| TopoCurate | 2603.01714 | Reflective Recovery metric for data curation |
| CARD | 2603.01089 (ICLR 2026) | Price penalty in loss, capability conditioning |
| GiGPO | 2505.10978 (NeurIPS/ICLR) | Step-level advantage, works for multi-turn revision |
| Graph-GRPO | 2603.02701 | Edge-level credit, already implemented in SAGE |
| Budget-Aware Routing | 2602.21227 | Per-step cheap/expensive model selection |
| AgentConductor | 2602.17100 | Direct competitor: SFT+GRPO, Qwen2.5-3B, no adaptation |
| AdaptOrch | 2602.16873 | Rule-based routing, 4 canonical templates, no learning |

## Architecture

### Training Environment (Multi-Turn with Memory)

```
SageTopologyEnv v2:

  reset(prompt, task_id):
    1. Query EpisodicMemory (SQLite) → top-3 similar past episodes
    2. Format memory context → inject in observation
    3. Return {text: prompt + memory, anchor: hash(prompt)}

  Turn 0 — GENERATE:
    model generates YAML with adaptation metadata
    → parse, build TopologyGraph (Rust), assign models (ProviderPool)
    → structural reward
    → anchor = topology_generator:difficulty:yaml_hash

  Turns 1..N — EXECUTE + DECIDE (incremental):
    For each checkpoint node:
      1. Execute node via ProviderPool.resolve(model_tier)
      2. TopologyController.evaluate_and_decide() → quality score
      3. If quality < threshold: present to model → "Action?"
      4. Model responds: "upgrade_model" or "continue"
         → anchor = role:quality_bucket:context_hash
      5. If upgrade: re-execute with fallback_tier
    For non-checkpoint nodes: execute silently

  Terminal — EVALUATE:
    1. Sandbox test → PASSED/FAILED
    2. RewardFlow: state-graph from K rollouts → PageRank → per-node rewards
    3. Resilience bonus + price penalty
    4. Store episode in EpisodicMemory (SQLite)
    5. StepRewardVector for GiGPO
```

### Reward Function (5 signals)

```
R_total = 0.20 × R_structural
        + 0.35 × R_execution
        + 0.20 × R_rewardflow
        + 0.15 × R_resilience
        + 0.10 × R_cost_efficiency
```

**R_structural** (existing): YAML format + Rust TopologyDensity.s_complex + PyHybridVerifier. Range [0.0, 1.0].

**R_execution** (existing): Sandbox test. PASSED=1.0, WRONG_ANSWER=0.5, RUNTIME_ERROR=0.3, TIMEOUT=0.2, NO_CODE=0.0.

**R_rewardflow** (new, arXiv 2603.18859):
- For prompt P, collect K=4 rollouts (different topologies)
- Build state-graph: state = (role, quality_bucket), transitions = consecutive nodes
- Terminal states get execution reward
- BFS/PageRank (damping=0.85, 20 iters) propagates backward
- Per-node reward = R_state of that node in the graph

**R_resilience** (new):
- 0.0 = no adaptation triggered or adaptation failed
- 0.3 = adaptation triggered + succeeded (node upgraded, output improved)
- 0.5 = adaptation triggered + succeeded + terminal PASSED

**R_cost_efficiency** (new, inspired by CARD 2603.01089):
- cost = sum(price_per_node) + sum(price_per_upgrade)
- budget_ref = {simple: $0.01, moderate: $0.05, complex: $0.20}
- R_cost = 1.0 - tanh(cost / budget_ref[difficulty])

### Episodic Memory (Training)

SQLite database persisting across epochs:

```sql
CREATE TABLE episodes (
    task_id TEXT,
    prompt_hash TEXT,
    domain TEXT,
    topology_yaml TEXT,
    n_nodes INTEGER,
    difficulty TEXT,
    outcome TEXT,           -- PASSED/FAILED/WRONG_ANSWER/...
    total_reward REAL,
    per_node_results TEXT,  -- JSON array
    adaptations_triggered INTEGER,
    embedding BLOB,         -- 768-dim for semantic search
    created_at TEXT
);
```

**Query:** Before each generation, semantic search for top-3 similar past episodes by prompt embedding similarity. Format as text context injected in the model's observation.

**Store:** After each episode, persist outcome + topology + per-node results. Memory grows across epochs — later epochs have richer context.

**Local validation:** Offline enrichment (update prompts between epochs).
**Pod (GiGPO):** Online query (SQLite query at each reset()).

**Embedding computation:** Precomputed offline during dataset preparation using arctic-embed-m (already in SAGE, 109M params). Embeddings stored as 768-dim float32 BLOBs in SQLite. At training time, similarity search uses numpy cosine distance on precomputed embeddings — no model loading needed during training (preserves VRAM for Qwen3.5-4B).

### RewardFlow Integration with GRPOTrainer

RewardFlow operates at the **batch level**, not per-rollout:
1. GRPOTrainer generates `num_generations=4` completions per prompt (K=4 rollouts)
2. Each rollout is scored by the 4 non-RewardFlow signals (structural, execution, resilience, cost)
3. `RewardFlowPropagator.compute()` takes all K rollouts as input, builds the state-graph, propagates
4. Per-node RewardFlow scores are added as the 5th signal to each rollout's total reward
5. The combined 5-signal scalar goes to GRPO advantage estimation

This reuses the same K rollouts for both GRPO grouping and RewardFlow — no extra generation overhead. RewardFlow is a **reward shaping step** between generation and advantage computation.

### Recovery Data: 2 entries per scenario

Each recovery scenario produces 2 training entries:
- Entry A: `initial_topology` → target YAML (the "before" pattern)
- Entry B: `recovered_topology` → target YAML (the "after" pattern, post-adaptation)

This teaches the model both "what a topology with fallback looks like" AND "what the topology looks like after adaptation succeeds." Effective recovery data: 40 × 2 = 80 entries.

## Critical Design Decisions

### C1: Gate semantics — `conditional` maps to `gate: open` + `condition` field

The Rust `Gate` enum only supports `"open"` and `"closed"`. The V2 adaptive YAML uses `gate: conditional` in the training data (90 entries in adaptive, 40 in recovery). Rather than adding a third Gate variant, we use the existing `TopologyEdge.condition: Option<String>` field:

- `gate: conditional` in YAML → parsed as `gate: "open"`, `condition: "quality_check"` in Rust
- The `_build_topology_graph()` parser maps `conditional` → `open` + sets `condition`
- `PyTopologyExecutor.close_gate()` is called by the controller when the condition fails
- This leverages existing Rust infrastructure without enum changes

### C2: Reward weights — initial values subject to ablation (not hardcoded)

The 0.20/0.35/0.20/0.15/0.10 weights are **initial values** derived from:
- R_execution at 0.35: execution is the primary signal (sandbox pass/fail)
- R_structural at 0.20: format correctness, matching existing weight in reward.py
- R_rewardflow at 0.20: per-node credit (RewardFlow paper shows +12.3% gain)
- R_resilience at 0.15: adaptation bonus (novel signal, conservative weight)
- R_cost_efficiency at 0.10: cost penalty (CARD paper uses 0.01, we start at 0.10)

**Ablation plan:** During Phase A local training, sweep R_resilience ∈ {0.10, 0.15, 0.20} and R_cost ∈ {0.05, 0.10, 0.15} on 100-prompt subsample. Pick weights maximizing reward/mean growth rate. Document results in training logs.

The Python `reward.py` exposes weights as module-level constants (not buried in logic) so they are trivially adjustable without code changes.

### C3: TopologyNode backward compatibility — keyword args with defaults

New fields use keyword arguments with defaults in the PyO3 `#[pyo3(signature)]` macro:
```rust
#[pyo3(signature = (role, model_id, system=2, prompt="", fallback_tier="", is_checkpoint=false, max_retries=0))]
```
Existing callers that use positional args (`TopologyNode("coder", "fast", 2, "...")`) continue to work. New fields are only set when explicitly provided. The `with_id()` and `new()` Rust constructors also default the new fields.

### C4: Model actions — V2 supports 2 of 5 controller actions

The TopologyController has 5 actions: `continue`, `upgrade_model`, `prune_node`, `reroute_topology`, `spawn_subagent`. V2 training exposes only 2 to the model:
- `continue` — proceed to next node
- `upgrade_model` — re-execute current node with fallback_tier

The other 3 (`prune_node`, `reroute_topology`, `spawn_subagent`) are **deferred to V3** to keep the action space small for initial training. The controller still uses them in production pipeline — they are just not part of the GiGPO training loop.

## Topology YAML Format (V2 Adaptive)

```yaml
difficulty: moderate
reasoning: "This algorithm task needs coding + review. The coder gets fast tier
  but has a reasoner fallback if quality is low. Reviewer checkpoint ensures
  code quality before synthesis."
adaptation:
  checkpoints: [0, 1]
  max_upgrades: 1
  max_reroutes: 0
  quality_threshold: 0.5
nodes:
  - role: coder
    model_tier: fast
    fallback_tier: reasoner
    prompt: "Write a function that..."
  - role: reviewer
    model_tier: budget
    fallback_tier: ""
    prompt: "Review the code for edge cases..."
  - role: synthesizer
    model_tier: fast
    fallback_tier: ""
    prompt: "Produce the final, complete, self-contained Python solution..."
edges:
  - {from_idx: 0, to_idx: 1, flow_type: message, gate: conditional}
  - {from_idx: 1, to_idx: 2, flow_type: message, gate: open}
```

**Note:** `gate: conditional` in YAML is parsed as `gate: "open"` with `condition: "quality_check"` in Rust (see C1 above).

## Implementation Plan

### Phase 0: Rust Changes (sage-core)

**topology_graph.rs — TopologyNode** (+3 fields):
- `fallback_tier: String` (#[pyo3(get, set)], default: "")
- `is_checkpoint: bool` (#[pyo3(get, set)], default: false)
- `max_retries: u8` (#[pyo3(get)], default: 0)

**topology_graph.rs — TopologyGraph** (+3 fields):
- `max_upgrades: u8` (#[pyo3(get, set)], default: 1)
- `max_reroutes: u8` (#[pyo3(get, set)], default: 0)
- `quality_threshold: f32` (#[pyo3(get, set)], default: 0.5)

**reward.rs — RewardScore** (+2 fields):
- `resilience: f32` (#[pyo3(get)])
- `cost_efficiency: f32` (#[pyo3(get)])
- New method `compute_full()` with adaptation and cost params

**Build:** `cd sage-core && maturin develop --features smt,onnx,cognitive,tool-executor --release`
**Test:** `cargo test --no-default-features --features smt,tool-executor --lib`

### Phase 1: Python Changes

**topology_env.py** — Major rewrite:
- Incremental node execution (not all-upfront)
- Decision turns after checkpoint nodes
- Memory query in reset(), memory store in finalize()
- ProviderPool wired to TopologyRunner
- TopologyController passed to runner

**rewardflow.py** — New file:
```python
class RewardFlowPropagator:
    def __init__(self, damping: float = 0.85, max_iters: int = 20):
        ...
    def compute(self, rollouts: list[EpisodeTrace]) -> list[dict[int, float]]:
        """Build state-graph from K rollouts, propagate via PageRank.
        Returns: list of {node_idx: reward} dicts, one per rollout."""
    def _build_state_graph(self, rollouts) -> dict:
        """States = (role, quality_bucket). Transitions = consecutive nodes."""
    def _pagerank(self, graph, terminal_rewards) -> dict[str, float]:
        """Personalized PageRank with terminal states as seeds."""
```

**training_memory.py** — New file:
```python
class TrainingMemory:
    def __init__(self, db_path: str = "data/training_memory.db"):
        ...
    def query_similar(self, prompt_embedding: np.ndarray, k: int = 3) -> list[dict]:
        """Cosine similarity on precomputed 768-dim embeddings. Returns top-k episodes."""
    def store_episode(self, trace: EpisodeTrace, embedding: np.ndarray) -> None:
        """Persist outcome + topology + per-node results to SQLite."""
    def format_context(self, episodes: list[dict]) -> str:
        """Format top-K episodes as text for model observation injection."""
    def precompute_embeddings(self, prompts: list[str]) -> None:
        """Offline: compute arctic-embed-m embeddings, store in DB."""
```

**reward.py** — Extend:
- Add R_resilience scoring from trace
- Add R_cost_efficiency from node costs
- Integrate RewardFlow at batch level
- New weights: 0.20/0.35/0.20/0.15/0.10

**convert_sft_to_verl.py** — Extend:
- Load gpt54_adaptive_topologies.jsonl (120 entries)
- Load gpt54_static_to_adaptive.jsonl (60 entries, use topology_adaptive)
- Load gpt54_recovery_scenarios.jsonl (40 entries, use initial_topology)
- Total: 2185 entries

### Phase 2: Local Training Setup

**Model:** Qwen3.5-4B with Unsloth QLoRA (~5GB VRAM on RTX 3500 Ada 12GB)
**Framework:** Unsloth + TRL GRPOTrainer
**Script:** `scripts/train_local_grpo.py`

**Phase A — Structural (GRPO, $0 API):**
- Dataset: 2185 prompts, 3-5 epochs, batch_size=4
- Reward: format + density + adaptation_bonus (presence of fallback_tier, checkpoints)
- Memory: starts empty, offline enrichment between epochs
- Duration: ~2-4h
- Success: reward/mean > 0.5, loss decreasing

**Phase B — Execution (GRPO, API calls):**
- Dataset: 600 curated, 5 epochs, batch_size=2
- Reward: full 5-signal (structural + execution + rewardflow + resilience + cost)
- ProviderPool: 8 providers active
- Memory: pre-populated from Phase A results
- Duration: ~4-8h
- Success: reward/mean > 0.6, PASSED rate > 30%

**Auto-recovery in training script:**
- OOM → halve batch_size (4→2→1), retry
- API timeout → skip, use structural reward fallback
- Provider rate limit (429) → exponential backoff (1s→2s→4s) + fallback to structural reward for that batch
- NaN loss → rollback to last checkpoint, reduce lr 50%
- Unknown error → log full traceback, stop cleanly

### Phase 3: Pod Deployment (after local validation)

**Model:** Qwen3.5-9B on H100 80GB
**Framework:** verl-agent with GiGPO (multi-turn)
**Key difference from local:** True multi-turn revision (model decides at each checkpoint), online memory query, GiGPO step-level advantage

## Data

| Source | Entries | Type |
|--------|---------|------|
| SFT v2 combined | ~1532 | Static topologies (BigCodeBench, CodeContests) |
| RAFT Phase 2 | 199 | Execution-verified, static |
| GPT-5.4 complex | 144 | 5-7 nodes, static |
| GPT-5.4 codeforces | 20 | Competition, static |
| GPT-5.4 deep reasoning | 20 | Chain-of-thought |
| GPT-5.4 simple calibrated | 20 | 1-3 nodes |
| GPT-5.4 error correction | 20 | v1→v2 pairs |
| GPT-5.4 audit | 10 | Improved |
| **GPT-5.4 adaptive (NEW)** | **120** | Topologies with adaptation metadata |
| **GPT-5.4 static→adaptive (NEW)** | **60** | Static to adaptive conversion |
| **GPT-5.4 recovery (NEW)** | **80** | Recovery scenarios: 40 initial + 40 recovered topologies |
| **TOTAL** | **~2225** | (exact count after dedup via convert_sft_to_verl.py) |

Curated for Phase B: 600 entries, adaptive data prioritized (all 260 adaptive entries included).

## Files Changed/Created

| Action | File | Change |
|--------|------|--------|
| MODIFY | sage-core/src/topology/topology_graph.rs | +3 TopologyNode fields, +3 TopologyGraph fields |
| MODIFY | sage-core/src/topology/reward.rs | +2 RewardScore fields, compute_full() |
| MODIFY | sage-python/src/sage/verl/topology_env.py | Multi-turn, memory, ProviderPool, controller |
| MODIFY | sage-python/src/sage/verl/reward.py | +resilience, +cost_efficiency, RewardFlow |
| MODIFY | sage-python/scripts/verl/convert_sft_to_verl.py | +3 new data sources |
| CREATE | sage-python/src/sage/verl/rewardflow.py | PageRank reward propagation |
| CREATE | sage-python/src/sage/verl/training_memory.py | SQLite episodic memory |
| CREATE | sage-python/scripts/train_local_grpo.py | Unsloth GRPO local training |

## Success Criteria

### Local Validation (Qwen3.5-4B)
- [ ] Training completes without crash (both phases)
- [ ] reward/mean increases across epochs
- [ ] Model generates valid adaptive YAML (with fallback_tier, checkpoints)
- [ ] Memory is populated and queried (SQLite has entries)
- [ ] ProviderPool resolves correctly (multi-provider execution)
- [ ] At least 1 adaptation triggered and succeeded during Phase B

### Pod Training (Qwen3.5-9B)
- [ ] GiGPO step_advantage is non-zero (proves multi-turn works)
- [ ] BigCodeBench Hard > 40% (above current 37.8%)
- [ ] Learned topologies beat static templates on 20-task sample
- [ ] Model uses memory context to improve decisions

## What Makes This Novel

1. **Learned topology + runtime adaptation** — model generates DAGs with embedded recovery policies
2. **Multi-turn revision via GiGPO** — model learns WHEN and HOW to correct
3. **Episodic memory during training** — model exploits cumulative experience
4. **Multi-provider per-node** — model learns cost/quality tradeoffs across 8 providers
5. **RewardFlow per-node credit** — dense reward signal via topology-aware PageRank
6. **CARD-style price penalty** — explicit cost optimization in reward

**Publication title:** "Learned Adaptive Multi-Agent Topologies with Runtime Recovery and Episodic Memory"

No existing system combines these six elements. Closest: AgentConductor (learned, static, single-provider), OpenSage (adaptive, not learned), CARD (conditional, not RL-trained with adaptation).
