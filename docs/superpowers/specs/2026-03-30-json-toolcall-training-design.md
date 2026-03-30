# JSON Tool-Call Training Pipeline — Design Spec

**Date**: 2026-03-30
**Branch**: `local`
**Replaces**: YAML-based training (deprecated after ablation showing format matters more than hyperparams)

## Goal

Train Qwen3-4B-Instruct to orchestrate the full SAGE Rust+Python pipeline via `<tool_call>` JSON, using all 7 SAGE tools. Deploy as Path 6 in TopologyEngine.

## Key Decision: Why Tool-Call JSON

| Evidence | Source |
|----------|--------|
| 91% YAML malformation rate in pod training | TRAINING_LOG.md |
| reward.py scores `<tool_call>` +1.0 vs YAML +0.5 | reward.py:_score_format() |
| Qwen3 Instruct has native `<tool_call>` (Hermes format) | Qwen docs |
| LLMs have 100x more JSON than YAML in pretraining | Common knowledge |
| Main branch already switched (commit e129c48) | git log |
| RL-Struct validates GRPO on Qwen3-4B for JSON (arXiv 2512.00319) | arXiv |

## Output Format

The model produces `<tool_call>`-wrapped JSON:

```xml
<tool_call>
{"name": "create_topology", "arguments": {
  "difficulty": "moderate",
  "reasoning": "Multi-step code task with verification",
  "nodes": [
    {"role": "coder", "model_tier": "codex", "prompt": "..."},
    {"role": "reviewer", "model_tier": "fast", "prompt": "..."}
  ],
  "edges": [{"from_idx": 0, "to_idx": 1, "flow_type": "message"}]
}}
</tool_call>
```

## 7 SAGE Tools

Defined in the system prompt, matching the Rust+Python pipeline:

| Tool | Description | Rust Module |
|------|-------------|-------------|
| `create_topology` | Design multi-agent DAG | TopologyEngine |
| `route_task` | Classify S1/S2/S3 | SystemRouter + kNN (92%) |
| `assign_models` | Map model_tier → real model | ModelAssigner (cards.toml) |
| `verify_topology` | Formal verification | HybridVerifier (Z3/OxiZ + LTL) |
| `adapt_topology` | Runtime adaptation | Online evolution (upgrade/reroute) |
| `execute_code` | Sandbox execution | 3-layer (tree-sitter → Wasm → subprocess) |
| `manage_memory` | S-MMU operations | Arrow STM + SQLite + Entity Graph |

## Training Phases

### Phase A — SFT Tool-Call Warmup

**Purpose**: Teach the model SAGE topology content in its native `<tool_call>` format.

- **Model**: `Qwen/Qwen3-4B-Instruct` (NF4 4-bit, LoRA r=32)
- **Data**: 1880 topologies converted to `<tool_call>` JSON format
  - System prompt includes 7 tool definitions (~1800 tokens)
  - Ground truth: `<tool_call>{"name": "create_topology", "arguments": <topology>}</tool_call>`
  - Only `create_topology` has ground truth; other 6 tools learned via RL
- **Hyperparams**: lr=2e-5, 2 epochs, batch=1, grad_accum=8, cosine schedule
- **Duration**: ~30 min on RTX 3500 Ada
- **Exit criteria**: loss < 1.5, model generates valid `<tool_call>` JSON

### Phase B — GRPO Structural Reward

**Purpose**: Optimize topology quality via RL with hierarchical reward.

- **Reward** (RL-Struct inspired, arXiv 2512.00319):
  - Validity: is it parseable JSON? (weight 1.0) — from `_score_format()`
  - Structure: nodes, edges, roles present? (weight 1.0) — from `_score_structure()`
  - Format: `<tool_call>` wrapper? (weight 0.5) — from `_score_format()` bonus
  - Rust density: TopologyEngine score (weight 0.5) — from `_score_rust_density()`
  - All ALREADY computed by `compute_score()` in reward.py
- **Hyperparams** (DAPO-inspired, arXiv 2503.14476):
  - lr=5e-6, temp=1.0, K=4, KL=0 (no KL penalty)
  - max_completion_length=1024
  - 250 steps, LoRA r=32
- **Duration**: ~2-4h (250 steps × ~30-60s/step)
- **Exit criteria**: N1 avg > 0.5, P(reward > 0.3) > 50%

### Phase C — Execution Reward

**Purpose**: The model learns which topologies produce working code.

- **Reward**: `SAGE_VERL_EXEC=1` — 30% structural + 70% execution
  - TopologyRunner executes real API calls (7 providers)
  - Sandbox tests generated code
  - Graduated: PASSED=1.5, WRONG_ANSWER=1.0, RUNTIME_ERROR=0.7, TIMEOUT=0.5
- **Phase C bonuses**: `SAGE_TRAINING_PHASE=C`
  - model_tier correctness: +0.1 × tier_ratio
  - checkpoints placed: +0.1
  - provider_hints: +0.05
  - hybrid LLM+code: +0.1
- **API cost**: ~$0.10/experiment (budget models)
- **Exit criteria**: MASBENCH > 67%, BigCodeBench Hard > 37.8%

## Data Pipeline

### Conversion Script: `convert_sft_to_toolcall.py`

Transforms existing SFT data:

```
Input:  {"prompt": "...", "topology": {...}, "topology_yaml": "..."}
Output: {"prompt": "...", "topology": {...}, "topology_toolcall": "<tool_call>\n{...}\n</tool_call>"}
```

System prompt template (baked per entry):
```
You are a multi-agent topology designer for the YGN-SAGE framework.

<tools>
[{"type": "function", "function": {"name": "create_topology", ...}},
 {"type": "function", "function": {"name": "route_task", ...}},
 ... (7 tools total)]
</tools>

For each task, call the appropriate tool(s) using <tool_call> JSON format.
```

### Datasets

| File | Entries | Format | Use |
|------|---------|--------|-----|
| `topology_sft_v2_toolcall.jsonl` | 1880 | `<tool_call>` JSON | Phase A SFT |
| `sft_complex_heavy_toolcall.jsonl` | 2843 | `<tool_call>` JSON, complex 5x | Phase A variant |
| `verl_topology_train.parquet` | 12303 | Prompts only | Phase B/C GRPO |
| `holdout_50_toolcall.json` | 50 | `<tool_call>` JSON | N1 evaluation |

## Infrastructure (reused from Phase 0)

| Component | Status | Changes needed |
|-----------|--------|----------------|
| `autoresearch_loop.py` | Working | None |
| `eval_reward_holdout.py` | Working (batched) | Update chat template for Instruct |
| `eval_masbench_local.py` | Working | None |
| `eval_bigcodebench_local.py` | Working | None |
| `experiments/journal.jsonl` | 4 entries | Continues |
| `train_local_qwen3_4b.py` | Working | Support tool-call format + Instruct model |

## Hardware Constraints

- RTX 3500 Ada 12 GB VRAM (WDDM, Windows)
- `nvidia-smi -lgc 3105` required
- No vLLM — native generation, batched inference
- DAPO not natively in TRL 0.29.1 — use GRPO with DAPO-inspired params (KL=0, temp=1.0)
- HF_HUB_OFFLINE=1 after first model download
- Keep ≥15 GB free on C: for checkpoints

## Evaluation Cascade (unchanged)

| Level | What | Cost | When |
|-------|------|------|------|
| N1 | Reward on 50 holdout | $0 | Every experiment |
| N2 | MASBENCH depth 20 tasks | ~$0.50 | N1 improves |
| N3 | BigCodeBench Hard 20 tasks | ~$2.00 | N2 improves |

## Exit Criteria (full pipeline)

| Metric | Target | Proof |
|--------|--------|-------|
| N1 reward avg | > 0.7 | Journal entry |
| MASBENCH depth | > 67% (main) | N2 eval |
| BigCodeBench Hard | > 37.8% (current) | N3 eval |
| Path 6 deployed | HuggingFace + SAGE_ENABLE_PATH6 | Working inference |

## References

- RL-Struct (arXiv 2512.00319): GRPO + hierarchical reward, Qwen3-4B, 89.7% JSON accuracy
- DAPO (arXiv 2503.14476): Token-level loss, asymmetric clip, KL=0
- ToolOrchestra (arXiv 2511.21689): Nemotron GRPO for multi-turn tool calling
- The Conductor (arXiv 2512.04388): SFT → GRPO, binary reward converges in 200 iters
- AgentConductor (arXiv 2602.17100): RL topology evolution, 97.5% HumanEval
- Graph-GRPO (arXiv 2603.02701): Edge-level credit assignment
