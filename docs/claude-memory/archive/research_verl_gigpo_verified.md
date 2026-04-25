---
name: veRL + GiGPO Verified Research (March 20, 2026)
description: Verified findings on GiGPO step-level gradient, veRL API, AgentConductor, Graph-GRPO, Qwen3.5-9B issues — evidence-based decisions for topology training
type: reference
---

## GiGPO Step-Level Gradient — VERIFIED in source code

- Read `core_gigpo.py` from verl-agent (langfengQ/verl-agent)
- Each step is a SEPARATE training sample with its own tokens
- Step advantage broadcasts to ALL tokens of that step's response: `scores.unsqueeze(-1).tile([1, response_length]) * response_mask`
- `build_step_group()` clusters observations by hash (exact) or SequenceMatcher (fuzzy threshold 0.95)
- Combined: `scores = episode_advantages + step_advantage_w * step_advantages`
- **For single-action (model acts once at step 0): GiGPO = GRPO exactly**
- Step-level only helps if model generates at EACH step

## veRL API for Multi-Turn (verified from docs)

### Two approaches:
1. **BaseTool** — tool-calling pattern, model calls tools at each step
   - `execute()` → `(ToolResponse, float_score, dict_info)`
   - Config: `rollout.multi_turn: True`, `rollout.name: "sglang"`
2. **BaseInteraction** — conversational feedback, model sees env responses
   - `generate_response()` → `(should_terminate, response_text, score, metadata)`
   - Better fit for SageTopologyEnv

### Reward function API:
```python
# veRL signature (sync or async, auto-detected):
def compute_score(data_source, solution_str, ground_truth, extra_info) -> float
# OR async with reward router:
async def compute_score(data_source, solution_str, ground_truth, extra_info, reward_router_address, reward_model_tokenizer) -> dict
```

### RewardLoopWorker:
- `compute_score_batch` splits into individual items via `asyncio.create_task` + `asyncio.gather`
- Async recommended for external API calls (DeepSeek, Gemini)

## AgentConductor — EXACT technical details (2602.17100)

- Model: Qwen2.5-3B-Instruct (NOT 9B)
- Algorithm: GRPO standard (NOT GiGPO)
- SFT: LLaMA-Factory, lr=1e-4, batch_size=4, LoRA
- RL: veRL + vLLM, standardized advantages
- Data: 4500 samples (2700 competition 2-turn + 300 basic), GPT-4o generated
- S_complex = exp(S_node + 2·S_edge + S_depth)
  - S_node = exp(-|V|/N_max(l))
  - S_edge = exp(-|E|/(|V|(|V|-0.5)))
  - S_depth = 1 - s/|V|
- N_max: easy=4, medium=7, hard=10 (matches SAGE implementation)
- Reward: r_e(execution) + r_g(density)
  - r_e: PASSED=1.5, WRONG_ANSWER=1.0, RUNTIME_ERROR=0.7, NO_YAML=-2.0
  - r_g: S_complex if |V|≤N_max, tanh((N_max-|V|)/N_max) otherwise
- 2nd-turn: feedback z_k appended to history, model regenerates topology
- No BigCodeBench submission

## Graph-GRPO — Edge-level credit (2603.02701)

- Per-edge success rate: S_ij = Σ[I(e_ij∈G_k)·r_k] / Σ[I(e_ij∈G_k)]
- Edge advantage: A_ij = (S_ij - μ_S) / (σ_S + ε)
- Loss: L(θ) = (1/|E|) Σ[-A_ij log π_θ(e_ij|Q) + β D_KL]
- K=16 topologies per prompt (Bernoulli sampling from learned probability matrix)
- DAG constraint: (P_θ)_ij = σ(h_i W h_j^T) if j<i, else 0
- Result: 92.45% vs 90.67% graph-level (+1.82%)
- No code released
- First to apply GRPO to discrete structure search

## Qwen3.5-9B Issues — VERIFIED March 20, 2026

- Architecture: Gated DeltaNet + full attention hybrid (DENSE, NOT MoE — the MoE is 35B-A3B only)
- Released: Feb 27, 2026
- vLLM < 0.17.0: CONFIRMED not supported (issue #35391)
- vLLM 0.17.0: CONFIRMED CUDA illegal memory bug (issue #36408). WORKAROUND: disable MTP (`num_speculative_tokens=0`)
- Docker `verlai/verl:vllm017.latest` EXISTS (pushed 2026-03-12, verified via pagination beyond first 100 tags)
- Contents: CUDA 12.9.1, Python 3.12, PyTorch 2.10.0, vLLM 0.17.0, FA 2.8.3, cuDNN 9.16, TE 2.12
- Warning: CUDA 12.9.1 requires NVIDIA driver >= 575.57.08 (use Secure Cloud or H100 nodes)
- Issue #5441: NPU tracking only, NOT a GPU bug
- **Qwen3.5-9B IS USABLE on H100 with Docker verlai/verl:vllm017.latest + MTP disabled (num_speculative_tokens=0)**
- **FALLBACK: Qwen2.5-7B-Instruct if Qwen3.5 still crashes**

## veRL Engineering Handbook Key Settings (from HuggingFace blog)

- gpu_memory_utilization=0.8 optimal (0.6 wastes 20GB, 0.9 causes OOM)
- lora_rank >= 32 minimum, 64 recommended for 9B
- tensor_model_parallel_size=1 for < 14B models
- train_batch_size=1024 (default 16 too conservative)
- rollout.n=5 (n=10 adds +70% time for marginal gain)
- enable_gradient_checkpointing=True
- param_offload=True for ref model
- Total cost: ~$40 on RunPod for Qwen2.5-3B training

## SAGE Advantages Over AgentConductor

Already implemented:
- Larger model capacity (9B vs 3B)
- Formal verification (OxiZ SMT) — they have none
- 5-path topology generation (MAP-Elites, MCTS, CMA-ME, LLM synthesis, templates) — they use templates only
- Online evolution (MAP-Elites + bandit) — they don't self-adapt
- BigCodeBench ready (37.8%) — they haven't submitted

To add:
- Edge-level credit (Graph-GRPO): +1.82%
- 2nd-turn correction data (they have 2700, we have 0)
- GPT-5.4 Pro distillation (5 prompt types)
- K=16 instead of K=8
- Async Reward Loop (parallelize API calls)

## SAGE Strategic Position — March 20, 2026

**SAGE can be FIRST to combine all three:**
1. Topology RL (AgentConductor has this)
2. Edge-level credit (Graph-GRPO has this)
3. Multi-turn correction (AgentConductor has this)
**No existing system combines all three as of March 20, 2026.**

## Additional SOTA Papers (Feb-March 2026)

- **Dr. MAS** (2602.08847) — Same author as verl-agent (langfengQ). Agent-wise advantage normalization. Code: github.com/langfengQ/DrMAS
- **AT-GRPO** (2510.11062) — Agent- and Turn-wise grouped RL. +3.87-7.62% on coding/math
- **Tree-GRPO** (2509.21240, ICLR 2026) — Tree search for LLM agent RL. 1.5x more rollouts at same budget. Code: github.com/AMAP-ML/Tree-GRPO
- **M-GRPO** (2511.13288) — Multi-agent deep research with hierarchical credit assignment
- **Hindsight Credit Assignment** (2603.08754) — Credit assignment for long-horizon LLM agents

## BigCodeBench Leaderboard Status

- Last release: v0.2.4 (March 2025) — STALE
- Frontier 2026 models (Claude 4, Gemini 3, o3) NOT submitted by vendors
- SAGE at 37.8% is competitive, no multi-agent system has submitted
- First agentic submission would be a significant milestone

**How to apply:** Use this as the evidence base for the RunPod plan. Start with Qwen2.5-7B (safe), implement edge-level credit, add 2nd-turn data, then migrate to Qwen3.5-9B when vLLM stabilizes. SAGE's unique position: first to combine topology RL + edge credit + multi-turn correction.
