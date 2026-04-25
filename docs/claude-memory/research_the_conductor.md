---
name: The Conductor — Critical Competitor (ICLR 2026, Sakana AI)
description: Qwen2.5-7B GRPO topology + multi-provider, BigCodeBench 40.0%. Invalidates SAGE's original differentiation claim.
type: reference
---

## The Conductor (arXiv 2512.04388, ICLR 2026)
- **Authors:** Sakana AI + University of Michigan + Institute of Science Tokyo
- **Published:** December 2025, accepted ICLR 2026, last revision March 1, 2026
- **Code:** NOT released (as of March 22, 2026)

## What It Does
- Qwen2.5-7B trained via GRPO end-to-end
- Designs communication topologies in natural language
- Supports RECURSIVE topologies (Conductor selects itself as worker)
- Multi-provider: GPT-5, Claude-Sonnet-4, Gemini-2.5-Pro, DeepSeek-R1-32B, Gemma3-27B, Qwen3-32B

## Benchmarks
- LiveCodeBench: 83.93%
- GPQA-Diamond: 87.5%
- MATH500: 99.4%
- **BigCodeBench: 37.86% (40.0% with recursion)**
- MMLU: 94.1%
- AIME25: 93.3%

## Impact on SAGE
**The old claim "no system combines learned + adaptive + multi-provider" is INVALID.**
The Conductor does all three (learned via GRPO, recursive adaptation, 6 providers).

## What SAGE Still Has Over The Conductor
1. **Rust engine** — formal verification (SMT/LTL), Rust density
2. **Fallback tiers + checkpoints** — adaptation embedded IN the topology YAML
3. **Edge-level credit** — GiGPO step-level + Graph-GRPO per-edge
4. **Episodic memory** — model learns from past episodes across training
5. **Self-hosted** — Qwen3.5-9B runs locally (Q8 GGUF 12GB)
6. **Open-source** — Conductor has no code released
7. **RewardFlow** — per-node PageRank credit (dense rewards)

## Updated Target
SAGE must beat **40.0%** on BigCodeBench Hard (not 37.8%).

**How to apply:** The Conductor is now the primary competitor, not AgentConductor. Update all strategy documents. The differentiation claim should be: "SAGE is the only OPEN-SOURCE system combining RL-trained topology with Rust formal verification, edge-level credit, episodic memory, and self-hosted multi-provider orchestration with fallback tiers."
