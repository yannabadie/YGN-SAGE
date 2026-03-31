# local — Training local Qwen3-4B sur RTX 3500 Ada 12 GB

## Scope
Training Qwen3-4B comme Path 6 du TopologyEngine SAGE.
Le modèle apprend à orchestrer le pipeline Rust+Python complet via `<tool_call>` JSON.
2 outils : `create_topology` (DAG design) + `adapt_topology` (runtime adaptation).
Les 5 autres modules SAGE (routing, assignment, verification, execution, memory) sont gérés par Rust.

## Stack
- **Model**: Qwen/Qwen3-4B (base, 4-bit NF4, tool-call natif via chat template Hermes 4168 chars)
- **Format**: `<tool_call>` JSON — pas YAML (91% malformation YAML historique, reward +1.0 vs +0.5)
- **LoRA**: rank 32, alpha 64
- **Trainer**: TRL 0.29.1 SFTTrainer + GRPOTrainer, PEFT, bitsandbytes
- **Reward**: `sage.verl.reward.compute_score` — priority: tool_call > JSON > YAML
- **HuggingFace**: yannabadie/sage-topology-policy-local

## Results (March 31, 2026)

### Phase A — SFT Tool-Call (DONE)
- Data: 1880 topologies in `<tool_call>` JSON format
- Loss: 1.59 → 0.225 (4.1x better than YAML 0.92)
- **N1 avg: 0.865** (YAML baseline was 0.391, +121%)
- Simple 0.780, Moderate 0.949, Complex 0.837

### Phase B — GRPO Structural (DONE, ceiling reached)
- 20 steps, K=4, lr=5e-6, KL=0
- N1 avg: 0.849 (structural reward saturated, no improvement over SFT)

### Phase C.1 — SFT Adaptive (DONE)
- Continued SFT from 0.865 checkpoint on 3000 adaptive entries
- Data: 1880 topologies with checkpoints + 1120 adapt_topology decisions
- Loss: 0.142
- **N1 avg: 0.922** (+6.6% vs Phase A)
- Simple 0.930, Moderate 0.966, Complex 0.854

### NOT DONE
- GRPO execution reward (SAGE_VERL_EXEC=1) — never tested with real API calls
- MASBENCH depth validation — never run
- BigCodeBench Hard — never run
- Path 6 wiring in TopologyEngine — not integrated yet

## 7 Providers (all verified March 31)
| Provider | URL | Status |
|----------|-----|--------|
| DeepSeek | api.deepseek.com/v1 | OK |
| OpenAI | api.openai.com/v1 | OK |
| Google | generativelanguage.googleapis.com | OK (SAGE_SSL_VERIFY=false) |
| Grok/xAI | api.x.ai/v1 | OK |
| Kimi | api.moonshot.ai/v1 | OK |
| MiniMax | api.minimax.io/v1 | OK |
| OpenRouter | openrouter.ai/api/v1 | OK (Qwen 3.5 Plus) |

## Key Files
| File | Purpose |
|------|---------|
| `scripts/train_local_qwen3_4b.py` | SFT + GRPO pipeline (supports continued SFT via --adapter + --sft-data) |
| `scripts/sage_tool_schemas.py` | 2 tool definitions (create_topology + adapt_topology) + system prompt |
| `scripts/eval_reward_holdout.py` | N1 evaluator (batched, batch_size=4-8, ~10 min) |
| `scripts/autoresearch_loop.py` | Autoresearch experiment loop |
| `scripts/convert_sft_to_toolcall.py` | Convert YAML SFT data → `<tool_call>` JSON |
| `scripts/enrich_topologies_adaptive.py` | Add adaptation fields (checkpoints, provider_hints) |
| `scripts/generate_adapt_decisions.py` | Generate adapt_topology training data (5139 decisions) |
| `scripts/generate_expert_topologies.py` | Claude Opus 4.6 distilled topologies (8 expert examples) |
| `scripts/upload_to_hf.py` | Push model + data to HuggingFace |
| `scripts/filter_sft_data.py` | Filter/resample SFT data by source/difficulty |

## Datasets (local, also on HuggingFace)
| File | Entries | Format | Phase |
|------|---------|--------|-------|
| `data/topology_sft_v2_toolcall.jsonl` | 1880 | `<tool_call>` JSON | A SFT |
| `data/topology_sft_v2_adaptive_toolcall.jsonl` | 1880 | `<tool_call>` + adaptation | C SFT |
| `data/adapt_decisions_toolcall.jsonl` | 5139 | adapt_topology decisions | C SFT |
| `data/expert_topologies.jsonl` | 8 | Claude Opus distilled | C SFT |
| `data/phase_c_combined.jsonl` | 7027 | Combined Phase C | C SFT |
| `data/verl_topology_train.parquet` | 12303 | Prompts | B GRPO |
| `experiments/holdout_50_toolcall.json` | 50 | Stratified holdout | Eval |

## Model Checkpoints (local, backed up on HuggingFace)
| Path | Phase | N1 Score |
|------|-------|----------|
| `models/toolcall_qwen3_4b/sft_checkpoint/` | A SFT | 0.865 |
| `models/toolcall_qwen3_4b_phase_c/sft_checkpoint/` | C SFT | 0.922 |

## Commands
```bash
# Prerequisites (EVERY session)
nvidia-smi -lgc 3105                                  # Lock GPU clocks
cd /c/Code/worktrees/local/sage-python
pip install -e ".[all,dev]"                           # Reinstall sage after any merge

# SFT
HF_HUB_OFFLINE=1 python -u scripts/train_local_qwen3_4b.py \
  --config experiments/configs/toolcall_sft_full.json --sft-only

# Continued SFT (from existing adapter)
HF_HUB_OFFLINE=1 python -u scripts/train_local_qwen3_4b.py \
  --adapter models/toolcall_qwen3_4b/sft_checkpoint \
  --sft-data data/phase_c_combined.jsonl --sft-epochs 1 --sft-lr 1e-5 --sft-only

# GRPO
HF_HUB_OFFLINE=1 python -u scripts/train_local_qwen3_4b.py \
  --adapter models/toolcall_qwen3_4b_phase_c/sft_checkpoint \
  --max-samples 40 --num-generations 4 --max-completion-length 1024 --batch-size 2

# GRPO with execution reward (real API calls)
SAGE_SSL_VERIFY=false SAGE_VERL_EXEC=1 SAGE_TRAINING_PHASE=C \
  HF_HUB_OFFLINE=1 python -u scripts/train_local_qwen3_4b.py \
  --adapter models/toolcall_qwen3_4b_phase_c/sft_checkpoint \
  --max-samples 20 --num-generations 4 --max-completion-length 1024 --batch-size 2

# N1 Eval
HF_HUB_OFFLINE=1 python -u scripts/eval_reward_holdout.py \
  --adapter models/toolcall_qwen3_4b_phase_c/sft_checkpoint --batch-size 4

# MASBENCH
SAGE_SSL_VERIFY=false python scripts/eval_masbench_local.py \
  --adapter models/toolcall_qwen3_4b_phase_c/sft_checkpoint --limit 20

# Upload to HuggingFace
python scripts/upload_to_hf.py
```

## Architecture
```
Qwen3-4B (learned policy)       Rust pipeline (hardcoded)
├── create_topology ──────────→ TopologyEngine (6 paths)
│   (DAG: nodes + edges +        ├── ModelAssigner (cards.toml, 20 models)
│    difficulty + reasoning)     ├── SystemRouter + kNN (92%)
│                                ├── HybridVerifier (Z3/OxiZ + LTL)
└── adapt_topology ───────────→ ├── TopologyRunner + ProviderPool (7 providers)
    (continue/upgrade/reroute)   └── S-MMU (Arrow STM + SQLite + Entity Graph)
```

## Remaining Work
1. MASBENCH depth 20 tasks (~$0.50) — validate multi-agent benefit
2. Wire Path 6 in `llm_caller.py` → use local Qwen3-4B model
3. BigCodeBench Hard 20 tasks (~$2) — prove framework delta
4. Update HF README with Phase C results
5. Optional: GRPO execution reward for further improvement beyond 0.922

## Constraints
- RTX 3500 Ada 12 GB VRAM, WDDM, Windows
- `nvidia-smi -lgc 3105` required every session (GPU defaults to 300 MHz)
- `HF_HUB_OFFLINE=1` for model loading (SSL issues with HuggingFace)
- `SAGE_SSL_VERIFY=false` for API calls (corporate SSL proxy)
- Never run `git push` while GPU model is loaded (git eats RAM, causes OOM)
- `pip install -e ".[all,dev]"` required after any merge from main

## Design Decisions (with evidence)
1. **JSON tool-call > YAML** — LLMs have 100x more JSON in pretraining, SFT loss 0.225 vs 0.92
2. **2 tools > 7** — Rust modules outperform 4B model (kNN 92%). Saves ~500 tokens context
3. **Qwen3-4B base** — has native tool-call (Hermes template). No separate Instruct variant
4. **SFT before GRPO** — mathematically proven: P(valid JSON | pi_base) ≈ 0 → GRPO advantage ≈ 0
5. **Continued SFT** — Phase C trains on existing adapter, doesn't start from scratch

## References
- RL-Struct (arXiv 2512.00319) — GRPO + JSON, Qwen3-4B, 89.7% accuracy
- DAPO (arXiv 2503.14476) — Token-level loss, KL=0, asymmetric clip
- The Conductor (arXiv 2512.04388) — SFT → GRPO for topology
- AgentConductor (arXiv 2602.17100) — 97.5% HumanEval with 3B
- V2 Adaptive Design — docs/superpowers/specs/2026-03-20-v2-adaptive-topology-design.md
