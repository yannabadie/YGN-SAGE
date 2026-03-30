# local — Training local Qwen3-4B sur RTX 3500 Ada 12 GB

## Scope
Training Qwen3-4B comme Path 6 du TopologyEngine SAGE.
Le modèle apprend à orchestrer le pipeline Rust+Python complet via `<tool_call>` JSON.
Itération rapide avant déploiement pod H100.

## Stack
- **Model**: Qwen/Qwen3-4B (base, 4-bit NF4, tool-call natif via chat template Hermes)
- **Format**: `<tool_call>` JSON — pas YAML (91% malformation YAML historique, +1.0 reward bonus)
- **LoRA**: rank 32, alpha 64, cibles attention + MLP
- **Trainer**: TRL 0.29.1 SFTTrainer + GRPOTrainer, PEFT, bitsandbytes
- **Reward**: `sage.verl.reward.compute_score` — scores tool_call +1.0, JSON +0.5, YAML +0.5
- **Outils**: 2 tools seulement (create_topology + adapt_topology). Les 5 autres (routing, assignment, verification, execution, memory) sont gérés par Rust.

## Key Files
- `scripts/train_local_qwen3_4b.py` — Pipeline SFT → GRPO
- `scripts/sage_tool_schemas.py` — 2 tool definitions + system prompt
- `scripts/eval_reward_holdout.py` — N1 evaluator (batched, ~10 min)
- `scripts/autoresearch_loop.py` — Boucle expérimentale autonome
- `scripts/convert_sft_to_toolcall.py` — Conversion YAML → tool-call JSON
- `scripts/enrich_topologies_adaptive.py` — Enrichissement adaptation/checkpoints
- `scripts/generate_expert_topologies.py` — Distillation Claude Opus 4.6

## Données
| Fichier | Entrées | Format | Usage |
|---------|---------|--------|-------|
| `topology_sft_v2_toolcall.jsonl` | 1880 | `<tool_call>` JSON | Phase A SFT |
| `topology_sft_v2_adaptive.jsonl` | 1880 (83% checkpoints) | `<tool_call>` JSON + adaptation | Phase C |
| `expert_topologies.jsonl` | 8 | `<tool_call>` JSON, 63% complex | Distillation haute qualité |
| `verl_topology_train.parquet` | 12303 | Prompts | Phase B GRPO |
| `holdout_50_toolcall.json` | 50 | `<tool_call>` JSON | Évaluation N1 |

## Commands
```bash
# Prerequisites
nvidia-smi -lgc 3105                    # Lock GPU clocks (CRITIQUE)
powercfg //setactive 8c5e7fda-...       # High Performance

# Phase A: SFT warmup (~60 min)
cd sage-python
HF_HUB_OFFLINE=1 python -u scripts/train_local_qwen3_4b.py \
  --config experiments/configs/toolcall_sft_full.json --sft-only

# Phase B: GRPO structural
HF_HUB_OFFLINE=1 python -u scripts/train_local_qwen3_4b.py \
  --adapter models/toolcall_qwen3_4b/sft_checkpoint \
  --max-samples 40 --num-generations 2 --max-completion-length 512

# N1 Eval (batched, ~10 min)
HF_HUB_OFFLINE=1 python -u scripts/eval_reward_holdout.py \
  --adapter models/toolcall_qwen3_4b/sft_checkpoint

# Autoresearch loop
python scripts/autoresearch_loop.py --eval-only \
  --adapter models/toolcall_qwen3_4b/sft_checkpoint
```

## Résultats (2026-03-30)

### YAML baseline (abandonné)
- SFT loss: 2.74 → 0.92 (470 steps, 56 min)
- GRPO post-SFT: avg reward 0.40, max 0.93
- Problème: 91% malformation YAML, signal discret

### JSON Tool-Call (en cours)
- SFT loss: 1.59 → 0.30 (80 steps, ~20 min) — 2.4x plus rapide que YAML
- Format natif Qwen3 → le modèle apprend le CONTENU, pas le FORMAT
- Training en cours, Phase A

## Architecture des outils
```
Qwen3-4B (learned)          Rust pipeline (hardcoded)
├── create_topology  ──────→ TopologyEngine (6 paths)
│   (DAG design)             ├── ModelAssigner (cards.toml)
│                            ├── SystemRouter + kNN (92%)
└── adapt_topology   ──────→ ├── HybridVerifier (Z3/OxiZ)
    (upgrade/reroute)        ├── TopologyRunner + ProviderPool
                             └── S-MMU (Arrow + SQLite)
```

## Roadmap
- Phase 0: Autoresearch infrastructure DONE
- Phase A: SFT tool-call warmup IN PROGRESS (loss 0.30)
- Phase B: GRPO structural reward (pending)
- Phase C: Execution reward + adaptation (pending)
- Phase 4: Deploy Path 6 + BigCodeBench proof (pending)

## Décisions clés (avec justification)
1. **JSON tool-call > YAML** — LLMs ont 100x plus de JSON en pretraining, reward +1.0 vs +0.5
2. **2 outils > 7** — Les 5 modules Rust surpassent le 4B (kNN 92%). Economise ~500 tokens context
3. **Qwen3-4B base** — A déjà tool-call natif (chat template Hermes 4168 chars). Pas de variante Instruct séparée
4. **SFT avant GRPO** — Prouvé mathématiquement: P(JSON valide | pi_base) = 0 → GRPO advantage = 0

## Contraintes hardware
- RTX 3500 Ada 12 GB VRAM, WDDM, Windows
- `nvidia-smi -lgc 3105` obligatoire (sinon GPU à 300 MHz)
- HF_HUB_OFFLINE=1 (SSL issues avec HuggingFace)
- Garder >=15 GB libres sur C: pour checkpoints

## Références
- RL-Struct (arXiv 2512.00319) — GRPO + JSON, Qwen3-4B, 89.7% accuracy
- DAPO (arXiv 2503.14476) — Token-level loss, KL=0
- The Conductor (arXiv 2512.04388) — SFT → GRPO for topology
- AgentConductor (arXiv 2602.17100) — 97.5% HumanEval with 3B
