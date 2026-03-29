# local — Training local sur RTX 3500 Ada 12 GB

## Scope
Training Qwen3-4B avec TRL+PEFT+bitsandbytes GRPO sur GPU local.
Itération rapide pour tester reward/data avant déploiement pod H100.

**PAS d'Unsloth** — triton incompatible Windows (feedback_training_issues.md).

## Stack
- **Model**: Qwen/Qwen3-4B (base, 4-bit NF4 via bitsandbytes)
- **LoRA**: rank 32, alpha 64, target all attention + MLP projections
- **Trainer**: TRL GRPOTrainer 0.29.1 + PEFT 0.18.1
- **Reward**: `sage.verl.reward.compute_score` (Phase A = binary YAML checks)
- **Data**: `data/verl_topology_train.parquet` (12303 entries)

## Key Files
- `sage-python/scripts/train_local_qwen3_4b.py` — Script principal
- `sage-python/src/sage/verl/reward.py` — Reward function (SAGE_TRAINING_PHASE=A)
- `sage-python/data/verl_topology_train.parquet` — 12303 entries
- `sage-python/models/local_qwen3_4b_grpo/live_metrics.jsonl` — Metrics temps réel

## Commands
```bash
# Prerequisites (one-time)
nvidia-smi -lgc 3105                    # Lock GPU clocks to max (CRITICAL)
powercfg //setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  # High Performance

# Smoke test (2 steps, ~5 min)
cd sage-python && python scripts/train_local_qwen3_4b.py --smoke

# Training rapide (20 samples, ~25 min)
python scripts/train_local_qwen3_4b.py --max-samples 20 --num-generations 2 \
  --max-completion-length 512 --grad-accum 2

# Training complet (12303 samples — LENT, ~40h sans vLLM)
python scripts/train_local_qwen3_4b.py

# Monitor en temps réel
cat models/local_qwen3_4b_grpo/live_metrics.jsonl
```

## Hardware
- GPU: NVIDIA RTX 3500 Ada (12 GB VRAM), Ada Lovelace
- VRAM usage: ~5.7 GB idle, ~6 GB generation, ~12 GB backward pass
- **CRITIQUE**: `nvidia-smi -lgc 3105` obligatoire — sans ça le GPU tourne à 300 MHz (10% vitesse)

## Contraintes Windows
1. **Pas de triton/Unsloth** — utiliser TRL+PEFT+bitsandbytes directement
2. **Pas de vLLM** — generation autoregressive native (lent)
3. **WDDM mode** — GPU scheduling overhead, utilization % trompeuse
4. **Power management** — lock clocks + High Performance power plan obligatoire
5. **gradient_checkpointing** dans GRPOConfig gère le KV cache toggle automatiquement

## Résultats actuels (2026-03-29)
- **Smoke test**: OK (reward 0.19 en 2 steps)
- **20 samples run**: avg reward 0.134, max 0.225, step time ~77s
- **Plateau**: reward ~0.13-0.15 (Phase A = checks binaires, signal discret)
- **clipped_ratio=1.0**: le modèle ne génère jamais EOS

## Diagnostic reward Phase A
La reward function Phase A (`_score_structure`) utilise des checks binaires:
- nodes présents? +0.3
- edges présents? +0.2
- roles corrects? +0.3
- reasoning? +0.2
→ Seulement ~21 valeurs uniques possibles.
Toutes les rewards observées (0.08-0.22) = YAML invalide.

**Pour dépasser 0.3**: besoin que le modèle génère du YAML valide avec nodes+edges.
Options: plus de samples, SFT warmup, ou passer à Phase C (SAGE_TRAINING_PHASE=C).

## Espace disque
- Cache HF: `~/.cache/huggingface/hub/` — Qwen3-4B = 7.6 GB
- **Nemotron-8B SUPPRIMÉ** du cache local (31 GB récupérés) — c'est pour le pod H100
- Garder ≥20 GB libres sur C: pour les checkpoints
