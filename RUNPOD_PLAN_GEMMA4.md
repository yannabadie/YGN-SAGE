# YGN-SAGE Gemma4 — Plan d'entraînement topology (RunPod H200)

> **Ce document est la référence pour l'opérateur (humain ou Claude Code) sur le pod.**
> Mis à jour: 2026-04-03 — Nouveau modèle MoE Gemma4-26B-A4B.

---

## Contexte

- **Branche** : `local-GM4`
- **Modèle** : `google/gemma-4-26B-A4B-it` (MoE, 25.2B total / 3.8B actifs, 128 experts top-8+1)
- **Hypothèse** : MoE qualité supérieure → meilleures topologies que Qwen3-4B (Phase C : 0.922 N1, 40% MASBENCH)
- **Format** : on entraîne Gemma4 à générer `<tool_call>` JSON (format SAGE) via SFT + GRPO
- **License** : Apache 2.0

### Pourquoi Gemma4 ?

Gemma4-26B-A4B-it est un MoE avec seulement 3.8B paramètres actifs par forward pass (comparable en coût inférence à Qwen3-4B) mais dispose de 25.2B paramètres totaux répartis sur 128 experts. L'hypothèse est que la diversité des experts produit des topologies plus variées et de meilleure qualité que Qwen3-4B, tout en gardant un coût inférence faible pour Path 6.

---

## Infrastructure

| Composant | Spécification |
|-----------|---------------|
| GPU | 1x H200 SXM 141GB |
| Framework | TRL (SFTTrainer + GRPOTrainer) + PEFT LoRA |
| Modèle | google/gemma-4-26B-A4B-it |
| Précision | BF16 (pas de quantification — H200 a assez de VRAM) |
| LoRA | r=16, alpha=32, targets: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj |

---

## Gotchas critiques

1. **`mm_token_type_ids`** — OBLIGATOIRE dans chaque forward pass (même text-only). `Gemma4DataCollator` le gère automatiquement
2. **`remove_unused_columns=False`** — sinon Trainer drop `mm_token_type_ids`
3. **BF16 obligatoire** — fp16 cause NaN sur Gemma4
4. **Ghost thoughts** — Gemma4 peut générer des "ghost" thought channels (`<|channel>thought`) — surveiller dans les outputs
5. **Tokenizer** — Gemma4 utilise `<|turn>` / `<turn|>` (PAS `<|im_start|>` / `<|im_end|>`)
6. **Format drift** — Le modèle a un format de chat natif différent de notre `<tool_call>` JSON ; le SFT doit être suffisant pour ancrer le format avant GRPO

---

## Phase 1 : SFT (2-4h estimées)

### Objectif
Le modèle apprend à générer `<tool_call>` JSON (format SAGE) au lieu de son format natif.

### Configuration
- **Script** : `python scripts/train_gemma4_topology.py --sft-data data/v2_gemma4_balanced.jsonl --sft-only`
- **Dataset** : 8950 exemples rééquilibrés (5694 create_topology 64% + 1686 adapt 19% + 1570 multi-turn 18%)
- **Epochs** : 2
- **Learning rate** : 1e-5
- **VRAM estimé** : ~66GB (model 50GB + LoRA + optimizer + activations)

### Critères de succès
- [ ] loss < 0.3
- [ ] N1 avg >= 0.85

### Évaluation
```bash
python scripts/eval_reward_holdout_gemma4.py \
    --adapter models/gemma4_topology/sft_checkpoint \
    --model google/gemma-4-26B-A4B-it
```

---

## Phase 2 : GRPO (4-8h estimées)

### Objectif
Améliorer la qualité des topologies via reinforcement learning avec reward 5-signal.

### Configuration
- **Script** : `SAGE_VERL_EXEC=0 SAGE_TRAINING_PHASE=C python scripts/train_gemma4_topology.py --adapter models/gemma4_topology/sft_checkpoint --data data/v2_gemma4_balanced.jsonl`
- **K** : 4 rollouts
- **Learning rate** : 3e-6
- **Beta** : 0.0
- **Temperature** : 1.0
- **Format drift monitoring** : callback every 50 steps

### Critères de succès
- [ ] N1 avg >= 0.90
- [ ] Format compliance > 85%

### Risque principal
Format drift vers format natif Gemma4 — augmenter `format_reward` si nécessaire. Le GRPO peut « oublier » le format SFT si le reward structurel n'est pas assez fort.

---

## Phase 3 : Évaluation comparative

### Benchmarks
1. **MASBENCH depth** (50 tasks) : Gemma4 GRPO vs Qwen3-4B Phase C (baseline 40%)
2. **BigCodeBench Hard** si résultats prometteurs

### Commandes
```bash
# MASBENCH
SAGE_ENABLE_PATH6=1 \
SAGE_PATH6_ADAPTER="${GRPO_OUTPUT}" \
SAGE_PATH6_MODEL="google/gemma-4-26B-A4B-it" \
SAGE_PATH6_MODEL_TYPE=gemma4 \
python3 -m sage.bench --type masbench --limit 50

# BigCodeBench (si MASBENCH > 40%)
SAGE_ENABLE_PATH6=1 \
SAGE_PATH6_ADAPTER="${GRPO_OUTPUT}" \
SAGE_PATH6_MODEL="google/gemma-4-26B-A4B-it" \
SAGE_PATH6_MODEL_TYPE=gemma4 \
python3 -m sage.bench --type bigcodebench --subset hard --split instruct --limit 50
```

### Décision
- **MASBENCH > 40%** : garder Gemma4, wire dans Path 6
- **MASBENCH <= 40%** : rester sur Qwen3-4B, archiver résultats

---

## Post-Training

1. Upload adapter HuggingFace : `yannabadie/sage-topology-policy-gemma4`
2. Si MASBENCH > 40% : wire dans Path 6 via `SAGE_PATH6_MODEL_TYPE=gemma4`
3. Sauvegarder métriques dans `sage-python/experiments/`

---

## Monitoring

```bash
# Suivre le training en temps réel
tail -f models/gemma4_topology/sft_metrics.jsonl | python3 -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l)
    print(f'step={d[\"step\"]:4d} loss={d[\"loss\"]:.4f}')
"

# Vérifier VRAM
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Vérifier format compliance pendant GRPO
grep 'format_compliance' models/gemma4_topology/grpo_metrics.jsonl | tail -5
```

---

## Fallbacks

| Problème | Solution |
|----------|----------|
| SFT loss ne converge pas | lr → 5e-6, epochs → 3 |
| NaN dans loss | Vérifier BF16 partout (pas fp16) |
| `mm_token_type_ids` pas passé au modèle | Subclass SFTTrainer, override `compute_loss` |
| Format drift GRPO > 20% | Augmenter `format_reward`, réduire temperature à 0.8 |
| OOM | `gradient_checkpointing` + reduce `max_length` |
| Gemma4 ghost thoughts | Ajouter `<|channel>` aux banned tokens |
| LoRA ne converge pas | Augmenter rank (r=32), vérifier que tous les modules targets sont corrects |
| GRPO reward plateau | Vérifier que les topologies générées sont variées (pas de mode collapse) |

---

## Baseline de comparaison

| Métrique | Qwen3-4B Phase C | Target Gemma4 |
|----------|------------------|---------------|
| N1 avg reward | 0.922 | >= 0.90 |
| MASBENCH depth | 40% (4/10) | > 40% |
| Format validity | ~95% | > 85% |

---

## Références

| Papier | Usage |
|--------|-------|
| Gemma 4 Technical Report (Google, 2026) | Architecture MoE, mm_token_type_ids, BF16 |
| DAPO (arXiv 2503.14476) | Token-level loss, asymmetric clipping |
| AgentConductor (arXiv 2602.17100) | RL topology evolution, S_complex density |
| Graph-GRPO (arXiv 2603.02701) | Edge-level credit pour topologies |
