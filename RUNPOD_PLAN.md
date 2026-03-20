# YGN-SAGE — Plan d'exécution RunPod H100

> **Ce fichier est le seul document à suivre sur le pod.**
> Tout le code est implémenté et testé (51/51 tests). Aucun code à écrire sur le pod.

## Objectif

Entraîner une politique de génération de topologies multi-agents via veRL GRPO sur Qwen3.5-9B.
Résultat attendu : un adaptateur LoRA qui génère des topologies YAML optimales pour des tâches de code,
surpassant AgentConductor (Qwen2.5-3B, arXiv 2602.17100) sur BigCodeBench Hard.

## Contexte technique

| Composant | Valeur |
|-----------|--------|
| Modèle | `Qwen/Qwen3.5-9B` (dense 9B, GatedDeltaNet + attention, Apache 2.0) |
| Fallback | `Qwen/Qwen2.5-7B-Instruct` (si Qwen3.5 crashe) |
| Framework | veRL 0.7.1 + vLLM 0.17.0 (pre-installed in Docker) |
| Docker | `verlai/verl:vllm017.latest` |
| Algorithme | GRPO standard (GiGPO = GRPO pour single-action, vérifié dans core_gigpo.py) |
| LoRA | r=64, alpha=32, target=all-linear |
| GPU | 1x H100 80GB (ou A100 80GB) |
| VRAM estimé | ~68 GB (model 18 + KV cache 45 + optimizer 1 + activations 4) |
| Données | **1965 entries** (SFT v2 1532 + RAFT 199 + GPT-5.4 Pro 234) dans le repo |
| Bug vLLM | `num_speculative_tokens=0` obligatoire (CUDA bug #36408 avec MTP) |
| Reward | Format YAML + structure + Rust density (AgentConductor Eq.9) |
| Innovation | Edge-level credit (Graph-GRPO, arXiv 2603.02701) |

## Prérequis locaux (déjà fait)

- [x] `sage.verl.reward` — fonction de reward veRL (compute_score)
- [x] `sage.verl.edge_credit` — credit par edge Graph-GRPO
- [x] `scripts/verl/train_topology.sh` — config training vérifiée
- [x] `scripts/verl/validate_setup.py` — validation 8 points
- [x] `scripts/verl/benchmark_post_train.py` — éval post-training
- [x] `scripts/verl/convert_sft_to_verl.py` — conversion JSONL → Parquet
- [x] `scripts/verl/export_for_local.py` — export LoRA pour 12GB local
- [x] 51 tests passing, 0 régression

---

## Étapes sur le pod

### Étape 1 — Créer le pod RunPod

1. Aller sur RunPod.io
2. Créer un pod GPU :
   - **GPU** : H100 80GB SXM (ou A100 80GB)
   - **Docker image** : `verlai/verl:vllm017.latest`
   - **Disk** : 150GB+ (modèle 9B bf16 = 18GB, checkpoints, vLLM cache)
   - **Volume** : monter sur `/workspace`
   - **Contenu** : CUDA 12.9.1, PyTorch 2.10.0, vLLM 0.17.0, FlashAttention 2.8.3, TransformerEngine 2.12, cuDNN 9.16 — tout pré-installé. Seul veRL Python package à installer (1 pip install).
   - **Warning** : CUDA 12.9.1 requiert driver NVIDIA >= 575.57.08. Utiliser RunPod **Secure Cloud** (drivers récents garantis) ou nodes H100 (toujours R575+).
3. Attendre que le pod démarre
4. SSH sur le pod

### Étape 2 — Cloner et configurer

```bash
# Cloner le repo
git clone https://github.com/yannabadie/YGN-SAGE.git /workspace/YGN-SAGE
cd /workspace/YGN-SAGE
git checkout VeRLGIGPO

# Vérifier la branche
git log --oneline -3
# Doit montrer le dernier commit de VeRLGIGPO
```

### Étape 3 — Configurer les clés API

Les données de training sont DANS le repo (1965 entries, pas de scp nécessaire).

```bash
# Sur le pod — configurer les clés API :
export DEEPSEEK_API_KEY="<ta clé DeepSeek>"   # REQUIS — primary provider (Chat V3.2, no CoT waste)
export GOOGLE_API_KEY="<ta clé Google>"       # REQUIS — fallback + fast/budget nodes
export HF_TOKEN="<ton token HuggingFace>"     # REQUIS — télécharger Qwen3.5-9B
export WANDB_API_KEY="<ta clé W&B>"           # Optionnel — dashboard training

# Vérifier :
echo "Google: ${GOOGLE_API_KEY:0:10}..."
echo "HF: ${HF_TOKEN:0:10}..."
```

### Étape 4 — Setup automatique (~5 min)

```bash
cd /workspace/YGN-SAGE
bash sage-python/scripts/verl/setup_runpod.sh
```

Ce script :
1. Vérifie le GPU
2. Vérifie que veRL est installé (dans le Docker)
3. Installe le SDK SAGE (`pip install -e "sage-python/.[all,dev]"`)
4. Build sage-core Rust (`maturin develop --features smt,onnx,cognitive,tool-executor --release`)
5. Convertit les données SFT → Parquet veRL
6. Affiche un résumé de vérification

### Étape 5 — Valider l'environnement

```bash
cd /workspace/YGN-SAGE/sage-python
python3 scripts/verl/validate_setup.py
```

**Les 8 checks doivent passer :**
1. GPU — H100/A100 détecté
2. veRL — importable
3. vLLM — version >= 0.17.0
4. sage-core — TopologyGraph, TopologyReward, PyHybridVerifier
5. SAGE SDK — TopologyRunner importable
6. Reward — compute_score retourne un float
7. Data — parquet lisible, N entries
8. API keys — au moins 1 configurée

**Si un check échoue** : corriger avant de continuer. Voir la section Troubleshooting ci-dessous.

### Étape 6 — Lancer le training

Le script fait **2 phases automatiquement** :
- **Phase A** (5 epochs, ~30 min) : reward structural uniquement → apprend le format YAML ($0 API)
- **Phase B** (5 epochs, ~4-8h) : reward execution multi-provider → apprend ce qui MARCHE (~$60-80 API)

```bash
cd /workspace/YGN-SAGE/sage-python

# Optionnel : curater 500 prompts diversifiés (GSM8K cappé, GPT-5.4 Pro prioritaire)
python3 scripts/verl/curate_training_data.py

# Lancer (web terminal : utilise screen/tmux pour ne pas perdre le process)
screen -S train
bash scripts/verl/train_topology.sh 2>&1 | tee train.log
# Ctrl+A D pour détacher, screen -r train pour rattacher
```

**Signaux à surveiller :**
- `reward/mean` devrait augmenter progressivement
- `actor/loss` devrait diminuer
- `kl_divergence` devrait rester < 0.1
- GPU utilization devrait être > 90%

```bash
# Vérifier GPU :
nvidia-smi

# Vérifier les métriques :
grep -E "(reward|loss|epoch)" train.log | tail -10

# Si W&B configuré, ouvrir le dashboard dans le navigateur
```

### Étape 7 — Exporter le modèle

```bash
cd /workspace/YGN-SAGE/sage-python
python3 scripts/verl/export_for_local.py \
    --checkpoint models/topology_verl/ \
    --output models/topology_verl_local/
```

### Étape 8 — Benchmarks post-training

```bash
# Évaluer avec le modèle entraîné
python3 scripts/verl/benchmark_post_train.py \
    --bench all \
    --limit 20 \
    --model models/topology_verl_local/
```

### Étape 9 — Récupérer les résultats

```bash
# Depuis la MACHINE LOCALE :
scp -r <pod>:/workspace/YGN-SAGE/sage-python/models/topology_verl_local/ sage-python/models/
scp <pod>:/workspace/YGN-SAGE/sage-python/train.log sage-python/data/
```

### Étape 10 — Commit et arrêter le pod

```bash
# Sur le pod :
cd /workspace/YGN-SAGE
git add sage-python/scripts/verl/ sage-python/src/sage/verl/ sage-python/tests/
git commit -m "feat: veRL GRPO training results on H100"
git push origin VeRLGIGPO

# Arrêter le pod sur RunPod.io pour ne plus payer
```

---

## Troubleshooting

### vLLM crashe avec Qwen3.5-9B

```
CUDA error: an illegal memory access was encountered
```

**Cause** : MTP speculative decoding (vLLM bug #36408).

**Fix** : vérifier que `num_speculative_tokens=0` est dans train_topology.sh (déjà configuré).

**Si ça crashe encore** : changer le modèle :
```bash
SAGE_MODEL="Qwen/Qwen2.5-7B-Instruct" bash scripts/verl/train_topology.sh
```

### OOM (Out of Memory)

**Réduire progressivement :**
1. `gpu_memory_utilization=0.6` (au lieu de 0.7)
2. `train_batch_size=32` (au lieu de 64)
3. `rollout.n=3` (au lieu de 5)
4. `actor.fsdp_config.optimizer_offload=True`

### sage-core ne compile pas

```bash
# Installer Rust si absent
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Rebuild
cd /workspace/YGN-SAGE/sage-core
pip install maturin
maturin develop --features smt,onnx,cognitive,tool-executor --release
```

### Reward toujours 0.0

- Vérifier que `sage_core` est importable : `python3 -c "from sage_core import TopologyReward; print('OK')"`
- Vérifier que les données parquet ont le bon format : `python3 -c "import pandas as pd; print(pd.read_parquet('data/verl_topology_train.parquet').columns.tolist())"`
- Tester manuellement : `python3 -c "from sage.verl.reward import compute_score; print(compute_score('t','nodes:\n- role: coder','',{}))"`

### Training diverge (reward baisse)

- Réduire `kl_loss_coef` de 0.04 à 0.02
- Augmenter `temperature` de 0.4 à 0.6
- Vérifier que le modèle n'est pas corrompu : `python3 -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('Qwen/Qwen3.5-9B', trust_remote_code=True).architectures)"`

---

## Ce qui fait de SAGE un système unique

| Innovation | Source | Status |
|-----------|--------|--------|
| Edge-level credit (Graph-GRPO) | arXiv 2603.02701 | Implémenté (`sage.verl.edge_credit`) |
| Formal verification (OxiZ SMT) | Rust sage-core | Actif (sub-0.1ms) |
| 5-path topology generation | MAP-Elites + MCTS + CMA-ME + LLM + templates | Actif |
| Per-difficulty density bounds | AgentConductor Eq.13 | Implémenté (N_max: 4/7/10) |
| Online evolution | MAP-Elites + bandit | Infrastructure prête |
| kNN routing (92% GT) | arXiv 2505.12601 | Actif |

**Aucun système ne combine les 3** (topology RL + edge credit + multi-turn correction) au 20 mars 2026.
SAGE est positionné pour être le premier.

---

## Fichiers clés

```
sage-python/
├── src/sage/verl/
│   ├── __init__.py              # Package veRL
│   ├── reward.py                # compute_score pour veRL (format+structure+density+edge_credit)
│   └── edge_credit.py           # Graph-GRPO per-edge advantage (arXiv 2603.02701)
├── scripts/verl/
│   ├── setup_runpod.sh          # Setup automatique du pod
│   ├── train_topology.sh        # Config training GRPO (Qwen3.5-9B, LoRA r=64)
│   ├── convert_sft_to_verl.py   # JSONL → Parquet veRL
│   ├── reward_topology.py       # Reward legacy (scripts/, backward compat)
│   ├── validate_setup.py        # Validation 8 points
│   ├── benchmark_post_train.py  # Eval BigCodeBench post-training
│   ├── export_for_local.py      # Export LoRA pour 12GB local
│   └── README.md                # Documentation
├── tests/
│   ├── test_verl_reward.py      # 20 tests reward
│   └── test_edge_credit.py      # 11 tests edge credit
└── data/                                       # TOUT EST DANS LE REPO — pas de scp
    ├── topology_sft_v2_combined.jsonl          # 1532 base entries
    ├── topology_raft_phase2.jsonl              # 199 execution-verified
    ├── topology_sft_gpt54_complex.jsonl        # 144 complex topologies
    ├── topology_gpt54_codeforces_gcj.jsonl     # 20 Codeforces/GCJ (GPT-5.4 Pro)
    ├── gpt54_deep_reasoning.jsonl              # 20 deep reasoning (GPT-5.4 Pro)
    ├── gpt54_simple_calibrated.jsonl           # 20 simple calibrated (GPT-5.4 Pro)
    ├── gpt54_error_correction.jsonl            # 20 error→correction pairs
    ├── gpt54_audit.jsonl                       # 10 audit improvements
    ├── topology_sft_gpt54_pro.jsonl            # 40 combined GPT-5.4 Pro
    ├── gpt54_preferences2.jsonl                # 20 preference pairs (futur DPO)
    └── verl_topology_train.parquet             # 1965 entries (auto-généré)
```

---

## Coût estimé

| Config | Durée | Coût RunPod |
|--------|-------|-------------|
| H100 80GB SXM | ~3h training | ~$12 (secure pod ~$4/h) |
| A100 80GB SXM | ~5h training | ~$10 (secure pod ~$2/h) |

Référence : Engineering Handbook HuggingFace — Qwen2.5-3B sur 4xA100 = ~$40. Notre 9B sur 1xH100 devrait être comparable.
