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
| Framework | veRL 0.7.1 + vLLM 0.17.0 |
| Docker | `verlai/verl:vllm017.latest` |
| Algorithme | GRPO standard (GiGPO = GRPO pour single-action, vérifié dans core_gigpo.py) |
| LoRA | r=64, alpha=32, target=all-linear |
| GPU | 1x H100 80GB (ou A100 80GB) |
| VRAM estimé | ~68 GB (model 18 + KV cache 45 + optimizer 1 + activations 4) |
| Données | 1880 entries (BigCodeBench 60%, GSM8K 20%, RAFT 11%, CodeContests 9%) |
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
   - **Disk** : 100GB+ (modèle 9B + checkpoints)
   - **Volume** : monter sur `/workspace`
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
# Doit montrer: f1cfbff feat: implement veRL reward + edge credit...
```

### Étape 3 — Uploader les données et clés API

```bash
# Depuis la MACHINE LOCALE (pas le pod) :
scp sage-python/data/topology_sft_v2_combined.jsonl <pod>:/workspace/YGN-SAGE/sage-python/data/

# Sur le pod — configurer les clés API :
export GOOGLE_API_KEY="<ta clé Google>"
export DEEPSEEK_API_KEY="<ta clé DeepSeek>"
export WANDB_API_KEY="<ta clé W&B>"  # optionnel, pour le dashboard

# Vérifier :
echo "Google: ${GOOGLE_API_KEY:0:10}..."
echo "DeepSeek: ${DEEPSEEK_API_KEY:0:10}..."
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

### Étape 6 — Lancer le training (~2-4h)

```bash
cd /workspace/YGN-SAGE/sage-python

# Lancer en background avec log
nohup bash scripts/verl/train_topology.sh > train.log 2>&1 &
echo $! > train.pid
echo "Training PID: $(cat train.pid)"

# Monitorer
tail -f train.log
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
└── data/
    ├── topology_sft_v2_combined.jsonl  # 1880 entries (UPLOADER sur le pod)
    ├── PROMPTS_GPT54_PRO.md            # 5 prompts pour distillation GPT-5.4 Pro
    └── verl_topology_train.parquet     # Généré par convert_sft_to_verl.py
```

---

## Coût estimé

| Config | Durée | Coût RunPod |
|--------|-------|-------------|
| H100 80GB SXM | ~3h training | ~$12 (secure pod ~$4/h) |
| A100 80GB SXM | ~5h training | ~$10 (secure pod ~$2/h) |

Référence : Engineering Handbook HuggingFace — Qwen2.5-3B sur 4xA100 = ~$40. Notre 9B sur 1xH100 devrait être comparable.
