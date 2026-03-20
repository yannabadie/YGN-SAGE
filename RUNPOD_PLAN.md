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
| Framework | veRL 0.7.1 + vLLM 0.17.0 (pré-installé dans Docker) |
| Docker | `verlai/verl:vllm017.latest` (CUDA 12.9.1, PyTorch 2.10, FA 2.8.3) |
| Algorithme | GRPO standard (GiGPO = GRPO pour single-action, vérifié) |
| LoRA | r=64, alpha=32, target=all-linear |
| GPU | 1x H100 80GB SXM (Secure Cloud, driver >= 575) |
| VRAM estimé | ~68 GB (model 18 + KV 45 + optimizer 1 + activations 4) |
| Données | **1965 entries** dans le repo (499 curated recommandé) |
| Bug vLLM | `num_speculative_tokens=0` obligatoire (CUDA bug #36408) |
| Innovation | Edge-level credit (Graph-GRPO, arXiv 2603.02701) |

## Stratégie de training (Mixed Reward, 2 phases)

| Phase | Epochs | Reward | Providers | Coût API |
|-------|--------|--------|-----------|----------|
| **A: Structural** | 5 | Format YAML + structure + Rust density | Aucun | $0 |
| **B: Execution** | 5 | TopologyRunner + ProviderPool multi-provider | **Tous les 8** | ~$50 |

**Phase B utilise les 8 providers** pour que le modèle apprenne l'orchestration multi-provider :
chaque nœud de topologie est exécuté par le provider assigné via `ModelAssigner` → `ProviderPool.resolve()`.

## Providers utilisés pendant le training (Phase B)

| Provider | Modèles | Prix (in/out par 1M) | Rôle dans la topologie |
|----------|---------|---------------------|----------------------|
| DeepSeek | deepseek-chat, deepseek-reasoner | $0.14*/$0.42 | Primary reasoner + budget (*cache avg) |
| Google | gemini-3.1-pro, flash-lite, 3-flash | $0.25-$2/$1.50-$12 | Reasoner + fast tier |
| xAI | grok-4-1-fast | $0.20/$0.50 | Budget tier (2M context) |
| OpenAI | gpt-5.4, gpt-5.4-mini, gpt-5.4-nano | $0.20-$2.50/$1.25-$15 | Flagship + budget |
| MiniMax | minimax-m2.7 | $0.30/$1.20 | Budget tier |
| Kimi | kimi-k2.5 | $0.60/$2.50 | Agent Swarm |
| OpenRouter | qwen/qwen3.5-plus-02-15 | $0.26/$1.56 | Qwen3.5-Plus |
| Codex | gpt-5.3-codex | CLI subprocess | Code specialist |

## Coût estimé (vérifié mars 20, 2026)

| Poste | Dataset curated (499) | Dataset full (1965) |
|-------|----------------------|---------------------|
| Phase A (structural, $0 API) | ~20 min GPU | ~1h GPU |
| Phase B API (8 providers) | ~$50 | ~$200 |
| RunPod H100 Secure ($3.09/h) | ~$3.50 (1h) | ~$13 (4h) |
| **TOTAL** | **~$54** | **~$213** |

**Recommandation : dataset curated (499) pour le premier run à ~$54.**

## Prérequis locaux (déjà fait)

- [x] `sage.verl.reward` — reward mixte (structural + execution multi-provider)
- [x] `sage.verl.edge_credit` — Graph-GRPO edge-level credit
- [x] `scripts/verl/train_topology.sh` — 2 phases automatiques (A structural + B execution)
- [x] `scripts/verl/curate_training_data.py` — 499 prompts curated
- [x] `scripts/verl/validate_setup.py` — validation 8 points
- [x] `scripts/verl/benchmark_post_train.py` — éval post-training avec modèle entraîné
- [x] `scripts/verl/convert_sft_to_verl.py` — auto-charge 10 sources → parquet
- [x] `scripts/verl/export_for_local.py` — export LoRA pour 12GB local
- [x] `sage-core/config/cards.toml` — 20 modèles, 8 providers, prix mars 2026
- [x] `sage-python/src/sage/providers/connector.py` — 7 configs API + Codex CLI
- [x] 51 tests passing, 0 régression

---

## Étapes sur le pod

### Étape 1 — Créer le pod RunPod

1. Aller sur [runpod.io](https://runpod.io)
2. **Deploy** → **GPU Pods** → **Deploy**
3. Choisir **H100 80GB SXM** en **Secure Cloud** (driver >= 575 garanti)
4. Docker image : `verlai/verl:vllm017.latest`
5. Disk : **150 GB**
6. Volume : `/workspace`
7. Lancer et ouvrir le **web terminal**

### Étape 2 — Cloner et configurer

```bash
git clone https://github.com/yannabadie/YGN-SAGE.git /workspace/YGN-SAGE
cd /workspace/YGN-SAGE
git checkout VeRLGIGPO
git log --oneline -3
```

### Étape 3 — Configurer TOUTES les clés API

**IMPORTANT : toutes les clés sont nécessaires pour le training multi-provider (Phase B).**

```bash
# REQUIS — providers principaux
export DEEPSEEK_API_KEY="..."         # DeepSeek Chat V3.2 (primary, pas de rate limits)
export GOOGLE_API_KEY="..."           # Gemini 3.1 Pro + Flash-Lite + 3-Flash
export OPENAI_API_KEY="..."           # GPT-5.4 + Mini + Nano
export HF_TOKEN="..."                # Télécharger Qwen3.5-9B

# REQUIS — providers secondaires (multi-provider training)
export GROK_API_KEY="..."             # Grok 4.1 Fast (xAI, $0.20/$0.50, 2M context)
export MINIMAX_API_KEY="..."          # MiniMax M2.7 ($0.30/$1.20)
export KIMI_API_KEY="..."             # Kimi K2.5 ($0.60/$2.50, Agent Swarm)
export OPEN_ROUTER_API_KEY="..."      # Qwen3.5-Plus via OpenRouter ($0.26/$1.56)

# OPTIONNEL
export WANDB_API_KEY="..."            # Dashboard W&B

# Vérifier :
echo "DeepSeek: ${DEEPSEEK_API_KEY:0:10}..."
echo "Google: ${GOOGLE_API_KEY:0:10}..."
echo "OpenAI: ${OPENAI_API_KEY:0:10}..."
echo "xAI: ${GROK_API_KEY:0:10}..."
echo "MiniMax: ${MINIMAX_API_KEY:0:10}..."
echo "Kimi: ${KIMI_API_KEY:0:10}..."
echo "OpenRouter: ${OPEN_ROUTER_API_KEY:0:10}..."
echo "HF: ${HF_TOKEN:0:10}..."
```

### Étape 4 — Setup automatique (~5 min)

```bash
cd /workspace/YGN-SAGE
bash sage-python/scripts/verl/setup_runpod.sh
```

Ce script (8 étapes) :
1. Vérifie le GPU (H100/A100, >= 40GB VRAM)
2. Vérifie vLLM >= 0.17.0
3. Installe veRL 0.7.1 depuis source (--no-deps)
4. Installe SAGE Python SDK
5. Build sage-core Rust (maturin, ~3 min)
6. Vérifie le modèle Qwen3.5-9B
7. Convertit les données → parquet veRL (1965 entries)
8. Verification finale

### Étape 5 — Valider l'environnement

```bash
cd /workspace/YGN-SAGE/sage-python
python3 scripts/verl/validate_setup.py
```

**Les 8 checks doivent passer** (GPU, veRL, vLLM, sage-core, SAGE SDK, reward, data, API keys).

### Étape 6 — Lancer le training

Le script fait **2 phases automatiquement** :
- **Phase A** (5 epochs, ~20 min) : reward structural, $0 API
- **Phase B** (5 epochs, ~50 min) : reward execution multi-provider, ~$50 API

```bash
cd /workspace/YGN-SAGE/sage-python

# Curater 499 prompts diversifiés
python3 scripts/verl/curate_training_data.py

# Lancer (web terminal : screen pour ne pas perdre le process)
screen -S train
bash scripts/verl/train_topology.sh 2>&1 | tee train.log
# Ctrl+A D pour détacher, screen -r train pour rattacher
```

**Signaux à surveiller :**
- `reward/mean` augmente progressivement
- `actor/loss` diminue
- `kl_divergence` < 0.1
- GPU utilization > 90%

```bash
nvidia-smi
grep -E "(reward|loss|epoch)" train.log | tail -10
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
python3 scripts/verl/benchmark_post_train.py \
    --bench all --limit 20 \
    --model models/topology_verl_local/
```

### Étape 9 — Récupérer les résultats (web terminal)

```bash
# Depuis le web terminal, zip les résultats :
cd /workspace/YGN-SAGE/sage-python
tar czf /workspace/results.tar.gz models/topology_verl_local/ train.log
# Télécharger via l'interface web RunPod ou scp
```

### Étape 10 — Commit et arrêter le pod

```bash
cd /workspace/YGN-SAGE
git add sage-python/src/sage/verl/ sage-python/scripts/verl/ sage-python/tests/
git commit -m "feat: veRL GRPO training results — Qwen3.5-9B on H100"
git push origin VeRLGIGPO

# IMPORTANT : arrêter le pod sur RunPod.io pour ne plus payer
```

---

## Troubleshooting

### vLLM crashe avec Qwen3.5-9B

```
CUDA error: an illegal memory access was encountered
```

**Fix** : `num_speculative_tokens=0` (déjà dans train_topology.sh).
**Si ça crashe encore** : `SAGE_MODEL="Qwen/Qwen2.5-7B-Instruct" bash scripts/verl/train_topology.sh`

### OOM (Out of Memory)

Réduire : `gpu_memory_utilization=0.6` → `train_batch_size=32` → `rollout.n=3` → `optimizer_offload=True`

### Provider API échoue pendant Phase B

Le reward function a un **smart fallback** : si l'exécution échoue (provider down, rate limit), elle retourne le score structural. Le training continue sans interruption.

### Reward toujours 0.0

```bash
python3 -c "from sage_core import TopologyReward; print('OK')"
python3 -c "from sage.verl.reward import compute_score; print(compute_score('t','nodes:\n- role: coder','',{}))"
```

---

## Architecture du reward multi-provider

```
SAGE_VERL_EXEC=0 (Phase A):
  compute_score() = (format + structure + rust_density) / 3
  → Pas d'appels API, le modèle apprend le format YAML

SAGE_VERL_EXEC=1 (Phase B):
  compute_score() = 0.3 × structural + 0.7 × execution
  → execution = evaluate_topology():
    1. Parse YAML → TopologyGraph (Rust)
    2. ModelAssigner assigne model_id (cards.toml, 20 modèles)
    3. TopologyRunner exécute chaque nœud via ProviderPool
       → Nœud "reasoner" → gemini-3.1-pro ou deepseek-reasoner
       → Nœud "fast" → gemini-flash-lite ou grok-4-1-fast
       → Nœud "budget" → deepseek-chat ou gpt-5.4-nano
    4. Code extrait du dernier nœud (synthesizer)
    5. Code testé en sandbox (BigCodeBench / HumanEval tests)
    6. Reward gradué : PASSED=1.5, WRONG_ANSWER=1.0, RUNTIME_ERROR=0.7
    7. + Rust density scoring (AgentConductor Eq.9)
    8. + Edge-level credit (Graph-GRPO, arXiv 2603.02701)
```

---

## Fichiers clés

```
sage-python/
├── src/sage/verl/
│   ├── reward.py                # compute_score (structural + execution multi-provider)
│   └── edge_credit.py           # Graph-GRPO per-edge advantage
├── src/sage/grpo/
│   └── execution_reward.py      # evaluate_topology() + TopologyRunner integration
├── scripts/verl/
│   ├── setup_runpod.sh          # 8-step setup
│   ├── train_topology.sh        # Phase A + Phase B automatiques
│   ├── curate_training_data.py  # 499 prompts curated
│   ├── convert_sft_to_verl.py   # 10 sources → parquet (1965 entries)
│   ├── validate_setup.py        # 8-point validation
│   ├── benchmark_post_train.py  # BigCodeBench eval avec modèle entraîné
│   └── export_for_local.py      # Export LoRA
├── tests/
│   ├── test_verl_reward.py      # 20 tests
│   └── test_edge_credit.py      # 11 tests
└── data/                        # TOUT DANS LE REPO — pas de scp
    ├── verl_topology_curated.parquet  # 499 entries (recommandé)
    ├── verl_topology_train.parquet    # 1965 entries (full)
    └── (10 fichiers .jsonl sources)

sage-core/config/cards.toml      # 20 modèles, 8 providers, prix mars 2026
```
