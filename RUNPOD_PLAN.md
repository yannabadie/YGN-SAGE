# YGN-SAGE — Plan d'exécution RunPod H100

> **Ce fichier est le seul document à suivre sur le pod.**
> Tout le code est implémenté et testé. Aucun code à écrire sur le pod.

## Objectif

Entraîner une politique de génération de topologies multi-agents via **verl-agent GiGPO**
sur Qwen3.5-9B. GiGPO (Group-in-Group Policy Optimization) fournit du credit assignment
step-level : chaque nœud de topologie = un step dans l'épisode, avec des anchor states
pour comparer les actions entre trajectoires.

## Pourquoi GiGPO, pas GRPO standard

```
GRPO standard (flat reward):
  Step 0: YAML → reward = 0.7
  Le modèle sait que cette topologie vaut 0.7, mais pas POURQUOI.

GiGPO (step-level reward):
  Step 0: YAML         → reward_0 = 0.8 (bonne structure)
  Step 1: nœud coder   → reward_1 = 0.3 (code médiocre)    anchor = coder:moderate:abc123
  Step 2: nœud reviewer→ reward_2 = 0.1 (review inutile)   anchor = reviewer:moderate:def456
  Step 3: synthesizer  → reward_3 = 0.0 (CRASH)            anchor = synthesizer:moderate:789

  GiGPO compare: "quand un reviewer voit le même contexte (même anchor),
  quel type de review produit de meilleurs résultats en aval?"
  → Credit assignment temporel que GRPO ne peut pas faire.
```

## Contexte technique

| Composant | Valeur |
|-----------|--------|
| Modèle | `Qwen/Qwen3.5-9B` (dense 9B, GatedDeltaNet + attention, Apache 2.0) |
| Fallback | `Qwen/Qwen2.5-7B-Instruct` |
| Framework | **verl-agent** (fork de veRL avec GiGPO, github.com/langfengQ/verl-agent) |
| Docker | `verlai/verl:vllm017.latest` (CUDA 12.9.1, PyTorch 2.10, vLLM 0.17) |
| Algorithme | **GiGPO** (`adv_estimator=gigpo`, `step_advantage_w=1.0`) |
| Environnement | `SageTopologyEnv` — multi-step gym (1 YAML + N nœuds) |
| LoRA | r=64, alpha=32, target=all-linear |
| GPU | 1x H100 80GB SXM (Secure Cloud, driver >= 575) |
| Données | **1965 entries** full (Phase A) + **499 curated** (Phase B) |
| Bug vLLM | `num_speculative_tokens=0` obligatoire (CUDA bug #36408) |
| Innovation | GiGPO step-level + Graph-GRPO edge credit (arXiv 2603.02701) |

## Stratégie de training (Mixed GiGPO, 2 phases)

| Phase | Epochs | Dataset | Reward | Env | Coût API |
|-------|--------|---------|--------|-----|----------|
| **A: Structural GiGPO** | 5 | Full (1965) | Format + structure + density | Multi-step (anchors structurels) | $0 |
| **B: Execution GiGPO** | 5 | Curated (499) | Real multi-provider execution | Multi-step (8 providers réels) | ~$50 |

## Providers (8 actifs pendant Phase B)

| Provider | Modèles | Rôle topologie | Prix (in/out par 1M) |
|----------|---------|----------------|---------------------|
| DeepSeek | deepseek-chat, reasoner | Primary reasoner + budget | $0.14*/$0.42 |
| Google | gemini-3.1-pro, flash-lite, 3-flash | Reasoner + fast tier | $0.25-$2/$1.50-$12 |
| xAI | grok-4-1-fast | Budget (2M context) | $0.20/$0.50 |
| OpenAI | gpt-5.4, mini, nano | Flagship + budget | $0.20-$2.50/$1.25-$15 |
| MiniMax | minimax-m2.7 | Budget | $0.30/$1.20 |
| Kimi | kimi-k2.5 | Agent Swarm | $0.60/$2.50 |
| OpenRouter | qwen3.5-plus | Qwen3.5-Plus | $0.26/$1.56 |
| Codex | gpt-5.3-codex | Code specialist | CLI |

## Coût estimé

| Poste | Dataset curated | Dataset full |
|-------|----------------|-------------|
| Phase A (structural GiGPO, $0 API) | ~1h GPU | ~2h GPU |
| Phase B API (8 providers) | ~$50 | ~$200 |
| RunPod H100 Secure ($3.09/h) | ~$6 (~2h) | ~$18 (~6h) |
| **TOTAL** | **~$56** | **~$218** |

## Prérequis locaux (déjà fait)

- [x] `sage.verl.topology_env` — SageTopologyEnv (multi-step gym avec anchor states)
- [x] `sage.verl.step_reward` — StepRewardVector (per-node reward pour GiGPO)
- [x] `sage.verl.reward` — compute_score (structural + execution multi-provider)
- [x] `sage.verl.edge_credit` — Graph-GRPO edge-level credit
- [x] `scripts/verl/train_topology.sh` — GiGPO config (adv_estimator=gigpo, multi_turn=True)
- [x] `scripts/verl/setup_runpod.sh` — installe **verl-agent** (pas veRL vanilla)
- [x] `scripts/verl/curate_training_data.py` — 499 prompts curated
- [x] `sage-core/config/cards.toml` — 20 modèles, 8 providers, prix mars 2026
- [x] `sage-python/src/sage/providers/connector.py` — 7 configs API + Codex CLI + OpenRouter

---

## Étapes sur le pod

### Étape 1 — Créer le pod RunPod

1. [console.runpod.io](https://console.runpod.io) → **Pods** → **+ Deploy**
2. GPU : **H100 80GB SXM**, **Secure Cloud**
3. Template : cliquer **"Go to my templates"** → **New Template** :
   - Name : `SAGE-GiGPO`
   - Container Image : `verlai/verl:vllm017.latest`
   - Container Disk : **50 GB**
   - Volume Disk : **100 GB**
   - Volume Mount : `/workspace`
   - Env var : `HF_TOKEN` = `<ton token HuggingFace>`
   - Save → Deploy On-Demand
4. Attendre ~5-10 min (image Docker 13.4GB)

### Étape 2 — Connecter (web terminal ou SSH)

**Web terminal** : Pod → Connect → Start Web Terminal

**SSH** (si web terminal ne marche pas) :
```bash
ssh <pod-id>@ssh.runpod.io -i <ta-clé>
```

### Étape 3 — Clone + .env

```bash
git clone https://github.com/yannabadie/YGN-SAGE.git /workspace/YGN-SAGE
cd /workspace/YGN-SAGE && git checkout VeRLGIGPO
```

```bash
cat > .env << 'EOF'
DEEPSEEK_API_KEY="<ta clé>"
GOOGLE_API_KEY="<ta clé>"
OPENAI_API_KEY="<ta clé>"
GROK_API_KEY="<ta clé>"
MINIMAX_API_KEY="<ta clé>"
KIMI_API_KEY="<ta clé>"
OPEN_ROUTER_API_KEY="<ta clé>"
HF_TOKEN="<ton token>"
EOF
```

### Étape 4 — Setup (~5-8 min)

```bash
bash sage-python/scripts/verl/setup_runpod.sh
```

Ce script (8 étapes) :
1. Vérifie GPU (H100/A100, >= 40GB VRAM)
2. Vérifie vLLM >= 0.17.0
3. **Installe verl-agent** (fork avec GiGPO, pas veRL vanilla)
4. Installe SAGE Python SDK
5. Build sage-core Rust (~3 min)
6. Vérifie le modèle Qwen3.5-9B
7. Convertit données → parquet (1965 entries)
8. Vérification finale

### Étape 5 — Valider

```bash
cd sage-python && python3 scripts/verl/validate_setup.py
```

**8/8 checks doivent passer.**

### Étape 6 — Lancer le training GiGPO

```bash
cd /workspace/YGN-SAGE/sage-python

# Curater 499 prompts
python3 scripts/verl/curate_training_data.py

# Lancer (screen pour web terminal)
screen -S train
bash scripts/verl/train_topology.sh 2>&1 | tee train.log
# Ctrl+A D pour détacher, screen -r train pour rattacher
```

**Le script fait 2 phases automatiquement** :
- **Phase A** (~1h) : GiGPO structural sur 1965 prompts — $0 API
- **Phase B** (~1-2h) : GiGPO execution sur 499 prompts — ~$50 API (8 providers)

**Signaux :**
- `reward/mean` augmente
- `step_advantage` non-nul (preuve que GiGPO fonctionne)
- GPU > 90%

### Étape 7 — Export + Benchmark

```bash
python3 scripts/verl/export_for_local.py --checkpoint models/topology_verl/ --output models/topology_verl_local/
python3 scripts/verl/benchmark_post_train.py --bench all --limit 20 --model models/topology_verl_local/
```

### Étape 8 — Récupérer + arrêter

```bash
tar czf /workspace/results.tar.gz models/topology_verl_local/ train.log
# Télécharger via file browser RunPod ou scp
```

**STOP le pod** sur console.runpod.io dès que fini.

---

## Troubleshooting

### verl-agent ne s'installe pas
```bash
cd /workspace && rm -rf verl-agent
git clone https://github.com/langfengQ/verl-agent.git
cd verl-agent && pip install -e . && cd /workspace/YGN-SAGE
```

### GiGPO: "gigpo not found in adv_estimator"
verl-agent n'est pas installé (veRL vanilla ne supporte pas GiGPO).
Vérifie : `python3 -c "from gigpo.core_gigpo import compute_gigpo_outcome_advantage; print('OK')"`

### vLLM CUDA crash
`num_speculative_tokens=0` (déjà dans train_topology.sh).
Fallback : `SAGE_MODEL="Qwen/Qwen2.5-7B-Instruct" bash scripts/verl/train_topology.sh`

### OOM
Réduire : `gpu_memory_utilization=0.6` → `train_batch_size=32` → `rollout.n=3`

### Provider API échoue en Phase B
Smart fallback : reward structural si execution échoue. Training continue.

---

## Architecture GiGPO multi-step

```
SageTopologyEnv.reset(prompt) → obs = {text: prompt, anchor: hash(prompt)}
                                          │
                Model generates YAML ◄─────┘
                          │
SageTopologyEnv.step(yaml) → obs, reward_0 (structural), done=False
                          │   anchor_0 = topology_generator:difficulty:yaml_hash
                          │
                Model sees node 0 result, generates response
                          │
SageTopologyEnv.step(response) → obs, reward_1 (node quality), done=False
                          │       anchor_1 = role:difficulty:context_hash
                          │
                ... repeat for each node ...
                          │
SageTopologyEnv.step(response) → obs, reward_N (terminal), done=True
                                  anchor_N = terminal:status

GiGPO advantage:
  A'(i,k) = A_episode(i) + ω × A_step(i,k)

  A_episode = (R_total - mean) / std     ← same as GRPO
  A_step    = (R_from_k - mean_group) / std_group
              where group = all (trajectory,step) pairs sharing anchor_k
```

---

## Fichiers clés

```
sage-python/
├── src/sage/verl/
│   ├── topology_env.py      # SageTopologyEnv (multi-step gym, anchor states)
│   ├── step_reward.py       # StepRewardVector (per-node rewards for GiGPO)
│   ├── reward.py            # compute_score (structural + execution)
│   └── edge_credit.py       # Graph-GRPO per-edge advantage
├── src/sage/execution/
│   └── __init__.py          # evaluate_topology(), extract_python_code, providers
├── scripts/verl/
│   ├── setup_runpod.sh      # Installs verl-agent (GiGPO), not vanilla veRL
│   ├── train_topology.sh    # adv_estimator=gigpo, multi_turn=True, 2 phases
│   ├── curate_training_data.py
│   ├── convert_sft_to_verl.py
│   ├── validate_setup.py
│   ├── benchmark_post_train.py
│   └── export_for_local.py
└── data/                    # TOUT DANS LE REPO
    ├── verl_topology_curated.parquet  # 499 (Phase B)
    ├── verl_topology_train.parquet    # 1965 (Phase A)
    └── (10 fichiers .jsonl sources)

sage-core/config/cards.toml  # 20 modèles, 8 providers
```
