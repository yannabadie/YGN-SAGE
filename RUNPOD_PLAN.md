# YGN-SAGE V2 — Plan d'exécution RunPod H100

> **Ce fichier est le seul document à suivre sur le pod.**
> Tout le code est implémenté et testé (46 tests, 0 failures). Aucun code à écrire sur le pod.

## Objectif

Entraîner Qwen3.5-9B via **verl-agent GiGPO** pour générer des topologies multi-agents adaptatives.
Le modèle apprend **3 choses simultanément** :
1. **Comment structurer** une topologie YAML (nodes, edges, model_tiers)
2. **Où placer** les checkpoints (quels nœuds sont fragiles)
3. **Quand upgrader** vs continuer vs rerouter (coût-bénéfice de l'adaptation)

C'est LE différenciateur vs The Conductor (ICLR 2026), CARD, AgentConductor, AdaptOrch.
Aucun concurrent n'entraîne un modèle à prendre des micro-décisions d'adaptation en cours d'exécution.

## Pourquoi GiGPO, pas GRPO

```
GRPO (flat reward):
  Step 0: modèle génère YAML → reward = 0.7
  Pas de credit assignment — le modèle ne sait pas POURQUOI 0.7.

GiGPO avec micro-décisions (step-level):
  Step 0: YAML                      → reward_0 = 0.8    anchor = prompt_hash
  Step 1: nœud coder exécuté        → reward_1 = 0.3    anchor = coder:moderate:abc
  Step 2: CHECKPOINT quality=0.3    → modèle décide "upgrade"
          [mask=0 pour l'obs, mask=1 pour "upgrade"]
  Step 3: coder ré-exécuté (reasoner)→ reward_3 = 0.8   anchor = upgrade:coder
  Step 4: nœud reviewer             → reward_4 = 0.6    anchor = reviewer:moderate:def
  Step 5: synthesizer               → reward_5 = 0.9    anchor = synthesizer:moderate:ghi
  Terminal: PASSED                   → reward_T = 1.0

  GiGPO groupe: "4 trajectoires arrivent au checkpoint coder avec quality=0.3.
  Trajectoires A,B choisissent upgrade → reward 0.8. C,D choisissent continue → reward 0.2.
  Advantage positif assigné à upgrade au anchor decision:coder:moderate:low."
```

**verl-agent masque automatiquement** les tokens d'observation (mask=0). Seuls les tokens
générés par le modèle (YAML, "continue"/"upgrade"/"reroute") reçoivent des gradients.

## Contexte technique

| Composant | Valeur |
|-----------|--------|
| Modèle | `Qwen/Qwen3.5-9B` (dense 9B, GDN + attention, Apache 2.0) |
| Fallback modèle | `Qwen/Qwen2.5-7B-Instruct` (si GDN crash vLLM) |
| Framework | **verl-agent** (github.com/langfengQ/verl-agent) |
| Docker | `verlai/verl:vllm011.latest` |
| Algorithme | **GiGPO** (`algorithm.adv_estimator=gigpo`) |
| GiGPO params | `algorithm.gigpo.{enable_similarity=True, similarity_thresh=0.85, step_advantage_w=1.0, mode=mean_norm}` |
| Environnement | `SageTopologyEnv` — 4-state machine (awaiting_yaml → executing → awaiting_decision → terminal) |
| LoRA | r=64, alpha=32, target=all-linear |
| GPU | 1x H100 80GB SXM |
| Données | **2225 entries** (Phase A) + **~600 curated** (Phase B) |
| Tokenizer | Patché (`patch_tokenizer.py` — supprime `<think>` de Qwen3.5) |
| Innovation | **Micro-décisions GiGPO** — le modèle décide upgrade/continue/reroute aux checkpoints |

## Données d'entraînement (2225 entries)

| Source | Entries | Contenu |
|--------|---------|---------|
| SFT v2 combined | 1532 | BigCodeBench, CodeContests, statiques |
| RAFT Phase 2 | 199 | Execution-verified |
| GPT-5.4 complex | 144 | 5-7 nœuds |
| GPT-5.4 adaptive (V2) | 120 | fallback_tier + checkpoints + gates |
| GPT-5.4 recovery (V2) | 80 | Scénarios échec → recovery (init + recovered) |
| GPT-5.4 static→adaptive | 60 | Migration statique → adaptatif |
| Autres GPT-5.4 | 90 | Codeforces, reasoning, simple, audit, correction |

**260 entrées adaptatives** (12% du dataset) enseignent les micro-décisions.

## Stratégie de training

| Phase | Epochs | Dataset | Reward | Micro-décisions | Coût API |
|-------|--------|---------|--------|-----------------|----------|
| **A: Structural** | 5 | 2225 | Format + density + adaptation bonus | Oui (checkpoints en structural) | $0 |
| **B: Execution** | 3 | ~600 curated | Structural + real multi-provider | Oui (checkpoints + vrai quality) | ~$50-80 |

## Coût estimé

| Poste | Coût |
|-------|------|
| RunPod H100 Secure (~4h, $3.09/h) | ~$13 |
| API Phase B (600 × 3 epochs × ~3 nœuds × $0.001) | ~$50-80 |
| **TOTAL** | **~$63-93** |

---

## Étapes sur le pod

### Étape 1 — Créer le pod

1. [console.runpod.io](https://console.runpod.io) → **Pods** → **+ Deploy**
2. GPU : **H100 80GB SXM**, **Secure Cloud**
3. Template :
   - Container Image : `verlai/verl:vllm011.latest`
   - Container Disk : **50 GB**, Volume Disk : **100 GB**, Volume Mount : `/workspace`
   - Env var : `HF_TOKEN` = `<ton token>`
4. Attendre ~5-10 min

### Étape 2 — Clone + .env

```bash
git clone https://github.com/yannabadie/YGN-SAGE.git /workspace/YGN-SAGE
cd /workspace/YGN-SAGE && git checkout VeRLGIGPO

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

### Étape 3 — Setup (~8 min)

```bash
bash sage-python/scripts/verl/setup_runpod.sh
```

Ce script (9 étapes) :
1. Vérifie GPU (H100/A100, >= 40GB VRAM)
2. Vérifie vLLM
3. **Installe flash-linear-attention + causal-conv1d** (Qwen3.5 GDN fast path)
4. **Installe verl-agent** (pas veRL vanilla)
5. Installe SAGE Python SDK + build sage-core Rust (~3 min)
6. **Patche le tokenizer Qwen3.5** (supprime `<think>` mode)
7. Convertit données → parquet (2225 entries)
8. Curate Phase B subset (~600 entries)
9. Vérification finale (10 checks)

### Étape 4 — Valider

```bash
cd sage-python && python3 scripts/verl/validate_setup.py
```

**10/10 checks doivent passer** (GPU, veRL, vLLM, flash-linear-attention, sage-core, SDK, reward, data, API keys, patched tokenizer).

### Étape 5 — Lancer le training

```bash
cd /workspace/YGN-SAGE/sage-python
screen -S train
bash scripts/verl/train_topology.sh 2>&1 | tee train.log
# Ctrl+A D pour détacher
```

**Le script fait** :
- **Step 0** : Validation GiGPO config (vérifie les params contre verl-agent)
- **Phase A** (~1-2h) : GiGPO structural, 2225 prompts, micro-décisions en mode structural
- **Phase B** (~1-2h) : GiGPO execution, ~600 curated, 8 providers réels

**Signaux de succès :**
- `reward/mean` augmente au fil des epochs
- `step_advantage` **non-nul** (preuve que GiGPO micro-décisions fonctionnent)
- Des steps avec `anchor = decision:*` apparaissent dans les logs
- GPU utilization > 80%

**Si Qwen3.5-9B crash (GDN vLLM bug)** :
```bash
SAGE_MODEL="Qwen/Qwen2.5-7B-Instruct" bash scripts/verl/train_topology.sh
```

### Étape 6 — Benchmark

```bash
python3 scripts/verl/benchmark_post_train.py --bench all --limit 20
```

**Targets** :
- BigCodeBench Hard > **40.0%** (battre The Conductor)
- Topologies adaptatives > templates statiques

### Étape 7 — Export → HuggingFace → Q8 GGUF

```bash
python3 scripts/verl/post_training_pipeline.py all
```

4 sous-étapes (~30 min) :
1. `export` — LoRA depuis checkpoint veRL
2. `merge` — Merge LoRA + Qwen3.5-9B base → float16
3. `push` — Upload vers `yannabadie/sage-topology-policy-v2`
4. `quantize` — Q8_0 GGUF pour RTX 3500 Ada 12GB local

**Résultat HuggingFace :**
- `yannabadie/sage-topology-policy-v2` — merged float16 (~18GB)
- `yannabadie/sage-topology-policy-v2/gguf/sage-topology-v2-Q8_0.gguf` (~9.5GB)

### Étape 8 — Stop pod

```bash
python3 -c "from huggingface_hub import list_repo_files; print(list_repo_files('yannabadie/sage-topology-policy-v2'))"
```
**STOP** sur console.runpod.io dès que confirmé.

---

## Troubleshooting

### Qwen3.5-9B GDN crash vLLM
```bash
# Le GDN hybrid a des bugs vLLM actifs. Fallback :
SAGE_MODEL="Qwen/Qwen2.5-7B-Instruct" bash scripts/verl/train_topology.sh
# Ou : ajouter --enforce-eager dans les params vLLM (2x plus lent)
```

### GiGPO: "gigpo not found"
```bash
# verl-agent pas installé
cd /workspace && rm -rf verl-agent
git clone https://github.com/langfengQ/verl-agent.git
cd verl-agent && pip install -e . && cd /workspace/YGN-SAGE/sage-python
```

### OOM
```bash
# Réduire progressivement :
# 1. gpu_memory_utilization=0.6
# 2. train_batch_size=32
# 3. rollout.n=3 (au lieu de 4)
# 4. ppo_micro_batch_size_per_gpu=4
```

### Provider API fail en Phase B
Le reward fallback structural si l'exécution échoue. Training continue.

---

## Architecture micro-décisions (V2)

```
SageTopologyEnv — Machine à 4 états :

  ┌─────────────────────────────────────────────────────────────────┐
  │  AWAITING_YAML                                                  │
  │  reset() → obs = {prompt + memory_context}                     │
  │  model generates YAML                                           │
  │  step(yaml) → parse, start incremental execution               │
  └───────────────────────────┬─────────────────────────────────────┘
                              ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  EXECUTING (incrémental, nœud par nœud)                        │
  │  _execute_single_node(cursor) → trace                          │
  │  Si checkpoint → qualité estimée → AWAITING_DECISION           │
  │  Si pas checkpoint → cursor++ → continuer                      │
  │  Si fin → TERMINAL                                              │
  └──────────┬───────────────────────────────┬──────────────────────┘
             ▼                               ▼
  ┌──────────────────────┐     ┌────────────────────────────────────┐
  │  AWAITING_DECISION   │     │  TERMINAL                          │
  │  obs = [CHECKPOINT]  │     │  sandbox test → PASSED/FAILED      │
  │  model: continue /   │     │  resilience bonus if upgrade worked│
  │         upgrade /    │     │  StepRewardVector for GiGPO        │
  │         reroute      │     │  store to episodic memory          │
  │  [mask=1 sur action] │     └────────────────────────────────────┘
  └──────────┬───────────┘
             ▼
     Retour à EXECUTING
```

**Tokens masking (verl-agent automatique) :**
- `mask=0` : observations env ([CHECKPOINT] Node 0...) → pas de gradients
- `mask=1` : actions modèle (YAML, "upgrade", "continue") → gradients GiGPO

---

## Fichiers clés

```
sage-python/
├── src/sage/verl/
│   ├── topology_env.py      # 4-state machine, micro-décisions, episodic memory
│   ├── step_reward.py       # StepRewardVector (per-step rewards for GiGPO)
│   ├── reward.py            # 5-signal: structural + execution + rewardflow + resilience + cost
│   ├── edge_credit.py       # Graph-GRPO per-edge advantage
│   ├── rewardflow.py        # PageRank per-node credit (arXiv 2603.18859)
│   ├── training_memory.py   # SQLite episodic memory
│   └── env_register.py      # verl-agent env registration (monkey-patch)
├── scripts/verl/
│   ├── setup_runpod.sh      # 9 étapes : deps + flash-linear-attention + patch tokenizer
│   ├── train_topology.sh    # GiGPO config (algorithm.gigpo.*), 2 phases
│   ├── patch_tokenizer.py   # Supprime <think> mode de Qwen3.5
│   ├── validate_setup.py    # 10 checks pré-training
│   ├── post_training_pipeline.py  # Export → Merge → HF → Q8 GGUF
│   ├── curate_training_data.py
│   ├── convert_sft_to_verl.py     # 11 sources → 2225 entries
│   └── benchmark_post_train.py
├── tests/
│   ├── test_verl_micro_decisions.py  # 7 tests micro-décisions
│   ├── test_verl_v2.py              # 20 tests V2 (memory, rewardflow, env)
│   └── test_verl_reward.py          # 19 tests reward
└── data/
    ├── verl_topology_train.parquet   # 2225 entries (0 /no_think)
    ├── verl_topology_curated.parquet # ~600 (Phase B)
    └── 11 fichiers .jsonl sources

sage-core/config/cards.toml          # 18 modèles, 7 providers
```

## Tests : 46 passed, 0 failed

```
tests/test_verl_micro_decisions.py  — 7 tests (micro-décisions)
tests/test_verl_v2.py               — 20 tests (V2 adaptive)
tests/test_verl_reward.py           — 19 tests (reward functions)
sage-core (Rust)                     — 351 tests
Total                                — 397 tests, 0 failures
```
