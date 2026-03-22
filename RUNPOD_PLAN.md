# YGN-SAGE V2 — Plan d'entraînement topology (RunPod H100)

> **Ce document est la référence pour l'opérateur (humain ou Claude Code) sur le pod.**
> Il décrit 3 phases progressives. Chaque phase a ses propres critères de succès.

## Vision

Entraîner un modèle (Qwen3.5-9B) qui génère des topologies multi-agents **adaptatives** — capables de se corriger en cours d'exécution. C'est le Path 6 de la DynamicTopologyEngine de YGN-SAGE.

**Concurrents à battre :**
- **The Conductor** (arXiv 2512.04388, ICLR 2026, Sakana AI) — Qwen2.5-7B GRPO, 6 providers, BigCodeBench 40.0%. Pas open-source.
- **AgentConductor** (arXiv 2602.17100) — Qwen2.5-3B GRPO, density S_complex, CodeContests 38.8%. Pas open-source.
- **CARD** (arXiv 2603.01089, ICLR 2026) — GCN conditionnel, price penalty. Code MIT (github.com/Warma10032/CARD).

**Différenciation SAGE :** Seul système open-source combinant RL topology + micro-décisions aux checkpoints + Rust formal verification + 8 providers + episodic memory + edge-level credit.

---

## Les 7 compétences du modèle de topologie

Le modèle doit apprendre à :

| # | Compétence | Comment elle est enseignée | Phase |
|---|------------|---------------------------|-------|
| 1 | **Structurer un DAG valide** — YAML avec nodes, edges, acyclique, connexe | `_score_format` + `_score_structure` + Rust `PyHybridVerifier` | A |
| 2 | **Assigner le bon model_tier** — reasoner/fast/budget selon rôle et difficulté | `_score_cost_efficiency` (CARD 2603.01089) + `ModelAssigner` Rust | A |
| 3 | **Placer les checkpoints** — nœuds fragiles où l'env évalue la qualité | 260 entrées adaptatives avec `adaptation.checkpoints` | A |
| 4 | **Choisir le fallback_tier** — tier d'upgrade si quality < threshold | 80 entrées recovery (before/after), `_score_resilience` | A |
| 5 | **Micro-décisions temps réel** — upgrade/continue/reroute aux checkpoints | GiGPO step-level anchors, machine 4 états `SageTopologyEnv` | **C** |
| 6 | **Adapter complexité↔difficulté** — 1-2 nœuds simple, 4-7 nœuds complex | `TopologyDensity.compute()` Rust, S_complex (AgentConductor) | A |
| 7 | **Reasoning explicite** — justification liée à la tâche | `_score_structure` +0.2, 2223/2225 entrées avec reasoning | A |

---

## Les 3 phases

### Phase A — Structural (single-turn, $0 API)

**Ce qu'on entraîne :** Le modèle génère du YAML en un shot. Pas d'interaction multi-step.

**Ce que le modèle apprend :** Compétences 1, 2, 3, 4, 6, 7 — structurer un DAG, assigner les tiers, placer checkpoints et fallbacks, adapter la complexité, écrire un reasoning.

**Ce que le modèle n'apprend PAS :** Compétence 5 (micro-décisions). Il ne voit jamais si ses topologies fonctionnent réellement. Le reward est purement structural.

**Framework :** verl 0.7.1 (ou verl-agent) + GiGPO. Single-turn : le modèle génère, le reward évalue le YAML statiquement.

**GiGPO en Phase A :** Fonctionne via step advantages token-level — GiGPO assigne du crédit aux tokens du YAML qui contribuent au reward. Ce n'est PAS le multi-step anchor grouping (ça c'est Phase C). Mais c'est déjà mieux que GRPO flat.

**Reward :**
```
R = _score_format(yaml)           — YAML valide ? [-2.0, +1.0]
  + _score_structure(yaml)         — nodes, edges, roles, reasoning [0.0, 1.0]
  + _score_rust_density(yaml)      — Rust TopologyDensity S_complex [0.0, 1.0]
  + bonus adaptation               — +0.1 si adaptation block, +0.1 si fallback_tier
```
Refs : `_score_format`, `_score_structure` dans reward.py. `TopologyDensity` dans sage-core/src/topology/density.rs. S_complex inspiré d'AgentConductor (2602.17100).

**Dataset :** 2225 entries (11 sources, dont 260 adaptatives V2).

**Config :**
```
Modèle: Qwen/Qwen3.5-9B (patched tokenizer, no <think>)
Algorithme: GiGPO (adv_estimator=gigpo, params dynamiques depuis ppo_trainer.yaml)
LoRA: r=64, alpha=32, all-linear
Epochs: 5
Batch: 64 (train), K=4 rollouts
Coût API: $0
Durée estimée: 1-2h H100
```

**Critères de succès Phase A :**
- [ ] `reward/mean` > 0.7 en fin de training
- [ ] Le modèle génère du YAML parsable > 90% du temps
- [ ] Les topologies incluent des `adaptation` blocks > 50% des cas pour les tâches moderate/complex
- [ ] Les `fallback_tier` sont présents sur les nœuds checkpoint
- [ ] Le `reasoning` est spécifique à la tâche (pas du boilerplate)

**Résultat attendu :** Un modèle qui génère du YAML structural de qualité, comparable à AgentConductor en format, avec les champs adaptatifs V2 en plus. **Pas encore meilleur que The Conductor** (pas d'execution reward).

---

### Phase B — Execution (single-turn, ~$50-80 API)

**Ce qu'on ajoute :** Le reward inclut maintenant l'exécution réelle. Chaque topologie est exécutée via `TopologyRunner` + `ProviderPool` avec les 8 providers LLM. Le code produit est testé en sandbox.

**Ce que le modèle apprend en plus :** Que `model_tier: reasoner` sur le planner + `fast` sur le coder produit de MEILLEURS résultats que tout en `budget`. Que certaines combinaisons de rôles fonctionnent et d'autres non. Le lien causal entre la topologie et le résultat.

**Framework :** Même que Phase A, mais `SAGE_VERL_EXEC=1`.

**Reward :**
```
R = 0.30 × R_structural           — format + density + verifier (identique Phase A)
  + 0.70 × R_execution            — PASSED=1.0, WRONG_ANSWER=0.5, RUNTIME_ERROR=0.3, TIMEOUT=0.2
```
Le modèle voit la CONSÉQUENCE de ses choix de topologie.

Refs : `evaluate_topology()` dans execution/__init__.py. `TopologyRunner` dans topology/runner.py. 8 providers dans config/cards.toml.

**Dataset :** ~600 curated (meilleur 27% du dataset, adaptatives prioritaires).

**Config :**
```
Modèle: Checkpoint Phase A (LoRA)
Algorithme: GiGPO (même config)
Epochs: 3
Batch: 32 (train), K=4 rollouts
Providers: DeepSeek, Google, OpenAI, xAI, MiniMax, Kimi, OpenRouter, Codex
Coût API: ~$50-80
Durée estimée: 1-2h H100
```

**Critères de succès Phase B :**
- [ ] `reward/mean` > 0.5 (execution reward est plus dur que structural)
- [ ] PASSED rate > 30% sur les 600 prompts
- [ ] Le modèle choisit des `model_tier` différenciés (pas tout en `budget` ni tout en `reasoner`)
- [ ] Les topologies complex ont plus de nœuds que les simples (S_complex calibré)
- [ ] BigCodeBench Hard (20 tasks) > 38% (amélioration sur 37.8% baseline)

**Résultat attendu :** Un modèle dont les topologies FONCTIONNENT. Comparable ou légèrement supérieur à AgentConductor. Pas encore au niveau de The Conductor (40%) car pas de micro-décisions.

---

### Phase C — Micro-décisions multi-step (futur, nécessite env integration)

**Ce qu'on ajoute :** L'environnement `SageTopologyEnv` (machine 4 états) est branché dans le training loop. Le modèle interagit avec l'env : il génère le YAML, voit les résultats nœud par nœud, et prend des décisions upgrade/continue/reroute aux checkpoints.

**Ce que le modèle apprend en plus :** Compétence 5 — QUAND upgrader vs continuer. Le coût-bénéfice de l'adaptation. C'est LE différenciateur vs tous les concurrents.

**Framework :** verl-agent (pas verl vanilla) avec multi-step env, OU custom training loop avec `SageTopologyEnv`.

**GiGPO en Phase C :** Les VRAIS anchor states fonctionnent. `decision:coder:moderate:low` groupe toutes les trajectoires où le coder a produit du mauvais code. GiGPO compare les décisions upgrade vs continue pour ce même anchor. Step-level advantage réel.

**Reward :**
```
R = 0.20 × R_structural
  + 0.35 × R_execution
  + 0.20 × R_rewardflow            — PageRank per-node credit (arXiv 2603.18859)
  + 0.15 × R_resilience            — bonus adaptation triggered + succeeded
  + 0.10 × R_cost_efficiency       — CARD price penalty (arXiv 2603.01089)
```
5 signaux. Refs : `rewardflow.py` (RewardFlow), `_score_resilience` + `_score_cost_efficiency` dans reward.py.

**Modules requis (tous implémentés, pas encore branchés dans le training loop) :**
- `topology_env.py` — Machine 4 états, 46 tests passent
- `rewardflow.py` — PageRank propagation
- `training_memory.py` — SQLite episodic memory cross-épisode
- `edge_credit.py` — Graph-GRPO per-edge advantage (arXiv 2603.02701)

**Token masking (critique) :** verl-agent masque automatiquement les observations (mask=0). Seuls les tokens générés par le modèle (YAML, "upgrade", "continue") reçoivent des gradients. À vérifier sur le premier batch.

**Critères de succès Phase C :**
- [ ] `step_advantage` non-nul dans les logs (preuve que GiGPO multi-step fonctionne)
- [ ] Des anchors `decision:*` apparaissent (preuve que le modèle prend des décisions)
- [ ] Le modèle choisit "upgrade" quand quality < threshold ET "continue" quand quality > threshold
- [ ] Les topologies avec adaptation activée ont un meilleur terminal reward que celles sans
- [ ] BigCodeBench Hard (20 tasks) > **40.0%** (battre The Conductor)
- [ ] Résilience : au moins 20% des épisodes ont une adaptation déclenchée et réussie

**Résultat attendu :** Le vrai SAGE V2. Supérieur à The Conductor grâce aux micro-décisions apprises + edge-level credit + episodic memory + 5-signal reward. Publiable.

---

## Progression réaliste

```
Phase A (maintenant) → Fondation structurelle
  ✓ YAML valide, model_tiers, checkpoints, fallbacks, reasoning
  ✗ Pas d'exécution, pas de micro-décisions
  Comparable à : AgentConductor (structural)

Phase B (après A) → Exécution réelle
  ✓ Le modèle voit si ses topologies marchent
  ✗ Pas de micro-décisions temps réel
  Comparable à : AgentConductor + multi-provider

Phase C (après B) → SOTA complet
  ✓ Micro-décisions, mémoire, edge credit, 5 signaux
  Supérieur à : The Conductor, CARD, AgentConductor
```

---

## Recherche qui inspire chaque composant

| Composant SAGE | Papier source | arXiv | Contribution |
|---------------|---------------|-------|-------------|
| GiGPO training | GiGPO | [2505.10978](https://arxiv.org/abs/2505.10978) | Step-level anchor states |
| Density function S_complex | AgentConductor | [2602.17100](https://arxiv.org/abs/2602.17100) | Penalise over/under-budget topologies |
| Price penalty reward | CARD | [2603.01089](https://arxiv.org/abs/2603.01089) | `1.0 - tanh(cost/budget)` |
| Per-node PageRank credit | RewardFlow | [2603.18859](https://arxiv.org/abs/2603.18859) | Dense reward sans model acting each step |
| Edge-level advantage | Graph-GRPO | [2603.02701](https://arxiv.org/abs/2603.02701) | Per-edge success rate |
| kNN routing pré-topologie | kNN routing | [2505.12601](https://arxiv.org/abs/2505.12601) | 92% accuracy S1/S2/S3 |
| Topology > model selection | AdaptOrch | [2602.16873](https://arxiv.org/abs/2602.16873) | Var_tau/Var_M ≥ 20 |
| Runtime agent pruning | AgentDropout | [2503.18891](https://arxiv.org/abs/2503.18891) | -21.6% tokens |
| Self-programming agents | OpenSage | [2602.16891](https://arxiv.org/abs/2602.16891) | Runtime agent creation |
| Cognitive architecture | CoALA | Référencé | 4-tier memory (working/episodic/semantic/procedural) |
| Z3 quality labeling | FoVer | Référencé | Z3 auto-labels for training data |
| Recursive topologies | The Conductor | [2512.04388](https://arxiv.org/abs/2512.04388) | Conductor sélectionne lui-même comme worker |
| Data curation | TopoCurate | [2603.01714](https://arxiv.org/abs/2603.01714) | Reflective Recovery metric |
| Multi-agent PRM | MASPRM | [2510.24803](https://arxiv.org/abs/2510.24803) | Bradley-Terry per-agent credit |
| Contextual bandit routing | PILOT | [2508.21141](https://arxiv.org/abs/2508.21141) | LinUCB budget-aware |

---

## Contexte technique (pod)

| Composant | Valeur |
|-----------|--------|
| Template RunPod | `Runpod Pytorch 2.4.0` |
| Modèle | `Qwen/Qwen3.5-9B` (GDN + attention, Apache 2.0) |
| Fallback | `Qwen/Qwen2.5-7B-Instruct` (si GDN crash vLLM) |
| Framework | verl 0.7.1 (Phase A/B) → verl-agent (Phase C) |
| Algorithme | GiGPO (`adv_estimator=gigpo`, params dynamiques) |
| LoRA | r=64, alpha=32, target=all-linear |
| GPU | 1x H100 80GB SXM |
| Données | 2225 entries (Phase A) → ~600 curated (Phase B) |
| Tokenizer | Patché (`patch_tokenizer.py` — supprime `<think>` Qwen3.5) |
| Providers Phase B | DeepSeek, Google, OpenAI, xAI, MiniMax, Kimi, OpenRouter, Codex |

## Coût estimé

| Phase | GPU | API | Total |
|-------|-----|-----|-------|
| A (structural) | ~$6 (~2h) | $0 | ~$6 |
| B (execution) | ~$6 (~2h) | ~$50-80 | ~$56-86 |
| C (micro-décisions) | ~$10 (~3h) | ~$50-80 | ~$60-90 |
| **Total 3 phases** | ~$22 | ~$100-160 | **~$122-182** |

---

## Étapes sur le pod (Phase A)

### 1. Setup

```bash
git clone https://github.com/yannabadie/YGN-SAGE.git /workspace/YGN-SAGE
cd /workspace/YGN-SAGE && git checkout VeRLGIGPO
# Copier .env avec les clés API
bash sage-python/scripts/verl/setup_runpod.sh
```

### 2. Valider

```bash
cd sage-python && python3 scripts/verl/validate_setup.py
```

### 3. Training Phase A

```bash
screen -S train
bash scripts/verl/train_topology.sh 2>&1 | tee train.log
```

**Signaux de succès :** `reward/mean` augmente, YAML parsable > 90%, adaptation blocks présents.

### 4. Training Phase B (après Phase A)

```bash
export SAGE_VERL_EXEC=1
# Relancer avec le checkpoint Phase A et le dataset curated
```

### 5. Post-training

```bash
python3 scripts/verl/post_training_pipeline.py all
# → export LoRA → merge Qwen3.5-9B → push HuggingFace → Q8 GGUF
```

**Résultat :** `yannabadie/sage-topology-policy-v2` sur HuggingFace + Q8_0 GGUF (~9.5GB) pour local.

---

## Troubleshooting

### Qwen3.5-9B GDN crash vLLM
```bash
SAGE_MODEL="Qwen/Qwen2.5-7B-Instruct" bash scripts/verl/train_topology.sh
# Ou: --enforce-eager (2x lent mais fonctionne)
```

### causal_conv1d / flash_attn incompatible
```bash
pip install causal-conv1d --force-reinstall --no-build-isolation --no-cache-dir
pip install flash-attn --force-reinstall --no-build-isolation --no-cache-dir
```

### GiGPO params rejetés par Hydra
Le Step 0 construit les args dynamiquement depuis `ppo_trainer.yaml`. Les params non reconnus sont automatiquement retirés.

### OOM
Réduire : `gpu_memory_utilization=0.6` → `train_batch_size=32` → `rollout.n=3`

### Provider API fail Phase B
Fallback structural si execution échoue. Training continue.

---

## ExoCortex (500+ papiers recherche)

Accessible via OpenAI Assistants API :
```python
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
response = client.responses.create(
    model="gpt-4o-mini",
    input="What does RewardFlow do?",
    tools=[{"type": "file_search", "vector_store_ids": ["ygnsageresearch-wii7kwkqozrd"]}]
)
```

---

## Fichiers clés

```
sage-python/src/sage/verl/
├── topology_env.py       # 4-state machine (Phase C)
├── step_reward.py        # StepRewardVector for GiGPO
├── reward.py             # 5-signal reward
├── edge_credit.py        # Graph-GRPO (Phase C)
├── rewardflow.py         # PageRank credit (Phase C)
├── training_memory.py    # Episodic memory (Phase C)
└── env_register.py       # verl-agent registration (Phase C)

sage-python/scripts/verl/
├── train_topology.sh     # GiGPO config, 2 phases
├── patch_tokenizer.py    # Qwen3.5 <think> removal
├── setup_runpod.sh       # 9-step setup
├── validate_setup.py     # 10 pre-flight checks
├── post_training_pipeline.py  # Export → HF → GGUF
└── convert_sft_to_verl.py     # 11 sources → 2225 entries

sage-core/src/topology/
├── topology_graph.rs     # TopologyNode + TopologyGraph (6 adaptive fields)
├── reward.rs             # RewardScore (resilience + cost_efficiency)
├── density.rs            # S_complex density function
└── verifier.rs           # PyHybridVerifier (acyclicity, connectivity)
```

## Tests : 397 (46 Python + 351 Rust), 0 failures
