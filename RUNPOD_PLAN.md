# YGN-SAGE V2 — Plan d'entraînement topology (RunPod H100)

> **Ce document est la référence pour l'opérateur (humain ou Claude Code) sur le pod.**
> Il décrit 3 phases progressives. Chaque phase a ses propres critères de succès.

---

## Qu'est-ce que YGN-SAGE ?

**YGN-SAGE** (Self-Adaptive Generation Engine) est un **Agent Development Kit** — un agent autonome de type Claude Code, Devin ou OpenSage, mais piloté par un moteur multi-agents **apprenant**. C'est un OpenSage (arXiv 2602.16891, UC Berkeley) profondément amélioré : là où OpenSage crée ses agents par prompting à chaque run (et repart de zéro), SAGE **apprend** de ses exécutions passées via RL et adapte ses topologies en temps réel.

### Philosophie : 3 principes fondateurs

**1. Rust first, Python tolerant**
Tout ce qui est performance-critique est en Rust (sage-core, compilé via PyO3/maturin) : TopologyGraph (petgraph), Density S_complex, HybridVerifier (SMT/LTL), QualityLabeler (tree-sitter + OxiZ), S-MMU (4-graph mémoire), ModelAssigner, SystemRouter, ContextualBandit. Python sert uniquement à l'orchestration (pipeline, agent loop, providers, training). Le coeur Rust garantit des latences sub-milliseconde pour le scoring et la vérification — critique pendant le training RL où chaque step est évalué.

**2. Zero heuristics**
Aucun seuil hardcodé. Chaque décision est soit formellement vérifiée (Z3/OxiZ SMT), soit apprise (ONNX, RL, bandit), soit backed par un papier de recherche. Le QualityLabeler Rust utilise tree-sitter + Z3 — pas de "if len(output) > 10". Le routing utilise kNN sur arctic-embed-m (92% accuracy) — pas de regex sur des mots-clés. Les reward weights (0.20/0.35/0.20/0.15/0.10) sont des valeurs initiales sujettes à ablation, pas des constantes magiques.

**3. Evidence before assertions**
On ne claim pas que ça marche — on prouve. 404 tests (357 Rust + 47 Python). BigCodeBench Hard comme benchmark principal (pas HumanEval qui est saturé). Chaque décision architecturale a une référence papier (voir table en bas).

### Architecture : 5 piliers cognitifs

```
                         ┌─────────────────────────────────┐
                         │          YGN-SAGE Agent          │
                         │   (type Claude Code / OpenSage)  │
                         └───────────────┬─────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
   sage-core (Rust)              sage-python (Python)            sage-discover
   Performance-critical          Orchestration                   Knowledge Pipeline
   ├── TopologyEngine            ├── Pipeline (5-stage)          ├── arXiv → ExoCortex
   │   ├── 6 paths generation    ├── AgentLoop                  └── 500+ papers RAG
   │   ├── MAP-Elites + CMA-ME   ├── 8 Providers (LLM)
   │   ├── MCTS search           ├── Memory 4-tier
   │   └── Path 6: learned ←──── ├── verl/ (RL training)
   ├── SystemRouter (S1/S2/S3)   │   ├── topology_env.py (4-state machine)
   ├── ModelAssigner              │   ├── reward.py (5-signal)
   ├── QualityLabeler (Z3)        │   ├── rewardflow.py (PageRank)
   ├── S-MMU (4-graph memory)     │   └── training_memory.py (SQLite)
   ├── SmtVerifier (OxiZ)         ├── A2A server (v1.0)
   └── HybridVerifier (LTL)      └── Bench (BigCodeBench, EvalPlus)
```

**Pilier 1 — Topology** : 7 chemins de génération de topologie (S-MMU retrieval → archive MAP-Elites → LLM synthesis → mutation → MCTS → **Path 6: learned policy** → templates). Path 6 est le modèle entraîné par RL. La DynamicTopologyEngine choisit le meilleur path via contextual bandit.

**Pilier 2 — Tools** : `AgentTool.from_agent()` transforme n'importe quel agent en outil. Sandbox 3 couches (tree-sitter → Wasm WASI → subprocess). Dynamic sub-agent creation (`agent_mgmt.py`) comme OpenSage.

**Pilier 3 — Memory** : 4 tiers inspirés de CoALA (cognitive architecture) — Rust Arrow STM → SQLite Episodic → Entity Semantic → ExoCortex RAG. S-MMU (Semantic Memory Management Unit) en Rust avec 4 types d'edges (temporal, semantic, causal, entity) et ULID chunks.

**Pilier 4 — Evolution** : MAP-Elites quality-diversity + CMA-ME + MCTS topology search. Online evolution câblée dans le pipeline Stage 5 (Learn). Les topologies qui marchent sont archivées et réutilisées.

**Pilier 5 — Strategy** : Routing cognitif S1/S2/S3 (Kahneman). kNN primary (92%), Rust SystemRouter (88%). ContextualBandit Thompson sampling pour le choix de topologie.

### Pipeline : comment une tâche est traitée

```
1. CLASSIFY  — kNN routing (arctic-embed-m, 92%) → S1 (simple) / S2 (moderate) / S3 (complex)
2. DECOMPOSE — TaskPlanner → TaskDAG + features DAG (ω, δ, γ d'AdaptOrch)
3. TOPOLOGY  — DynamicTopologyEngine choisit parmi 6 paths → TopologyGraph (Rust petgraph)
4. ASSIGN    — ModelAssigner (Rust) : affinity 0.4 + domain 0.4 + cost 0.2 → model_id par nœud
5. EXECUTE   — TopologyRunner + ProviderPool (8 providers) → exécution nœud par nœud
6. LEARN     — QualityEstimator Z3 → Bandit update + MAP-Elites archive + episodic memory
```

### Ce qui différencie SAGE d'OpenSage

| Aspect | OpenSage (Berkeley) | YGN-SAGE |
|--------|-------------------|----------|
| Topologie | Promptée à chaque run | **Apprise par RL** (GiGPO) |
| Adaptation | Runtime prompting | **Apprise** (checkpoints, fallback_tier dans le YAML) |
| Mémoire | Graph-based hierarchical | **4-tier** (STM → Episodic → Semantic → ExoCortex) + cross-episode training |
| Verification | Aucune | **Formelle** (Rust SMT/LTL, QualityLabeler Z3) |
| Providers | Multi-model (GPT/Claude/Gemini) | **8 providers** câblés DANS le training |
| Engine | Python pur | **Rust core** (PyO3, sub-ms latency) |
| Self-programming | Agents créent des agents | **Idem** (`agent_mgmt.py`) + topologies apprises |
| Code | 404 (pas publié) | **Open-source MIT** |
| Protocol | Aucun standard | **A2A v1.0** (Google) |
| Benchmark | SWE-bench Pro 59% | BigCodeBench Hard 37.8% (→ cible >40%) |

### Interface cible : pi-mono

En production, SAGE sera pilotable via **pi-mono** (github.com/badlogic/pi-mono) — un toolkit TypeScript pour agents AI avec multi-provider API unifiée (`@mariozechner/pi-ai`), agent runtime (`pi-agent-core`), et web UI (`pi-web-ui`). L'intégration se fait via le serveur A2A de SAGE (`a2a_server.py`) qui expose les skills topology/code/research.

### Standard A2A

SAGE implémente le protocole **Agent-to-Agent v1.0** (Google) via `a2a_server.py`. L'agent est exposé comme un `AgentCard` avec 3 skills (general, code, research). N'importe quel client A2A (Google ADK, LangGraph, CrewAI) peut déléguer des tâches à SAGE. Le modèle de topologie entraîné produira des YAML qui respectent les conventions A2A pour l'interopérabilité.

---

## Vision training

Entraîner **DeepSeek-R1-0528-Qwen3-8B** (MIT license) via GiGPO pour générer des topologies multi-agents **adaptatives** — capables de se corriger en cours d'exécution. C'est le Path 6 de la DynamicTopologyEngine.

**Pourquoi DeepSeek-R1-0528-Qwen3-8B :**
- Architecture **Qwen3-8B transformer standard** (pas de GDN/Mamba2 = zéro flashinfer/causal_conv1d)
- Distillé sur les **traces de raisonnement R1-0528** → AIME 86.0% (+10pp vs Qwen3-8B), bat Qwen3-32B
- LiveCodeBench 60.5%, GPQA 61.1% — raisonnement de niveau frontier dans 8B
- **MIT license** — plus permissif qu'Apache (Qwen3) et Falcon-LLM
- Compatible verl GRPO (même arch que Qwen3-8B, exemple officiel `run_qwen3-8b.sh`)
- `enable_thinking` contrôlable via chat_template (comme Qwen3)
- GGUF Q8_0 (~8.5GB) tient sur RTX 3500 Ada 12GB local
- Tokenizer DeepSeek-R1-0528 (pas Qwen3 standard — à patcher pour désactiver `<think>`)

**Avantage clé vs CARD (2603.01089) :** CARD conditionne sur des feature vectors d'environnement via GCN. SAGE conditionne sur le **raisonnement profond** du modèle R1 — le chain-of-thought est la "condition", bien plus riche qu'un vecteur GCN. Le modèle raisonne sur la tâche PUIS structure la topologie.

**Concurrents :**
- **The Conductor** (2512.04388, ICLR 2026) — Qwen2.5-7B GRPO, 6 providers, BigCodeBench 40.0%. Pas open-source.
- **AgentConductor** (2602.17100) — Qwen2.5-3B GRPO, density S_complex, CodeContests 38.8%. Pas open-source.
- **CARD** (2603.01089, ICLR 2026) — GCN conditionnel, price penalty. Code MIT.

**Différenciation SAGE :** Seul système open-source combinant RL topology + R1-level reasoning + micro-décisions checkpoints + Rust formal verification + 8 providers + episodic memory + edge-level credit.

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
Modèle: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B (patched tokenizer)
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
| Template RunPod | `Runpod Pytorch 2.4.0` (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`) |
| Modèle | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` (Qwen3 transformer, MIT, R1 reasoning distilled) |
| Fallback | `Qwen/Qwen3-8B` (même arch, sans R1 distillation) |
| Framework | verl 0.7.1 + GiGPO plugin (Phase A/B) → verl-agent (Phase C) |
| Algorithme | GiGPO (`adv_estimator=gigpo`, `enable_similarity=True`, `similarity_thresh=0.85`) |
| LoRA | r=64, alpha=32, target=all-linear |
| GPU | 1x H100 80GB SXM (~39GB utilisés / 81GB total) |
| Données | 2225 entries (Phase A) → ~600 curated (Phase B) |
| Tokenizer | R1-0528 tokenizer — patcher `<think>` via `enable_thinking=False` dans vLLM ou `patch_tokenizer.py` |
| Attention | SDPA natif PyTorch (`VLLM_ATTENTION_BACKEND=TORCH_SDPA`) — pas de flashinfer |
| Providers Phase B | DeepSeek, Google, OpenAI, xAI, MiniMax, Kimi, OpenRouter, Codex |
| GGUF local | Q8_0 (~8.5GB) sur RTX 3500 Ada 12GB |

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
# → export LoRA → merge DeepSeek-R1-0528-Qwen3-8B → push HuggingFace → Q8 GGUF
```

**Résultat :** `yannabadie/sage-topology-policy-v2` sur HuggingFace + Q8_0 GGUF (~9.5GB) pour local.

---

## Troubleshooting

### Modèle crash vLLM
```bash
# DeepSeek-R1-0528-Qwen3-8B est transformer standard — ne devrait PAS crash.
# Si problème quand même, fallback :
SAGE_MODEL="Qwen/Qwen3-8B" bash scripts/verl/train_topology_v3.sh
# Ou: ajouter VLLM_ATTENTION_BACKEND=TORCH_SDPA (déjà dans train_topology_v3.sh)
```

### flash_attn / causal_conv1d non nécessaires
DeepSeek-R1-0528-Qwen3-8B est transformer standard. Utiliser SDPA natif :
```bash
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
# Pas besoin de flash_attn ni causal_conv1d
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

## Tests : 404 (47 Python + 357 Rust), 0 failures
