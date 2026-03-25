# YGN-SAGE V2 — Plan d'entraînement topology (RunPod H100)

> **Ce document est la référence pour l'opérateur (humain ou Claude Code) sur le pod.**
> Il décrit 3 phases progressives. Chaque phase a ses propres critères de succès.

---

## Qu'est-ce que YGN-SAGE ?

**YGN-SAGE** (Self-Adaptive Generation Engine) est un **Agent Development Kit** — un agent autonome de type Claude Code, Devin ou OpenSage, mais piloté par un moteur multi-agents **apprenant**. C'est un OpenSage (arXiv 2602.16891, UC Berkeley) profondément amélioré : là où OpenSage crée ses agents par prompting à chaque run (et repart de zéro), SAGE **apprend** de ses exécutions passées via RL et adapte ses topologies en temps réel.

### Philosophie : 3 principes fondateurs

**1. Rust first, Python tolerant**
Tout ce qui est performance-critique est en Rust (sage-core, compilé via PyO3/maturin) : TopologyGraph (petgraph), Density S_complex, HybridVerifier (SMT/LTL), QualityLabeler (tree-sitter + OxiZ), S-MMU (4-graph mémoire), ModelAssigner, SystemRouter, ContextualBandit. Python sert uniquement à l'orchestration (pipeline, agent loop, providers, training). Le coeur Rust garantit des latences sub-milliseconde pour le scoring et la vérification — critique pendant le training RL où chaque step est évalué.

**2. Minimal heuristics**
Les décisions critiques sont soit formellement vérifiées (Z3/OxiZ SMT), soit apprises (RL, bandit), soit backed par un papier. Les seuils d'adaptation (THETA_GOOD=0.7, THETA_CRITICAL=0.3) sont des priors calibrés, pas des constantes magiques. Le QualityEstimator retourne None (abstention) quand il ne peut pas évaluer — le contrôleur utilise un default 0.5 explicitement tracké. Le QualityLabeler Rust utilise tree-sitter + Z3 — pas de "if len(output) > 10". Le routing utilise kNN sur arctic-embed-m (92% accuracy) — pas de regex sur des mots-clés. Les reward weights (0.20/0.35/0.20/0.15/0.10) sont des valeurs initiales sujettes à ablation, pas des constantes magiques.

**3. Evidence before assertions**
On ne claim pas que ça marche — on prouve. 2067+ tests (1778 Python + 289 Rust base). BigCodeBench Hard comme benchmark principal (pas HumanEval qui est saturé). Chaque décision architecturale a une référence papier (voir table en bas).

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
   │   ├── MAP-Elites + CMA-ME   ├── 7 Providers + Codex
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
5. EXECUTE   — TopologyRunner + ProviderPool (7 providers + Codex) → exécution nœud par nœud
6. LEARN     — QualityEstimator Z3 → Bandit update + MAP-Elites archive + episodic memory
```

### Ce qui différencie SAGE d'OpenSage

| Aspect | OpenSage (Berkeley) | YGN-SAGE |
|--------|-------------------|----------|
| Topologie | Promptée à chaque run | **Apprise par RL** (GiGPO) |
| Adaptation | Runtime prompting | **Apprise** (checkpoints, fallback_tier dans le YAML) |
| Mémoire | Graph-based hierarchical | **4-tier** (STM → Episodic → Semantic → ExoCortex) + cross-episode training |
| Verification | Aucune | **Formelle** (Rust SMT/LTL, QualityLabeler Z3) |
| Providers | Multi-model (GPT/Claude/Gemini) | **7 providers + Codex** câblés DANS le training |
| Engine | Python pur | **Rust core** (PyO3, sub-ms latency) |
| Self-programming | Agents créent des agents | **Idem** (`agent_mgmt.py`) + topologies apprises |
| Code | 2067+ tests | **Open-source MIT** |
| Protocol | Aucun standard | **A2A v1.0** (Google) |
| Benchmark | SWE-bench Pro 59% | BigCodeBench Hard 37.8% (→ cible >40%) |

### Interface cible : pi-mono

En production, SAGE sera pilotable via **pi-mono** (github.com/badlogic/pi-mono) — un toolkit TypeScript pour agents AI avec multi-provider API unifiée (`@mariozechner/pi-ai`), agent runtime (`pi-agent-core`), et web UI (`pi-web-ui`). L'intégration se fait via le serveur A2A de SAGE (`a2a_server.py`) qui expose les skills topology/code/research.

### Standard A2A

SAGE implémente le protocole **Agent-to-Agent v1.0** (Google) via `a2a_server.py`. L'agent est exposé comme un `AgentCard` avec 3 skills (general, code, research). N'importe quel client A2A (Google ADK, LangGraph, CrewAI) peut déléguer des tâches à SAGE. Le modèle de topologie entraîné produira des YAML qui respectent les conventions A2A pour l'interopérabilité.

---

## Vision training

Entraîner **nvidia/Nemotron-Orchestrator-8B** (NVIDIA Open Model License) via GiGPO pour générer des topologies multi-agents **adaptatives** — capables de se corriger en cours d'exécution. C'est le Path 6 de la DynamicTopologyEngine.

**Pourquoi Nemotron-Orchestrator-8B (et pas Qwen3-8B vanilla, DeepSeek-R1, etc.) :**
- **GRPO-trained for orchestration** : entraîné par NVIDIA via GRPO multi-objectif (accuracy + cost + preference) pour décider quel modèle/outil utiliser à chaque étape — exactement nos décisions GiGPO de checkpoint (arXiv 2511.21689, github.com/NVlabs/ToolOrchestra)
- **Base Qwen3-8B** : même architecture transformer = même script verl, zéro modification training loop
- **HLE 37.1%** : bat GPT-5 sur Humanity's Last Exam, excelle aussi sur Tau2-Bench et FRAMES
- **vLLM compatible** : architecture Qwen3, SDPA natif, aucun bug kernel
- **1× H100 confortable** : ~17GB actor + ~22GB vLLM = 39GB/81GB, marge large
- **NVIDIA Open Model License** : fine-tuning + commercial + derivatives OK (permissive)
- **GGUF disponible** : `Mungert/Nemotron-Orchestrator-8B-GGUF` pour inférence locale
- **Pas de tokenizer patching** : `/no_think` dans le system prompt suffit (hérité de Qwen3)

**Modèles rejetés et pourquoi :**
| Modèle | Raison du rejet |
|--------|----------------|
| Qwen/Qwen3-8B (vanilla) | Bon modèle de base mais pas pré-entraîné pour l'orchestration — Nemotron ajoute le GRPO orchestration par-dessus |
| DeepSeek-R1-0528-Qwen3-8B | `<think>` CoT gaspille 40× les tokens utiles ; tokenizer patching fragile |
| Qwen3.5-9B | GDN kernel crashes vLLM (issue #34948), flashinfer compilation 5h+ |
| Falcon-H1R-7B | Mamba2 hybrid = mêmes problèmes causal_conv1d ; Falcon license restrictive |
| Qwen3-14B | Pas de script verl officiel ; tight VRAM sur 1× H100 avec vLLM |
| Klear-AgentForge-8B | SWE-bench 39% (excellent) mais README vide, pas de script verl, risque intégration |

**Pourquoi upgrader de Qwen3-8B vanilla :** Nemotron-Orchestrator-8B est un Qwen3-8B fine-tuné par NVIDIA spécifiquement pour les décisions d'orchestration (quel outil, quel modèle, quand déléguer). Notre GiGPO s'appuie donc sur des poids déjà orientés orchestration — warm start parfait.

**Concurrents :**
- **The Conductor** (2512.04388, ICLR 2026) — Qwen2.5-7B GRPO, 6 providers, BigCodeBench 40.0%. Pas open-source.
- **AgentConductor** (2602.17100) — Qwen2.5-3B GRPO, density S_complex, CodeContests 38.8%. Pas open-source.
- **CARD** (2603.01089, ICLR 2026) — GCN conditionnel, price penalty. Code MIT.

**Différenciation SAGE :** Seul système open-source combinant RL topology + GiGPO multi-step micro-décisions + Rust formal verification + 7 providers + Codex + episodic memory + edge-level credit.

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

### Phase A — Structural warm-up (single-turn, $0 API)

**Ce qu'on entraîne :** Le modèle génère du YAML en un shot. Pas d'interaction multi-step.

**Ce que le modèle apprend :** Compétences 1, 2, 3, 4, 6, 7 — structurer un DAG, assigner les tiers, placer checkpoints et fallbacks, adapter la complexité, écrire un reasoning.

**Ce que le modèle n'apprend PAS :** Compétence 5 (micro-décisions). Il ne voit jamais si ses topologies fonctionnent réellement. Le reward est purement structural.

**Framework :** verl 0.7.1 + GRPO. Single-turn : le modèle génère, le reward évalue le YAML statiquement.

**GRPO en Phase A :** Pour du single-turn, GRPO est fonctionnellement équivalent à GiGPO (pas de multi-step à grouper). C'est un warm-up qui enseigne le format. La vraie Phase C dynamique utilise GiGPO avec des anchor states.

**Reward :**
```
R = _score_format(yaml)           — YAML valide ? [-2.0, +1.0]
  + _score_structure(yaml)         — nodes, edges, roles, reasoning [0.0, 1.0]
  + _score_rust_density(yaml)      — Rust TopologyDensity S_complex [0.0, 1.0]
  + bonus adaptation               — +0.1 si adaptation block, +0.1 si fallback_tier
```
Refs : `_score_format`, `_score_structure` dans reward.py. `TopologyDensity` dans sage-core/src/topology/density.rs. S_complex inspiré d'AgentConductor (2602.17100).

**Dataset :** 12,303 entries (verl_topology_train.parquet).

**Config :**
```
Modèle: nvidia/Nemotron-Orchestrator-8B (SFT-merged, patched tokenizer)
Algorithme: GRPO via verl 0.7.1 (single-turn, warm-up pour Phase C)
LoRA: r=64, alpha=32, all-linear
LR: 1e-6 (V5 fix, was 5e-5 in V3/V4)
max_response_length: 1024 (V5 fix, was 512)
Epochs: 3
Batch: 32 (train), K=4 rollouts
Coût API: $0
Durée estimée: ~29h H100
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

**Ce qu'on ajoute :** Le reward inclut maintenant l'exécution réelle. Chaque topologie est exécutée via `TopologyRunner` + `ProviderPool` avec les 7 providers + Codex LLM. Le code produit est testé en sandbox.

**Ce que le modèle apprend en plus :** Que `model_tier: reasoner` sur le planner + `fast` sur le coder produit de MEILLEURS résultats que tout en `budget`. Que certaines combinaisons de rôles fonctionnent et d'autres non. Le lien causal entre la topologie et le résultat.

**Framework :** Même que Phase A, mais `SAGE_VERL_EXEC=1`.

**Reward :**
```
R = 0.30 × R_structural           — format + density + verifier (identique Phase A)
  + 0.70 × R_execution            — PASSED=1.0, WRONG_ANSWER=0.5, RUNTIME_ERROR=0.3, TIMEOUT=0.2
```
Le modèle voit la CONSÉQUENCE de ses choix de topologie.

Refs : `evaluate_topology()` dans execution/__init__.py. `TopologyRunner` dans topology/runner.py. 7 providers + Codex dans config/cards.toml.

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

### Phase C — Micro-décisions multi-step (READY, scripts + env integration wired)

**Ce qu'on ajoute :** L'environnement `SageTopologyEnv` (machine 4 états) est branché dans le training loop. Le modèle interagit avec l'env : il génère le YAML, voit les résultats noeud par noeud, et prend des décisions upgrade/continue/reroute aux checkpoints.

**Ce que le modèle apprend en plus :** Compétence 5 — QUAND upgrader vs continuer. Le coût-bénéfice de l'adaptation. C'est LE différenciateur vs tous les concurrents.

**La machine 4 états (`topology_env.py`) :**
```
awaiting_yaml ──[model generates YAML]──> executing
executing ──[run nodes one by one]──> awaiting_decision (at checkpoint)
                                  └──> terminal (all nodes done)
awaiting_decision ──[model: continue]──> executing (resume)
                  ├──[model: upgrade]──> executing (re-run node with fallback_tier)
                  └──[model: reroute]──> terminal (abort, -0.3 penalty)
```
The model takes REAL actions at 2 types of steps:
1. **Step 0 (awaiting_yaml):** Generate the YAML topology (same as Phase A/B)
2. **Checkpoint steps (awaiting_decision):** Decide continue/upgrade/reroute based on node output quality

**Micro-decision reward constants (`topology_env.py`):**
- `_REWARD_UPGRADE_COST = -0.05` — every upgrade has a cost
- `_REWARD_REROUTE_PENALTY = -0.30` — reroute = abort + restart
- `_REWARD_UPGRADE_SUCCESS = +0.15` — upgrade improved quality

**GiGPO en Phase C :** Les VRAIS anchor states fonctionnent. `decision:coder:moderate:low` groupe toutes les trajectoires où le coder a produit du mauvais code. GiGPO compare les décisions upgrade vs continue pour ce même anchor. Step-level advantage réel.

#### Approche retenue : Custom GiGPO loop (train_phase_c_custom.py)

**Pourquoi pas verl-agent :** L'installation de verl-agent a échoué sur le pod (git auth).
Le script `train_phase_c_custom.py` implémente GiGPO directement avec PyTorch + PEFT, sans dépendance externe. Il exerce la machine 4 états complète de `SageTopologyEnv`.

```bash
# Sur le pod, après Phase A convergée :
python3 scripts/verl/train_phase_c_custom.py \
    --model /workspace/sft_merged_model \
    --checkpoint /workspace/topology_verl_output \
    --data data/verl_topology_phase_c.parquet \
    --output /workspace/topology_verl_phase_c \
    --epochs 3 --lr 5e-7 --k 4 --batch-size 4 \
    --memory-db /workspace/training_memory.db
```

**Dataset Phase C :** `verl_topology_phase_c.parquet` — 12,303 entries dont 43% avec checkpoints (enrichi via `enrich_dataset_checkpoints.py`). Le mix single-turn/multi-step permet au modèle de consolider le format YAML tout en apprenant les micro-décisions.

**Ce qui rend cette Phase C RÉELLEMENT dynamique :**
1. Le modèle prend >1 décision par épisode (YAML + N checkpoint decisions)
2. Les décisions sont groupées par anchor state → GiGPO normalise within-group
3. Edge credit (Graph-GRPO) et RewardFlow (PageRank) distribuent le crédit par nœud/edge
4. L'épisodic memory (SQLite) réinjecte les expériences passées dans les observations
5. Le reward est majoritairement execution-based (35% exec + 20% rewardflow + 15% resilience)

**Avantages :** Simple, pas de dépendance verl-agent, exerce la machine 4 états complète.
**Inconvénients :** Plus lent (PyTorch generate au lieu de vLLM), pas de token masking.

**Reward (5 signaux) :**
```
R = 0.20 * R_structural            — format + density + verifier
  + 0.35 * R_execution             — PASSED=1.0, WRONG_ANSWER=0.5, ...
  + 0.20 * R_rewardflow            — PageRank per-node credit (arXiv 2603.18859)
  + 0.15 * R_resilience            — bonus adaptation triggered + succeeded
  + 0.10 * R_cost_efficiency       — CARD price penalty (arXiv 2603.01089)
```
Refs : `rewardflow.py` (RewardFlow), `_score_resilience` + `_score_cost_efficiency` dans reward.py.

**Modules (tous implémentés ET wired) :**
- `topology_env.py` — Machine 4 états, 7 micro-decision tests passent
- `env_register.py` — Monkey-patch verl-agent `make_envs()`, config extraction from Hydra
- `rewardflow.py` — PageRank propagation
- `training_memory.py` — SQLite episodic memory cross-épisode (via `SAGE_TRAINING_MEMORY_DB` env var)
- `edge_credit.py` — Graph-GRPO per-edge advantage (arXiv 2603.02701)
- `step_reward.py` — `StepRewardVector` with `to_verl_format()` for GiGPO

**Token masking (critique pour Approach A) :** verl-agent masque automatiquement les observations (mask=0). Seuls les tokens générés par le modèle (YAML, "upgrade", "continue") reçoivent des gradients. A vérifier sur le premier batch.

**Config Phase C :**
```
Modèle: Checkpoint Phase A/B (LoRA sur Nemotron-Orchestrator-8B)
Algorithme: GiGPO (multi-step anchor grouping)
Env: SageTopologyEnv (4-state machine, max_steps=10)
LoRA: r=64, alpha=32, all-linear (continue from Phase A/B)
LR: 5e-7 (lower than Phase A to preserve structural knowledge)
Epochs: 3
Batch: 32 (train), K=4 rollouts
Temperature: 0.7 (diversity for GiGPO grouping)
Providers: DeepSeek, Google, OpenAI, xAI, MiniMax, Kimi, OpenRouter, Codex
Memory: SQLite episodic (cross-epoch learning)
Coût API: ~$50-80
Durée estimée: 2-3h H100
```

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
| Template RunPod | `Runpod Pytorch 2.4.0` (runpod-torch-v240) |
| GPU | **1x H100 NVL 94GB** |
| Disque système | **110 GB** |
| Disque stockage | **80 GB** (monté sur `/workspace`) |
| Modèle | `nvidia/Nemotron-Orchestrator-8B` (Qwen3 architecture, NVIDIA Open Model License, GRPO-trained orchestrator) |
| Framework | verl 0.7.1 + GiGPO plugin |
| Algorithme | GiGPO (`adv_estimator=gigpo`, `enable_similarity=True`, `similarity_thresh=0.85`) |
| LoRA | r=64, alpha=32, target=all-linear |
| VRAM estimé | ~17GB actor + ~22GB vLLM = **~39GB / 94GB** — marge 55GB |
| Dataset | **12,303 entries** (Phase A) → ~600 curated (Phase B) |
| Tokenizer | Nemotron tokenizer (Qwen3 base) — `<think>` désactivé via `/no_think` system prompt |
| Attention | SDPA natif PyTorch (`VLLM_ATTENTION_BACKEND=TORCH_SDPA`) |
| Providers Phase B | DeepSeek, Google, OpenAI, xAI, MiniMax, Kimi, OpenRouter, Codex |
| Post-training | Merge → HuggingFace (`yannabadie/sage-topology-policy-v2`) + GGUF Q8_0 (8.7GB, local RTX 3500) |
| Opérateur | **Claude Code** en mode autonome (`--dangerously-skip-permissions`) sous user `yann` |

## Coût estimé

| Phase | GPU | API | Total |
|-------|-----|-----|-------|
| A (structural) | ~$6 (~2h) | $0 | ~$6 |
| B (execution) | ~$6 (~2h) | ~$50-80 | ~$56-86 |
| C (micro-décisions) | ~$10 (~3h) | ~$50-80 | ~$60-90 |
| **Total 3 phases** | ~$22 | ~$100-160 | **~$122-182** |

---

## Étapes sur le pod

### Étape 0 — Créer le pod RunPod

1. [console.runpod.io](https://console.runpod.io) → Deploy → GPU Cloud
2. Template : **Runpod Pytorch 2.4.0** (`runpod-torch-v240`)
3. GPU : **H100 NVL** (94 GB)
4. Container Disk : **110 GB**
5. Volume Disk : **80 GB** (monté sur `/workspace`)
6. Deploy

### Étape 1 — Créer l'utilisateur et installer Claude Code

Se connecter en root (web terminal ou SSH), puis :

```bash
# 1. Créer l'utilisateur yann avec tous les droits
useradd -m -s /bin/bash -G sudo yann
echo "yann ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/yann
chmod 440 /etc/sudoers.d/yann

# 2. Donner accès au workspace
chown -R yann:yann /workspace

# 3. Installer Claude Code
curl -fsSL https://claude.ai/install.sh | bash

# 4. Configurer le PATH pour yann
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/yann/.bashrc

# 5. Basculer sur yann
su - yann
```

### Étape 2 — Cloner le repo et configurer

```bash
# En tant que yann
git clone https://github.com/yannabadie/YGN-SAGE.git /workspace/YGN-SAGE
cd /workspace/YGN-SAGE && git checkout VeRLGIGPO

# Créer le .env avec les clés API
cat > .env << 'EOF'
DEEPSEEK_API_KEY="<ta clé>"
GOOGLE_API_KEY="<ta clé>"
OPENAI_API_KEY="<ta clé>"
GROK_API_KEY="<ta clé>"
MINIMAX_API_KEY="<ta clé>"
KIMI_API_KEY="<ta clé>"
OPEN_ROUTER_API_KEY="<ta clé>"
HF_TOKEN="<ton token>"
ANTHROPIC_API_KEY="<ta clé Anthropic pour Claude Code>"
EOF
```

### Étape 3 — Lancer Claude Code en mode autonome

```bash
cd /workspace/YGN-SAGE
claude --dangerously-skip-permissions
```

### Étape 4 — Prompt pour Claude Code

Coller ce prompt dans Claude Code :

```
Lis RUNPOD_PLAN.md et AI-ARCHITECTURE.md. Tu es sur un pod RunPod H100 NVL 94GB.
User: yann (sudo). Branche: VeRLGIGPO.

Exécute le plan COMPLET de bout en bout (3 phases + post-training) :

PHASE A — Structural GiGPO ($0 API)
1. bash sage-python/scripts/verl/setup_runpod.sh
2. cd sage-python && python3 scripts/verl/validate_setup.py (10/10 checks)
3. bash scripts/verl/train_topology_v3.sh 2>&1 | tee /workspace/train_phase_a.log
   Critères succès: reward/mean > 0.7, YAML parsable > 90%

PHASE B — Execution GiGPO (~$50-80 API)
4. export SAGE_VERL_EXEC=1
5. Relancer le training avec le dataset curated et le checkpoint Phase A
   Critères succès: reward/mean > 0.5, PASSED rate > 30%

PHASE C — Micro-décisions multi-step (le différenciateur SAGE)
6. Approche A (préférée): bash scripts/verl/train_topology_phase_c.sh 2>&1 | tee /workspace/train_phase_c.log
   OU Approche B (fallback si verl-agent ne marche pas):
   python3 scripts/verl/train_phase_c_custom.py --model /workspace/patched_nemotron_orchestrator \
     --checkpoint [checkpoint Phase B] --data data/verl_topology_curated.parquet \
     --output /workspace/topology_verl_phase_c --epochs 3 --k 4 --memory-db /workspace/training_memory.db
   Critères succès: step_advantage non-nul, anchors decision:* dans les logs,
   upgrade quand quality < threshold, BigCodeBench Hard > 40%

POST-TRAINING
7. python3 scripts/verl/benchmark_post_train.py --bench all --limit 20
8. python3 scripts/verl/post_training_pipeline.py all
   → export LoRA → merge Nemotron-Orchestrator-8B → push HuggingFace → GGUF Q8_0

Le modèle est nvidia/Nemotron-Orchestrator-8B (Qwen3 arch, GRPO-trained orchestrator).
Si le tokenizer a <think> mode actif, patch avec patch_tokenizer.py.
Si vLLM crash, essaie VLLM_ATTENTION_BACKEND=TORCH_SDPA.
Si verl-agent env registration échoue en Phase C, utilise train_phase_c_custom.py (Approche B).

Push le modèle sur HuggingFace: yannabadie/sage-topology-policy-v2
  - Merged float16 (~16GB) pour serveurs/pods
  - GGUF Q8_0 (~8.7GB) dans /gguf/ pour local RTX 3500 Ada 12GB
Commit et push les résultats + logs sur GitHub (branche VeRLGIGPO).

Si un step échoue, diagnostique, fixe, et réessaie. Log tout dans /workspace/.
L'ExoCortex (500+ papiers) est accessible via:
python3 -c "from openai import OpenAI; import os; c=OpenAI(api_key=os.environ['OPENAI_API_KEY']); r=c.responses.create(model='gpt-4o-mini', input='query', tools=[{'type':'file_search','vector_store_ids':['ygnsageresearch-wii7kwkqozrd']}]); print(r.output_text)"
```

### Signaux de succès

| Phase | Critère | Target |
|-------|---------|--------|
| A (structural) | `reward/mean` | > 0.7 |
| A | YAML parsable | > 90% |
| A | Adaptation blocks | > 50% moderate/complex |
| B (execution) | `reward/mean` | > 0.5 |
| B | PASSED rate | > 30% |
| B | BigCodeBench Hard (20 tasks) | > 38% |
| **C (micro-décisions)** | `step_advantage` | **non-nul** |
| **C** | Anchors `decision:*` | **dans les logs** |
| **C** | Model chooses upgrade when quality < threshold | **observable** |
| **C** | BigCodeBench Hard (20 tasks) | **> 40%** (battre The Conductor) |

### 4. Training Phase B (après Phase A)

```bash
export SAGE_VERL_EXEC=1
# Relancer avec le checkpoint Phase A et le dataset curated
```

### 5. Training Phase C (après Phase B)

**Approche A (verl-agent multi-turn) :**
```bash
screen -S train_c
bash scripts/verl/train_topology_phase_c.sh 2>&1 | tee train_phase_c.log
```

**Signaux de succès :**
- `step_advantage` non-null dans les logs (GiGPO multi-step fonctionne)
- Des anchors `decision:coder:moderate:low` apparaissent
- Le modèle prend des décisions variées (pas tout "continue")

**Si Approche A échoue** (verl-agent env_manager incompatible) :
```bash
# Fallback: custom training loop
python3 scripts/verl/train_phase_c_custom.py \
    --checkpoint /workspace/topology_verl_output \
    --data data/verl_topology_curated.parquet \
    --output /workspace/topology_verl_phase_c_custom \
    --epochs 3 --lr 5e-7 --k 4 --batch-size 8 \
    --memory-db /workspace/topology_training_memory.db \
    2>&1 | tee train_phase_c_custom.log
```

### 6. Post-training

```bash
python3 scripts/verl/post_training_pipeline.py all
# -> export LoRA -> merge Nemotron-Orchestrator-8B -> push HuggingFace -> Q8 GGUF
```

**Résultat :** `yannabadie/sage-topology-policy-v2` sur HuggingFace + Q8_0 GGUF (~9.5GB) pour local.

---

## Troubleshooting

### Modèle crash vLLM
```bash
# Nemotron-Orchestrator-8B est Qwen3 architecture (transformer standard) — ne devrait PAS crash.
# Si problème quand même :
VLLM_ATTENTION_BACKEND=TORCH_SDPA bash scripts/verl/train_topology_v3.sh
# Ou: ajouter VLLM_ATTENTION_BACKEND=TORCH_SDPA (déjà dans train_topology_v3.sh)
```

### flash_attn / causal_conv1d non nécessaires
Nemotron-Orchestrator-8B est Qwen3 architecture (transformer standard). Utiliser SDPA natif :
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
├── train_topology.sh           # GiGPO config, Phase A/B (verl-agent)
├── train_topology_v3.sh        # GiGPO Phase A only (verl 0.7.1)
├── train_topology_phase_c.sh   # Phase C: multi-step micro-decisions (verl-agent)
├── train_phase_c_custom.py     # Phase C fallback: custom GiGPO loop (no verl)
├── patch_tokenizer.py          # Qwen3.5 <think> removal
├── setup_runpod.sh             # 9-step setup
├── validate_setup.py           # 10 pre-flight checks
├── post_training_pipeline.py   # Export -> HF -> GGUF
└── convert_sft_to_verl.py      # 11 sources -> 2225 entries

sage-core/src/topology/
├── topology_graph.rs     # TopologyNode + TopologyGraph (6 adaptive fields)
├── reward.rs             # RewardScore (resilience + cost_efficiency)
├── density.rs            # S_complex density function
└── verifier.rs           # PyHybridVerifier (acyclicity, connectivity)
```

## Tests : 2067+ (1778 Python + 289 Rust), 0 failures
