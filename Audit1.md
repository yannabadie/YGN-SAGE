# Audit Technique Impitoyable — YGN-SAGE

**Date** : 24 mars 2026  
**Auditeur** : Chercheur senior MAS/ML (sans complaisance)  
**Base** : AI-ARCHITECTURE.md (Branch: VeRLGIGPO, 404 tests passing)

---

## 1. Génération topologique : adaptativité réelle ou déclarative ?

### Mécanisme de génération : réellement adaptatif ?

**Oui, mais partiellement.** Le `TopologyEngine` (Rust, `topology/engine.rs`) implémente 6 chemins de génération :

| Path | Mécanisme | Adaptatif ? |
|------|-----------|-------------|
| 1 | S-MMU hit (similarité > 0.7, qualité > 0.5) → clone | ✅ Oui, basé sur retrieval |
| 2 | MAP-Elites archive lookup | ✅ Oui, quality-diversity |
| 3 | LLM synthesis (Python callback) | ⚠️ Prompt-based |
| 4 | Mutation (best-from-archive + random) | ✅ Oui, évolution |
| 5 | MCTS search (UCB1 over mutations) | ✅ Oui, search-based |
| 6 | Template fallback (S1→seq, S2→avr, S3→debate) | ❌ Non, statique |

**Preuve d'adaptativité runtime** : `TopologyController` (`topology_controller.py`) expose 4 actions : `upgrade_model`, `spawn_subagent`, `reroute`, `prune`. L'`agent_loop.py` mentionne "S3 escalation on repeated failure" et "DriftMonitor for sliding-window analysis".

### Feedback loop : chemin exact

```
Observation → QualityEstimator.estimate() [5 signals: non-empty, length, code, errors, AVR]
           ↓
Décision → ContextualBandit.record_outcome() [Beta/Gamma posteriors, Thompson sampling]
           ↓
Reconfiguration → TopologyEngine.record_outcome() → MapElitesArchive.insert() [si Pareto-dominant]
               → S-MMU.register_chunk() [pour retrieval futur]
```

**Manque identifié** : Le document indique "veRL training pending" pour GiGPO. La boucle d'apprentissage policy-gradient n'est pas encore active en production — seulement le bandit et MAP-Elites.

### Granularité d'adaptation

| Niveau | Implémenté ? | Composant |
|--------|--------------|-----------|
| Ajout/suppression d'agents | ✅ | `mutations.rs` : add_node, remove_node |
| Rerouting de flux | ✅ | `mutations.rs` : add_edge, remove_edge |
| Changement modèle par agent | ✅ | `model_assigner.rs` : per-node assignment |
| Modification prompts/paramètres | ⚠️ | Non documenté explicitement |
| Changement système (S1/S2/S3) | ✅ | `mutations.rs` : change_system (±1) |

### Comparaison état de l'art

| Projet | Mécanisme | Différence vs YGN-SAGE |
|--------|-----------|------------------------|
| **AutoGen** | GroupChat patterns fixes | SAGE : topologies apprises, pas de patterns hardcodés |
| **CrewAI** | Process (sequential/hierarchical) | SAGE : 8 templates + évolution + LLM synthesis |
| **LangGraph** | Conditional edges (manuelles) | SAGE : edges appris via Graph-GRPO edge credit |
| **MetaGPT** | SOP graphs statiques par task type | SAGE : topologies dynamiques via S-MMU retrieval |
| **AgentConductor** (arXiv 2602.17100) | S_complex density function | SAGE : implémentation complète citée comme référence |

**Apport concret de YGN-SAGE** :
1. GiGPO multi-step pour politique de topologie (pas juste single-step GRPO)
2. MAP-Elites 4D grid (108 cells) pour quality-diversity
3. Assignment hétérogène par noeud (pas un modèle pour tout le DAG)
4. S-MMU 4 vues orthogonales pour retrieval de topologies passées

### Verdict

**[Adaptatif avec apprentissage (partiel)]**

Le mécanisme d'adaptation runtime existe (controller, bandit, MAP-Elites). Cependant, l'apprentissage de politique via GiGPO est marqué "pending" — le système adapte via bandit Thompson sampling et archive lookup, mais pas via policy gradient end-to-end. C'est plus qu'un système semi-dynamique, moins qu'un système pleinement appris.

---

## 2. Mémoire et apprentissage continu : viable ou vaporware ?

### Mémoire dédiée à la génération topologique

**Existence confirmée** : `TopologySmmuBridge` (`topology/smmu_bridge.rs`) stocke les `TopologyOutcome` dans S-MMU :

```rust
TopologyOutcome {
    topology_id, task_summary, keywords, embedding,
    template, quality, cost, latency, structural_features
}
```

**Structures de données** :
- `MultiViewMMU` : 1 graphe unique avec 4 types d'arêtes (Temporal, Semantic, Causal, Entity)
- Stockage : in-memory `DiGraph<ChunkMetadata, MultiEdge>` (petgraph)
- Persistance : `compact_to_arrow_with_meta()` → Arrow RecordBatch (7 colonnes)
- Cache RAG : `RagCache` (DashMap, FIFO+TTL)

**Mécanisme de rétention/oubli** :
- Auto-GC à 10 000 chunks
- `page_out_candidates()` : éviction par pertinence (graph distance)
- Utility-based eviction : `recency * access_count`

### Apprentissage continu : implémenté ou architecturé ?

| Composant | Statut | Preuve |
|-----------|--------|--------|
| ContextualBandit | ✅ Fonctionnel | Beta/Gamma posteriors, Thompson sampling, SQLite persistence (feature: cognitive) |
| MAP-Elites Archive | ✅ Fonctionnel | 4D grid, 108 cells, Pareto insertion |
| GiGPO Policy Training | ⚠️ Pending | "veRL training pending" — env ready, update pas encore wire |
| ModelRegistry Telemetry | ✅ Fonctionnel | Calibrated affinity : w = min(count/50, 0.8), blend card + observed |

**Verdict** : Ce n'est pas du vaporware. Le bandit et MAP-Elites sont opérationnels. Mais l'apprentissage de politique (le coeur de la revendication "learned topology policy") n'est pas encore actif.

### Si c'est du prompt engineering itératif ?

Le Path 3 (LLM synthesis) est du prompt engineering : "3-stage LLM topology pipeline: role assignment → structure design → validation". Ce n'est pas déguisé — c'est documenté comme tel. **Viabilité** : pas de limite de context window mentionnée pour ce path (le LLM génère du YAML, pas du CoT long). Risque faible.

### Scalabilité mémoire

| Mécanisme | Présent ? | Détail |
|-----------|-----------|--------|
| Compaction | ✅ | Arrow RecordBatch, zero-copy |
| Summarization | ✅ | `MemoryEvent::summary(content)` flag |
| Éviction | ✅ | Semantic paging + utility-based + auto-GC 10K |
| Overflow protection | ✅ | Hard limit 10 000 chunks |

**Problème** : `RustEntityGraph` — "SQLite persistence not yet wired; in-memory only". Si le système tourne longtemps, l'entity graph va saturer la RAM.

### Verdict

**[Fonctionnel mais incomplet]**

La mémoire S-MMU est production-ready (éviction, compaction, GC). L'apprentissage bandit/MAP-Elites est fonctionnel. Mais :
1. GiGPO policy training pending
2. Entity graph persistence not wired
3. 1965 training entries — volume faible pour policy learning

---

## 3. Choix du modèle à entraîner

### Entraîne-t-il un modèle ou orchestre-t-il ?

**Les deux.**

**Orchestration** : `ModelRegistry` gère 8 backends (Google, OpenAI, DeepSeek, xAI, Kimi, MiniMax, OpenRouter, etc.) via `providers/connector.py`. Routing intelligent basé sur :
- S1/S2/S3 affinity
- Domain score
- Cost/latency telemetry
- Calibrated affinity (card prior + observed quality)

**Entraînement** : `verl/topology_env.py` — GiGPO sur **Qwen3.5-9B** via veRL. Dataset : 1965 training entries dans `TrainingMemory` (SQLite).

### Le choix est-il justifié ?

**Qwen3.5-9B** :
- ✅ Taille raisonnable (9B = trainable sur 1-2 H100)
- ✅ Open weights (fine-tuning possible)
- ⚠️ Justification absente du document — pourquoi pas Llama-3.1-8B ? Mistral-Nemo-12B ?
- ⚠️ 1965 entries — très faible pour policy learning. Graph-GRPO papers utilisent typiquement 10K-100K trajectories.

### Recommandation

**Modèle à entraîner** : Conserver Qwen3.5-9B OU passer à **Llama-3.1-8B-Instruct**.

**Justification** :
| Critère | Qwen3.5-9B | Llama-3.1-8B |
|---------|------------|--------------|
| Taille | 9B | 8B |
| Community support | Moyen | Excellent |
| Tool use capability | Bon | Excellent |
| Coût fine-tuning | ~$1500 (1000 GPU-h H100) | ~$1200 |

**Dataset nécessaire** : Minimum 10K trajectories pour policy learning stable. Actuel : 1965. **Gap : 5x**.

**Le fine-tuning a-t-il un sens ?**

**Oui, mais conditionnel.** Si l'objectif est d'apprendre une politique de topologie (quel DAG pour quelle tâche), le fine-tuning est justifié — le routing seul ne peut pas apprendre des structures multi-agents complexes. **Cependant** :
- Avec 1965 entries, risque de overfitting élevé
- Alternative : in-context learning + RAG sur topologies passées (déjà via S-MMU)
- Recommendation : augmenter dataset avant de lancer le training

---

## 4. Type d'entraînement : est-ce le bon ?

### Type utilisé : GiGPO (Group-in-Group Policy Optimization)

**Référence** : arXiv 2505.10978

**Reward function** (`verl/reward.py`) :
```
1. Format scoring: YAML validity [-2.0, +1.0]
2. Structure scoring: roles/edges/capabilities [0.0, 1.0]
3. Execution scoring: sandbox pass@1 [0.0, 1.0]
4. Edge credit: Graph-GRPO integration
```

**Problèmes identifiés** :

1. **Reward misalignment potentiel** : La reward favorise la validité YAML et structurelle, pas nécessairement la réussite de la tâche utilisateur. Un DAG valide mais inefficace peut scorer haut.

2. **Credit assignment** : Graph-GRPO (`edge_credit.py`) et RewardFlow (`rewardflow.py`) sont implémentés — c'est un point fort. Mais la combinaison des deux n'est pas clairement pondérée.

3. **Dataset volume** : 1965 entries — insuffisant pour GiGPO stable. Risque de variance élevée des gradients.

### Coût/bénéfice estimé

| Poste | Estimation |
|-------|------------|
| GPU-hours (Qwen3.5-9B, 10K trajectories) | 500-1000 H100 hours |
| Coût cloud (RunPod H100 @ $2/h) | $1000-2500 |
| Temps dev pour wire veRL | 5-10 jours |
| Gain attendu vs bandit-only | +15-25% sur tâches complexes (S3) |

**Verdict** : Le coût est raisonnable. Le gain est incertain sans dataset plus large. **Recommandation** : collecter 10K+ trajectories avant training.

### Si aucun entraînement ?

Ce n'est pas le cas — l'infrastructure est là. Mais "pending" signifie que la valeur principale (politique apprise) n'est pas encore délivrée. **Choix rationnel** : mieux vaut wire correctement que lancer un training prématuré.

---

## 5. Améliorations — classées par impact/effort

#### Amélioration 1 : Persistance de l'Entity Graph

- **Problème identifié** : `RustEntityGraph` — "SQLite persistence not yet wired; in-memory only" (`memory/entity_graph.rs`). Risque d'overflow RAM sur sessions longues.
- **Impact si non résolu** : Crash mémoire après ~10K entities. Perte de contexte causal/semantic entre sessions.
- **Solution proposée** : Wire SQLite WAL pour `RustEntityGraph` (même schema que `TrainingMemory`). Ajouter `save_checkpoint()` et `load_checkpoint()` methods.
- **Effort estimé** : 3-5 jours-dev (Rust + SQLite binding)
- **Papier/Référence** : CoALA (Cognitive Architectures for Language Agents) — 3-tier memory avec persistance

#### Amélioration 2 : Expansion du Dataset d'Entraînement

- **Problème identifié** : 1965 training entries (`verl/training_memory.py`). Insuffisant pour GiGPO policy learning stable.
- **Impact si non résolu** : Overfitting, variance élevée, politique non généralisable.
- **Solution proposée** : Pipeline de synthèse : (1) collecter topologies réussies via S-MMU, (2) générer variantes via mutation operators, (3) exécuter et labeler via `QualityLabeler`. Cible : 10K trajectories.
- **Effort estimé** : 7-10 jours-dev (automation + validation)
- **Papier/Référence** : TopoCurate (arXiv 2603.01714) — topology-aware data curation

#### Amélioration 3 : Wire Complet veRL GiGPO

- **Problème identifié** : "veRL training pending" (`verl/topology_env.py`). L'env est prêt, mais l'update policy n'est pas wire dans `boot.py`.
- **Impact si non résolu** : Le système n'apprend pas de politique — seulement bandit + archive lookup. Valeur principale non délivrée.
- **Solution proposée** : (1) Ajouter phase 8 dans `boot.py` pour init veRL trainer, (2) Connecter `StepRewardVector` à veRL config, (3) Ajouter CLI command `sage train --epochs N`.
- **Effort estimé** : 5-7 jours-dev (Python + config)
- **Papier/Référence** : GiGPO (arXiv 2505.10978) Section 4 — training loop

#### Amélioration 4 : OOD Detection pour kNN Router

- **Problème identifié** : `RustKnnRouter` a "OOD rejection via threshold on nearest distance" mais le threshold n'est pas calibré dynamiquement.
- **Impact si non résolu** : Routing erroné sur tâches hors-distribution → mauvaise topologie → échec exécution.
- **Solution proposée** : (1) Ajouter calibration du threshold via validation set, (2) Fallback à SystemRouter (structural features) si OOD détecté, (3) Log OOD rate pour monitoring.
- **Effort estimé** : 2-3 jours-dev (Rust + validation)
- **Papier/Référence** : kNN Routing (arXiv 2505.12601) — OOD rejection section

#### Amélioration 5 : Isolation Sandbox Renforcée

- **Problème identifié** : `sandbox/subprocess.rs` — "No seccomp/namespace/cgroup isolation (Audit3 F-02)". `tool_executor.rs` — "Subprocess has no OS-level sandboxing".
- **Impact si non résolu** : Risque sécurité si code malveillant exécuté. Inacceptable pour production.
- **Solution proposée** : (1) Activer eBPF module (actuellement disabled : "solana_rbpf CI issues"), (2) OU utiliser Docker-in-Docker avec resource limits, (3) OU firecracker microVM pour isolation forte.
- **Effort estimé** : 10-15 jours-dev (sécurité + CI)
- **Papier/Référence** : FoVer (Formal Verification of LLM outputs) — sandbox requirements

---

## 6. Évaluation globale

| Dimension | Score /10 | Justification (1 ligne) |
|-----------|-----------|-------------------------|
| Originalité architecturale | 8 | S-MMU 4 vues + MAP-Elites + GiGPO = combinaison inédite |
| Maturité du code | 6 | 404 tests passing, mais veRL training pending, entity graph non persisté |
| Adaptativité topologique (réelle) | 7 | Bandit + MAP-Elites fonctionnels, policy learning pas encore wire |
| Viabilité mémoire/apprentissage | 6 | S-MMU production-ready, mais entity graph in-memory only, dataset 5x trop faible |
| Pertinence du choix de training | 6 | Qwen3.5-9B raisonnable, mais 1965 entries insuffisant pour GiGPO stable |
| Potentiel si améliorations appliquées | 8 | Avec dataset 10K + veRL wire + persistance = système production-ready |

---

## Synthèse sans fard

**Ce que YGN-SAGE fait bien** :
- Architecture Rust/Python bien séparée (perf vs orchestration)
- S-MMU multi-view = innovation mémoire réelle
- MAP-Elites + Bandit = adaptation fonctionnelle (pas juste déclarative)
- Verification formelle (OxiZ SMT, LTL) = rigueur rare

**Ce qui manque** :
- Dataset d'entraînement 5x trop faible (1965 vs 10K requis)
- veRL GiGPO "pending" = valeur principale non délivrée
- Entity graph non persisté = risque overflow RAM
- Sandbox subprocess = isolation insuffisante (Audit3 F-02)

**Verdict final** : **6.5/10** — Architecture solide, implémentation partiellement mature. Le système est utilisable en mode bandit/MAP-Elites, mais la promesse "learned topology policy" n'est pas encore tenue. Priorité : wire veRL + expand dataset + persist entity graph.