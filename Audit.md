## Périmètre audité

Branche auditée : `VeRLGIGPO`. HEAD réellement audité : `05d179613a3fc7fe9c5fbf09053b1c1a52be6dfa` (`05d1796`, commit “fix: Phase A V5 — reward shaping + response length + lr fix”). Le document `AI-ARCHITECTURE.md` de cette même branche annonce pourtant `Commit HEAD : 9832b7b`, donc la documentation interne est déjà en dérive par rapport au code effectivement audité. [EVIDENCE: AI-ARCHITECTURE.md:1 ; header] [STATUT: Observé] [REACHABILITY: Inconnue] [VALIDATION: Aucune]. ([GitHub][1])

## Points d’entrée réels

`Serving runtime` : le point d’entrée réel est `sage.boot.boot_agent_system()` puis `AgentSystem.run()`, avec un appel au pipeline uniquement si celui-ci est câblé et que le provider n’est pas `mock`, sinon bypass/mock/legacy fallback. [EVIDENCE: sage-python/src/sage/boot.py:159-167,1029-1047 ; AgentSystem.run, boot_agent_system] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Pipeline]

`Training offline` : les seuls points d’entrée explicites sont `sage-python/scripts/verl/train_topology.sh` et `sage-python/scripts/verl/train_topology_v5.sh`, qui appellent `python3 -m verl.trainer.main_ppo` et dépendent d’un environnement externe `verl-agent` / `verl-071`. [EVIDENCE: sage-python/scripts/verl/train_topology.sh:38-44,214-417 ; train_topology + sage-python/scripts/verl/train_topology_v5.sh:52-155 ; train_topology_v5] [STATUT: Observé] [REACHABILITY: Training] [VALIDATION: Pipeline]

`Tests / benchmarks / CI` : le repo expose des tests Python/Rust et un CLI `python -m sage.bench`, mais la CI ne déclenche que sur `main`/`master`, et le smoke test réel API ne tourne que sur push `master`. Cette branche n’a donc pas de garde-fou CI direct de type “push branch = preuve runtime”. [EVIDENCE: .github/workflows/ci.yml:1-1 ; on/push/pull_request/master-only smoke + sage-python/src/sage/bench/**main**.py:1-6 ; CLI benchmark] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Benchmark]. ([GitHub][2])

## Composants critiques inspectés

Topologie : `topology_engine.rs`, `llm_caller.py`, `pipeline.py`, `topology_controller.py`, `topology/runner.py`. Routing / assignment : `system_router.rs`, `adaptive_router.py`, `model_assigner.rs`, `model_registry.rs`. Mémoire : `memory/working.py`, `verl/training_memory.py`, `verl/topology_env.py`. Training / reward / datasets : `verl/reward.py`, `scripts/verl/train_topology*.sh`, `verl/env_register.py`. Pipeline principal : `boot.py`, `pipeline.py`. Tests / CI / benchmarks : `tests/test_boot_topology.py`, `tests/test_engine_persistence.py`, `tests/test_env_package.py`, `.github/workflows/ci.yml`, `sage.bench`. [EVIDENCE: multiples chemins ci-dessus ; symboles divers] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Aucune]

## Problème dominant

Le problème dominant n’est pas une subtilité algorithmique ; c’est un écart systémique entre les claims et les chemins réellement reachables. Le README vend un moteur qui “apprend” la topologie, une mémoire 4-tiers durable, une quality pipeline “zero heuristics”, un Path 6 Nemotron GiGPO V2, un reward d’exécution RLVR et `pipeline.py` comme “ONLY execution path”, alors que le code reachable sur cette branche livre surtout une orchestration heuristique avec fallbacks, abstentions, mocks, feature gates et training structurel par défaut. [EVIDENCE: README.md:6-15 ; claims principaux + sage-python/src/sage/boot.py:159-167,1029-1095 ; pipeline optionnel/fallback + sage-python/src/sage/quality_estimator.py:31-56 ; backend absent + sage-python/src/sage/verl/reward.py:259-320 ; structural fallback] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Aucune]. ([GitHub][3])

## 1. Génération topologique : adaptativité réelle ou déclarative ?

### Serving runtime

Claim -> Evidence -> Reachability -> Validation -> Verdict : “`pipeline.py` est l’ONLY execution path” -> le README l’écrit explicitement, mais `AgentSystem.run()` n’appelle `self.pipeline.run()` que si le pipeline est câblé et que le provider n’est pas `mock`, puis retombe sur le legacy path sur exception ; `boot_agent_system()` n’instancie le pipeline que si `model_assigner` et `provider_pool` existent -> Runtime -> Pipeline sans test d’exclusivité -> Verdict : **faux** ; implémentation **Partiellement câblé**. [EVIDENCE: README.md:15 ; project structure + sage-python/src/sage/boot.py:159-167,1029-1047 ; AgentSystem.run, boot_agent_system] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Pipeline]. ([GitHub][3])

Claim -> Evidence -> Reachability -> Validation -> Verdict : “moteur topologique dynamique 6-path réellement adaptatif” -> la génération par requête existe bien, mais l’appel principal passe `task_embedding=None`, la voie LLM est explicitement “skipped in pure Rust” et déportée côté Python, et le reroute runtime n’est pas une mutation in-place du graphe mais une sortie sentinelle `__REROUTE__` qui force `pipeline.py` à relancer Stage 2 + Stage 3 sur un nouveau graphe -> Runtime -> Pipeline, pas de benchmark lié au head -> Verdict : **survendu** ; adaptativité réelle **Semi-dynamique**, pas “Dynamique réactif” et encore moins “Adaptatif avec apprentissage”. [EVIDENCE: sage-core/src/topology/topology_engine.rs:1-15,167-293 ; DynamicTopologyEngine::generate + sage-python/src/sage/boot.py:239-279 ; AgentSystem.run + sage-python/src/sage/pipeline.py:232-245,668-704 ; _stage_select_topology, Stage 4 reroute + sage-python/src/sage/topology/runner.py:245-324 ; TopologyRunner.run] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Pipeline]. ([GitHub][3])

La génération topologique n’est pas design-time. Elle est surtout **par requête** sur le main path. Le démarrage ne fait qu’un bootstrap synthétique S-MMU avec `bootstrap_s1/s2/s3` et des métriques neutres `0.5 / 0 / 0`. Le cross-run n’existe que si la persistance `save_state/load_state` est exposée par la feature Rust `cognitive`, ce qui n’est ni garanti ni imposé par le boot path. [EVIDENCE: sage-python/src/sage/boot.py:239-247,712-738,740-760 ; generation, persistence, bootstrap + sage-core/src/topology/pyo3_wrappers.rs:202-223 ; save_state/load_state cfg(feature="cognitive")] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Test]

Claim -> Evidence -> Reachability -> Validation -> Verdict : “Path 6 learned policy V2 Nemotron-Orchestrator-8B GiGPO auto-downloaded from `sage-topology-policy-v2`” -> le README le dit, mais le code reachable charge `HF_POLICY_REPO = "yannabadie/sage-topology-policy"`, base `microsoft/Phi-4-mini-instruct`, exige CUDA, et `pipeline.py` ne reconstruit qu’un graphe de nœuds sans arêtes ni `model_id` -> Runtime opt-in -> Aucune -> Verdict : **matériellement faux** pour cette branche ; implémentation **Partiellement câblé**. [EVIDENCE: README.md:11-12 ; Path 6 claim + sage-python/src/sage/topology/llm_caller.py:263-337,373-384 ; generate_topology_from_policy + sage-python/src/sage/pipeline.py:251-275 ; _stage_select_topology] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Aucune]. ([GitHub][3])

Feedback loop réel : `output node` (`TopologyRunner._execute_node`) -> `quality / consistency / importance / emergent_subtask` (`TopologyController._compute_quality`, `compute_consistency_score`, `compute_importance_score`) -> `decision.action` (`evaluate_and_decide`) -> `upgrade_model` / `prune_node` / `spawn_subagent` / `__REROUTE__` (`TopologyRunner.run`) -> soit réexécution locale du nœud, soit skip, soit sous-agent, soit régénération **externe** du graphe par `pipeline.py`. La boucle est cassée pour un claim “appris” parce que la qualité bascule à `0.5` si l’estimateur abstient et la consistance vaut `1.0` si `sage.consistency` manque. [EVIDENCE: sage-python/src/sage/topology_controller.py:74-233 ; evaluate_and_decide, _compute_quality + sage-python/src/sage/topology/runner.py:245-324 ; TopologyRunner.run + sage-python/src/sage/pipeline.py:668-704 ; reroute handling] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Pipeline]

Granularité réelle : changement de modèle par nœud **oui** ; prune/skip **oui** ; spawn de sous-agent **oui** ; reroute **externe au DAG actif** **oui** ; ajout/suppression de nœuds du graphe actif en serving **non démontré** ; mutation de prompts au runtime main path **non démontrée**. [EVIDENCE: sage-python/src/sage/topology/runner.py:275-329 ; upgrade/prune/spawn/reroute + sage-python/src/sage/pipeline.py:668-704 ; external reroute + sage-core/src/topology/topology_engine.rs:415-456 ; mutation/MCTS seulement à la génération] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Pipeline]

Claim -> Evidence -> Reachability -> Validation -> Verdict : “politique apprise de routing/model assignment” -> le router principal décide S1/S2/S3 par mots-clés/formal keywords et seuils de complexité, puis choisit un modèle sous contrainte de budget ; `ModelAssigner` score les modèles avec des poids fixes `0.4 / 0.4 / 0.2` ; `AdaptiveRouter` annonce explicitement que le stage 3 online learning est “not yet implemented” -> Runtime -> Pipeline, pas de preuve de gain branchée au head -> Verdict : **surtout heuristique**, pas politique apprise sur le main path. [EVIDENCE: sage-core/src/routing/system_router.rs:22-31,364-577 ; FORMAL_KEYWORDS, route_integrated, decide_system + sage-core/src/routing/model_assigner.rs:10-12,28-120 ; WEIGHT_* , assign_models_inner + sage-python/src/sage/strategy/adaptive_router.py:1-8,234-246 ; stage 3 reserved, record_feedback] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Pipeline]

Claim -> Evidence -> Reachability -> Validation -> Verdict : “Z3 Quality Pipeline remplace l’heuristique, zero false signals” -> le README l’annonce done, mais `QualityEstimator` dit explicitement que le modèle ONNX n’est pas livré et peut abstain, tandis que `TopologyController` remplace cette abstention par `0.5` et `compute_consistency_score()` retourne `1.0` si le module manque -> Runtime -> Aucune -> Verdict : **faux sur le main path** ; implémentation **Partiellement câblé**. [EVIDENCE: README.md:13 ; QualityLabeler zero heuristics + sage-python/src/sage/quality_estimator.py:31-56,58-84 ; QualityEstimator + sage-python/src/sage/topology_controller.py:183-214 ; defaults 0.5/1.0 + sage-core/src/lib.rs:80-89 ; feature-gated QualityLabeler] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Aucune]. ([GitHub][3])

### Training offline

Le training offline simule lui aussi de la “reconfiguration”, mais l’environnement `SageTopologyEnv` tombe en mode structural par défaut quand `SAGE_VERL_EXEC=0`, quand le provider manque, ou quand un nœud échoue ; le checkpoint “upgrade/reroute” existe, mais la partie “exécution réelle” n’est donc pas le chemin dominant du scaffold. [EVIDENCE: sage-python/src/sage/verl/topology_env.py:459-524,530-550 ; _execute_single_node, _structural_stub, _estimate_quality] [STATUT: Observé] [REACHABILITY: Training] [VALIDATION: Test]

### Expérimental / dead code

`ShadowRouter` est explicitement désactivé par défaut à cause d’une divergence annoncée de `49.6%`. Ce n’est pas une capacité validée ; c’est un chemin de trace désactivé. [EVIDENCE: sage-python/src/sage/boot.py:622-640 ; ShadowRouter init/logging] [STATUT: Observé] [REACHABILITY: Dead code] [VALIDATION: Aucune]

### Scénarios obligatoires

`Nominal` : le pipeline classe heuristiquement, décompose en single-node si `TaskPlanner` manque, génère une topologie via `engine.generate(..., None, ...)`, assigne des modèles par score fixe, puis exécute via `TopologyRunner`. [EVIDENCE: sage-python/src/sage/pipeline.py:182-245,393-445,645-667 ; classify, decompose, select, assign, execute] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Pipeline]

`Dégradé` : si le provider d’un nœud tombe, `TopologyRunner` retente avec le provider par défaut ; si la qualité est inconnue, le contrôleur décide quand même avec `0.5` ; si un reroute arrive, le pipeline **reconstruit** un nouveau graphe au lieu de muter l’ancien. [EVIDENCE: sage-python/src/sage/topology/runner.py:145-189,275-324 ; provider fallback, reroute sentinel + sage-python/src/sage/topology_controller.py:193-214 ; default quality/consistency + sage-python/src/sage/pipeline.py:668-704 ; external reroute] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Pipeline]

`Croissance` : quand le coût topologique dépasse le budget, `pipeline.py` dégrade vers single-node ; `SystemRouter` et `ModelAssigner` calculent les coûts sur des hypothèses fixes de tokens ; `ModelRegistry` ne persiste pas sa télémétrie. Le système ne montre donc pas une adaptation stable à la croissance, il montre surtout des heuristiques bornées. [EVIDENCE: sage-python/src/sage/pipeline.py:293-361 ; budget degradation, cost estimate + sage-core/src/routing/system_router.rs:542-577 ; token assumptions + sage-core/src/routing/model_registry.rs:44-47,161-175 ; in-memory telemetry] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Aucune]

Comparaison mécanisme contre mécanisme : LangGraph et CrewAI documentent des workflows/flows stateful avec persistance et contrôle explicite ; AutoGen documente surtout l’orchestration par conversation multi-agent ; MetaGPT documente des rôles et SOPs. Par rapport à eux, YGN-SAGE tente sur le papier quelque chose de plus ambitieux sur l’adaptation topologique, mais sur ce head il reconfigure moins proprement que LangGraph/CrewAI à l’exécution, il apprend moins que son README ne le prétend, et il montre moins de preuve expérimentale branchée que les frameworks d’orchestration qu’il veut dépasser. [EVIDENCE: langgraph docs ; memory/control-flow | CrewAI docs ; flows/state persistence | AutoGen docs ; multi-agent conversation | MetaGPT docs ; SOP/roles] [STATUT: Observé] [REACHABILITY: Inconnue] [VALIDATION: Aucune]. ([LangChain][4])

**Verdict axe 1 : `[Semi-dynamique]`.** [EVIDENCE: synthèse des preuves ci-dessus ; topology/runtime] [STATUT: Inféré] [REACHABILITY: Runtime] [VALIDATION: Pipeline]

## 2. Mémoire et apprentissage continu : viable ou vaporware ?

### Serving runtime

Claim -> Evidence -> Reachability -> Validation -> Verdict : “4-tier hierarchical memory” -> le README vend une mémoire durable 4 tiers, mais sans extension Rust `WorkingMemory` bascule vers un mock Python dont `compact_to_arrow*()` renvoie `"0"`, `retrieve_relevant_chunks()` renvoie `[]` et `get_latest_arrow_chunk()` renvoie `None` -> Runtime -> Aucune -> Verdict : **faux sur le main path non compilé** ; implémentation **Squelette**. [EVIDENCE: README.md:10 ; memory claim + sage-python/src/sage/memory/working.py:47-120 ; _PyWorkingMemory + sage-python/src/sage/boot.py:795-802 ; degradation warning] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Aucune]. ([GitHub][3])

La mémoire dédiée à la génération topologique existe seulement **partiellement** au sens strict : write path runtime via `pipeline._stage_learn()` et `AgentSystem._record_topology_outcome()`, retrieve path via `TopologyEngine::try_smmu_hit()`, usage théorique dans `generate()`. Mais le main path sabote la sémantique : `boot.py` appelle `generate(task, None, ...)`, enregistre souvent `record_outcome(..., None, ...)`, et `try_smmu_hit()` commente que la vraie retrieval sémantique n’est pas câblée dans `generate()` et retombe sur un anchor simplifié. Verdict : mémoire topologique **Partiellement câblé**, pas mémoire adaptative robuste. [EVIDENCE: sage-python/src/sage/pipeline.py:805-833 ; _stage_learn + sage-python/src/sage/boot.py:242-245,442-450 ; generate with None, record_outcome None + sage-core/src/topology/topology_engine.rs:295-323 ; try_smmu_hit] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Pipeline]

Claim -> Evidence -> Reachability -> Validation -> Verdict : “continuous learning” -> `AdaptiveRouter` déclare explicitement que le stage 3 online learning n’est pas implémenté, la télémétrie du `ModelRegistry` est un `HashMap` en mémoire, et la persistance `save_state/load_state` du moteur n’existe que sous feature `cognitive` ; le système apprend donc au mieux des posteriors/bandits en RAM ou en persistance optionnelle, pas une politique durable branchée partout -> Runtime/Training -> tests de persistence, pas de validation de gain -> Verdict : **Squelette présent**. [EVIDENCE: sage-python/src/sage/strategy/adaptive_router.py:1-8 ; reserved stage 3 + sage-core/src/routing/model_registry.rs:44-47,161-175,226-247 ; in-memory telemetry + sage-core/src/topology/pyo3_wrappers.rs:202-223 ; cfg(feature="cognitive") + sage-python/tests/test_engine_persistence.py:29-107 ; round-trip only] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Test]

### Training offline

Le seul stockage réellement persistant **et relu automatiquement** est la `TrainingMemory` SQLite de l’environnement veRL : `store_episode()` écrit, `query_similar()` relit, `format_context()` réinjecte dans `reset()`. Mais ce circuit n’influence que le training offline, pas la décision topologique du serving runtime. [EVIDENCE: sage-python/src/sage/verl/training_memory.py:54-140 ; TrainingMemory + sage-python/src/sage/verl/topology_env.py:133-175,712-733 ; reset, _finalize_episode] [STATUT: Observé] [REACHABILITY: Training] [VALIDATION: Test]

### Expérimental / dead code

La persistance moteur existe en tests, mais elle est feature-gated et testée surtout en round-trip de fichiers SQLite, pas en amélioration de qualité, pas en mémoire longue utile à la décision de serving. [EVIDENCE: sage-python/tests/test_engine_persistence.py:29-107 ; round-trip persistence + sage-core/src/topology/pyo3_wrappers.rs:202-223 ; cognitive feature gate] [STATUT: Observé] [REACHABILITY: Dead code] [VALIDATION: Test]

### Scalabilité

Scalabilité mémoire/rétention : aucune politique d’éviction n’est visible pour `TrainingMemory`, le query set est simplement borné aux 500 meilleurs épisodes avant cosine, la télémétrie modèle ne garde que 100 latences récentes par modèle, et rien ne montre une compaction/oubli cross-session de la mémoire utile en serving. [EVIDENCE: sage-python/src/sage/verl/training_memory.py:98-128 ; query_similar limit 500 + sage-core/src/routing/model_registry.rs:15-18,167-172 ; bounded latencies + sage-python/src/sage/memory/working.py:103-120 ; mock retrieval] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Aucune]

### Scénarios obligatoires

`Nominal` : si `QualityEstimator` renvoie un float, Stage 5 écrit bien bandit + archive et peut calculer un embedding réel ; il y a donc un write path exploitable. [EVIDENCE: sage-python/src/sage/pipeline.py:771-833 ; _stage_learn] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Pipeline]

`Dégradé` : si l’estimateur abstient ou manque, `pipeline.py` n’écrit plus rien au bandit, `boot.py` force `0.5` neutre, et le contrôleur décide quand même avec `0.5/1.0` par défaut ; la “mémoire” apprend donc un signal fragile ou rien du tout. [EVIDENCE: sage-python/src/sage/pipeline.py:776-803 ; bandit abstention + sage-python/src/sage/boot.py:429-434 ; neutral 0.5 + sage-python/src/sage/topology_controller.py:193-214 ; defaults] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Pipeline]

`Croissance` : sur session longue/cross-run, la persistance n’est disponible que si feature `cognitive` et tests round-trip simples ; aucune preuve d’un apprentissage durable utile à la décision topologique en production. [EVIDENCE: sage-core/src/topology/pyo3_wrappers.rs:202-223 ; save/load cfg(feature) + sage-python/tests/test_engine_persistence.py:40-107 ; round-trip only] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Test]

**Verdict axe 2 : `[Squelette présent]`.** [EVIDENCE: synthèse des preuves ci-dessus ; memory/learning] [STATUT: Inféré] [REACHABILITY: Runtime] [VALIDATION: Test]

## 3. Choix du modèle à entraîner

Le projet entraîne **très peu** de choses au sens “modèle livré et branché” ; il orchestre surtout des modèles existants. Le modèle d’exécution d’agent n’est pas entraîné ici : il est assigné par heuristique/budget dans `ModelAssigner` puis exécuté via providers externes avec fallback provider. [EVIDENCE: sage-core/src/routing/model_assigner.rs:28-120 ; assign_models_inner + sage-python/src/sage/topology/runner.py:140-189 ; provider resolution/fallback] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Pipeline]

Le seul candidat réel à entraîner est le **modèle de synthèse topologique** du Path 6. Mais son contrat de sortie n’est pas aligné avec le serving : le loader runtime pointe encore sur l’adaptateur legacy `yannabadie/sage-topology-policy`, charge une base `Phi-4-mini-instruct`, exige CUDA, et Stage 2 n’utilise que des nœuds sans arêtes ni `model_id`. [EVIDENCE: sage-python/src/sage/topology/llm_caller.py:263-337,373-384 ; generate_topology_from_policy + sage-python/src/sage/pipeline.py:251-271 ; learned_policy topology] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Aucune]. ([GitHub][5])

Le “modèle” de scoring/routing appris n’existe pas vraiment sur le main path : `AdaptiveRouter` réserve le stage 3 à plus tard, `SystemRouter` est à seuils heuristiques, et le modèle ONNX de qualité n’est pas livré. [EVIDENCE: sage-python/src/sage/strategy/adaptive_router.py:1-8 ; stage 3 reserved + sage-core/src/routing/system_router.rs:509-577 ; decide_system + sage-python/src/sage/quality_estimator.py:31-56 ; ONNX not shipped] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Aucune]. ([GitHub][6])

Le judge/reward model n’est pas un judge stable ; c’est un mélange structural + branche d’exécution conditionnelle. Ce n’est pas un bon candidat de fine-tuning tant que les fallbacks structurels et les abstentions ne sont pas durcis. [EVIDENCE: sage-python/src/sage/verl/reward.py:259-320 ; compute_score + sage-python/src/sage/quality_estimator.py:58-84 ; abstention] [STATUT: Observé] [REACHABILITY: Training] [VALIDATION: Aucune]. ([GitHub][7])

Recommandation : **aucun fine-tuning supplémentaire n’a de sens à court terme tant que le contrat runtime n’est pas refermé**. Le meilleur candidat réel à fine-tuner, une fois le contrat corrigé, est le **policy model de topologie** ; il doit sortir un graphe complet (`nodes + edges + tiers + checkpoints`) directement consommable par Stage 2. Le gain attendu à court terme vient davantage de `schema strict + prompting + retrieval + routing + évaluation` que d’un RL/fine-tuning de plus. [EVIDENCE: sage-python/src/sage/topology/llm_caller.py:263-384 ; policy path + sage-python/src/sage/pipeline.py:251-271 ; nodes-only + sage-python/src/sage/strategy/adaptive_router.py:1-8 ; no online learning + sage-python/src/sage/quality_estimator.py:31-56 ; missing ONNX] [STATUT: Inféré] [REACHABILITY: Runtime] [VALIDATION: Aucune]

## 4. Type d’entraînement : est-ce le bon ?

Claim -> Evidence -> Reachability -> Validation -> Verdict : “branche GiGPO / veRL prête” -> `train_topology.sh` est bien un scaffold multi-step GiGPO en deux phases avec `algorithm.adv_estimator=gigpo`, mais il dépend d’un overlay externe `/workspace/verl-agent` et d’`agent_system.environments.env_manager`, non vendored ; `env_register.py` monkey-patche cet external manager et renvoie `skipped` si absent -> Training -> tests d’interface, pas run reproductible fourni -> Verdict : **Partiellement câblé**. [EVIDENCE: sage-python/scripts/verl/train_topology.sh:38-44,95-101,214-283,292-417 ; overlay + phases + sage-python/src/sage/verl/env_register.py:66-108 ; registration patch] [STATUT: Observé] [REACHABILITY: Training] [VALIDATION: Pipeline]. ([GitHub][8])

Claim -> Evidence -> Reachability -> Validation -> Verdict : “V2 Nemotron-Orchestrator-8B GiGPO done/opt-in” -> le script V5 au head dit explicitement `Algorithm: GRPO via verl 0.7.1 (single-turn, GiGPO-equivalent)` et lance `algorithm.adv_estimator=grpo`; ce n’est pas du GiGPO multi-step -> Training -> aucune preuve d’intégration serving sur head -> Verdict : **claim faux** pour le script actif V5 ; **Doc-only / Partiellement câblé** selon qu’on parle du README ou du scaffold. [EVIDENCE: sage-python/scripts/verl/train_topology_v5.sh:5-7,132-145 ; header + trainer args + README.md:11-13 ; GiGPO/Nemotron claim] [STATUT: Observé] [REACHABILITY: Training] [VALIDATION: Aucune]. ([GitHub][9])

Claim -> Evidence -> Reachability -> Validation -> Verdict : “execution-based reward, not format-only” -> `reward.py` renvoie purement le score structural si `SAGE_VERL_EXEC!=1`, YAML invalide, provider absent ou exception ; `topology_env.py` exécute par défaut un stub structural quand `SAGE_VERL_EXEC=0`, quand le provider est absent, ou quand un nœud plante -> Training -> tests d’interface, pas de run head lié à un checkpoint -> Verdict : l’exécution-récompense existe comme **branche conditionnelle**, mais le training réel reste structurel par défaut et en fallback ; **Partiellement câblé**. [EVIDENCE: sage-python/src/sage/verl/reward.py:259-320 ; compute_score + sage-python/src/sage/verl/topology_env.py:459-524,530-550 ; _execute_single_node, _structural_stub, _estimate_quality] [STATUT: Observé] [REACHABILITY: Training] [VALIDATION: Test]. ([GitHub][7])

Je n’ai pas trouvé, dans les entrypoints audités, de pipeline SFT complet livré de bout en bout ; `train_topology_v5.sh` suppose un modèle déjà “SFT-merged” en `/workspace/sft_merged_model`, et `llm_caller.py` suppose un adaptateur HF déjà entraîné. Verdict : la partie SFT fournie par le repo est au mieux une dépendance amont, pas un pipeline livré ici. [EVIDENCE: sage-python/scripts/verl/train_topology_v5.sh:65-71 ; SFT-merged base + sage-python/src/sage/topology/llm_caller.py:263-337 ; adapter loader] [STATUT: Observé] [REACHABILITY: Training] [VALIDATION: Aucune]. ([GitHub][9])

Claim -> Evidence -> Reachability -> Validation -> Verdict : “+5.5pp / 89.6% / 92% / 86%” -> le README publie ces chiffres et le CLI benchmark sait sauvegarder un JSON + truth pack, mais je n’ai pas trouvé dans les artefacts reachables audités le report JSON, la seed, la config provider, le checkpoint/adaptateur exact ou la CI qui reproduit ces nombres sur ce head -> Inconnue -> harness oui, artefacts de validation manquants -> Verdict : **Doc-only** pour les chiffres, **Inconcluant** pour la reproductibilité. [EVIDENCE: README.md:13-15 ; benchmark numbers + sage-python/src/sage/bench/**main**.py:1-6 ; report saver + .github/workflows/ci.yml:1-1,117-135 ; pas de preuve e2e head-bound] [STATUT: Inconcluant] [REACHABILITY: Inconnue] [VALIDATION: Benchmark]. ([GitHub][3])

Artefacts manquants bloquants : report benchmark `JSON/JSONL` correspondant aux scores du README, commande exacte + seeds + provider/model config, checkpoint/adaptateur ayant produit Path 6 V2, et dépôt/vendor du `verl-agent` requis. Sans eux, je peux auditer le scaffold, pas valider la performance annoncée. Ces artefacts changeraient le verdict uniquement s’ils reliaient le head `05d1796` à un run reproductible. [EVIDENCE: sage-python/src/sage/bench/**main**.py:1-6 ; report paths + sage-python/scripts/verl/train_topology.sh:38-44,95-101 ; external deps + README.md:13-15 ; claimed scores] [STATUT: Observé] [REACHABILITY: Inconnue] [VALIDATION: Aucune]. ([GitHub][10])

L’estimation coût/bénéfice sérieuse serait trop spéculative : le repo ne fournit que des commentaires de script (`~$50-80 API + ~$20-40 H100` pour `train_topology.sh`, `~29h H100 NVL` pour V5), pas un run validé lié au head. Je traite donc ces montants comme **doc-only**, pas comme base d’arbitrage. [EVIDENCE: sage-python/scripts/verl/train_topology.sh:20-26 ; cost comments + sage-python/scripts/verl/train_topology_v5.sh:40-42 ; time estimate] [STATUT: Observé] [REACHABILITY: Training] [VALIDATION: Aucune]

Mon jugement : lancer davantage de RL avant de durcir le contrat de sortie du policy model et d’empêcher les fallbacks structurels est un mauvais ordre des opérations. Le frein actuel n’est pas “pas assez de RL”, c’est “reward et serving pas assez fermés”. [EVIDENCE: sage-python/src/sage/verl/reward.py:305-320 ; structural fallback + sage-python/src/sage/pipeline.py:251-271 ; Path 6 nodes only + sage-python/src/sage/quality_estimator.py:31-56 ; abstention/no shipped ONNX] [STATUT: Inféré] [REACHABILITY: Training] [VALIDATION: Aucune]

## 5. Améliorations — classées par impact/effort

### Amélioration 1 : Couvrir le vrai chemin de recherche en CI

**Problème identifié** : la CI ne protège pas directement cette branche et le seul smoke réel API est `master`-only ; les claims de recherche peuvent donc dériver sans garde-fou branch-bound. [EVIDENCE: .github/workflows/ci.yml:1-1,117-135 ; branch filters, integration-smoke] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Aucune]

**Impact si non résolu** : chaque claim “done” peut rester vrai en doc mais faux sur la branche active.

**Solution proposée** : ajouter une matrice CI sur `VeRLGIGPO` ou `**`, avec un smoke test `boot -> pipeline -> reroute -> provider failure -> fallback`, et un job qui force `smt,tool-executor,cognitive` pour empêcher la validation sur chemin dégradé uniquement.

**Effort estimé** : 1 à 2 jours-dev, complexité faible à moyenne.

**Papier/Référence** : GitHub Actions branch filters + smoke-test de chemin critique.

### Amélioration 2 : Faire échouer explicitement le reward “execution mode” quand l’exécution réelle n’a pas eu lieu

**Problème identifié** : en `SAGE_VERL_EXEC=1`, `reward.py` et `topology_env.py` retombent silencieusement sur du structurel si le provider manque, si le YAML est invalide, ou si l’exécution plante. [EVIDENCE: sage-python/src/sage/verl/reward.py:305-320 ; compute_score + sage-python/src/sage/verl/topology_env.py:459-524 ; structural stub paths] [STATUT: Observé] [REACHABILITY: Training] [VALIDATION: Test]

**Impact si non résolu** : le RL “apprend” surtout à survivre aux fallbacks, pas à optimiser la performance d’exécution.

**Solution proposée** : en mode exec, retourner une pénalité dure et loggée si provider absent / fallback structural / timeout ; séparer strictement `reward_structural` et `reward_execution` dans les artefacts d’entraînement.

**Effort estimé** : 2 à 4 jours-dev, complexité moyenne.

**Papier/Référence** : AgentConductor / Graph-GRPO ; la reward n’a de sens que si elle mesure bien le mécanisme vendu.

### Amélioration 3 : Câbler de vrais embeddings topologiques partout et arrêter la pollution du bootstrap neutre

**Problème identifié** : le main path appelle `generate(..., None, ...)`, enregistre souvent `record_outcome(..., None, ...)`, et le bootstrap injecte des épisodes synthétiques notés `0.5 / 0 / 0`. [EVIDENCE: sage-python/src/sage/boot.py:242-245,442-450,748-760 ; generate with None, record_outcome None, bootstrap + sage-core/src/topology/topology_engine.rs:295-323 ; simplified retrieval] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Pipeline]

**Impact si non résolu** : la mémoire topologique est contaminée par des signaux pauvres et la retrieval sémantique reste largement fictive.

**Solution proposée** : calculer l’embedding avant le premier `generate()`, interdire `record_outcome` sans embedding sur les runs réels, marquer les entrées bootstrap comme `synthetic=true` et les exclure de la retrieval par défaut.

**Effort estimé** : 3 à 5 jours-dev, complexité moyenne.

**Papier/Référence** : MAP-Elites / quality-diversity ; la qualité du descripteur conditionne tout le reste.

### Amélioration 4 : Rendre la mémoire/runtime capability explicite et persister la télémétrie utile

**Problème identifié** : la mémoire de travail peut basculer vers un mock qui renvoie des valeurs factices, et la télémétrie de `ModelRegistry` vit seulement en RAM. [EVIDENCE: sage-python/src/sage/memory/working.py:95-120 ; dummy returns + sage-core/src/routing/model_registry.rs:44-47,161-175 ; in-memory telemetry] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Aucune]

**Impact si non résolu** : les claims de mémoire durable et d’apprentissage continu restent faux au premier redémarrage ou sur une installation partielle.

**Solution proposée** : capability handshake au boot (`memory_ready`, `quality_ready`, `formal_ready`), refus explicite des claims “memory/evolution enabled” si les features manquent, et persistance SQLite de la télémétrie routing/quality au même titre que le moteur topologique.

**Effort estimé** : 3 à 6 jours-dev, complexité moyenne.

**Papier/Référence** : architectures stateful type LangGraph ; la persistance doit être une propriété runtime, pas une hypothèse.

### Amélioration 5 : Aligner Path 6 avec l’artefact réellement servi

**Problème identifié** : le README promet `sage-topology-policy-v2` / Nemotron GiGPO, mais le runtime charge encore l’adaptateur legacy `sage-topology-policy` sur base `Phi-4-mini-instruct`, et Stage 2 n’utilise qu’un sous-ensemble du graphe. [EVIDENCE: README.md:11-12 ; Path 6 claim + sage-python/src/sage/topology/llm_caller.py:263-337 ; HF_POLICY_REPO/base model + sage-python/src/sage/pipeline.py:251-271 ; nodes only] [STATUT: Observé] [REACHABILITY: Runtime] [VALIDATION: Aucune]

**Impact si non résolu** : tout gain de training sur le policy model reste invérifiable et mal mesuré côté serving.

**Solution proposée** : versionner explicitement l’artefact servi (`manifest.json` avec commit/checkpoint/base model), faire sortir `nodes + edges + model tiers + adaptation metadata`, puis reconstruire le vrai `TopologyGraph` via `parse_and_build_topology()` avant toute benchmark gate.

**Effort estimé** : 4 à 7 jours-dev, complexité moyenne.

**Papier/Référence** : structured generation / constrained decoding ; un policy model n’est utile que si son output contract est stable.

## 6. Évaluation globale

| Dimension                             | Score /10 | Justification (1 ligne)                                                                                                    |
| ------------------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------- |
| Originalité architecturale            | 6         | Split Rust/Python + archive/bandit/topology search intéressant, mais l’originalité est plus forte que la preuve reachable. |
| Maturité du code                      | 4         | Beaucoup de fallback, de feature gates, de doc drift et de chemins “best effort” non verrouillés.                          |
| Adaptativité topologique (réelle)     | 3         | Génération par requête et reroute externe existent, mais pas de mutation runtime propre ni de politique apprise prouvée.   |
| Viabilité mémoire/apprentissage       | 3         | Write paths partiels, mocks en serving, persistance optionnelle, online learning surtout absent.                           |
tural/execution se mélangent.               |
| Potentiel si améliorations appliquées | 6         | Le squelette peut devenir crédible assez vite si CI, reward, Path 6 et mémoire sont recâblés proprement.                   |

Verdict final : **le repo n’est pas vide, mais ses claims de recherche dépassent d’au moins une à deux couches d’intégration son implémentation réellement reachable**. Le cœur livré sur `VeRLGIGPO` est une orchestration heuristique semi-dynamique avec hooks expérimentaux, pas un moteur topologique appris, durable et validé de bout en bout. [EVIDENCE: synthèse de l’audit ; boot/pipeline/topology/reward/memory/training] [STATUT: Inféré] [REACHABILITY: Runtime] [VALIDATION: Aucune]

[1]: https://github.com/yannabadie/YGN-SAGE/tree/VeRLGIGPO?utm_source=chatgpt.com "GitHub - yannabadie/YGN-SAGE at VeRLGIGPO · GitHub"
[2]: https://raw.githubusercontent.com/yannabadie/YGN-SAGE/VeRLGIGPO/.github/workflows/ci.yml "https://raw.githubusercontent.com/yannabadie/YGN-SAGE/VeRLGIGPO/.github/workflows/ci.yml"
[3]: https://raw.githubusercontent.com/yannabadie/YGN-SAGE/VeRLGIGPO/README.md "https://raw.githubusercontent.com/yannabadie/YGN-SAGE/VeRLGIGPO/README.md"
[4]: https://www.langchain.com/langgraph "https://www.langchain.com/langgraph"
[5]: https://raw.githubusercontent.com/yannabadie/YGN-SAGE/VeRLGIGPO/sage-python/src/sage/topology/llm_caller.py "https://raw.githubusercontent.com/yannabadie/YGN-SAGE/VeRLGIGPO/sage-python/src/sage/topology/llm_caller.py"
[6]: https://raw.githubusercontent.com/yannabadie/YGN-SAGE/VeRLGIGPO/sage-python/src/sage/quality_estimator.py "https://raw.githubusercontent.com/yannabadie/YGN-SAGE/VeRLGIGPO/sage-python/src/sage/quality_estimator.py"
[7]: https://raw.githubusercontent.com/yannabadie/YGN-SAGE/VeRLGIGPO/sage-python/src/sage/verl/reward.py "https://raw.githubusercontent.com/yannabadie/YGN-SAGE/VeRLGIGPO/sage-python/src/sage/verl/reward.py"
[8]: https://raw.githubusercontent.com/yannabadie/YGN-SAGE/VeRLGIGPO/sage-python/scripts/verl/train_topology.sh "https://raw.githubusercontent.com/yannabadie/YGN-SAGE/VeRLGIGPO/sage-python/scripts/verl/train_topology.sh"
[9]: https://raw.githubusercontent.com/yannabadie/YGN-SAGE/VeRLGIGPO/sage-python/scripts/verl/train_topology_v5.sh "https://raw.githubusercontent.com/yannabadie/YGN-SAGE/VeRLGIGPO/sage-python/scripts/verl/train_topology_v5.sh"
[10]: https://github.com/yannabadie/YGN-SAGE/commit/05d179613a3fc7fe9c5fbf09053b1c1a52be6dfa "https://github.com/yannabadie/YGN-SAGE/commit/05d179613a3fc7fe9c5fbf09053b1c1a52be6dfa"
