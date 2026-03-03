# YGN-SAGE : Synthèse de Recherche NotebookLM (MARL & ADK Architecture)

## Introduction
Ce rapport synthétise les recherches croisées effectuées via NotebookLM sur le corpus de référence du projet YGN-SAGE (OpenSage, AlphaEvolve, PSRO, VAD-CFR). L'objectif est d'aligner l'implémentation technique de nos 5 piliers cognitifs avec l'état de l'art de la recherche en agents autonomes et théorie des jeux.

---

## 1. Synergie Évolution / Stratégie : Optimisation dynamique par MAP-Elites

L'intégration de l'évolution (via AlphaEvolve) et de la stratégie algorithmique permet de dépasser les heuristiques manuelles en traitant le code source des algorithmes comme un génome mutable. 

### 1.1 VAD-CFR (Volatility-Adaptive Discounted CFR)
VAD-CFR remplace l'escompte statique par des paramètres réactifs basés sur la volatilité locale de l'apprentissage.

*   **Volatilité (EWMA)** : 
    $M_t = \max_a |r^t(a)|$ (Amplitude max du regret instantané)
    $EWMA_t = 0.1 \cdot M_t + 0.9 \cdot EWMA_{t-1}$
    $v_t = \min(1.0, \frac{EWMA_t}{2.0})$ (Volatilité normalisée)
*   **Escompte Adaptatif** :
    $\alpha_t = \max(0.1, 1.5 - 0.5 \cdot v_t)$
    $\beta_t = \min(\alpha_t, -0.1 - 0.5 \cdot v_t)$
    $D^+_t = \frac{(t+1)^{\alpha_t}}{(t+1)^{\alpha_t} + 1.0}$ (Regret positif)
    $D^-_t = \frac{(t+1)^{\beta_t}}{(t+1)^{\beta_t} + 1.0}$ (Regret négatif)
*   **Hard Warm-Start** :
    La mise à jour de la politique moyenne est suspendue jusqu'à l'itération $T=500$ pour éliminer le bruit initial.
*   **Pondération de la Politique** :
    $\gamma_t = \min(4.0, 2.0 + 1.5 \cdot v_t)$
    $Weight_t = (t+1)^{\gamma_t} \cdot (1 + \frac{v_t}{2.0})^{0.5}$

### 1.2 SHOR-PSRO (Smoothed Hybrid Optimistic Regret PSRO)
SHOR-PSRO utilise un méta-solveur hybride pour accélérer la convergence vers l'équilibre de Nash.

*   **Lissage Softmax (Exploitation)** :
    $\sigma_{Softmax}(a) = \frac{\exp((u(a) - \max u) / \tau)}{\sum \exp((u(a') - \max u) / \tau)}$
*   **Mélange Hybride (Hybrid Blending)** :
    $\sigma_{hybrid} = (1 - \lambda) \cdot \sigma_{ORM} + \lambda \cdot \sigma_{Softmax}$
    $\sigma_{ORM}$ est calculé via Optimistic Regret Matching avec un bonus de diversité.
*   **Recuit Dynamique (Dynamic Annealing)** :
    Pour $p = \min(1.0, \frac{iter}{75})$ :
    - $\lambda = 0.30 - 0.25 \cdot p$ (Mélange)
    - $\tau = 0.50 - 0.49 \cdot p$ (Température Softmax)
    - $div = 0.05 - 0.049 \cdot p$ (Bonus de diversité)

## 2. Mémoire Graphes et RAG : Architecture hiérarchique et synchronisation

La recherche préconise une architecture de mémoire unifiée combinant Neo4j et des index vectoriels pour synchroniser la latence à court terme et la persistence à long terme.

*   **Mémoire de Travail (Low Latency)** : Structurée comme un graphe d'exécution direct (`AgentRun` -> `Event` -> `RawToolResponse`). Pour éviter la saturation du contexte, un Agent de Mémoire dédié compresse l'historique ancien en événements résumés. Ce graphe est *read-only* pour les agents d'exécution afin de garantir l'intégrité du flux de raisonnement.
*   **Mémoire Long Terme (Cross-Task)** : Un graphe de connaissances Neo4j distinct. Lorsqu'une information clé est identifiée, elle est indexée vectoriellement (ex: `text-embedding-3-small`) et insérée comme un nœud sémantique structuré (ex: `code_understanding`, `bug_pattern`) avec ses relations contextuelles.
*   **Synchronisation RAG** : La récupération (retrieval) combine la localisation topologique dans le graphe (recherche par sauts/relations) et la similarité vectorielle fine pour extraire le contenu exact.

## 3. Topologie Dynamique Multi-Agents et Sandboxing Docker

Pour gérer la limite de la fenêtre de contexte, YGN-SAGE doit adopter une topologie structurelle auto-générée capable de déléguer les processus cognitifs.

*   **Délégation Verticale** : L'agent `sage-discover` instancie dynamiquement des sous-agents spécialisés (ex: `gdb_helper`) avec un set d'outils restreint et une mission précise. Ce sous-agent opère dans son propre contexte et ne renvoie qu'une synthèse actionnable au parent.
*   **Isolation par Sandboxing (Docker)** : Chaque invocation d'outil sensible ou sous-agent de débogage s'exécute dans un conteneur Docker isolé.
*   **Optimisation par Snapshots** : L'état des outils (fichiers compilés, dépendances) est sauvegardé via des snapshots d'images Docker. Cela permet un "warm boot" des environnements de test, réduisant drastiquement la latence d'initialisation des bacs à sable.
*   **Exécution Asynchrone** : Les tâches lourdes s'exécutent en arrière-plan via des handles de processus, permettant à l'agent principal de poursuivre son raisonnement ou de paralléliser d'autres recherches horizontales.

---

## Conclusion
L'architecture de YGN-SAGE doit évoluer vers une gestion plus granulaire des environnements (Docker snapshots) et une méta-stratégie auto-adaptative (VAD-CFR évolué). La mémoire ne doit plus être une simple liste de messages, mais un graphe actif géré par un agent de compression dédié.
