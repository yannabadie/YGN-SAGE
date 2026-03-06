# Audit YGN-SAGE — Verdict sans filtre

**Repo**: `yannabadie/YGN-SAGE` | **Période**: 2-6 mars 2026 | **126 commits en 4 jours**

---

## Verdict global : Prototype ambitieux, crédible en structure, mais creux en substance réelle

YGN-SAGE est un **Agent Development Kit** architecturalement bien pensé qui souffre d'un problème fondamental : **le ratio documentation/plans vs. code fonctionnel est inversé**. Il y a ~13 300 lignes de plans/docs pour ~8 700 lignes de code source — c'est la signature classique d'un projet piloté par AI (Gemini CLI + Claude Code) où l'ambition architecturale a devancé l'implémentation réelle.

---

## Ce qui vaut vraiment quelque chose

### 1. Architecture propre et modulaire
La séparation `sage-core` (Rust/PyO3) + `sage-python` (SDK) + `sage-discover` (pipeline) + `ui/` (dashboard) est saine. Le `boot.py` câble tout proprement. Le workspace Cargo est minimal. Les frontières de responsabilité sont claires.

### 2. Le noyau Rust est réel
Ce n'est pas du code fantôme. Les dépendances sont légitimes et correctement utilisées :

- **petgraph** pour le S-MMU multi-graphe (temporel, sémantique, causal, entité) — 295 lignes propres
- **solana_rbpf** pour l'eBPF sandbox — vrai chargement ELF + exécution bytecode
- **z3** pour la validation formelle — `prove_memory_safety()` et `check_loop_bound()` sont corrects
- **arrow** pour la mémoire colonne — compaction Tier 2 fonctionnelle
- **dashmap** pour le cache FIFO+TTL — lock-free, avec TTL atomique

Les 1 727 lignes Rust sont denses et utiles. C'est la partie la plus solide du projet.

### 3. Le framework d'agent est fonctionnel
Le cycle `perceive → think → act → learn` dans `agent_loop.py` (480 lignes) implémente :

- Routage S1/S2/S3 avec retry budgets indépendants
- Boucle AVR (Act-Verify-Refine) pour S2 avec exécution sandbox
- Estimation de coût par modèle/tier
- Self-braking par entropie (CGRS)
- AgentEvent structuré pour l'observabilité

### 4. La pipeline Knowledge (sage-discover) est complète
Discovery → Curation → Ingestion → Migration — c'est un vrai pipeline ETL pour la recherche académique avec arXiv, Semantic Scholar, HuggingFace. Bien testé (45 tests).

### 5. CI/CD opérationnel
3 jobs GitHub Actions : Rust (fmt + clippy + test), Python SDK, Python Discover. C'est plus que ce que 90% des projets perso ont.

---

## Ce qui ne vaut pas grand-chose (ou qui est trompeur)

### 1. Les benchmarks sont 100% synthétiques et trompeurs

Le `official_benchmark_proof.json` affiche un **"speedup" de 2 050x** et une "success rate" de 100%. En réalité :

- Le "benchmark SciEvo" mute aléatoirement un octet dans du bytecode eBPF (incrémenter `r0` de 1 à 10). Ce n'est pas de l'évolution, c'est un `random.randint`.
- Le "benchmark RE-Bench" compare l'exécution d'une instruction `mov r0, 42` dans solana_rbpf à... rien. Il n'y a pas de baseline Python réelle. Le "2 050x speedup" est un nombre inventé.

**Le GEMINI.md lui-même interdit les "fake benchmarks"** — et pourtant ils sont dans le repo.

### 2. La "Cognitive Routing" S1/S2/S3 est un comparateur de seuils

Le `MetacognitiveController.route()` est fondamentalement :

```
si complexity > 0.7 → S3
si complexity < 0.35 et uncertainty < 0.3 → S1
sinon → S2
```

L'évaluation heuristique cherche des mots-clés comme "debug", "fix", "optimize". L'appel LLM pour l'assessment passe par Gemini Flash Lite — c'est une bonne idée, mais c'est un classifieur binaire glorifié, pas de la "métacognition".

### 3. L'évolution DGM ne s'auto-modifie pas

Le README affirme "Darwin Godel Machine (DGM) capable of self-modifying the mutator logic". En réalité, SAMPO choisit 1 parmi 5 directives textuelles fixes ("Optimize performance", "Fix edge cases"...) et l'injecte dans le prompt du LLM mutateur. C'est du prompt engineering, pas de l'auto-modification.

### 4. Les tests sont exhaustifs en couverture mais superficiels en profondeur

Les 200+ tests Python sont quasi-intégralement mockés. `test_agent.py` mocke le LLM. `test_evolution.py` teste la population/mutateur avec des valeurs synthétiques. `test_metacognition.py` teste les seuils.

**Aucun test d'intégration réel** ne connecte le flux complet boot → metacognition → LLM → tools → memory → response. Les tests prouvent que le code ne crashe pas, pas qu'il *fonctionne*.

### 5. 13 000+ lignes de "plans" — pour 4 jours de développement

Les fichiers dans `docs/plans/` totalisent 13 266 lignes. Certains sont des artefacts de planification Gemini qui n'ont jamais été implémentés (ex: `2026-03-05-opensage-surpass-implementation.md` — 1 771 lignes de plan, zéro commit de code correspondant tracé).

### 6. La mémoire épisodique est volatile

Le README note "in-memory only, no cross-session persistence" — mais c'est sous-estimé. À chaque redémarrage, tout est perdu. Pour un "Self-Adaptive Generation Engine", c'est une lacune critique.

---

## Analyse structurelle

| Métrique | Valeur | Évaluation |
|----------|--------|------------|
| Lignes de code source | ~8 700 (5.5K Python + 1.7K Rust + 1.5K Discover) | Correct pour un proto |
| Lignes de tests | ~4 000 | Bon ratio (46%) |
| Lignes de documentation/plans | ~13 300 | **Disproportionné** |
| Commits | 126 en 4 jours | Vélocité AI |
| Dépendances Rust réelles | petgraph, solana_rbpf, z3, arrow, dashmap, pyo3 | Solide |
| Tests d'intégration réels | 0 | **Critique** |
| Production deployments | 0 | Proto uniquement |
| Contributeurs | 1 (+ AI agents) | Normal pour un proto perso |

---

## Plan d'amélioration concret (3 axes)

### Axe 1 — Élaguer et consolider (Semaine 1-2)

**Objectif** : Réduire la surface de confabulation, garder ce qui marche.

1. **Supprimer tous les benchmarks synthétiques** (`benchmark_results.json`, `official_benchmark_proof.json`, `cybergym_benchmark_proof.json`, `benchmark_dashboard.html`). Ils décrédibilisent le projet.

2. **Archiver `docs/plans/` dans une branche `archive/plans`**. Ne garder dans `master` que les ADR (Architecture Decision Records) qui reflètent des décisions *implémentées*.

3. **Supprimer `conductor/`** (déjà vidé mais le dossier traîne), `memory-bank/` (artefact Gemini), `research_journal/` (notes internes).

4. **Réécrire le README en 100 lignes max**. Le README actuel fait ~400 lignes avec des claims non vérifiées. Un proto honnête a besoin de : what it does, how to run it, what works, what doesn't.

### Axe 2 — Prouver que ça marche (Semaine 2-4)

**Objectif** : Un chemin d'exécution réel, bout en bout, documenté.

5. **Écrire 3 tests d'intégration end-to-end** avec le MockProvider qui vérifient le flux complet :
   - `boot_agent_system(mock) → system.run("simple question") → S1 routing → response`
   - `boot_agent_system(mock) → system.run("debug this code") → S2 routing → AVR loop → response`
   - `boot_agent_system(mock) → system.run("prove X") → S3 routing → Z3 validation → response`

6. **Enregistrer une session réelle** avec Gemini Flash (pas Codex) sur une tâche non triviale, stocker la trace `AgentEvent` dans le repo comme preuve. Un `.jsonl` de 50 événements vaut plus que 13K lignes de plans.

7. **Persister la mémoire épisodique** — SQLite suffit. 30 lignes de code. Sans ça, "Self-Adaptive" est un mensonge.

8. **Brancher un vrai benchmark** : prendre une tâche de SWE-bench Lite ou HumanEval, faire tourner l'agent, mesurer et publier le résultat brut (même mauvais). Un résultat honnête de 15% sur HumanEval vaut infiniment plus qu'un "2 050x speedup" inventé.

### Axe 3 — Trouver le use case (Semaine 4+)

**Objectif** : Décider ce que YGN-SAGE *est* réellement.

Tu as 3 options, et il faut en choisir **une seule** :

**Option A — ADK pour ERP/MES** (pivot vers ton métier)
Fusionner avec tes MCP servers `x3-oracle` et `cegid-oracle`. YGN-SAGE devient le runtime d'agent qui orchestre des requêtes ERP complexes. Le S1/S2/S3 routing a un vrai sens ici : S1 pour les lookups simples, S2 pour les requêtes multi-tables, S3 pour la validation formelle de cohérence de données.

**Option B — Plugin MetaYGN** (merge dans ton autre projet)
Le noyau Rust (eBPF sandbox, Z3 validator, FIFO cache, Arrow memory) est excellent comme *bibliothèque de composants* pour MetaYGN. Extraire `sage-core` comme crate indépendante, abandonner le Python SDK standalone.

**Option C — Honest Research PoC** (publier tel quel)
Nettoyer, ajouter un `LIMITATIONS.md` brutal, publier comme "exploration architecturale" avec un blog post. Pas de prétention à l'utilisation en production. Valeur = portfolio technique.

---

## Score final

| Critère | Note /10 | Commentaire |
|---------|----------|-------------|
| Architecture | 8 | Bien pensée, modulaire, extensible |
| Qualité du code Rust | 7 | Dense, correct, bien typé |
| Qualité du code Python | 6 | Clean mais trop de surface mockée |
| Tests | 4 | Volume OK, profondeur insuffisante |
| Documentation | 3 | Beaucoup de bruit, peu de signal |
| Honnêteté des claims | 2 | Benchmarks synthétiques, termes "ASI/DGM" exagérés |
| Production-readiness | 1 | Pas testable en conditions réelles sans API keys |
| **Potentiel réel** | **7** | **Les fondations sont là, il faut élaguer et prouver** |

---

*Audit réalisé par analyse statique complète du repo (126 commits, 54+ fichiers source, tous les plans et benchmarks examinés). Aucune exécution de code effectuée.*
