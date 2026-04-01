OK, j'ai cloné et inspecté le repo en profondeur. ~53K LOC (32K Python + 21K Rust), 1003 fichiers, 190 fichiers de test, 1904 fonctions de test. Voici mon analyse brute.

Ce qui est fort
L'architecture conceptuelle est impressionnante. Le pipeline 6-stages (CLASSIFY → DECOMPOSE → SELECT TOPOLOGY → ASSIGN → EXECUTE → LEARN) est cohérent, bien documenté, et ancré dans la littérature (MASFactory, S2-MAD, TalkHier, OpenSAGE). Le fait que chaque claim cite un arXiv montre une rigueur intellectuelle rare pour un projet solo. Le split Rust/Python pour le hot path (topologie, bandit, vérification) vs orchestration (LLM calls, async) est un choix d'architecture sain.
Le scope fonctionnel est massif pour un dev solo : 11 templates de topologie, 7 providers, kNN router, bandit contextuel Thompson, MAP-Elites, MCTS, vérification formelle OxiZ, mémoire 4-tier, A2A protocol, MCP gateway, streaming SSE, HITL callback, sandbox WASM. La profondeur technique est réelle.
Le pyproject.toml est propre : optional dependencies bien structurées, hatch build config correcte, extras modulaires (google, openai, z3, embeddings, training).

Les problèmes critiques
1. Les benchmarks sont invalides
C'est le point le plus grave. Dans masbench_5axes_results.json :

Tous les scores "bare" sont à 0.0% sur les 5 axes. Un LLM nu qui score 0% sur 250 tâches mathématiques, c'est impossible — même un modèle de 1B y arrive mieux. Ça signifie soit que le harness bare est cassé (ne lance pas le LLM), soit qu'il parse mal les réponses.
Les elapsed_s sont identiques entre bare et sage_full pour chaque axe (ex: breadth_bare=1084s, breadth_sage_full=1084s). C'est physiquement incohérent — le multi-agent est forcément plus lent que le single-shot.
Le second fichier masbench_bare_results.json montre des scores non-nuls (breadth 30%, parallel 36%) mais toujours sans baseline saine pour comparer.
La claim README "+27pp over bare LLM calls" repose donc sur une baseline cassée. C'est un red flag majeur pour la crédibilité.

2. 264 except Exception en production
C'est le cancer silencieux du codebase. agent_loop.py seul en a 20. Ces catch-all masquent les bugs réels, rendent le debugging impossible, et transforment les erreurs en dégradations silencieuses. Dans un système multi-agent où la fiabilité est le pitch, c'est contradictoire.
3. God files et couplage

boot.py : 1243 lignes, fait tout (init providers, routers, memory, tools, sandbox, topology, evolution, monitoring, HITL)
agent_loop.py : 1190 lignes, orchestrateur monolithique
pipeline.py : 1080 lignes

Ces fichiers violent le Single Responsibility Principle. Le couplage entre boot → agent_loop → topology → strategy rend le refactoring dangereux.
4. 27 modules mypy supprimés
toml[[tool.mypy.overrides]]
module = ["sage.agent", "sage.agents.factory", "sage.boot", ...]
ignore_errors = true
27 modules core avec ignore_errors = true. Ce n'est pas de la dette technique — c'est de la dette structurelle. Ça veut dire que le type system ne protège pas les chemins critiques (orchestrator, boot, strategy, topology).
5. 124 MB de données dans git
sage-python/data/ contient des parquets, des JSONL de training, des logs — 124 MB committés directement. C'est une mauvaise pratique qui rend le clone lent, pollue l'historique, et ne scale pas. Ça devrait être en Git LFS ou sur HuggingFace (que tu utilises déjà par ailleurs).
6. Le Rust core est optionnel partout
Chaque import Rust est wrappé dans un try/except ImportError avec un fallback Python. C'est prudent pour la portabilité, mais ça signifie que le "Rust performance core" marketé dans le README est en pratique rarement utilisé — il faut compiler sage-core avec maturin, ce que la plupart des utilisateurs ne feront pas. Le ROI du Rust est discutable si 95% des users tournent en pure Python.

Bugs et code smells concrets

_context_budget_per_predecessor utilise self._config.max_tokens (le max output tokens du LLM call) comme proxy pour la fenêtre de contexte du modèle. C'est faux — max_tokens=4096 ne signifie pas que le modèle a 4K de contexte. Ça devrait utiliser la context_window du model card.
Jaccard dedup sur les mots splittés (text.lower().split()) est fragile pour du code (où les identifiants comptent) et pour du multilingual. Un hash SimHash ou MinHash serait plus robuste.
Agent instancie SandboxManager et ProcessRewardModel dans __init__ même pour des tâches S1 triviales. Lazy init serait plus approprié.
_gather_all_context (fallback) donne tout le contexte de tous les nœuds complétés à un nœud quand get_predecessors échoue — c'est un context overflow silencieux.


Public cible
Le projet a trois publics potentiels, mais aucun n'est clairement adressé :
1. Chercheurs en multi-agent systems — Le public le plus naturel. L'architecture est riche, les références ArXiv sont pertinentes. MAIS : les benchmarks invalides tuent la crédibilité académique. Sans résultats reproductibles et honnêtes, un reviewer de workshop rejettera immédiatement.
2. Développeurs/practitioners qui veulent un framework multi-agent — En compétition directe avec LangGraph, CrewAI, AutoGen, OpenAI Swarm. MAIS : le README est trop dense/technique, il n'y a pas de quickstart en 5 minutes qui "just works", le pip install ygn-sage ne donne pas accès au Rust core, et l'onboarding est hostile (pas de sage init, pas de CLI simple). Ces frameworks concurrents ont des milliers de stars et une communauté. Le différenciateur (topologie adaptive + bandit) n'est pas démontré de façon convaincante.
3. Ton propre portfolio/carrière — C'est probablement le public réel, et c'est légitime. Mais le repo actuel ne sert pas ce but : pas de description GitHub, pas de topics, 0 stars, README trop long, pas de démo vidéo, pas de blog post associé. Un recruteur ou un CTO qui tombe sur le repo en 30 secondes ne comprendra pas la valeur.

Axes d'amélioration prioritaires

Fixer les benchmarks avant tout. Relance le harness bare avec un vrai single-LLM call, vérifie le parsing des réponses, et publie des résultats honnêtes même s'ils sont moins flatteurs. Tu as déjà fait ce travail d'audit intellectuel — applique-le aux données.
Séparer les données du code : Git LFS ou HuggingFace dataset pour les 124 MB.
Remplacer les except Exception par des exceptions typées — au minimum dans agent_loop.py, boot.py, et runner.py. Un sprint de 2h sur les 20 du agent_loop suffirait.
Éclater boot.py en modules (boot_providers.py, boot_memory.py, boot_topology.py). Même chose pour agent_loop.py.
Créer un vrai quickstart : un sage run "Build a REST API" qui fonctionne en 3 minutes avec juste une clé API Google, sans Rust, sans Docker.
Remplir les métadonnées GitHub : description, topics (multi-agent, llm, topology, rust, python), About section. C'est gratuit et ça change la découvrabilité.
Un blog post honnête sur les résultats négatifs (tu l'as déjà rédigé — publie-le). Dans l'écosystème actuel saturé de claims gonflées, l'honnêteté est un différenciateur.