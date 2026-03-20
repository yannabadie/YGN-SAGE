# Prompts ChatGPT 5.4 Pro — 5 types de données pour YGN-SAGE

> Utilise ChatGPT Pro avec réflexion étendue + recherche web activée.
> Sauvegarde chaque réponse dans le fichier .jsonl indiqué.
> Validation à la fin de chaque session.

---

## Type 1 — Paires erreur→correction (PRIORITÉ HAUTE)

> Fichier: `data/gpt54_error_correction.jsonl`
> Objectif: 100+ paires. Enseigne la récupération d'échec.

```
Tu es un expert en orchestration multi-agents pour la programmation. Je développe YGN-SAGE, un système qui génère des topologies d'agents (graphes DAG en YAML) pour résoudre des tâches de code.

J'ai besoin de données d'ENTRAÎNEMENT pour apprendre à corriger des topologies qui échouent.

Pour chaque entrée, génère une paire :
- topology_v1 : une topologie qui a un DÉFAUT réaliste (noeud manquant, mauvais ordre, prompt trop vague, pas de tester, reviewer absent)
- error_type : le type d'erreur ("WRONG_ANSWER", "RUNTIME_ERROR", "TIMEOUT", "NO_CODE")
- error_description : explication de pourquoi ça échoue
- topology_v2 : la topologie CORRIGÉE qui résout le problème

FORMAT JSON STRICT, une ligne par paire :
{"task_id": "ErrorPair/N", "prompt": "<la tâche de programmation>", "topology_v1": {"reasoning": "...", "difficulty": "...", "nodes": [...], "edges": [...]}, "error_type": "WRONG_ANSWER", "error_description": "Le coder ne gère pas les cas limites car il n'y a pas de tester", "topology_v2": {"reasoning": "Ajout d'un tester qui vérifie les edge cases avant le synthesizer", "difficulty": "...", "nodes": [...], "edges": [...]}}

RÈGLES :
- Le DERNIER noeud de topology_v2 DOIT être un "synthesizer" avec prompt : "Produce the final, complete, self-contained Python solution incorporating all feedback. Return ONLY the code inside a single ```python fenced block. No explanation, no commentary."
- topology_v1 doit avoir un défaut RÉALISTE (pas un défaut évident)
- topology_v2 doit corriger le défaut de manière MINIMALE (pas tout refaire)
- Les prompts doivent être SPÉCIFIQUES à la tâche
- Utilise ta recherche web pour trouver des tâches réelles (Codeforces, LeetCode, BigCodeBench)

Génère 20 paires. Retourne UNIQUEMENT les 20 lignes JSON.
```

---

## Type 2 — Préférences A vs B (pour futur DPO)

> Fichier: `data/gpt54_preferences.jsonl`
> Objectif: 200+ paires. Données de préférence pour DPO.

```
Tu es un expert en évaluation de topologies multi-agents pour le code.

Pour chaque tâche de programmation, je te donne 2 topologies différentes (A et B). Tu dois juger laquelle est MEILLEURE et expliquer pourquoi.

FORMAT JSON STRICT, une ligne par jugement :
{"task_id": "Pref/N", "prompt": "<tâche>", "topology_a": {"nodes": [...], "edges": [...]}, "topology_b": {"nodes": [...], "edges": [...]}, "preferred": "A" ou "B", "reasoning": "<pourquoi A ou B est meilleur — cite des critères concrets>", "score_a": 0.0-1.0, "score_b": 0.0-1.0}

CRITÈRES D'ÉVALUATION :
1. Parcimonie : moins de noeuds pour le même résultat = mieux
2. Spécificité des prompts : "vérifie les off-by-one dans le binary search" > "review the code"
3. Couverture des rôles : planner+coder+reviewer+tester > coder seul pour les tâches complexes
4. Structure des edges : le flow d'information est-il logique?
5. Adéquation difficulté/taille : une tâche simple ne doit pas avoir 7 noeuds

Pour chaque entrée :
- Crée une tâche réaliste (utilise ta recherche web pour des problèmes réels)
- Génère topology_a (bonne) et topology_b (moins bonne, mais pas évidemment mauvaise)
- Alterne qui est A et qui est B (évite le biais "A est toujours meilleur")
- Le dernier noeud de CHAQUE topologie doit être un synthesizer

Génère 20 jugements. Retourne UNIQUEMENT les 20 lignes JSON.
```

---

## Type 3 — Topologies avec reasoning profond (chain-of-thought)

> Fichier: `data/gpt54_deep_reasoning.jsonl`
> Objectif: 100+ entrées. Le modèle apprend POURQUOI chaque noeud existe.

```
Tu es un architecte de systèmes multi-agents spécialisé en programmation compétitive.

Pour chaque tâche, analyse EN PROFONDEUR puis génère une topologie. Le champ "reasoning" doit être un VRAI raisonnement de 200-500 mots qui explique :
- Quel algorithme résout cette tâche
- Pourquoi chaque noeud est NÉCESSAIRE (pas optionnel)
- Quels edge cases le reviewer/tester doit vérifier
- Pourquoi cet ordre de noeuds est optimal
- Quel model_tier chaque noeud nécessite et pourquoi

FORMAT JSON STRICT :
{"task_id": "Deep/N", "prompt": "<tâche détaillée>", "topology": {"reasoning": "<200-500 mots d'analyse profonde>", "difficulty": "complex", "nodes": [{"role": "...", "prompt": "<prompt spécifique qui référence l'algorithme>", "model_tier": "..."}], "edges": [{"from_idx": N, "to_idx": N, "flow_type": "..."}]}, "node_count": N, "edge_count": N, "difficulty": "complex"}

RÈGLES :
- Utilise ta RECHERCHE WEB pour trouver des problèmes de compétition réels (Codeforces div1/div2, ICPC, Google Code Jam 2024-2025)
- Minimum 5 noeuds par topologie (ce sont des tâches complexes)
- Le reasoning doit mentionner l'ALGORITHME par son nom (BFS, dynamic programming, segment tree, etc.)
- Les prompts de noeuds doivent référencer les contraintes SPÉCIFIQUES du problème (N <= 10^5, etc.)
- Dernier noeud = synthesizer avec l'instruction ```python

Génère 20 topologies. Retourne UNIQUEMENT les 20 lignes JSON.
```

---

## Type 4 — Topologies SIMPLES calibrées (rééquilibrage)

> Fichier: `data/gpt54_simple_calibrated.jsonl`
> Objectif: 100+ entrées. Topologies 1-2 noeuds pour tâches simples.

```
Tu es un expert en systèmes multi-agents MINIMALISTES.

Principe fondamental : ne PAS sur-ingéniérer. Une tâche simple (tri de liste, manipulation de string, calcul mathématique) ne nécessite PAS 5 agents. Un seul coder suffit.

Pour chaque tâche SIMPLE, génère une topologie MINIMALE :
- 1 noeud : juste un "coder" + synthesizer (2 noeuds max)
- 2 noeuds : un "planner" + "coder" + synthesizer (3 noeuds max) SEULEMENT si le planning est nécessaire

FORMAT JSON STRICT :
{"task_id": "Simple/N", "prompt": "<tâche simple>", "topology": {"reasoning": "<pourquoi cette topologie minimale suffit — 2-3 phrases>", "difficulty": "simple", "nodes": [...], "edges": [...]}, "node_count": N, "edge_count": N, "difficulty": "simple"}

RÈGLES :
- difficulty = "simple" TOUJOURS
- Maximum 3 noeuds (coder + optionnel planner + synthesizer obligatoire)
- Le reasoning doit JUSTIFIER pourquoi c'est simple (pas de "it's straightforward")
- Les tâches doivent être variées : string, math, list, dict, file I/O, regex, datetime
- Dernier noeud = synthesizer
- PAS de reviewer ni tester pour les tâches simples

Génère 20 topologies. Retourne UNIQUEMENT les 20 lignes JSON.
```

---

## Type 5 — Audit et amélioration de données existantes

> Fichier: `data/gpt54_audit.jsonl`
> Objectif: Améliorer 50+ topologies existantes.

```
Tu es un auditeur expert en topologies multi-agents. Je te donne des topologies de notre dataset d'entraînement. Pour chacune, tu dois :
1. Identifier les DÉFAUTS (prompt vague, noeud inutile, edge manquant, mauvais model_tier)
2. Proposer une version AMÉLIORÉE
3. Expliquer ce que tu as changé

FORMAT JSON STRICT :
{"task_id": "<original_task_id>", "prompt": "<la tâche>", "original": {"nodes": [...], "edges": [...]}, "issues": ["prompt du coder trop vague", "pas de tester pour un algo de graphe"], "improved": {"reasoning": "...", "difficulty": "...", "nodes": [...], "edges": [...]}, "changes": "<résumé des changements>"}

Voici 10 topologies à auditer :

[COLLE ICI 10 ENTRÉES DE topology_sft_v2_combined.jsonl]

Pour chaque topologie :
- Le dernier noeud de "improved" DOIT être un synthesizer avec l'instruction ```python
- Ne change PAS la difficulté sauf si elle est clairement fausse
- Garde les noeuds qui sont bons, ne change que ce qui est défaillant
- Si la topologie est déjà bonne, dis "NO_ISSUES" et copie-la telle quelle

Retourne UNIQUEMENT les 10 lignes JSON.
```

> Pour ce prompt, tu dois d'abord extraire 10 entrées du dataset :
> ```bash
> cd sage-python
> python -c "
> import json
> with open('data/topology_sft_v2_combined.jsonl', encoding='utf-8') as f:
>     lines = [json.loads(l) for l in f]
> # Prendre 5 BigCodeBench + 5 CodeContests
> bcb = [l for l in lines if l['task_id'].startswith('BigCodeBench')][:5]
> cc = [l for l in lines if l['task_id'].startswith('CodeContests')][:5]
> for e in bcb + cc:
>     print(json.dumps({'task_id': e['task_id'], 'prompt': e['prompt'], 'topology': e['topology']}, ensure_ascii=False))
> "
> ```
> Copie le résultat à la place de [COLLE ICI...] dans le prompt.

---

## Validation de toutes les données

Après avoir généré les 5 types, valide :

```bash
cd sage-python
python -c "
import json, os

files = {
    'error_correction': 'data/gpt54_error_correction.jsonl',
    'preferences': 'data/gpt54_preferences.jsonl',
    'deep_reasoning': 'data/gpt54_deep_reasoning.jsonl',
    'simple_calibrated': 'data/gpt54_simple_calibrated.jsonl',
    'audit': 'data/gpt54_audit.jsonl',
}

for name, path in files.items():
    if not os.path.exists(path):
        print(f'{name}: NOT FOUND')
        continue
    valid = 0
    total = 0
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            total += 1
            try:
                e = json.loads(line)
                # Check it has topology with nodes
                topo = e.get('topology_v2', e.get('improved', e.get('topology', {})))
                nodes = topo.get('nodes', [])
                assert len(nodes) >= 1, 'no nodes'
                # Check last node is synthesizer
                if nodes[-1].get('role') == 'synthesizer':
                    valid += 1
                else:
                    print(f'  {name} line {i}: last node is {nodes[-1].get(\"role\")}, not synthesizer')
                    valid += 1  # count anyway for preferences/audit
            except Exception as ex:
                print(f'  {name} line {i}: {ex}')
    print(f'{name}: {valid}/{total} valid')
"
```

## Intégration dans le training

Quand tous les fichiers sont prêts :

```bash
# Combiner toutes les nouvelles données
cat data/gpt54_error_correction.jsonl \
    data/gpt54_deep_reasoning.jsonl \
    data/gpt54_simple_calibrated.jsonl \
    > data/topology_sft_gpt54_pro.jsonl

# Reconstruire le dataset combiné
python scripts/build_sft_v2_combined.py \
    --extra data/topology_sft_gpt54_pro.jsonl \
    --output data/topology_sft_v3_combined.jsonl

# Les préférences et l'audit sont pour des usages séparés (DPO, qualité)
```
