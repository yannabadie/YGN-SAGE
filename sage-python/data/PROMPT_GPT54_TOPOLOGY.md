# Prompt pour ChatGPT 5.4 Pro — Génération de topologies multi-agents

> Copie-colle ce prompt dans ChatGPT Pro avec réflexion étendue + recherche web activée.
> Sauvegarde chaque réponse dans un fichier .jsonl (une ligne JSON par topologie).

---

## Le prompt à copier

```
Tu es un expert en systèmes multi-agents pour la résolution de problèmes de programmation compétitive.

Je développe YGN-SAGE, un framework qui orchestre des agents LLM en topologies (graphes dirigés). Un petit modèle (9B paramètres) apprend via reinforcement learning à générer ces topologies optimales.

J'ai besoin de données d'entraînement : pour chaque tâche de programmation, génère UNE topologie optimale au format JSON strict.

RÈGLES STRICTES :
1. Chaque topologie a 3 à 7 nœuds selon la difficulté
2. Le DERNIER nœud DOIT être un "synthesizer" avec ce prompt exact dans son champ prompt : "Produce the final, complete, self-contained Python solution incorporating all feedback. Return ONLY the code inside a single ```python fenced block. No explanation, no commentary."
3. Les rôles disponibles : planner, coder, reviewer, tester, debugger, analyst, synthesizer
4. Les model_tier : "reasoner" (tâches complexes), "fast" (tâches simples), "budget" (vérification)
5. Les flow_type pour les edges : "control", "message", "state"
6. La difficulté : "simple" (1-2 nœuds), "moderate" (3-4 nœuds), "complex" (5-7 nœuds)
7. Chaque prompt de nœud doit être SPÉCIFIQUE à la tâche (pas générique)

FORMAT JSON STRICT (une seule ligne par topologie) :
{"task_id": "...", "prompt": "...", "topology": {"reasoning": "...", "difficulty": "...", "nodes": [...], "edges": [...]}, "node_count": N, "edge_count": N, "difficulty": "..."}

EXEMPLE pour une tâche modérée :
{"task_id": "Custom/reverse_linked_list", "prompt": "Implement a function to reverse a singly linked list in-place.", "topology": {"reasoning": "This requires careful pointer manipulation. A planner identifies the approach, a coder implements, and a reviewer checks edge cases.", "difficulty": "moderate", "nodes": [{"role": "planner", "prompt": "Analyze the linked list reversal problem. Identify the three-pointer technique (prev, current, next). List edge cases: empty list, single node, circular reference.", "model_tier": "reasoner"}, {"role": "coder", "prompt": "Implement reverse_linked_list(head) using the three-pointer technique from the plan. Handle None input. Use iterative approach, not recursive.", "model_tier": "reasoner"}, {"role": "reviewer", "prompt": "Review the reversal code. Check: does it handle head=None? Does it return the new head? Is the loop termination correct? Are there memory leaks?", "model_tier": "fast"}, {"role": "synthesizer", "prompt": "Produce the final, complete, self-contained Python solution incorporating all feedback. Return ONLY the code inside a single ```python fenced block. No explanation, no commentary.", "model_tier": "fast"}], "edges": [{"from_idx": 0, "to_idx": 1, "flow_type": "control"}, {"from_idx": 1, "to_idx": 2, "flow_type": "message"}, {"from_idx": 2, "to_idx": 3, "flow_type": "message"}]}, "node_count": 4, "edge_count": 3, "difficulty": "moderate"}

MAINTENANT, génère 20 topologies pour ces tâches de programmation COMPLEXES (5-7 nœuds chacune). Utilise ta recherche web pour trouver des problèmes réels de compétitions de programmation (Codeforces, ICPC, Google Code Jam). Chaque topologie doit avoir des rôles VRAIMENT distincts — pas de padding.

Retourne UNIQUEMENT les 20 lignes JSON, une par ligne, sans texte autour.
```

---

## Comment utiliser

1. Ouvre ChatGPT Pro (5.4) avec réflexion étendue
2. Colle le prompt ci-dessus
3. ChatGPT génère 20 lignes JSON
4. Copie la réponse dans un fichier texte
5. Répète 5-10 fois avec des variations :
   - "génère 20 topologies pour des tâches BigCodeBench HARD"
   - "génère 20 topologies pour des algorithmes de graphes"
   - "génère 20 topologies pour du machine learning appliqué"
   - "génère 20 topologies SIMPLES (1-2 nœuds) pour des fonctions utilitaires"
6. Concatène tous les fichiers : `cat gpt54_*.jsonl > topology_sft_gpt54_pro.jsonl`

## Validation

```bash
cd sage-python
python -c "
import json, yaml
valid = 0
with open('data/topology_sft_gpt54_pro.jsonl') as f:
    for i, line in enumerate(f):
        try:
            e = json.loads(line)
            nodes = e['topology']['nodes']
            assert len(nodes) >= 1
            assert nodes[-1]['role'] == 'synthesizer'
            valid += 1
        except:
            print(f'Line {i}: INVALID')
print(f'{valid} valid topologies')
"
```
