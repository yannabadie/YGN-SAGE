---
name: No Heuristic Tuning
description: User explicitly rejected hardcoded heuristic tuning - demands research-backed principled approaches
type: feedback
---

User rejected adding more keywords / adjusting thresholds to fix routing accuracy:
"L'approche n'est pas sérieuse, rajouter de l'heuristique, hardcoder des valeurs arbitraires... Ca ne colle pas au projet."

Instead, user expects:
1. Research on ArXiv, GitHub, ExoCortex, web for SOTA approaches
2. Principled, learned methods backed by papers
3. Implementation grounded in published results

Successfully applied: kNN routing (arXiv 2505.12601) replaced keyword heuristic, improving from 52% to 92% accuracy.
