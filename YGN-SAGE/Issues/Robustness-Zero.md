---
title: "MASBENCH Robustness 0%"
type: issue
severity: P2
status: ouvert
tags:
  - issue
  - benchmark
created: 2026-04-07
---

# MASBENCH Robustness 0%

## Probleme

Sur l'axe robustness : bare 0.0% ET SAGE 0.0%.
Aucune des deux approches ne passe les taches de robustesse.

## Hypotheses

1. **Bug benchmark** : les taches de robustesse sont peut-etre mal formulees ou impossibles
2. **Hors scope** : le type de robustesse teste (adversarial? noise?) n'est peut-etre pas ce que SAGE adresse
3. **Evaluation trop stricte** : le grader rejette peut-etre des reponses partiellement correctes

## Action requise

1. Examiner les taches robustness individuellement
2. Verifier le grader/evaluateur
3. Comparer avec ce que d'autres frameworks obtiennent sur cet axe

## Impact

Si c'est un bug benchmark, c'est pas grave.
Si les taches sont legits et 0%, ca signifie que SAGE n'apporte rien sur la robustesse.
