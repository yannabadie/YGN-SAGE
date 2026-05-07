---
name: cgpro YGN-SAGE project centralization
description: Yann directive 2026-05-06 — toutes les conv cgpro pour YGN-SAGE doivent être centralisées dans le projet ChatGPT "YGN-SAGE" (gizmoId g-p-69ed9637e63c8191b61c9741b50d1c01). Garder la même conv vivante pour plusieurs questions sur le même topic, démarrer une nouvelle conv (sans --resume) pour un nouveau topic afin que l'auto-routing projet se déclenche.
type: feedback
originSessionId: 88857be6-7048-463a-8ee4-cb3b4cca20fd
---
# cgpro YGN-SAGE project centralization

**Rule (Yann 2026-05-06)** : pour YGN-SAGE, toutes les conversations cgpro doivent vivre dans le projet ChatGPT **"YGN-SAGE"** (gizmoId `g-p-69ed9637e63c8191b61c9741b50d1c01`, déjà lié au cwd depuis 2026-04-28). Et il est explicitement OK de garder la même conv vivante pour plusieurs questions sur le même topic — c'est même recommandé pour la continuité.

**Why** : suivi cohérent dans un seul folder ChatGPT, project memory pre-pendée à chaque message dans une conv du projet, plus facile pour Yann de retrouver l'historique.

**How to apply** :

| Situation | Bonne commande |
|---|---|
| Nouveau topic / nouvelle cycle (e.g. Phase 1.5 ToolPolicy DESIGN) | `cgpro ask --save cgpro_<slug>_<date> "..."` SANS `--resume`. Auto-routes vers YGN-SAGE project + crée un alias pour future continuity. |
| Continuer un thread existant sur le MÊME topic (e.g. round-N de cycle-13 K) | `cgpro ask --resume <name|id> "..."` — garde la conv vivante pour ce topic. |
| Vérifier l'état du link | `cgpro project show` (mais nécessite daemon stopped si daemon-mode actif). `cgpro project list` pour voir tous les projets disponibles. |
| Re-lier si le lien casse | `cgpro project link "YGN-SAGE"` ou par gizmoId. |

**Trap connu** (CLAUDE.md verified 2026-04-28) : `--resume <name|id>` DISABLES project auto-routing. Une conv resumed reste à son emplacement original. Donc pour qu'une conv se retrouve dans le projet YGN-SAGE, elle doit être créée SANS `--resume` ET pendant que `cgpro project show` confirme un lien actif.

**Active threads à connaître pour résumer** (à enrichir au fil du temps) :
- `cgpro_pi_mono_pivot_20260505` — cycle-12+ pi-mono pivot strategic thread (cycle-12 cycle-13 B chain).
- `Analyse approfondie de repo` (id `69fb0d11-9bd8-8390-a074-edb6826f8cb6`) — ALIRE.md remediation thread (cycle-13 K Phase 0 + 0.6 + 0.6b + 0.6c + 0.6d). NOTE: ouverte hors-projet à l'origine ; les rounds 1-6 utilisent `--resume` donc restent à leur emplacement.

**Operating rule pour l'avenir** :
1. Pour un NOUVEAU topic post cycle-13 K (e.g. Phase 1.5 ToolPolicy DESIGN, Phase 2.1 facade rewrite DESIGN, etc.) : créer une nouvelle conv SANS `--resume`, avec `--save cgpro_<slug>_<date>` pour bookmark. Elle se retrouvera DANS le projet YGN-SAGE.
2. Pour continuer le travail sur ce topic ensuite : `--resume cgpro_<slug>_<date>`. Continuité OK même si `--resume` désactive l'auto-routing — la conv était déjà dans le projet à sa création.
3. Ne PAS migrer rétroactivement les threads pré-existants hors-projet (e.g. `Analyse approfondie de repo`) — ça casserait la continuité. À leur clôture, en démarrer une nouvelle in-project pour la suite.
