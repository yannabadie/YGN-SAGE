---
title: "BigCodeBench Hard — Avril 2026"
type: benchmark
benchmark: bigcodebench
date: 2026-04-01
tags:
  - benchmark
  - bigcodebench
---

# BigCodeBench Hard — Avril 2026

## Configuration

- **Subset** : Hard
- **Split** : Instruct
- **Limite** : 148 taches
- **Modele** : budget (DeepSeek primaire)

## Resultat

| Metrique | Score |
|----------|-------|
| Pass rate | **37.8%** (56/148) |
| Leaderboard (gele) | 33.1% (o3-mini) |
| SOTA (The Conductor) | 40.0% |

## Commande

```bash
python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 148
```

> [!warning] Comparaison biaisee
> Le leaderboard BigCodeBench est **gele depuis avril 2025**.
> Les modeles frontier actuels (GPT-5.4, Opus 4.6) ne sont pas soumis.
> Le 37.8% bat le leaderboard gele (33.1%) mais probablement pas les modeles recents.
>
> **Le vrai chiffre qui compte** : le delta framework (SAGE vs bare LLM avec le meme modele).
> Ce delta n'est pas mesure ici — c'est MASBENCH qui le mesure.

## Notes

- BigCodeBench Hard est non-sature (ICLR '25), donc pertinent
- Mais la comparaison avec un leaderboard gele est trompeuse
- Competitor direct : The Conductor a 40.0% (ICLR 2026)
