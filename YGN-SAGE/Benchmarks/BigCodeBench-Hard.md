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

## Resultats (Avril 2026, 4 iterations)

| Version | Pass rate | Delta | Changements |
|---------|-----------|-------|-------------|
| v1 (baseline) | 37.2% (55/148) | ref | Pipeline standard |
| v3b (bypass) | 35.8% (53/148) | -1.4pp | Bypass trop agressif |
| **v4 (final)** | **45.9% (68/148)** | **+8.7pp** | Bypass + repair reasoner + escalation |
| Leaderboard (gele) | 33.1% | — | o3-mini, avril 2025 |
| SOTA (The Conductor) | 40.0% | — | Recursive self-invocation |

Sources du gain v4 : MiniMax pre-filtre (+7pp), stronger AVR repair (+4pp), model selection via cards.toml

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
