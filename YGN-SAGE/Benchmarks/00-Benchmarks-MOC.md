---
title: Benchmarks MOC
type: moc
tags:
  - benchmarks
  - moc
updated: 2026-04-07
---

# Benchmarks

## Resultats actuels (Avril 2026)

| Benchmark | Score | Notes |
|-----------|-------|-------|
| [[MASBENCH-2026-04|MASBENCH]] | +27pp delta (non-pondere) | Regression parallel -6pp |
| [[BigCodeBench-Hard|BigCodeBench Hard]] | v1: 37.2%, **v4: 45.9%** (+8.7pp) | Repair + escalation + model selection |
| [[Routing-GT|Routing GT]] | **93.3%** (56/60) | kNN primary |
| [[TopologyBench|TopologyBench]] | 94.0% mean (9/9) | 4.3pp spread |
| HumanEval+ | 89.6% (147/164) | **NE PAS benchmarker** (sature) |

## Regles de benchmark

> [!danger] NE PAS benchmarker
> - **HumanEval+** : sature, mesure le LLM pas le framework
> - **MBPP+** : idem
> - **GSM8K** : idem

> [!success] A benchmarker
> - **BigCodeBench** : non-sature, ICLR '25
> - **routing_gt** : 50 taches, instantane
> - **ablation** : mesure l'impact de chaque composant
> - **MASBENCH** : delta framework (bare vs SAGE)
