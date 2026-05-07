---
title: Benchmarks MOC
type: moc
tags:
  - benchmarks
  - moc
updated: 2026-04-23
---

# Benchmarks

## Resultats actuels (Avril 2026)

| Benchmark | Score | Notes |
|-----------|-------|-------|
| [[MASBENCH-2026-04\|MASBENCH]] | +27pp delta (non-pondere) | Regression parallel -6pp |
| [[BigCodeBench-Hard\|BigCodeBench Hard]] | v1: 37.2%, **v4: 45.9%** (+8.7pp) | Repair + escalation + model selection |
| [[Routing-GT\|Routing GT]] | archive snapshot, was `evidence_pending`; current status: `routing.knn_92pct` `delivered` floor ≥50/60 LOO-CV in `docs/CLAIMS.yaml` (historic 93.3% non-autoritative) | kNN primary |
| [[TopologyBench\|TopologyBench]] | 94.0% mean (9/9) | 4.3pp spread |
| SWE-bench Lite (Docker-graded) | **10%** (1/10) v15 Apr 21 | +40% patch-gen rate. Blocker bottleneck = diff-emission quality, pas tool-set (voir 2026-04-22 parity smoke) |
| Diff-context verifier observe-smoke Apr 23 | N=10, 2 PATCH / 8 EMPTY, 2/2 content_mismatch post-fix | `docs/benchmarks/2026-04-23-diff-verifier-observe-smoke/findings.md`. Parser false-negative corrigé par `711008a`. |
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
