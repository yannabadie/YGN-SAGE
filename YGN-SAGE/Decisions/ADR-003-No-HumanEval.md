---
title: "ADR-003: No HumanEval Benchmarking"
type: adr
status: accepte
date: 2026-03-01
tags:
  - adr
  - benchmarks
---

# ADR-003: Ne pas benchmarker sur HumanEval+/MBPP+/GSM8K

## Contexte

Les benchmarks satures (HumanEval+ 89.6%) mesurent la qualite du LLM sous-jacent,
pas la valeur ajoutee du framework multi-agent.

## Decision

Benchmarker uniquement sur des benchmarks non-satures qui mesurent le delta framework :
- BigCodeBench Hard (ICLR '25)
- MASBENCH (multi-axe)
- routing_gt (interne)
- ablation (impact par composant)

## Consequences

- Positives : Pas de chiffres trompeurs, focus sur la vraie valeur ajoutee
- Negatives : Moins de chiffres "impressionnants" pour le marketing
- Risques : Difficulte a comparer avec d'autres frameworks qui publient sur HumanEval

## Evidence

- HumanEval+ 89.6% (147/164) — atteint mais non mis en avant
- BigCodeBench Hard 37.8% — non-sature, pertinent
- MASBENCH +27pp — mesure le delta framework
