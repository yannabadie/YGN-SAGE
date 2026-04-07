---
title: Decisions MOC
type: moc
tags:
  - decisions
  - moc
updated: 2026-04-07
---

# Architecture Decision Records

| ADR | Decision | Status |
|-----|----------|--------|
| [[ADR-001-kNN-over-Heuristic]] | kNN (93.3%) plutot que heuristic (45%) pour le routing | accepte |
| [[ADR-002-Rust-First]] | Rust pour le core, Python pour l'orchestration | accepte |
| [[ADR-003-No-HumanEval]] | Ne pas benchmarker sur HumanEval+/MBPP+/GSM8K | accepte |
| [[ADR-004-Thompson-Sampling]] | Thompson sampling pour le bandit (pas UCB) | accepte |
| [[ADR-005-ShadowRouter-Deprecated]] | Deprecier le ShadowRouter dual-path | accepte |
| [[ADR-006-BigCodeBench-Limits]] | BigCodeBench Hard : topology pas le levier (omega=1.3) | accepte |
