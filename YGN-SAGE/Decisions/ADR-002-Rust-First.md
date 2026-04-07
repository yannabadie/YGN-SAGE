---
title: "ADR-002: Rust First, Python Tolerant"
type: adr
status: accepte
date: 2026-01-15
tags:
  - adr
  - architecture
---

# ADR-002: Rust First, Python Tolerant

## Contexte

Le framework doit etre performant (topology engine, SMT verification, memory) tout en restant
accessible (orchestration, providers, benchmarks).

## Decision

Performance-critical en Rust (sage-core via PyO3), Python pour l'orchestration.

## Alternatives considerees

| Option | Pour | Contre |
|--------|------|--------|
| **Rust core + Python SDK** | Performance + accessibilite | Deux langages a maintenir |
| Pure Python | Un seul langage | Trop lent pour SMT, topology, memory |
| Pure Rust | Performance maximale | Inaccessible pour la communaute |

## Consequences

- Positives : sub-0.1ms SMT proofs, zero-copy Arrow memory, SIMD operations
- Negatives : Complexite de build (maturin, PyO3), deux sets de tests
- Risques : Fragmentation de la logique entre Rust et Python (ex: routing en Rust ET Python)

## Evidence

- sage-core : ~20K LOC Rust, 429 tests (avril 7 2026)
- sage-python : ~31K LOC Python, 2001 tests, 0 failures (avril 7 2026)
- PyO3 bindings fonctionnels
