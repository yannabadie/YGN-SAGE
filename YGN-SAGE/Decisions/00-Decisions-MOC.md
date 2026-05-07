---
title: Decisions MOC
type: moc
tags:
  - decisions
  - moc
updated: 2026-04-22
---

# Architecture Decision Records

| ADR | Decision | Status |
|-----|----------|--------|
| [[ADR-001-kNN-over-Heuristic]] | kNN plutot que heuristic pour le routing — archive snapshot, was `evidence_pending`; historic accuracy figures non-autoritative. Current status: `routing.knn_92pct` `delivered` floor ≥50/60 LOO-CV in `docs/CLAIMS.yaml`. | accepte |
| [[ADR-002-Rust-First]] | Rust pour le core, Python pour l'orchestration | accepte |
| [[ADR-003-No-HumanEval]] | Ne pas benchmarker sur HumanEval+/MBPP+/GSM8K | accepte |
| [[ADR-004-Thompson-Sampling]] | Thompson sampling pour le bandit (pas UCB) | accepte |
| [[ADR-005-ShadowRouter-Deprecated]] | Deprecier le ShadowRouter dual-path | accepte |
| [[ADR-006-BigCodeBench-Limits]] | BigCodeBench Hard : topology pas le levier (omega=1.3) | accepte |
| [[ADR-007-F7-Routing]] | F7 routing : floor domain-aware, promotion producer roles | accepte |
| [[ADR-008-PRM-Gate-Domain]] | Z3 PRM uniquement sur math/formal, AVR sur code/general | accepte |
| [[ADR-009-Telemetry-And-Routing-Plumbing]] | Wire-up telemetry + per-model routing + quota-aware health | accepte |
| [[ADR-010-Bypass-Audit-Methodology]] | Methodologie bypass-audit (grep #[pyclass] + empirical validation) | accepte |
| [[ADR-011-Singleton-vs-Factory-Asymmetry]] | Singleton AgentLoop doit re-configurer tout ce que la factory set | accepte |
| [[ADR-012-TopologyController-Rust-Port]] | TopologyController ported: 6 decision paths + state Rust-primary, helpers Python | accepte |
| [[ADR-010-Meta-Harness-Divergence]] | Meta-Harness : divergence vs paper (hyperparam tuner, pas structural search) | accepte |
| [[ADR-013-Wasm-Sandbox-Default]] | Wasm sandbox (RustPython embarque) est le chemin Python par defaut ; SAGE_UNSAFE_UNSANDBOXED supprime ; 40/40 red-team bloque | accepte |
