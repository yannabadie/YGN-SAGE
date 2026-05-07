---
title: Architecture MOC
type: moc
tags:
  - architecture
  - moc
updated: 2026-04-07
---

# Architecture — YGN-SAGE

## These centrale

> **La topologie multi-agent compte plus que le choix du modele.**
> AdaptOrch (arXiv 2602.16873) : Var_topology / Var_model >= 20 sur les taches difficiles.

## 5 Piliers Cognitifs

1. [[Pillar-1-Topology|Topology]] — TopologyEngine 6 chemins, MAP-Elites, CMA-ME, MCTS, 11 templates
2. [[Pillar-2-Tools|Tools]] — Sandbox 3 couches (tree-sitter → Wasm WASI → subprocess), ToolForge
3. [[Pillar-3-Memory|Memory]] — 4 tiers (Arrow STM → SQLite Episodic → Semantic → ExoCortex)
4. [[Pillar-4-Evolution|Evolution]] — MAP-Elites + CMA-ME, 7 operateurs mutation, LLM-as-mutator
5. [[Pillar-5-Strategy|Strategy]] — S1/S2/S3 routing Kahneman, kNN primary (`routing.knn_92pct` `delivered` floor ≥50/60 LOO-CV in `docs/CLAIMS.yaml`; historical 92% provenance only), bandit Thompson

## Pipeline

[[Pipeline|CLASSIFY → DECOMPOSE → TOPOLOGY → ASSIGN → EXECUTE → LEARN]]

## Infrastructure

- [[Provider-Architecture|7 Providers]] — DeepSeek (primaire), Google, OpenAI, xAI, Kimi, MiniMax, OpenRouter
- **Rust core** : ~20K LOC, PyO3 bindings, features: smt, onnx, cognitive, tool-executor
- **Python SDK** : ~31K LOC, 175 modules
- **sage-discover** : Pipeline arXiv → ExoCortex (partiellement casse)
- [[Protocols|Protocoles]] — A2A (3 skills), MCP (run_task + tools dynamiques), Docker multi-stage

## Carte des composants

```
sage-core/          sage-python/              sage-discover/
├─ topology/        ├─ pipeline.py (49KB)     ├─ store.py
├─ routing/         ├─ boot.py (29KB)         ├─ mcp.py
├─ memory/          ├─ agent_loop*.py          ├─ knowledge/
├─ verification/    ├─ topology/               └─ mcp_gateway.py [CASSE]
├─ sandbox/         ├─ evolution/
└─ config/          ├─ memory/
   cards.toml       ├─ providers/ (7)
                    ├─ bench/
                    ├─ verl/ (training)
                    └─ protocols/ (A2A, MCP)
```

## Gap architecture vs realite

| Composant | Documente | Implemente | Integre pipeline |
|-----------|-----------|------------|-----------------|
| TopologyEngine 6-path | oui | oui | oui |
| kNN Router — archive snapshot, was `evidence_pending`; current status: `routing.knn_92pct` `delivered` floor ≥50/60 LOO-CV in `docs/CLAIMS.yaml` (historic 93.3% non-autoritative) | oui | oui | oui |
| SystemRouter Rust | oui | oui | oui |
| Memory 4-tier | oui | oui | **partiel** (consolidation incomplete) |
| Evolution online | oui | oui | **opt-in** (pas defaut) |
| Z3 Quality training | oui | oui | **partiel** (pas wire au bootstrap) |
| Path 6 learned | oui | oui | **opt-in** (SAGE_ENABLE_PATH6=1) |
| sage-discover | oui | **partiel** | **casse** (P1 issues) |
| Sandbox formel | oui | oui | **durci** (pas formellement sur) |
