---
title: "Routing GT"
type: benchmark
benchmark: routing_gt
date: 2026-04-01
tags:
  - benchmark
  - routing
---

# Routing Ground Truth

> **⚠ Non-autoritative / archive** — Obsidian-vault snapshot, historic reference only. Authoritative status: `docs/CLAIMS.yaml` — routing accuracy claims (`routing.knn_92pct` / `routing.system_router_88pct`) are `evidence_pending`.

## Resultat (historic / non-autoritative)

| Routeur | Accuracy (historic) | Status |
|---------|----------|--------|
| **kNN** | historic 93.3% (56/60) — non-autoritative; `routing.knn_92pct` `evidence_pending` | Primaire |
| SystemRouter (Rust) | historic 88% — non-autoritative; `routing.system_router_88pct` `evidence_pending` | Fallback |
| ComplexityRouter | historic 45% (27/60) — non-autoritative; `evidence_pending` | Priority-3 emergency fallback only, NOT dead code (AUDIT2 2026-04-24 corrected) |
| ShadowRouter | historic 49.6% divergence — non-autoritative | **DEPRECATED** |

## Commande

```bash
python -m sage.bench --type routing_gt
```

## Notes

- 60 taches etiquetees manuellement (S1/S2/S3)
- 60 exemplaires kNN, Rust ONNX acceleration
- Embeddings : arctic-embed-m 768-dim
- Execution instantanee (<1s)
- Benchmark le plus fiable du projet (pas de variabilite LLM)
