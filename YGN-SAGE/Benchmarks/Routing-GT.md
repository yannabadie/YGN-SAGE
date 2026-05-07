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

> **⚠ Non-autoritative / archive snapshot** — Obsidian-vault, historic reference only. Routing accuracy claims (`routing.knn_92pct` / `routing.system_router_88pct`) were `evidence_pending` in this archive snapshot; current authoritative status: `docs/CLAIMS.yaml` (`delivered` floors ≥50/60 + ≥52/60 since 2026-05-07).

## Resultat (historic / non-autoritative archive snapshot)

| Routeur | Accuracy (historic) | Status |
|---------|----------|--------|
| **kNN** | historic 93.3% (56/60) — non-autoritative archive snapshot; `routing.knn_92pct` was `evidence_pending` in this snapshot, current status: `delivered` floor ≥50/60 LOO-CV in `docs/CLAIMS.yaml` | Primaire |
| SystemRouter (Rust) | historic 88% — non-autoritative archive snapshot; `routing.system_router_88pct` was `evidence_pending` in this snapshot, current status: `delivered` floor ≥52/60 in `docs/CLAIMS.yaml` | Fallback |
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
