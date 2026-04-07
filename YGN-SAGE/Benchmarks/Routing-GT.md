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

## Resultat

| Routeur | Accuracy | Status |
|---------|----------|--------|
| **kNN** | **93.3%** (56/60) | Primaire |
| SystemRouter (Rust) | 88% | Fallback |
| ComplexityRouter | 45% (27/60) | **DEAD CODE** |
| ShadowRouter | 49.6% divergence | **DEPRECATED** |

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
