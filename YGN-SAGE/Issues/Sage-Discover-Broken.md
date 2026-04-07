---
title: "sage-discover Gateway Cassee"
type: issue
severity: P1
status: partiellement-fixe
tags:
  - issue
  - sage-discover
  - mcp
created: 2026-04-07
---

# sage-discover Gateway Cassee

## Probleme

`sage-discover/mcp_gateway.py` importait `sage.evolution.ebpf_evaluator.EbpfEvaluator`
qui n'existe pas.

## Fixes appliques

- Imports morts supprimes (improvement_log.txt, 3 avril)
- Debug mode desactive par defaut (HOST/DEBUG env vars)

## Problemes restants

1. **Runtime non verifie** : les imports sont fixes mais personne n'a verifie que la gateway demarre effectivement
2. **Workflow MCP chainability** (P2) : `discover_mcp.py` retourne des IDs sans persister → `curate_mcp.py` ne les trouve pas → "paper not found"
3. **Pas d'authentification** sur le endpoint HTTP

## Action requise

```bash
# Verifier que la gateway demarre
cd sage-discover && python -m sage_discover.mcp_gateway
```

Si ca crashe, il y a encore des imports morts ou des dependances manquantes.
