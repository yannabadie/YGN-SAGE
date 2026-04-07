---
title: Protocoles (A2A, MCP, Docker)
type: architecture
tags:
  - architecture
  - protocols
  - a2a
  - mcp
  - docker
updated: 2026-04-07
---

# Protocoles — A2A, MCP, Docker

> [!info] Dashboard UI
> Le dashboard web (`ui/`) n'est pas prioritaire.
> L'interface utilisateur se fera sur **pi-mono** avec une version custom.

## A2A (Google Agent-to-Agent) — a2a-sdk 0.3.25

**Fichier** : `sage-python/src/sage/protocols/a2a_server.py`
**SDK** : `a2a-sdk[http-server]>=0.3` (v1.0 n'existe pas sur PyPI)
**Agent version** : 0.2.0

### 3 Skills exposees via AgentCard

| Skill | Description |
|-------|-------------|
| General Task Execution | Pipeline cognitif complet, auto S1/S2/S3 |
| Code Generation & Analysis | Generation, review, fix avec verification formelle |
| Knowledge Retrieval | Recherche ExoCortex (500+ papers) |

### Fonctionnement

- Wrappe le pipeline SAGE comme `AgentExecutor`
- Streaming per-node via `TaskArtifactUpdateEvent`
- Lifecycle via `TaskStatusUpdateEvent`
- Support annulation (`cancel()`)
- Fallback synchrone si streaming indisponible

## MCP Server

**Fichier** : `sage-python/src/sage/protocols/mcp_server.py`

### Outils exposes

| Outil | Description |
|-------|-------------|
| `run_task` | Meta-tool : execute via pipeline cognitif complet (auto/S1/S2/S3) |
| Tous les outils du ToolRegistry | Enregistrement dynamique de chaque outil comme MCP tool |

### Ressource

- `sage://events/recent` : les 20 derniers evenements agents (lecture seule)

### Implementation

- FastMCP server wrapper
- Wrapping dynamique des outils (`tool_registry._tools`)
- Resultats serialises JSON

## Unified Serve CLI

**Fichier** : `sage-python/src/sage/protocols/serve.py`

```bash
python -m sage.protocols.serve --mcp --mcp-port 8001 --a2a --a2a-port 8002 --host 0.0.0.0
```

- Boot SAGE une seule fois, partage entre MCP et A2A
- Requiert `pip install ygn-sage[mcp]` et/ou `pip install ygn-sage[a2a]`
- MCP sur streamable-http, A2A sur uvicorn

## Docker

**Fichiers** : `docker-compose.yml` (3.9) + `Dockerfile` (multi-stage)

### Services

| Service | Ports | Commande |
|---------|-------|----------|
| sage | 8000 (dashboard), 8001 (MCP), 8002 (A2A) | `python -m sage.protocols.serve --mcp --a2a` |
| dashboard (profil `full`) | 8080→8000 | `python ui/app.py` |

### Build multi-stage

1. **Builder** (Rust slim) : compile sage-core via maturin
2. **Runtime** (Python 3.13-slim) : installe wheels + deps, copie app
3. Optimise pour Google Cloud Run (serverless, scale-to-zero)

### Variables d'environnement

API keys (Google, OpenAI, DeepSeek, Grok, Kimi, MiniMax) + dashboard token.
Volume : `sage-data:/root/.sage`
