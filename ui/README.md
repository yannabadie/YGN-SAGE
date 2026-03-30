# YGN-SAGE Agent UI

Real-time cognitive agent interface for YGN-SAGE. Built with FastAPI (backend) and modular vanilla JS (frontend).

## Quick Start

```bash
python ui/app.py    # Starts on http://localhost:8000
```

## Architecture

**Backend** (`app.py`): FastAPI with 15 REST/SSE endpoints + WebSocket event stream.

**Frontend** (`static/`): Modular ES modules, zero build step. Tailwind CSS CDN + Chart.js + Cytoscape.js + ELK.js + Marked.js.

### Tabs

| Tab | Module | Description |
|-----|--------|-------------|
| **Chat** | `js/chat.js` | Conversational interface with SSE streaming, markdown rendering, multi-turn history |
| **Dashboard** | `js/dashboard.js` | Control panel: task input, response, memory tiers, guardrails, routing pipeline, stats |
| **Topology** | `js/topology.js` | Interactive DAG graph (Cytoscape.js + ELK.js) with 3-flow edge model (Control/Message/State) |
| **Providers** | `js/providers.js` | Provider health cards with circuit breaker state, latency, error rates |

### Always Visible

| Component | Module | Description |
|-----------|--------|-------------|
| **Event Stream** | `js/events.js` | Real-time event log from WebSocket, filterable by phase |
| **Header** | `index.html` | S1/S2/S3 indicators, step count, cost, model, WebSocket status |

### Shared Infrastructure

| Module | Purpose |
|--------|---------|
| `js/state.js` | Reactive pub/sub state store — all modules subscribe |
| `js/ws.js` | WebSocket manager with auto-reconnect |
| `css/sage.css` | Shared styles, animations, component classes |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve the SPA shell |
| GET | `/api/state` | Dashboard summary stats |
| POST | `/api/task` | Submit task to queue |
| GET | `/api/tasks` | List task history |
| POST | `/api/stop` | Cancel current task |
| POST | `/api/reset` | Reset all state |
| GET | `/api/providers` | List available models |
| POST | `/api/benchmark` | Launch benchmark |
| GET | `/api/memory/stats` | 4-tier memory stats |
| GET | `/api/topology` | Agent pool topology |
| GET | `/api/evolution` | MAP-Elites evolution state |
| POST | `/api/chat/stream` | SSE streaming chat response |
| GET | `/api/topology/graph` | Cytoscape.js-compatible topology graph |
| GET | `/api/providers/health` | Provider health + circuit breaker |
| GET | `/api/routing/pipeline` | 4-stage routing pipeline state |

## Research Backing

- **3-flow edge model**: MASFactory (arXiv 2603.06007) — Control + Message + State edge types
- **ELK Sugiyama layout**: Hierarchical DAG rendering for agent pipelines
- **Causal event tracing**: AgentTrace (arXiv 2603.14688) — event linking in stream
- **kNN routing visualization**: arXiv 2505.12601 — 92% GT accuracy, 4-stage pipeline
- **Streaming-first**: Vercel AI SDK 6 pattern — SSE for chat, WebSocket for events

## Authentication

Set `SAGE_DASHBOARD_TOKEN` env var. Both REST and WebSocket require the token. Without it: open dev mode.
