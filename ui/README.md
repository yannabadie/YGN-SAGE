# Dashboard

Real-time control dashboard for YGN-SAGE agent monitoring. Built with FastAPI (backend) and a single-file HTML frontend.

## Quick Start

```bash
python ui/app.py    # Starts on http://localhost:8000
```

## Components

### `app.py` -- FastAPI Backend

REST API and WebSocket server for the dashboard.

- **WebSocket `/ws`** -- Pushes all `AgentEvent` instances from the EventBus in real-time. Authenticated via bearer token when `SAGE_DASHBOARD_TOKEN` is set. Binds to localhost only (audit fix A2).
- **REST API** -- Endpoints for querying agent state, benchmark results, and configuration. Protected by HTTPBearer authentication when token is configured.
- **DashboardState** -- In-memory state tracking for active agents, routing statistics, and guardrail outcomes.
- **Open dev mode** -- When no `SAGE_DASHBOARD_TOKEN` is set, the dashboard runs without authentication for local development.

### `static/index.html` -- Frontend Dashboard

Single-file dark-theme dashboard built with Tailwind CSS and Chart.js.

**Sections:**

- **Routing S1/S2/S3** -- Live distribution of task complexity tiers with bar charts.
- **Response** -- Current agent response and metadata.
- **Memory 4-tier** -- Status of Working Memory (Tier 0), Episodic (Tier 1), Semantic (Tier 2), and ExoCortex (Tier 3).
- **Guardrails** -- Real-time guardrail check results and block events.
- **Events** -- Scrolling event log fed by the WebSocket connection.
- **Benchmarks** -- Latest benchmark results (HumanEval pass@1, routing accuracy).

## Authentication

Set the `SAGE_DASHBOARD_TOKEN` environment variable to enable authentication. Both REST and WebSocket endpoints require the token. Without it, the dashboard runs in open dev mode.
