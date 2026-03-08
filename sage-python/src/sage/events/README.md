# Events

In-process event bus for agent observability and inter-component communication.

## Modules

### `bus.py` -- EventBus

Central publish-subscribe event system used by all YGN-SAGE components.

**Core API:**

- `emit(event)` -- Publish an `AgentEvent` to all subscribers.
- `subscribe(callback)` -- Register a synchronous callback for all events.
- `stream()` -- Async iterator yielding events in real-time. Used by the dashboard WebSocket endpoint (`/ws`) to push events to the UI.
- `query(phase, last_n)` -- Retrieve recent events filtered by phase (PERCEIVE, THINK, ACT, LEARN) and count.

**Event Types:**

Events follow the `AgentEvent` schema with fields for phase, timestamp, payload, and metadata. Key event types include:

- `PERCEIVE`, `THINK`, `ACT`, `LEARN` -- Agent loop phases
- `ROUTING` -- ComplexityRouter tier selection (S1/S2/S3)
- `GUARDRAIL_CHECK`, `GUARDRAIL_BLOCK` -- Guardrail outcomes
- `BENCH_RESULT` -- Benchmark task completion

## Integration

The EventBus is instantiated once in `boot.py` and shared across all components. The dashboard (`ui/app.py`) consumes events via `stream()` over WebSocket, replacing the earlier JSONL file polling approach.
