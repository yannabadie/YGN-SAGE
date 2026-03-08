# Guardrails

Three-layer guardrail system for input validation, runtime safety, and output verification.

## Architecture

Guardrails execute at three points in the agent loop:

1. **Input (PERCEIVE phase)** -- Validates the task before any LLM call. Blocks execution if severity is "block".
2. **Runtime (ACT phase)** -- Checks generated code before sandbox execution. Best-effort (does not block).
3. **Output (LEARN phase)** -- Validates the final result before returning. Enforces cost and schema constraints.

All guardrail events are emitted on the EventBus for observability.

## Modules

### `base.py` -- Core Abstractions

- `GuardrailResult` -- Outcome of a guardrail check (passed, severity, message).
- `Guardrail` -- Abstract base class. Subclasses implement `check(context) -> GuardrailResult`.
- `GuardrailPipeline` -- Ordered chain of guardrails. Runs all checks and aggregates results. Wired into the agent system via `boot.py`.

### `builtin.py` -- Built-in Guardrails

- `CostGuardrail` -- Enforces per-task and cumulative cost budgets. Blocks execution when cost exceeds threshold.
- `SchemaGuardrail` -- Validates LLM output against a JSON schema. Added during audit response (Task C10-C13) to catch malformed responses.

## Wiring

The `GuardrailPipeline` is instantiated in `boot.py` and injected into `AgentSystem`. The agent loop calls it at each phase boundary. Blocked results emit `GUARDRAIL_BLOCK` events.
