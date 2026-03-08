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

- `CostGuardrail` -- Enforces per-task and cumulative cost budgets. Blocks execution when cost exceeds threshold. Uses API `usage_metadata` (Google Gemini `prompt_token_count`/`candidates_token_count`) when available, falls back to `len(text)//4`.
- `OutputGuardrail` -- Warns when free-text output is empty, too long, or looks like a refusal. **Default in the pipeline** for the common case where the agent returns plain text. Replaces `SchemaGuardrail` as the default output guardrail.
- `SchemaGuardrail` -- Validates LLM output against a JSON schema. Use this instead of `OutputGuardrail` when the agent is expected to return structured JSON with required fields.

## Wiring

The `GuardrailPipeline` is instantiated in `boot.py` and injected into `AgentSystem`. The agent loop calls it at each phase boundary. Blocked results emit `GUARDRAIL_BLOCK` events.
