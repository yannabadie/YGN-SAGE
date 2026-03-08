# Routing

Dynamic model routing with capability constraints and performance feedback.

## Modules

### `dynamic.py` -- DynamicRouter

Capability-constrained model selection with runtime feedback. Given a task and its complexity tier (from ComplexityRouter), DynamicRouter selects the best available model based on:

- **Capability matrix** -- Each provider declares capabilities (coding, reasoning, formal verification). The router filters to models that satisfy the task's requirements.
- **Performance feedback** -- Historical success rates, latency, and cost are factored into scoring. Poor-performing models are deprioritized.
- **Budget constraints** -- Respects per-task and cumulative cost limits from CostTracker.

**Safety:** Handles edge cases where all models are filtered out (fixed IndexError on empty scored list, identified during audit). Falls back gracefully when no model matches constraints.

## Integration

The DynamicRouter sits between the ComplexityRouter (which determines S1/S2/S3 tier) and the LLM providers (which execute the call). The flow is:

```
Task -> ComplexityRouter (tier) -> DynamicRouter (model) -> LLM Provider (execution)
```

Feedback from task outcomes flows back to update the router's scoring model.
