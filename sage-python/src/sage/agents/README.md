# agents

Multi-agent composition patterns for YGN-SAGE. Provides four primitives for combining agents into higher-order workflows, plus a factory for dynamic agent creation from LLM-generated blueprints.

## Modules

### `sequential.py` -- SequentialAgent

Chains agents in series. Each agent's output becomes the next agent's input. Accepts an optional `shared_state` dict that is accessible to all agents in the chain.

- **Key exports**: `SequentialAgent`, `Runnable` (Protocol)
- **Usage**: `await SequentialAgent(name="pipe", agents=[a, b, c]).run(task)`

### `parallel.py` -- ParallelAgent

Runs multiple agents concurrently via `asyncio.gather` and combines results through a pluggable aggregator function. The default aggregator joins outputs as `[agent_name]: result` blocks.

- **Key exports**: `ParallelAgent`
- **Usage**: `await ParallelAgent(name="fan-out", agents=[a, b], aggregator=fn).run(task)`

### `loop_agent.py` -- LoopAgent

Iterates a single agent in a loop, feeding its output back as input. Terminates when `exit_condition(output)` returns `True` or `max_iterations` is reached.

- **Key exports**: `LoopAgent`
- **Usage**: `await LoopAgent(name="refine", agent=a, max_iterations=5, exit_condition=fn).run(task)`

### `handoff.py` -- Handoff

Transfers control to a specialist agent. Supports an `input_filter` to transform the task before handoff and an `on_handoff` callback for logging or side-effects.

- **Key exports**: `Handoff`, `HandoffResult`
- **Usage**: `result = await Handoff(target=specialist, input_filter=fn).run(task)`

### `factory.py` -- DynamicAgentFactory

Creates `ModelAgent` instances from `AgentBlueprint` dataclasses (parsed from LLM decomposition output) and `ModelProfile` objects. Supports `[CODE]`, `[REASON]`, and `[GENERAL]` tags.

- **Key exports**: `DynamicAgentFactory`, `AgentBlueprint`

## Protocol

All composition agents and their children must satisfy the `Runnable` protocol:

```python
class Runnable(Protocol):
    name: str
    async def run(self, task: str) -> str: ...
```
