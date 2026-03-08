# Tools

Agent tool system providing structured capabilities to the agent runtime.

## Modules

### `registry.py` -- ToolRegistry

Central registry for agent tools. Tools are registered by name with typed input/output schemas. The agent loop queries the registry during the ACT phase to resolve tool calls from LLM output.

### `base.py` -- Tool Base Class

Abstract base class for tool implementations. Defines the interface: `name`, `description`, `parameters` schema, and `execute()` method.

### `builtin.py` -- Built-in Tools

Standard utility tools available to all agents (e.g., code execution, file operations).

### `memory_tools.py` -- AgeMem Tools (7 tools)

Memory-related tools exposed to the agent, split across two tiers:

- **STM (3 tools)** -- Short-term memory operations: store, recall, and list items in working memory (Tier 0 Arrow buffer).
- **LTM (4 tools)** -- Long-term memory operations: store/recall from episodic memory (Tier 1 SQLite), query semantic memory (Tier 2 graph), and search causal chains.

### `exocortex_tools.py` -- ExoCortex Tools (2 tools)

- `search_exocortex` -- Active query against the ExoCortex (Google GenAI File Search API) for research papers and knowledge base content.
- `refresh_knowledge` -- Triggers re-indexing of ExoCortex sources.

### `agent_mgmt.py` -- Agent Management Tools

Tools for dynamic sub-agent creation and lifecycle management within the agent pool.

### `meta.py` -- Meta Tools

Introspection tools allowing agents to query their own capabilities, configuration, and runtime state.

### `generated_tools/` -- Generated Tool Definitions

Directory for dynamically generated or externally defined tool specifications.
