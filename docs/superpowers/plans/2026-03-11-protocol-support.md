# Protocol Support (MCP + A2A) — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MCP server and A2A agent protocol support to YGN-SAGE, enabling interoperability with Claude Desktop, LangGraph, CrewAI, Google ADK, and any MCP/A2A client. Both protocols behind optional feature flags.

**Architecture:** Create `sage-python/src/sage/protocols/` package with two thin wrappers: `mcp_server.py` (exposes ToolRegistry + agent-as-tool via MCP) and `a2a_server.py` (wraps AgentLoop as A2A AgentExecutor). Both mount on existing FastAPI or run standalone. Feature-flagged via optional deps.

**Tech Stack:** Python 3.12, `mcp>=1.7` (MCP Python SDK), `a2a-sdk[http-server]` (A2A Python SDK), FastAPI/Starlette

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `sage-python/src/sage/protocols/__init__.py` | Create | Feature detection (HAS_MCP, HAS_A2A) |
| `sage-python/src/sage/protocols/mcp_server.py` | Create | MCP server: tool registration, agent-as-tool, resources |
| `sage-python/src/sage/protocols/a2a_server.py` | Create | A2A server: AgentExecutor, AgentCard, skill auto-discovery |
| `sage-python/src/sage/protocols/serve.py` | Create | Unified CLI: `python -m sage.protocols.serve --mcp --a2a` |
| `sage-python/pyproject.toml` | Modify | Add mcp, a2a, protocols optional dep groups |
| `sage-python/tests/test_mcp_server.py` | Create | MCP server unit tests |
| `sage-python/tests/test_a2a_server.py` | Create | A2A server unit tests |

---

### Task 1: Add Optional Dependencies

**Files:**
- Modify: `sage-python/pyproject.toml`

- [ ] **Step 1: Add optional dependency groups**

In `pyproject.toml` `[project.optional-dependencies]`:

```toml
mcp = ["mcp>=1.7"]
a2a = ["a2a-sdk[http-server]>=0.3"]
protocols = ["ygn-sage[mcp,a2a]"]
```

- [ ] **Step 2: Update `all` group to include protocols**

```toml
all = [
    "ygn-sage[google,openai,arrow,z3,embeddings,ui,bench,mcp,a2a]",
]
```

- [ ] **Step 3: Commit**

```bash
git add sage-python/pyproject.toml
git commit -m "feat(protocols): add mcp and a2a optional dependency groups"
```

---

### Task 2: Create Protocol Package with Feature Detection

**Files:**
- Create: `sage-python/src/sage/protocols/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# sage-python/tests/test_protocol_init.py
def test_protocol_package_importable():
    import sage.protocols
    assert hasattr(sage.protocols, "HAS_MCP")
    assert hasattr(sage.protocols, "HAS_A2A")
    assert isinstance(sage.protocols.HAS_MCP, bool)
    assert isinstance(sage.protocols.HAS_A2A, bool)
```

- [ ] **Step 2: Create the package**

```python
# sage-python/src/sage/protocols/__init__.py
"""Protocol support for MCP and A2A interoperability.

Both protocols are behind optional dependencies:
  pip install ygn-sage[mcp]        # MCP server
  pip install ygn-sage[a2a]        # A2A agent
  pip install ygn-sage[protocols]  # Both
"""

try:
    import mcp  # noqa: F401
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

try:
    import a2a  # noqa: F401
    HAS_A2A = True
except ImportError:
    HAS_A2A = False
```

- [ ] **Step 3: Commit**

```bash
git add sage-python/src/sage/protocols/__init__.py sage-python/tests/test_protocol_init.py
git commit -m "feat(protocols): create protocols package with MCP/A2A feature detection"
```

---

### Task 3: MCP Server Wrapper

**Files:**
- Create: `sage-python/src/sage/protocols/mcp_server.py`
- Create: `sage-python/tests/test_mcp_server.py`

- [ ] **Step 1: Write the failing test**

```python
# sage-python/tests/test_mcp_server.py
"""Test MCP server wrapper."""
import pytest

pytest.importorskip("mcp", reason="mcp package not installed")

def test_create_mcp_server_returns_server():
    from sage.protocols.mcp_server import create_mcp_server
    from sage.tools.registry import ToolRegistry
    registry = ToolRegistry()
    server = create_mcp_server(tool_registry=registry)
    assert server is not None

def test_mcp_server_registers_run_task_tool():
    from sage.protocols.mcp_server import create_mcp_server
    from sage.tools.registry import ToolRegistry
    registry = ToolRegistry()
    server = create_mcp_server(tool_registry=registry)
    # The server should have at minimum the run_task meta-tool
    tools = server.list_tools()
    tool_names = [t.name for t in tools]
    assert "run_task" in tool_names
```

- [ ] **Step 2: Implement MCP server wrapper**

```python
# sage-python/src/sage/protocols/mcp_server.py
"""MCP (Model Context Protocol) server wrapper for YGN-SAGE.

Exposes YGN-SAGE tools via MCP protocol, enabling interoperability
with Claude Desktop, Cursor, and any MCP client.

Usage:
    from sage.protocols.mcp_server import create_mcp_server
    server = create_mcp_server(tool_registry, agent_loop, event_bus)
    server.run(transport="streamable-http", host="0.0.0.0", port=8001)
"""
from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

_log = logging.getLogger(__name__)


def create_mcp_server(
    tool_registry: Any | None = None,
    agent_loop: Any | None = None,
    event_bus: Any | None = None,
) -> FastMCP:
    """Create an MCP server exposing YGN-SAGE capabilities.

    Parameters
    ----------
    tool_registry:
        ToolRegistry instance. All registered tools are exposed via MCP.
    agent_loop:
        AgentLoop instance. If provided, adds a `run_task` meta-tool.
    event_bus:
        EventBus instance. If provided, exposes recent events as MCP resource.
    """
    server = FastMCP(
        "YGN-SAGE",
        json_response=True,
    )

    # Meta-tool: run a task through the full SAGE pipeline
    @server.tool()
    async def run_task(task: str, system: int = 0) -> str:
        """Run a task through the YGN-SAGE cognitive pipeline.

        Args:
            task: The task to execute.
            system: Force cognitive system (0=auto, 1=S1, 2=S2, 3=S3).

        Returns:
            The agent's response.
        """
        if agent_loop is None:
            return "Error: AgentLoop not configured. Start SAGE with full boot sequence."
        try:
            result = await agent_loop.run(task)
            return result if isinstance(result, str) else str(result)
        except Exception as exc:
            return f"Error: {exc}"

    # Register all tools from ToolRegistry
    if tool_registry is not None:
        for name, tool in tool_registry._tools.items():
            _register_tool_as_mcp(server, name, tool)

    # Expose EventBus as resource (read-only)
    if event_bus is not None:
        @server.resource("sage://events/recent")
        async def recent_events() -> str:
            """Recent agent events (last 20)."""
            import json
            events = event_bus.query(last_n=20)
            return json.dumps([
                {"phase": e.phase, "content": e.content[:200]}
                for e in events
            ], indent=2)

    _log.info("MCP server created with %d tools", len(server._tool_manager._tools))
    return server


def _register_tool_as_mcp(server: FastMCP, name: str, tool: Any) -> None:
    """Register a SAGE tool as an MCP tool."""
    try:
        spec = tool.spec if hasattr(tool, "spec") else None
        description = spec.description if spec else f"SAGE tool: {name}"

        @server.tool(name=name, description=description)
        async def _wrapper(**kwargs: Any) -> str:
            try:
                handler = tool.handler if hasattr(tool, "handler") else tool
                result = handler(**kwargs)
                if hasattr(result, "__await__"):
                    result = await result
                return str(result)
            except Exception as exc:
                return f"Error in {name}: {exc}"
    except Exception as exc:
        _log.debug("Failed to register tool %s as MCP: %s", name, exc)
```

- [ ] **Step 3: Run tests**

```bash
cd sage-python && pip install mcp>=1.7 && python -m pytest tests/test_mcp_server.py -v --tb=short
```

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/protocols/mcp_server.py sage-python/tests/test_mcp_server.py
git commit -m "feat(protocols): add MCP server wrapper exposing ToolRegistry + run_task meta-tool"
```

---

### Task 4: A2A Agent Wrapper

**Files:**
- Create: `sage-python/src/sage/protocols/a2a_server.py`
- Create: `sage-python/tests/test_a2a_server.py`

- [ ] **Step 1: Write the failing test**

```python
# sage-python/tests/test_a2a_server.py
"""Test A2A agent wrapper."""
import pytest

pytest.importorskip("a2a", reason="a2a-sdk not installed")

def test_build_agent_card():
    from sage.protocols.a2a_server import build_agent_card
    card = build_agent_card(name="test-sage", url="http://localhost:8002")
    assert card.name == "test-sage"
    assert len(card.skills) >= 1  # At least the "general" skill
```

- [ ] **Step 2: Implement A2A agent wrapper**

```python
# sage-python/src/sage/protocols/a2a_server.py
"""A2A (Agent-to-Agent) protocol server for YGN-SAGE.

Exposes YGN-SAGE as an A2A-compatible agent, enabling delegation from
Google ADK, LangGraph, CrewAI, or any A2A client.

Usage:
    from sage.protocols.a2a_server import create_a2a_app
    app = create_a2a_app(agent_loop, tool_registry, event_bus)
    uvicorn.run(app, host="0.0.0.0", port=8002)
"""
from __future__ import annotations

import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils.helpers import new_agent_text_message

_log = logging.getLogger(__name__)


class SageAgentExecutor(AgentExecutor):
    """Wraps YGN-SAGE AgentLoop as an A2A AgentExecutor."""

    def __init__(self, agent_loop: Any | None = None):
        self._agent_loop = agent_loop

    async def execute(self, context: Any, event_queue: Any) -> None:
        """Execute a task via the SAGE cognitive pipeline."""
        # Extract task text from A2A message
        task_text = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, "text"):
                    task_text += part.text

        if not task_text:
            await event_queue.enqueue_event(
                new_agent_text_message("Error: empty task")
            )
            return

        if self._agent_loop is None:
            await event_queue.enqueue_event(
                new_agent_text_message("Error: AgentLoop not configured")
            )
            return

        try:
            result = await self._agent_loop.run(task_text)
            text = result if isinstance(result, str) else str(result)
            await event_queue.enqueue_event(new_agent_text_message(text))
        except Exception as exc:
            _log.error("A2A execution error: %s", exc)
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: {exc}")
            )

    async def cancel(self, context: Any, event_queue: Any) -> None:
        """Cancel not yet supported."""
        raise NotImplementedError("Task cancellation not yet supported")


def build_agent_card(
    name: str = "YGN-SAGE",
    url: str = "http://localhost:8002",
    description: str | None = None,
) -> AgentCard:
    """Build an A2A AgentCard describing SAGE capabilities."""
    return AgentCard(
        name=name,
        description=description or (
            "YGN-SAGE: Self-Adaptive Generation Engine with cognitive routing "
            "(S1/S2/S3), formal verification, evolutionary topology search, "
            "4-tier memory, and 7-provider model selection."
        ),
        url=url,
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="general",
                name="General Task Execution",
                description="Execute any task through the cognitive pipeline with automatic S1/S2/S3 routing.",
                tags=["general", "coding", "reasoning", "math"],
                examples=["Write a Python function", "Prove sqrt(2) is irrational"],
            ),
            AgentSkill(
                id="code",
                name="Code Generation & Analysis",
                description="Generate, review, and fix code with formal verification.",
                tags=["code", "python", "rust", "review"],
                examples=["Implement a binary search tree", "Fix this bug"],
            ),
            AgentSkill(
                id="research",
                name="Knowledge Retrieval",
                description="Search ExoCortex research store (500+ papers) and answer questions.",
                tags=["research", "papers", "knowledge"],
                examples=["What is MAP-Elites?", "Summarize PSRO"],
            ),
        ],
    )


def create_a2a_app(
    agent_loop: Any | None = None,
    tool_registry: Any | None = None,
    event_bus: Any | None = None,
    name: str = "YGN-SAGE",
    url: str = "http://localhost:8002",
) -> A2AStarletteApplication:
    """Create an A2A Starlette application wrapping SAGE."""
    agent_card = build_agent_card(name=name, url=url)
    executor = SageAgentExecutor(agent_loop=agent_loop)
    task_store = InMemoryTaskStore()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )
    _log.info("A2A server created: %s at %s", name, url)
    return app
```

- [ ] **Step 3: Run tests**

```bash
cd sage-python && pip install "a2a-sdk[http-server]" && python -m pytest tests/test_a2a_server.py -v --tb=short
```

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/protocols/a2a_server.py sage-python/tests/test_a2a_server.py
git commit -m "feat(protocols): add A2A agent wrapper with AgentCard + 3 skills"
```

---

### Task 5: Unified Serve CLI

**Files:**
- Create: `sage-python/src/sage/protocols/serve.py`

- [ ] **Step 1: Create unified serve command**

```python
# sage-python/src/sage/protocols/serve.py
"""Unified protocol server CLI.

Usage:
    python -m sage.protocols.serve --mcp --a2a --dashboard
    python -m sage.protocols.serve --mcp --port 8001
    python -m sage.protocols.serve --a2a --port 8002
"""
from __future__ import annotations

import argparse
import asyncio
import logging

_log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="YGN-SAGE Protocol Server")
    parser.add_argument("--mcp", action="store_true", help="Start MCP server")
    parser.add_argument("--a2a", action="store_true", help="Start A2A agent server")
    parser.add_argument("--mcp-port", type=int, default=8001, help="MCP server port")
    parser.add_argument("--a2a-port", type=int, default=8002, help="A2A server port")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    args = parser.parse_args()

    if not args.mcp and not args.a2a:
        parser.error("Specify at least one of --mcp or --a2a")

    # Boot SAGE system
    from sage.boot import boot
    system = asyncio.run(boot())

    if args.mcp:
        from sage.protocols import HAS_MCP
        if not HAS_MCP:
            print("Error: mcp package not installed. Run: pip install ygn-sage[mcp]")
            return
        from sage.protocols.mcp_server import create_mcp_server
        server = create_mcp_server(
            tool_registry=system.tool_registry,
            agent_loop=system.agent_loop,
            event_bus=system.event_bus,
        )
        print(f"MCP server starting on {args.host}:{args.mcp_port}")
        server.run(transport="streamable-http", host=args.host, port=args.mcp_port)

    if args.a2a:
        from sage.protocols import HAS_A2A
        if not HAS_A2A:
            print("Error: a2a-sdk not installed. Run: pip install ygn-sage[a2a]")
            return
        from sage.protocols.a2a_server import create_a2a_app
        import uvicorn
        app = create_a2a_app(
            agent_loop=system.agent_loop,
            tool_registry=system.tool_registry,
            event_bus=system.event_bus,
            url=f"http://{args.host}:{args.a2a_port}",
        )
        print(f"A2A agent starting on {args.host}:{args.a2a_port}")
        uvicorn.run(app, host=args.host, port=args.a2a_port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add sage-python/src/sage/protocols/serve.py
git commit -m "feat(protocols): add unified serve CLI for MCP + A2A servers"
```

---

### Task 6: Update Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add protocols section to CLAUDE.md**

After the Dashboard section:

```markdown
### Protocols (MCP + A2A)
- `protocols/__init__.py` - Feature detection (HAS_MCP, HAS_A2A)
- `protocols/mcp_server.py` - MCP server: ToolRegistry → MCP tools, `run_task` meta-tool, EventBus resource
- `protocols/a2a_server.py` - A2A agent: AgentLoop → AgentExecutor, AgentCard with 3 skills (general, code, research)
- `protocols/serve.py` - Unified CLI: `python -m sage.protocols.serve --mcp --a2a`
```

And in Development Commands:

```markdown
### Protocol Servers
```bash
pip install ygn-sage[protocols]                           # Install MCP + A2A deps
python -m sage.protocols.serve --mcp --mcp-port 8001     # MCP server
python -m sage.protocols.serve --a2a --a2a-port 8002     # A2A agent
python -m sage.protocols.serve --mcp --a2a               # Both
```
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add MCP + A2A protocol support to CLAUDE.md"
```

---

## Summary

| Task | What | LOC | Risk |
|------|------|-----|------|
| 1. Optional deps | pyproject.toml | ~5 | None |
| 2. Feature detection | protocols/__init__.py | ~15 | None |
| 3. MCP server | mcp_server.py + test | ~120 + 30 | Low |
| 4. A2A agent | a2a_server.py + test | ~150 + 20 | Low |
| 5. Unified CLI | serve.py | ~60 | Low |
| 6. Documentation | CLAUDE.md | ~20 | None |

**Total:** ~420 LOC production + ~50 LOC tests. Both protocols behind optional deps — zero impact on existing code.
