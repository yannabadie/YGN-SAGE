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
    parser.add_argument(
        "--host",
        default=None,
        help=(
            "Bind address. Default 127.0.0.1. Set SAGE_PROTOCOL_BIND_ALL=1 "
            "to default to 0.0.0.0 (public). Explicit --host overrides env."
        ),
    )
    args = parser.parse_args()

    if not args.mcp and not args.a2a:
        parser.error("Specify at least one of --mcp or --a2a")

    # AUDIT3 / A19: localhost-by-default + security warning when public
    # bind lacks a bearer token. See sage.protocols.auth.
    from sage.protocols.auth import resolve_bind_host, warn_insecure_bind
    args.host = resolve_bind_host(args.host)
    warn_insecure_bind(args.host)

    # Boot SAGE system
    from sage.boot import boot_agent_system
    system = boot_agent_system()

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
        # A19 / AUDIT.md §6 S7 — build the streamable HTTP app manually
        # so we can install bearer-token middleware before serving.
        # FastMCP.run() would build + serve atomically, leaving no hook.
        import uvicorn
        from sage.protocols.auth import bearer_token_from_env, require_bearer_middleware
        from starlette.middleware.base import BaseHTTPMiddleware
        mcp_app = server.streamable_http_app()
        if bearer_token_from_env() is not None:
            mcp_app.add_middleware(BaseHTTPMiddleware, dispatch=require_bearer_middleware())
            _log.info("MCP bearer-token middleware installed (SAGE_PROTOCOL_BEARER_TOKEN set)")
        print(f"MCP server starting on {args.host}:{args.mcp_port}")
        uvicorn.run(mcp_app, host=args.host, port=args.mcp_port)

    if args.a2a:
        from sage.protocols import HAS_A2A
        if not HAS_A2A:
            print("Error: a2a-sdk not installed. Run: pip install ygn-sage[a2a]")
            return
        from sage.protocols.a2a_server import create_a2a_app
        import uvicorn
        app = create_a2a_app(
            agent_loop=system.agent_loop,
            pipeline=system.pipeline,  # enables streaming + cancellation
            tool_registry=system.tool_registry,
            event_bus=system.event_bus,
            url=f"http://{args.host}:{args.a2a_port}",
        )
        print(f"A2A agent starting on {args.host}:{args.a2a_port}")
        uvicorn.run(app, host=args.host, port=args.a2a_port)


if __name__ == "__main__":
    main()
