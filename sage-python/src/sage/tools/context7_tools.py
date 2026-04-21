"""Context7 agent tool: lookup_library_docs.

C2c (2026-04-21). Closes the gap that the C2b smoke surfaced: the
2026-04-21 ExoCortex audit flagged 0 `search_exocortex` calls across
SWE-bench runs, C2b tried to fix it by surfacing `search_exocortex` in
the prompt with a "library API contracts" framing, but the C2b smoke
(10/10 tasks, 315 tool calls) still hit 0 because the
`search_exocortex` tool's own schema description targets research
domains (MARL / cognitive arch / formal verification / evol comp /
memory systems) — not django/astropy/requests/sqlalchemy library docs.

This module registers a purpose-built tool backed by the Context7
public HTTP API (https://context7.com/api/v2/...). Context7 is designed
specifically for on-demand library-docs lookup and indexes most major
open-source libraries. The tool does `resolve` + `query` in one
invocation so the LLM gets "library name + question → docs snippets"
with a single call.

Configuration:

    CONTEXT7_API_KEY=ctx7sk-...   (preferred)
    CONTEXT7=ctx7sk-...           (legacy fallback)

If neither env var is set, `create_context7_tools()` returns an empty
list and the tool is simply not registered — same no-op pattern as
`search_exocortex` when `SAGE_EXOCORTEX_STORE` is missing. Boot logs a
one-shot WARN so misconfigurations are visible.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from sage.llm._ssl import ssl_verify
from sage.tools.base import Tool

_log = logging.getLogger(__name__)

_CONTEXT7_BASE = "https://context7.com/api/v2"
_RESOLVE_PATH = "/libs/search"
_DOCS_PATH = "/context"
_TIMEOUT_S = 30.0
_MAX_SNIPPETS = 6
_MAX_INFO_ITEMS = 4


def _load_api_key() -> str | None:
    """Return the Context7 API key or None.

    Checks `CONTEXT7_API_KEY` first (the documented env var name), then
    falls back to `CONTEXT7` for users with the legacy config.
    """
    return os.environ.get("CONTEXT7_API_KEY") or os.environ.get("CONTEXT7") or None


def _format_snippets(payload: dict[str, Any]) -> str:
    """Render Context7's JSON response into an LLM-readable string.

    Context7 returns:
        {"codeSnippets": [{"codeTitle": ..., "codeList": [{"code": ...}]}, ...],
         "infoSnippets": [{"content": ...}, ...]}

    We cap output to keep prompts / tool-result blocks manageable.
    """
    lines: list[str] = []

    infos = payload.get("infoSnippets") or []
    if infos:
        lines.append("## Documentation")
        for item in infos[:_MAX_INFO_ITEMS]:
            content = (item.get("content") or "").strip()
            if content:
                lines.append(content)
                lines.append("")

    snippets = payload.get("codeSnippets") or []
    if snippets:
        lines.append("## Code examples")
        for snippet in snippets[:_MAX_SNIPPETS]:
            title = (snippet.get("codeTitle") or "").strip() or "(untitled)"
            lines.append(f"### {title}")
            for code_block in snippet.get("codeList") or []:
                code = (code_block.get("code") or "").strip()
                if code:
                    lines.append("```")
                    lines.append(code)
                    lines.append("```")
            lines.append("")

    if not lines:
        return "No documentation found for this query."
    return "\n".join(lines).rstrip() + "\n"


async def _resolve_library_id(
    client: httpx.AsyncClient,
    api_key: str,
    library_name: str,
    query: str,
) -> str | None:
    """Call Context7's search endpoint and return the top library id."""
    resp = await client.get(
        _CONTEXT7_BASE + _RESOLVE_PATH,
        params={"libraryName": library_name, "query": query},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    body = resp.json()
    results = body.get("results") or []
    if not results:
        return None
    top = results[0]
    return top.get("id") or None


async def _fetch_docs(
    client: httpx.AsyncClient,
    api_key: str,
    library_id: str,
    query: str,
) -> dict[str, Any]:
    """Call Context7's context endpoint and return the raw JSON payload."""
    resp = await client.get(
        _CONTEXT7_BASE + _DOCS_PATH,
        params={"libraryId": library_id, "query": query},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict):
        raise ValueError(f"unexpected Context7 response shape: {type(body).__name__}")
    return body


async def _lookup(api_key: str, library_name: str, query: str) -> str:
    """Resolve the library id then fetch docs; return a formatted string."""
    async with httpx.AsyncClient(verify=ssl_verify(), timeout=_TIMEOUT_S) as client:
        try:
            library_id = await _resolve_library_id(client, api_key, library_name, query)
        except httpx.HTTPError as e:
            return f"Context7 library search failed: {type(e).__name__}: {e}"

        if not library_id:
            return (
                f"Context7 has no indexed library matching {library_name!r}. "
                "Try the official spelling (e.g. 'Next.js' not 'nextjs') or "
                "fall back to `execute_bash` for repo-local exploration."
            )

        try:
            payload = await _fetch_docs(client, api_key, library_id, query)
        except httpx.HTTPError as e:
            return f"Context7 docs query failed ({library_id}): {type(e).__name__}: {e}"
        except (ValueError, json.JSONDecodeError) as e:
            return f"Context7 returned unparseable response for {library_id}: {e}"

    return _format_snippets(payload)


_DESCRIPTION = (
    "Look up OFFICIAL documentation and code examples for a third-party "
    "library or framework via Context7 (context7.com). Use this for "
    "library API contracts, deprecation notices, idiomatic patterns, and "
    "version-specific behavior (e.g., 'How does requests.Response.json "
    "behave on empty body?', 'Django 3.0 FILE_UPLOAD_PERMISSION default', "
    "'astropy Table.write formats'). This tool queries a dedicated "
    "documentation index — it does NOT browse the checked-out repo; use "
    "`execute_bash` (grep/sed/git) for that. Pass the official library "
    "name with punctuation ('Next.js' not 'nextjs') and a specific question."
)


def create_context7_tools(api_key: str | None = None) -> list[Tool]:
    """Create Context7-backed tools if an API key is available.

    Parameters
    ----------
    api_key : str | None, optional
        Bearer token. When None (default), resolves from env:
        `CONTEXT7_API_KEY` then `CONTEXT7`. If still None, returns an
        empty list — the tool is simply not registered, matching the
        ExoCortex no-op pattern.
    """
    key = api_key or _load_api_key()
    if not key:
        _log.info(
            "Context7: CONTEXT7_API_KEY not set — lookup_library_docs not registered"
        )
        return []

    @Tool.define(
        name="lookup_library_docs",
        description=_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "library_name": {
                    "type": "string",
                    "description": (
                        "Official library name with correct punctuation "
                        "(e.g. 'Django', 'Next.js', 'requests', 'astropy')."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Specific question about the library's API or "
                        "behavior. Be precise — 'JWT auth setup with "
                        "Django REST Framework 3.14' beats 'auth'."
                    ),
                },
            },
            "required": ["library_name", "query"],
        },
    )
    async def lookup_library_docs(library_name: str, query: str) -> str:
        return await _lookup(key, library_name, query)

    return [lookup_library_docs]
