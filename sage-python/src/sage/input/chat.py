"""Chat-source normalizer.

Per spec decisions (2026-04-21):

* Q1 — bash is OFF by default in chat mode. The default `tools_filter`
  allows `search_exocortex` and read-only memory/context lookup. Users
  opt into bash via `SAGE_CHAT_ALLOW_BASH=1` or (later) a `/shell`
  REPL command.
* Q3 — chat mode does NOT enforce a response format. `response_format`
  stays `TEXT` and the prompt builder skips the format block; the LLM
  picks prose / code / markdown based on the user's request.
"""
from __future__ import annotations

import os

from sage.input.types import ResponseFormat, TaskInput

CHAT_DEFAULT_TOOLS: list[str] = [
    "search_exocortex",
    "refresh_knowledge",
    "search_memory",
    "retrieve_context",
    "summarize_context",
    "filter_context",
    "search_causal_chain",
    "list_active_agents",
]
"""Safe read-only tools enabled by default in chat mode.

Excluded by design: `bash`, `create_python_tool`, `create_bash_tool`,
`store_memory`, `update_memory`, `delete_memory`, `create_agent`,
`call_agent`, `sage_recurse`. These mutate state, spawn processes, or
recurse — all opt-in only.
"""

_BASH_OPT_IN_ENV = "SAGE_CHAT_ALLOW_BASH"


def _bash_allowed() -> bool:
    """True iff the caller set SAGE_CHAT_ALLOW_BASH to a truthy value."""
    value = os.environ.get(_BASH_OPT_IN_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_chat(user_message: str) -> TaskInput:
    """Convert a raw chat message into a `TaskInput`.

    Parameters
    ----------
    user_message:
        The exact text the user typed. Whitespace and case are
        preserved; no interpretation is applied here — the pipeline
        downstream decides what to do with it.
    """
    tools_filter: list[str] | None
    if _bash_allowed():
        tools_filter = None
    else:
        tools_filter = list(CHAT_DEFAULT_TOOLS)

    return TaskInput(
        prompt=user_message,
        response_format=ResponseFormat.TEXT,
        tools_filter=tools_filter,
        source="chat",
    )
