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
    # Research + library lookup (post-C2c chat pivot)
    "search_exocortex",
    "refresh_knowledge",
    "lookup_library_docs",
    # Memory / context inspection (read-only)
    "search_memory",
    "retrieve_context",
    "summarize_context",
    "filter_context",
    "search_causal_chain",
    "list_active_agents",
    # P0.1 (2026-04-22 audit remediation): typed repo tools are safe
    # by construction — path-jail, argv-list execution, scrubbed env,
    # output caps. Chat users asking "what's in my repo?" / "run my
    # tests" / "show the diff" get them without needing
    # SAGE_CHAT_ALLOW_BASH. Full-power bash stays gated by the
    # existing opt-in env var / /shell REPL command.
    "read_file",
    "search_repo",
    "list_files",
    "run_tests",
    "apply_patch",
    "git_diff",
]
"""Safe read-only tools enabled by default in chat mode.

Excluded by design: `bash`, `create_python_tool`, `create_bash_tool`,
`store_memory`, `update_memory`, `delete_memory`, `create_agent`,
`call_agent`, `sage_recurse`. These mutate state, spawn processes, or
recurse — all opt-in only.

Added 2026-04-22 (post C2c benchmark-validation pivot): `lookup_library_docs`
joins the chat-default allowlist. The C2b/C2c/C2b-resmoke triple proved the
tool's benchmark impact cannot be cleanly measured on our current SWE-bench
Lite / BCB slices without a variance-controlled gate we don't yet want to
fund (~11 h + $60-120 per validation pass). Instead, we ship it
opportunistically: chat users asking "how does Django X work" / "what changed
in requests 2.28" get the tool for free with zero claim about benchmark lift.
If a future cross-library benchmark (SWE-bench Pro, synthetic) justifies it,
the plan file `crystalline-crafting-shore.md` Step 4 spells out the
variance-controlled validation that would earn the infrastructure step.
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
