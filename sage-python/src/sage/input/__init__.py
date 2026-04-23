"""Universal input adapter — one normalized entry point for chat + benches.

See `docs/superpowers/specs/2026-04-21-universal-input-adapter-design.md`
for the full design. Every source (chat REPL, SWE-bench, BigCodeBench, a
future API endpoint) produces a `TaskInput` via its own `normalize_*`
function, and the pipeline consumes that one shape.
"""
from sage.input.bcb import normalize_bcb, render_bcb_prompt
from sage.input.chat import CHAT_DEFAULT_TOOLS, normalize_chat
from sage.input.swebench import (
    SWEBENCH_SYSTEM_TEMPLATE,
    SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE,
    get_swebench_template,
    normalize_swebench,
    render_swebench_prompt,
)
from sage.input.types import ResponseFormat, TaskInput

__all__ = [
    "CHAT_DEFAULT_TOOLS",
    "ResponseFormat",
    "SWEBENCH_SYSTEM_TEMPLATE",
    "SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE",
    "TaskInput",
    "get_swebench_template",
    "normalize_bcb",
    "normalize_chat",
    "normalize_swebench",
    "render_bcb_prompt",
    "render_swebench_prompt",
]
