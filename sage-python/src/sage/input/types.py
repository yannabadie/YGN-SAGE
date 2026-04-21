"""TaskInput + ResponseFormat: the normalized shape any source maps to."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ResponseFormat(str, Enum):
    """Expected shape of the model's final answer.

    Benches (PATCH, CODE, JSON) need parseable output and inject the
    corresponding natural-language constraint. Chat mode (TEXT) does
    not — the LLM decides prose vs. markdown vs. fenced code based on
    the user's request.
    """

    TEXT = "text"
    CODE = "code"
    PATCH = "patch"
    JSON = "json"
    SEARCH_REPLACE = "search_replace"


@dataclass
class TaskInput:
    """The single normalized input to the pipeline.

    Chat, SWE-bench, BigCodeBench, and any future source all produce
    this shape via their own `normalize_*` function. The pipeline
    (via `perceive`) consumes it to build the system + user prompts.
    """

    prompt: str
    response_format: ResponseFormat = ResponseFormat.TEXT
    hints: dict = field(default_factory=dict)
    instructions: str = ""
    tools_filter: list[str] | None = None
    expected_length_hint: int = 0
    source: str = "chat"
