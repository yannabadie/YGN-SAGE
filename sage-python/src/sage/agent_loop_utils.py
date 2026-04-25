"""Utility functions for the agent loop.

Extracted from agent_loop.py to reduce file size.
These are pure functions with no dependency on AgentLoop state.
Re-exported by agent_loop.py for backward compatibility.
"""
from __future__ import annotations

import ast
import math
import re
import shlex

from sage.constants import STAGNATION_WINDOW


# Cost per 1K tokens (USD) -- derived from cards.toml at boot, not hardcoded.
_COST_PER_1K: dict[str, float] = {}


def _load_cost_table() -> None:
    """Populate _COST_PER_1K from ModelCard catalog (cards.toml)."""
    global _COST_PER_1K
    if _COST_PER_1K:
        return  # already loaded
    try:
        from sage_core import ModelRegistry  # type: ignore[import-not-found]
        from pathlib import Path
        for p in [
            Path.cwd() / "sage-core" / "config" / "cards.toml",
            Path.cwd().parent / "sage-core" / "config" / "cards.toml",
            Path.cwd() / "config" / "cards.toml",
        ]:
            if p.exists():
                reg = ModelRegistry.from_toml_file(str(p))
                for card in reg.all_models():
                    # Average of input + output cost per 1M, converted to per 1K
                    avg_per_m = (card.cost_input_per_m + card.cost_output_per_m) / 2
                    _COST_PER_1K[card.id] = avg_per_m / 1000
                break
    except (ImportError, IOError, OSError):
        pass  # Rust unavailable -- DEFAULT_COST_PER_1K used as fallback


def _estimate_tokens(text: str, actual_count: int | None = None) -> int:
    """Return actual token count from API if available, else rough estimate."""
    if actual_count is not None and actual_count > 0:
        return actual_count
    return max(1, len(text) // 4)


def _text_entropy(text: str) -> float:
    """Shannon entropy of character distribution (normalised 0-1)."""
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    ent = -sum((c / n) * math.log2(c / n) for c in freq.values() if c > 0)
    max_ent = math.log2(max(len(freq), 2))
    return ent / max_ent if max_ent > 0 else 0.0


def _extract_code_blocks(text: str) -> list[str]:
    """Extract fenced code blocks from markdown-style LLM output."""
    pattern = r"```(?:\w+)?\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


def _strip_markdown_fences(code: str) -> str:
    """Strip leading/trailing markdown fences from a code string."""
    code = code.strip()
    # Remove leading ```python or ``` line
    if code.startswith("```"):
        first_newline = code.find("\n")
        if first_newline != -1:
            code = code[first_newline + 1:]
        else:
            code = code[3:]
    # Remove trailing ```
    if code.rstrip().endswith("```"):
        code = code.rstrip()[:-3]
    return code.strip()


def _validate_code_syntax(code: str) -> tuple[bool, str]:
    """Validate Python code syntax via ast.parse().

    Returns (is_valid, error_message). If valid, error_message is empty.
    The error_message uses SLF (Single-Line Feedback) format for concise LLM guidance.
    """
    cleaned = _strip_markdown_fences(code)
    if not cleaned:
        return False, "SyntaxError: empty code block after stripping markdown fences"
    try:
        ast.parse(cleaned, mode="exec")
        return True, ""
    except SyntaxError as e:
        line_info = f" (line {e.lineno})" if e.lineno else ""
        return False, f"SyntaxError{line_info}: {e.msg}"


def _is_stagnating(error_history: list[str], window: int = STAGNATION_WINDOW) -> bool:
    """Detect stagnation: True if the last `window` errors are identical.

    This means the LLM is producing the same broken code repeatedly
    and retrying will not help -- escalation is needed.
    """
    if len(error_history) < window:
        return False
    recent = error_history[-window:]
    return all(e == recent[0] for e in recent)


def _is_code_task(task: str) -> bool:
    """Detect if task is primarily about code generation.

    Used to skip episodic/semantic memory injection for code tasks,
    which Sprint 3 evidence shows degrades accuracy (30% vs 50% no-memory).
    """
    lower = task.lower()
    return bool(re.search(
        r'\b(?:implement|code|function|class|method|algorithm|program|'
        r'write\s+(?:a\s+)?(?:function|method|class|code|script)|'
        r'python|javascript|rust|java|def\s|return\s)\b', lower
    ))


def _shell_quote(code: str) -> str:
    """Shell-quote a code string for subprocess execution."""
    return shlex.quote(code)
