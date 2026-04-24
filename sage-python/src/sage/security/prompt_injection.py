"""Prompt-injection detection starter (AUDIT3 #10 / A13 roadmap).

Minimum-viable regex-based detector — NOT a full solution.

Production-grade prompt-injection defense in 2026 uses:
- PromptGuard-2 (Meta, 2024) — BERT-based classifier
- Lakera Guard — external API, multi-signal
- Azure AI Content Safety Prompt Shields (GA Nov 2024)
- OpenAI instruction-hierarchy training (GPT-4o priv separation)
- Microsoft spotlighting (arXiv 2403.14720)

This module intentionally stays at the regex tier because:
1. Integration into the agent loop is deferred to a second phase.
2. False-positive tuning needs real traffic; regex is a starter safe
   default (opt-in via env SAGE_PROMPT_INJECTION_STRICT=1).
3. Replacing with PromptGuard-2 / Lakera is a drop-in when the
   classifier is selected.

Opt-in API:
    from sage.security.prompt_injection import check, detect

    if not check(user_task_text):
        log.warning("possible prompt injection detected")

    # strict mode: raise on detection
    check(user_task_text, strict=True)   # raises PromptInjectionError

    # env-driven: SAGE_PROMPT_INJECTION_STRICT=1 implicit strict
    check(user_task_text)                # raises if env set

See AUDIT3 claim #10 / roadmap.md A13 / phase3-severity-sota.md.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

__all__ = [
    "InjectionMatch",
    "PROMPT_INJECTION_PATTERNS",
    "PromptInjectionError",
    "check",
    "detect",
]


class PromptInjectionError(Exception):
    """Raised by `check` when strict=True and an injection pattern matches."""


@dataclass(frozen=True)
class InjectionMatch:
    """One regex hit against the input text."""

    pattern_name: str
    match_text: str
    start: int
    end: int
    severity: str  # "medium" | "high"


# Regex patterns are case-insensitive. Each value is a (compiled_regex,
# severity) pair so we can score hits without a second lookup.
#
# Careful anchoring: "ignore previous instructions" requires the full
# phrase to avoid false-positives on code like "ignore blank lines".
PROMPT_INJECTION_PATTERNS: dict[str, tuple[re.Pattern[str], str]] = {
    # "ignore (all )? previous instructions"
    "ignore_previous_instructions": (
        re.compile(
            r"ignore\s+(?:all\s+|any\s+)?previous\s+instructions",
            re.IGNORECASE,
        ),
        "high",
    ),
    # "You are now DAN" or "You are now an uncensored/unrestricted/jailbroken"
    "jailbreak_role_reassignment": (
        re.compile(
            r"You\s+are\s+now\s+"
            r"(?:DAN|an?\s+(?:uncensored|unrestricted|jailbroken))"
            r"(?:\s+(?:model|assistant|AI))?",
            re.IGNORECASE,
        ),
        "high",
    ),
    # "system prompt override" / "system prompt reset"
    "system_prompt_override": (
        re.compile(
            r"system\s+prompt\s+(?:override|reset|bypass)",
            re.IGNORECASE,
        ),
        "high",
    ),
    # "reveal your system prompt / instructions / training"
    "reveal_prompt": (
        re.compile(
            r"reveal\s+your\s+(?:system\s+prompt|instructions|training)",
            re.IGNORECASE,
        ),
        "medium",
    ),
    # Chat-template smuggling markers (OpenAI / Anthropic / generic).
    "chat_template_marker": (
        re.compile(
            r"<\|im_start\|>|<\|system\|>|<\|im_end\|>|<\|end\|>",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Llama-style instruction markers (plain [INST] or the angle-
    # bracketed <[INST]> smuggling variant).
    "llama_inst_marker": (
        re.compile(
            r"\[INST\]|\[/INST\]",
            re.IGNORECASE,
        ),
        "medium",
    ),
    # base64 payload smuggling (≥40 chars). "base64:" prefix lowers FP risk.
    "base64_smuggling": (
        re.compile(
            r"base64:[A-Za-z0-9+/]{40,}={0,2}",
            re.IGNORECASE,
        ),
        "medium",
    ),
}


def detect(text: str) -> list[InjectionMatch]:
    """Scan `text` for all known prompt-injection patterns.

    Returns every match (duplicates included). Empty list when clean.
    """
    if not text:
        return []

    matches: list[InjectionMatch] = []
    for name, (pattern, severity) in PROMPT_INJECTION_PATTERNS.items():
        for m in pattern.finditer(text):
            matches.append(
                InjectionMatch(
                    pattern_name=name,
                    match_text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    severity=severity,
                )
            )
    return matches


def _env_strict() -> bool:
    """Env-gated strict mode. Default OFF to avoid surprise raises."""
    return os.environ.get("SAGE_PROMPT_INJECTION_STRICT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def check(text: str, *, strict: bool | None = None) -> bool:
    """Return True when `text` is clean, False otherwise.

    When `strict` is True (explicitly OR via SAGE_PROMPT_INJECTION_STRICT
    env) and any pattern matches, raises `PromptInjectionError` instead
    of returning False. Explicit `strict` arg wins over env.
    """
    matches = detect(text)
    if not matches:
        return True

    effective_strict = _env_strict() if strict is None else strict
    if effective_strict:
        names = ", ".join(sorted({m.pattern_name for m in matches}))
        raise PromptInjectionError(
            f"Prompt-injection pattern(s) detected: {names}. "
            f"Set SAGE_PROMPT_INJECTION_STRICT=0 or pass strict=False to downgrade."
        )
    return False
