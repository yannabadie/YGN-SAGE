"""Structural feature extraction for adaptive routing (Stage 0).

Python reimplementation of sage-core/src/routing/features.rs (287 LOC).
Field-for-field, method-for-method compatible with the Rust PyO3 class.
Pure string processing — no SIMD benefit in Rust for this workload.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Keyword groups (exact match with features.rs) ────────────────────
ALGO_KEYWORDS: list[str] = [
    "implement", "build", "algorithm", "optimize", "compiler",
    "concurrent", "distributed", "consensus", "lock-free",
]
CODE_KEYWORDS: list[str] = [
    "write", "create", "code", "function", "class", "method",
    "parse", "regex", "query", "endpoint", "decorator",
]
DEBUG_KEYWORDS: list[str] = [
    "debug", "fix", "error", "crash", "bug",
    "race condition", "deadlock", "oom", "memory leak",
]
DESIGN_KEYWORDS: list[str] = [
    "design", "architect", "refactor", "schema", "system",
    "prove", "induction", "complexity",
]
UNCERTAINTY_KEYWORDS: list[str] = [
    "maybe", "possibly", "explore", "investigate",
    "intermittent", "sometimes", "random", "flaky",
]
TOOL_KEYWORDS: list[str] = [
    "file", "search", "run", "execute", "compile",
    "test", "deploy", "download", "upload",
]


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(kw in text for kw in keywords)


def _count_keywords(text: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


@dataclass
class StructuralFeatures:
    """Cheap structural features extracted from a task string."""

    word_count: int = 0
    has_code_block: bool = False
    has_question_mark: bool = False
    keyword_complexity: float = 0.0
    keyword_uncertainty: float = 0.0
    tool_required: bool = False

    @classmethod
    def extract(cls, task: str) -> StructuralFeatures:
        """Extract structural features from a task string.

        Mirrors features.rs::StructuralFeatures::extract_from() exactly.
        """
        lower = task.lower() if task else ""
        word_count = len(task.split())
        has_code_block = "```" in lower or "~~~" in lower
        has_question_mark = "?" in task
        tool_required = _has_any(lower, TOOL_KEYWORDS)

        # Uncertainty score
        uncertainty_hits = _count_keywords(lower, UNCERTAINTY_KEYWORDS)
        keyword_uncertainty = min(uncertainty_hits * 0.25, 1.0)

        # Complexity score (elif priority: algo > debug > design > code)
        complexity = 0.2  # base
        if _has_any(lower, ALGO_KEYWORDS):
            complexity += 0.35
        elif _has_any(lower, DEBUG_KEYWORDS):
            complexity += 0.30
        elif _has_any(lower, DESIGN_KEYWORDS):
            complexity += 0.20
        elif _has_any(lower, CODE_KEYWORDS):
            complexity += 0.15

        if has_code_block:
            complexity += 0.1

        if word_count > 100:
            complexity += 0.15
        elif word_count > 50:
            complexity += 0.1
        elif word_count > 20:
            complexity += 0.05

        keyword_complexity = max(0.0, min(complexity, 1.0))

        return cls(
            word_count=word_count,
            has_code_block=has_code_block,
            has_question_mark=has_question_mark,
            keyword_complexity=keyword_complexity,
            keyword_uncertainty=keyword_uncertainty,
            tool_required=tool_required,
        )

    def __repr__(self) -> str:
        return (
            f"StructuralFeatures(words={self.word_count}, "
            f"code_block={self.has_code_block}, "
            f"question={self.has_question_mark}, "
            f"complexity={self.keyword_complexity:.3f}, "
            f"uncertainty={self.keyword_uncertainty:.3f}, "
            f"tool={self.tool_required})"
        )
