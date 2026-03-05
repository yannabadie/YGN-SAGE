"""Curator module: heuristic filter + LLM relevance scoring.

Stage 1 applies fast heuristics to reject noise (short abstracts, stale uncited
papers, blocklist patterns).  Stage 2 calls an LLM to score each surviving
candidate on a 0-10 relevance scale for YGN-SAGE research.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from discover.discovery import PaperCandidate
from sage.llm.base import LLMProvider, Message, Role

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELEVANCE_THRESHOLD: int = 6
STALE_DAYS: int = 90
MIN_ABSTRACT_LENGTH: int = 100
BLOCKLIST_PATTERNS: list[str] = ["survey of surveys", "correction to", "erratum"]

# ---------------------------------------------------------------------------
# CuratedPaper dataclass
# ---------------------------------------------------------------------------


@dataclass
class CuratedPaper:
    """A paper that has passed curation (heuristic + LLM scoring)."""

    candidate: PaperCandidate
    relevance_score: int
    reason: str
    key_insights: list[str] = field(default_factory=list)
    pdf_path: Path | None = None


# ---------------------------------------------------------------------------
# Stage 1 — heuristic filter
# ---------------------------------------------------------------------------


def heuristic_filter(candidates: list[PaperCandidate]) -> list[PaperCandidate]:
    """Fast heuristic filter that rejects obvious noise.

    Rejection rules:
    - Abstract shorter than MIN_ABSTRACT_LENGTH
    - Published > STALE_DAYS ago AND citation_count == 0
    - Title matches any BLOCKLIST_PATTERNS (case-insensitive)
    """
    today = date.today()
    stale_cutoff = today - timedelta(days=STALE_DAYS)
    passed: list[PaperCandidate] = []

    for c in candidates:
        # Rule 1: short abstract
        if len(c.abstract) < MIN_ABSTRACT_LENGTH:
            logger.debug("Rejected (short abstract): %s", c.title)
            continue

        # Rule 2: stale + uncited
        if c.published < stale_cutoff and c.citation_count == 0:
            logger.debug("Rejected (stale+uncited): %s", c.title)
            continue

        # Rule 3: blocklist patterns
        title_lower = c.title.lower()
        if any(pattern in title_lower for pattern in BLOCKLIST_PATTERNS):
            logger.debug("Rejected (blocklist): %s", c.title)
            continue

        passed.append(c)

    logger.info(
        "Heuristic filter: %d/%d candidates passed", len(passed), len(candidates)
    )
    return passed


# ---------------------------------------------------------------------------
# Stage 2 — LLM scoring
# ---------------------------------------------------------------------------

CURATION_PROMPT = """\
You are a research paper curator for YGN-SAGE, an AI agent development kit built on \
5 cognitive pillars: Topology, Tools, Memory, Evolution, Strategy.

Rate each paper below on a scale of 0-10 for relevance to YGN-SAGE research.
Consider relevance to: multi-agent RL, cognitive architectures, formal verification, \
evolutionary computation (MAP-Elites, QD), memory systems (episodic, semantic, RAG).

For each paper, provide:
- score: integer 0-10
- reason: one-sentence justification
- key_insights: list of 1-3 key insights from the abstract

Papers:
{papers}

Respond ONLY with a JSON array (no markdown fences):
[{{"score": <int>, "reason": "<str>", "key_insights": ["<str>", ...]}}]
"""


async def llm_score(
    candidates: list[PaperCandidate],
    llm: LLMProvider,
    batch_size: int = 20,
) -> list[CuratedPaper]:
    """Score candidates via LLM in batches and return CuratedPaper list."""
    curated: list[CuratedPaper] = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        papers_text = "\n\n".join(
            f"[{j}] Title: {c.title}\n    Abstract: {c.abstract[:500]}"
            for j, c in enumerate(batch)
        )

        prompt = CURATION_PROMPT.format(papers=papers_text)
        messages = [Message(role=Role.USER, content=prompt)]

        try:
            response = await llm.generate(messages)
            content = response.content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0].strip()
            scores: list[dict[str, Any]] = json.loads(content)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("LLM scoring parse failed: %s", exc)
            # Assign neutral score on failure
            for c in batch:
                curated.append(
                    CuratedPaper(
                        candidate=c,
                        relevance_score=5,
                        reason="LLM scoring failed",
                    )
                )
            continue

        for j, c in enumerate(batch):
            if j < len(scores):
                s = scores[j]
                curated.append(
                    CuratedPaper(
                        candidate=c,
                        relevance_score=int(s.get("score", 5)),
                        reason=str(s.get("reason", "No reason provided")),
                        key_insights=list(s.get("key_insights", [])),
                    )
                )
            else:
                curated.append(
                    CuratedPaper(
                        candidate=c,
                        relevance_score=5,
                        reason="LLM scoring failed",
                    )
                )

    return curated


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


async def curate(
    candidates: list[PaperCandidate],
    llm: LLMProvider,
) -> list[CuratedPaper]:
    """Full curation pipeline: heuristic filter -> LLM score -> threshold.

    Returns only papers with relevance_score >= RELEVANCE_THRESHOLD.
    """
    # Stage 1: heuristic filter
    filtered = heuristic_filter(candidates)

    if not filtered:
        logger.info("No candidates survived heuristic filter")
        return []

    # Stage 2: LLM scoring
    scored = await llm_score(filtered, llm)

    # Stage 3: threshold
    passing = [cp for cp in scored if cp.relevance_score >= RELEVANCE_THRESHOLD]
    logger.info(
        "Curation complete: %d/%d papers pass threshold (%d)",
        len(passing),
        len(scored),
        RELEVANCE_THRESHOLD,
    )
    return passing
