"""Discovery module: scan arXiv, Semantic Scholar, and HuggingFace for papers.

Scans 3 academic sources across 5 research domains and deduplicates results
by normalized title, keeping the entry with the highest citation count.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------------

DOMAINS: dict[str, dict[str, Any]] = {
    "marl": {
        "arxiv_categories": ["cs.MA", "cs.AI", "cs.GT"],
        "keywords": [
            "multi-agent reinforcement learning",
            "cooperative MARL",
            "competitive multi-agent",
            "communication in MARL",
        ],
    },
    "cognitive_architectures": {
        "arxiv_categories": ["cs.AI", "cs.CL", "cs.NE"],
        "keywords": [
            "cognitive architecture",
            "metacognition",
            "System 1 System 2",
            "dual process theory AI",
        ],
    },
    "formal_verification": {
        "arxiv_categories": ["cs.LO", "cs.SE", "cs.PL"],
        "keywords": [
            "formal verification neural networks",
            "SMT solver AI",
            "process reward model",
            "program synthesis verification",
        ],
    },
    "evolutionary_computation": {
        "arxiv_categories": ["cs.NE", "cs.AI"],
        "keywords": [
            "MAP-Elites",
            "quality diversity",
            "evolutionary algorithm LLM",
            "neural architecture search",
        ],
    },
    "memory_systems": {
        "arxiv_categories": ["cs.AI", "cs.CL", "cs.IR"],
        "keywords": [
            "episodic memory AI",
            "retrieval augmented generation",
            "knowledge graph memory",
            "working memory transformer",
        ],
    },
}

# ---------------------------------------------------------------------------
# PaperCandidate dataclass
# ---------------------------------------------------------------------------


@dataclass
class PaperCandidate:
    """A candidate paper discovered from any source."""

    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    source: str  # "arxiv" | "s2" | "hf"
    domain: str
    published: date
    pdf_url: str | None
    citation_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_title(title: str) -> str:
    """Lowercase and strip punctuation for deduplication."""
    return _PUNCT_RE.sub("", title.lower()).strip()


def deduplicate(candidates: list[PaperCandidate]) -> list[PaperCandidate]:
    """Deduplicate papers by normalized title, keeping highest citation count."""
    seen: dict[str, PaperCandidate] = {}
    for c in candidates:
        key = _normalize_title(c.title)
        if key not in seen or c.citation_count > seen[key].citation_count:
            seen[key] = c
    return list(seen.values())


# ---------------------------------------------------------------------------
# Source: arXiv
# ---------------------------------------------------------------------------


async def discover_arxiv(
    domain: str,
    since: date,
    max_results: int = 20,
) -> list[PaperCandidate]:
    """Search arXiv for papers in *domain* published after *since*.

    Uses the ``arxiv`` library (wrapped in ``asyncio.to_thread`` for async).
    Returns an empty list if the library is not installed.
    """
    try:
        import arxiv  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("arxiv library not installed -- skipping arXiv source")
        return []

    domain_info = DOMAINS.get(domain)
    if domain_info is None:
        return []

    categories = " OR ".join(f"cat:{c}" for c in domain_info["arxiv_categories"])
    keywords = " OR ".join(f'"{k}"' for k in domain_info["keywords"])
    query = f"({categories}) AND ({keywords})"

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    def _fetch() -> list[PaperCandidate]:
        client = arxiv.Client()
        results: list[PaperCandidate] = []
        for r in client.results(search):
            pub_date = r.published.date() if hasattr(r.published, "date") else r.published
            if pub_date < since:
                continue
            results.append(
                PaperCandidate(
                    paper_id=r.entry_id,
                    title=r.title,
                    authors=[a.name for a in r.authors],
                    abstract=r.summary,
                    source="arxiv",
                    domain=domain,
                    published=pub_date,
                    pdf_url=r.pdf_url,
                    citation_count=0,  # arXiv doesn't provide citation counts
                )
            )
        return results

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as exc:
        logger.error("arXiv discovery failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Source: Semantic Scholar
# ---------------------------------------------------------------------------


async def discover_semantic_scholar(
    domain: str,
    since: date,
    max_results: int = 20,
) -> list[PaperCandidate]:
    """Search Semantic Scholar for papers in *domain* published after *since*.

    Uses the ``semanticscholar`` library. Returns an empty list if not installed.
    """
    try:
        from semanticscholar import SemanticScholar  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("semanticscholar library not installed -- skipping S2 source")
        return []

    domain_info = DOMAINS.get(domain)
    if domain_info is None:
        return []

    query = " ".join(domain_info["keywords"][:2])

    def _fetch() -> list[PaperCandidate]:
        sch = SemanticScholar()
        results: list[PaperCandidate] = []
        try:
            papers = sch.search_paper(query, limit=max_results)
        except Exception as exc:
            logger.error("Semantic Scholar search failed: %s", exc)
            return []

        for p in papers:
            # Parse publication date
            pub_date_str = getattr(p, "publicationDate", None)
            if pub_date_str is None:
                continue
            if isinstance(pub_date_str, str):
                try:
                    pub_date = date.fromisoformat(pub_date_str)
                except ValueError:
                    continue
            elif isinstance(pub_date_str, date):
                pub_date = pub_date_str
            else:
                # Try .date() for datetime-like objects
                try:
                    pub_date = pub_date_str.date()
                except (AttributeError, TypeError):
                    continue

            if pub_date < since:
                continue

            authors = []
            if hasattr(p, "authors") and p.authors:
                authors = [a.get("name", str(a)) if isinstance(a, dict) else getattr(a, "name", str(a)) for a in p.authors]

            results.append(
                PaperCandidate(
                    paper_id=getattr(p, "paperId", "") or "",
                    title=getattr(p, "title", "") or "",
                    authors=authors,
                    abstract=getattr(p, "abstract", "") or "",
                    source="s2",
                    domain=domain,
                    published=pub_date,
                    pdf_url=getattr(p, "url", None),
                    citation_count=getattr(p, "citationCount", 0) or 0,
                )
            )
        return results

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as exc:
        logger.error("Semantic Scholar discovery failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Source: HuggingFace
# ---------------------------------------------------------------------------


async def discover_hf(
    domain: str,
    max_results: int = 10,
) -> list[PaperCandidate]:
    """Search HuggingFace Hub for papers in *domain*.

    Uses ``huggingface_hub.HfApi().list_papers()``. Returns an empty list if
    the library is not installed or the API call fails.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("huggingface_hub not installed -- skipping HF source")
        return []

    domain_info = DOMAINS.get(domain)
    if domain_info is None:
        return []

    query = domain_info["keywords"][0]

    def _fetch() -> list[PaperCandidate]:
        api = HfApi()
        results: list[PaperCandidate] = []
        try:
            papers = api.list_papers(query=query)
        except Exception as exc:
            logger.error("HuggingFace paper search failed: %s", exc)
            return []

        count = 0
        for p in papers:
            if count >= max_results:
                break
            pub_date = date.today()
            if hasattr(p, "publishedAt") and p.publishedAt:
                try:
                    pub_date = p.publishedAt.date() if hasattr(p.publishedAt, "date") else date.fromisoformat(str(p.publishedAt)[:10])
                except (ValueError, AttributeError):
                    pass

            authors: list[str] = []
            if hasattr(p, "authors") and p.authors:
                authors = [getattr(a, "name", str(a)) if not isinstance(a, str) else a for a in p.authors]

            results.append(
                PaperCandidate(
                    paper_id=getattr(p, "id", "") or "",
                    title=getattr(p, "title", "") or "",
                    authors=authors,
                    abstract=getattr(p, "summary", "") or "",
                    source="hf",
                    domain=domain,
                    published=pub_date,
                    pdf_url=None,
                    citation_count=0,
                )
            )
            count += 1
        return results

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as exc:
        logger.error("HuggingFace discovery failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def discover(
    since: date,
    query: str,
    domains: list[str] | None = None,
) -> list[PaperCandidate]:
    """Run all 3 sources per domain in parallel, then deduplicate.

    Parameters
    ----------
    since : date
        Only include papers published on or after this date.
    query : str
        Optional free-text query (currently unused; domain keywords drive search).
    domains : list[str] | None
        Subset of DOMAINS keys to scan. Defaults to all 5 domains.
    """
    target_domains = domains if domains else list(DOMAINS.keys())

    tasks: list[asyncio.Task[list[PaperCandidate]]] = []
    for dom in target_domains:
        tasks.append(asyncio.ensure_future(discover_arxiv(dom, since)))
        tasks.append(asyncio.ensure_future(discover_semantic_scholar(dom, since)))
        tasks.append(asyncio.ensure_future(discover_hf(dom)))

    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates: list[PaperCandidate] = []
    for result in all_results:
        if isinstance(result, BaseException):
            logger.error("Discovery task failed: %s", result)
            continue
        candidates.extend(result)

    return deduplicate(candidates)
