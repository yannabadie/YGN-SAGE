"""src/discover/frontier.py — MAP-Elites frontier explorer for research discovery."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

from discover.discovery import DOMAINS, discover

logger = logging.getLogger(__name__)

DOMAINS_LIST = list(DOMAINS.keys())


@dataclass
class FrontierDescriptor:
    """4D behavior descriptor for MAP-Elites archive."""
    domain_idx: int
    recency: float
    citation_velocity: float
    novelty: float


@dataclass
class FrontierEntry:
    paper_id: str
    descriptor: FrontierDescriptor
    fitness: float


@dataclass
class FrontierReport:
    coverage: float
    total_papers: int
    empty_regions: int
    generations_run: int


class FrontierArchive:
    """MAP-Elites grid archive for research frontier coverage."""

    def __init__(self, bins: list[int] | None = None):
        self._bins = bins or [5, 4, 4, 3]
        self._grid: dict[tuple[int, ...], FrontierEntry] = {}

    def _descriptor_to_cell(self, desc: FrontierDescriptor) -> tuple[int, ...]:
        d = min(desc.domain_idx, self._bins[0] - 1)
        r = min(int(desc.recency * self._bins[1]), self._bins[1] - 1)
        c = min(int(desc.citation_velocity * self._bins[2]), self._bins[2] - 1)
        n = min(int(desc.novelty * self._bins[3]), self._bins[3] - 1)
        return (d, r, c, n)

    def try_insert(self, paper_id: str, desc: FrontierDescriptor, fitness: float) -> bool:
        cell = self._descriptor_to_cell(desc)
        existing = self._grid.get(cell)
        if existing is None or fitness > existing.fitness:
            self._grid[cell] = FrontierEntry(paper_id, desc, fitness)
            return True
        return False

    def size(self) -> int:
        return len(self._grid)

    def coverage(self) -> float:
        total = 1
        for b in self._bins:
            total *= b
        return len(self._grid) / total

    def get_empty_cells(self) -> list[tuple[int, ...]]:
        all_cells = set()
        for d in range(self._bins[0]):
            for r in range(self._bins[1]):
                for c in range(self._bins[2]):
                    for n in range(self._bins[3]):
                        all_cells.add((d, r, c, n))
        return list(all_cells - set(self._grid.keys()))

    def cell_to_descriptor(self, cell: tuple[int, ...]) -> FrontierDescriptor:
        return FrontierDescriptor(
            domain_idx=cell[0],
            recency=(cell[1] + 0.5) / self._bins[1],
            citation_velocity=(cell[2] + 0.5) / self._bins[2],
            novelty=(cell[3] + 0.5) / self._bins[3],
        )

    def get_best_per_dimension(self, dim: int) -> dict[int, FrontierEntry]:
        best: dict[int, FrontierEntry] = {}
        for cell, entry in self._grid.items():
            val = cell[dim]
            if val not in best or entry.fitness > best[val].fitness:
                best[val] = entry
        return best


class FrontierExplorer:
    """MAP-Elites-based research frontier explorer."""

    def __init__(self, store: Any, embedder: Any, llm: Any = None, bins: list[int] | None = None):
        self._store = store
        self._embedder = embedder
        self._llm = llm
        self._archive = FrontierArchive(bins=bins)

    def _compute_descriptor(self, domain: str, published: date | None = None,
                             citation_count: int = 0, paper_embedding: np.ndarray | None = None) -> FrontierDescriptor:
        domain_idx = DOMAINS_LIST.index(domain) if domain in DOMAINS_LIST else 0

        if published:
            days_old = (date.today() - published).days
            recency = min(days_old / 365.0, 1.0)
        else:
            recency = 0.5

        citation_velocity = min(citation_count / 50.0, 1.0)

        novelty = 1.0
        if paper_embedding is not None and self._archive.size() > 0:
            results = self._store.search_dense(paper_embedding, limit=1)
            if results and results[0].get("score", 0) > 0:
                novelty = max(0.0, 1.0 - results[0]["score"])

        return FrontierDescriptor(domain_idx=domain_idx, recency=recency,
                                  citation_velocity=citation_velocity, novelty=novelty)

    async def seed(self) -> None:
        for domain in DOMAINS_LIST:
            dummy_vec = self._embedder.embed_text(domain)
            results = self._store.search_dense(dummy_vec, limit=20, domain=domain)
            for r in results:
                payload = r.get("payload", {})
                desc = self._compute_descriptor(
                    domain=payload.get("domain", domain),
                    citation_count=payload.get("citation_count", 0),
                )
                self._archive.try_insert(r["id"], desc, fitness=r.get("score", 0.5))

    async def _generate_query(self, target: FrontierDescriptor) -> str:
        domain_name = DOMAINS_LIST[target.domain_idx] if target.domain_idx < len(DOMAINS_LIST) else "marl"
        recency_hint = "very recent (last month)" if target.recency < 0.1 else "from the past year"
        novelty_hint = "highly novel, underexplored" if target.novelty > 0.7 else "well-established"

        if self._llm is None:
            keywords = DOMAINS[domain_name]["keywords"]
            return keywords[0] if keywords else domain_name

        from sage.llm.base import Message, Role
        prompt = (f"Generate a specific arXiv search query for:\n"
                  f"- Domain: {domain_name}\n"
                  f"- Paper type: {recency_hint}\n"
                  f"- Desired novelty: {novelty_hint}\n\n"
                  f"Return only the search query string, no explanation.")
        messages = [Message(role=Role.USER, content=prompt)]
        response = await self._llm.generate(messages)
        return response.content.strip()

    async def explore(self, generations: int = 5, batch_size: int = 10) -> FrontierReport:
        for gen in range(generations):
            empty_cells = self._archive.get_empty_cells()
            if not empty_cells:
                logger.info("Archive fully covered at generation %d", gen)
                break

            targets = empty_cells[:batch_size]
            for cell in targets:
                target_desc = self._archive.cell_to_descriptor(cell)
                query = await self._generate_query(target_desc)

                try:
                    since = date.today() - timedelta(days=int(target_desc.recency * 365) + 7)
                    domain = DOMAINS_LIST[target_desc.domain_idx] if target_desc.domain_idx < len(DOMAINS_LIST) else None
                    candidates = await discover(since=since, query=query, domains=[domain] if domain else None)
                except Exception as e:
                    logger.warning("Discovery failed for query '%s': %s", query, e)
                    continue

                for paper in candidates[:5]:
                    embedding = self._embedder.embed_text(f"{paper.title}. {paper.abstract}")
                    desc = self._compute_descriptor(domain=paper.domain, published=paper.published,
                                                    citation_count=paper.citation_count, paper_embedding=embedding)
                    self._archive.try_insert(paper.paper_id, desc, fitness=0.5)

            logger.info("Generation %d: coverage=%.2f%%, archive_size=%d",
                        gen, self._archive.coverage() * 100, self._archive.size())

        return FrontierReport(coverage=self._archive.coverage(), total_papers=self._archive.size(),
                              empty_regions=len(self._archive.get_empty_cells()), generations_run=generations)
