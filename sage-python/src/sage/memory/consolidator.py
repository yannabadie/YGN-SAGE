"""Inter-tier memory consolidation pipeline.

Closes the information silo between memory tiers by periodically mining
episodic entries for entities/relations and feeding them into semantic
and causal memory.

Research basis:
- MAGMA (2601.03236): 4 orthogonal memory graphs with cross-graph consolidation
- AMA-Bench (2602.22769): Memory fails without causal links
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sage.constants import CONSOLIDATION_BATCH_SIZE

log = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Result of one consolidation pass."""

    processed: int = 0
    entities_added: int = 0
    causal_edges_added: int = 0
    skipped_already_consolidated: int = 0


class MemoryConsolidator:
    """Consolidates episodic memories into semantic and causal tiers.

    Flow:
    1. Scan episodic memory for unconsolidated entries
    2. Run MemoryAgent.extract() on each to produce entities/relations
    3. Feed ExtractionResult into SemanticMemory.add_extraction()
    4. For sequential steps, create causal edges (step N "enabled" step N+1)
    5. Mark entries as consolidated (update metadata)

    Parameters
    ----------
    episodic : EpisodicMemory
        Tier 1 episodic store.
    semantic : SemanticMemory
        Tier 2 semantic entity graph.
    causal : CausalMemory | None
        Tier 2b causal graph. If None, causal edges are skipped.
    memory_agent : MemoryAgent
        Entity/relation extractor.
    batch_size : int
        Max entries to consolidate per pass.
    """

    def __init__(
        self,
        episodic: Any,
        semantic: Any,
        causal: Any | None,
        memory_agent: Any,
        batch_size: int = CONSOLIDATION_BATCH_SIZE,
    ) -> None:
        self.episodic = episodic
        self.semantic = semantic
        self.causal = causal
        self.memory_agent = memory_agent
        self.batch_size = batch_size

    async def consolidate(self) -> ConsolidationResult:
        """Run one consolidation pass. Returns stats."""
        result = ConsolidationResult()

        try:
            entries = await self.episodic.list_all(limit=self.batch_size)
        except Exception as e:
            log.warning("Consolidation: failed to list episodic entries: %s", e)
            return result

        # Filter to unconsolidated entries
        unconsolidated = []
        for entry in entries:
            meta = entry.get("metadata") or {}
            if meta.get("consolidated"):
                result.skipped_already_consolidated += 1
            else:
                unconsolidated.append(entry)

        if not unconsolidated:
            return result

        prev_entities: list[str] = []

        for entry in unconsolidated:
            content = entry.get("content", "")
            if not content or len(content) < 20:
                continue

            try:
                extraction = await self.memory_agent.extract(content[:1000])
            except Exception as e:
                log.debug("Consolidation: extraction failed for '%s': %s", entry.get("key"), e)
                continue

            if extraction.entities:
                self.semantic.add_extraction(extraction)
                result.entities_added += len(extraction.entities)

                # Causal edges: sequential episodes imply causal flow
                if self.causal and prev_entities and extraction.entities:
                    try:
                        # Last entity of previous step -> first entity of this step
                        src = prev_entities[-1]
                        tgt = extraction.entities[0]
                        self.causal.add_entity(src)
                        self.causal.add_entity(tgt)
                        self.causal.add_causal_edge(src, tgt, cause_type="enabled")
                        result.causal_edges_added += 1
                    except Exception as e:
                        log.debug("Consolidation: causal edge failed: %s", e)

                prev_entities = extraction.entities

            # Mark as consolidated
            try:
                existing_meta = entry.get("metadata") or {}
                existing_meta["consolidated"] = True
                await self.episodic.update(
                    entry["key"],
                    metadata=existing_meta,
                )
            except Exception as e:
                log.debug("Consolidation: failed to mark '%s' as consolidated: %s", entry.get("key"), e)

            result.processed += 1

        log.info(
            "Consolidation: processed=%d entities=%d causal_edges=%d skipped=%d",
            result.processed,
            result.entities_added,
            result.causal_edges_added,
            result.skipped_already_consolidated,
        )
        return result
