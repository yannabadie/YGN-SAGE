"""Autonomous Memory Agent: extracts entities/relations, manages compression."""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    entities: list[str] = field(default_factory=list)
    relationships: list[tuple[str, str, str]] = field(default_factory=list)
    summary: str = ""


class MemoryAgent:
    """Runs asynchronously to compress working memory into graph knowledge.

    Extracts entities and relationships from agent events using heuristic
    or LLM-powered extraction. Currently stores in-memory; graph DB
    persistence (Neo4j/Qdrant) is a planned future enhancement.
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_tier: str = "budget",
        compress_threshold: int = 50,
    ):
        self.use_llm = use_llm
        self.llm_tier = llm_tier
        self.compress_threshold = compress_threshold

    def should_compress(self, event_count: int) -> bool:
        return event_count > self.compress_threshold

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relationships from text."""
        if self.use_llm:
            return await self._llm_extract(text)
        return self._heuristic_extract(text)

    def _heuristic_extract(self, text: str) -> ExtractionResult:
        """Fast heuristic extraction (no LLM cost)."""
        # Extract capitalized terms as entities
        entities = list(set(re.findall(r'\b[A-Z][A-Za-z0-9_-]{2,}\b', text)))

        # Simple verb-based relationship extraction
        relationships = []
        for ent in entities:
            pattern = rf'{re.escape(ent)}\s+(verif\w+|uses?|creates?|calls?|returns?)\s+(\w+)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            for verb, obj in matches:
                relationships.append((ent, verb, obj))

        return ExtractionResult(
            entities=entities[:20],
            relationships=relationships[:10],
            summary=text[:200] if len(text) > 200 else text,
        )

    async def _llm_extract(self, text: str) -> ExtractionResult:
        """LLM-powered extraction with structured output."""
        from pydantic import BaseModel
        from sage.llm.router import ModelRouter
        from sage.llm.google import GoogleProvider
        from sage.llm.base import Message, Role

        class KGExtraction(BaseModel):
            entities: list[str]
            relationships: list[list[str]]  # [subject, predicate, object]
            summary: str

        config = ModelRouter.get_config(
            self.llm_tier, temperature=0.1, json_schema=KGExtraction,
        )
        provider = GoogleProvider()
        response = await provider.generate(
            messages=[
                Message(role=Role.SYSTEM, content=(
                    "Extract entities and relationships from the text. "
                    "Return JSON with entities (list of names), "
                    "relationships (list of [subject, predicate, object]), "
                    "and a one-sentence summary."
                )),
                Message(role=Role.USER, content=text),
            ],
            config=config,
        )

        try:
            parsed = KGExtraction.model_validate_json(response.content)
            return ExtractionResult(
                entities=parsed.entities,
                relationships=[tuple(r) for r in parsed.relationships if len(r) == 3],
                summary=parsed.summary,
            )
        except Exception as e:
            log.warning(f"LLM extraction failed: {e}, falling back to heuristic")
            return self._heuristic_extract(text)
