"""Memory Compressor: uses LLMs to summarize and persist memory."""
from __future__ import annotations

import logging
import re

from sage.llm.base import LLMProvider, Message, Role
from sage.memory.embedder import Embedder
from sage.memory.working import WorkingMemory


class MemoryCompressor:
    """Agent that monitors working memory and performs compression & persistence."""

    def __init__(
        self,
        llm: LLMProvider,
        compression_threshold: int = 20,
        keep_recent: int = 5,
    ):
        self.llm = llm
        self.compression_threshold = compression_threshold
        self.keep_recent = keep_recent
        self.logger = logging.getLogger(__name__)
        self.internal_state: str = ""
        self._hash_warned: bool = False
        # Embedder for S-MMU semantic edges — default to hash fallback.
        # The boot sequence (boot.py) will inject a real one later.
        self.embedder: Embedder = Embedder(force_hash=True)

    async def generate_internal_state(self, new_observation: str) -> str:
        """MEM1: generate rolling internal state by merging previous IS with new observation."""
        if self.internal_state:
            prompt = (
                f"You are maintaining a rolling internal state for an AI agent.\n"
                f"Previous state:\n{self.internal_state}\n\n"
                f"New observation:\n{new_observation}\n\n"
                f"Produce an updated internal state that merges the previous state "
                f"with the new observation. Be concise (max 3 sentences). "
                f"Drop details that are no longer relevant."
            )
        else:
            prompt = (
                f"You are maintaining a rolling internal state for an AI agent.\n"
                f"First observation:\n{new_observation}\n\n"
                f"Produce a concise internal state summary (max 2 sentences)."
            )

        response = await self.llm.generate(
            messages=[Message(role=Role.USER, content=prompt)]
        )
        self.internal_state = response.content or new_observation[:200]
        return self.internal_state

    async def step(self, working_memory: WorkingMemory) -> bool:
        """Check and perform compression if threshold is met (memory pressure trigger)."""
        if self.embedder.is_hash_fallback and not self._hash_warned:
            self.logger.warning(
                "Compressor using hash-based embeddings (not semantic). "
                "S-MMU semantic retrieval quality degraded."
            )
            self._hash_warned = True
        if working_memory.event_count() < self.compression_threshold:
            return False

        self.logger.info(f"Compressing memory for agent {working_memory.agent_id}")

        # 1. Identify events to compress (all except keep_recent)
        all_events = working_memory.recent_events(working_memory.event_count())
        if len(all_events) <= self.keep_recent:
            return False
        to_compress = all_events[:-self.keep_recent] if self.keep_recent > 0 else all_events
        context = "\n".join([f"[{e['type']}] {e['content']}" for e in to_compress])

        # 2. Generate Summary & Key Discoveries via LLM
        prompt = f"""Analyze the following agent execution history and provide:
1. A concise summary of actions and results.
2. A list of KEY DISCOVERIES (facts, code patterns, or bugs) worth long-term storage.

History:
{context}

Format:
SUMMARY: <text>
DISCOVERIES:
- <discovery 1>
- <discovery 2>
"""
        response_obj = await self.llm.generate(
            messages=[Message(role=Role.USER, content=prompt)]
        )
        response = response_obj.content or ""
        
        # Simple parsing (could be improved with structured output)
        summary = ""
        discoveries = []
        lines = response.split("\n")
        current_section = None
        for line in lines:
            if line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
            elif line.startswith("DISCOVERIES:"):
                current_section = "discoveries"
            elif current_section == "discoveries" and line.strip().startswith("-"):
                discoveries.append(line.strip().lstrip("- ").strip())

        # 3. Update Working Memory — compress_old_events keeps N recent + prepends summary
        working_memory.compress(self.keep_recent, summary or "No summary generated.")

        # 4. Compact to Arrow + S-MMU (best-effort — failure must not break compression)
        try:
            # Extract keywords: words > 3 chars from summary, max 10
            keywords = self._extract_keywords(summary) if summary else []
            # Compute embedding from summary text
            embedding = self.embedder.embed(summary) if summary else None
            working_memory.compact_to_arrow_with_meta(
                keywords=keywords,
                embedding=embedding,
                summary=summary or None,
            )
        except Exception:
            self.logger.warning(
                "S-MMU compaction failed (best-effort); compression still succeeded",
                exc_info=True,
            )

        return True

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract keywords from text: unique words > 3 chars, max 10."""
        words = re.findall(r"[A-Za-z]+", text)
        seen: set[str] = set()
        keywords: list[str] = []
        for w in words:
            lower = w.lower()
            if len(w) > 3 and lower not in seen:
                seen.add(lower)
                keywords.append(w)
                if len(keywords) >= 10:
                    break
        return keywords
