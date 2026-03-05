"""Memory Compressor: uses LLMs to summarize and persist memory to Graph/Vector DBs."""
from __future__ import annotations

import logging
from typing import Any, Protocol, List
from datetime import datetime, timezone

from sage.llm.base import LLMProvider, Message, Role
from sage.memory.working import WorkingMemory

class GraphDatabase(Protocol):
    """Protocol for Neo4j or similar graph DB interaction."""
    def create_node(self, label: str, properties: dict[str, Any]) -> str: ...
    def create_relationship(self, from_id: str, to_id: str, rel_type: str) -> None: ...

class VectorDatabase(Protocol):
    """Protocol for Qdrant or similar vector DB interaction."""
    def upsert(self, collection: str, text: str, metadata: dict[str, Any]) -> str: ...

class MemoryCompressor:
    """Agent that monitors working memory and performs SOTA compression & persistence."""

    def __init__(
        self,
        llm: LLMProvider,
        graph_db: GraphDatabase | None = None,
        vector_db: VectorDatabase | None = None,
        compression_threshold: int = 20,
        keep_recent: int = 5,
    ):
        self.llm = llm
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.compression_threshold = compression_threshold
        self.keep_recent = keep_recent
        self.logger = logging.getLogger(__name__)
        self.internal_state: str = ""

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
                discoveries.append(line.strip().replace("-", "", 1).strip())

        # 3. Persist to Graph & Vector DBs (GraphRAG Sync)
        summary_node_id = None
        if self.graph_db:
            summary_node_id = self.graph_db.create_node("SummaryEvent", {
                "agent_id": working_memory.agent_id,
                "content": summary,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            for discovery in discoveries:
                disc_node_id = self.graph_db.create_node("Discovery", {
                    "content": discovery,
                    "agent_id": working_memory.agent_id
                })
                self.graph_db.create_relationship(summary_node_id, disc_node_id, "CONTAINS_DISCOVERY")
                
                if self.vector_db:
                    self.vector_db.upsert("discoveries", discovery, {
                        "agent_id": working_memory.agent_id,
                        "graph_node_id": disc_node_id
                    })

        # 4. Update Working Memory — compress_old_events keeps N recent + prepends summary
        working_memory.compress(self.keep_recent, summary or "No summary generated.")
        return True
