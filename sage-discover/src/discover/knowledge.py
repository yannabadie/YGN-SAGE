"""Knowledge Bridge: Programmatic access to NotebookLM for research grounding.

Uses the notebooklm-py SDK to query the 'Cerveau Externe' of YGN-SAGE.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

try:
    from notebooklm import NotebookLMClient
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


class NotebookLMBridge:
    """Async bridge to NotebookLM for grounding research in SOTA papers."""

    def __init__(self, notebook_id: str | None = None):
        self.notebook_id = notebook_id or os.getenv("NOTEBOOKLM_DEFAULT_ID")
        self._enabled = HAS_SDK and self.notebook_id is not None

    async def ask_research_question(self, question: str) -> str:
        """Ask a question grounded in the current notebook sources."""
        if not self._enabled:
            return "Knowledge Bridge disabled: SDK missing or NOTEBOOKLM_DEFAULT_ID not set."

        try:
            async with NotebookLMClient() as client:
                # SOTA 2026: Direct programmatic inquiry to the notebook
                # Note: We use the 'ask' feature which leverages NotebookLM's RAG
                response = await client.notebooks.ask(self.notebook_id, question)
                return response.text
        except Exception as e:
            return f"Error querying NotebookLM: {e}"

    async def get_sota_insights(self, domain: str) -> List[str]:
        """Retrieve key insights for a specific research domain (e.g., MARL, Sorting)."""
        prompt = f"What are the latest SOTA techniques for optimizing {domain} algorithms, specifically regarding hardware acceleration or meta-strategy?"
        response = await self.ask_research_question(prompt)
        
        # Simple extraction logic: split by bullets if present
        if "\n*" in response or "\n-" in response:
            return [line.strip("* -") for line in response.split("\n") if line.strip().startswith(("*", "-"))]
        return [response]

    @property
    def is_active(self) -> bool:
        return self._enabled
