"""ExoCortex: Persistent managed RAG via Google GenAI File Search API.

Replaces the fragile NotebookLM CLI bridge with native API integration.
Stores persist indefinitely. Free storage. Automatic chunking/embedding.
"""
from __future__ import annotations

import os
import logging
from typing import Any

log = logging.getLogger(__name__)

# Known stable store name -- avoids silent disable when .env is not loaded.
# Resolution order: explicit param > SAGE_EXOCORTEX_STORE env var > DEFAULT_STORE.
DEFAULT_STORE = "fileSearchStores/ygnsageresearch-wii7kwkqozrd"


class ExoCortex:
    """Persistent RAG store backed by Google GenAI File Search API."""

    def __init__(self, store_name: str | None = None):
        self.store_name = store_name or os.environ.get("SAGE_EXOCORTEX_STORE") or DEFAULT_STORE
        self._api_key = os.environ.get("GOOGLE_API_KEY", "")

    @property
    def is_available(self) -> bool:
        return bool(self._api_key and self.store_name)

    def get_file_search_tool(self) -> Any | None:
        """Return a types.Tool for injection into Gemini generate() calls.

        Returns None if no store is configured or API unavailable.
        """
        if not self.store_name or not self._api_key:
            return None
        try:
            from google.genai import types
            return types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[self.store_name]
                )
            )
        except (ImportError, AttributeError) as e:
            log.warning(f"FileSearch tool unavailable: {e}")
            return None

    async def create_store(self, display_name: str) -> str:
        """Create a new FileSearchStore. Returns the store resource name."""
        import asyncio
        from google import genai

        def _create():
            client = genai.Client(api_key=self._api_key)
            return client.file_search_stores.create(
                config={"display_name": display_name}
            )

        store = await asyncio.to_thread(_create)
        self.store_name = store.name
        log.info("Created ExoCortex store: %s", store.name)
        return store.name

    async def upload(self, file_path: str, display_name: str | None = None) -> None:
        """Upload and index a file into the store."""
        if not self.store_name:
            raise RuntimeError("No store configured. Call create_store() first.")
        import asyncio
        from google import genai

        def _upload():
            import time
            client = genai.Client(api_key=self._api_key)
            operation = client.file_search_stores.upload_to_file_search_store(
                file=file_path,
                file_search_store_name=self.store_name,
                config={"display_name": display_name or file_path},
            )
            while not operation.done:
                time.sleep(2)
                operation = client.operations.get(operation)

        await asyncio.to_thread(_upload)
        log.info("Uploaded %s to ExoCortex store", file_path)

    async def delete_store(self) -> None:
        """Delete the current store."""
        if not self.store_name:
            return
        import asyncio
        from google import genai

        store_name = self.store_name

        def _delete():
            client = genai.Client(api_key=self._api_key)
            client.file_search_stores.delete(name=store_name)

        await asyncio.to_thread(_delete)
        log.info("Deleted ExoCortex store: %s", self.store_name)
        self.store_name = None

    def query(self, question: str, domain: str | None = None) -> str:
        """Synchronous query for tool use. Returns grounded answer or empty string."""
        if not self.store_name or not self._api_key:
            return ""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self._api_key)
            tools = [types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[self.store_name]
                )
            )]
            config = types.GenerateContentConfig(
                tools=tools,
                system_instruction="Answer based on the indexed documents. Be precise and cite sources.",
            )
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=question,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            log.warning("ExoCortex query failed: %s", e)
            return ""


class RagCacheFallback:
    """Pure-Python LRU+TTL cache (fallback when sage_core unavailable)."""

    def __init__(self, max_entries: int = 1000, ttl_seconds: int = 3600):
        self._cache: dict[int, tuple[float, bytes]] = {}
        self._max = max_entries
        self._ttl = ttl_seconds

    def put(self, query_hash: int, data: bytes) -> None:
        import time
        if len(self._cache) >= self._max:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        self._cache[query_hash] = (time.time(), data)

    def get(self, query_hash: int) -> bytes | None:
        import time
        entry = self._cache.get(query_hash)
        if entry is None:
            return None
        ts, data = entry
        if time.time() - ts > self._ttl:
            del self._cache[query_hash]
            return None
        return data

    def stats(self) -> tuple[int, int, int]:
        return (0, 0, len(self._cache))  # No hit/miss tracking in fallback

    def clear(self) -> None:
        self._cache.clear()


def get_rag_cache(max_entries: int = 1000, ttl_seconds: int = 3600):
    """Get the best available RAG cache (Rust or Python fallback)."""
    try:
        import sage_core
        return sage_core.RagCache(max_entries, ttl_seconds)
    except (ImportError, AttributeError):
        return RagCacheFallback(max_entries, ttl_seconds)
