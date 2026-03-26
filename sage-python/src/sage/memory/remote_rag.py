"""ExoCortex: Persistent managed RAG via Google GenAI File Search API.

Replaces the fragile NotebookLM CLI bridge with native API integration.
Stores persist indefinitely. Free storage. Automatic chunking/embedding.

Implements the KnowledgeStore protocol (see rag_backend.py) for vendor-
independent RAG access.
"""
from __future__ import annotations

import os
import logging
from typing import Any

log = logging.getLogger(__name__)

# Known stable store name -- avoids silent disable when .env is not loaded.
# Resolution order: explicit param > SAGE_EXOCORTEX_STORE env var > DEFAULT_STORE.
DEFAULT_STORE = "fileSearchStores/ygnsageresearch-wii7kwkqozrd"

# Default model for ExoCortex queries.
# Resolution order: explicit param > SAGE_EXOCORTEX_MODEL env var > _DEFAULT_MODEL.
_DEFAULT_MODEL = "gemini-2.5-flash"


class ExoCortex:
    """Persistent RAG store backed by Google GenAI File Search API.

    Conforms to the ``KnowledgeStore`` protocol (``rag_backend.py``),
    enabling drop-in replacement with Qdrant, FAISS, or other backends.
    """

    def __init__(self, store_name: str | None = None, model_id: str | None = None):
        self._store_name = store_name or os.environ.get("SAGE_EXOCORTEX_STORE") or DEFAULT_STORE
        self._model_id = model_id or os.environ.get("SAGE_EXOCORTEX_MODEL") or _DEFAULT_MODEL
        self._api_key = os.environ.get("GOOGLE_API_KEY", "")
        self._client = None

    # -- KnowledgeStore protocol: store_name property --------------------------

    def _get_client(self):
        """Return a cached genai.Client, creating one if needed."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
            from sage.llm._ssl import patch_genai_ssl
            patch_genai_ssl(self._client)
        return self._client

    @property
    def store_name(self) -> str:
        """Human-readable name of the store backend."""
        return self._store_name

    @store_name.setter
    def store_name(self, value: str | None) -> None:
        self._store_name = value

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
            from sage.llm._ssl import patch_genai_ssl
            patch_genai_ssl(client)
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
            from sage.llm._ssl import patch_genai_ssl
            patch_genai_ssl(client)
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
            from sage.llm._ssl import patch_genai_ssl
            patch_genai_ssl(client)
            client.file_search_stores.delete(name=store_name)

        await asyncio.to_thread(_delete)
        log.info("Deleted ExoCortex store: %s", self.store_name)
        self.store_name = None

    def query(self, question: str, domain: str | None = None) -> str:
        """Synchronous query for tool use. Returns grounded answer or empty string."""
        if not self.store_name or not self._api_key:
            return ""
        try:
            from google.genai import types

            client = self._get_client()
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
                model=self._model_id,
                contents=question,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            log.warning("ExoCortex query failed: %s", e)
            return ""

    # -- KnowledgeStore protocol methods ---------------------------------------

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the knowledge store (KnowledgeStore protocol).

        Delegates to the synchronous ``query()`` method via asyncio.to_thread.
        Returns a list of result dicts with ``content`` and ``score`` keys.

        Note: Google File Search API returns a single grounded answer rather
        than ranked chunks, so results contain at most one entry with
        score=1.0. Future backends (Qdrant, FAISS) will return proper
        ranked results.
        """
        import asyncio

        result = await asyncio.to_thread(self.query, query)
        if not result:
            return []
        return [{"content": result, "score": 1.0}]

    async def ingest(self, content: str, metadata: dict | None = None) -> str:
        """Ingest content into the store (KnowledgeStore protocol).

        Writes *content* to a temporary file and uploads it via the
        Google File Search API.  Returns the temp file path used as the
        document identifier.

        For bulk file ingestion, prefer the existing ``upload()`` method
        which accepts a file path directly.
        """
        import asyncio
        import tempfile

        if not self.store_name:
            raise RuntimeError("No store configured. Call create_store() first.")

        display_name = (metadata or {}).get("display_name", "ingested-content")
        suffix = (metadata or {}).get("suffix", ".txt")

        # Write content to a temp file, then delegate to upload()
        def _write_temp() -> str:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, prefix="sage_ingest_"
            ) as f:
                f.write(content)
                return f.name

        tmp_path = await asyncio.to_thread(_write_temp)
        try:
            await self.upload(tmp_path, display_name=display_name)
        finally:
            # Best-effort cleanup
            try:
                import os as _os
                _os.unlink(tmp_path)
            except OSError:
                pass
        return tmp_path


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
