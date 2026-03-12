"""Episodic memory - medium-term, cross-session experience storage.

Supports two backends:
- **SQLite** (when ``db_path`` is given): persistent across restarts, uses aiosqlite.
- **In-memory** (default, ``db_path=None``): fast, no dependencies, backward-compatible
  with the original list-based implementation.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


class EpisodicMemory:
    """Episodic store with keyword search, CRUD, and optional SQLite persistence.

    Args:
        db_path: Path to a SQLite database file.  When *None* (default) the
            store lives entirely in memory — matching the pre-v2 behaviour.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path
        # In-memory fallback
        self._entries: list[dict[str, Any]] = []
        # Cached SQLite connection (lazy, see _get_connection)
        self._conn = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the backing table if using SQLite.  Safe to call repeatedly."""
        if self._db_path is None:
            return  # no-op for in-memory mode

        db = await self._get_connection()
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                key        TEXT PRIMARY KEY,
                content    TEXT NOT NULL,
                metadata   TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
            """
        )
        await db.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        """Lazy-initialize SQLite on first use (safe to call repeatedly)."""
        if self._db_path is not None and not hasattr(self, "_initialized"):
            await self.initialize()
            self._initialized = True

    async def _get_connection(self):
        """Return a cached aiosqlite connection, creating one if needed."""
        if self._conn is None:
            import aiosqlite
            self._conn = await aiosqlite.connect(self._db_path)
        return self._conn

    async def close(self) -> None:
        """Close the cached SQLite connection if open."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def store(
        self,
        key: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store (or overwrite) an episodic memory entry."""
        await self._ensure_initialized()
        meta = metadata or {}

        if self._db_path is not None:
            now = datetime.now(timezone.utc).isoformat()
            db = await self._get_connection()
            await db.execute(
                """
                INSERT OR REPLACE INTO episodes (key, content, metadata, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, content, json.dumps(meta), now),
            )
            await db.commit()
        else:
            # In-memory: overwrite if key exists, else append
            for entry in self._entries:
                if entry["key"] == key:
                    entry["content"] = content
                    entry["metadata"] = meta
                    return
            self._entries.append({"key": key, "content": content, "metadata": meta})

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search episodic memory by keyword matching.

        Returns up to *top_k* results, each a dict with ``key``, ``content``,
        ``metadata`` fields.
        """
        await self._ensure_initialized()
        if self._db_path is not None:
            return await self._sqlite_search(query, top_k)

        # --- In-memory keyword scoring (unchanged from v1) ---
        query_lower = query.lower()
        scored: list[tuple[int, dict[str, Any]]] = []
        for entry in self._entries:
            content_lower = entry["content"].lower()
            key_lower = entry["key"].lower()
            score = sum(
                1
                for word in query_lower.split()
                if word in content_lower or word in key_lower
            )
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    async def update(
        self,
        key: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing entry.  Returns *False* if the key is not found."""
        await self._ensure_initialized()
        if self._db_path is not None:
            return await self._sqlite_update(key, content, metadata)

        for entry in self._entries:
            if entry["key"] == key:
                if content is not None:
                    entry["content"] = content
                if metadata is not None:
                    entry["metadata"] = metadata
                return True
        return False

    async def delete(self, key: str) -> bool:
        """Delete an entry by key.  Returns *False* if not found."""
        await self._ensure_initialized()
        if self._db_path is not None:
            return await self._sqlite_delete(key)

        for i, entry in enumerate(self._entries):
            if entry["key"] == key:
                self._entries.pop(i)
                return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def count(self) -> int:
        """Return the total number of stored entries."""
        await self._ensure_initialized()
        if self._db_path is not None:
            db = await self._get_connection()
            async with db.execute("SELECT COUNT(*) FROM episodes") as cur:
                row = await cur.fetchone()
                return row[0] if row else 0

        return len(self._entries)

    async def list_all(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return entries, most-recent first.  Capped at *limit*."""
        await self._ensure_initialized()
        if self._db_path is not None:
            import aiosqlite
            db = await self._get_connection()
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT key, content, metadata FROM episodes ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
                return [
                    {
                        "key": r["key"],
                        "content": r["content"],
                        "metadata": json.loads(r["metadata"]),
                    }
                    for r in rows
                ]

        # In-memory: return most recently appended first
        return list(reversed(self._entries[-limit:]))

    # ------------------------------------------------------------------
    # Backward-compatibility helpers
    # ------------------------------------------------------------------

    def list_keys(self) -> list[str]:
        """Return all entry keys (sync, in-memory only — kept for v1 compat)."""
        return [e["key"] for e in self._entries]

    # ------------------------------------------------------------------
    # SQLite internals
    # ------------------------------------------------------------------

    async def _sqlite_search(
        self, query: str, top_k: int
    ) -> list[dict[str, Any]]:
        """Score-based search over SQLite rows using word matching."""
        import aiosqlite

        db = await self._get_connection()
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT key, content, metadata FROM episodes") as cur:
            rows = await cur.fetchall()

        words = query.lower().split()
        scored: list[tuple[int, dict[str, Any]]] = []
        for r in rows:
            text = (r["key"] + " " + r["content"]).lower()
            score = sum(1 for w in words if w in text)
            if score > 0:
                scored.append(
                    (
                        score,
                        {
                            "key": r["key"],
                            "content": r["content"],
                            "metadata": json.loads(r["metadata"]),
                        },
                    )
                )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    async def _sqlite_update(
        self,
        key: str,
        content: str | None,
        metadata: dict[str, Any] | None,
    ) -> bool:
        db = await self._get_connection()
        # Check existence first
        async with db.execute(
            "SELECT key FROM episodes WHERE key = ?", (key,)
        ) as cur:
            if await cur.fetchone() is None:
                return False

        parts: list[str] = []
        params: list[Any] = []
        if content is not None:
            parts.append("content = ?")
            params.append(content)
        if metadata is not None:
            parts.append("metadata = ?")
            params.append(json.dumps(metadata))

        if parts:
            params.append(key)
            await db.execute(
                f"UPDATE episodes SET {', '.join(parts)} WHERE key = ?",
                params,
            )
            await db.commit()
        return True

    async def _sqlite_delete(self, key: str) -> bool:
        db = await self._get_connection()
        cursor = await db.execute("DELETE FROM episodes WHERE key = ?", (key,))
        await db.commit()
        return cursor.rowcount > 0
