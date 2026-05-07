"""Rust MultiViewMMU Python wrapper (Phase 8 boundary)."""

from __future__ import annotations

from typing import Any

import sage_core


class RustMultiViewMMU:
    """Thin façade over ``sage_core.MultiViewMMU``."""

    def __init__(self) -> None:
        self._inner = sage_core.MultiViewMMU()

    def register_chunk(
        self,
        start_time: int,
        end_time: int,
        summary: str,
        keywords: list[str],
        embedding: list[float] | None = None,
        parent_chunk_id: str | None = None,
    ) -> str:
        return self._inner.register_chunk(
            start_time, end_time, summary, keywords, embedding, parent_chunk_id,
        )

    def retrieve_relevant(self, chunk_id: str, max_hops: int) -> Any:
        return self._inner.retrieve_relevant(chunk_id, max_hops)

    def chunk_count(self) -> int:
        return self._inner.chunk_count()

    def save_json(self, path: str) -> None:
        self._inner.save_json(path)

    @staticmethod
    def load_json(path: str) -> "RustMultiViewMMU":
        inner = sage_core.MultiViewMMU.load_json(path)
        mmu = RustMultiViewMMU.__new__(RustMultiViewMMU)
        mmu._inner = inner  # type: ignore[attr-defined]
        return mmu
