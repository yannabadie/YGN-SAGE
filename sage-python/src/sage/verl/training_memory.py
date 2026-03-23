"""SQLite episodic memory for topology training.

Stores episode outcomes across epochs so the model can learn
from past successes/failures on similar tasks.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

import numpy as np

log = logging.getLogger("training_memory")


class TrainingMemory:
    """SQLite-backed episodic memory for training loop."""

    def __init__(self, db_path: str = "data/training_memory.db"):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                prompt_hash TEXT,
                domain TEXT,
                topology_yaml TEXT,
                n_nodes INTEGER,
                difficulty TEXT,
                outcome TEXT,
                total_reward REAL,
                per_node_results TEXT,
                adaptations_triggered INTEGER DEFAULT 0,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    def store_episode(
        self,
        task_id: str,
        prompt_hash: str,
        domain: str,
        topology_yaml: str,
        n_nodes: int,
        difficulty: str,
        outcome: str,
        total_reward: float,
        per_node_results: list[dict],
        adaptations_triggered: int,
        embedding: np.ndarray,
    ) -> None:
        """Persist one episode outcome."""
        self._conn.execute(
            """INSERT INTO episodes
               (task_id, prompt_hash, domain, topology_yaml, n_nodes,
                difficulty, outcome, total_reward, per_node_results,
                adaptations_triggered, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id, prompt_hash, domain, topology_yaml, n_nodes,
                difficulty, outcome, total_reward,
                json.dumps(per_node_results),
                adaptations_triggered,
                embedding.tobytes(),
            ),
        )
        self._conn.commit()

    def query_similar(self, query_embedding: np.ndarray, k: int = 3, domain: str = "") -> list[dict]:
        """Find top-k similar episodes by cosine similarity on embeddings.

        Parameters
        ----------
        query_embedding : np.ndarray
            The embedding vector to compare against stored episodes.
        k : int
            Number of top results to return.
        domain : str
            Optional domain prefilter. When non-empty, only episodes with
            matching domain are considered.
        """
        query = "SELECT * FROM episodes WHERE embedding IS NOT NULL"
        params: list[str] = []
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        query += " ORDER BY created_at DESC LIMIT 500"
        rows = self._conn.execute(query, params).fetchall()

        if not rows:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        scored = []
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            sim = float(np.dot(query_norm, emb_norm))
            scored.append((sim, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, row_dict in scored[:k]:
            row_dict.pop("embedding", None)
            if row_dict.get("per_node_results"):
                try:
                    row_dict["per_node_results"] = json.loads(row_dict["per_node_results"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(row_dict)
        return results

    def format_context(self, episodes: list[dict]) -> str:
        """Format episodes as text for model observation."""
        if not episodes:
            return ""
        lines = ["[Past experience on similar tasks]:"]
        for ep in episodes:
            outcome = ep.get("outcome", "?")
            reward = ep.get("total_reward", 0)
            diff = ep.get("difficulty", "?")
            n = ep.get("n_nodes", "?")
            adapt = ep.get("adaptations_triggered", 0)
            lines.append(
                f"- {diff}, {n} nodes, {outcome} (reward={reward:.2f})"
                + (f", {adapt} adaptations" if adapt else "")
            )
        return "\n".join(lines)

    def count(self) -> int:
        """Number of stored episodes."""
        return self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
