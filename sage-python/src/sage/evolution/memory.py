"""EvolutionMemory — persistent store for mutation history and learned skills.

Inspired by CORAL (arXiv 2604.01658): persistent attempts/skills enable
3-10x improvement rate over memoryless evolution.

Design choices:
- SQLite + WAL for append-heavy mutation logging
- No LLM calls for skill extraction — pure SQL aggregation
- Temporal decay on skills to prevent stale knowledge
- Domain-scoped queries to prevent cross-domain overfitting
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Defaults (overridable via constants.py)
_DEFAULT_DB_PATH = os.path.join(os.path.expanduser("~"), ".sage", "evolution_memory.db")
_MIN_SAMPLES = 5
_SKILL_TOP_K = 3
_DECAY_HALF_LIFE_DAYS = 30.0


@dataclass
class MutationRecord:
    """A single mutation attempt with full context."""

    id: str = ""
    generation: int = 0
    parent_id: str = ""
    child_id: str = ""
    sampo_action: int = 0
    parent_score: float = 0.0
    child_score: float = 0.0
    accepted: bool = False
    eval_stage: str = ""
    eval_details: dict[str, Any] = field(default_factory=dict)
    domain: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class Skill:
    """A learned pattern from evolution history."""

    id: str = ""
    pattern: str = ""
    sampo_action: int = 0
    success_rate: float = 0.0
    sample_count: int = 0
    domain: str = ""
    min_parent_score: float = 0.0
    max_parent_score: float = 1.0
    created_at: float = 0.0
    last_used_at: float = 0.0


class EvolutionMemory:
    """SQLite persistent store for evolution attempts + extracted skills.

    Two tables:
    - mutations: raw log of every mutation attempt (parent, child, score, action)
    - skills: aggregated patterns (action × domain × score_bucket → success_rate)

    Skills are extracted via SQL aggregation (no LLM), with temporal decay.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(Path(db_path or _DEFAULT_DB_PATH).expanduser())
        self._db: Any = None  # aiosqlite connection
        self._initialized = False

    async def _ensure_init(self) -> None:
        """Lazy init: create tables on first async call."""
        if self._initialized:
            return
        await self.initialize()

    async def initialize(self) -> None:
        """Create tables if they don't exist. Uses WAL mode for write throughput."""
        if self._initialized:
            return
        import aiosqlite

        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS mutations (
                id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                sampo_action INTEGER NOT NULL,
                parent_score REAL NOT NULL,
                child_score REAL NOT NULL,
                accepted INTEGER NOT NULL,
                eval_stage TEXT NOT NULL DEFAULT '',
                eval_details TEXT NOT NULL DEFAULT '{}',
                domain TEXT NOT NULL DEFAULT '',
                timestamp REAL NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                pattern TEXT NOT NULL,
                sampo_action INTEGER NOT NULL,
                success_rate REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                domain TEXT NOT NULL DEFAULT '',
                min_parent_score REAL NOT NULL DEFAULT 0.0,
                max_parent_score REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL,
                last_used_at REAL NOT NULL
            )
        """)
        # Index for fast skill queries
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_skills_domain_score
            ON skills (domain, min_parent_score, max_parent_score)
        """)
        await self._db.commit()
        self._initialized = True
        log.info("EvolutionMemory initialized at %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def record_mutation(self, record: MutationRecord) -> None:
        """Persist a mutation attempt."""
        await self._ensure_init()
        if not self._db:
            return
        await self._db.execute(
            """INSERT OR REPLACE INTO mutations
               (id, generation, parent_id, child_id, sampo_action,
                parent_score, child_score, accepted, eval_stage, eval_details,
                domain, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id, record.generation, record.parent_id, record.child_id,
                record.sampo_action, record.parent_score, record.child_score,
                int(record.accepted), record.eval_stage,
                json.dumps(record.eval_details),
                record.domain, record.timestamp,
            ),
        )
        await self._db.commit()

    async def extract_skills(self, min_samples: int = _MIN_SAMPLES) -> int:
        """Extract skills from mutation history via SQL aggregation.

        Groups mutations by (sampo_action, domain, parent_score_bucket)
        and creates/updates skills where sample_count >= min_samples.
        Returns number of skills added/updated.
        """
        await self._ensure_init()
        if not self._db:
            return 0

        now = time.time()
        # Bucket parent_score into 0.1 increments (0.0, 0.1, ..., 0.9)
        cursor = await self._db.execute("""
            SELECT
                sampo_action,
                domain,
                CAST(parent_score * 10 AS INTEGER) / 10.0 AS score_bucket,
                COUNT(*) AS cnt,
                SUM(accepted) AS successes,
                MIN(parent_score) AS min_ps,
                MAX(parent_score) AS max_ps
            FROM mutations
            GROUP BY sampo_action, domain, score_bucket
            HAVING cnt >= ?
        """, (min_samples,))

        rows = await cursor.fetchall()
        added = 0
        for action, domain, bucket, cnt, successes, min_ps, max_ps in rows:
            success_rate = successes / cnt if cnt > 0 else 0.0
            skill_id = f"s{action}-{domain or 'any'}-{bucket:.1f}"

            await self._db.execute(
                """INSERT INTO skills (id, pattern, sampo_action, success_rate,
                   sample_count, domain, min_parent_score, max_parent_score,
                   created_at, last_used_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                   success_rate = excluded.success_rate,
                   sample_count = excluded.sample_count,
                   min_parent_score = excluded.min_parent_score,
                   max_parent_score = excluded.max_parent_score,
                   last_used_at = excluded.last_used_at""",
                (
                    skill_id,
                    f"SAMPO action {action} on {domain or 'any'} domain, score ~{bucket:.1f}",
                    action, success_rate, cnt, domain or "",
                    min_ps, max_ps, now, now,
                ),
            )
            added += 1

        await self._db.commit()
        if added:
            log.info("EvolutionMemory: extracted %d skills from %d mutation groups", added, len(rows))
        return added

    async def query_skills(
        self,
        domain: str = "",
        parent_score: float = 0.0,
        top_k: int = _SKILL_TOP_K,
    ) -> list[Skill]:
        """Query relevant skills for a mutation context.

        Filters by domain (or 'any') and parent_score range.
        Returns top-k by success_rate.
        """
        await self._ensure_init()
        if not self._db:
            return []

        cursor = await self._db.execute(
            """SELECT id, pattern, sampo_action, success_rate, sample_count,
                      domain, min_parent_score, max_parent_score, created_at, last_used_at
               FROM skills
               WHERE (domain = ? OR domain = '')
                 AND min_parent_score <= ?
                 AND max_parent_score >= ?
               ORDER BY success_rate DESC
               LIMIT ?""",
            (domain, parent_score + 0.15, parent_score - 0.15, top_k),
        )
        rows = await cursor.fetchall()

        # Update last_used_at for retrieved skills
        now = time.time()
        skill_ids = [row[0] for row in rows]
        if skill_ids:
            placeholders = ",".join("?" * len(skill_ids))
            await self._db.execute(
                f"UPDATE skills SET last_used_at = ? WHERE id IN ({placeholders})",
                [now, *skill_ids],
            )
            await self._db.commit()

        return [
            Skill(
                id=r[0], pattern=r[1], sampo_action=r[2], success_rate=r[3],
                sample_count=r[4], domain=r[5], min_parent_score=r[6],
                max_parent_score=r[7], created_at=r[8], last_used_at=r[9],
            )
            for r in rows
        ]

    async def decay_skills(self, half_life_days: float = _DECAY_HALF_LIFE_DAYS) -> int:
        """Apply temporal decay to skill success rates.

        Skills unused for >half_life_days have their success_rate halved.
        Skills with success_rate < 0.05 after decay are deleted.
        Returns number of skills decayed/deleted.
        """
        await self._ensure_init()
        if not self._db:
            return 0

        now = time.time()
        half_life_s = half_life_days * 86400

        cursor = await self._db.execute(
            "SELECT id, success_rate, last_used_at FROM skills"
        )
        rows = await cursor.fetchall()

        decayed = 0
        for skill_id, rate, last_used in rows:
            age_s = now - last_used
            if age_s <= 0:
                continue
            decay_factor = math.pow(0.5, age_s / half_life_s)
            new_rate = rate * decay_factor

            if new_rate < 0.05:
                await self._db.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
            else:
                await self._db.execute(
                    "UPDATE skills SET success_rate = ? WHERE id = ?",
                    (new_rate, skill_id),
                )
            decayed += 1

        if decayed:
            await self._db.commit()
        return decayed

    async def stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        await self._ensure_init()
        if not self._db:
            return {"mutations": 0, "skills": 0}

        cur_m = await self._db.execute("SELECT COUNT(*) FROM mutations")
        cur_s = await self._db.execute("SELECT COUNT(*) FROM skills")
        cur_acc = await self._db.execute(
            "SELECT AVG(accepted) FROM mutations"
        )
        (n_mutations,) = await cur_m.fetchone()
        (n_skills,) = await cur_s.fetchone()
        (avg_accepted,) = await cur_acc.fetchone()

        return {
            "mutations": n_mutations,
            "skills": n_skills,
            "avg_acceptance_rate": round(avg_accepted or 0, 3),
            "db_path": self._db_path,
        }
