"""Tests for EvolutionMemory — persistent evolution knowledge (CORAL-inspired)."""
from __future__ import annotations

import os
import time

import pytest

from sage.evolution.memory import EvolutionMemory, MutationRecord, Skill


@pytest.fixture
async def evo_mem(tmp_path):
    """Fresh EvolutionMemory in a temp directory."""
    db_path = tmp_path / "test_evo.db"
    mem = EvolutionMemory(db_path=str(db_path))
    await mem.initialize()
    yield mem
    await mem.close()


def _make_record(**overrides) -> MutationRecord:
    defaults = dict(
        generation=1, parent_id="p1", child_id="c1",
        sampo_action=2, parent_score=0.5, child_score=0.7,
        accepted=True, eval_stage="unittest", eval_details={"pass": True},
        domain="code",
    )
    defaults.update(overrides)
    return MutationRecord(**defaults)


class TestEvolutionMemory:

    @pytest.mark.asyncio
    async def test_record_mutation(self, evo_mem):
        record = _make_record()
        await evo_mem.record_mutation(record)
        stats = await evo_mem.stats()
        assert stats["mutations"] == 1

    @pytest.mark.asyncio
    async def test_record_multiple(self, evo_mem):
        for i in range(10):
            await evo_mem.record_mutation(_make_record(
                child_id=f"c{i}", accepted=(i % 2 == 0),
            ))
        stats = await evo_mem.stats()
        assert stats["mutations"] == 10
        assert 0.4 <= stats["avg_acceptance_rate"] <= 0.6

    @pytest.mark.asyncio
    async def test_extract_skills_minimum_samples(self, evo_mem):
        """Skills require min_samples mutations to be extracted."""
        # Only 3 records — below threshold of 5
        for i in range(3):
            await evo_mem.record_mutation(_make_record(child_id=f"c{i}"))

        added = await evo_mem.extract_skills(min_samples=5)
        assert added == 0

        stats = await evo_mem.stats()
        assert stats["skills"] == 0

    @pytest.mark.asyncio
    async def test_extract_skills_sufficient_samples(self, evo_mem):
        """Skills are extracted when enough mutations exist."""
        for i in range(8):
            await evo_mem.record_mutation(_make_record(
                child_id=f"c{i}",
                accepted=(i < 6),  # 75% success
                parent_score=0.5,
            ))

        added = await evo_mem.extract_skills(min_samples=5)
        assert added >= 1

        skills = await evo_mem.query_skills(domain="code", parent_score=0.5)
        assert len(skills) >= 1
        assert skills[0].success_rate == pytest.approx(0.75, abs=0.01)
        assert skills[0].sample_count == 8

    @pytest.mark.asyncio
    async def test_query_skills_by_domain_and_score(self, evo_mem):
        """Skills are filtered by domain and parent score range."""
        # Code domain, low score
        for i in range(6):
            await evo_mem.record_mutation(_make_record(
                child_id=f"code_low_{i}", domain="code",
                parent_score=0.2, accepted=True,
            ))
        # Math domain, high score
        for i in range(6):
            await evo_mem.record_mutation(_make_record(
                child_id=f"math_high_{i}", domain="math",
                parent_score=0.8, accepted=False,
            ))

        await evo_mem.extract_skills(min_samples=5)

        # Query code at 0.2 — should find code skill
        code_skills = await evo_mem.query_skills(domain="code", parent_score=0.2)
        assert len(code_skills) >= 1
        assert all(s.domain == "code" or s.domain == "" for s in code_skills)

        # Query math at 0.8 — should find math skill (0% success)
        math_skills = await evo_mem.query_skills(domain="math", parent_score=0.8)
        assert len(math_skills) >= 1

    @pytest.mark.asyncio
    async def test_decay_skills(self, evo_mem):
        """Skill success rates decay over time."""
        for i in range(6):
            await evo_mem.record_mutation(_make_record(
                child_id=f"c{i}", accepted=True,
            ))
        await evo_mem.extract_skills(min_samples=5)

        # Check initial rate
        skills_before = await evo_mem.query_skills(domain="code", parent_score=0.5)
        assert len(skills_before) >= 1
        rate_before = skills_before[0].success_rate

        # Simulate aging by backdating last_used_at
        await evo_mem._db.execute(
            "UPDATE skills SET last_used_at = ?",
            (time.time() - 60 * 86400,),  # 60 days ago
        )
        await evo_mem._db.commit()

        decayed = await evo_mem.decay_skills(half_life_days=30.0)
        assert decayed >= 1

        skills_after = await evo_mem.query_skills(domain="code", parent_score=0.5)
        if skills_after:
            # Rate should be ~25% of original (2 half-lives)
            assert skills_after[0].success_rate < rate_before * 0.5

    @pytest.mark.asyncio
    async def test_persistence_across_sessions(self, tmp_path):
        """Data survives close + reopen."""
        db_path = str(tmp_path / "persist_test.db")

        mem1 = EvolutionMemory(db_path=db_path)
        await mem1.initialize()
        await mem1.record_mutation(_make_record(child_id="persist_1"))
        await mem1.close()

        mem2 = EvolutionMemory(db_path=db_path)
        await mem2.initialize()
        stats = await mem2.stats()
        assert stats["mutations"] == 1
        await mem2.close()

    @pytest.mark.asyncio
    async def test_skill_injection_format(self, evo_mem):
        """Skills have the fields needed for prompt injection."""
        for i in range(6):
            await evo_mem.record_mutation(_make_record(
                child_id=f"c{i}", sampo_action=2, accepted=True,
            ))
        await evo_mem.extract_skills(min_samples=5)

        skills = await evo_mem.query_skills(domain="code", parent_score=0.5)
        assert len(skills) >= 1
        s = skills[0]
        # All fields needed for prompt injection exist
        assert isinstance(s.pattern, str) and len(s.pattern) > 0
        assert isinstance(s.sampo_action, int)
        assert 0.0 <= s.success_rate <= 1.0
        assert s.sample_count >= 5

    @pytest.mark.asyncio
    async def test_stats(self, evo_mem):
        stats = await evo_mem.stats()
        assert stats["mutations"] == 0
        assert stats["skills"] == 0
        assert "db_path" in stats
