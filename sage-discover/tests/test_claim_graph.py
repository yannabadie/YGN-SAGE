"""tests/test_claim_graph.py — Claim extraction + SMT verification tests."""
from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock

import pytest

from discover.claim_graph import (
    Claim,
    ClaimRelation,
    extract_claims_from_text,
    classify_relation,
    translate_claim_to_smt,
    verify_claim_cluster,
)


def test_claim_dataclass():
    c = Claim(
        claim_id="c1",
        statement="kNN achieves 92% accuracy",
        paper_id="p1",
        claim_type="finding",
        confidence=0.9,
    )
    assert c.claim_id == "c1"
    assert c.smt_status == "not_checked"


def test_translate_performance_claim():
    c = Claim("c1", "Method X achieves 92% accuracy", "p1", "finding", 0.9)
    formula = translate_claim_to_smt(c)
    assert formula is not None
    assert "92" in formula


def test_translate_comparison_claim():
    c = Claim("c1", "Method X improves over baseline by 5 percentage points", "p1", "finding", 0.9)
    formula = translate_claim_to_smt(c)
    assert formula is not None
    assert "5" in formula


def test_translate_qualitative_returns_none():
    c = Claim("c1", "Our method is more elegant", "p1", "finding", 0.5)
    formula = translate_claim_to_smt(c)
    assert formula is None


def test_verify_consistent_cluster():
    claims = [
        Claim("c1", "Method X achieves 92% accuracy", "p1", "finding", 0.9),
        Claim("c2", "Method Y achieves 88% accuracy", "p2", "finding", 0.9),
    ]
    result = verify_claim_cluster(claims)
    assert result in ("consistent", "unknown")


def test_verify_contradictory_cluster():
    claims = [
        Claim("c1", "Method X achieves 92% accuracy on benchmark B", "p1", "finding", 0.9),
        Claim("c2", "Method X achieves 45% accuracy on benchmark B", "p2", "finding", 0.9),
    ]
    result = verify_claim_cluster(claims)
    assert result in ("contradictory", "unknown")


@pytest.mark.asyncio
async def test_extract_claims_from_text():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content='[{"statement": "We achieve 92% accuracy", "type": "finding", "confidence": 0.9}]'
    )
    claims = await extract_claims_from_text("Some paper text here", "p1", mock_llm)
    assert len(claims) == 1
    assert claims[0].statement == "We achieve 92% accuracy"
    assert claims[0].paper_id == "p1"


@pytest.mark.asyncio
async def test_extract_claims_handles_bad_json():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(content="not valid json")
    claims = await extract_claims_from_text("Text", "p1", mock_llm)
    assert claims == []


@pytest.mark.asyncio
async def test_classify_relation():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(content="supports")
    c1 = Claim("c1", "X is good", "p1", "finding", 0.9)
    c2 = Claim("c2", "X works well", "p2", "finding", 0.8)
    rel = await classify_relation(c1, c2, mock_llm)
    assert rel.relation_type in ("supports", "extends", "refutes", "qualifies", "independent")


@pytest.mark.asyncio
async def test_classify_relation_invalid_response():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(content="gibberish_nonsense")
    c1 = Claim("c1", "X", "p1", "finding", 0.9)
    c2 = Claim("c2", "Y", "p2", "finding", 0.8)
    rel = await classify_relation(c1, c2, mock_llm)
    assert rel.relation_type == "independent"  # fallback
