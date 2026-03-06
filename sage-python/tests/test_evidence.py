"""Tests for EvidenceRecord and EvidenceLevel."""
from __future__ import annotations

import pytest
from sage.evidence import EvidenceRecord, EvidenceLevel


def test_evidence_record_creation():
    er = EvidenceRecord(
        level=EvidenceLevel.HEURISTIC,
        proof_strength=0.3,
        external_validity=False,
        coverage=0.12,
        assumptions=["labels calibrated to heuristic"],
    )
    assert er.level == EvidenceLevel.HEURISTIC
    assert er.proof_strength == 0.3


def test_evidence_level_ordering():
    assert EvidenceLevel.HEURISTIC.value < EvidenceLevel.SOLVER_PROVED.value


def test_evidence_record_to_dict():
    er = EvidenceRecord(level=EvidenceLevel.EMPIRICALLY_VALIDATED, proof_strength=0.95)
    d = er.to_dict()
    assert d["level"] == "empirically_validated"
    assert "timestamp" in d


def test_evidence_readme_projection():
    """README uses simple string, not full record."""
    er = EvidenceRecord(level=EvidenceLevel.CHECKED, proof_strength=0.6)
    assert er.readme_label() == "checked"
