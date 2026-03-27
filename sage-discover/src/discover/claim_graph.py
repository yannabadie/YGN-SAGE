"""src/discover/claim_graph.py — Claim extraction, relation classification, SMT verification."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Try importing OxiZ from sage_core
try:
    import sage_core
    HAS_SMT = hasattr(sage_core, "SmtVerifier")
except ImportError:
    HAS_SMT = False


@dataclass
class Claim:
    claim_id: str
    statement: str
    paper_id: str
    claim_type: str  # finding | method | limitation | hypothesis
    confidence: float
    section: str = "unknown"
    smt_status: str = "not_checked"  # not_checked | consistent | contradictory | unknown
    smt_formula: str | None = None
    relations: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ClaimRelation:
    source_id: str
    target_id: str
    relation_type: str  # supports | extends | refutes | qualifies | independent


# --- Claim Extraction ---

CLAIM_EXTRACTION_PROMPT = """\
Extract the main scientific claims from this text.
For each claim, provide a JSON array with objects containing:
- "statement": the claim in one sentence
- "type": one of "finding", "method", "limitation", "hypothesis"
- "confidence": float 0.0-1.0 (how certain the authors are)

Text:
{text}

Respond ONLY with a JSON array (no markdown fences):"""


async def extract_claims_from_text(
    text: str,
    paper_id: str,
    llm: Any,
) -> list[Claim]:
    """Extract scientific claims from text using an LLM."""
    from sage.llm.base import Message, Role

    prompt = CLAIM_EXTRACTION_PROMPT.format(text=text[:4000])
    messages = [Message(role=Role.USER, content=prompt)]
    response = await llm.generate(messages)

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse claims JSON from LLM response")
        return []

    claims = []
    for i, item in enumerate(items):
        claims.append(Claim(
            claim_id=f"{paper_id}_c{i}",
            statement=item.get("statement", ""),
            paper_id=paper_id,
            claim_type=item.get("type", "finding"),
            confidence=float(item.get("confidence", 0.5)),
        ))
    return claims


# --- Relation Classification ---

RELATION_PROMPT = """\
What is the relationship between these two scientific claims?

Claim A: "{claim_a}"
Claim B: "{claim_b}"

Choose exactly one:
- supports: B provides evidence for A
- extends: B builds upon A with new contributions
- refutes: B contradicts A
- qualifies: B limits the scope of A
- independent: no direct relationship

Respond with a single word:"""

VALID_RELATIONS = {"supports", "extends", "refutes", "qualifies", "independent"}


async def classify_relation(
    claim_a: Claim,
    claim_b: Claim,
    llm: Any,
) -> ClaimRelation:
    """Classify the relationship between two claims."""
    from sage.llm.base import Message, Role

    prompt = RELATION_PROMPT.format(claim_a=claim_a.statement, claim_b=claim_b.statement)
    messages = [Message(role=Role.USER, content=prompt)]
    response = await llm.generate(messages)

    rel_type = response.content.strip().lower()
    if rel_type not in VALID_RELATIONS:
        rel_type = "independent"

    return ClaimRelation(
        source_id=claim_a.claim_id,
        target_id=claim_b.claim_id,
        relation_type=rel_type,
    )


# --- SMT Translation ---

_PERF_PATTERN = re.compile(
    r"(?:achieve|attain|reach|obtain|report)s?\s+(\d+(?:\.\d+)?)\s*%\s*(?:accuracy|precision|recall|F1|score)",
    re.IGNORECASE,
)
_IMPROVE_PATTERN = re.compile(
    r"improv(?:e|es|ing)\s+(?:over|upon|compared to)\s+.*?by\s+(\d+(?:\.\d+)?)\s*(?:percentage|pp|%)",
    re.IGNORECASE,
)
_COMPARE_PATTERN = re.compile(
    r"(\w+)\s+(?:achieves?|attains?)\s+(\d+(?:\.\d+)?)\s*%.*?(?:on|for)\s+(?:benchmark\s+)?(\w+)",
    re.IGNORECASE,
)


def translate_claim_to_smt(claim: Claim) -> str | None:
    """Translate a quantitative claim to SMT-LIB2 formula.

    Tries patterns in order of specificity:
    1. Method+benchmark comparison (most specific, best for grouping)
    2. Improvement over baseline
    3. Generic performance claim (least specific)
    """
    text = claim.statement

    # Most specific first: "Method achieves X% on benchmark B"
    m = _COMPARE_PATTERN.search(text)
    if m:
        method = m.group(1).lower()
        val = int(float(m.group(2)))
        benchmark = m.group(3).lower()
        var = f"perf_{method}_{benchmark}"
        return f"(= {var} {val})"

    # "improves over baseline by X%"
    m = _IMPROVE_PATTERN.search(text)
    if m:
        delta = int(float(m.group(1)))
        var = f"improvement_{claim.claim_id}"
        return f"(= {var} {delta})"

    # Generic: "achieves X% accuracy"
    m = _PERF_PATTERN.search(text)
    if m:
        val = int(float(m.group(1)))
        var = f"perf_{claim.claim_id}"
        return f"(= {var} {val})"

    return None


# --- SMT Verification ---

def _extract_numeric_value(formula: str) -> int | None:
    """Extract the numeric value from an SMT formula like '(= var 92)'."""
    m = re.search(r"\b(\d+)\)?$", formula.strip())
    return int(m.group(1)) if m else None


def _extract_variable_key(formula: str) -> str | None:
    """Extract the variable name from an SMT formula for grouping."""
    # Match patterns like perf_method_benchmark or perf_claimid
    m = re.search(r"(perf_\w+|improvement_\w+)", formula)
    if not m:
        return None
    key = m.group(1)
    # Normalize: strip the claim-specific suffix to group by method+benchmark
    # e.g., "perf_method_benchmark" stays, "perf_c1" becomes "perf"
    parts = key.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])  # perf_method_benchmark
    return key


def verify_claim_cluster(claims: list[Claim]) -> str:
    """Verify logical consistency of a cluster of related claims.

    Uses sage_core.SmtVerifier.verify_arithmetic(val1, val2, tolerance) to check
    if quantitative claims about the same subject are consistent (tolerance=0).

    Returns: "consistent" | "contradictory" | "unknown"
    """
    if not HAS_SMT:
        return "unknown"

    # Extract numeric assertions from claims
    assertions: list[tuple[str, int, Claim]] = []  # (variable_key, value, claim)
    for claim in claims:
        formula = translate_claim_to_smt(claim)
        if formula:
            claim.smt_formula = formula
            key = _extract_variable_key(formula)
            val = _extract_numeric_value(formula)
            if key and val is not None:
                assertions.append((key, val, claim))

    if len(assertions) < 2:
        return "unknown"

    # Group by variable key and check consistency within each group
    groups: dict[str, list[tuple[int, Claim]]] = {}
    for key, val, claim in assertions:
        groups.setdefault(key, []).append((val, claim))

    try:
        verifier = sage_core.SmtVerifier()
        found_contradiction = False

        for key, entries in groups.items():
            if len(entries) < 2:
                continue
            # Check all pairs within this group
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    val_a, claim_a = entries[i]
                    val_b, claim_b = entries[j]
                    # tolerance=0: values must be exactly equal to be consistent
                    consistent = verifier.verify_arithmetic(val_a, val_b, 0)
                    if not consistent:
                        claim_a.smt_status = "contradictory"
                        claim_b.smt_status = "contradictory"
                        found_contradiction = True

        if found_contradiction:
            return "contradictory"

        # All checked pairs were consistent
        for _, _, claim in assertions:
            if claim.smt_status == "not_checked":
                claim.smt_status = "consistent"
        return "consistent"

    except Exception as e:
        logger.warning("SMT verification failed: %s", e)
        return "unknown"
