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
    """Translate a quantitative claim to SMT-LIB2 formula."""
    text = claim.statement

    m = _PERF_PATTERN.search(text)
    if m:
        val = int(float(m.group(1)))
        var = f"perf_{claim.claim_id}"
        return f"(= {var} {val})"

    m = _IMPROVE_PATTERN.search(text)
    if m:
        delta = int(float(m.group(1)))
        var = f"improvement_{claim.claim_id}"
        return f"(= {var} {delta})"

    m = _COMPARE_PATTERN.search(text)
    if m:
        method = m.group(1).lower()
        val = int(float(m.group(2)))
        benchmark = m.group(3).lower()
        var = f"perf_{method}_{benchmark}"
        return f"(= {var} {val})"

    return None


# --- SMT Verification ---

def verify_claim_cluster(claims: list[Claim]) -> str:
    """Verify logical consistency of a cluster of related claims.

    Returns: "consistent" | "contradictory" | "unknown"
    """
    if not HAS_SMT:
        return "unknown"

    formulas = []
    variables = set()

    for claim in claims:
        formula = translate_claim_to_smt(claim)
        if formula:
            claim.smt_formula = formula
            formulas.append(formula)
            for var_match in re.finditer(r"(?:perf|improvement)_\w+", formula):
                variables.add(var_match.group())

    if len(formulas) < 2:
        return "unknown"

    try:
        verifier = sage_core.SmtVerifier()
        verifier.set_logic("QF_LIA")

        for var in variables:
            verifier.declare_const(var, "Int")

        for formula in formulas:
            verifier.assert_(formula)

        result = verifier.check_sat()
        if result == "sat":
            return "consistent"
        elif result == "unsat":
            return "contradictory"
        else:
            return "unknown"
    except Exception as e:
        logger.warning("SMT verification failed: %s", e)
        return "unknown"
