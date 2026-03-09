"""Routing quality benchmark: measures routing accuracy against human-labeled ground truth.

Unlike the self-consistency benchmark (routing.py), this measures whether
the ComplexityRouter assigns appropriate complexity levels to tasks with
known difficulty.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sage.strategy.metacognition import ComplexityRouter

log = logging.getLogger(__name__)

# Ground truth: (task, minimum_acceptable_system, rationale)
# Labels assigned by human judgment, NOT calibrated against the router.
GROUND_TRUTH: list[tuple[str, int, str]] = [
    # S1 — trivial tasks (factual, arithmetic, simple lookup)
    ("What is 2+2?", 1, "trivial arithmetic"),
    ("What is the capital of France?", 1, "trivial factual recall"),
    ("Convert 100 Celsius to Fahrenheit", 1, "simple formula"),
    ("What day of the week is Christmas 2026?", 1, "calendar lookup"),
    ("Translate 'hello' to Spanish", 1, "simple translation"),
    ("How many continents are there?", 1, "basic geography"),
    ("What is the square root of 144?", 1, "basic math"),
    ("Define the word 'algorithm'", 1, "dictionary definition"),
    ("What color is the sky on a clear day?", 1, "trivial common knowledge"),
    ("Name three fruits", 1, "simple enumeration"),
    ("What is 15% of 200?", 1, "percentage calculation"),
    ("Spell the word 'necessary'", 1, "trivial spelling"),
    ("What year did World War II end?", 1, "basic history"),
    ("Convert 5 miles to kilometers", 1, "unit conversion"),
    ("What is the chemical symbol for water?", 1, "basic science"),

    # S2 — moderate tasks (short code, simple analysis, structured output)
    ("Write a Python function to check if a number is prime", 2, "simple algorithm"),
    ("Write a bubble sort implementation in Python", 2, "basic sorting algorithm"),
    ("Explain the difference between a list and a tuple in Python", 2, "conceptual comparison"),
    ("Write a SQL query to find duplicate emails in a users table", 2, "moderate SQL"),
    ("Parse this JSON and extract all email addresses", 2, "data extraction"),
    ("Write a regex to validate email addresses", 2, "regex pattern"),
    ("Implement a stack using a Python list", 2, "simple data structure"),
    ("Write a function to reverse a linked list", 2, "classic algorithm"),
    ("Create a REST API endpoint with Flask that returns user data", 2, "web development"),
    ("Write unit tests for a calculator class", 2, "testing"),
    ("Explain how HTTPS works", 2, "technical explanation"),
    ("Write a Python decorator that logs function calls", 2, "intermediate Python"),
    ("Implement binary search on a sorted array", 2, "classic algorithm"),
    ("Write a function to find the longest common subsequence", 2, "dynamic programming"),
    ("Design a database schema for a blog with posts and comments", 2, "schema design"),

    # S3 — complex tasks (multi-step reasoning, architecture, debugging)
    ("Debug a race condition in async Rust code with deadlock on Arc<Mutex>", 3, "complex concurrent debugging"),
    ("Implement a B+ tree with concurrent insert and delete operations", 3, "complex data structure"),
    ("Design a distributed consensus protocol for a 5-node cluster", 3, "distributed systems"),
    ("Analyze this codebase and propose a refactoring strategy for the legacy module", 3, "architectural analysis"),
    ("Write a compiler frontend (lexer + parser) for a simple expression language", 3, "compiler design"),
    ("Implement a lock-free concurrent hash map in Rust", 3, "advanced concurrent programming"),
    ("Design a multi-tenant SaaS architecture with data isolation", 3, "system architecture"),
    ("Prove that this algorithm terminates for all inputs using structural induction", 3, "formal proof"),
    ("Implement RAFT consensus with leader election and log replication", 3, "distributed protocol"),
    ("Build a query optimizer for a SQL engine supporting joins and aggregations", 3, "database internals"),
    ("Debug why this Kubernetes pod keeps crashing with OOM despite 8GB limit", 3, "complex ops debugging"),
    ("Design an event sourcing system with CQRS for a financial trading platform", 3, "complex architecture"),
    ("Implement a garbage collector using tri-color marking", 3, "systems programming"),
    ("Analyze the time complexity of this recursive function with memoization", 3, "algorithmic analysis"),
    ("Build a real-time collaborative editor with operational transforms", 3, "complex distributed system"),
]


@dataclass
class RoutingQualityResult:
    """Result of routing quality benchmark."""
    total: int
    correct: int
    under_routed: int  # S1 when S3 was needed (dangerous)
    over_routed: int   # S3 when S1 sufficed (wasteful)
    accuracy: float
    under_routing_rate: float
    over_routing_rate: float
    details: list[dict]


def run_routing_quality(router: ComplexityRouter | None = None) -> RoutingQualityResult:
    """Run routing quality benchmark against ground truth.

    Args:
        router: ComplexityRouter to test. Uses default if None.

    Returns:
        RoutingQualityResult with accuracy and breakdown.
    """
    if router is None:
        router = ComplexityRouter()

    correct = 0
    under_routed = 0
    over_routed = 0
    details = []

    for task, min_system, rationale in GROUND_TRUTH:
        profile = router.assess_complexity(task)
        decision = router.route(profile)
        actual = decision.system

        is_correct = actual >= min_system
        is_under = actual < min_system
        is_over = actual > min_system

        if is_correct:
            correct += 1
        if is_under:
            under_routed += 1
        if is_over:
            over_routed += 1

        details.append({
            "task": task[:60],
            "expected_min": min_system,
            "actual": actual,
            "correct": is_correct,
            "complexity": round(profile.complexity, 3),
            "rationale": rationale,
        })

    total = len(GROUND_TRUTH)
    return RoutingQualityResult(
        total=total,
        correct=correct,
        under_routed=under_routed,
        over_routed=over_routed,
        accuracy=correct / total if total else 0.0,
        under_routing_rate=under_routed / total if total else 0.0,
        over_routing_rate=over_routed / total if total else 0.0,
        details=details,
    )
