"""Routing quality benchmark with human-labeled ground truth.

Unlike the self-consistency benchmark (routing.py), this tests against
independently labeled tasks. The labels represent MINIMUM required system:
  1 = S1 (trivial), 2 = S2 (code/reasoning), 3 = S3 (formal/complex)
"""
from sage.strategy.metacognition import ComplexityRouter

GROUND_TRUTH: list[tuple[str, int, str]] = [
    # S1 tasks (trivial — should NOT be over-routed)
    ("What is 2+2?", 1, "trivial arithmetic"),
    ("What's the capital of France?", 1, "trivial factual"),
    ("Translate 'hello' to French", 1, "trivial translation"),
    ("What color is the sky?", 1, "trivial factual"),
    ("List 3 fruits", 1, "trivial enumeration"),
    ("Who wrote Romeo and Juliet?", 1, "trivial factual"),
    ("What is 15% of 200?", 1, "simple math"),
    ("Define 'photosynthesis' in one sentence", 1, "trivial definition"),
    ("Is 17 a prime number?", 1, "trivial check"),
    ("What year did WWII end?", 1, "trivial factual"),
    # S2 tasks (code/reasoning — need empirical validation)
    ("Write a function to check if a number is prime", 2, "simple algorithm"),
    ("Write a Python bubble sort implementation", 2, "basic algorithm"),
    ("Create a REST API endpoint with FastAPI", 2, "framework code"),
    ("Write unit tests for a calculator class", 2, "test code"),
    ("Implement a binary search tree insert method", 2, "data structure"),
    ("Write a regex to validate email addresses", 2, "pattern matching"),
    ("Create a Python decorator for caching", 2, "intermediate Python"),
    ("Write a function to merge two sorted arrays", 2, "algorithm"),
    ("Implement a simple linked list in Python", 2, "data structure"),
    ("Write a CSV parser without using the csv module", 2, "parsing logic"),
    # S3 tasks (formal/complex — need deep reasoning)
    ("Prove that sqrt(2) is irrational", 3, "mathematical proof"),
    ("Design a distributed consensus protocol", 3, "distributed systems"),
    ("Write a formal specification for a banking transaction system", 3, "formal spec"),
    ("Implement a lock-free concurrent queue in Rust", 3, "concurrent programming"),
    ("Prove the correctness of quicksort using loop invariants", 3, "algorithm proof"),
    ("Design a capability-based security model for microservices", 3, "security architecture"),
    ("Analyze the time complexity of the Ackermann function", 3, "complexity theory"),
    ("Implement a type checker for a simple lambda calculus", 3, "PL theory"),
    ("Design a CRDT for collaborative text editing", 3, "distributed data"),
    ("Write a Z3 proof for mutual exclusion in Peterson's algorithm", 3, "formal verification"),
]


def run_routing_quality() -> dict:
    """Run routing quality benchmark against ground truth."""
    router = ComplexityRouter()
    results = {"correct": 0, "over_routed": 0, "under_routed": 0, "total": len(GROUND_TRUTH)}
    details = []

    for task, min_system, rationale in GROUND_TRUTH:
        profile = router.assess_complexity(task)
        if profile.complexity <= 0.50:
            routed = 1
        elif profile.complexity > 0.65:
            routed = 3
        else:
            routed = 2

        if routed >= min_system:
            results["correct"] += 1
        if routed > min_system:
            results["over_routed"] += 1
        if routed < min_system:
            results["under_routed"] += 1

        details.append({
            "task": task[:60], "expected": min_system, "routed": routed,
            "complexity": round(profile.complexity, 3),
            "correct": routed >= min_system,
        })

    results["accuracy"] = results["correct"] / results["total"]
    results["over_routing_rate"] = results["over_routed"] / results["total"]
    results["under_routing_rate"] = results["under_routed"] / results["total"]
    results["details"] = details
    return results
