import sys
import types

# Mock sage_core so sage package can import without Rust extension
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

from sage.topology.kg_rlvr import ProcessRewardModel, FormalKnowledgeGraph

def test_extract_reasoning_steps():
    prm = ProcessRewardModel()
    content = """
    <think>
    First, I assert bounds(50, 100) to ensure memory safety.
    Then, I assert loop(i) where i is bounded.
    </think>
    Final answer here.
    """
    steps = prm.extract_reasoning_steps(content)
    assert len(steps) == 2
    assert "assert bounds(50, 100)" in steps[0]
    
def test_calculate_r_path():
    prm = ProcessRewardModel()
    
    # Valid System 3 reasoning (Z3 provable bounds)
    valid_content = """
    <think>
    I am writing a memory access. Let's assert bounds(5, 100).
    Because the bounds are safe, it will not crash.
    </think>
    """
    score, details = prm.calculate_r_path(valid_content)
    assert score > 0.0
    assert details["verifiable_ratio"] > 0.0
    
    # Invalid System 3 reasoning (Z3 provable hallucination)
    hallucination_content = """
    <think>
    Let's assert bounds(150, 100). This should be fine.
    </think>
    """
    score, details = prm.calculate_r_path(hallucination_content)
    assert score == -1.0 # Provably false
    
    # No reasoning (System 1)
    invalid_content = "Just giving the answer directly."
    score, details = prm.calculate_r_path(invalid_content)
    assert score == -1.0
    assert "error" in details


def test_arithmetic_verification():
    kg = FormalKnowledgeGraph()
    score = kg.verify_step("assert arithmetic(2+2, 4)")
    assert isinstance(score, float)


def test_process_reward_model_with_think_blocks():
    prm = ProcessRewardModel()
    content = """<think>
assert bounds(5, 100)
assert loop(iterations)
checking ebpf latency
</think>"""
    r_path, details = prm.calculate_r_path(content)
    assert details["total_steps"] == 3
    assert isinstance(r_path, float)


def test_score_with_z3_validator():
    """Test the Z3Validator backend."""
    prm = ProcessRewardModel()
    score, details = prm.score_with_z3(["bounds(5, 100)", "bounds(0, 10)"])
    assert isinstance(score, float)
    assert "backend" in details or "error" in details


def test_score_with_z3_detects_violation():
    """Test that Z3 catches bounds violations."""
    prm = ProcessRewardModel()
    score, details = prm.score_with_z3(["bounds(200, 100)"])
    if "error" not in details:
        assert score == -1.0
        assert not details["safe"]


def test_verify_invariant_blocks_code_injection():
    """Verify that malicious pre/post strings cannot execute arbitrary code."""
    kg = FormalKnowledgeGraph()
    malicious_pre = "__import__('os').system('echo pwned')"
    malicious_post = "x > 0"
    result = kg.verify_invariant(malicious_pre, malicious_post)
    assert result is False, "Malicious input must fail-closed (return False), not pass"


def test_verify_invariant_accepts_valid_z3_expressions():
    """Verify that legitimate Z3 constraint strings still work."""
    kg = FormalKnowledgeGraph()
    if not kg.has_z3:
        return  # Skip if z3 not installed
    result = kg.verify_invariant("x > 0", "x > -1")
    assert result is True
    result = kg.verify_invariant("x > 10", "x > 20")
    assert result is False


def test_verify_invariant_fails_closed_on_unparseable():
    """Verify that unparseable expressions fail-closed (return False)."""
    kg = FormalKnowledgeGraph()
    result = kg.verify_invariant("not a valid expression ???", "also garbage")
    assert result is False, "Unparseable input must fail-closed"


def test_verify_arithmetic_evaluates_expr():
    """Z3-03: verify_arithmetic must actually evaluate the expression."""
    fkg = FormalKnowledgeGraph()

    # Simple constant - should work
    assert fkg.verify_arithmetic("42", 42) is True
    assert fkg.verify_arithmetic("42", 43) is False
    assert fkg.verify_arithmetic("42", 43, tolerance=1) is True

    # Expression that ast.literal_eval can't handle
    # _safe_z3_eval should try, but simple arithmetic strings
    # won't parse as Z3 without z3 syntax, so fail-closed
    assert fkg.verify_arithmetic("invalid", 0) is False

    # Without Z3 - fail-closed
    fkg.has_z3 = False
    assert fkg.verify_arithmetic("42", 42) is False


def test_fail_closed_without_z3():
    """Z3-07: Without z3-solver, formal verification must fail-closed."""
    fkg = FormalKnowledgeGraph()
    fkg.has_z3 = False

    # prove_memory_safety: should use Python bounds check
    assert fkg.prove_memory_safety(5, 10) is True   # 0 <= 5 < 10
    assert fkg.prove_memory_safety(-1, 10) is False  # -1 < 0
    assert fkg.prove_memory_safety(10, 10) is False  # 10 >= 10

    # All others: fail-closed (False)
    assert fkg.check_loop_bound("n", 100) is False
    assert fkg.verify_arithmetic("2+2", 4) is False
    assert fkg.verify_invariant("x > 0", "x > 0") is False
