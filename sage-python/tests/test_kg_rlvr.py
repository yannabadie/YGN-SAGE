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
