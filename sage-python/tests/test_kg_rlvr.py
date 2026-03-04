from sage.topology.kg_rlvr import ProcessRewardModel, SimpleKnowledgeGraph

def test_extract_reasoning_steps():
    prm = ProcessRewardModel()
    content = """
    <think>
    First, I need to understand that YGN-SAGE uses SAMPO.
    Then, SAMPO prevents Amnesia.
    </think>
    Final answer here.
    """
    steps = prm.extract_reasoning_steps(content)
    assert len(steps) == 2
    assert "YGN-SAGE uses SAMPO" in steps[0]
    
def test_calculate_r_path():
    prm = ProcessRewardModel()
    
    # Valid System 3 reasoning
    valid_content = """
    <think>
    YGN-SAGE uses SAMPO to stabilize loops.
    Because SAMPO prevents Amnesia, it is better.
    </think>
    """
    score, details = prm.calculate_r_path(valid_content)
    assert score > 0.0
    assert details["verifiable_ratio"] > 0.0
    
    # No reasoning (System 1)
    invalid_content = "Just giving the answer directly."
    score, details = prm.calculate_r_path(invalid_content)
    assert score == -1.0
    assert "error" in details
