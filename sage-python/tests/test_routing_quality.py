from sage.bench.routing_quality import run_routing_quality


def test_routing_quality_above_threshold():
    """Routing must achieve minimum accuracy on ground-truth labels.

    Production routing uses kNN (92%) or ONNX BERT.
    The degraded keyword-count heuristic (ComplexityRouter sync path)
    is a last-resort fallback — lower threshold applies.
    """
    results = run_routing_quality()
    # Degraded heuristic: 40% expected (only matches complex keywords).
    # kNN/ONNX production paths: 80%+ expected (tested via routing_gt benchmark).
    assert results["accuracy"] >= 0.33, (
        f"Routing accuracy {results['accuracy']:.1%} < 33% threshold. "
        f"Even degraded heuristic should match trivial S1 tasks."
    )
