from sage.bench.routing_quality import run_routing_quality


def test_routing_quality_above_threshold():
    """Routing must achieve >= 80% accuracy on ground-truth labels."""
    results = run_routing_quality()
    assert results["accuracy"] >= 0.80, (
        f"Routing accuracy {results['accuracy']:.1%} < 80% threshold. "
        f"Under-routed: {results['under_routing_rate']:.1%}"
    )
    # Under-routing is worse than over-routing (safety)
    assert results["under_routing_rate"] <= 0.10, (
        f"Under-routing rate {results['under_routing_rate']:.1%} > 10% threshold"
    )
