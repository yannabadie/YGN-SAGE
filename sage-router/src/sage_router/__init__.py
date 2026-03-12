"""sage-router: Cost-aware S1/S2/S3 LLM routing.

Standalone routing library extracted from YGN-SAGE.
pip install sage-router          # Zero dependencies
pip install sage-router[knn]     # With kNN routing (numpy)
"""
from sage_router.metacognition import ComplexityRouter, CognitiveProfile, RoutingDecision
from sage_router.quality_estimator import QualityEstimator

__all__ = [
    "ComplexityRouter",
    "CognitiveProfile",
    "RoutingDecision",
    "QualityEstimator",
]

__version__ = "0.1.0"
