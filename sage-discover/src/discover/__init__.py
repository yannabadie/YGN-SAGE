"""sage-discover: Flagship Research & Discovery Agent.

Brings together all 5 YGN-SAGE cognitive pillars to autonomously
explore research domains, generate hypotheses, evolve solutions,
and evaluate discoveries.
"""

__version__ = "0.1.0"

try:
    from discover.workflow import DiscoverWorkflow, DiscoverConfig
    from discover.researcher import ResearchAgent
    from discover.pipeline import run_pipeline, PipelineReport

    __all__ = [
        "DiscoverWorkflow",
        "DiscoverConfig",
        "ResearchAgent",
        "run_pipeline",
        "PipelineReport",
    ]
except ImportError:
    # sage-python not installed; only discover.store and other standalone modules available
    __all__ = []
