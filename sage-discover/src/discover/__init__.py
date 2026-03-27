"""sage-discover: Knowledge Discovery Engine with SMT-verified claims."""

__version__ = "0.2.0"

from discover.pipeline import run_pipeline, PipelineReport

__all__ = [
    "run_pipeline",
    "PipelineReport",
]
