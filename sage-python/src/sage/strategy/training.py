"""Training pipeline for AdaptiveRouter BERT classifier.

Collects routing feedback, exports training data, triggers retraining.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)
TRAINING_DATA_DIR = Path.home() / ".sage" / "routing_training"


@dataclass
class TrainingExample:
    task: str
    routed_tier: int
    actual_quality: float
    latency_ms: float
    cost_usd: float
    label: int  # computed optimal tier


def compute_label(routed_tier: int, quality: float, cost_usd: float) -> int:
    """Compute optimal routing label from outcome.

    - quality >= 0.8 -> current tier was sufficient
    - quality < 0.5 and tier < 3 -> should have escalated
    - S3 with low quality stays S3 (no S4)
    """
    if quality < 0.5 and routed_tier < 3:
        return routed_tier + 1
    return routed_tier


def export_training_data(
    feedback: list[dict],
    output_path: Path | None = None,
) -> Path:
    """Export feedback buffer to JSONL training file."""
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        import time

        output_path = TRAINING_DATA_DIR / f"feedback_{int(time.time())}.jsonl"

    examples = []
    for fb in feedback:
        label = compute_label(
            fb["routed_tier"], fb["actual_quality"], fb.get("cost_usd", 0.0)
        )
        ex = TrainingExample(
            task=fb["task"],
            routed_tier=fb["routed_tier"],
            actual_quality=fb["actual_quality"],
            latency_ms=fb.get("latency_ms", 0.0),
            cost_usd=fb.get("cost_usd", 0.0),
            label=label,
        )
        examples.append(ex)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")

    log.info("Exported %d training examples to %s", len(examples), output_path)
    return output_path
