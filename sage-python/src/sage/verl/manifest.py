"""Training manifest — artefact contract between pipeline stages.

Each training stage (SFT, GRPO, Phase C, export) writes a manifest.json
to its output directory. The next stage reads it to find the correct
input artefact. The runtime loader reads the final manifest to load
the trained policy model.

This is the SINGLE source of truth for artefact provenance.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("manifest")

MANIFEST_FILENAME = "sage_training_manifest.json"
SCHEMA_VERSION = "1.0"


@dataclass
class TrainingManifest:
    """Artefact contract for the Nemotron E2E training pipeline.

    Written by: sft_warmup.py, train_topology_v5.sh, train_phase_c_custom.py,
                post_training_pipeline.py
    Read by:    next training stage, llm_caller.py (runtime)
    """
    base_model: str = "nvidia/Nemotron-Orchestrator-8B"
    stage: str = ""           # "sft" | "grpo_warmup" | "phase_c" | "merged" | "exported"
    format: str = ""          # "lora" | "merged" | "gguf"
    chat_template: str = "qwen3"
    schema_version: str = SCHEMA_VERSION
    runtime_compatible: bool = False  # True only after merge/export
    output_path: str = ""     # absolute path to the artefact
    parent_manifest: str = "" # path to the previous stage manifest (provenance chain)
    dataset: str = ""         # dataset used for this stage
    dataset_size: int = 0
    algorithm: str = ""       # "sft" | "grpo" | "gigpo_custom"
    lr: float = 0.0
    epochs: int = 0
    steps_completed: int = 0
    final_reward_mean: float = 0.0
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str) -> str:
        """Write manifest.json to the given directory. Returns the path."""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        path = Path(directory) / MANIFEST_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        log.info("Manifest written: %s (stage=%s)", path, self.stage)
        return str(path)

    @classmethod
    def load(cls, path: str) -> TrainingManifest:
        """Load manifest from a file or directory."""
        p = Path(path)
        if p.is_dir():
            p = p / MANIFEST_FILENAME
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {p}")
        with open(p) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def find_latest(cls, base_dir: str, stage: str | None = None) -> TrainingManifest | None:
        """Find the most recent manifest in a directory tree."""
        candidates = []
        for p in Path(base_dir).rglob(MANIFEST_FILENAME):
            try:
                m = cls.load(str(p))
                if stage is None or m.stage == stage:
                    candidates.append(m)
            except Exception:
                continue
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.timestamp or "")
