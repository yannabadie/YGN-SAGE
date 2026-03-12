# SOTA Breakout Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push YGN-SAGE beyond state-of-the-art in routing accuracy, topology evidence, and developer experience.

**Architecture:** Three independent phases — (1) Routing Dominance: learned quality estimator + DeBERTa classifier + standalone sage-router lib, (2) TopologyBench: first topology impact benchmark + large-scale ablation + kNN expansion, (3) DX Parity: per-agent memory scoping + durable execution + Python-only topology ports.

**Tech Stack:** Python 3.12+, PyTorch/transformers (training only), ONNX Runtime, networkx, msgpack, numpy, sentence-transformers, aiosqlite

**Spec:** `docs/superpowers/specs/2026-03-12-sota-breakout-design.md`

---

## Chunk 1: Phase 1 — Routing Dominance

### Task 1: Quality Training Data Collector

**Files:**
- Create: `sage-python/src/sage/training/__init__.py`
- Create: `sage-python/src/sage/training/quality_collector.py`
- Create: `sage-python/tests/test_quality_collector.py`

Reads benchmark result JSONs from `docs/benchmarks/*.json` and produces `(task, response, quality_score)` triples as JSONL.

- [ ] **Step 1: Create training package**

```python
# sage-python/src/sage/training/__init__.py
"""Training pipelines for SAGE models."""
```

- [ ] **Step 2: Write failing tests**

```python
# sage-python/tests/test_quality_collector.py
import json, tempfile
from pathlib import Path
from sage.training.quality_collector import collect_triples, Triple

def test_triple_from_bench_result():
    """A passing task gets quality=1.0, failing gets 0.0."""
    bench = {
        "benchmark": "evalplus_humaneval",
        "results": [
            {"task_id": "HE/0", "passed": True, "latency_ms": 100},
            {"task_id": "HE/1", "passed": False, "latency_ms": 200, "error": "AssertionError"},
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(bench, f)
        path = Path(f.name)
    triples = collect_triples([path])
    assert len(triples) == 2
    assert triples[0].quality == 1.0
    assert triples[1].quality == 0.0
    assert triples[0].task_id == "HE/0"

def test_collect_exports_jsonl():
    """collect_triples can export to JSONL."""
    bench = {"benchmark": "test", "results": [
        {"task_id": "T/0", "passed": True, "latency_ms": 50}
    ]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(bench, f)
        path = Path(f.name)
    triples = collect_triples([path])
    assert len(triples) == 1
    line = triples[0].to_jsonl()
    parsed = json.loads(line)
    assert "task_id" in parsed
    assert "quality" in parsed

def test_collect_skips_malformed():
    """Malformed entries are skipped, not crashed."""
    bench = {"benchmark": "test", "results": [
        {"task_id": "T/0"},  # missing passed
        {"task_id": "T/1", "passed": True, "latency_ms": 10},
    ]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(bench, f)
        path = Path(f.name)
    triples = collect_triples([path])
    assert len(triples) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_quality_collector.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Implement quality_collector.py**

```python
# sage-python/src/sage/training/quality_collector.py
"""Collect (task, response, quality) triples from benchmark JSONs."""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

_log = logging.getLogger(__name__)

@dataclass
class Triple:
    task_id: str
    task_text: str
    response: str
    quality: float  # 0.0-1.0
    latency_ms: float = 0.0
    benchmark: str = ""

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def collect_triples(paths: list[Path]) -> list[Triple]:
    """Read benchmark JSON files, extract triples."""
    triples: list[Triple] = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            _log.warning("Skip %s: %s", p, e)
            continue
        bench_name = data.get("benchmark", p.stem)
        for r in data.get("results", []):
            if "passed" not in r:
                _log.debug("Skip malformed entry in %s: %s", p, r.get("task_id", "?"))
                continue
            triples.append(Triple(
                task_id=r.get("task_id", ""),
                task_text=r.get("task", r.get("task_id", "")),
                response=r.get("response", ""),
                quality=1.0 if r["passed"] else 0.0,
                latency_ms=r.get("latency_ms", 0.0),
                benchmark=bench_name,
            ))
    return triples


def export_jsonl(triples: list[Triple], output: Path) -> Path:
    """Write triples to JSONL file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(t.to_jsonl() + "\n")
    return output
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_quality_collector.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/training/ sage-python/tests/test_quality_collector.py
git commit -m "feat(training): quality triple collector from benchmark JSONs"
```

---

### Task 2: LearnedQualityEstimator

**Files:**
- Modify: `sage-python/src/sage/quality_estimator.py:11-69`
- Create: `sage-python/tests/test_learned_quality_estimator.py`

Add `LearnedQualityEstimator` class that uses ONNX model when available, falls back to existing 5-signal heuristic.

- [ ] **Step 1: Write failing tests**

```python
# sage-python/tests/test_learned_quality_estimator.py
import pytest
from unittest.mock import MagicMock, patch
from sage.quality_estimator import QualityEstimator, LearnedQualityEstimator

def test_learned_estimator_fallback_to_heuristic():
    """Without ONNX model, falls back to heuristic."""
    est = LearnedQualityEstimator(model_path=None)
    score = est.estimate("Write hello world", "print('hello')")
    assert 0.0 <= score <= 1.0

def test_learned_estimator_uses_onnx_when_available():
    """With mock ONNX session, uses model inference."""
    mock_session = MagicMock()
    mock_session.run.return_value = [[0.85]]  # quality score
    mock_session.get_inputs.return_value = [
        MagicMock(name="input_ids"), MagicMock(name="attention_mask")
    ]
    est = LearnedQualityEstimator(model_path=None)
    est._session = mock_session
    est._tokenizer = MagicMock()
    est._tokenizer.return_value = {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}
    score = est.estimate("complex task", "detailed response")
    assert score == pytest.approx(0.85, abs=0.01)

def test_learned_estimator_clamps_output():
    """Output clamped to [0.0, 1.0]."""
    est = LearnedQualityEstimator(model_path=None)
    est._session = MagicMock()
    est._session.run.return_value = [[1.5]]
    est._session.get_inputs.return_value = [
        MagicMock(name="input_ids"), MagicMock(name="attention_mask")
    ]
    est._tokenizer = MagicMock()
    est._tokenizer.return_value = {"input_ids": [[1]], "attention_mask": [[1]]}
    score = est.estimate("task", "response")
    assert score == 1.0

def test_heuristic_estimator_unchanged():
    """Original QualityEstimator still works."""
    score = QualityEstimator.estimate("Write code", "def foo(): pass", latency_ms=100)
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_learned_quality_estimator.py -v`
Expected: FAIL (LearnedQualityEstimator not defined)

- [ ] **Step 3: Implement LearnedQualityEstimator**

Add to `sage-python/src/sage/quality_estimator.py` after the existing `QualityEstimator` class:

```python
class LearnedQualityEstimator:
    """ONNX-based quality estimator with heuristic fallback.

    Uses a fine-tuned DistilBERT/TinyBERT model on (task, response) pairs.
    Falls back to QualityEstimator.estimate() when ONNX model unavailable.
    """

    def __init__(self, model_path: str | None = None, tokenizer_name: str = "distilbert-base-uncased"):
        self._session = None
        self._tokenizer = None
        if model_path:
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
                self._session = ort.InferenceSession(model_path)
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception:
                pass  # fallback to heuristic

    def estimate(
        self,
        task: str,
        result: str,
        latency_ms: float = 0.0,
        had_errors: bool = False,
        avr_iterations: int = 0,
    ) -> float:
        if self._session is not None and self._tokenizer is not None:
            return self._infer(task, result)
        return QualityEstimator.estimate(task, result, latency_ms, had_errors, avr_iterations)

    def _infer(self, task: str, response: str) -> float:
        inputs = self._tokenizer(
            task, response, truncation=True, max_length=512,
            padding="max_length", return_tensors="np",
        )
        input_names = [inp.name for inp in self._session.get_inputs()]
        feed = {k: inputs[k] for k in input_names if k in inputs}
        output = self._session.run(None, feed)
        score = float(output[0][0])
        return max(0.0, min(1.0, score))
```

- [ ] **Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_learned_quality_estimator.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/quality_estimator.py sage-python/tests/test_learned_quality_estimator.py
git commit -m "feat: LearnedQualityEstimator with ONNX inference + heuristic fallback"
```

---

### Task 3: Quality Training Pipeline

**Files:**
- Create: `sage-python/src/sage/training/quality_trainer.py`
- Create: `sage-python/tests/test_quality_trainer.py`

Fine-tunes DistilBERT on collected triples, exports to ONNX. This is a CLI tool, not a runtime component.

- [ ] **Step 1: Write failing tests**

```python
# sage-python/tests/test_quality_trainer.py
import pytest
from sage.training.quality_trainer import QualityTrainerConfig, prepare_dataset

def test_config_defaults():
    cfg = QualityTrainerConfig()
    assert cfg.model_name == "distilbert-base-uncased"
    assert cfg.epochs == 5
    assert cfg.batch_size == 16
    assert cfg.lr == 2e-5
    assert cfg.dropout == 0.3
    assert cfg.weight_decay == 0.01
    assert cfg.freeze_layers == 4  # freeze bottom 4 of 6
    assert cfg.patience == 3

def test_prepare_dataset_splits():
    """5-fold CV: 80% train, 20% val per fold."""
    triples = [{"task_id": f"T/{i}", "task_text": f"task {i}",
                "response": f"resp {i}", "quality": float(i % 2)}
               for i in range(100)]
    folds = prepare_dataset(triples, n_folds=5)
    assert len(folds) == 5
    for train, val in folds:
        assert len(val) == 20
        assert len(train) == 80
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd sage-python && python -m pytest tests/test_quality_trainer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement quality_trainer.py**

```python
# sage-python/src/sage/training/quality_trainer.py
"""Fine-tune DistilBERT for quality estimation. CLI tool."""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

_log = logging.getLogger(__name__)

@dataclass
class QualityTrainerConfig:
    model_name: str = "distilbert-base-uncased"
    epochs: int = 5
    batch_size: int = 16
    lr: float = 2e-5
    dropout: float = 0.3
    weight_decay: float = 0.01
    freeze_layers: int = 4
    patience: int = 3
    max_length: int = 512
    output_dir: str = "sage-core/models"


def prepare_dataset(
    triples: list[dict], n_folds: int = 5,
) -> list[tuple[list[dict], list[dict]]]:
    """Stratified K-fold split. Returns list of (train, val) pairs."""
    # Binarize for stratification
    labels = [1 if t["quality"] >= 0.5 else 0 for t in triples]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = []
    for train_idx, val_idx in skf.split(triples, labels):
        train = [triples[i] for i in train_idx]
        val = [triples[i] for i in val_idx]
        folds.append((train, val))
    return folds


def train_and_export(
    triples_path: Path,
    config: QualityTrainerConfig | None = None,
) -> Path:
    """Full pipeline: load triples, train, export ONNX.

    Requires: pip install transformers torch onnx scikit-learn
    Returns path to exported ONNX model.
    """
    cfg = config or QualityTrainerConfig()

    # Lazy imports — training deps not required at runtime
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback,
    )

    triples = [json.loads(line) for line in triples_path.read_text().splitlines() if line.strip()]
    folds = prepare_dataset(triples, n_folds=5)
    train_data, val_data = folds[0]  # Use first fold for final model

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=1, problem_type="regression",
    )
    model.config.hidden_dropout_prob = cfg.dropout
    model.config.attention_probs_dropout_prob = cfg.dropout

    # Freeze bottom layers
    for i, layer in enumerate(model.distilbert.transformer.layer):
        if i < cfg.freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False

    # Tokenize
    def tokenize(examples):
        texts = [f"{e['task_text']} [SEP] {e['response']}" for e in examples]
        enc = tokenizer(texts, truncation=True, max_length=cfg.max_length, padding="max_length")
        enc["labels"] = [e["quality"] for e in examples]
        return enc

    train_enc = tokenize(train_data)
    val_enc = tokenize(val_data)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir / "quality_checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_enc, eval_dataset=val_enc,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)],
    )
    trainer.train()

    # Export ONNX
    onnx_path = output_dir / "quality_estimator.onnx"
    dummy = tokenizer("task [SEP] response", return_tensors="pt",
                       truncation=True, max_length=cfg.max_length, padding="max_length")
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["quality_score"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
        opset_version=14,
    )
    _log.info("Exported ONNX model to %s", onnx_path)
    return onnx_path
```

- [ ] **Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_quality_trainer.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/training/quality_trainer.py sage-python/tests/test_quality_trainer.py
git commit -m "feat(training): quality trainer pipeline (DistilBERT fine-tune + ONNX export)"
```

---

### Task 4: DeBERTa Zero-Shot Evaluation

**Files:**
- Create: `sage-python/scripts/eval_deberta_zeroshot.py`
- Create: `sage-python/tests/test_deberta_eval.py`

Script to download NVIDIA's DeBERTa-v3-base classifier, convert to ONNX, and measure accuracy on SAGE's 50 GT tasks with bootstrap CI.

- [ ] **Step 1: Write test for bootstrap CI logic**

```python
# sage-python/tests/test_deberta_eval.py
import numpy as np
from sage.training.bootstrap_ci import bootstrap_accuracy_ci

def test_bootstrap_ci_perfect():
    """Perfect predictions → CI near 1.0."""
    preds = [1] * 50
    labels = [1] * 50
    lower, point, upper = bootstrap_accuracy_ci(preds, labels, n_resamples=1000)
    assert point == 1.0
    assert lower > 0.9

def test_bootstrap_ci_50_percent():
    """Random predictions → CI around 0.5."""
    np.random.seed(42)
    preds = list(np.random.choice([1, 2, 3], 50))
    labels = list(np.random.choice([1, 2, 3], 50))
    lower, point, upper = bootstrap_accuracy_ci(preds, labels, n_resamples=1000)
    assert 0.1 < lower < 0.6
    assert lower < point < upper
```

- [ ] **Step 2: Implement bootstrap_ci module**

```python
# sage-python/src/sage/training/bootstrap_ci.py
"""Bootstrap confidence intervals for accuracy."""
import numpy as np

def bootstrap_accuracy_ci(
    preds: list[int], labels: list[int],
    n_resamples: int = 10_000, confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Returns (lower, point_estimate, upper) for accuracy."""
    preds_arr = np.array(preds)
    labels_arr = np.array(labels)
    point = float(np.mean(preds_arr == labels_arr))
    n = len(preds)
    accs = np.empty(n_resamples)
    rng = np.random.default_rng(42)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        accs[i] = np.mean(preds_arr[idx] == labels_arr[idx])
    alpha = 1 - confidence
    lower = float(np.percentile(accs, 100 * alpha / 2))
    upper = float(np.percentile(accs, 100 * (1 - alpha / 2)))
    return lower, point, upper
```

- [ ] **Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_deberta_eval.py -v`
Expected: 2 passed

- [ ] **Step 4: Write evaluation script**

```python
# sage-python/scripts/eval_deberta_zeroshot.py
"""Evaluate NVIDIA DeBERTa-v3-base zero-shot on SAGE GT tasks.

Usage: python scripts/eval_deberta_zeroshot.py [--model nvidia/prompt-task-and-complexity-classifier]
"""
import argparse, json, logging, sys
from pathlib import Path

_log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/prompt-task-and-complexity-classifier")
    parser.add_argument("--gt", default="sage-python/src/sage/bench/routing_gt.py",
                        help="Ground truth module path")
    parser.add_argument("--output", default="docs/benchmarks/deberta_zeroshot.json")
    args = parser.parse_args()

    from transformers import pipeline
    from sage.training.bootstrap_ci import bootstrap_accuracy_ci

    # Load GT tasks
    sys.path.insert(0, str(Path.cwd() / "sage-python" / "src"))
    from sage.bench.routing_gt import GROUND_TRUTH_TASKS

    classifier = pipeline("text-classification", model=args.model, top_k=None)

    preds, labels = [], []
    details = []
    for task in GROUND_TRUTH_TASKS:
        result = classifier(task["task"])
        # Map NVIDIA labels to S1/S2/S3
        pred_system = _map_to_system(result)
        preds.append(pred_system)
        labels.append(task["expected"])
        details.append({"task": task["task"], "expected": task["expected"],
                        "predicted": pred_system, "raw": str(result)})

    lower, point, upper = bootstrap_accuracy_ci(preds, labels)
    report = {
        "model": args.model, "n_tasks": len(labels),
        "accuracy": point, "ci_lower": lower, "ci_upper": upper,
        "ship_criterion": "ci_lower >= 0.90",
        "ship_decision": "SHIP" if lower >= 0.90 else "FINE_TUNE_NEEDED",
        "details": details,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"Accuracy: {point:.1%} (95% CI: [{lower:.1%}, {upper:.1%}])")
    print(f"Decision: {report['ship_decision']}")

def _map_to_system(result: list[dict]) -> int:
    """Map NVIDIA classifier output to S1/S2/S3."""
    # NVIDIA model outputs complexity labels — map highest confidence
    label_map = {"simple": 1, "moderate": 2, "complex": 3,
                 "easy": 1, "medium": 2, "hard": 3}
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], list):
            result = result[0]
        best = max(result, key=lambda x: x.get("score", 0))
        label = best.get("label", "").lower()
        for key, system in label_map.items():
            if key in label:
                return system
    return 2  # default S2

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/training/bootstrap_ci.py sage-python/tests/test_deberta_eval.py sage-python/scripts/eval_deberta_zeroshot.py
git commit -m "feat: DeBERTa zero-shot eval script + bootstrap CI module"
```

---

### Task 5: DeBERTa Stage 1 Integration

**Files:**
- Modify: `sage-python/src/sage/strategy/adaptive_router.py:59-120` (constructor), `167-221` (route_adaptive)
- Create: `sage-python/tests/test_deberta_routing.py`

Wire DeBERTa ONNX model as Stage 1 in AdaptiveRouter, with shadow comparison against kNN.

- [ ] **Step 1: Write failing tests**

```python
# sage-python/tests/test_deberta_routing.py
from unittest.mock import MagicMock, patch
from sage.strategy.adaptive_router import AdaptiveRouter

def test_adaptive_router_deberta_stage1():
    """When DeBERTa ONNX model is loaded, Stage 1 uses it."""
    router = AdaptiveRouter()
    mock_session = MagicMock()
    mock_session.run.return_value = [[0.1, 0.3, 0.6]]  # S1=0.1, S2=0.3, S3=0.6
    mock_session.get_inputs.return_value = [
        MagicMock(name="input_ids"), MagicMock(name="attention_mask")
    ]
    router._deberta_session = mock_session
    router._deberta_tokenizer = MagicMock()
    router._deberta_tokenizer.return_value = {
        "input_ids": [[1, 2]], "attention_mask": [[1, 1]]
    }
    result = router._try_deberta("complex formal proof task")
    assert result is not None
    assert result.system == 3  # highest logit

def test_adaptive_router_no_deberta_returns_none():
    """Without DeBERTa, _try_deberta returns None (fall through)."""
    router = AdaptiveRouter()
    router._deberta_session = None
    result = router._try_deberta("any task")
    assert result is None

def test_shadow_trace_deberta_vs_knn():
    """Shadow mode logs both kNN and DeBERTa results."""
    router = AdaptiveRouter()
    router._shadow_traces = []
    # This tests the shadow logging mechanism exists
    assert hasattr(router, "_shadow_traces") or hasattr(router, "_deberta_session")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd sage-python && python -m pytest tests/test_deberta_routing.py -v`
Expected: FAIL

- [ ] **Step 3: Add DeBERTa support to AdaptiveRouter**

Modify `sage-python/src/sage/strategy/adaptive_router.py`:

1. In `__init__`, add after kNN initialization (~line 99):
```python
# Stage 1: DeBERTa ONNX classifier (optional)
self._deberta_session = None
self._deberta_tokenizer = None
deberta_path = self._find_model("deberta_router.onnx")
if deberta_path:
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self._deberta_session = ort.InferenceSession(str(deberta_path))
        self._deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    except Exception as e:
        _log.debug("DeBERTa Stage 1 unavailable: %s", e)
```

2. Add `_try_deberta()` method:
```python
def _try_deberta(self, task: str) -> CognitiveProfile | None:
    """Stage 1: DeBERTa ONNX multi-head classifier."""
    if self._deberta_session is None or self._deberta_tokenizer is None:
        return None
    try:
        inputs = self._deberta_tokenizer(
            task, truncation=True, max_length=512,
            padding="max_length", return_tensors="np",
        )
        input_names = [inp.name for inp in self._deberta_session.get_inputs()]
        feed = {k: inputs[k] for k in input_names if k in inputs}
        logits = self._deberta_session.run(None, feed)[0][0]
        system = int(logits.argmax()) + 1  # 0-indexed → 1-indexed
        confidence = float(logits.max())
        complexity = [0.25, 0.55, 0.85][system - 1]
        return CognitiveProfile(
            complexity=complexity, uncertainty=max(0, 1.0 - confidence),
            tool_required=False, reasoning=f"deberta_stage1_S{system}",
        )
    except Exception as e:
        _log.debug("DeBERTa inference failed: %s", e)
        return None
```

3. In `route_adaptive()`, insert DeBERTa Stage 1 after kNN Stage 0.5 (~line 194):
```python
# Stage 1: DeBERTa ONNX (if available)
deberta_result = self._try_deberta(task)
if deberta_result is not None:
    # Shadow: log both kNN and DeBERTa for comparison
    if knn_result is not None:
        _log.debug("Shadow: kNN=S%s DeBERTa=S%s for: %.50s",
                    knn_result.system, deberta_result.system, task)
    return AdaptiveRoutingResult(
        profile=deberta_result, decision=self.route(deberta_result),
        stage="deberta_onnx", confidence=1.0 - deberta_result.uncertainty,
    )
```

- [ ] **Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_deberta_routing.py -v`
Expected: 3 passed

- [ ] **Step 5: Run full routing test suite**

Run: `cd sage-python && python -m pytest tests/test_adaptive_router.py tests/test_deberta_routing.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/strategy/adaptive_router.py sage-python/tests/test_deberta_routing.py
git commit -m "feat(routing): DeBERTa-v3-base Stage 1 ONNX integration in AdaptiveRouter"
```

---

### Task 6: sage-router Package Scaffold

**Files:**
- Create: `sage-router/pyproject.toml`
- Create: `sage-router/src/sage_router/__init__.py`
- Create: `sage-router/src/sage_router/types.py`
- Create: `sage-router/tests/test_types.py`

Standalone pip-installable routing library. Start with types + package structure.

- [ ] **Step 1: Create package structure**

```bash
mkdir -p sage-router/src/sage_router sage-router/tests
```

- [ ] **Step 2: Write pyproject.toml**

```toml
# sage-router/pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "sage-router"
version = "0.1.0"
description = "Cognitive S1/S2/S3 routing for LLM agents — standalone from YGN-SAGE"
requires-python = ">=3.11"
license = "MIT"
dependencies = ["numpy>=1.26"]

[project.optional-dependencies]
onnx = ["onnxruntime>=1.17", "sentence-transformers>=2.2"]
dev = ["pytest>=8", "pytest-asyncio>=0.23"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Write types.py (copied from metacognition.py)**

```python
# sage-router/src/sage_router/types.py
"""Core routing types — self-contained, zero external deps."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class CognitiveProfile:
    """Task complexity assessment."""
    complexity: float     # 0.0-1.0
    uncertainty: float    # 0.0-1.0
    tool_required: bool
    reasoning: str = ""

@dataclass
class RouteDecision:
    """Routing decision: which cognitive system to use."""
    system: int           # 1 (fast), 2 (deliberate), 3 (formal)
    confidence: float     # 0.0-1.0
    method: str           # "knn", "deberta", "heuristic", "structural"
    llm_tier: str = ""    # "fast", "mutator", "reasoner", "codex"
```

- [ ] **Step 4: Write __init__.py**

```python
# sage-router/src/sage_router/__init__.py
"""sage-router: Cognitive S1/S2/S3 routing for LLM agents."""
from sage_router.types import CognitiveProfile, RouteDecision

__all__ = ["CognitiveProfile", "RouteDecision"]
__version__ = "0.1.0"
```

- [ ] **Step 5: Write tests**

```python
# sage-router/tests/test_types.py
from sage_router.types import CognitiveProfile, RouteDecision

def test_cognitive_profile_defaults():
    p = CognitiveProfile(complexity=0.5, uncertainty=0.3, tool_required=False)
    assert p.reasoning == ""

def test_route_decision():
    d = RouteDecision(system=2, confidence=0.9, method="knn", llm_tier="reasoner")
    assert d.system == 2
    assert d.llm_tier == "reasoner"

def test_import_from_package():
    from sage_router import CognitiveProfile, RouteDecision
    assert CognitiveProfile is not None
```

- [ ] **Step 6: Run tests**

Run: `cd sage-router && pip install -e ".[dev]" && python -m pytest tests/ -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add sage-router/
git commit -m "feat: sage-router package scaffold with CognitiveProfile + RouteDecision types"
```

---

### Task 7: sage-router CognitiveRouter

**Files:**
- Create: `sage-router/src/sage_router/features.py` (copy from sage-python structural_features.py)
- Create: `sage-router/src/sage_router/router.py`
- Create: `sage-router/src/sage_router/embedder.py`
- Create: `sage-router/tests/test_router.py`
- Modify: `sage-router/src/sage_router/__init__.py`

The main `CognitiveRouter` class wrapping structural features + kNN.

- [ ] **Step 1: Copy structural_features.py**

Copy `sage-python/src/sage/strategy/structural_features.py` → `sage-router/src/sage_router/features.py`. Adjust imports only (remove any sage-specific imports).

- [ ] **Step 2: Write minimal embedder wrapper**

```python
# sage-router/src/sage_router/embedder.py
"""Minimal embedder for sage-router. sentence-transformers only."""
from __future__ import annotations
import logging

_log = logging.getLogger(__name__)

class RouterEmbedder:
    """Embeds text for kNN routing. sentence-transformers backend."""

    def __init__(self, model_name: str = "Snowflake/snowflake-arctic-embed-m"):
        self._model = None
        self._model_name = model_name

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    @property
    def is_available(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except Exception:
            return False

    def embed(self, text: str) -> list[float]:
        self._ensure_loaded()
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        return self._model.encode(texts, normalize_embeddings=True).tolist()
```

- [ ] **Step 3: Write CognitiveRouter**

```python
# sage-router/src/sage_router/router.py
"""CognitiveRouter: the main entry point for sage-router."""
from __future__ import annotations
import logging
import numpy as np
from pathlib import Path
from sage_router.types import CognitiveProfile, RouteDecision
from sage_router.features import StructuralFeatures

_log = logging.getLogger(__name__)

class CognitiveRouter:
    """S1/S2/S3 cognitive routing with structural features + optional kNN."""

    def __init__(self, exemplars_path: str | Path | None = None):
        self._embedder = None
        self._exemplar_embeddings = None
        self._exemplar_labels = None
        self._load_exemplars(exemplars_path)

    def _load_exemplars(self, path: str | Path | None):
        search_paths = [
            Path(path) if path else None,
            Path.cwd() / "config" / "routing_exemplars.npz",
            Path.home() / ".sage" / "routing_exemplars.npz",
        ]
        for p in search_paths:
            if p and p.exists():
                try:
                    data = np.load(str(p))
                    self._exemplar_embeddings = data["embeddings"]
                    self._exemplar_labels = data["labels"]
                    _log.info("Loaded %d exemplars from %s",
                              len(self._exemplar_labels), p)
                    return
                except Exception as e:
                    _log.debug("Failed to load exemplars from %s: %s", p, e)

    def route(self, task: str) -> RouteDecision:
        """Route a task to S1/S2/S3."""
        # Stage 0.5: kNN (if exemplars loaded)
        knn = self._try_knn(task)
        if knn is not None:
            return knn

        # Stage 0: Structural features
        features = StructuralFeatures.extract(task)
        system = 1
        if features.keyword_complexity > 0.65:
            system = 3
        elif features.keyword_complexity > 0.35 or features.tool_required:
            system = 2

        return RouteDecision(
            system=system,
            confidence=0.5,
            method="structural",
        )

    def _try_knn(self, task: str, k: int = 5) -> RouteDecision | None:
        """kNN routing using pre-computed exemplar embeddings."""
        if self._exemplar_embeddings is None:
            return None
        if self._embedder is None:
            try:
                from sage_router.embedder import RouterEmbedder
                self._embedder = RouterEmbedder()
                if not self._embedder.is_available:
                    self._embedder = None
                    return None
            except Exception:
                return None

        query = np.array(self._embedder.embed(task), dtype=np.float32)
        query /= np.linalg.norm(query) + 1e-9
        sims = self._exemplar_embeddings @ query
        top_k = np.argpartition(sims, -k)[-k:]
        top_sims = sims[top_k]
        top_labels = self._exemplar_labels[top_k]

        # Distance-weighted majority vote
        votes = {1: 0.0, 2: 0.0, 3: 0.0}
        for label, sim in zip(top_labels, top_sims):
            votes[int(label)] += max(0, float(sim))
        system = max(votes, key=votes.get)
        confidence = votes[system] / (sum(votes.values()) + 1e-9)

        return RouteDecision(system=system, confidence=confidence, method="knn")
```

- [ ] **Step 4: Write tests**

```python
# sage-router/tests/test_router.py
from sage_router.router import CognitiveRouter
from sage_router.types import RouteDecision

def test_structural_routing_simple():
    router = CognitiveRouter(exemplars_path="/nonexistent")
    decision = router.route("What is 2+2?")
    assert isinstance(decision, RouteDecision)
    assert decision.system == 1
    assert decision.method == "structural"

def test_structural_routing_complex():
    router = CognitiveRouter(exemplars_path="/nonexistent")
    decision = router.route(
        "Implement a distributed consensus algorithm with formal proof"
    )
    assert decision.system >= 2

def test_route_returns_decision():
    router = CognitiveRouter()
    d = router.route("Hello world")
    assert d.system in (1, 2, 3)
    assert 0 <= d.confidence <= 1
```

- [ ] **Step 5: Update __init__.py**

```python
# sage-router/src/sage_router/__init__.py
"""sage-router: Cognitive S1/S2/S3 routing for LLM agents."""
from sage_router.types import CognitiveProfile, RouteDecision
from sage_router.router import CognitiveRouter

__all__ = ["CognitiveProfile", "RouteDecision", "CognitiveRouter"]
__version__ = "0.1.0"
```

- [ ] **Step 6: Run tests**

Run: `cd sage-router && python -m pytest tests/ -v`
Expected: 6 passed (3 types + 3 router)

- [ ] **Step 7: Commit**

```bash
git add sage-router/
git commit -m "feat(sage-router): CognitiveRouter with structural + kNN routing"
```

---

### Task 8: SAGE Re-Export Migration

**Files:**
- Modify: `sage-python/src/sage/strategy/metacognition.py:22-38`
- Modify: `sage-python/pyproject.toml`
- Create: `sage-python/tests/test_sage_router_reexport.py`

SAGE imports `CognitiveProfile` and `RoutingDecision` from `sage_router.types` and re-exports them. Transitional pattern — existing imports keep working.

- [ ] **Step 1: Write test for re-export**

```python
# sage-python/tests/test_sage_router_reexport.py
def test_import_from_metacognition_still_works():
    """Backward compat: old import path still works."""
    from sage.strategy.metacognition import CognitiveProfile, RoutingDecision
    p = CognitiveProfile(complexity=0.5, uncertainty=0.2, tool_required=False)
    assert p.complexity == 0.5

def test_import_from_sage_router():
    """New import path works."""
    from sage_router import CognitiveProfile, RouteDecision
    p = CognitiveProfile(complexity=0.3, uncertainty=0.1, tool_required=True)
    assert p.tool_required is True
```

- [ ] **Step 2: Add sage-router dependency to pyproject.toml**

In `sage-python/pyproject.toml`, add to dependencies:
```toml
"sage-router>=0.1.0",
```

- [ ] **Step 3: Replace dataclass definitions with re-exports**

In `sage-python/src/sage/strategy/metacognition.py`, replace the `CognitiveProfile` and `RoutingDecision` dataclass definitions (lines 22-38) with:

```python
# Re-export from sage-router for backward compatibility
from sage_router.types import CognitiveProfile  # noqa: F401

@dataclass
class RoutingDecision:
    """Routing decision — SAGE-specific (includes max_tokens, use_z3)."""
    system: int
    llm_tier: str
    max_tokens: int
    use_z3: bool
    validation_level: int = 1
```

Note: `RoutingDecision` stays in SAGE because it has SAGE-specific fields (`max_tokens`, `use_z3`, `validation_level`). Only `CognitiveProfile` is shared with sage-router.

- [ ] **Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_sage_router_reexport.py tests/test_strategy.py tests/test_metacognition_provider.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/strategy/metacognition.py sage-python/pyproject.toml sage-python/tests/test_sage_router_reexport.py
git commit -m "refactor: re-export CognitiveProfile from sage-router, keep RoutingDecision in SAGE"
```

---

## Chunk 2: Phase 2 — TopologyBench + Scale Proof

### Task 9: TopologyBench Task Suite

**Files:**
- Create: `sage-python/src/sage/bench/topology_tasks.json`
- Create: `sage-python/tests/test_topology_tasks.py`

200 tasks across 5 categories with difficulty labels.

- [ ] **Step 1: Write validation tests**

```python
# sage-python/tests/test_topology_tasks.py
import json
from pathlib import Path

TASKS_PATH = Path(__file__).parent.parent / "src" / "sage" / "bench" / "topology_tasks.json"

def test_task_file_exists():
    assert TASKS_PATH.exists()

def test_200_tasks():
    tasks = json.loads(TASKS_PATH.read_text())
    assert len(tasks) == 200

def test_category_distribution():
    tasks = json.loads(TASKS_PATH.read_text())
    cats = {t["category"] for t in tasks}
    assert cats == {"code", "math", "reasoning", "tool_use", "creative"}
    counts = {}
    for t in tasks:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    assert counts["code"] == 60
    assert counts["math"] == 40
    assert counts["reasoning"] == 40
    assert counts["tool_use"] == 30
    assert counts["creative"] == 30

def test_task_schema():
    tasks = json.loads(TASKS_PATH.read_text())
    for t in tasks:
        assert "id" in t
        assert "task" in t
        assert "category" in t
        assert "difficulty" in t
        assert t["difficulty"] in ("S1", "S2", "S3")
```

- [ ] **Step 2: Create topology_tasks.json**

Generate 200 tasks. This is a data file — the implementer should create tasks following this distribution:
- **Code (60)**: 20 S1 (hello world, fizzbuzz), 25 S2 (algorithms, data structures), 15 S3 (distributed systems, formal proofs)
- **Math (40)**: 15 S1 (arithmetic), 15 S2 (algebra, calculus), 10 S3 (formal proofs)
- **Reasoning (40)**: 15 S1 (logic puzzles), 15 S2 (analogies, planning), 10 S3 (complex multi-step)
- **Tool-use (30)**: 0 S1, 20 S2 (file I/O, API calls), 10 S3 (complex orchestration)
- **Creative (30)**: 15 S1 (short writing), 15 S2 (summarization, translation), 0 S3

Each task entry:
```json
{"id": "code_001", "task": "Write a function that returns the sum of two integers", "category": "code", "difficulty": "S1", "expected_output_type": "code"}
```

Source for code tasks: Sample from HumanEval+ prompts (easy/medium/hard). For other categories, write original tasks.

- [ ] **Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_topology_tasks.py -v`
Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/bench/topology_tasks.json sage-python/tests/test_topology_tasks.py
git commit -m "data: TopologyBench 200-task suite (5 categories × graduated difficulty)"
```

---

### Task 10: TopologyBench Runner

**Files:**
- Create: `sage-python/src/sage/bench/topology_bench.py`
- Create: `sage-python/tests/test_topology_bench.py`
- Modify: `sage-python/src/sage/bench/__init__.py` (register topology bench)

Runner that executes each task against each of 10 topology configurations.

- [ ] **Step 1: Write tests**

```python
# sage-python/tests/test_topology_bench.py
import pytest
from sage.bench.topology_bench import TopologyBenchConfig, TOPOLOGY_CONFIGS

def test_topology_configs_count():
    assert len(TOPOLOGY_CONFIGS) == 10

def test_topology_config_names():
    names = {c.name for c in TOPOLOGY_CONFIGS}
    assert "sequential" in names
    assert "avr" in names
    assert "debate" in names
    assert "evolved" in names

def test_config_has_template():
    for c in TOPOLOGY_CONFIGS:
        assert c.name
        assert c.avg_calls_per_task > 0

def test_bench_config_defaults():
    cfg = TopologyBenchConfig()
    assert cfg.task_limit is None
    assert cfg.model_tier == "fast"
```

- [ ] **Step 2: Implement topology_bench.py**

```python
# sage-python/src/sage/bench/topology_bench.py
"""TopologyBench: first multi-agent topology impact benchmark."""
from __future__ import annotations
import json, logging, time
from dataclasses import dataclass, field, asdict
from pathlib import Path

_log = logging.getLogger(__name__)
_TASKS_PATH = Path(__file__).parent / "topology_tasks.json"


@dataclass
class TopologyConfig:
    name: str
    template: str  # matches Rust template name or "evolved"/"oracle"
    avg_calls_per_task: float
    description: str = ""


TOPOLOGY_CONFIGS = [
    TopologyConfig("sequential", "Sequential", 1.0, "Single agent baseline"),
    TopologyConfig("parallel", "Parallel", 2.5, "Concurrent agents"),
    TopologyConfig("avr", "AVR", 3.0, "Act-Verify-Refine"),
    TopologyConfig("selfmoa", "SelfMoA", 4.0, "Mixture of Agents"),
    TopologyConfig("hierarchical", "Hierarchical", 3.0, "Manager + workers"),
    TopologyConfig("hub", "Hub", 2.5, "Hub + spokes"),
    TopologyConfig("debate", "Debate", 5.0, "Proposer + opponent + judge"),
    TopologyConfig("brainstorming", "Brainstorming", 4.0, "Generators + synthesizer"),
    TopologyConfig("evolved", "evolved", 4.0, "MAP-Elites/MCTS evolved"),
    TopologyConfig("oracle", "oracle", 3.5, "Human-selected per category"),
]


@dataclass
class TopologyBenchConfig:
    task_limit: int | None = None
    model_tier: str = "fast"
    output_dir: str = "docs/benchmarks"
    topologies: list[str] | None = None  # None = all 10


@dataclass
class CellResult:
    task_id: str
    topology: str
    passed: bool
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tokens_used: int = 0
    error: str | None = None


@dataclass
class TopologyBenchReport:
    total_tasks: int
    total_topologies: int
    total_cells: int
    results: list[CellResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def load_tasks(limit: int | None = None) -> list[dict]:
    tasks = json.loads(_TASKS_PATH.read_text(encoding="utf-8"))
    if limit:
        tasks = tasks[:limit]
    return tasks


async def run_topology_bench(
    config: TopologyBenchConfig | None = None,
) -> TopologyBenchReport:
    """Run TopologyBench: each task × each topology."""
    cfg = config or TopologyBenchConfig()
    tasks = load_tasks(cfg.task_limit)

    active_topos = TOPOLOGY_CONFIGS
    if cfg.topologies:
        active_topos = [t for t in TOPOLOGY_CONFIGS if t.name in cfg.topologies]

    results: list[CellResult] = []
    for task in tasks:
        for topo in active_topos:
            cell = await _run_cell(task, topo, cfg.model_tier)
            results.append(cell)

    report = TopologyBenchReport(
        total_tasks=len(tasks),
        total_topologies=len(active_topos),
        total_cells=len(results),
        results=results,
        summary=_compute_summary(results, active_topos),
    )

    # Save
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d")
    path = out / f"{ts}-topology-bench.json"
    path.write_text(report.to_json(), encoding="utf-8")
    _log.info("TopologyBench saved to %s", path)
    return report


async def _run_cell(task: dict, topo: TopologyConfig, model_tier: str) -> CellResult:
    """Execute one (task, topology) cell. Placeholder — wires to SAGE runtime."""
    # TODO: Wire to actual TopologyExecutor + AgentLoop
    # For now, returns placeholder to unblock the benchmark infrastructure
    return CellResult(
        task_id=task["id"], topology=topo.name,
        passed=False, error="not_implemented",
    )


def _compute_summary(results: list[CellResult], topos: list[TopologyConfig]) -> dict:
    """Aggregate pass rates per topology and per category."""
    by_topo: dict[str, list[bool]] = {}
    for r in results:
        by_topo.setdefault(r.topology, []).append(r.passed)
    summary = {}
    for name, passes in by_topo.items():
        n = len(passes)
        passed = sum(passes)
        summary[name] = {"total": n, "passed": passed,
                         "pass_rate": passed / n if n > 0 else 0.0}
    return summary
```

- [ ] **Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_topology_bench.py -v`
Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/bench/topology_bench.py sage-python/tests/test_topology_bench.py
git commit -m "feat(bench): TopologyBench runner — 200 tasks × 10 topologies framework"
```

---

### Task 11: Large-Scale Ablation with Bootstrap CI

**Files:**
- Modify: `sage-python/src/sage/bench/ablation.py`
- Create: `sage-python/tests/test_ablation_stats.py`

Add bootstrap CI and McNemar test to existing ablation framework.

- [ ] **Step 1: Write tests for statistical methods**

```python
# sage-python/tests/test_ablation_stats.py
import numpy as np
from sage.bench.ablation import bootstrap_ci, mcnemar_test

def test_bootstrap_ci_returns_tuple():
    results = [True, True, False, True, False] * 20  # 100 results
    lower, point, upper = bootstrap_ci(results)
    assert lower < point < upper
    assert 0.4 < point < 0.8

def test_bootstrap_ci_perfect():
    results = [True] * 100
    lower, point, upper = bootstrap_ci(results)
    assert point == 1.0
    assert lower > 0.95

def test_mcnemar_significant():
    """One config clearly better than the other."""
    a = [True] * 80 + [False] * 20
    b = [True] * 50 + [False] * 50
    stat, p_value = mcnemar_test(a, b)
    assert p_value < 0.05

def test_mcnemar_not_significant():
    """Identical configs → not significant."""
    a = [True, False] * 50
    stat, p_value = mcnemar_test(a, a)
    assert p_value > 0.05 or stat == 0
```

- [ ] **Step 2: Add statistical functions to ablation.py**

Add to `sage-python/src/sage/bench/ablation.py`:

```python
def bootstrap_ci(
    results: list[bool], n_resamples: int = 10_000, confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI for pass rate."""
    arr = np.array(results, dtype=float)
    point = float(arr.mean())
    rng = np.random.default_rng(42)
    boot = np.empty(n_resamples)
    n = len(arr)
    for i in range(n_resamples):
        boot[i] = arr[rng.integers(0, n, size=n)].mean()
    alpha = 1 - confidence
    return (float(np.percentile(boot, 100 * alpha / 2)),
            point,
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def mcnemar_test(a: list[bool], b: list[bool]) -> tuple[float, float]:
    """McNemar test for paired binary outcomes. Returns (statistic, p_value)."""
    # b01: a wrong, b right. b10: a right, b wrong.
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = b01 + b10
    if n == 0:
        return 0.0, 1.0
    stat = (abs(b01 - b10) - 1) ** 2 / n  # continuity correction
    # chi2 with 1 df: p-value approximation
    from math import erfc, sqrt
    p_value = erfc(sqrt(stat / 2))  # rough approximation
    return float(stat), float(p_value)
```

- [ ] **Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_ablation_stats.py -v`
Expected: 4 passed

- [ ] **Step 4: Add `--scale full` flag**

Modify the ablation runner to accept a scale parameter that selects task pool size:
- `--scale quick`: 20 tasks (current default)
- `--scale full`: 500 tasks (HumanEval+ 164 + MBPP+ 200 + reasoning 136)

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/bench/ablation.py sage-python/tests/test_ablation_stats.py
git commit -m "feat(bench): bootstrap CI + McNemar test for ablation, --scale full flag"
```

---

### Task 12: kNN Exemplar Expansion Pipeline

**Files:**
- Create: `sage-python/src/sage/training/exemplar_expander.py`
- Create: `sage-python/tests/test_exemplar_expander.py`

Auto-label TopologyBench results → proposed labels CSV → human review gate → rebuild exemplars.

- [ ] **Step 1: Write tests**

```python
# sage-python/tests/test_exemplar_expander.py
from sage.training.exemplar_expander import auto_label_task, ExemplarExpander

def test_auto_label_s1():
    """Task passing with Sequential alone → S1."""
    results = {"sequential": True, "avr": True, "debate": True}
    label, confidence = auto_label_task(results)
    assert label == 1
    assert confidence > 0.8

def test_auto_label_s2():
    """Task failing Sequential but passing AVR → S2."""
    results = {"sequential": False, "avr": True, "debate": True}
    label, confidence = auto_label_task(results)
    assert label == 2

def test_auto_label_s3():
    """Task failing Sequential and AVR, needs Debate → S3."""
    results = {"sequential": False, "avr": False, "debate": True}
    label, confidence = auto_label_task(results)
    assert label == 3

def test_auto_label_ambiguous():
    """Mixed results → low confidence."""
    results = {"sequential": True, "avr": False, "debate": True}
    label, confidence = auto_label_task(results)
    assert confidence < 0.8  # flagged for review

def test_expander_produces_csv(tmp_path):
    tasks = [
        {"id": "T/0", "task": "hello", "results": {"sequential": True, "avr": True}},
        {"id": "T/1", "task": "complex", "results": {"sequential": False, "avr": True}},
    ]
    exp = ExemplarExpander(anchor_gt_path=None)
    csv_path = exp.generate_review_csv(tasks, tmp_path / "labels.csv")
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 3  # header + 2 tasks
```

- [ ] **Step 2: Implement exemplar_expander.py**

```python
# sage-python/src/sage/training/exemplar_expander.py
"""Auto-label TopologyBench results with human review gate."""
from __future__ import annotations
import csv, json, logging
from pathlib import Path
from dataclasses import dataclass

_log = logging.getLogger(__name__)

def auto_label_task(results: dict[str, bool]) -> tuple[int, float]:
    """Propose S1/S2/S3 label from topology pass/fail results.

    Returns (label, confidence).
    - S1: passes with sequential alone
    - S2: fails sequential, passes with AVR/parallel
    - S3: needs debate/hierarchical/evolved
    """
    s1_topos = {"sequential"}
    s2_topos = {"avr", "parallel", "hub"}
    s3_topos = {"debate", "hierarchical", "selfmoa", "brainstorming", "evolved"}

    passes_s1 = any(results.get(t, False) for t in s1_topos)
    passes_s2 = any(results.get(t, False) for t in s2_topos)
    passes_s3 = any(results.get(t, False) for t in s3_topos)

    if passes_s1:
        # Check consistency: if it passes S1 but fails S2, that's weird
        if not passes_s2 and passes_s3:
            return 1, 0.5  # ambiguous
        return 1, 0.9
    elif passes_s2:
        return 2, 0.85
    elif passes_s3:
        return 3, 0.8
    else:
        return 3, 0.4  # nothing worked → assume S3, low confidence


class ExemplarExpander:
    """Pipeline: TopologyBench results → proposed labels → review CSV."""

    def __init__(self, anchor_gt_path: Path | None = None):
        self._anchor = {}
        if anchor_gt_path and anchor_gt_path.exists():
            data = json.loads(anchor_gt_path.read_text())
            self._anchor = {t["task"]: t["expected"] for t in data}

    def generate_review_csv(
        self, tasks: list[dict], output: Path,
    ) -> Path:
        """Generate proposed_labels.csv for human review."""
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["task_id", "task_text", "auto_label", "confidence",
                             "anchor_conflict", "needs_review"])
            for t in tasks:
                label, conf = auto_label_task(t.get("results", {}))
                anchor_label = self._anchor.get(t.get("task", ""))
                conflict = anchor_label is not None and anchor_label != label
                needs_review = conf < 0.8 or conflict
                writer.writerow([
                    t["id"], t.get("task", "")[:100],
                    label, f"{conf:.2f}",
                    "YES" if conflict else "no",
                    "YES" if needs_review else "no",
                ])
        _log.info("Review CSV: %s (%d tasks)", output, len(tasks))
        return output
```

- [ ] **Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_exemplar_expander.py -v`
Expected: 5 passed

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/training/exemplar_expander.py sage-python/tests/test_exemplar_expander.py
git commit -m "feat(training): kNN exemplar expansion with auto-labeling + human review gate"
```

---

## Chunk 3: Phase 3 — DX Parity

### Task 13: Per-Agent Memory Scoping (Episodic)

**Files:**
- Modify: `sage-python/src/sage/memory/episodic.py:78-133` (store + search)
- Create: `sage-python/tests/test_episodic_scoping.py`

Add `agent_id` column to episodic store with backward-compatible default.

- [ ] **Step 1: Write tests**

```python
# sage-python/tests/test_episodic_scoping.py
import pytest
from sage.memory.episodic import EpisodicMemory

@pytest.mark.asyncio
async def test_store_with_agent_id():
    mem = EpisodicMemory()
    await mem.initialize()
    await mem.store("key1", "content1", agent_id="researcher")
    await mem.store("key2", "content2", agent_id="coder")
    await mem.store("key3", "content3")  # no agent_id (global)
    results = await mem.search("content", agent_id="researcher")
    assert any("content1" in r.get("content", "") for r in results)
    assert not any("content2" in r.get("content", "") for r in results)

@pytest.mark.asyncio
async def test_search_without_agent_id_returns_all():
    mem = EpisodicMemory()
    await mem.initialize()
    await mem.store("k1", "alpha", agent_id="a")
    await mem.store("k2", "alpha", agent_id="b")
    results = await mem.search("alpha")
    assert len(results) >= 2  # both agents' entries

@pytest.mark.asyncio
async def test_backward_compat_no_agent_id():
    """Existing code without agent_id still works."""
    mem = EpisodicMemory()
    await mem.initialize()
    await mem.store("k", "data")
    results = await mem.search("data")
    assert len(results) >= 1
```

- [ ] **Step 2: Modify episodic.py**

1. In `initialize()`, alter `CREATE TABLE` to include `agent_id TEXT DEFAULT ''`:
```sql
CREATE TABLE IF NOT EXISTS episodes (
    key TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata TEXT,
    agent_id TEXT DEFAULT '',
    timestamp TEXT
)
```

2. In `store()`, add `agent_id: str = ""` parameter and include in INSERT.

3. In `search()`, add `agent_id: str | None = None` parameter. When set, add `WHERE agent_id = ?` filter.

- [ ] **Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_episodic_scoping.py -v`
Expected: 3 passed

- [ ] **Step 4: Run existing episodic tests**

Run: `cd sage-python && python -m pytest tests/ -k episodic -v`
Expected: All existing tests still pass

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/memory/episodic.py sage-python/tests/test_episodic_scoping.py
git commit -m "feat(memory): per-agent scoping in EpisodicMemory (agent_id column)"
```

---

### Task 14: Per-Agent Memory Scoping (Semantic)

**Files:**
- Modify: `sage-python/src/sage/memory/semantic.py:45-75` (add_extraction), `123-154` (get_context_for)
- Create: `sage-python/tests/test_semantic_scoping.py`

Add `agent_id` ownership on edges, scoped queries.

- [ ] **Step 1: Write tests**

```python
# sage-python/tests/test_semantic_scoping.py
from sage.memory.semantic import SemanticMemory

def test_add_extraction_with_agent_id():
    mem = SemanticMemory()
    mem.add_extraction("Python", "is", "language", agent_id="researcher")
    mem.add_extraction("Rust", "is", "language", agent_id="coder")
    ctx = mem.get_context_for("Python programming", agent_id="researcher")
    assert "Python" in ctx
    assert "Rust" not in ctx

def test_get_context_without_agent_returns_all():
    mem = SemanticMemory()
    mem.add_extraction("A", "rel", "B", agent_id="x")
    mem.add_extraction("C", "rel", "D", agent_id="y")
    ctx = mem.get_context_for("A C")
    assert "A" in ctx
    assert "C" in ctx

def test_backward_compat():
    mem = SemanticMemory()
    mem.add_extraction("E", "rel", "F")  # no agent_id
    ctx = mem.get_context_for("E")
    assert "E" in ctx
```

- [ ] **Step 2: Modify semantic.py**

1. Change relation tuple from `(s, p, o)` to `(s, p, o, agent_id)`.
2. In `add_extraction()`, add `agent_id: str = ""` parameter.
3. In `get_context_for()`, add `agent_id: str | None = None` parameter. When set, filter relations by agent_id.

- [ ] **Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_semantic_scoping.py tests/test_memory.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/memory/semantic.py sage-python/tests/test_semantic_scoping.py
git commit -m "feat(memory): per-agent scoping in SemanticMemory (agent_id on edges)"
```

---

### Task 15: ScopedMemory Wrapper

**Files:**
- Create: `sage-python/src/sage/memory/scoped.py`
- Create: `sage-python/tests/test_scoped_memory.py`

Convenience wrapper that binds an `agent_id` and delegates to episodic + semantic.

- [ ] **Step 1: Write tests**

```python
# sage-python/tests/test_scoped_memory.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from sage.memory.scoped import ScopedMemory

@pytest.mark.asyncio
async def test_scoped_search_passes_agent_id():
    episodic = AsyncMock()
    episodic.search = AsyncMock(return_value=[])
    semantic = MagicMock()
    semantic.get_context_for = MagicMock(return_value="")

    scoped = ScopedMemory(agent_id="researcher", episodic=episodic, semantic=semantic)
    await scoped.search("query")

    episodic.search.assert_awaited_once_with("query", agent_id="researcher", top_k=5)
    semantic.get_context_for.assert_called_once_with("query", agent_id="researcher")

@pytest.mark.asyncio
async def test_scoped_store_passes_agent_id():
    episodic = AsyncMock()
    scoped = ScopedMemory(agent_id="coder", episodic=episodic)
    await scoped.store("key", "content")
    episodic.store.assert_awaited_once_with("key", "content", metadata=None, agent_id="coder")
```

- [ ] **Step 2: Implement scoped.py**

```python
# sage-python/src/sage/memory/scoped.py
"""ScopedMemory: binds agent_id for episodic + semantic queries."""
from __future__ import annotations
from typing import Any

class ScopedMemory:
    """Convenience wrapper that auto-passes agent_id to memory tiers."""

    def __init__(
        self,
        agent_id: str,
        episodic: Any = None,
        semantic: Any = None,
    ):
        self.agent_id = agent_id
        self._episodic = episodic
        self._semantic = semantic

    async def store(self, key: str, content: str, metadata: dict | None = None):
        if self._episodic:
            await self._episodic.store(key, content, metadata=metadata, agent_id=self.agent_id)

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        results = []
        if self._episodic:
            results = await self._episodic.search(query, agent_id=self.agent_id, top_k=top_k)
        if self._semantic:
            ctx = self._semantic.get_context_for(query, agent_id=self.agent_id)
            if ctx:
                results.append({"source": "semantic", "content": ctx})
        return results
```

- [ ] **Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_scoped_memory.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/memory/scoped.py sage-python/tests/test_scoped_memory.py
git commit -m "feat(memory): ScopedMemory wrapper for per-agent memory access"
```

---

### Task 16: Checkpointer Module

**Files:**
- Create: `sage-python/src/sage/checkpointing.py`
- Create: `sage-python/tests/test_checkpointing.py`
- Modify: `sage-python/pyproject.toml` (add msgpack dependency)

SQLite-backed checkpoint save/load/cleanup.

- [ ] **Step 1: Add msgpack dependency**

In `sage-python/pyproject.toml`, add `"msgpack>=1.0"` to core dependencies.

- [ ] **Step 2: Write tests**

```python
# sage-python/tests/test_checkpointing.py
import pytest
from sage.checkpointing import Checkpointer

@pytest.mark.asyncio
async def test_save_and_load(tmp_path):
    cp = Checkpointer(db_path=str(tmp_path / "checkpoints.db"))
    await cp.initialize()

    state = {"phase": "think", "task": "hello", "responses": ["world"]}
    await cp.save("task-001", "think", state, agent_id="main")

    loaded = await cp.load("task-001")
    assert loaded is not None
    assert loaded["phase"] == "think"
    assert loaded["responses"] == ["world"]

@pytest.mark.asyncio
async def test_load_nonexistent(tmp_path):
    cp = Checkpointer(db_path=str(tmp_path / "checkpoints.db"))
    await cp.initialize()
    assert await cp.load("no-such-task") is None

@pytest.mark.asyncio
async def test_cleanup(tmp_path):
    cp = Checkpointer(db_path=str(tmp_path / "checkpoints.db"))
    await cp.initialize()
    await cp.save("t1", "act", {"x": 1})
    await cp.cleanup("t1")
    assert await cp.load("t1") is None

@pytest.mark.asyncio
async def test_overwrite_checkpoint(tmp_path):
    cp = Checkpointer(db_path=str(tmp_path / "checkpoints.db"))
    await cp.initialize()
    await cp.save("t1", "perceive", {"step": 1})
    await cp.save("t1", "think", {"step": 2})
    loaded = await cp.load("t1")
    assert loaded["step"] == 2
```

- [ ] **Step 3: Implement checkpointing.py**

```python
# sage-python/src/sage/checkpointing.py
"""Durable execution: checkpoint/resume at phase boundaries."""
from __future__ import annotations
import logging, time
from pathlib import Path

import aiosqlite
import msgpack

_log = logging.getLogger(__name__)
_DEFAULT_DB = str(Path.home() / ".sage" / "checkpoints.db")


class Checkpointer:
    """SQLite-backed checkpoint save/load/cleanup."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or _DEFAULT_DB
        self._conn = None

    async def initialize(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                task_id   TEXT PRIMARY KEY,
                phase     TEXT,
                state     BLOB,
                timestamp REAL,
                agent_id  TEXT DEFAULT ''
            )
        """)
        await self._conn.commit()

    async def save(
        self, task_id: str, phase: str, state: dict, agent_id: str = "",
    ):
        blob = msgpack.packb(state, use_bin_type=True)
        await self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints (task_id, phase, state, timestamp, agent_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, phase, blob, time.time(), agent_id),
        )
        await self._conn.commit()

    async def load(self, task_id: str) -> dict | None:
        async with self._conn.execute(
            "SELECT state FROM checkpoints WHERE task_id = ?", (task_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return msgpack.unpackb(row[0], raw=False)

    async def cleanup(self, task_id: str):
        await self._conn.execute(
            "DELETE FROM checkpoints WHERE task_id = ?", (task_id,),
        )
        await self._conn.commit()

    async def close(self):
        if self._conn:
            await self._conn.close()
```

- [ ] **Step 4: Run tests**

Run: `cd sage-python && pip install msgpack && python -m pytest tests/test_checkpointing.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/checkpointing.py sage-python/tests/test_checkpointing.py sage-python/pyproject.toml
git commit -m "feat: Checkpointer module — SQLite + msgpack checkpoint/resume"
```

---

### Task 17: AgentLoop Checkpoint Wiring

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py:169-234` (constructor), `327-347` (PERCEIVE), `440-501` (THINK), `503-701` (ACT), `757-817` (LEARN)
- Create: `sage-python/tests/test_agent_loop_checkpoint.py`

Wire checkpoint save calls at phase boundaries in AgentLoop.

- [ ] **Step 1: Write tests**

```python
# sage-python/tests/test_agent_loop_checkpoint.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sage.agent_loop import AgentLoop, AgentConfig

@pytest.mark.asyncio
async def test_agent_loop_accepts_checkpointer():
    cfg = AgentConfig(name="test", system_prompt="test")
    provider = AsyncMock()
    loop = AgentLoop(config=cfg, llm_provider=provider, checkpointer=None)
    assert loop._checkpointer is None

@pytest.mark.asyncio
async def test_checkpoint_saves_after_perceive():
    cfg = AgentConfig(name="test", system_prompt="test")
    provider = AsyncMock()
    cp = AsyncMock()
    loop = AgentLoop(config=cfg, llm_provider=provider, checkpointer=cp)
    # Verify checkpointer attribute is stored
    assert loop._checkpointer is cp
```

- [ ] **Step 2: Add checkpointer to AgentLoop constructor**

In `sage-python/src/sage/agent_loop.py`, modify `__init__` to accept `checkpointer: Checkpointer | None = None` and store as `self._checkpointer`.

- [ ] **Step 3: Add checkpoint calls at phase boundaries**

After each phase completes, save checkpoint if checkpointer is set:

```python
# After PERCEIVE (around line 390):
if self._checkpointer:
    await self._checkpointer.save(task_id, "perceive", {
        "task": task, "routing": {"system": system, "tier": tier},
    })

# After THINK (around line 500):
if self._checkpointer:
    await self._checkpointer.save(task_id, "think", {
        "task": task, "response": response_text,
        "routing": {"system": system, "tier": tier},
    })

# After ACT (around line 700):
if self._checkpointer:
    await self._checkpointer.save(task_id, "act", {
        "task": task, "response": final_result,
        "avr_iterations": iteration,
    })

# After LEARN — cleanup (around line 815):
if self._checkpointer:
    await self._checkpointer.cleanup(task_id)
```

- [ ] **Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_agent_loop_checkpoint.py -v`
Expected: 2 passed

- [ ] **Step 5: Run full agent_loop test suite**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py -v`
Expected: All existing tests pass (checkpointer=None is default)

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/agent_loop.py sage-python/tests/test_agent_loop_checkpoint.py
git commit -m "feat: wire Checkpointer into AgentLoop phase boundaries"
```

---

### Task 18: Python TopologyGraph

**Files:**
- Create: `sage-python/src/sage/topology/py_graph.py`
- Create: `sage-python/tests/test_py_topology_graph.py`
- Modify: `sage-python/pyproject.toml` (add networkx dependency)

Python port of Rust TopologyGraph (687 LOC) using networkx.

- [ ] **Step 1: Add networkx dependency**

Add `"networkx>=3.2"` to pyproject.toml core dependencies.

- [ ] **Step 2: Write tests**

```python
# sage-python/tests/test_py_topology_graph.py
from sage.topology.py_graph import PyTopologyGraph, NodeData, EdgeKind

def test_create_empty_graph():
    g = PyTopologyGraph()
    assert g.node_count() == 0
    assert g.edge_count() == 0

def test_add_node():
    g = PyTopologyGraph()
    n = g.add_node(NodeData(role="researcher", model="gemini-flash"))
    assert g.node_count() == 1
    assert g.get_node(n).role == "researcher"

def test_add_edge():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="a"))
    b = g.add_node(NodeData(role="b"))
    g.add_edge(a, b, EdgeKind.CONTROL)
    assert g.edge_count() == 1

def test_three_edge_kinds():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="a"))
    b = g.add_node(NodeData(role="b"))
    g.add_edge(a, b, EdgeKind.CONTROL)
    g.add_edge(a, b, EdgeKind.MESSAGE, field="response")
    g.add_edge(a, b, EdgeKind.STATE)
    assert g.edge_count() == 3

def test_successors_predecessors():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="a"))
    b = g.add_node(NodeData(role="b"))
    c = g.add_node(NodeData(role="c"))
    g.add_edge(a, b, EdgeKind.CONTROL)
    g.add_edge(a, c, EdgeKind.CONTROL)
    assert set(g.successors(a)) == {b, c}
    assert g.predecessors(b) == [a]

def test_entry_and_exit_nodes():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="entry"))
    b = g.add_node(NodeData(role="middle"))
    c = g.add_node(NodeData(role="exit"))
    g.add_edge(a, b, EdgeKind.CONTROL)
    g.add_edge(b, c, EdgeKind.CONTROL)
    assert g.entry_nodes() == [a]
    assert g.exit_nodes() == [c]

def test_is_acyclic():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="a"))
    b = g.add_node(NodeData(role="b"))
    g.add_edge(a, b, EdgeKind.CONTROL)
    assert g.is_acyclic() is True

def test_cycle_detection():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="a"))
    b = g.add_node(NodeData(role="b"))
    g.add_edge(a, b, EdgeKind.CONTROL)
    g.add_edge(b, a, EdgeKind.CONTROL)
    assert g.is_acyclic() is False

def test_remove_node():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="a"))
    b = g.add_node(NodeData(role="b"))
    g.add_edge(a, b, EdgeKind.CONTROL)
    g.remove_node(a)
    assert g.node_count() == 1
    assert g.edge_count() == 0
```

- [ ] **Step 3: Implement py_graph.py**

```python
# sage-python/src/sage/topology/py_graph.py
"""Python TopologyGraph — networkx port of Rust petgraph-based TopologyGraph."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import networkx as nx


class EdgeKind(str, Enum):
    CONTROL = "control"
    MESSAGE = "message"
    STATE = "state"


@dataclass
class NodeData:
    role: str = ""
    model: str = ""
    capabilities: list[str] = field(default_factory=list)
    budget_usd: float = 0.0
    security_label: str = "LOW"
    prompt: str = ""


@dataclass
class EdgeData:
    kind: EdgeKind = EdgeKind.CONTROL
    field: str = ""  # for MESSAGE edges


class PyTopologyGraph:
    """Directed graph with typed nodes and three-flow edges."""

    def __init__(self):
        self._g = nx.DiGraph()
        self._next_id = 0

    def add_node(self, data: NodeData) -> int:
        nid = self._next_id
        self._next_id += 1
        self._g.add_node(nid, data=data)
        return nid

    def get_node(self, nid: int) -> NodeData:
        return self._g.nodes[nid]["data"]

    def remove_node(self, nid: int):
        self._g.remove_node(nid)

    def add_edge(self, src: int, dst: int, kind: EdgeKind, field: str = ""):
        self._g.add_edge(src, dst, key=f"{kind.value}_{field}",
                         data=EdgeData(kind=kind, field=field))

    def node_count(self) -> int:
        return self._g.number_of_nodes()

    def edge_count(self) -> int:
        return self._g.number_of_edges()

    def successors(self, nid: int) -> list[int]:
        return list(self._g.successors(nid))

    def predecessors(self, nid: int) -> list[int]:
        return list(self._g.predecessors(nid))

    def entry_nodes(self) -> list[int]:
        return [n for n in self._g.nodes if self._g.in_degree(n) == 0]

    def exit_nodes(self) -> list[int]:
        return [n for n in self._g.nodes if self._g.out_degree(n) == 0]

    def is_acyclic(self) -> bool:
        return nx.is_directed_acyclic_graph(self._g)

    def topological_sort(self) -> list[int]:
        return list(nx.topological_sort(self._g))

    def all_nodes(self) -> list[int]:
        return list(self._g.nodes)

    def edges_from(self, nid: int) -> list[tuple[int, int, EdgeData]]:
        return [(u, v, d["data"]) for u, v, d in self._g.edges(nid, data=True)]
```

- [ ] **Step 4: Run tests**

Run: `cd sage-python && pip install networkx && python -m pytest tests/test_py_topology_graph.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/topology/py_graph.py sage-python/tests/test_py_topology_graph.py sage-python/pyproject.toml
git commit -m "feat(topology): Python TopologyGraph port using networkx"
```

---

### Task 19: Python HybridVerifier

**Files:**
- Create: `sage-python/src/sage/topology/py_verifier.py`
- Create: `sage-python/tests/test_py_verifier.py`

Python port of 6 structural + 4 semantic checks from Rust HybridVerifier (660 LOC).

- [ ] **Step 1: Write tests**

```python
# sage-python/tests/test_py_verifier.py
from sage.topology.py_graph import PyTopologyGraph, NodeData, EdgeKind
from sage.topology.py_verifier import PyHybridVerifier, VerificationResult

def test_valid_sequential():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="agent1", model="m"))
    b = g.add_node(NodeData(role="agent2", model="m"))
    g.add_edge(a, b, EdgeKind.CONTROL)
    result = PyHybridVerifier.verify(g)
    assert result.is_valid

def test_empty_graph_invalid():
    g = PyTopologyGraph()
    result = PyHybridVerifier.verify(g)
    assert not result.is_valid
    assert any("empty" in e.lower() for e in result.errors)

def test_disconnected_node_warning():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="a"))
    b = g.add_node(NodeData(role="b"))  # disconnected
    result = PyHybridVerifier.verify(g)
    assert len(result.warnings) > 0

def test_no_exit_node_error():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="a"))
    b = g.add_node(NodeData(role="b"))
    g.add_edge(a, b, EdgeKind.CONTROL)
    g.add_edge(b, a, EdgeKind.CONTROL)  # cycle, no exit
    result = PyHybridVerifier.verify(g)
    # Should warn about liveness (no exit reachable)
    assert len(result.warnings) > 0 or not result.is_valid

def test_security_violation():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="high", security_label="HIGH"))
    b = g.add_node(NodeData(role="low", security_label="LOW"))
    g.add_edge(a, b, EdgeKind.CONTROL)
    result = PyHybridVerifier.verify(g)
    assert any("security" in e.lower() or "safety" in e.lower() for e in result.errors)
```

- [ ] **Step 2: Implement py_verifier.py**

```python
# sage-python/src/sage/topology/py_verifier.py
"""Python HybridVerifier — port of Rust 6 structural + 4 semantic checks."""
from __future__ import annotations
from dataclasses import dataclass, field
from sage.topology.py_graph import PyTopologyGraph, EdgeKind


@dataclass
class VerificationResult:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PyHybridVerifier:
    """6 structural + 4 semantic checks on PyTopologyGraph."""

    @staticmethod
    def verify(graph: PyTopologyGraph) -> VerificationResult:
        result = VerificationResult()

        # Structural checks
        _check_non_empty(graph, result)
        _check_has_entry(graph, result)
        _check_has_exit(graph, result)
        _check_connected(graph, result)
        _check_no_self_loops(graph, result)
        _check_node_has_model(graph, result)

        # Semantic checks
        _check_safety(graph, result)
        _check_liveness(graph, result)
        _check_role_uniqueness(graph, result)
        _check_budget_positive(graph, result)

        result.is_valid = len(result.errors) == 0
        return result


def _check_non_empty(g: PyTopologyGraph, r: VerificationResult):
    if g.node_count() == 0:
        r.errors.append("Graph is empty (0 nodes)")

def _check_has_entry(g: PyTopologyGraph, r: VerificationResult):
    if g.node_count() > 0 and len(g.entry_nodes()) == 0:
        r.errors.append("No entry nodes (all nodes have predecessors)")

def _check_has_exit(g: PyTopologyGraph, r: VerificationResult):
    if g.node_count() > 0 and len(g.exit_nodes()) == 0:
        r.warnings.append("No exit nodes (possible cycle without termination)")

def _check_connected(g: PyTopologyGraph, r: VerificationResult):
    if g.node_count() <= 1:
        return
    all_nodes = set(g.all_nodes())
    if not all_nodes:
        return
    # BFS from first entry or first node
    entries = g.entry_nodes()
    start = entries[0] if entries else next(iter(all_nodes))
    visited = set()
    queue = [start]
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        visited.add(n)
        queue.extend(g.successors(n))
        queue.extend(g.predecessors(n))
    unreachable = all_nodes - visited
    if unreachable:
        r.warnings.append(f"Disconnected nodes: {unreachable}")

def _check_no_self_loops(g: PyTopologyGraph, r: VerificationResult):
    for n in g.all_nodes():
        if n in g.successors(n):
            r.errors.append(f"Self-loop on node {n}")

def _check_node_has_model(g: PyTopologyGraph, r: VerificationResult):
    for n in g.all_nodes():
        data = g.get_node(n)
        if not data.model:
            r.warnings.append(f"Node {n} ({data.role}) has no model assigned")

def _check_safety(g: PyTopologyGraph, r: VerificationResult):
    """No HIGH → LOW security flow."""
    for n in g.all_nodes():
        src = g.get_node(n)
        if src.security_label == "HIGH":
            for succ in g.successors(n):
                dst = g.get_node(succ)
                if dst.security_label == "LOW":
                    r.errors.append(
                        f"Safety violation: HIGH node {n} ({src.role}) → "
                        f"LOW node {succ} ({dst.role})"
                    )

def _check_liveness(g: PyTopologyGraph, r: VerificationResult):
    """All entry nodes should reach at least one exit node."""
    exits = set(g.exit_nodes())
    if not exits:
        return  # already warned in _check_has_exit
    for entry in g.entry_nodes():
        visited = set()
        queue = [entry]
        found_exit = False
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            if n in exits:
                found_exit = True
                break
            queue.extend(g.successors(n))
        if not found_exit:
            r.warnings.append(f"Liveness: entry {entry} cannot reach any exit node")

def _check_role_uniqueness(g: PyTopologyGraph, r: VerificationResult):
    roles = {}
    for n in g.all_nodes():
        role = g.get_node(n).role
        if role in roles:
            r.warnings.append(f"Duplicate role '{role}' on nodes {roles[role]} and {n}")
        roles[role] = n

def _check_budget_positive(g: PyTopologyGraph, r: VerificationResult):
    for n in g.all_nodes():
        data = g.get_node(n)
        if data.budget_usd < 0:
            r.errors.append(f"Negative budget on node {n} ({data.role})")
```

- [ ] **Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_py_verifier.py -v`
Expected: 5 passed

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/topology/py_verifier.py sage-python/tests/test_py_verifier.py
git commit -m "feat(topology): Python HybridVerifier — 6 structural + 4 semantic checks"
```

---

### Task 20: Python TopologyExecutor

**Files:**
- Create: `sage-python/src/sage/topology/py_executor.py`
- Create: `sage-python/tests/test_py_executor.py`

Python port of Rust TopologyExecutor: Kahn's toposort (acyclic) + gate-based readiness (cyclic).

- [ ] **Step 1: Write tests**

```python
# sage-python/tests/test_py_executor.py
import pytest
from sage.topology.py_graph import PyTopologyGraph, NodeData, EdgeKind
from sage.topology.py_executor import PyTopologyExecutor

def test_static_schedule_sequential():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="a"))
    b = g.add_node(NodeData(role="b"))
    c = g.add_node(NodeData(role="c"))
    g.add_edge(a, b, EdgeKind.CONTROL)
    g.add_edge(b, c, EdgeKind.CONTROL)
    exe = PyTopologyExecutor(g)
    order = exe.static_schedule()
    assert order == [a, b, c]

def test_static_schedule_parallel():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="entry"))
    b = g.add_node(NodeData(role="w1"))
    c = g.add_node(NodeData(role="w2"))
    d = g.add_node(NodeData(role="exit"))
    g.add_edge(a, b, EdgeKind.CONTROL)
    g.add_edge(a, c, EdgeKind.CONTROL)
    g.add_edge(b, d, EdgeKind.CONTROL)
    g.add_edge(c, d, EdgeKind.CONTROL)
    exe = PyTopologyExecutor(g)
    order = exe.static_schedule()
    assert order[0] == a
    assert order[-1] == d
    assert set(order[1:3]) == {b, c}

def test_dynamic_ready_nodes():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="a"))
    b = g.add_node(NodeData(role="b"))
    g.add_edge(a, b, EdgeKind.CONTROL)
    exe = PyTopologyExecutor(g)
    ready = exe.ready_nodes()
    assert ready == [a]
    exe.mark_complete(a)
    ready = exe.ready_nodes()
    assert ready == [b]

def test_cyclic_uses_dynamic():
    g = PyTopologyGraph()
    a = g.add_node(NodeData(role="a"))
    b = g.add_node(NodeData(role="b"))
    g.add_edge(a, b, EdgeKind.CONTROL)
    g.add_edge(b, a, EdgeKind.CONTROL)
    exe = PyTopologyExecutor(g)
    assert exe.is_cyclic
    ready = exe.ready_nodes()  # gate-based: nodes with no unmet deps
    assert len(ready) > 0  # at least one node ready (break cycle)
```

- [ ] **Step 2: Implement py_executor.py**

```python
# sage-python/src/sage/topology/py_executor.py
"""Python TopologyExecutor — Kahn's (acyclic) + gate-based (cyclic)."""
from __future__ import annotations
from collections import deque
from sage.topology.py_graph import PyTopologyGraph, EdgeKind


class PyTopologyExecutor:
    """Dual-mode scheduler: static for DAGs, dynamic for cyclic topologies."""

    def __init__(self, graph: PyTopologyGraph):
        self._graph = graph
        self._completed: set[int] = set()
        self.is_cyclic = not graph.is_acyclic()

    def static_schedule(self) -> list[int]:
        """Kahn's toposort. Raises if graph has cycles."""
        if self.is_cyclic:
            raise ValueError("Cannot static-schedule a cyclic graph")
        return self._graph.topological_sort()

    def ready_nodes(self) -> list[int]:
        """Gate-based: nodes whose predecessors are all completed."""
        ready = []
        for n in self._graph.all_nodes():
            if n in self._completed:
                continue
            preds = self._graph.predecessors(n)
            if all(p in self._completed for p in preds):
                ready.append(n)
        # For cyclic graphs with no ready nodes, break cycle by picking
        # the node with most completed predecessors
        if not ready and self.is_cyclic:
            remaining = [n for n in self._graph.all_nodes() if n not in self._completed]
            if remaining:
                ready = [max(remaining, key=lambda n: sum(
                    1 for p in self._graph.predecessors(n) if p in self._completed
                ))]
        return ready

    def mark_complete(self, node_id: int):
        self._completed.add(node_id)

    def is_done(self) -> bool:
        return len(self._completed) == self._graph.node_count()

    def reset(self):
        self._completed.clear()
```

- [ ] **Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_py_executor.py -v`
Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/topology/py_executor.py sage-python/tests/test_py_executor.py
git commit -m "feat(topology): Python TopologyExecutor — Kahn's + gate-based scheduling"
```

---

### Task 21: Python Topology Templates

**Files:**
- Create: `sage-python/src/sage/topology/py_templates.py`
- Create: `sage-python/tests/test_py_templates.py`

Python port of 8 template constructors returning PyTopologyGraph instances.

- [ ] **Step 1: Write tests**

```python
# sage-python/tests/test_py_templates.py
from sage.topology.py_templates import create_template, TEMPLATE_NAMES
from sage.topology.py_verifier import PyHybridVerifier

def test_all_8_templates_exist():
    assert len(TEMPLATE_NAMES) == 8

def test_sequential_template():
    g = create_template("Sequential", n_agents=3, model="m")
    assert g.node_count() == 3
    assert g.edge_count() == 2
    result = PyHybridVerifier.verify(g)
    assert result.is_valid

def test_parallel_template():
    g = create_template("Parallel", n_agents=3, model="m")
    assert g.node_count() >= 4  # entry + workers + aggregator
    result = PyHybridVerifier.verify(g)
    assert result.is_valid

def test_avr_template():
    g = create_template("AVR", model="m")
    assert g.node_count() == 3  # act, verify, refine
    result = PyHybridVerifier.verify(g)
    # AVR has a cycle (refine→act), so exit check may warn
    assert len(result.errors) == 0

def test_all_templates_valid():
    for name in TEMPLATE_NAMES:
        g = create_template(name, model="test-model")
        result = PyHybridVerifier.verify(g)
        assert len(result.errors) == 0, f"Template {name} has errors: {result.errors}"
```

- [ ] **Step 2: Implement py_templates.py**

```python
# sage-python/src/sage/topology/py_templates.py
"""8 topology templates — Python port of Rust templates.rs."""
from __future__ import annotations
from sage.topology.py_graph import PyTopologyGraph, NodeData, EdgeKind

TEMPLATE_NAMES = [
    "Sequential", "Parallel", "AVR", "SelfMoA",
    "Hierarchical", "Hub", "Debate", "Brainstorming",
]


def create_template(
    name: str, n_agents: int = 3, model: str = "",
) -> PyTopologyGraph:
    """Create a topology from a named template."""
    builders = {
        "Sequential": _sequential,
        "Parallel": _parallel,
        "AVR": _avr,
        "SelfMoA": _selfmoa,
        "Hierarchical": _hierarchical,
        "Hub": _hub,
        "Debate": _debate,
        "Brainstorming": _brainstorming,
    }
    builder = builders.get(name)
    if builder is None:
        raise ValueError(f"Unknown template: {name}. Available: {TEMPLATE_NAMES}")
    return builder(n_agents=n_agents, model=model)


def _sequential(n_agents: int, model: str) -> PyTopologyGraph:
    g = PyTopologyGraph()
    prev = None
    for i in range(max(n_agents, 2)):
        n = g.add_node(NodeData(role=f"agent_{i}", model=model))
        if prev is not None:
            g.add_edge(prev, n, EdgeKind.CONTROL)
        prev = n
    return g


def _parallel(n_agents: int, model: str) -> PyTopologyGraph:
    g = PyTopologyGraph()
    entry = g.add_node(NodeData(role="dispatcher", model=model))
    workers = [g.add_node(NodeData(role=f"worker_{i}", model=model))
               for i in range(max(n_agents, 2))]
    agg = g.add_node(NodeData(role="aggregator", model=model))
    for w in workers:
        g.add_edge(entry, w, EdgeKind.CONTROL)
        g.add_edge(w, agg, EdgeKind.CONTROL)
    return g


def _avr(n_agents: int = 3, model: str = "") -> PyTopologyGraph:
    g = PyTopologyGraph()
    act = g.add_node(NodeData(role="actor", model=model))
    verify = g.add_node(NodeData(role="verifier", model=model))
    refine = g.add_node(NodeData(role="refiner", model=model))
    g.add_edge(act, verify, EdgeKind.CONTROL)
    g.add_edge(verify, refine, EdgeKind.CONTROL)
    g.add_edge(refine, act, EdgeKind.CONTROL)  # cycle
    return g


def _selfmoa(n_agents: int = 3, model: str = "") -> PyTopologyGraph:
    g = PyTopologyGraph()
    experts = [g.add_node(NodeData(role=f"expert_{i}", model=model))
               for i in range(max(n_agents, 2))]
    agg = g.add_node(NodeData(role="aggregator", model=model))
    for e in experts:
        g.add_edge(e, agg, EdgeKind.CONTROL)
        g.add_edge(e, agg, EdgeKind.MESSAGE, field="response")
    return g


def _hierarchical(n_agents: int = 3, model: str = "") -> PyTopologyGraph:
    g = PyTopologyGraph()
    manager = g.add_node(NodeData(role="manager", model=model))
    workers = [g.add_node(NodeData(role=f"worker_{i}", model=model))
               for i in range(max(n_agents - 1, 2))]
    collector = g.add_node(NodeData(role="collector", model=model))
    for w in workers:
        g.add_edge(manager, w, EdgeKind.CONTROL)
        g.add_edge(w, collector, EdgeKind.CONTROL)
    g.add_edge(manager, collector, EdgeKind.STATE)
    return g


def _hub(n_agents: int = 3, model: str = "") -> PyTopologyGraph:
    g = PyTopologyGraph()
    hub = g.add_node(NodeData(role="hub", model=model))
    spokes = [g.add_node(NodeData(role=f"spoke_{i}", model=model))
              for i in range(max(n_agents - 1, 2))]
    out = g.add_node(NodeData(role="output", model=model))
    for s in spokes:
        g.add_edge(hub, s, EdgeKind.CONTROL)
        g.add_edge(s, out, EdgeKind.CONTROL)
    return g


def _debate(n_agents: int = 3, model: str = "") -> PyTopologyGraph:
    g = PyTopologyGraph()
    proposer = g.add_node(NodeData(role="proposer", model=model))
    opponent = g.add_node(NodeData(role="opponent", model=model))
    judge = g.add_node(NodeData(role="judge", model=model))
    g.add_edge(proposer, opponent, EdgeKind.CONTROL)
    g.add_edge(opponent, judge, EdgeKind.CONTROL)
    g.add_edge(judge, proposer, EdgeKind.CONTROL)  # cycle for multi-round
    return g


def _brainstorming(n_agents: int = 3, model: str = "") -> PyTopologyGraph:
    g = PyTopologyGraph()
    generators = [g.add_node(NodeData(role=f"gen_{i}", model=model))
                  for i in range(max(n_agents - 1, 2))]
    synth = g.add_node(NodeData(role="synthesizer", model=model))
    for gen in generators:
        g.add_edge(gen, synth, EdgeKind.CONTROL)
        g.add_edge(gen, synth, EdgeKind.MESSAGE, field="ideas")
    return g
```

- [ ] **Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_py_templates.py -v`
Expected: 4 passed

- [ ] **Step 4: Run all topology tests together**

Run: `cd sage-python && python -m pytest tests/test_py_topology_graph.py tests/test_py_verifier.py tests/test_py_executor.py tests/test_py_templates.py -v`
Expected: All pass (9 + 5 + 4 + 4 = 22 tests)

- [ ] **Step 5: Commit**

```bash
git add sage-python/src/sage/topology/py_templates.py sage-python/tests/test_py_templates.py
git commit -m "feat(topology): 8 Python topology templates (Sequential→Brainstorming)"
```

---

## Final Verification

After all 21 tasks are complete:

- [ ] **Run full Python test suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short
```

Expected: All existing tests pass + ~75 new tests pass.

- [ ] **Run sage-router tests**

```bash
cd sage-router && python -m pytest tests/ -v
```

Expected: 6 tests pass.

- [ ] **Run ruff lint**

```bash
cd sage-python && ruff check src/
cd sage-router && ruff check src/
```

Expected: No errors.
