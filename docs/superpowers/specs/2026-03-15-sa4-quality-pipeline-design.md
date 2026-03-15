# SA-4: Z3 Auto-Label Quality Pipeline — Design Spec

## Problem

The QualityEstimator is a 5-signal heuristic (Pearson r=0.34). Heuristics produce mediocre results — proven by research (ETH-SRI ICLR '25, Cascade Routing 2410.10347). The bandit receives imprecise signals and cannot converge. The evolution engine cannot select the best topologies. The TopologyController cannot make good upgrade/prune decisions.

**Principle: zero heuristics.** Quality scoring must be either formally verified (Z3 proofs) or learned (trained model). When neither is available, abstain — do not emit a false signal.

## Architecture

```
Task + Response
      │
      ▼
Rust QualityLabeler (sage-core/src/verification/quality_labeler.rs)
  ├── tree-sitter AST parse (syntax valid?)
  ├── OxiZ verify_arithmetic (math correct?)
  ├── OxiZ prove_memory_safety (bounds safe?)
  ├── OxiZ verify_invariant (loops correct?)
  └── Execution test (subprocess, pass/fail?)
      │
      ▼
  QualityLabel { score: f32, checks: Vec<CheckResult> }
      │
      ├──[offline]──▶ JSONL accumulation → train DistilBERT → ONNX export
      │                                                          │
      ▼                                                          ▼
Pipeline Stage 5 LEARN                           Rust RustQualityEstimator
  │                                                (ort ONNX inference)
  │                                                          │
  ▼                                                          ▼
Bandit.record(quality)                           Pipeline Stage 5 LEARN
                                                  Bandit.record(quality)
```

Two modes:
- **Labeling mode** (offline): Z3 labeler generates ground-truth labels for training
- **Inference mode** (runtime): Trained ONNX model scores in <1ms via Rust ort

No heuristic fallback. If ONNX model not available AND Z3 labeler cannot assess (non-code task), quality = None → bandit skips recording.

## Components

### 1. Rust: `sage-core/src/verification/quality_labeler.rs` (NEW)

PyO3 class `QualityLabeler` with a single method:

```rust
#[pyclass]
pub struct QualityLabeler {
    smt: SmtVerifier,
    // tree-sitter validator reused from sandbox/validator.rs
}

#[pymethods]
impl QualityLabeler {
    #[new]
    pub fn new() -> Self { ... }

    /// Label a (task, response) pair with formal verification.
    /// Returns None if the response is not assessable (no code, no math).
    /// Returns Some(QualityLabel) with score 0.0-1.0 based on verified properties.
    pub fn label(&self, task: &str, response: &str) -> Option<QualityLabel> {
        // 1. Extract code blocks from response
        // 2. If no code found, return None (not assessable)
        // 3. Run formal checks:
        //    - syntax_valid: tree-sitter parse succeeds (bool)
        //    - no_dangerous_imports: validator check (bool)
        //    - arithmetic_correct: extract numeric assertions, verify via OxiZ (Option<bool>)
        //    - memory_safe: extract bounds, verify via OxiZ (Option<bool>)
        //    - invariants_hold: extract loop invariants, verify via OxiZ (Option<bool>)
        // 4. Score = verified_count / applicable_count
        //    (only count checks that were applicable — don't penalize for N/A)
    }
}

#[pyclass]
pub struct QualityLabel {
    #[pyo3(get)]
    pub score: f32,          // 0.0-1.0
    #[pyo3(get)]
    pub checks_passed: u32,  // number of checks that passed
    #[pyo3(get)]
    pub checks_total: u32,   // number of applicable checks
    #[pyo3(get)]
    pub assessable: bool,    // false if response had no verifiable content
}
```

Feature gate: `smt` + `tool-executor` (needs SmtVerifier + tree-sitter).

**Not a heuristic:** Each check is a formal proof (SAT/UNSAT). Score = proven_properties / applicable_properties. No weights, no thresholds, no magic numbers.

### 2. Rust: `sage-core/src/verification/quality_estimator_onnx.rs` (NEW)

PyO3 class `RustLearnedQualityEstimator`:

```rust
#[pyclass]
pub struct RustLearnedQualityEstimator {
    session: ort::Session,
    tokenizer: tokenizers::Tokenizer,
}

#[pymethods]
impl RustLearnedQualityEstimator {
    /// Load ONNX model + tokenizer from paths.
    #[new]
    pub fn new(model_path: &str, tokenizer_path: &str) -> PyResult<Self> { ... }

    /// Estimate quality of (task, response) pair.
    /// Returns score 0.0-1.0 from the trained DistilBERT model.
    pub fn estimate(&self, task: &str, response: &str) -> f32 { ... }
}
```

Feature gate: `onnx` (needs ort + tokenizers, same as RustEmbedder).

Pattern: identical to `memory/embedder.rs` — load ONNX model, tokenize input, run inference, return scalar.

### 3. Python: `sage-python/scripts/collect_quality_labels.py` (NEW)

Offline script that uses QualityLabeler to generate training data:

```python
async def collect_labels(dataset="humaneval", limit=None):
    """Run tasks through SAGE, label with Z3, save JSONL."""
    labeler = QualityLabeler()  # Rust
    system = boot_agent_system(...)

    for task_id, task in load_tasks(dataset):
        response = await system.run(task["prompt"])
        label = labeler.label(task["prompt"], response)
        if label and label.assessable:
            save_jsonl({
                "task": task["prompt"],
                "response": response,
                "score": label.score,
                "checks_passed": label.checks_passed,
                "checks_total": label.checks_total,
            })
```

Output: `data/quality_labels.jsonl` — ground truth for training.

### 4. Python: `sage-python/scripts/train_quality_model.py` (EXISTS, extend)

Already exists. Extend to:
- Read `data/quality_labels.jsonl` (Z3 labels, not heuristic triples)
- Train DistilBERT regression head
- Export ONNX to `models/quality_estimator_v2.onnx`
- Validate on held-out set (must report Pearson r)

### 5. Python: `sage-python/src/sage/quality_estimator.py` (MODIFY)

Replace the current 5-signal heuristic:

```python
class QualityEstimator:
    def __init__(self):
        # Priority 1: Rust ONNX learned model
        self._learned = self._try_load_onnx()
        # Priority 2: Rust Z3 labeler (formal verification)
        self._labeler = self._try_load_labeler()
        # No priority 3: NO heuristic fallback

    def estimate(self, task: str, response: str, latency_ms: float = 0) -> float | None:
        """Return quality score or None if not assessable."""
        if self._learned:
            return self._learned.estimate(task, response)
        if self._labeler:
            label = self._labeler.label(task, response)
            if label and label.assessable:
                return label.score
            return None  # not assessable
        return None  # nothing available — abstain
```

### 6. Pipeline Stage 5 LEARN (MODIFY)

Already wired (today's fix). Change to respect None:

```python
quality = self.quality_estimator.estimate(ctx.task, ctx.result, ctx.latency_ms)
if quality is not None and self.bandit:
    self.bandit.record(decision_id, quality, cost, latency_ms)
# If quality is None: abstain — don't pollute bandit with false signals
```

## What This Enables

1. **Bandit convergence**: precise quality signals → Thompson sampling converges to optimal model/topology selection
2. **Evolution (SA-3)**: MAP-Elites fitness function uses verified quality, not guesswork
3. **TopologyController**: upgrade/prune decisions based on formal verification, not heuristic thresholds
4. **Self-training**: the system generates its own training data via Z3, trains its own model, loads it at runtime

## Success Criteria

- QualityLabeler produces non-None labels for ≥80% of code tasks
- Trained DistilBERT achieves Pearson r ≥ 0.6 on held-out set (vs current r=0.34)
- Bandit converges faster (measured by regret curve on BigCodeBench)
- Zero heuristic scoring paths remain in production code

## Files

| File | Action | LOC estimate |
|------|--------|-------------|
| `sage-core/src/verification/quality_labeler.rs` | CREATE | ~200 |
| `sage-core/src/verification/quality_estimator_onnx.rs` | CREATE | ~100 |
| `sage-core/src/verification/mod.rs` | MODIFY | +10 |
| `sage-core/src/lib.rs` | MODIFY | +5 (PyO3 exports) |
| `sage-python/src/sage/quality_estimator.py` | REWRITE | ~80 |
| `sage-python/src/sage/pipeline.py` | MODIFY | ~5 (None handling) |
| `sage-python/scripts/collect_quality_labels.py` | CREATE | ~100 |
| `sage-python/scripts/train_quality_model.py` | MODIFY | ~50 |
| `sage-python/tests/test_quality_labeler.py` | CREATE | ~100 |
| `sage-core` Cargo.toml | MODIFY | +2 (feature wiring) |

Total: ~650 LOC, mostly Rust.
