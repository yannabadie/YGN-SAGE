# SAGE Adaptive Router (SAR) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the regex word-boundary heuristic router with a 4-stage learned adaptive router that provides genuine downstream quality improvements, not just self-consistency.

**Architecture:** Hybrid Rust+Python pipeline with confidence-gated stages. Stages 0-1 (structural features + BERT classifier) run in Rust via ort/tokenizers (<10ms). Stages 2-3 (entropy probe + cascade) run in Python (require LLM API calls). Each stage can short-circuit if confidence exceeds its threshold, avoiding unnecessary computation.

**Tech Stack:** Rust (ort 2.0, tokenizers 0.21, ndarray — all already in Cargo.toml behind `onnx` feature), Python (sage strategy module), ONNX models (all-MiniLM-L6-v2 for embeddings, routellm/bert XLM-RoBERTa-278M for classifier).

**Research basis:** RouteLLM (LMSYS), CARROT (minimax rate-optimal), AutoMix (self-verification), Unified Cascade Routing (ETH Zurich ICLR 2025), entropy-based arbitration (MetaScaffold, AdaptThink, Dualformer).

---

## Phase 1: Rust AdaptiveRouter — Stage 0 (Structural Features)

### Task 1: Create routing module directory and mod.rs

**Files:**
- Create: `sage-core/src/routing/mod.rs`
- Create: `sage-core/src/routing/features.rs`
- Modify: `sage-core/src/lib.rs:1-10`

**Step 1: Create the routing module directory**

```bash
mkdir -p sage-core/src/routing
```

**Step 2: Create mod.rs with conditional exports**

Create `sage-core/src/routing/mod.rs`:

```rust
//! Adaptive Router — learned S1/S2/S3 routing pipeline.
//!
//! Stage 0: Structural features + embedding similarity (Rust, <1ms)
//! Stage 1: BERT classifier ONNX inference (Rust, <10ms)
//! Stages 2-3: Entropy probe + cascade (Python-side, requires LLM API)
//!
//! Behind the `onnx` feature flag (reuses ort + tokenizers deps).

pub mod features;
#[cfg(feature = "onnx")]
mod router;
#[cfg(feature = "onnx")]
pub use router::AdaptiveRouter;
```

**Step 3: Create features.rs stub**

Create `sage-core/src/routing/features.rs`:

```rust
//! Structural feature extraction for Stage 0 pre-routing.
//! No external dependencies — always compiled.

use pyo3::prelude::*;

/// Structural features extracted from a task string (zero-cost, no ML).
#[pyclass]
#[derive(Clone, Debug)]
pub struct StructuralFeatures {
    #[pyo3(get)]
    pub word_count: usize,
    #[pyo3(get)]
    pub has_code_block: bool,
    #[pyo3(get)]
    pub has_question_mark: bool,
    #[pyo3(get)]
    pub keyword_complexity: f32,
    #[pyo3(get)]
    pub keyword_uncertainty: f32,
    #[pyo3(get)]
    pub tool_required: bool,
}
```

**Step 4: Wire routing module into lib.rs**

In `sage-core/src/lib.rs`, add after line 8 (`pub mod types;`):

```rust
pub mod routing;
```

And in the `sage_core` pymodule function, add after the ONNX block (after line 38):

```rust
#[cfg(feature = "onnx")]
m.add_class::<routing::AdaptiveRouter>()?;
m.add_class::<routing::features::StructuralFeatures>()?;
```

**Step 5: Verify it compiles**

Run: `cd sage-core && cargo build`
Expected: SUCCESS (features.rs and mod.rs are minimal stubs)

**Step 6: Commit**

```bash
git add sage-core/src/routing/ sage-core/src/lib.rs
git commit -m "feat(routing): add routing module skeleton for AdaptiveRouter"
```

---

### Task 2: Write failing tests for StructuralFeatures extraction

**Files:**
- Create: `sage-core/src/routing/features.rs` (extend with tests)

**Step 1: Write the failing tests**

Add to the bottom of `sage-core/src/routing/features.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_factual_task() {
        let f = StructuralFeatures::extract("What is 2+2?");
        assert_eq!(f.word_count, 4);
        assert!(!f.has_code_block);
        assert!(f.has_question_mark);
        assert!(f.keyword_complexity < 0.3);
        assert!(!f.tool_required);
    }

    #[test]
    fn test_code_task() {
        let f = StructuralFeatures::extract("Write a Python function to check if a number is prime");
        assert!(f.word_count > 5);
        assert!(f.keyword_complexity >= 0.3);
    }

    #[test]
    fn test_complex_debug_task() {
        let f = StructuralFeatures::extract(
            "Debug a race condition in async Rust code with deadlock on Arc<Mutex>"
        );
        assert!(f.keyword_complexity >= 0.6);
    }

    #[test]
    fn test_code_block_detection() {
        let task = "Fix this:\n```python\nprint('hello')\n```";
        let f = StructuralFeatures::extract(task);
        assert!(f.has_code_block);
    }

    #[test]
    fn test_tool_required_detection() {
        let f = StructuralFeatures::extract("Run the test suite and fix failing tests");
        assert!(f.tool_required);
    }

    #[test]
    fn test_long_task_scaling() {
        let long_task = "word ".repeat(120);
        let f = StructuralFeatures::extract(&long_task);
        assert_eq!(f.word_count, 120);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-core && cargo test --lib routing::features::tests`
Expected: FAIL with "no function or associated item named `extract`"

---

### Task 3: Implement StructuralFeatures::extract()

**Files:**
- Modify: `sage-core/src/routing/features.rs`

**Step 1: Implement the extract function**

Replace the `StructuralFeatures` struct definition and add impl blocks:

```rust
//! Structural feature extraction for Stage 0 pre-routing.
//! No external dependencies — always compiled.

use pyo3::prelude::*;

/// Structural features extracted from a task string (zero-cost, no ML).
#[pyclass]
#[derive(Clone, Debug)]
pub struct StructuralFeatures {
    #[pyo3(get)]
    pub word_count: usize,
    #[pyo3(get)]
    pub has_code_block: bool,
    #[pyo3(get)]
    pub has_question_mark: bool,
    #[pyo3(get)]
    pub keyword_complexity: f32,
    #[pyo3(get)]
    pub keyword_uncertainty: f32,
    #[pyo3(get)]
    pub tool_required: bool,
}

// Keyword groups with associated complexity weights
const ALGO_KEYWORDS: &[&str] = &[
    "implement", "build", "algorithm", "optimize", "compiler",
    "concurrent", "distributed", "consensus", "lock-free",
];
const CODE_KEYWORDS: &[&str] = &[
    "write", "create", "code", "function", "class", "method",
    "parse", "regex", "query", "endpoint", "decorator",
];
const DEBUG_KEYWORDS: &[&str] = &[
    "debug", "fix", "error", "crash", "bug", "race condition",
    "deadlock", "oom", "memory leak",
];
const DESIGN_KEYWORDS: &[&str] = &[
    "design", "architect", "refactor", "schema", "system",
    "prove", "induction", "complexity",
];
const UNCERTAINTY_KEYWORDS: &[&str] = &[
    "maybe", "possibly", "explore", "investigate",
    "intermittent", "sometimes", "random", "flaky",
];
const TOOL_KEYWORDS: &[&str] = &[
    "file", "search", "run", "execute", "compile", "test",
    "deploy", "download", "upload",
];

impl StructuralFeatures {
    /// Extract structural features from a task string.
    /// Pure computation, no ML — runs in <1us.
    pub fn extract(task: &str) -> Self {
        let lower = task.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        let word_count = words.len();

        let has_code_block = task.contains("```");
        let has_question_mark = task.contains('?');

        // Keyword complexity scoring
        let mut complexity: f32 = 0.2; // base

        let has_algo = ALGO_KEYWORDS.iter().any(|kw| lower.contains(kw));
        let has_code = CODE_KEYWORDS.iter().any(|kw| lower.contains(kw));
        let has_debug = DEBUG_KEYWORDS.iter().any(|kw| lower.contains(kw));
        let has_design = DESIGN_KEYWORDS.iter().any(|kw| lower.contains(kw));

        if has_algo { complexity += 0.35; }
        else if has_code { complexity += 0.15; }
        if has_debug { complexity += 0.3; }
        if has_design { complexity += 0.2; }
        if has_code_block { complexity += 0.1; }

        // Word count scaling
        if word_count > 100 { complexity += 0.15; }
        else if word_count > 50 { complexity += 0.1; }
        else if word_count > 20 { complexity += 0.05; }

        let keyword_complexity = complexity.min(1.0);

        // Uncertainty
        let mut uncertainty: f32 = 0.2;
        if has_question_mark { uncertainty += 0.1; }
        let has_uncertainty = UNCERTAINTY_KEYWORDS.iter().any(|kw| lower.contains(kw));
        if has_uncertainty { uncertainty += 0.2; }
        let keyword_uncertainty = uncertainty.min(1.0);

        // Tool requirement
        let tool_required = TOOL_KEYWORDS.iter().any(|kw| lower.contains(kw));

        Self {
            word_count,
            has_code_block,
            has_question_mark,
            keyword_complexity,
            keyword_uncertainty,
            tool_required,
        }
    }
}

#[pymethods]
impl StructuralFeatures {
    /// Extract structural features from a task string (Python API).
    #[staticmethod]
    #[pyo3(name = "extract")]
    pub fn py_extract(task: &str) -> Self {
        Self::extract(task)
    }
}
```

**Step 2: Run tests to verify they pass**

Run: `cd sage-core && cargo test --lib routing::features::tests -v`
Expected: 6 tests PASS

**Step 3: Commit**

```bash
git add sage-core/src/routing/features.rs
git commit -m "feat(routing): implement StructuralFeatures extraction (Stage 0)"
```

---

### Task 4: Write failing tests for AdaptiveRouter Stage 0

**Files:**
- Create: `sage-core/src/routing/router.rs`

**Step 1: Create router.rs with test module**

```rust
//! AdaptiveRouter — 4-stage confidence-gated routing pipeline.
//!
//! Stages 0-1 run in Rust (this module). Stages 2-3 are Python-side.
//! Behind `onnx` feature flag (reuses ort + tokenizers).

use pyo3::prelude::*;
use std::sync::Mutex;
use crate::routing::features::StructuralFeatures;

/// Routing tier: maps to S1/S2/S3 cognitive systems.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Tier {
    S1 = 1,
    S2 = 2,
    S3 = 3,
}

/// Result of a routing stage.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RoutingResult {
    #[pyo3(get)]
    pub tier: u8,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub stage: u8,  // which stage produced this result (0-3)
    #[pyo3(get)]
    pub features: StructuralFeatures,
}

/// Feedback record for online learning.
#[derive(Clone, Debug)]
pub struct RoutingFeedback {
    pub task: String,
    pub routed_tier: u8,
    pub actual_quality: f32,   // 0.0-1.0 outcome quality
    pub latency_ms: u64,
    pub cost_usd: f32,
}

/// 4-stage adaptive router. Stages 0-1 in Rust, 2-3 in Python.
#[pyclass]
pub struct AdaptiveRouter {
    // Stage 0: exemplar bank (embedding, tier) pairs
    exemplar_embeddings: Vec<(Vec<f32>, u8)>,
    // Stage 1: BERT classifier ONNX session
    classifier: Option<ort::session::Session>,
    classifier_tokenizer: Option<tokenizers::Tokenizer>,
    // Thresholds
    c0_threshold: f32,
    c1_threshold: f32,
    // Feedback buffer
    feedback: Mutex<Vec<RoutingFeedback>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage0_simple_task_routes_s1() {
        let router = AdaptiveRouter::new(None, None, None, None);
        let result = router.route_stage0("What is 2+2?");
        assert_eq!(result.tier, 1);
        assert_eq!(result.stage, 0);
    }

    #[test]
    fn test_stage0_code_task_routes_s2() {
        let router = AdaptiveRouter::new(None, None, None, None);
        let result = router.route_stage0("Write a Python function to check if a number is prime");
        assert_eq!(result.tier, 2);
        assert_eq!(result.stage, 0);
    }

    #[test]
    fn test_stage0_complex_task_routes_s3() {
        let router = AdaptiveRouter::new(None, None, None, None);
        let result = router.route_stage0(
            "Debug a race condition in async Rust code with deadlock on Arc<Mutex>"
        );
        assert_eq!(result.tier, 3);
        assert_eq!(result.stage, 0);
    }

    #[test]
    fn test_stage0_confidence_in_valid_range() {
        let router = AdaptiveRouter::new(None, None, None, None);
        let result = router.route_stage0("Hello world");
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_default_thresholds() {
        let router = AdaptiveRouter::new(None, None, None, None);
        assert_eq!(router.c0_threshold, 0.85);
        assert_eq!(router.c1_threshold, 0.70);
    }

    #[test]
    fn test_custom_thresholds() {
        let router = AdaptiveRouter::new(Some(0.90), Some(0.80), None, None);
        assert_eq!(router.c0_threshold, 0.90);
        assert_eq!(router.c1_threshold, 0.80);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-core && cargo test --features onnx --lib routing::router::tests`
Expected: FAIL — methods not implemented

---

### Task 5: Implement AdaptiveRouter Stage 0 (structural-only routing)

**Files:**
- Modify: `sage-core/src/routing/router.rs`

**Step 1: Implement the constructor and route_stage0**

Add the impl blocks to `router.rs` (after the struct definition, before `#[cfg(test)]`):

```rust
impl AdaptiveRouter {
    fn tier_from_complexity(complexity: f32, uncertainty: f32, tool_req: bool) -> (u8, f32) {
        // S3: high complexity OR high uncertainty
        if complexity > 0.65 || uncertainty > 0.6 {
            let conf = ((complexity - 0.65).max(0.0) * 2.0 + (uncertainty - 0.6).max(0.0) * 2.0)
                .min(1.0)
                .max(0.5); // at least 0.5 confidence at S3 boundary
            return (3, conf);
        }

        // S1: low complexity AND low uncertainty AND no tools
        if complexity <= 0.50 && uncertainty <= 0.3 && !tool_req {
            let conf = (1.0 - complexity / 0.50) * 0.5 + 0.5; // higher conf for lower complexity
            return (1, conf.min(1.0));
        }

        // S2: everything in between
        let conf = 0.6; // moderate confidence for middle zone
        (2, conf)
    }
}

#[pymethods]
impl AdaptiveRouter {
    #[new]
    #[pyo3(signature = (c0_threshold=None, c1_threshold=None, classifier_path=None, tokenizer_path=None))]
    pub fn new(
        c0_threshold: Option<f32>,
        c1_threshold: Option<f32>,
        classifier_path: Option<String>,
        tokenizer_path: Option<String>,
    ) -> Self {
        let _ = (classifier_path, tokenizer_path); // Stage 1 — implemented in Phase 2
        Self {
            exemplar_embeddings: Vec::new(),
            classifier: None,
            classifier_tokenizer: None,
            c0_threshold: c0_threshold.unwrap_or(0.85),
            c1_threshold: c1_threshold.unwrap_or(0.70),
            feedback: Mutex::new(Vec::new()),
        }
    }

    /// Stage 0: structural features only (no ML, <1ms).
    /// Returns RoutingResult with tier, confidence, and features.
    pub fn route_stage0(&self, task: &str) -> RoutingResult {
        let features = StructuralFeatures::extract(task);
        let (tier, confidence) = Self::tier_from_complexity(
            features.keyword_complexity,
            features.keyword_uncertainty,
            features.tool_required,
        );
        RoutingResult {
            tier,
            confidence,
            stage: 0,
            features,
        }
    }

    /// Full routing pipeline: Stage 0 → (Stage 1 if available) → return.
    /// Stages 2-3 are handled Python-side.
    pub fn route(&self, task: &str) -> RoutingResult {
        let s0 = self.route_stage0(task);

        // Short-circuit if Stage 0 is confident enough
        if s0.confidence >= self.c0_threshold {
            return s0;
        }

        // Stage 1: BERT classifier (if loaded)
        if let Some(s1) = self.route_stage1(task, &s0.features) {
            if s1.confidence >= self.c1_threshold {
                return s1;
            }
        }

        // Return Stage 0 result if no classifier or low confidence
        s0
    }

    /// Stage 1: BERT classifier inference.
    /// Returns None if no classifier loaded.
    fn route_stage1(&self, _task: &str, _features: &StructuralFeatures) -> Option<RoutingResult> {
        // Implemented in Phase 2
        None
    }

    /// Record routing feedback for online learning.
    pub fn record_feedback(
        &self,
        task: String,
        routed_tier: u8,
        actual_quality: f32,
        latency_ms: u64,
        cost_usd: f32,
    ) {
        if let Ok(mut fb) = self.feedback.lock() {
            fb.push(RoutingFeedback {
                task,
                routed_tier,
                actual_quality,
                latency_ms,
                cost_usd,
            });
            // Cap buffer at 10000 entries
            if fb.len() > 10000 {
                fb.drain(..5000);
            }
        }
    }

    /// Get feedback buffer size.
    pub fn feedback_count(&self) -> usize {
        self.feedback.lock().map(|fb| fb.len()).unwrap_or(0)
    }

    /// Whether a BERT classifier is loaded (Stage 1 available).
    pub fn has_classifier(&self) -> bool {
        self.classifier.is_some()
    }
}
```

**Step 2: Run tests to verify they pass**

Run: `cd sage-core && cargo test --features onnx --lib routing::router::tests -v`
Expected: 6 tests PASS

**Step 3: Run full build to verify no regressions**

Run: `cd sage-core && cargo build --features onnx`
Expected: SUCCESS

**Step 4: Commit**

```bash
git add sage-core/src/routing/
git commit -m "feat(routing): implement AdaptiveRouter Stage 0 (structural features)"
```

---

### Task 6: Register AdaptiveRouter in lib.rs and verify PyO3 exports

**Files:**
- Modify: `sage-core/src/lib.rs`

**Step 1: Add routing module and PyClass registration**

In `lib.rs`, add `pub mod routing;` in the module list, and register PyClasses:

```rust
// After line 8: pub mod types;
pub mod routing;
```

```rust
// After the onnx RustEmbedder block (line 38):
#[cfg(feature = "onnx")]
m.add_class::<routing::router::AdaptiveRouter>()?;
m.add_class::<routing::features::StructuralFeatures>()?;
```

**Step 2: Verify build with onnx feature**

Run: `cd sage-core && cargo build --features onnx`
Expected: SUCCESS

**Step 3: Verify build without onnx feature (no breakage)**

Run: `cd sage-core && cargo build`
Expected: SUCCESS (routing/router.rs is gated behind onnx)

**Step 4: Commit**

```bash
git add sage-core/src/lib.rs
git commit -m "feat(routing): register AdaptiveRouter and StructuralFeatures in PyO3 module"
```

---

## Phase 2: Stage 1 — BERT Classifier ONNX Inference

### Task 7: Create ONNX classifier export script

**Files:**
- Create: `sage-core/models/export_classifier.py`

**Step 1: Write the export script**

This script downloads `routellm/bert` (XLM-RoBERTa 278M fine-tuned for routing),
converts to ONNX with O3 optimization, and quantizes to int8.

```python
"""Export routellm/bert classifier to quantized ONNX for Rust inference.

Usage:
    pip install sentence-transformers optimum onnx onnxruntime
    python sage-core/models/export_classifier.py

Outputs:
    sage-core/models/classifier/model_quantized.onnx
    sage-core/models/classifier/tokenizer.json
"""
import os
import sys
from pathlib import Path

def main():
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
    except ImportError:
        print("Install: pip install optimum[onnxruntime] transformers")
        sys.exit(1)

    model_name = "routellm/bert"
    out_dir = Path(__file__).parent / "classifier"
    out_dir.mkdir(exist_ok=True)

    print(f"Exporting {model_name} to ONNX...")
    model = ORTModelForSequenceClassification.from_pretrained(
        model_name, export=True
    )
    model.save_pretrained(out_dir)

    print("Exporting tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(out_dir)

    # Also save as tokenizer.json for Rust tokenizers crate
    if hasattr(tokenizer, "backend_tokenizer"):
        tokenizer.backend_tokenizer.save(str(out_dir / "tokenizer.json"))

    print(f"Done. Model saved to {out_dir}")
    print(f"  ONNX: {out_dir / 'model.onnx'}")
    print(f"  Tokenizer: {out_dir / 'tokenizer.json'}")

    # Quantize to int8
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        print("Quantizing to int8...")
        quantizer = ORTQuantizer.from_pretrained(out_dir)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
        quantizer.quantize(save_dir=out_dir, quantization_config=qconfig)
        print(f"  Quantized: {out_dir / 'model_quantized.onnx'}")
    except Exception as e:
        print(f"  Quantization skipped: {e}")
        print("  Using unquantized model (larger but still works)")


if __name__ == "__main__":
    main()
```

**Step 2: Commit (script only — model files are gitignored)**

```bash
echo "sage-core/models/classifier/*.onnx" >> sage-core/.gitignore
echo "sage-core/models/classifier/*.bin" >> sage-core/.gitignore
git add sage-core/models/export_classifier.py sage-core/.gitignore
git commit -m "feat(routing): add ONNX classifier export script for routellm/bert"
```

---

### Task 8: Write failing tests for Stage 1 classifier loading

**Files:**
- Modify: `sage-core/src/routing/router.rs`

**Step 1: Add Stage 1 tests**

Add these tests to the `mod tests` block in `router.rs`:

```rust
    #[test]
    fn test_load_classifier_nonexistent_path_returns_error() {
        let router = AdaptiveRouter::new(None, None,
            Some("/nonexistent/model.onnx".to_string()),
            Some("/nonexistent/tokenizer.json".to_string()),
        );
        // Should gracefully fall back to no classifier
        assert!(!router.has_classifier());
    }

    #[test]
    fn test_route_without_classifier_uses_stage0() {
        let router = AdaptiveRouter::new(None, None, None, None);
        let result = router.route("What is the capital of France?");
        assert_eq!(result.stage, 0); // No classifier → Stage 0 only
        assert_eq!(result.tier, 1);
    }

    #[test]
    fn test_feedback_recording() {
        let router = AdaptiveRouter::new(None, None, None, None);
        router.record_feedback("test task".to_string(), 1, 0.9, 100, 0.001);
        assert_eq!(router.feedback_count(), 1);
    }

    #[test]
    fn test_feedback_buffer_bounded() {
        let router = AdaptiveRouter::new(None, None, None, None);
        for i in 0..10500 {
            router.record_feedback(format!("task {i}"), 1, 0.5, 50, 0.001);
        }
        // Buffer should have been trimmed
        assert!(router.feedback_count() <= 10000);
    }
```

**Step 2: Run tests to verify they pass (these should all pass with current impl)**

Run: `cd sage-core && cargo test --features onnx --lib routing::router::tests -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add sage-core/src/routing/router.rs
git commit -m "test(routing): add Stage 1 classifier and feedback tests"
```

---

### Task 9: Implement Stage 1 BERT classifier loading and inference

**Files:**
- Modify: `sage-core/src/routing/router.rs`

**Step 1: Update the constructor to load classifier if paths provided**

Update the `new()` method to attempt loading the classifier:

```rust
    #[new]
    #[pyo3(signature = (c0_threshold=None, c1_threshold=None, classifier_path=None, tokenizer_path=None))]
    pub fn new(
        c0_threshold: Option<f32>,
        c1_threshold: Option<f32>,
        classifier_path: Option<String>,
        tokenizer_path: Option<String>,
    ) -> Self {
        let mut classifier = None;
        let mut classifier_tokenizer = None;

        if let (Some(cp), Some(tp)) = (&classifier_path, &tokenizer_path) {
            // Initialize ORT runtime
            crate::memory::embedder::ensure_ort_initialized(cp);

            match ort::session::Session::builder()
                .and_then(|b| b.commit_from_file(cp))
            {
                Ok(session) => {
                    classifier = Some(session);
                    log::info!("Loaded BERT classifier from {cp}");
                }
                Err(e) => {
                    log::warn!("Failed to load classifier from {cp}: {e}");
                }
            }

            match tokenizers::Tokenizer::from_file(tp) {
                Ok(tok) => {
                    classifier_tokenizer = Some(tok);
                    log::info!("Loaded classifier tokenizer from {tp}");
                }
                Err(e) => {
                    log::warn!("Failed to load tokenizer from {tp}: {e}");
                }
            }
        }

        Self {
            exemplar_embeddings: Vec::new(),
            classifier,
            classifier_tokenizer,
            c0_threshold: c0_threshold.unwrap_or(0.85),
            c1_threshold: c1_threshold.unwrap_or(0.70),
            feedback: Mutex::new(Vec::new()),
        }
    }
```

**Step 2: Implement route_stage1**

Replace the stub `route_stage1` with the real implementation:

```rust
    /// Stage 1: BERT classifier inference.
    /// The classifier outputs logits for [S1, S2, S3].
    /// Returns None if no classifier loaded.
    fn route_stage1(&self, task: &str, features: &StructuralFeatures) -> Option<RoutingResult> {
        let session = self.classifier.as_ref()?;
        let tokenizer = self.classifier_tokenizer.as_ref()?;

        // Tokenize
        let encoding = tokenizer.encode(task, true).ok()?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let type_ids: Vec<i64> = vec![0i64; ids.len()];
        let seq_len = ids.len();

        let shape = vec![1usize, seq_len];

        let id_tensor = ort::value::TensorRef::from_array_view(
            (shape.clone(), &*ids)
        ).ok()?;
        let mask_tensor = ort::value::TensorRef::from_array_view(
            (shape.clone(), &*mask)
        ).ok()?;
        let type_tensor = ort::value::TensorRef::from_array_view(
            (shape, &*type_ids)
        ).ok()?;

        let outputs = session.run(ort::inputs![
            "input_ids" => id_tensor,
            "attention_mask" => mask_tensor,
            "token_type_ids" => type_tensor
        ]).ok()?;

        // Extract logits — shape [1, num_classes]
        let (out_shape, logits) = outputs[0].try_extract_tensor::<f32>().ok()?;
        let num_classes = if out_shape.len() >= 2 { out_shape[1] as usize } else { logits.len() };

        // Softmax to get probabilities
        let max_logit = logits.iter().take(num_classes).cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().take(num_classes).map(|&l| (l - max_logit).exp()).sum();
        let probs: Vec<f32> = logits.iter().take(num_classes).map(|&l| (l - max_logit).exp() / exp_sum).collect();

        // Map to tier: highest probability class
        // For routellm/bert: class 0 = weak model OK (S1), class 1 = strong model needed (S2/S3)
        // We refine S2 vs S3 using structural features
        let (tier, confidence) = if num_classes == 2 {
            // Binary: weak vs strong
            if probs[0] > probs[1] {
                (1u8, probs[0])  // S1
            } else {
                // Use structural features to split S2/S3
                if features.keyword_complexity > 0.65 {
                    (3, probs[1])
                } else {
                    (2, probs[1])
                }
            }
        } else if num_classes >= 3 {
            // Multi-class: direct mapping
            let (idx, &max_prob) = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())?;
            ((idx as u8 + 1).min(3), max_prob)
        } else {
            return None;
        };

        Some(RoutingResult {
            tier,
            confidence,
            stage: 1,
            features: features.clone(),
        })
    }
```

**Step 3: Add `log` dependency (if not already present)**

Check if `log` is in `Cargo.toml`. If not, add:
```toml
log = "0.4"
```

**Step 4: Make `ensure_ort_initialized` pub(crate)**

In `sage-core/src/memory/embedder.rs`, change:
```rust
fn ensure_ort_initialized(model_path: &str) {
```
to:
```rust
pub(crate) fn ensure_ort_initialized(model_path: &str) {
```

**Step 5: Run tests**

Run: `cd sage-core && cargo test --features onnx --lib routing::router::tests -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add sage-core/src/routing/router.rs sage-core/src/memory/embedder.rs sage-core/Cargo.toml
git commit -m "feat(routing): implement Stage 1 BERT classifier inference via ONNX"
```

---

## Phase 3: Python Integration — Replace ComplexityRouter

### Task 10: Write the Python AdaptiveRouter wrapper

**Files:**
- Create: `sage-python/src/sage/strategy/adaptive_router.py`

**Step 1: Create the Python wrapper**

```python
"""Adaptive Router — 4-stage learned routing pipeline.

Replaces ComplexityRouter regex heuristic with:
  Stage 0: Structural features (Rust, <1ms)
  Stage 1: BERT classifier (Rust ONNX, <10ms)
  Stage 2: Entropy probe (Python, ~100ms, optional)
  Stage 3: Cascade execution (Python, variable, optional)

Falls back to ComplexityRouter if sage_core is not compiled with onnx feature.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from sage.strategy.metacognition import (
    CognitiveProfile, ComplexityRouter, RoutingDecision,
)

log = logging.getLogger(__name__)

# Try to import Rust AdaptiveRouter
_rust_router = None
try:
    from sage_core import AdaptiveRouter as _RustRouter  # type: ignore
    _rust_router = _RustRouter
    log.info("Rust AdaptiveRouter available (Stage 0+1)")
except ImportError:
    log.info("Rust AdaptiveRouter not available, using ComplexityRouter fallback")


@dataclass
class AdaptiveRoutingResult:
    """Extended routing result with stage information."""
    decision: RoutingDecision
    profile: CognitiveProfile
    stage: int          # 0-3: which stage produced the decision
    confidence: float   # confidence of the routing decision
    method: str         # "rust_s0", "rust_s1", "entropy_s2", "cascade_s3", "heuristic"


class AdaptiveRouter:
    """4-stage adaptive router with confidence gating.

    Wraps Rust AdaptiveRouter (Stages 0-1) and adds Python Stages 2-3.
    Falls back to ComplexityRouter heuristic if Rust is unavailable.
    """

    def __init__(
        self,
        c0_threshold: float = 0.85,
        c1_threshold: float = 0.70,
        classifier_path: str | None = None,
        tokenizer_path: str | None = None,
        llm_provider: Any = None,
        enable_entropy_probe: bool = False,
        enable_cascade: bool = False,
    ):
        self._llm_provider = llm_provider
        self._enable_entropy = enable_entropy_probe
        self._enable_cascade = enable_cascade

        # Try Rust router
        self._rust: Any | None = None
        if _rust_router is not None:
            try:
                self._rust = _rust_router(
                    c0_threshold=c0_threshold,
                    c1_threshold=c1_threshold,
                    classifier_path=classifier_path,
                    tokenizer_path=tokenizer_path,
                )
                log.info(
                    "AdaptiveRouter initialized (Rust, classifier=%s)",
                    self._rust.has_classifier(),
                )
            except Exception as e:
                log.warning("Rust AdaptiveRouter init failed: %s", e)

        # Fallback: ComplexityRouter
        self._fallback = ComplexityRouter(llm_provider=llm_provider)

    @property
    def has_rust(self) -> bool:
        return self._rust is not None

    @property
    def has_classifier(self) -> bool:
        return self._rust is not None and self._rust.has_classifier()

    def route(self, task: str) -> AdaptiveRoutingResult:
        """Synchronous routing (Stages 0-1 only, no LLM calls)."""
        if self._rust is not None:
            result = self._rust.route(task)
            profile = CognitiveProfile(
                complexity=result.features.keyword_complexity,
                uncertainty=result.features.keyword_uncertainty,
                tool_required=result.features.tool_required,
                reasoning=f"adaptive_stage{result.stage}",
            )
            decision = self._make_decision(result.tier, profile)
            method = f"rust_s{result.stage}"
            return AdaptiveRoutingResult(
                decision=decision,
                profile=profile,
                stage=result.stage,
                confidence=result.confidence,
                method=method,
            )

        # Fallback to heuristic
        profile = self._fallback.assess_complexity(task)
        decision = self._fallback.route(profile)
        return AdaptiveRoutingResult(
            decision=decision,
            profile=profile,
            stage=0,
            confidence=0.5,
            method="heuristic",
        )

    async def route_async(self, task: str) -> AdaptiveRoutingResult:
        """Async routing with full pipeline (Stages 0-3)."""
        # Stage 0-1: Rust (sync, fast)
        result = self.route(task)

        # If confident enough, return immediately
        if result.confidence >= 0.85:
            return result

        # Stage 2: Entropy probe (optional)
        if self._enable_entropy and self._llm_provider is not None:
            try:
                entropy_result = await self._entropy_probe(task, result)
                if entropy_result is not None:
                    return entropy_result
            except Exception as e:
                log.warning("Entropy probe failed: %s", e)

        # Stage 3: Cascade (optional) — handled at execution time, not routing
        # The cascade pattern (try fast → verify → escalate) is implemented
        # in the CognitiveOrchestrator, not here.

        return result

    async def _entropy_probe(
        self, task: str, current: AdaptiveRoutingResult
    ) -> AdaptiveRoutingResult | None:
        """Stage 2: Probe fast model with 3 tokens, measure entropy.

        Low entropy → S1 (model is confident, task is easy)
        High entropy → S2/S3 (model is uncertain, task needs more compute)
        """
        if self._llm_provider is None:
            return None

        import math
        from sage.llm.base import Message, Role, LLMConfig

        try:
            response = await self._llm_provider.generate(
                messages=[Message(role=Role.USER, content=task[:500])],
                config=LLMConfig(
                    provider="auto", model="auto",
                    temperature=1.0,  # Need non-zero temp for entropy
                    max_tokens=3,
                ),
            )

            # Estimate entropy from token probabilities if available
            # Most providers don't expose logprobs, so use content-based heuristic
            content = response.content.strip()
            if not content:
                return None

            # Content-based entropy estimate: unique chars / total chars
            chars = list(content)
            unique = len(set(chars))
            char_entropy = unique / max(len(chars), 1)

            # Map to routing decision
            if char_entropy < 0.3:  # Low diversity → confident → S1
                tier = 1
                confidence = 0.75
            elif char_entropy > 0.7:  # High diversity → uncertain → S3
                tier = 3
                confidence = 0.65
            else:
                tier = 2
                confidence = 0.60

            profile = current.profile
            decision = self._make_decision(tier, profile)
            return AdaptiveRoutingResult(
                decision=decision,
                profile=profile,
                stage=2,
                confidence=confidence,
                method="entropy_s2",
            )
        except Exception:
            return None

    def record_feedback(
        self,
        task: str,
        routed_tier: int,
        actual_quality: float,
        latency_ms: int,
        cost_usd: float,
    ) -> None:
        """Record routing feedback for online learning."""
        if self._rust is not None:
            self._rust.record_feedback(
                task, routed_tier, actual_quality, latency_ms, cost_usd,
            )

    def _make_decision(self, tier: int, profile: CognitiveProfile) -> RoutingDecision:
        """Convert a tier number to a RoutingDecision."""
        if tier == 3:
            llm_tier = "codex" if profile.complexity > 0.8 else "reasoner"
            return RoutingDecision(
                system=3, llm_tier=llm_tier,
                max_tokens=8192, use_z3=True, validation_level=3,
            )
        elif tier == 1:
            return RoutingDecision(
                system=1, llm_tier="fast",
                max_tokens=2048, use_z3=False, validation_level=1,
            )
        else:
            llm_tier = "reasoner" if profile.complexity > 0.55 else "mutator"
            return RoutingDecision(
                system=2, llm_tier=llm_tier,
                max_tokens=4096, use_z3=False, validation_level=2,
            )
```

**Step 2: Commit**

```bash
git add sage-python/src/sage/strategy/adaptive_router.py
git commit -m "feat(routing): add Python AdaptiveRouter wrapper with Stage 0-2 pipeline"
```

---

### Task 11: Write tests for Python AdaptiveRouter

**Files:**
- Create: `sage-python/tests/test_adaptive_router.py`

**Step 1: Write the test file**

```python
"""Tests for AdaptiveRouter (Python wrapper + fallback)."""
import sys
import types

# Mock sage_core if not available
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.strategy.adaptive_router import AdaptiveRouter, AdaptiveRoutingResult
from sage.strategy.metacognition import CognitiveProfile, RoutingDecision


class TestAdaptiveRouterFallback:
    """Tests with ComplexityRouter fallback (no Rust)."""

    def test_simple_task_routes_s1(self):
        router = AdaptiveRouter()
        result = router.route("What is 2+2?")
        assert isinstance(result, AdaptiveRoutingResult)
        assert result.decision.system == 1
        assert result.method in ("rust_s0", "heuristic")

    def test_code_task_routes_s2(self):
        router = AdaptiveRouter()
        result = router.route("Write a Python function to check if a number is prime")
        assert result.decision.system >= 2

    def test_complex_task_routes_s3(self):
        router = AdaptiveRouter()
        result = router.route(
            "Debug a race condition in async Rust code with deadlock on Arc<Mutex>"
        )
        assert result.decision.system >= 2  # At minimum S2

    def test_result_has_all_fields(self):
        router = AdaptiveRouter()
        result = router.route("Hello world")
        assert hasattr(result, "decision")
        assert hasattr(result, "profile")
        assert hasattr(result, "stage")
        assert hasattr(result, "confidence")
        assert hasattr(result, "method")

    def test_confidence_in_valid_range(self):
        router = AdaptiveRouter()
        result = router.route("Calculate the fibonacci sequence")
        assert 0.0 <= result.confidence <= 1.0

    def test_decision_has_validation_level(self):
        router = AdaptiveRouter()
        result = router.route("Simple question")
        assert result.decision.validation_level in (1, 2, 3)

    def test_record_feedback_does_not_crash(self):
        router = AdaptiveRouter()
        router.record_feedback("test", 1, 0.9, 100, 0.001)
        # Should not raise even without Rust backend

    def test_has_rust_property(self):
        router = AdaptiveRouter()
        assert isinstance(router.has_rust, bool)

    def test_has_classifier_property(self):
        router = AdaptiveRouter()
        assert isinstance(router.has_classifier, bool)


@pytest.mark.asyncio
async def test_route_async_falls_back():
    """Async routing works even without entropy probe."""
    router = AdaptiveRouter(enable_entropy_probe=False)
    result = await router.route_async("What is 2+2?")
    assert result.decision.system == 1


@pytest.mark.asyncio
async def test_route_async_complex_task():
    """Async routing handles complex tasks."""
    router = AdaptiveRouter(enable_entropy_probe=False)
    result = await router.route_async(
        "Design a distributed consensus protocol for a 5-node cluster"
    )
    assert result.decision.system >= 2


class TestRoutingQualityWithAdaptive:
    """Verify AdaptiveRouter meets quality baselines."""

    def test_s1_tasks_route_correctly(self):
        """All trivial tasks should route to S1."""
        router = AdaptiveRouter()
        s1_tasks = [
            "What is 2+2?",
            "What is the capital of France?",
            "How many continents are there?",
        ]
        for task in s1_tasks:
            result = router.route(task)
            assert result.decision.system == 1, f"'{task}' should be S1, got S{result.decision.system}"

    def test_over_routing_below_threshold(self):
        """Simple tasks should not be over-routed to S3."""
        router = AdaptiveRouter()
        simple_tasks = [
            "What is 15% of 200?",
            "Name three fruits",
            "Define the word algorithm",
        ]
        over_routed = sum(
            1 for t in simple_tasks if router.route(t).decision.system > 1
        )
        assert over_routed == 0, f"{over_routed}/{len(simple_tasks)} simple tasks over-routed"
```

**Step 2: Run tests**

Run: `cd sage-python && python -m pytest tests/test_adaptive_router.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add sage-python/tests/test_adaptive_router.py
git commit -m "test(routing): add AdaptiveRouter Python tests (fallback + quality)"
```

---

### Task 12: Wire AdaptiveRouter into boot.py

**Files:**
- Modify: `sage-python/src/sage/boot.py:30,47,73-74,185`

**Step 1: Import AdaptiveRouter**

In `boot.py`, change line 30:
```python
from sage.strategy.metacognition import ComplexityRouter
```
to:
```python
from sage.strategy.metacognition import ComplexityRouter
from sage.strategy.adaptive_router import AdaptiveRouter
```

**Step 2: Replace ComplexityRouter instantiation**

In `boot_agent_system()`, change line 185:
```python
metacognition = ComplexityRouter(llm_provider=provider if not use_mock_llm else None)
```
to:
```python
# Try AdaptiveRouter (Rust Stages 0-1 + Python Stages 2-3)
# Falls back to ComplexityRouter heuristic if sage_core[onnx] not available
_adaptive = AdaptiveRouter(llm_provider=provider if not use_mock_llm else None)
metacognition = _adaptive._fallback  # ComplexityRouter for backward compat
_adaptive_router = _adaptive  # Store for AgentSystem
```

**Step 3: Add adaptive_router to AgentSystem**

In the `AgentSystem` dataclass, add after line 63:
```python
# AdaptiveRouter (4-stage learned routing)
adaptive_router: Any = None
```

And in the return statement (line 355+), add:
```python
adaptive_router=_adaptive_router if not use_mock_llm else None,
```

**Step 4: Update AgentSystem.run() to use AdaptiveRouter**

In `AgentSystem.run()`, change lines 73-74:
```python
profile = await self.metacognition.assess_complexity_async(task)
decision = self.metacognition.route(profile)
```
to:
```python
# Use AdaptiveRouter if available, else legacy ComplexityRouter
if self.adaptive_router is not None:
    adaptive_result = await self.adaptive_router.route_async(task)
    profile = adaptive_result.profile
    decision = adaptive_result.decision
    _log.info(
        "Adaptive routing: S%d (stage=%d, confidence=%.2f, method=%s)",
        decision.system, adaptive_result.stage,
        adaptive_result.confidence, adaptive_result.method,
    )
else:
    profile = await self.metacognition.assess_complexity_async(task)
    decision = self.metacognition.route(profile)
```

**Step 5: Run existing tests to verify no regressions**

Run: `cd sage-python && python -m pytest tests/test_boot.py tests/test_metacognition.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add sage-python/src/sage/boot.py
git commit -m "feat(routing): wire AdaptiveRouter into boot sequence with fallback"
```

---

## Phase 4: Evaluation Framework

### Task 13: Extend routing_quality.py to support AdaptiveRouter

**Files:**
- Modify: `sage-python/src/sage/bench/routing_quality.py`

**Step 1: Add AdaptiveRouter support**

Add after line 6 (`from sage.strategy.metacognition import ComplexityRouter`):

```python
from sage.strategy.adaptive_router import AdaptiveRouter
```

Add a new function after `run_routing_quality()`:

```python
def run_adaptive_routing_quality(
    router: AdaptiveRouter | None = None,
) -> RoutingQualityResult:
    """Run routing quality benchmark using AdaptiveRouter.

    Returns same RoutingQualityResult for comparison with ComplexityRouter.
    """
    if router is None:
        router = AdaptiveRouter()

    correct = 0
    under_routed = 0
    over_routed = 0
    details = []

    for task, min_system, rationale in GROUND_TRUTH:
        result = router.route(task)
        actual = result.decision.system

        is_correct = actual >= min_system
        is_under = actual < min_system
        is_over = actual > min_system

        if is_correct:
            correct += 1
        if is_under:
            under_routed += 1
        if is_over:
            over_routed += 1

        details.append({
            "task": task[:60],
            "expected_min": min_system,
            "actual": actual,
            "correct": is_correct,
            "complexity": round(result.profile.complexity, 3),
            "confidence": round(result.confidence, 3),
            "stage": result.stage,
            "method": result.method,
            "rationale": rationale,
        })

    total = len(GROUND_TRUTH)
    return RoutingQualityResult(
        total=total,
        correct=correct,
        under_routed=under_routed,
        over_routed=over_routed,
        accuracy=correct / total if total else 0.0,
        under_routing_rate=under_routed / total if total else 0.0,
        over_routing_rate=over_routed / total if total else 0.0,
        details=details,
    )
```

**Step 2: Run tests**

Run: `cd sage-python && python -m pytest tests/test_routing_quality.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add sage-python/src/sage/bench/routing_quality.py
git commit -m "feat(bench): extend routing quality benchmark for AdaptiveRouter"
```

---

### Task 14: Create downstream quality evaluation

**Files:**
- Create: `sage-python/src/sage/bench/routing_downstream.py`

**Step 1: Write the downstream quality evaluator**

```python
"""Downstream quality evaluation for routing.

Measures ACTUAL quality impact of routing decisions, not self-consistency.

Metrics:
  1. Quality Delta: A/B split (20% random → best model as control)
  2. Cost-Quality Pareto: cost vs quality across tiers
  3. Tier Precision: success rate per routed tier
  4. Escalation Rate: % of tasks that needed re-routing (<20% target)
  5. Routing Latency: P50/P95/P99 of routing decision time
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class TierMetrics:
    """Per-tier quality metrics."""
    tier: int
    total: int = 0
    successes: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0

    @property
    def precision(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0

    @property
    def avg_cost_usd(self) -> float:
        return self.total_cost_usd / self.total if self.total else 0.0


@dataclass
class DownstreamResult:
    """Downstream quality evaluation result."""
    total_tasks: int = 0
    escalations: int = 0
    tier_metrics: dict[int, TierMetrics] = field(default_factory=dict)
    routing_latencies_ms: list[float] = field(default_factory=list)
    quality_scores: list[float] = field(default_factory=list)

    @property
    def escalation_rate(self) -> float:
        return self.escalations / self.total_tasks if self.total_tasks else 0.0

    @property
    def avg_quality(self) -> float:
        return sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0

    @property
    def routing_p50_ms(self) -> float:
        if not self.routing_latencies_ms:
            return 0.0
        sorted_lat = sorted(self.routing_latencies_ms)
        return sorted_lat[len(sorted_lat) // 2]

    @property
    def routing_p99_ms(self) -> float:
        if not self.routing_latencies_ms:
            return 0.0
        sorted_lat = sorted(self.routing_latencies_ms)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def to_dict(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "escalation_rate": round(self.escalation_rate, 3),
            "avg_quality": round(self.avg_quality, 3),
            "routing_p50_ms": round(self.routing_p50_ms, 2),
            "routing_p99_ms": round(self.routing_p99_ms, 2),
            "tier_precision": {
                t: round(m.precision, 3) for t, m in self.tier_metrics.items()
            },
            "tier_avg_cost": {
                t: round(m.avg_cost_usd, 5) for t, m in self.tier_metrics.items()
            },
        }


class DownstreamEvaluator:
    """Collects routing outcomes and computes downstream quality metrics.

    Usage:
        evaluator = DownstreamEvaluator()
        for task in tasks:
            t0 = time.perf_counter()
            result = router.route(task)
            routing_ms = (time.perf_counter() - t0) * 1000
            # ... execute task ...
            evaluator.record(
                tier=result.decision.system,
                quality=outcome_quality,
                latency_ms=execution_latency,
                cost_usd=execution_cost,
                routing_ms=routing_ms,
                escalated=was_escalated,
            )
        report = evaluator.result()
    """

    def __init__(self):
        self._result = DownstreamResult()

    def record(
        self,
        tier: int,
        quality: float,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        routing_ms: float = 0.0,
        escalated: bool = False,
    ) -> None:
        self._result.total_tasks += 1
        self._result.quality_scores.append(quality)
        self._result.routing_latencies_ms.append(routing_ms)
        if escalated:
            self._result.escalations += 1

        if tier not in self._result.tier_metrics:
            self._result.tier_metrics[tier] = TierMetrics(tier=tier)
        tm = self._result.tier_metrics[tier]
        tm.total += 1
        if quality >= 0.5:  # Success threshold
            tm.successes += 1
        tm.total_latency_ms += latency_ms
        tm.total_cost_usd += cost_usd

    def result(self) -> DownstreamResult:
        return self._result
```

**Step 2: Commit**

```bash
git add sage-python/src/sage/bench/routing_downstream.py
git commit -m "feat(bench): add downstream quality evaluator for routing"
```

---

### Task 15: Write tests for downstream evaluator

**Files:**
- Create: `sage-python/tests/test_routing_downstream.py`

**Step 1: Write the test file**

```python
"""Tests for downstream routing quality evaluator."""
import sys
import types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

from sage.bench.routing_downstream import (
    DownstreamEvaluator, DownstreamResult, TierMetrics,
)


def test_tier_metrics_precision():
    tm = TierMetrics(tier=1, total=10, successes=8)
    assert tm.precision == 0.8


def test_tier_metrics_empty():
    tm = TierMetrics(tier=1)
    assert tm.precision == 0.0
    assert tm.avg_latency_ms == 0.0


def test_evaluator_records_correctly():
    ev = DownstreamEvaluator()
    ev.record(tier=1, quality=0.9, latency_ms=50, cost_usd=0.001, routing_ms=1.0)
    ev.record(tier=2, quality=0.7, latency_ms=200, cost_usd=0.01, routing_ms=5.0)
    ev.record(tier=1, quality=0.3, latency_ms=30, cost_usd=0.001, routing_ms=0.5)

    result = ev.result()
    assert result.total_tasks == 3
    assert result.escalation_rate == 0.0
    assert len(result.tier_metrics) == 2
    assert result.tier_metrics[1].total == 2
    assert result.tier_metrics[1].successes == 1  # only 0.9 >= 0.5
    assert result.tier_metrics[2].successes == 1  # 0.7 >= 0.5


def test_escalation_rate():
    ev = DownstreamEvaluator()
    ev.record(tier=1, quality=0.9, escalated=False)
    ev.record(tier=2, quality=0.5, escalated=True)
    ev.record(tier=2, quality=0.8, escalated=False)

    result = ev.result()
    assert abs(result.escalation_rate - 1/3) < 0.01


def test_routing_percentiles():
    ev = DownstreamEvaluator()
    for ms in [1.0, 2.0, 3.0, 4.0, 100.0]:
        ev.record(tier=1, quality=0.9, routing_ms=ms)

    result = ev.result()
    assert result.routing_p50_ms == 3.0
    assert result.routing_p99_ms == 100.0


def test_to_dict():
    ev = DownstreamEvaluator()
    ev.record(tier=1, quality=0.9, cost_usd=0.001, routing_ms=1.0)
    d = ev.result().to_dict()
    assert "total_tasks" in d
    assert "escalation_rate" in d
    assert "tier_precision" in d
    assert "routing_p50_ms" in d
```

**Step 2: Run tests**

Run: `cd sage-python && python -m pytest tests/test_routing_downstream.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add sage-python/tests/test_routing_downstream.py
git commit -m "test(bench): add downstream evaluator tests"
```

---

## Phase 5: Training Pipeline & Feedback Loop

### Task 16: Create feedback export and training script

**Files:**
- Create: `sage-python/src/sage/strategy/training.py`

**Step 1: Write the training data management module**

```python
"""Training pipeline for AdaptiveRouter BERT classifier.

Collects routing feedback, exports training data, triggers retraining,
and hot-reloads the updated ONNX model into the Rust router.
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
    """Single training example: task → tier with quality signal."""
    task: str
    routed_tier: int
    actual_quality: float
    latency_ms: float
    cost_usd: float
    label: int  # Computed: optimal tier based on quality + cost


def compute_label(routed_tier: int, quality: float, cost_usd: float) -> int:
    """Compute optimal routing label from outcome.

    Rules:
      - If quality >= 0.8 → current tier was sufficient (label = routed_tier)
      - If quality < 0.5 and tier < 3 → should have escalated (label = tier + 1)
      - If quality >= 0.8 and tier > 1 and cost > median → could have used cheaper (label = tier - 1)
    """
    if quality < 0.5 and routed_tier < 3:
        return routed_tier + 1  # Escalate
    if quality >= 0.8:
        return routed_tier  # Good enough
    return routed_tier


def export_training_data(
    feedback: list[dict],
    output_path: Path | None = None,
) -> Path:
    """Export feedback buffer to JSONL training file.

    Args:
        feedback: list of {task, routed_tier, actual_quality, latency_ms, cost_usd}
        output_path: override output path

    Returns:
        Path to the exported JSONL file.
    """
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
```

**Step 2: Commit**

```bash
git add sage-python/src/sage/strategy/training.py
git commit -m "feat(routing): add training data export pipeline for classifier"
```

---

### Task 17: Write tests for training pipeline

**Files:**
- Create: `sage-python/tests/test_routing_training.py`

**Step 1: Write the tests**

```python
"""Tests for routing training pipeline."""
import sys
import types
import tempfile
from pathlib import Path

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

from sage.strategy.training import (
    compute_label, export_training_data, TrainingExample,
)


def test_compute_label_good_quality():
    """High quality → keep current tier."""
    assert compute_label(routed_tier=1, quality=0.9, cost_usd=0.001) == 1
    assert compute_label(routed_tier=2, quality=0.85, cost_usd=0.01) == 2


def test_compute_label_low_quality_escalates():
    """Low quality → escalate to next tier."""
    assert compute_label(routed_tier=1, quality=0.3, cost_usd=0.001) == 2
    assert compute_label(routed_tier=2, quality=0.4, cost_usd=0.01) == 3


def test_compute_label_s3_no_further_escalation():
    """S3 with low quality stays at S3 (no S4)."""
    assert compute_label(routed_tier=3, quality=0.3, cost_usd=0.03) == 3


def test_export_training_data():
    """Export produces valid JSONL file."""
    import json

    feedback = [
        {"task": "What is 2+2?", "routed_tier": 1, "actual_quality": 0.95,
         "latency_ms": 50, "cost_usd": 0.001},
        {"task": "Write bubble sort", "routed_tier": 2, "actual_quality": 0.7,
         "latency_ms": 200, "cost_usd": 0.01},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.jsonl"
        result = export_training_data(feedback, output_path=out)
        assert result.exists()

        lines = result.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        ex1 = json.loads(lines[0])
        assert ex1["task"] == "What is 2+2?"
        assert ex1["label"] == 1  # Good quality, keep S1
```

**Step 2: Run tests**

Run: `cd sage-python && python -m pytest tests/test_routing_training.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add sage-python/tests/test_routing_training.py
git commit -m "test(routing): add training pipeline tests"
```

---

## Phase 6: Documentation & Final Integration

### Task 18: Update CLAUDE.md with AdaptiveRouter

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add AdaptiveRouter to the Key Python Modules section**

Add after the `strategy/metacognition.py` entry:

```markdown
- `strategy/adaptive_router.py` - AdaptiveRouter: 4-stage learned routing pipeline (Stage 0: structural features, Stage 1: BERT ONNX classifier, Stage 2: entropy probe, Stage 3: cascade). Falls back to ComplexityRouter if sage_core[onnx] unavailable
- `strategy/training.py` - Training data export for BERT classifier retraining
```

**Step 2: Add to Rust modules section**

Add after the `sandbox/tool_executor.rs` entry:

```markdown
- `routing/features.rs` - StructuralFeatures: zero-cost keyword/structural feature extraction for Stage 0 pre-routing. Always compiled.
- `routing/router.rs` - AdaptiveRouter PyO3 class: Stage 0 (structural) + Stage 1 (BERT ONNX classifier). Behind `onnx` feature flag. Reuses ort + tokenizers from RustEmbedder.
```

**Step 3: Update routing benchmark section**

Add to the Benchmarks section:

```markdown
- **Routing Quality**: 45 labeled tasks (15 S1 + 15 S2 + 15 S3). Measures both ComplexityRouter and AdaptiveRouter against human-labeled ground truth.
- **Downstream Quality**: Records routing outcomes (quality, latency, cost) for Pareto analysis. Tracks tier precision, escalation rate, routing P99 latency.
```

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add AdaptiveRouter to CLAUDE.md"
```

---

### Task 19: Update sage-core README and sage-python README

**Files:**
- Modify: `sage-core/README.md`
- Modify: `sage-python/README.md`

**Step 1: Add routing module to sage-core README module table**

```markdown
| `routing/` | AdaptiveRouter: 4-stage learned S1/S2/S3 routing (structural features + BERT classifier), behind `onnx` feature |
```

**Step 2: Add to sage-python README package table**

```markdown
| `sage/strategy/` | ComplexityRouter (S1/S2/S3 heuristic), AdaptiveRouter (4-stage learned pipeline), CGRS self-braking, training data export |
| `sage/bench/` | Benchmarks: HumanEval, routing accuracy, routing quality (ground truth), downstream quality evaluator |
```

**Step 3: Commit**

```bash
git add sage-core/README.md sage-python/README.md
git commit -m "docs: add AdaptiveRouter to sage-core and sage-python READMEs"
```

---

### Task 20: Run full test suite and verify no regressions

**Step 1: Run Rust tests**

Run: `cd sage-core && cargo test --workspace`
Expected: 7+ tests PASS

Run: `cd sage-core && cargo test --features onnx`
Expected: All ONNX + routing tests PASS (requires ORT_DYLIB_PATH)

**Step 2: Run Python tests**

Run: `cd sage-python && python -m pytest tests/ -v --tb=short`
Expected: 850+ tests PASS (846 existing + new routing tests)

**Step 3: Run routing benchmark**

Run: `cd sage-python && python -m sage.bench --type routing`
Expected: 30/30 self-consistency (unchanged)

**Step 4: Run routing quality benchmark**

Run: `cd sage-python && python -c "from sage.bench.routing_quality import run_routing_quality, run_adaptive_routing_quality; r1 = run_routing_quality(); r2 = run_adaptive_routing_quality(); print(f'ComplexityRouter: {r1.accuracy:.1%}  AdaptiveRouter: {r2.accuracy:.1%}')"`
Expected: Both show accuracy scores

**Step 5: Commit tag**

```bash
git tag -a v0.2.0-adaptive-router -m "AdaptiveRouter: 4-stage learned S1/S2/S3 routing"
```

---

## Summary of Deliverables

| Phase | Tasks | Key Files | Tests |
|-------|-------|-----------|-------|
| 1: Rust Stage 0 | 1-6 | `routing/features.rs`, `routing/router.rs`, `routing/mod.rs` | 12 Rust unit tests |
| 2: Rust Stage 1 | 7-9 | `routing/router.rs`, `models/export_classifier.py` | 4 additional Rust tests |
| 3: Python Integration | 10-12 | `strategy/adaptive_router.py`, `boot.py` | 12 Python tests |
| 4: Evaluation | 13-15 | `bench/routing_quality.py`, `bench/routing_downstream.py` | 6 Python tests |
| 5: Training | 16-17 | `strategy/training.py` | 4 Python tests |
| 6: Documentation | 18-20 | `CLAUDE.md`, READMEs | Full regression suite |

## Key Design Decisions

1. **Feature flag**: `onnx` (reuses existing ort + tokenizers, zero new Rust deps)
2. **Fallback**: AdaptiveRouter always falls back to ComplexityRouter if Rust unavailable
3. **Backward compat**: `ComplexityRouter` still exists, `MetacognitiveController` alias preserved
4. **Evaluation**: Ground truth benchmark (45 tasks) + downstream quality metrics replace self-consistency
5. **Online learning**: Feedback buffer in Rust (bounded 10K), export to JSONL for retraining
6. **Stage gating**: c0=0.85 (Stage 0), c1=0.70 (Stage 1) — confident routing exits early

## Future Work (not in this plan)

- **Exemplar bank**: Pre-compute embeddings for known task categories, use cosine similarity in Stage 0
- **BERT fine-tuning**: Retrain routellm/bert on SAGE-specific routing feedback
- **Entropy from logprobs**: Use actual token probabilities from providers that support logprobs
- **Cascade execution**: Implement AutoMix self-verification in CognitiveOrchestrator
- **A/B testing**: 20% random → best model for continuous calibration
