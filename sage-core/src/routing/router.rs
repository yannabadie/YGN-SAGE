//! AdaptiveRouter — 4-stage learned S1/S2/S3 routing.
//!
//! Stage 0: Structural features only (keyword complexity + uncertainty).
//! Stage 1: ONNX BERT classifier (routellm/bert) on task text.
//! Stage 2-3: Python-side dynamic routing with feedback.
//!
//! Behind the `onnx` feature flag (shares deps with RustEmbedder).

use ort::value::TensorRef;
use pyo3::prelude::*;
use std::sync::Mutex;

use crate::routing::features::StructuralFeatures;

// ── RoutingResult ────────────────────────────────────────────────────────────

/// Result of a routing stage.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RoutingResult {
    /// Tier: 1=S1 (fast), 2=S2 (reasoning), 3=S3 (formal).
    #[pyo3(get)]
    pub tier: u8,
    /// Confidence in the routing decision (0.0-1.0).
    #[pyo3(get)]
    pub confidence: f32,
    /// Which stage produced this result (0-3).
    #[pyo3(get)]
    pub stage: u8,
    /// Structural features used for routing.
    #[pyo3(get)]
    pub features: StructuralFeatures,
}

#[pymethods]
impl RoutingResult {
    fn __repr__(&self) -> String {
        format!(
            "RoutingResult(tier=S{}, confidence={:.3}, stage={})",
            self.tier, self.confidence, self.stage,
        )
    }
}

// ── RoutingFeedback ──────────────────────────────────────────────────────────

/// Feedback record for online learning.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct RoutingFeedback {
    pub task: String,
    pub routed_tier: u8,
    pub actual_quality: f32,
    pub latency_ms: u64,
    pub cost_usd: f32,
}

// ── AdaptiveRouter ───────────────────────────────────────────────────────────

const DEFAULT_C0_THRESHOLD: f32 = 0.85;
const DEFAULT_C1_THRESHOLD: f32 = 0.70;
const FEEDBACK_MAX: usize = 10_000;
const FEEDBACK_DRAIN: usize = 5_000;

/// 4-stage adaptive router. Stages 0-1 in Rust, 2-3 in Python.
///
/// Stage 0 uses structural features (keyword complexity, uncertainty, tool
/// requirement) to produce a tier with a confidence score. If confidence
/// exceeds `c0_threshold`, we return immediately. Otherwise, Stage 1
/// (ONNX classifier) is attempted.
#[pyclass]
pub struct AdaptiveRouter {
    #[allow(dead_code)]
    exemplar_embeddings: Vec<(Vec<f32>, u8)>,
    classifier: Mutex<Option<ort::session::Session>>,
    classifier_tokenizer: Option<tokenizers::Tokenizer>,
    c0_threshold: f32,
    c1_threshold: f32,
    feedback: Mutex<Vec<RoutingFeedback>>,
}

impl AdaptiveRouter {
    /// Map (complexity, uncertainty, tool_required) to (tier, confidence).
    fn tier_from_complexity(complexity: f32, uncertainty: f32, tool_req: bool) -> (u8, f32) {
        // S3: high complexity or high uncertainty.
        if complexity > 0.65 || uncertainty > 0.6 {
            let excess = (complexity - 0.65).max(0.0) + (uncertainty - 0.6).max(0.0);
            let confidence = (0.6 + excess).clamp(0.0, 1.0);
            return (3, confidence);
        }

        // S1: low complexity, low uncertainty, no tool use.
        if complexity <= 0.50 && uncertainty <= 0.3 && !tool_req {
            // Lower complexity = higher confidence in S1.
            let confidence = (1.0 - complexity).clamp(0.0, 1.0);
            return (1, confidence);
        }

        // S2: everything else.
        (2, 0.6)
    }

    /// Stage 1: ONNX BERT classifier inference.
    ///
    /// Tokenizes `task`, runs the classifier ONNX model, applies softmax,
    /// and maps output logits to a routing tier.
    ///
    /// Supports both binary (routellm/bert: class 0 = weak model OK, class 1 =
    /// strong model needed) and multi-class (class 0=S1, 1=S2, 2=S3) models.
    /// For binary output, uses `features.keyword_complexity` to split S2/S3.
    ///
    /// Dynamically discovers required model inputs from `session.inputs` so that
    /// models without `token_type_ids` (e.g., RoBERTa/XLM-RoBERTa) work correctly.
    fn route_stage1(
        &self,
        task: &str,
        features: &StructuralFeatures,
    ) -> Option<RoutingResult> {
        let mut session_guard = self.classifier.lock().unwrap();
        let session = session_guard.as_mut()?;
        let tokenizer = self.classifier_tokenizer.as_ref()?;

        // Tokenize with truncation to model max (512 tokens).
        let encoding = tokenizer.encode(task, true).ok()?;
        let raw_ids = encoding.get_ids();
        let raw_mask = encoding.get_attention_mask();

        let max_len: usize = 512;
        let seq_len = raw_ids.len().min(max_len);

        let ids: Vec<i64> = raw_ids[..seq_len].iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = raw_mask[..seq_len].iter().map(|&m| m as i64).collect();

        let shape = vec![1usize, seq_len];

        let id_tensor = TensorRef::from_array_view((shape.clone(), &*ids)).ok()?;
        let mask_tensor = TensorRef::from_array_view((shape.clone(), &*mask)).ok()?;

        // Discover required inputs from session metadata.
        // RoBERTa/XLM-RoBERTa do NOT use token_type_ids; BERT does.
        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        let outputs = if needs_token_type_ids {
            let type_ids = vec![0i64; seq_len];
            let type_tensor = TensorRef::from_array_view((shape, &*type_ids)).ok()?;
            session
                .run(ort::inputs![
                    "input_ids" => id_tensor,
                    "attention_mask" => mask_tensor,
                    "token_type_ids" => type_tensor
                ])
                .ok()?
        } else {
            session
                .run(ort::inputs![
                    "input_ids" => id_tensor,
                    "attention_mask" => mask_tensor
                ])
                .ok()?
        };

        // Extract logits — shape is [1, num_classes].
        let (out_shape, logits) = outputs[0].try_extract_tensor::<f32>().ok()?;
        let dims: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();
        let num_classes = if dims.len() == 2 { dims[1] } else { dims[0] };

        if num_classes == 0 {
            return None;
        }

        // Softmax over logits.
        let max_logit = logits
            .iter()
            .take(num_classes)
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits
            .iter()
            .take(num_classes)
            .map(|&l| (l - max_logit).exp())
            .sum();
        let probs: Vec<f32> = logits
            .iter()
            .take(num_classes)
            .map(|&l| (l - max_logit).exp() / exp_sum)
            .collect();

        // Map probabilities to routing tier.
        let (tier, confidence) = if num_classes == 2 {
            // Binary: class 0 = weak model sufficient (S1),
            //         class 1 = strong model needed (S2 or S3).
            if probs[0] > probs[1] {
                (1u8, probs[0])
            } else if features.keyword_complexity > 0.65 {
                // High structural complexity => formal verification tier.
                (3, probs[1])
            } else {
                (2, probs[1])
            }
        } else if num_classes >= 3 {
            // Multi-class: class 0=S1, 1=S2, 2=S3 (direct mapping).
            let (idx, &max_prob) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())?;
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
}

#[pymethods]
impl AdaptiveRouter {
    /// Create a new AdaptiveRouter.
    ///
    /// # Arguments
    /// * `c0_threshold` - Confidence threshold for Stage 0 (default 0.85).
    /// * `c1_threshold` - Confidence threshold for Stage 1 (default 0.70).
    /// * `classifier_path` - Path to ONNX classifier model (e.g., `models/classifier/model.onnx`).
    /// * `tokenizer_path` - Path to tokenizer JSON (e.g., `models/classifier/tokenizer.json`).
    #[new]
    #[pyo3(signature = (c0_threshold=None, c1_threshold=None, classifier_path=None, tokenizer_path=None))]
    pub fn new(
        py: Python<'_>,
        c0_threshold: Option<f32>,
        c1_threshold: Option<f32>,
        classifier_path: Option<String>,
        tokenizer_path: Option<String>,
    ) -> Self {
        // Extract sys.prefix while we hold the GIL — before entering Once::call_once
        let sys_prefix: Option<String> = py
            .import("sys")
            .ok()
            .and_then(|sys| sys.getattr("prefix").ok())
            .and_then(|p| p.extract().ok());
        let mut classifier = None;
        let mut classifier_tokenizer = None;

        if let (Some(ref cp), Some(ref tp)) = (&classifier_path, &tokenizer_path) {
            // Only attempt loading if both files exist on disk.
            // ORT initialization with nonexistent paths can hang on Windows.
            let cp_exists = std::path::Path::new(cp).exists();
            let tp_exists = std::path::Path::new(tp).exists();

            if cp_exists && tp_exists {
                // Ensure ORT_DYLIB_PATH is set before loading classifier
                // (thread-safe via OnceLock — resolved once, reused everywhere).
                if let Some(path) = crate::memory::embedder::resolve_ort_dylib_once(cp, sys_prefix.as_deref()) {
                    if std::env::var("ORT_DYLIB_PATH").is_err() {
                        std::env::set_var("ORT_DYLIB_PATH", path);
                    }
                }

                // Release the GIL before loading ORT — on Windows, LoadLibraryW
                // runs DllMain which may attempt GIL acquisition, causing deadlock.
                let cp_clone = cp.clone();
                let tp_clone = tp.clone();
                let (sess_result, tok_result) = py.allow_threads(|| {
                    let sess = ort::session::Session::builder()
                        .and_then(|mut b| b.commit_from_file(&cp_clone));
                    let tok = tokenizers::Tokenizer::from_file(&tp_clone);
                    (sess, tok)
                });

                match sess_result {
                    Ok(session) => {
                        classifier = Some(session);
                    }
                    Err(e) => {
                        eprintln!(
                            "[AdaptiveRouter] WARN: failed to load classifier ONNX: {e}"
                        );
                    }
                }

                match tok_result {
                    Ok(tok) => {
                        classifier_tokenizer = Some(tok);
                    }
                    Err(e) => {
                        eprintln!("[AdaptiveRouter] WARN: failed to load tokenizer: {e}");
                    }
                }
            } else {
                if !cp_exists {
                    eprintln!("[AdaptiveRouter] WARN: classifier model not found: {cp}");
                }
                if !tp_exists {
                    eprintln!("[AdaptiveRouter] WARN: tokenizer not found: {tp}");
                }
            }
        }

        Self {
            exemplar_embeddings: Vec::new(),
            classifier: Mutex::new(classifier),
            classifier_tokenizer,
            c0_threshold: c0_threshold.unwrap_or(DEFAULT_C0_THRESHOLD),
            c1_threshold: c1_threshold.unwrap_or(DEFAULT_C1_THRESHOLD),
            feedback: Mutex::new(Vec::new()),
        }
    }

    /// Stage 0: route using structural features only.
    ///
    /// Extracts `StructuralFeatures` from the task and maps complexity /
    /// uncertainty / tool-requirement to a tier.
    pub fn route_stage0(&self, task: &str) -> RoutingResult {
        let features = StructuralFeatures::extract_from(task);
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

    /// Full routing pipeline: Stage 0, then Stage 1 if confidence is low.
    ///
    /// Returns the first stage result whose confidence exceeds its threshold,
    /// or the best result seen so far.
    pub fn route(&self, task: &str) -> RoutingResult {
        let s0 = self.route_stage0(task);
        if s0.confidence >= self.c0_threshold {
            return s0;
        }

        // Stage 1: ONNX classifier (when loaded).
        if let Some(s1) = self.route_stage1(task, &s0.features) {
            if s1.confidence >= self.c1_threshold {
                return s1;
            }
        }

        // Fall through to Stage 0 result.
        s0
    }

    /// Record routing feedback for online learning.
    ///
    /// Feedback is buffered in memory. When the buffer exceeds 10,000 entries,
    /// the oldest 5,000 are drained.
    pub fn record_feedback(
        &self,
        task: String,
        routed_tier: u8,
        actual_quality: f32,
        latency_ms: u64,
        cost_usd: f32,
    ) {
        let mut fb = self.feedback.lock().unwrap();
        if fb.len() >= FEEDBACK_MAX {
            fb.drain(..FEEDBACK_DRAIN);
        }
        fb.push(RoutingFeedback {
            task,
            routed_tier,
            actual_quality,
            latency_ms,
            cost_usd,
        });
    }

    /// Number of feedback records currently buffered.
    pub fn feedback_count(&self) -> usize {
        self.feedback.lock().unwrap().len()
    }

    /// Whether an ONNX classifier is loaded (Stage 1 available).
    pub fn has_classifier(&self) -> bool {
        self.classifier.lock().unwrap().is_some()
    }

    /// Current Stage 0 confidence threshold.
    #[pyo3(name = "get_c0_threshold")]
    pub fn py_c0_threshold(&self) -> f32 {
        self.c0_threshold
    }

    /// Current Stage 1 confidence threshold.
    #[pyo3(name = "get_c1_threshold")]
    pub fn py_c1_threshold(&self) -> f32 {
        self.c1_threshold
    }
}

// ── Test-only constructor ────────────────────────────────────────────────────

#[cfg(test)]
impl AdaptiveRouter {
    /// Internal constructor for Rust unit tests (no GIL needed, no classifier).
    pub(crate) fn new_without_py(
        c0_threshold: Option<f32>,
        c1_threshold: Option<f32>,
    ) -> Self {
        Self {
            exemplar_embeddings: Vec::new(),
            classifier: Mutex::new(None),
            classifier_tokenizer: None,
            c0_threshold: c0_threshold.unwrap_or(DEFAULT_C0_THRESHOLD),
            c1_threshold: c1_threshold.unwrap_or(DEFAULT_C1_THRESHOLD),
            feedback: Mutex::new(Vec::new()),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage0_simple_task_routes_s1() {
        let router = AdaptiveRouter::new_without_py(None, None);
        let result = router.route_stage0("What is the capital of France?");
        assert_eq!(result.tier, 1, "Simple factual question should route to S1");
        assert_eq!(result.stage, 0);
    }

    #[test]
    fn test_stage0_code_task_routes_s2() {
        let router = AdaptiveRouter::new_without_py(None, None);
        // "write" -> CODE_KEYWORDS (complexity 0.35), "run" + "test" -> TOOL_KEYWORDS.
        // tool_required=true prevents S1, complexity <= 0.65 prevents S3 -> S2.
        let result = router.route_stage0(
            "Write a Python function to parse JSON, then run the test suite",
        );
        assert_eq!(result.tier, 2, "Code task with tool keyword should route to S2, got S{}", result.tier);
        assert_eq!(result.stage, 0);
    }

    #[test]
    fn test_stage0_complex_task_routes_s3() {
        let router = AdaptiveRouter::new_without_py(None, None);
        // "implement" -> ALGO_KEYWORDS (complexity base 0.2 + 0.35 = 0.55).
        // >100 words adds +0.15, pushing to 0.70 > 0.65 threshold -> S3.
        let padding = "using advanced techniques in the system ".repeat(16);
        let task = format!(
            "Implement a distributed consensus algorithm with lock-free data structures {}",
            padding,
        );
        let result = router.route_stage0(&task);
        assert!(
            result.features.word_count > 100,
            "Task should have >100 words, got {}",
            result.features.word_count,
        );
        assert_eq!(result.tier, 3, "Complex algo task (>100 words) should route to S3, got S{}", result.tier);
        assert_eq!(result.stage, 0);
    }

    #[test]
    fn test_stage0_confidence_in_valid_range() {
        let router = AdaptiveRouter::new_without_py(None, None);
        let tasks = [
            "Hello world",
            "Write a function to sort a list",
            "Implement a compiler for a subset of Haskell with formal verification",
            "Debug the intermittent flaky race condition in the concurrent queue",
        ];
        for task in &tasks {
            let result = router.route_stage0(task);
            assert!(
                (0.0..=1.0).contains(&result.confidence),
                "Confidence {} out of range for task: {}",
                result.confidence,
                task,
            );
        }
    }

    #[test]
    fn test_default_thresholds() {
        let router = AdaptiveRouter::new_without_py(None, None);
        assert!(
            (router.c0_threshold - 0.85).abs() < f32::EPSILON,
            "Default c0 should be 0.85, got {}",
            router.c0_threshold,
        );
        assert!(
            (router.c1_threshold - 0.70).abs() < f32::EPSILON,
            "Default c1 should be 0.70, got {}",
            router.c1_threshold,
        );
    }

    #[test]
    fn test_custom_thresholds() {
        let router = AdaptiveRouter::new_without_py(Some(0.90), Some(0.80));
        assert!(
            (router.c0_threshold - 0.90).abs() < f32::EPSILON,
            "Custom c0 should be 0.90, got {}",
            router.c0_threshold,
        );
        assert!(
            (router.c1_threshold - 0.80).abs() < f32::EPSILON,
            "Custom c1 should be 0.80, got {}",
            router.c1_threshold,
        );
    }

    #[test]
    fn test_load_classifier_nonexistent_returns_no_classifier() {
        // Passing no classifier path still creates a router — classifier just isn't loaded.
        let router = AdaptiveRouter::new_without_py(None, None);
        assert!(
            !router.has_classifier(),
            "Should have no classifier when model file does not exist",
        );
    }

    #[test]
    fn test_route_without_classifier_uses_stage0() {
        let router = AdaptiveRouter::new_without_py(None, None);
        let result = router.route("What is 2+2?");
        // Without a classifier, route() should fall through to stage0 result.
        assert_eq!(result.stage, 0, "Without classifier, route should use stage 0");
        assert_eq!(result.tier, 1, "Simple arithmetic question should be S1");
    }

    #[test]
    fn test_feedback_recording() {
        let router = AdaptiveRouter::new_without_py(None, None);
        assert_eq!(router.feedback_count(), 0);

        router.record_feedback("test task".to_string(), 2, 0.85, 150, 0.01);
        assert_eq!(router.feedback_count(), 1);

        router.record_feedback("another task".to_string(), 1, 0.95, 50, 0.001);
        assert_eq!(router.feedback_count(), 2);

        // Verify feedback data integrity via count.
        let fb = router.feedback.lock().unwrap();
        assert_eq!(fb[0].task, "test task");
        assert_eq!(fb[0].routed_tier, 2);
        assert!((fb[0].actual_quality - 0.85).abs() < f32::EPSILON);
        assert_eq!(fb[1].latency_ms, 50);
    }

    #[test]
    fn test_feedback_buffer_bounded() {
        let router = AdaptiveRouter::new_without_py(None, None);

        // Fill to capacity.
        for i in 0..FEEDBACK_MAX {
            router.record_feedback(
                format!("task_{}", i),
                1,
                0.9,
                100,
                0.01,
            );
        }
        assert_eq!(router.feedback_count(), FEEDBACK_MAX);

        // One more triggers drain of oldest 5000.
        router.record_feedback("overflow".to_string(), 2, 0.5, 200, 0.02);
        assert_eq!(
            router.feedback_count(),
            FEEDBACK_MAX - FEEDBACK_DRAIN + 1,
            "After overflow, buffer should be drained to {} + 1 new = {}",
            FEEDBACK_MAX - FEEDBACK_DRAIN,
            FEEDBACK_MAX - FEEDBACK_DRAIN + 1,
        );

        // Oldest entries should be gone — first remaining should be task_5000.
        let fb = router.feedback.lock().unwrap();
        assert_eq!(fb[0].task, format!("task_{}", FEEDBACK_DRAIN));
    }
}
