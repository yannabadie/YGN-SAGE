//! AdaptiveRouter — 5-stage learned S1/S2/S3 routing.
//!
//! Stage 0: Structural features only (keyword complexity + uncertainty).
//! Stage 0.5: kNN on pre-computed exemplar embeddings (arXiv 2505.12601).
//! Stage 1: ONNX BERT classifier (routellm/bert) on task text.
//! Stage 2-3: Python-side dynamic routing with feedback.
//!
//! Behind the `onnx` feature flag (shares deps with RustEmbedder).

use ort::value::TensorRef;
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::routing::features::StructuralFeatures;

// ── ShadowTrace ────────────────────────────────────────────────────────────

/// A single routing trace record for shadow comparison analysis.
#[derive(Clone, Debug)]
pub struct ShadowTrace {
    /// FNV-style hash of the task text.
    pub task_hash: u64,
    /// Tier chosen by Stage 0 (structural features).
    pub structural_tier: u8,
    /// Tier chosen by Stage 1 (ONNX), if available.
    pub onnx_tier: Option<u8>,
    /// Timestamp in milliseconds since Unix epoch.
    pub timestamp_ms: u64,
}

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
    shadow_traces: Mutex<Vec<ShadowTrace>>,
}

impl AdaptiveRouter {
    /// Stage 0.5: kNN routing on pre-computed exemplar embeddings.
    ///
    /// Computes cosine similarity (dot product on L2-normalized vectors) between
    /// the query embedding and all exemplars. Returns weighted majority vote.
    fn route_knn(&self, query_embedding: &[f32], k: usize, features: &StructuralFeatures) -> Option<RoutingResult> {
        if self.exemplar_embeddings.is_empty() || query_embedding.len() != 768 {
            return None;
        }

        let k = k.min(self.exemplar_embeddings.len());

        // Compute cosine similarities (dot product — both sides L2-normalized).
        let mut scored: Vec<(f32, u8)> = self
            .exemplar_embeddings
            .iter()
            .map(|(emb, label)| {
                let sim: f32 = emb.iter().zip(query_embedding.iter()).map(|(a, b)| a * b).sum();
                (sim, *label)
            })
            .collect();

        // Partial sort: top-k by descending similarity.
        scored.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        let top_k = &scored[..k];

        // OOD rejection: if nearest neighbor similarity < 0.3
        let max_sim = top_k.iter().map(|(s, _)| *s).fold(f32::NEG_INFINITY, f32::max);
        if max_sim < 0.3 {
            return None;
        }

        // Distance-weighted majority vote.
        let mut votes = [0.0f32; 4]; // index 1,2,3 for S1/S2/S3
        for &(sim, label) in top_k {
            let weight = sim.max(0.0);
            let idx = (label as usize).min(3);
            if idx > 0 {
                votes[idx] += weight;
            }
        }

        let (winner, &max_vote) = votes
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;

        let total_weight: f32 = votes[1..].iter().sum();
        let confidence = if total_weight > 0.0 { max_vote / total_weight } else { 0.0 };

        Some(RoutingResult {
            tier: winner as u8,
            confidence,
            stage: 0,
            features: features.clone(),
        })
    }

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
    fn route_stage1(&self, task: &str, features: &StructuralFeatures) -> Option<RoutingResult> {
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
                if let Some(path) =
                    crate::memory::embedder::resolve_ort_dylib_once(cp, sys_prefix.as_deref())
                {
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
                        eprintln!("[AdaptiveRouter] WARN: failed to load classifier ONNX: {e}");
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
            shadow_traces: Mutex::new(Vec::new()),
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
    /// or the best result seen so far. Appends a `ShadowTrace` for every call.
    pub fn route(&self, task: &str) -> RoutingResult {
        let s0 = self.route_stage0(task);

        // Attempt Stage 1 if Stage 0 confidence is below threshold.
        let s1_result = if s0.confidence < self.c0_threshold {
            self.route_stage1(task, &s0.features)
        } else {
            None
        };

        // Record shadow trace.
        let mut hasher = DefaultHasher::new();
        task.hash(&mut hasher);
        let task_hash = hasher.finish();
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        if let Ok(mut traces) = self.shadow_traces.lock() {
            traces.push(ShadowTrace {
                task_hash,
                structural_tier: s0.tier,
                onnx_tier: s1_result.as_ref().map(|r| r.tier),
                timestamp_ms,
            });
        }

        // Return best result.
        if let Some(ref s1) = s1_result {
            if s1.confidence >= self.c1_threshold {
                return s1.clone();
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

    /// Load pre-computed exemplar embeddings for kNN routing.
    ///
    /// # Arguments
    /// * `embeddings` - Flat list of f32 values (N * 768 elements).
    /// * `labels` - List of N labels (1=S1, 2=S2, 3=S3).
    ///
    /// Returns the number of exemplars loaded.
    pub fn load_exemplars(&mut self, embeddings: Vec<f32>, labels: Vec<u8>) -> usize {
        let dim = 768usize;
        let n = labels.len();
        if embeddings.len() != n * dim {
            return 0;
        }

        self.exemplar_embeddings = (0..n)
            .map(|i| {
                let start = i * dim;
                let mut vec = embeddings[start..start + dim].to_vec();
                // L2-normalize
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    vec.iter_mut().for_each(|x| *x /= norm);
                }
                (vec, labels[i])
            })
            .collect();

        n
    }

    /// Number of loaded exemplar embeddings.
    pub fn exemplar_count(&self) -> usize {
        self.exemplar_embeddings.len()
    }

    /// Whether kNN exemplars are loaded (Stage 0.5 available).
    pub fn has_knn(&self) -> bool {
        !self.exemplar_embeddings.is_empty()
    }

    /// Route a task using a pre-computed embedding (kNN Stage 0.5 + structural fallback).
    ///
    /// If kNN exemplars are loaded, tries kNN first. Falls back to structural + ONNX.
    pub fn route_with_embedding(&self, task: &str, embedding: Vec<f32>) -> RoutingResult {
        let features = StructuralFeatures::extract_from(task);

        // Stage 0.5: kNN on embedding
        if !self.exemplar_embeddings.is_empty() {
            if let Some(knn_result) = self.route_knn(&embedding, 5, &features) {
                return knn_result;
            }
        }

        // Fallback to structural + ONNX
        let (tier, confidence) = Self::tier_from_complexity(
            features.keyword_complexity,
            features.keyword_uncertainty,
            features.tool_required,
        );
        let s0 = RoutingResult { tier, confidence, stage: 0, features: features.clone() };

        if s0.confidence < self.c0_threshold {
            if let Some(s1) = self.route_stage1(task, &s0.features) {
                if s1.confidence >= self.c1_threshold {
                    return s1;
                }
            }
        }

        s0
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

    /// Number of shadow traces currently buffered.
    pub fn shadow_trace_count(&self) -> usize {
        self.shadow_traces.lock().unwrap().len()
    }

    /// Write buffered shadow traces as JSONL to `path`, clear the buffer,
    /// and return the number of traces written.
    pub fn flush_shadow_traces(&mut self, path: &str) -> PyResult<usize> {
        let mut traces = self.shadow_traces.lock().unwrap();
        let count = traces.len();
        if count == 0 {
            return Ok(0);
        }

        // Ensure parent directory exists.
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "failed to create directory {}: {}",
                    parent.display(),
                    e
                ))
            })?;
        }

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("failed to open {path}: {e}"))
            })?;

        use std::io::Write;
        for trace in traces.iter() {
            let onnx_tier = match trace.onnx_tier {
                Some(t) => format!("{t}"),
                None => "null".to_string(),
            };
            writeln!(
                file,
                "{{\"task_hash\":{},\"structural_tier\":{},\"onnx_tier\":{},\"timestamp_ms\":{}}}",
                trace.task_hash, trace.structural_tier, onnx_tier, trace.timestamp_ms,
            )
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("write error: {e}")))?;
        }
        traces.clear();
        Ok(count)
    }

    /// Retrain thresholds using buffered feedback.
    ///
    /// For each feedback entry, re-routes structurally and compares the
    /// structural tier to the actual quality-implied tier:
    /// - If structural over-routes to S1 (actual was S2/S3): lower c0_threshold
    /// - If structural over-routes to S3 (actual was S1/S2): raise c0_threshold
    /// Clamps to [0.5, 0.95]. Clears feedback after retraining.
    pub fn retrain_thresholds(&mut self) {
        let mut fb = self.feedback.lock().unwrap();
        if fb.is_empty() {
            return;
        }

        let mut c0_delta: f32 = 0.0;
        let mut c1_delta: f32 = 0.0;

        for entry in fb.iter() {
            // Determine the "actual" tier from quality:
            // high quality (>= 0.7) achieved => could have used a simpler tier
            // low quality (< 0.7) => needed a stronger tier
            let actual_tier = if entry.actual_quality >= 0.8 {
                // High quality — the routed tier was sufficient or over-provisioned
                entry.routed_tier
            } else if entry.actual_quality >= 0.5 {
                // Medium quality — likely needed one tier higher
                (entry.routed_tier + 1).min(3)
            } else {
                // Low quality — definitely needed a higher tier
                3
            };

            // Re-route structurally to get what Stage 0 would have said.
            let s0 = self.route_stage0(&entry.task);

            // Compare structural prediction to actual tier.
            if s0.tier < actual_tier {
                // Under-routed (e.g., S1 when S2/S3 was needed) => lower threshold
                // to make Stage 0 less confident, triggering Stage 1 more often.
                c0_delta -= 0.01;
            } else if s0.tier > actual_tier {
                // Over-routed (e.g., S3 when S1/S2 was fine) => raise threshold
                // to accept Stage 0's simpler routing more readily.
                c0_delta += 0.01;
            }

            // Same logic for c1 (ONNX vs actual).
            if entry.routed_tier < actual_tier {
                c1_delta -= 0.01;
            } else if entry.routed_tier > actual_tier {
                c1_delta += 0.01;
            }
        }

        self.c0_threshold = (self.c0_threshold + c0_delta).clamp(0.5, 0.95);
        self.c1_threshold = (self.c1_threshold + c1_delta).clamp(0.5, 0.95);
        fb.clear();
    }
}

// ── Test-only constructor ────────────────────────────────────────────────────

#[cfg(test)]
impl AdaptiveRouter {
    /// Internal constructor for Rust unit tests (no GIL needed, no classifier).
    pub(crate) fn new_without_py(c0_threshold: Option<f32>, c1_threshold: Option<f32>) -> Self {
        Self {
            exemplar_embeddings: Vec::new(),
            classifier: Mutex::new(None),
            classifier_tokenizer: None,
            c0_threshold: c0_threshold.unwrap_or(DEFAULT_C0_THRESHOLD),
            c1_threshold: c1_threshold.unwrap_or(DEFAULT_C1_THRESHOLD),
            feedback: Mutex::new(Vec::new()),
            shadow_traces: Mutex::new(Vec::new()),
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
        let result =
            router.route_stage0("Write a Python function to parse JSON, then run the test suite");
        assert_eq!(
            result.tier, 2,
            "Code task with tool keyword should route to S2, got S{}",
            result.tier
        );
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
        assert_eq!(
            result.tier, 3,
            "Complex algo task (>100 words) should route to S3, got S{}",
            result.tier
        );
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
        assert_eq!(
            result.stage, 0,
            "Without classifier, route should use stage 0"
        );
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
    fn test_shadow_trace_collection() {
        let router = AdaptiveRouter::new_without_py(None, None);
        assert_eq!(router.shadow_trace_count(), 0);
        router.route("hello world");
        assert!(router.shadow_trace_count() >= 1);
    }

    #[test]
    fn test_shadow_trace_fields() {
        let router = AdaptiveRouter::new_without_py(None, None);
        router.route("What is 2+2?");
        let traces = router.shadow_traces.lock().unwrap();
        assert_eq!(traces.len(), 1);
        let t = &traces[0];
        assert!(t.task_hash != 0);
        assert!(t.structural_tier >= 1 && t.structural_tier <= 3);
        assert!(t.onnx_tier.is_none()); // no classifier loaded
        assert!(t.timestamp_ms > 0);
    }

    #[test]
    fn test_flush_shadow_traces() {
        let router = AdaptiveRouter::new_without_py(None, None);
        router.route("task one");
        router.route("task two");
        assert_eq!(router.shadow_trace_count(), 2);

        let dir = std::env::temp_dir().join("sage_test_shadow");
        let path = dir.join("traces.jsonl");

        // Flush — need PyResult so call inner logic directly
        {
            let mut traces = router.shadow_traces.lock().unwrap();
            let count = traces.len();
            assert_eq!(count, 2);
            std::fs::create_dir_all(&dir).unwrap();
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&path)
                .unwrap();
            use std::io::Write;
            for trace in traces.iter() {
                let onnx_tier = match trace.onnx_tier {
                    Some(t) => format!("{t}"),
                    None => "null".to_string(),
                };
                writeln!(
                    file,
                    "{{\"task_hash\":{},\"structural_tier\":{},\"onnx_tier\":{},\"timestamp_ms\":{}}}",
                    trace.task_hash, trace.structural_tier, onnx_tier, trace.timestamp_ms,
                )
                .unwrap();
            }
            traces.clear();
        }

        assert_eq!(router.shadow_trace_count(), 0);
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert_eq!(lines.len(), 2);

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_retrain_thresholds_adjusts() {
        let mut router = AdaptiveRouter::new_without_py(None, None);
        let old_c0 = router.c0_threshold;
        // Record feedback simulating structural under-routing:
        // route to tier 1 but quality is low (0.3) => actual tier should be 3
        for _ in 0..50 {
            router.record_feedback(
                "complex task requiring formal verification".to_string(),
                1,   // routed_tier
                0.3, // actual_quality (low => needed higher tier)
                500,
                0.05,
            );
        }
        router.retrain_thresholds();
        // Should have adjusted (or hit clamp boundary)
        assert!(
            (router.c0_threshold - old_c0).abs() > f32::EPSILON
                || (router.c0_threshold - 0.5).abs() < f32::EPSILON,
            "c0_threshold should have changed from {old_c0}, got {}",
            router.c0_threshold,
        );
        // Feedback should be cleared
        assert_eq!(router.feedback_count(), 0);
    }

    #[test]
    fn test_retrain_thresholds_clamped() {
        let mut router = AdaptiveRouter::new_without_py(Some(0.94), None);
        // Over-routing feedback (high quality with high tier => should raise threshold)
        for _ in 0..200 {
            router.record_feedback(
                "hello".to_string(),
                3,    // routed to S3
                0.95, // high quality => didn't need S3
                50,
                0.001,
            );
        }
        router.retrain_thresholds();
        // Should be clamped to max 0.95
        assert!(
            router.c0_threshold <= 0.95,
            "c0 must be <= 0.95, got {}",
            router.c0_threshold
        );
        assert!(
            router.c0_threshold >= 0.5,
            "c0 must be >= 0.5, got {}",
            router.c0_threshold
        );
    }

    #[test]
    fn test_retrain_empty_feedback_noop() {
        let mut router = AdaptiveRouter::new_without_py(None, None);
        let old_c0 = router.c0_threshold;
        let old_c1 = router.c1_threshold;
        router.retrain_thresholds();
        assert!((router.c0_threshold - old_c0).abs() < f32::EPSILON);
        assert!((router.c1_threshold - old_c1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_c0_c1_field_access() {
        let router = AdaptiveRouter::new_without_py(Some(0.75), Some(0.60));
        assert!((router.c0_threshold - 0.75).abs() < f32::EPSILON);
        assert!((router.c1_threshold - 0.60).abs() < f32::EPSILON);
    }

    #[test]
    fn test_knn_load_exemplars() {
        let mut router = AdaptiveRouter::new_without_py(None, None);
        assert_eq!(router.exemplar_count(), 0);
        assert!(!router.has_knn());

        // 3 exemplars of dimension 768
        let dim = 768;
        let mut embeddings = vec![0.0f32; 3 * dim];
        // S1 exemplar: signal in dim 0
        embeddings[0] = 1.0;
        // S2 exemplar: signal in dim 1
        embeddings[dim + 1] = 1.0;
        // S3 exemplar: signal in dim 2
        embeddings[2 * dim + 2] = 1.0;

        let labels = vec![1u8, 2, 3];
        let n = router.load_exemplars(embeddings, labels);
        assert_eq!(n, 3);
        assert_eq!(router.exemplar_count(), 3);
        assert!(router.has_knn());
    }

    #[test]
    fn test_knn_routing_basic() {
        let mut router = AdaptiveRouter::new_without_py(None, None);
        let dim = 768;

        // Create 6 exemplars: 2 per system
        let mut embeddings = vec![0.0f32; 6 * dim];
        let labels = vec![1u8, 1, 2, 2, 3, 3];
        // S1 exemplars: strong signal in dims 0-9
        for i in 0..2 {
            for d in 0..10 {
                embeddings[i * dim + d] = 1.0;
            }
        }
        // S2 exemplars: strong signal in dims 10-19
        for i in 2..4 {
            for d in 10..20 {
                embeddings[i * dim + d] = 1.0;
            }
        }
        // S3 exemplars: strong signal in dims 20-29
        for i in 4..6 {
            for d in 20..30 {
                embeddings[i * dim + d] = 1.0;
            }
        }

        router.load_exemplars(embeddings, labels);

        // Query close to S3 cluster
        let mut query = vec![0.0f32; dim];
        for d in 20..30 {
            query[d] = 1.0;
        }
        // L2 normalize
        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        query.iter_mut().for_each(|x| *x /= norm);

        let features = StructuralFeatures::extract_from("test");
        let result = router.route_knn(&query, 5, &features);
        assert!(result.is_some());
        assert_eq!(result.unwrap().tier, 3);
    }

    #[test]
    fn test_knn_ood_rejection() {
        let mut router = AdaptiveRouter::new_without_py(None, None);
        let dim = 768;

        // Single exemplar
        let mut embeddings = vec![0.0f32; dim];
        embeddings[0] = 1.0;
        router.load_exemplars(embeddings, vec![1u8]);

        // Query orthogonal to exemplar → cosine sim ≈ 0 → OOD rejection
        let mut query = vec![0.0f32; dim];
        query[dim - 1] = 1.0;

        let features = StructuralFeatures::extract_from("test");
        let result = router.route_knn(&query, 1, &features);
        assert!(result.is_none(), "Orthogonal query should be OOD-rejected");
    }

    #[test]
    fn test_knn_load_bad_dimensions() {
        let mut router = AdaptiveRouter::new_without_py(None, None);
        // Mismatched: 3 labels but only 2*768 embedding values
        let n = router.load_exemplars(vec![0.0; 2 * 768], vec![1, 2, 3]);
        assert_eq!(n, 0);
        assert!(!router.has_knn());
    }

    #[test]
    fn test_route_with_embedding_uses_knn() {
        let mut router = AdaptiveRouter::new_without_py(None, None);
        let dim = 768;

        // 2 S3 exemplars
        let mut embeddings = vec![0.0f32; 2 * dim];
        for i in 0..2 {
            for d in 0..10 {
                embeddings[i * dim + d] = 1.0;
            }
        }
        router.load_exemplars(embeddings, vec![3u8, 3]);

        // Query matching S3 cluster
        let mut query = vec![0.0f32; dim];
        for d in 0..10 {
            query[d] = 1.0;
        }
        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        query.iter_mut().for_each(|x| *x /= norm);

        let result = router.route_with_embedding("hello", query);
        assert_eq!(result.tier, 3, "kNN should route to S3");
    }

    #[test]
    fn test_feedback_buffer_bounded() {
        let router = AdaptiveRouter::new_without_py(None, None);

        // Fill to capacity.
        for i in 0..FEEDBACK_MAX {
            router.record_feedback(format!("task_{}", i), 1, 0.9, 100, 0.01);
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
