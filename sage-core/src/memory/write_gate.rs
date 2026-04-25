//! Composite salience write gate — 5-signal scoring for memory writes.
//!
//! Research: arXiv 2603.15994 — composite salience gate achieves 100% accuracy
//! vs 13% with ungated writes. Signals: confidence, novelty (cosine distance),
//! reliability (source tier), recency (time decay), task relevance (keyword overlap).
//!
//! Exposes `RustCompositeWriteGate` via PyO3 with the same API as the Python version.

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;
use tracing::instrument;

/// Stop words for relevance scoring (same set as RustRelevanceGate).
const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "to", "of", "in", "for", "on", "with", "at", "by", "from", "and",
    "but", "or", "not", "this", "that", "it", "its",
];

/// Default reliability scores per source tier.
fn default_reliability(tier: &str) -> f32 {
    match tier {
        "codex" => 0.95,
        "reasoner" => 0.90,
        "fast" => 0.75,
        "budget" | "budget-alt" => 0.70,
        _ => 0.60,
    }
}

/// Result of a write gate evaluation.
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct WriteGateDecision {
    pub allowed: bool,
    pub salience_score: f32,
    pub reason: String,
    pub confidence: f32,
    pub novelty: f32,
    pub reliability: f32,
    pub recency: f32,
    pub relevance: f32,
}

/// Composite write gate with 5-signal salience scoring.
///
/// All weights from arXiv 2603.15994, subject to ablation.
#[pyclass]
pub struct RustCompositeWriteGate {
    threshold: f32,
    w_confidence: f32,
    w_novelty: f32,
    w_reliability: f32,
    w_recency: f32,
    w_relevance: f32,
    novelty_sim_threshold: f32,
    recency_halflife_s: f32,
    max_seen: usize,
    seen_embeddings: VecDeque<Vec<f32>>,
    seen_content: HashSet<u64>, // FNV hash of content for exact dedup
    write_count: u64,
    abstention_count: u64,
    task_start: Instant,
    stop_words: HashSet<&'static str>,
}

#[pymethods]
impl RustCompositeWriteGate {
    /// Create a new composite write gate with configurable weights.
    #[new]
    #[pyo3(signature = (
        threshold = 0.35,
        w_confidence = 0.25,
        w_novelty = 0.30,
        w_reliability = 0.20,
        w_recency = 0.10,
        w_relevance = 0.15,
        novelty_sim_threshold = 0.90,
        recency_halflife_s = 300.0,
        max_seen = 200,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        threshold: f32,
        w_confidence: f32,
        w_novelty: f32,
        w_reliability: f32,
        w_recency: f32,
        w_relevance: f32,
        novelty_sim_threshold: f32,
        recency_halflife_s: f32,
        max_seen: usize,
    ) -> Self {
        let stop_words: HashSet<&'static str> = STOP_WORDS.iter().copied().collect();
        Self {
            threshold,
            w_confidence,
            w_novelty,
            w_reliability,
            w_recency,
            w_relevance,
            novelty_sim_threshold,
            recency_halflife_s,
            max_seen,
            seen_embeddings: VecDeque::with_capacity(max_seen),
            seen_content: HashSet::new(),
            write_count: 0,
            abstention_count: 0,
            task_start: Instant::now(),
            stop_words,
        }
    }

    /// Reset the recency timer (call at task start).
    pub fn reset_task_timer(&mut self) {
        self.task_start = Instant::now();
    }

    /// Evaluate whether to allow a memory write.
    ///
    /// Args:
    ///     content: The content to write.
    ///     confidence: Model confidence (0-1).
    ///     task: Current task description (for relevance signal).
    ///     source_tier: Model tier that produced the content.
    ///     embedding: Pre-computed embedding (for novelty signal). None = novelty 1.0.
    #[instrument(skip(self, content, task, embedding))]
    #[pyo3(signature = (content, confidence, task = "", source_tier = "unknown", embedding = None))]
    pub fn evaluate(
        &mut self,
        content: &str,
        confidence: f32,
        task: &str,
        source_tier: &str,
        embedding: Option<Vec<f32>>,
    ) -> WriteGateDecision {
        // Hard check: empty content
        if content.is_empty() || content.trim().is_empty() {
            self.abstention_count += 1;
            return WriteGateDecision {
                allowed: false,
                salience_score: 0.0,
                reason: "Blocked: empty content".into(),
                confidence,
                novelty: 0.0,
                reliability: 0.0,
                recency: 0.0,
                relevance: 0.0,
            };
        }

        // Hard check: exact duplicate (via hash)
        let hash = fnv_hash(content);
        if self.seen_content.contains(&hash) {
            self.abstention_count += 1;
            return WriteGateDecision {
                allowed: false,
                salience_score: 0.0,
                reason: "Blocked: exact duplicate".into(),
                confidence,
                novelty: 0.0,
                reliability: 0.0,
                recency: 0.0,
                relevance: 0.0,
            };
        }

        // Signal 1: confidence (clamped)
        let sig_confidence = confidence.clamp(0.0, 1.0);

        // Signal 2: novelty (1 - max cosine similarity to seen embeddings)
        let sig_novelty = self.compute_novelty(&embedding);

        // Signal 3: reliability (source tier reputation)
        let sig_reliability = default_reliability(source_tier);

        // Signal 4: recency (exponential decay since task start)
        let elapsed = self.task_start.elapsed().as_secs_f32();
        let sig_recency = (-0.693 * elapsed / self.recency_halflife_s.max(1.0)).exp();

        // Signal 5: task relevance (keyword overlap)
        let sig_relevance = if task.is_empty() {
            0.5
        } else {
            self.keyword_overlap(task, content)
        };

        // Composite salience score
        let score = self.w_confidence * sig_confidence
            + self.w_novelty * sig_novelty
            + self.w_reliability * sig_reliability
            + self.w_recency * sig_recency
            + self.w_relevance * sig_relevance;

        if score < self.threshold {
            self.abstention_count += 1;
            return WriteGateDecision {
                allowed: false,
                salience_score: score,
                reason: format!(
                    "Blocked: salience {:.3} < threshold {:.3}",
                    score, self.threshold
                ),
                confidence: sig_confidence,
                novelty: sig_novelty,
                reliability: sig_reliability,
                recency: sig_recency,
                relevance: sig_relevance,
            };
        }

        // Allow — update state
        self.write_count += 1;
        self.seen_content.insert(hash);
        if let Some(ref emb) = embedding {
            if self.seen_embeddings.len() >= self.max_seen {
                self.seen_embeddings.pop_front();
            }
            self.seen_embeddings.push_back(emb.clone());
        }

        WriteGateDecision {
            allowed: true,
            salience_score: score,
            reason: "Allowed".into(),
            confidence: sig_confidence,
            novelty: sig_novelty,
            reliability: sig_reliability,
            recency: sig_recency,
            relevance: sig_relevance,
        }
    }

    /// Number of allowed writes.
    #[getter]
    pub fn write_count(&self) -> u64 {
        self.write_count
    }

    /// Number of blocked writes.
    #[getter]
    pub fn abstention_count(&self) -> u64 {
        self.abstention_count
    }

    /// Abstention rate (0.0 to 1.0).
    pub fn abstention_rate(&self) -> f32 {
        let total = self.write_count + self.abstention_count;
        if total == 0 {
            return 0.0;
        }
        self.abstention_count as f32 / total as f32
    }

    /// Statistics dict.
    pub fn stats(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("writes".into(), self.write_count as f64);
        m.insert("abstentions".into(), self.abstention_count as f64);
        m.insert("abstention_rate".into(), self.abstention_rate() as f64);
        m.insert("threshold".into(), self.threshold as f64);
        m.insert("seen_embeddings".into(), self.seen_embeddings.len() as f64);
        m
    }
}

impl RustCompositeWriteGate {
    /// Compute novelty as 1 - max_cosine_similarity to seen embeddings.
    fn compute_novelty(&self, embedding: &Option<Vec<f32>>) -> f32 {
        let emb = match embedding {
            Some(e) if !e.is_empty() => e,
            _ => return 1.0, // No embedding → fully novel
        };

        if self.seen_embeddings.is_empty() {
            return 1.0;
        }

        let norm_a = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        let mut max_sim: f32 = 0.0;

        for seen in &self.seen_embeddings {
            let dot: f32 = emb.iter().zip(seen.iter()).map(|(a, b)| a * b).sum();
            let norm_b = seen.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            let sim = dot / (norm_a * norm_b);
            max_sim = max_sim.max(sim);
        }

        if max_sim > self.novelty_sim_threshold {
            return 0.0; // Near-duplicate
        }

        1.0 - max_sim
    }

    /// Keyword overlap scoring (same logic as RustRelevanceGate).
    fn keyword_overlap(&self, task: &str, content: &str) -> f32 {
        let task_tokens = self.tokenize(task);
        if task_tokens.is_empty() {
            return 0.5;
        }
        let content_tokens = self.tokenize(content);
        let overlap = task_tokens.intersection(&content_tokens).count();
        overlap as f32 / task_tokens.len() as f32
    }

    /// Extract meaningful lowercase tokens (length >= 3, no stop words).
    fn tokenize(&self, text: &str) -> HashSet<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| w.len() >= 3 && !self.stop_words.contains(w))
            .map(String::from)
            .collect()
    }
}

/// FNV-1a hash for fast exact dedup.
fn fnv_hash(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_content_blocked() {
        let mut gate =
            RustCompositeWriteGate::new(0.35, 0.25, 0.30, 0.20, 0.10, 0.15, 0.90, 300.0, 200);
        let d = gate.evaluate("", 0.9, "", "unknown", None);
        assert!(!d.allowed);
        assert!(d.reason.contains("empty"));
    }

    #[test]
    fn test_duplicate_blocked() {
        let mut gate =
            RustCompositeWriteGate::new(0.35, 0.25, 0.30, 0.20, 0.10, 0.15, 0.90, 300.0, 200);
        let d1 = gate.evaluate("hello world", 0.9, "", "unknown", None);
        assert!(d1.allowed);
        let d2 = gate.evaluate("hello world", 0.9, "", "unknown", None);
        assert!(!d2.allowed);
        assert!(d2.reason.contains("duplicate"));
    }

    #[test]
    fn test_high_confidence_passes() {
        let mut gate =
            RustCompositeWriteGate::new(0.3, 0.25, 0.30, 0.20, 0.10, 0.15, 0.90, 300.0, 200);
        let d = gate.evaluate(
            "novel content about algorithms",
            0.9,
            "implement algorithms",
            "codex",
            None,
        );
        assert!(d.allowed);
        assert!(d.salience_score > 0.3);
    }

    #[test]
    fn test_novelty_blocks_similar_embeddings() {
        let mut gate =
            RustCompositeWriteGate::new(0.5, 0.25, 0.30, 0.20, 0.10, 0.15, 0.90, 300.0, 200);
        let emb1 = vec![1.0, 0.0, 0.0];
        let emb2 = vec![0.99, 0.01, 0.0]; // Very similar

        gate.evaluate("first entry", 0.9, "", "unknown", Some(emb1));
        let d = gate.evaluate("second entry", 0.9, "", "unknown", Some(emb2));
        assert!(d.novelty < 0.1);
    }

    #[test]
    fn test_novelty_high_for_orthogonal() {
        let mut gate =
            RustCompositeWriteGate::new(0.1, 0.25, 0.30, 0.20, 0.10, 0.15, 0.90, 300.0, 200);
        let emb1 = vec![1.0, 0.0, 0.0];
        let emb2 = vec![0.0, 1.0, 0.0];

        gate.evaluate("first", 0.9, "", "unknown", Some(emb1));
        let d = gate.evaluate("second", 0.9, "", "unknown", Some(emb2));
        assert!((d.novelty - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_reliability_tiers() {
        assert!(default_reliability("codex") > default_reliability("budget"));
        assert!(default_reliability("reasoner") > default_reliability("fast"));
        assert_eq!(default_reliability("unknown_tier"), 0.60);
    }

    #[test]
    fn test_keyword_overlap() {
        let gate =
            RustCompositeWriteGate::new(0.35, 0.25, 0.30, 0.20, 0.10, 0.15, 0.90, 300.0, 200);
        let score = gate.keyword_overlap(
            "implement quicksort algorithm",
            "quicksort implementation done",
        );
        assert!(score > 0.3);
    }

    #[test]
    fn test_stats() {
        let mut gate =
            RustCompositeWriteGate::new(0.01, 0.25, 0.30, 0.20, 0.10, 0.15, 0.90, 300.0, 200);
        gate.evaluate("a", 0.9, "", "unknown", None);
        gate.evaluate("b", 0.9, "", "unknown", None);
        let s = gate.stats();
        assert_eq!(s["writes"], 2.0);
    }

    #[test]
    fn test_fnv_hash_deterministic() {
        assert_eq!(fnv_hash("hello"), fnv_hash("hello"));
        assert_ne!(fnv_hash("hello"), fnv_hash("world"));
    }
}
