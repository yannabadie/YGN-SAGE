//! Topology-aware S-MMU bridge — stores and retrieves topology outcomes with
//! structural features, and injects bandit priors from similar past tasks.
//!
//! This is a deeper bridge than `routing::smmu_bridge::TopologyBridge`, which
//! only stores template + model_id. This bridge additionally tracks:
//! - `topology_id` (ULID referencing a `TopologyGraph`)
//! - Structural features: agent_count, max_depth, model_diversity_score
//! - Rich `TopologySuggestion` return values
//! - Bandit prior injection from retrieved suggestions

use crate::memory::smmu::MultiViewMMU;
use crate::routing::bandit::ContextualBandit;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, info_span};

// ── TopologyOutcome ─────────────────────────────────────────────────────────

/// Input data for recording a topology execution outcome in S-MMU.
///
/// Contains the task context (summary, keywords, embedding) plus the
/// topology outcome (topology_id, template, quality, cost, latency)
/// and structural features (agent_count, max_depth, model_diversity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyOutcome {
    /// ULID string referencing a `TopologyGraph`.
    pub topology_id: String,
    /// Human-readable task summary for S-MMU chunk.
    pub task_summary: String,
    /// Keywords / entity tags for entity-graph linking.
    pub keywords: Vec<String>,
    /// 384-dim embedding vector for semantic similarity.
    pub task_embedding: Option<Vec<f32>>,
    /// Template name (e.g., "sequential", "avr", "parallel").
    pub template: String,
    /// Quality score [0.0, 1.0].
    pub quality: f32,
    /// Cost in USD.
    pub cost: f32,
    /// Latency in milliseconds.
    pub latency_ms: f32,
    /// Number of agents in the topology.
    pub agent_count: u32,
    /// Maximum depth of the topology graph.
    pub max_depth: u32,
    /// Model diversity score [0.0, 1.0] — fraction of unique models.
    pub model_diversity: f32,
}

// ── OutcomeMeta ─────────────────────────────────────────────────────────────

/// Metadata stored alongside each S-MMU chunk, capturing topology-specific
/// data that the S-MMU itself does not track (it only stores summary/keywords/embedding).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeMeta {
    pub topology_id: String,
    pub template: String,
    pub quality: f32,
    pub cost: f32,
    pub latency_ms: f32,
    pub agent_count: u32,
    pub max_depth: u32,
    pub model_diversity: f32,
}

// ── TopologySuggestion ──────────────────────────────────────────────────────

/// A rich suggestion returned from S-MMU retrieval, combining topology
/// metadata with similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologySuggestion {
    /// ULID string referencing the original `TopologyGraph`.
    pub topology_id: String,
    /// Template name (e.g., "sequential", "avr", "parallel").
    pub template: String,
    /// Quality score [0.0, 1.0] from the original outcome.
    pub quality: f32,
    /// Cost in USD from the original outcome.
    pub cost: f32,
    /// Latency in milliseconds from the original outcome.
    pub latency_ms: f32,
    /// Semantic similarity score from S-MMU retrieval.
    pub similarity_score: f32,
    /// Number of agents in the suggested topology.
    pub agent_count: u32,
    /// Maximum depth of the suggested topology graph.
    pub max_depth: u32,
    /// Model diversity score [0.0, 1.0].
    pub model_diversity: f32,
}

// ── TopologySmmuBridge ──────────────────────────────────────────────────────

/// Deeper S-MMU bridge for topology routing — stores topology outcomes with
/// structural features and retrieves similar past tasks as rich suggestions.
///
/// Key differences from `routing::smmu_bridge::TopologyBridge`:
/// - Stores `topology_id` (ULID referencing a `TopologyGraph`)
/// - Includes structural features: agent_count, max_depth, model_diversity
/// - Returns `TopologySuggestion` with full metadata
/// - Has `inject_priors()` for bandit warm-start from similar tasks
pub struct TopologySmmuBridge {
    /// chunk_id -> topology-specific metadata.
    chunk_meta: HashMap<usize, OutcomeMeta>,
}

impl Default for TopologySmmuBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl TopologySmmuBridge {
    pub fn new() -> Self {
        Self {
            chunk_meta: HashMap::new(),
        }
    }

    /// Store a topology outcome in S-MMU.
    ///
    /// Registers the task context as an S-MMU chunk and stores the topology
    /// outcome metadata (topology_id, template, quality, cost, latency, structural
    /// features) locally. Returns the S-MMU chunk ID.
    pub fn record_outcome(
        &mut self,
        smmu: &mut MultiViewMMU,
        outcome: TopologyOutcome,
    ) -> usize {
        let _span = info_span!(
            "topology_smmu.record",
            topology_id = %outcome.topology_id,
            template = %outcome.template,
            quality = outcome.quality,
            agent_count = outcome.agent_count,
        )
        .entered();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let chunk_id = smmu.register_chunk(
            now,
            now,
            &outcome.task_summary,
            outcome.keywords,
            outcome.task_embedding,
            None,
        );

        let meta = OutcomeMeta {
            topology_id: outcome.topology_id.clone(),
            template: outcome.template.clone(),
            quality: outcome.quality,
            cost: outcome.cost,
            latency_ms: outcome.latency_ms,
            agent_count: outcome.agent_count,
            max_depth: outcome.max_depth,
            model_diversity: outcome.model_diversity,
        };

        self.chunk_meta.insert(chunk_id, meta);

        info!(
            chunk_id = chunk_id,
            topology_id = %outcome.topology_id,
            template = %outcome.template,
            quality = outcome.quality,
            "topology_outcome_recorded"
        );

        chunk_id
    }

    /// Query S-MMU for similar past tasks and return top-k topology suggestions.
    ///
    /// Uses semantic-heavy weights `[0.1, 0.7, 0.1, 0.1]` (temporal, semantic,
    /// causal, entity) since topology retrieval benefits most from semantic
    /// similarity between task descriptions.
    ///
    /// Returns `Vec<TopologySuggestion>` with full topology metadata.
    pub fn retrieve_similar(
        &self,
        smmu: &MultiViewMMU,
        query_chunk_id: usize,
        max_results: usize,
    ) -> Vec<TopologySuggestion> {
        let _span = info_span!(
            "topology_smmu.retrieve",
            query_chunk_id = query_chunk_id,
            max_results = max_results,
        )
        .entered();

        // Semantic weight heavily for topology retrieval
        let weights = [0.1, 0.7, 0.1, 0.1]; // temporal, semantic, causal, entity
        let results = smmu.retrieve_relevant(query_chunk_id, 2, weights);

        let suggestions: Vec<TopologySuggestion> = results
            .into_iter()
            .filter_map(|(chunk_id, score)| {
                self.chunk_meta.get(&chunk_id).map(|meta| TopologySuggestion {
                    topology_id: meta.topology_id.clone(),
                    template: meta.template.clone(),
                    quality: meta.quality,
                    cost: meta.cost,
                    latency_ms: meta.latency_ms,
                    similarity_score: score,
                    agent_count: meta.agent_count,
                    max_depth: meta.max_depth,
                    model_diversity: meta.model_diversity,
                })
            })
            .take(max_results)
            .collect();

        info!(
            found = suggestions.len(),
            "topology_suggestions_retrieved"
        );

        suggestions
    }

    /// Inject bandit priors from retrieved topology suggestions.
    ///
    /// For each suggestion with quality above a minimum threshold, adds the
    /// (model="suggested", template) arm to the bandit. This warm-starts
    /// exploration toward topologies that worked well for similar past tasks.
    ///
    /// The arm is registered with model_id "suggested" so the bandit can
    /// distinguish prior-injected arms from organically registered ones.
    pub fn inject_priors(
        &self,
        bandit: &mut ContextualBandit,
        suggestions: &[TopologySuggestion],
    ) {
        let _span = info_span!(
            "topology_smmu.inject_priors",
            suggestion_count = suggestions.len(),
        )
        .entered();

        let quality_threshold = 0.3;

        for suggestion in suggestions {
            if suggestion.quality >= quality_threshold {
                bandit.add_arm("suggested", &suggestion.template);
                info!(
                    template = %suggestion.template,
                    quality = suggestion.quality,
                    similarity = suggestion.similarity_score,
                    "bandit_prior_injected"
                );
            }
        }
    }

    /// Number of stored topology outcome chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunk_meta.len()
    }

    /// Get the metadata for a specific chunk ID (test/debug use).
    pub fn get_meta(&self, chunk_id: usize) -> Option<&OutcomeMeta> {
        self.chunk_meta.get(&chunk_id)
    }
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_outcome(
        topology_id: &str,
        summary: &str,
        template: &str,
        quality: f32,
        embedding: Option<Vec<f32>>,
    ) -> TopologyOutcome {
        TopologyOutcome {
            topology_id: topology_id.to_string(),
            task_summary: summary.to_string(),
            keywords: vec!["code".into(), "test".into()],
            task_embedding: embedding,
            template: template.to_string(),
            quality,
            cost: 0.01,
            latency_ms: 500.0,
            agent_count: 3,
            max_depth: 2,
            model_diversity: 0.67,
        }
    }

    #[test]
    fn test_new_bridge_is_empty() {
        let bridge = TopologySmmuBridge::new();
        assert_eq!(bridge.chunk_count(), 0);
    }

    #[test]
    fn test_default_is_empty() {
        let bridge = TopologySmmuBridge::default();
        assert_eq!(bridge.chunk_count(), 0);
    }

    #[test]
    fn test_record_increments_count() {
        let mut smmu = MultiViewMMU::new();
        let mut bridge = TopologySmmuBridge::new();

        let outcome = make_outcome(
            "01JTEST0001",
            "Sort an array",
            "avr",
            0.9,
            None,
        );
        bridge.record_outcome(&mut smmu, outcome);

        assert_eq!(bridge.chunk_count(), 1);
        assert_eq!(smmu.chunk_count(), 1);
    }

    #[test]
    fn test_record_returns_valid_chunk_id() {
        let mut smmu = MultiViewMMU::new();
        let mut bridge = TopologySmmuBridge::new();

        let outcome = make_outcome(
            "01JTEST0001",
            "Write quicksort",
            "sequential",
            0.8,
            Some(vec![1.0; 384]),
        );
        let id = bridge.record_outcome(&mut smmu, outcome);

        // First chunk should be ID 0
        assert_eq!(id, 0);
    }

    #[test]
    fn test_multiple_records() {
        let mut smmu = MultiViewMMU::new();
        let mut bridge = TopologySmmuBridge::new();

        for i in 0..5 {
            let outcome = make_outcome(
                &format!("01JTEST{:04}", i),
                &format!("Task {}", i),
                if i % 2 == 0 { "sequential" } else { "parallel" },
                0.5 + i as f32 * 0.1,
                None,
            );
            let id = bridge.record_outcome(&mut smmu, outcome);
            assert_eq!(id, i);
        }

        assert_eq!(bridge.chunk_count(), 5);
        assert_eq!(smmu.chunk_count(), 5);
    }

    #[test]
    fn test_retrieve_on_empty_bridge() {
        let smmu = MultiViewMMU::new();
        let bridge = TopologySmmuBridge::new();

        let results = bridge.retrieve_similar(&smmu, 0, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_retrieve_with_similar_embeddings() {
        let mut smmu = MultiViewMMU::new();
        let mut bridge = TopologySmmuBridge::new();

        let emb1 = vec![1.0; 384];
        let emb2 = vec![1.0; 384]; // Identical => cosine similarity = 1.0

        let outcome = make_outcome(
            "01JTEST0001",
            "Write a sorting function",
            "avr",
            0.9,
            Some(emb1),
        );
        bridge.record_outcome(&mut smmu, outcome);

        // Register a query chunk directly in S-MMU
        let query_id = smmu.register_chunk(
            0,
            0,
            "Sort an array",
            vec!["sort".into()],
            Some(emb2),
            None,
        );

        let results = bridge.retrieve_similar(&smmu, query_id, 5);
        assert!(!results.is_empty(), "Should find the similar task");
        assert_eq!(results[0].template, "avr");
        assert_eq!(results[0].topology_id, "01JTEST0001");
        assert!((results[0].quality - 0.9).abs() < 1e-5);
        assert!(results[0].similarity_score > 0.0);
    }

    #[test]
    fn test_structural_features_stored() {
        let mut smmu = MultiViewMMU::new();
        let mut bridge = TopologySmmuBridge::new();

        let outcome = TopologyOutcome {
            topology_id: "01JTEST_STRUCT".to_string(),
            task_summary: "Complex multi-agent pipeline".to_string(),
            keywords: vec!["pipeline".into()],
            task_embedding: None,
            template: "parallel".to_string(),
            quality: 0.85,
            cost: 0.05,
            latency_ms: 2000.0,
            agent_count: 5,
            max_depth: 3,
            model_diversity: 0.8,
        };
        let chunk_id = bridge.record_outcome(&mut smmu, outcome);

        let meta = bridge.get_meta(chunk_id).expect("meta should exist");
        assert_eq!(meta.topology_id, "01JTEST_STRUCT");
        assert_eq!(meta.agent_count, 5);
        assert_eq!(meta.max_depth, 3);
        assert!((meta.model_diversity - 0.8).abs() < 1e-5);
        assert!((meta.quality - 0.85).abs() < 1e-5);
        assert!((meta.cost - 0.05).abs() < 1e-5);
        assert!((meta.latency_ms - 2000.0).abs() < 1e-1);
    }

    #[test]
    fn test_inject_priors_adds_arms() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        let bridge = TopologySmmuBridge::new();

        assert_eq!(bandit.arm_count(), 0);

        let suggestions = vec![
            TopologySuggestion {
                topology_id: "01JTEST0001".to_string(),
                template: "avr".to_string(),
                quality: 0.9,
                cost: 0.01,
                latency_ms: 500.0,
                similarity_score: 0.95,
                agent_count: 3,
                max_depth: 2,
                model_diversity: 0.67,
            },
            TopologySuggestion {
                topology_id: "01JTEST0002".to_string(),
                template: "parallel".to_string(),
                quality: 0.8,
                cost: 0.02,
                latency_ms: 800.0,
                similarity_score: 0.85,
                agent_count: 4,
                max_depth: 2,
                model_diversity: 0.75,
            },
        ];

        bridge.inject_priors(&mut bandit, &suggestions);

        // Both suggestions have quality >= 0.3 threshold, so both arms should be added
        assert_eq!(bandit.arm_count(), 2);
    }

    #[test]
    fn test_inject_priors_filters_low_quality() {
        let mut bandit = ContextualBandit::create(0.995, 0.1);
        let bridge = TopologySmmuBridge::new();

        let suggestions = vec![
            TopologySuggestion {
                topology_id: "01JTEST_GOOD".to_string(),
                template: "avr".to_string(),
                quality: 0.9,
                cost: 0.01,
                latency_ms: 500.0,
                similarity_score: 0.95,
                agent_count: 3,
                max_depth: 2,
                model_diversity: 0.67,
            },
            TopologySuggestion {
                topology_id: "01JTEST_BAD".to_string(),
                template: "sequential".to_string(),
                quality: 0.1, // Below 0.3 threshold
                cost: 0.05,
                latency_ms: 3000.0,
                similarity_score: 0.7,
                agent_count: 1,
                max_depth: 1,
                model_diversity: 0.0,
            },
        ];

        bridge.inject_priors(&mut bandit, &suggestions);

        // Only the good suggestion should be injected
        assert_eq!(bandit.arm_count(), 1);
    }

    #[test]
    fn test_suggestion_fields_populated() {
        let mut smmu = MultiViewMMU::new();
        let mut bridge = TopologySmmuBridge::new();

        let emb = vec![1.0; 384];
        let outcome = TopologyOutcome {
            topology_id: "01JTEST_FULL".to_string(),
            task_summary: "Full-featured task".to_string(),
            keywords: vec!["full".into(), "test".into()],
            task_embedding: Some(emb.clone()),
            template: "avr".to_string(),
            quality: 0.92,
            cost: 0.03,
            latency_ms: 1200.0,
            agent_count: 4,
            max_depth: 3,
            model_diversity: 0.75,
        };
        bridge.record_outcome(&mut smmu, outcome);

        let query_id = smmu.register_chunk(
            0,
            0,
            "Similar task",
            vec!["test".into()],
            Some(emb),
            None,
        );

        let suggestions = bridge.retrieve_similar(&smmu, query_id, 5);
        assert!(!suggestions.is_empty());

        let s = &suggestions[0];
        assert_eq!(s.topology_id, "01JTEST_FULL");
        assert_eq!(s.template, "avr");
        assert!((s.quality - 0.92).abs() < 1e-5);
        assert!((s.cost - 0.03).abs() < 1e-5);
        assert!((s.latency_ms - 1200.0).abs() < 1e-1);
        assert_eq!(s.agent_count, 4);
        assert_eq!(s.max_depth, 3);
        assert!((s.model_diversity - 0.75).abs() < 1e-5);
        assert!(s.similarity_score > 0.0);
    }
}
