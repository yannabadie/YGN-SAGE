//! S-MMU bridge for topology routing — stores and retrieves task→topology mappings.
//!
//! The S-MMU lives in `memory::smmu` and the routing system lives in `routing::`.
//! This bridge coordinates between them, providing topology-specific helpers for
//! recording outcomes and retrieving similar past tasks.

use crate::memory::smmu::MultiViewMMU;
use std::collections::HashMap;

/// Input data for recording a topology outcome in S-MMU.
///
/// Contains the task context (summary, keywords, embedding) plus the
/// topology outcome (template, model, quality, cost, latency).
pub struct TopologyChunk {
    pub task_summary: String,
    pub keywords: Vec<String>,
    pub embedding: Option<Vec<f32>>,
    pub template: String,
    pub model_id: String,
    pub quality: f32,
    pub cost: f32,
    pub latency_ms: f32,
}

/// Metadata stored alongside each S-MMU chunk, capturing the routing outcome
/// that the S-MMU itself does not track (it only stores summary/keywords/embedding).
#[derive(Debug, Clone)]
#[allow(dead_code)] // cost/latency stored for future cost-aware retrieval
struct ChunkMeta {
    template: String,
    model_id: String,
    quality: f32,
    cost: f32,
    latency_ms: f32,
}

/// Bridge between the routing system and S-MMU.
///
/// Maps S-MMU chunk IDs to topology-specific metadata (template, model, quality,
/// cost, latency). Provides methods to record routing outcomes and retrieve
/// similar past tasks for informed routing decisions.
#[deprecated(since = "0.2.0", note = "Use Python routing/shadow.py ShadowRouter for routing-S-MMU integration")]
pub struct TopologyBridge {
    /// chunk_id → topology metadata (S-MMU only stores summary/keywords/embedding).
    chunk_meta: HashMap<String, ChunkMeta>,
}

impl Default for TopologyBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl TopologyBridge {
    pub fn new() -> Self {
        Self {
            chunk_meta: HashMap::new(),
        }
    }

    /// Store a topology outcome in S-MMU.
    ///
    /// Registers the task context as an S-MMU chunk and stores the routing
    /// outcome metadata (template, model, quality, cost, latency) locally.
    /// Returns the S-MMU chunk ID.
    pub fn record_outcome(&mut self, smmu: &mut MultiViewMMU, chunk: TopologyChunk) -> String {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let chunk_id = smmu.register_chunk(
            now,
            now,
            &chunk.task_summary,
            chunk.keywords,
            chunk.embedding,
            None,
        );

        self.chunk_meta.insert(
            chunk_id.clone(),
            ChunkMeta {
                template: chunk.template,
                model_id: chunk.model_id,
                quality: chunk.quality,
                cost: chunk.cost,
                latency_ms: chunk.latency_ms,
            },
        );

        chunk_id
    }

    /// Query S-MMU for similar past tasks and return top-k topology recommendations.
    ///
    /// Uses semantic-heavy weights `[0.1, 0.7, 0.1, 0.1]` (temporal, semantic,
    /// causal, entity) since topology retrieval benefits most from semantic
    /// similarity between task descriptions.
    ///
    /// Returns `Vec<(template, model_id, quality, similarity_score)>`.
    pub fn retrieve_similar(
        &self,
        smmu: &MultiViewMMU,
        query_chunk_id: &str,
        max_results: usize,
    ) -> Vec<(String, String, f32, f32)> {
        // Semantic weight heavily for topology retrieval
        let weights = [0.1, 0.7, 0.1, 0.1]; // temporal, semantic, causal, entity
        let results = smmu.retrieve_relevant(query_chunk_id, 2, weights);

        results
            .into_iter()
            .filter_map(|(chunk_id, score)| {
                self.chunk_meta.get(&chunk_id).map(|meta| {
                    (
                        meta.template.clone(),
                        meta.model_id.clone(),
                        meta.quality,
                        score,
                    )
                })
            })
            .take(max_results)
            .collect()
    }

    /// Number of stored topology chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunk_meta.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_bridge_is_empty() {
        let bridge = TopologyBridge::new();
        assert_eq!(bridge.chunk_count(), 0);
    }

    #[test]
    fn test_default_is_empty() {
        let bridge = TopologyBridge::default();
        assert_eq!(bridge.chunk_count(), 0);
    }

    #[test]
    fn test_record_increments_count() {
        let mut smmu = MultiViewMMU::new();
        let mut bridge = TopologyBridge::new();

        bridge.record_outcome(
            &mut smmu,
            TopologyChunk {
                task_summary: "Sort an array".into(),
                keywords: vec!["code".into(), "sort".into()],
                embedding: None,
                template: "avr".into(),
                model_id: "model-a".into(),
                quality: 0.9,
                cost: 0.01,
                latency_ms: 1000.0,
            },
        );

        assert_eq!(bridge.chunk_count(), 1);
        assert_eq!(smmu.chunk_count(), 1);
    }

    #[test]
    fn test_record_returns_valid_ulid() {
        let mut smmu = MultiViewMMU::new();
        let mut bridge = TopologyBridge::new();

        let id = bridge.record_outcome(
            &mut smmu,
            TopologyChunk {
                task_summary: "Write quicksort".into(),
                keywords: vec!["code".into()],
                embedding: Some(vec![1.0; 384]),
                template: "sequential".into(),
                model_id: "model-a".into(),
                quality: 0.8,
                cost: 0.005,
                latency_ms: 500.0,
            },
        );

        // Chunk ID should be a 26-char ULID string
        assert_eq!(id.len(), 26);
    }

    #[test]
    fn test_multiple_records() {
        let mut smmu = MultiViewMMU::new();
        let mut bridge = TopologyBridge::new();
        let mut ids = Vec::new();

        for i in 0..5 {
            let id = bridge.record_outcome(
                &mut smmu,
                TopologyChunk {
                    task_summary: format!("Task {}", i),
                    keywords: vec![format!("keyword_{}", i)],
                    embedding: None,
                    template: if i % 2 == 0 {
                        "sequential".into()
                    } else {
                        "parallel".into()
                    },
                    model_id: "model".into(),
                    quality: 0.5 + i as f32 * 0.1,
                    cost: 0.01,
                    latency_ms: 500.0,
                },
            );
            ids.push(id);
        }

        // All IDs should be unique ULIDs
        let unique: std::collections::HashSet<&str> = ids.iter().map(|s| s.as_str()).collect();
        assert_eq!(unique.len(), 5);
        assert_eq!(bridge.chunk_count(), 5);
        assert_eq!(smmu.chunk_count(), 5);
    }

    #[test]
    fn test_retrieve_on_empty_bridge() {
        let smmu = MultiViewMMU::new();
        let bridge = TopologyBridge::new();

        let results = bridge.retrieve_similar(&smmu, "nonexistent", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_retrieve_with_similar_embeddings() {
        let mut smmu = MultiViewMMU::new();
        let mut bridge = TopologyBridge::new();

        let emb1 = vec![1.0; 384];
        let emb2 = vec![1.0; 384]; // Identical embedding => cosine sim = 1.0

        // Record a topology outcome
        bridge.record_outcome(
            &mut smmu,
            TopologyChunk {
                task_summary: "Write a sorting function".into(),
                keywords: vec!["code".into(), "sort".into()],
                embedding: Some(emb1),
                template: "avr".into(),
                model_id: "model-a".into(),
                quality: 0.9,
                cost: 0.01,
                latency_ms: 1000.0,
            },
        );

        // Register a query chunk directly in S-MMU
        let query_id =
            smmu.register_chunk(0, 0, "Sort an array", vec!["sort".into()], Some(emb2), None);

        let results = bridge.retrieve_similar(&smmu, &query_id, 5);
        // Should find the similar task via semantic edge
        assert!(!results.is_empty(), "Should find the similar task");
        assert_eq!(results[0].0, "avr"); // template
        assert_eq!(results[0].1, "model-a"); // model_id
        assert!((results[0].2 - 0.9).abs() < 1e-5); // quality
        assert!(results[0].3 > 0.0); // similarity score > 0
    }
}
