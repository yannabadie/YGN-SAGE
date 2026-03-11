//! Multi-View Semantic Memory Management Unit (S-MMU)
//!
//! Replaces the old single-graph SemanticMMU with 4 orthogonal graphs:
//! - **Temporal**: chronological links between sequential chunks, weighted by time proximity
//! - **Semantic**: similarity links using cosine similarity on embeddings
//! - **Causal**: parent-child agent causality links
//! - **Entity**: shared keyword/entity links using Jaccard similarity

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

/// Metadata stored per compacted chunk in the S-MMU.
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    pub chunk_id: usize,
    pub start_time: i64,
    pub end_time: i64,
    pub summary: String,
    /// Optional embedding vector for semantic similarity.
    pub embedding: Option<Vec<f32>>,
    /// Keywords / entity tags for entity-graph linking.
    pub keywords: Vec<String>,
    /// Parent chunk ID for causal linking (e.g. parent agent's chunk).
    pub parent_chunk_id: Option<usize>,
}

/// Edge label identifying which graph view an edge belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeKind {
    Temporal,
    Semantic,
    Causal,
    Entity,
}

/// A weighted, labeled edge in the multi-view graph.
#[derive(Debug, Clone)]
pub struct MultiEdge {
    pub kind: EdgeKind,
    pub weight: f32,
}

/// Maximum number of recent chunks to compare against when building
/// semantic and entity edges. Limits `register_chunk()` from O(n) to O(K).
///
/// At K=128 the cost per registration is bounded regardless of total chunk
/// count, while still capturing the most temporally-relevant neighbors
/// (recent chunks are most likely to be semantically related in an agent
/// conversation).
pub const MAX_SEMANTIC_NEIGHBORS: usize = 128;

/// Multi-View S-MMU: 4 orthogonal views stored in a single DiGraph.
///
/// Each edge carries a `MultiEdge` that identifies its view (temporal, semantic,
/// causal, entity) and its weight. This allows unified traversal while keeping
/// the views logically separate.
#[derive(Debug, Clone)]
pub struct MultiViewMMU {
    pub graph: DiGraph<ChunkMetadata, MultiEdge>,
    pub chunk_map: HashMap<usize, NodeIndex>,
    /// Insertion-ordered ring of chunk IDs for recency-bounded scans.
    recent_ids: VecDeque<usize>,
    next_chunk_id: usize,
}

impl Default for MultiViewMMU {
    fn default() -> Self {
        Self {
            graph: DiGraph::new(),
            chunk_map: HashMap::new(),
            recent_ids: VecDeque::new(),
            next_chunk_id: 0,
        }
    }
}

impl MultiViewMMU {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of chunks registered in the S-MMU.
    pub fn chunk_count(&self) -> usize {
        self.chunk_map.len()
    }

    /// Register a new chunk and build all applicable edges.
    pub fn register_chunk(
        &mut self,
        start_time: i64,
        end_time: i64,
        summary: &str,
        keywords: Vec<String>,
        embedding: Option<Vec<f32>>,
        parent_chunk_id: Option<usize>,
    ) -> usize {
        let id = self.next_chunk_id;
        self.next_chunk_id += 1;

        let meta = ChunkMetadata {
            chunk_id: id,
            start_time,
            end_time,
            summary: summary.to_string(),
            embedding: embedding.clone(),
            keywords: keywords.clone(),
            parent_chunk_id,
        };

        let node_idx = self.graph.add_node(meta);
        self.chunk_map.insert(id, node_idx);

        // --- Temporal edges ---
        // Link to previous chunk (chronological sequence).
        if id > 0 {
            if let Some(&prev_idx) = self.chunk_map.get(&(id - 1)) {
                // Weight: time proximity = 1 / (1 + |gap|) where gap is in nanoseconds.
                let prev_end = self.graph[prev_idx].end_time;
                let gap = (start_time - prev_end).unsigned_abs() as f64;
                let weight = 1.0 / (1.0 + gap / 1_000_000_000.0); // normalise to seconds
                self.graph.add_edge(
                    prev_idx,
                    node_idx,
                    MultiEdge {
                        kind: EdgeKind::Temporal,
                        weight: weight as f32,
                    },
                );
            }
        }

        // --- Semantic edges ---
        // Compare against the most recent MAX_SEMANTIC_NEIGHBORS chunks only.
        // This bounds the cost from O(n) to O(K) per registration while still
        // capturing the most temporally-relevant neighbors (recent chunks are
        // most likely to be semantically related in an agent conversation).
        if let Some(ref emb) = embedding {
            let scan_count = self.recent_ids.len().min(MAX_SEMANTIC_NEIGHBORS);
            for i in 0..scan_count {
                // Walk backwards from the most recent entry.
                let cid = self.recent_ids[self.recent_ids.len() - 1 - i];
                if cid == id {
                    continue;
                }
                let nidx = match self.chunk_map.get(&cid) {
                    Some(&idx) => idx,
                    None => continue,
                };
                if let Some(ref other_emb) = self.graph[nidx].embedding {
                    let sim = cosine_similarity(emb, other_emb);
                    if sim > 0.5 {
                        self.graph.add_edge(
                            node_idx,
                            nidx,
                            MultiEdge {
                                kind: EdgeKind::Semantic,
                                weight: sim,
                            },
                        );
                        self.graph.add_edge(
                            nidx,
                            node_idx,
                            MultiEdge {
                                kind: EdgeKind::Semantic,
                                weight: sim,
                            },
                        );
                    }
                }
            }
        }

        // --- Causal edges ---
        // Link from parent chunk to this chunk.
        if let Some(pcid) = parent_chunk_id {
            if let Some(&parent_idx) = self.chunk_map.get(&pcid) {
                self.graph.add_edge(
                    parent_idx,
                    node_idx,
                    MultiEdge {
                        kind: EdgeKind::Causal,
                        weight: 1.0,
                    },
                );
            }
        }

        // --- Entity edges ---
        // Compare against the most recent MAX_SEMANTIC_NEIGHBORS chunks only
        // (same recency bound as semantic edges — O(K) instead of O(n)).
        if !keywords.is_empty() {
            let kw_set: HashSet<&str> = keywords.iter().map(|s| s.as_str()).collect();
            let scan_count = self.recent_ids.len().min(MAX_SEMANTIC_NEIGHBORS);
            for i in 0..scan_count {
                let cid = self.recent_ids[self.recent_ids.len() - 1 - i];
                if cid == id {
                    continue;
                }
                let nidx = match self.chunk_map.get(&cid) {
                    Some(&idx) => idx,
                    None => continue,
                };
                let other_kw = &self.graph[nidx].keywords;
                if other_kw.is_empty() {
                    continue;
                }
                let other_set: HashSet<&str> = other_kw.iter().map(|s| s.as_str()).collect();
                let jaccard = jaccard_similarity(&kw_set, &other_set);
                if jaccard > 0.0 {
                    self.graph.add_edge(
                        node_idx,
                        nidx,
                        MultiEdge {
                            kind: EdgeKind::Entity,
                            weight: jaccard,
                        },
                    );
                    self.graph.add_edge(
                        nidx,
                        node_idx,
                        MultiEdge {
                            kind: EdgeKind::Entity,
                            weight: jaccard,
                        },
                    );
                }
            }
        }

        // Track insertion order for recency-bounded scans.
        self.recent_ids.push_back(id);

        id
    }

    /// Retrieve chunks relevant to `active_chunk_id` by walking the multi-view
    /// graph up to `max_hops` hops.
    ///
    /// `weights` = `[temporal, semantic, causal, entity]` weighting factors
    /// (default `[1.0, 1.0, 1.0, 1.0]`).
    ///
    /// Returns `(chunk_id, aggregated_score)` sorted descending by score.
    pub fn retrieve_relevant(
        &self,
        active_chunk_id: usize,
        max_hops: usize,
        weights: [f32; 4],
    ) -> Vec<(usize, f32)> {
        let start_idx = match self.chunk_map.get(&active_chunk_id) {
            Some(&idx) => idx,
            None => return Vec::new(),
        };

        // BFS up to max_hops, accumulating scores.
        let mut scores: HashMap<usize, f32> = HashMap::new();
        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut frontier: Vec<(NodeIndex, f32, usize)> = vec![(start_idx, 1.0, 0)];
        visited.insert(start_idx);

        while let Some((node, incoming_score, depth)) = frontier.pop() {
            if depth >= max_hops {
                continue;
            }
            for edge_ref in self.graph.edges(node) {
                let target = edge_ref.target();
                let me = edge_ref.weight();
                let view_weight = match me.kind {
                    EdgeKind::Temporal => weights[0],
                    EdgeKind::Semantic => weights[1],
                    EdgeKind::Causal => weights[2],
                    EdgeKind::Entity => weights[3],
                };
                let propagated = incoming_score * me.weight * view_weight;
                let target_cid = self.graph[target].chunk_id;
                if target_cid != active_chunk_id {
                    *scores.entry(target_cid).or_insert(0.0) += propagated;
                }
                if !visited.contains(&target) {
                    visited.insert(target);
                    frontier.push((target, propagated, depth + 1));
                }
            }
        }

        let mut result: Vec<(usize, f32)> = scores.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn jaccard_similarity(a: &HashSet<&str>, b: &HashSet<&str>) -> f32 {
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

/// Python-facing wrapper for `MultiViewMMU`.
#[pyclass(name = "MultiViewMMU")]
pub struct PyMultiViewMMU {
    inner: MultiViewMMU,
}

#[pymethods]
impl PyMultiViewMMU {
    #[new]
    fn new() -> Self {
        Self {
            inner: MultiViewMMU::new(),
        }
    }

    /// Register a new chunk and build all applicable edges.
    /// Returns the chunk ID.
    fn register_chunk(
        &mut self,
        start_time: i64,
        end_time: i64,
        summary: &str,
        keywords: Vec<String>,
        embedding: Option<Vec<f32>>,
        parent_chunk_id: Option<usize>,
    ) -> usize {
        self.inner.register_chunk(
            start_time,
            end_time,
            summary,
            keywords,
            embedding,
            parent_chunk_id,
        )
    }

    /// Number of chunks registered in the S-MMU.
    fn chunk_count(&self) -> usize {
        self.inner.chunk_count()
    }

    /// Get the summary string for a given chunk ID.
    fn get_chunk_summary(&self, chunk_id: usize) -> Option<String> {
        self.inner
            .chunk_map
            .get(&chunk_id)
            .map(|&idx| self.inner.graph[idx].summary.clone())
    }

    /// Retrieve chunks relevant to `chunk_id` via multi-view BFS (up to `max_hops`).
    /// Uses default view weights `[0.4, 0.3, 0.2, 0.1]`.
    /// Returns list of `(chunk_id, score)` sorted descending by score.
    fn retrieve_relevant(&self, chunk_id: usize, max_hops: usize) -> Vec<(usize, f32)> {
        self.inner
            .retrieve_relevant(chunk_id, max_hops, [0.4, 0.3, 0.2, 0.1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 0.0, 1.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_jaccard_full_overlap() {
        let a: HashSet<&str> = ["x", "y"].iter().copied().collect();
        let sim = jaccard_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_jaccard_no_overlap() {
        let a: HashSet<&str> = ["x"].iter().copied().collect();
        let b: HashSet<&str> = ["y"].iter().copied().collect();
        let sim = jaccard_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    /// Verify that semantic edges are still created correctly under the
    /// recency-bounded scan (within MAX_SEMANTIC_NEIGHBORS).
    #[test]
    fn test_bounded_semantic_edges_within_limit() {
        let mut smmu = MultiViewMMU::new();

        // Create two chunks with identical embeddings — should get semantic edges.
        let emb = vec![1.0, 0.0, 0.5];
        let a = smmu.register_chunk(0, 1, "A", vec![], Some(emb.clone()), None);
        let b = smmu.register_chunk(2, 3, "B", vec![], Some(emb.clone()), None);

        // From A, B should be reachable via a semantic edge.
        let results = smmu.retrieve_relevant(a, 1, [0.0, 1.0, 0.0, 0.0]);
        let b_entry = results.iter().find(|(id, _)| *id == b);
        assert!(
            b_entry.is_some(),
            "B should be reachable from A via semantic edge"
        );
        assert!(
            (b_entry.unwrap().1 - 1.0).abs() < 1e-5,
            "Identical embeddings should yield weight ~1.0"
        );
    }

    /// Verify that entity edges are still created correctly under the
    /// recency-bounded scan.
    #[test]
    fn test_bounded_entity_edges_within_limit() {
        let mut smmu = MultiViewMMU::new();

        let a = smmu.register_chunk(0, 1, "A", vec!["rust".into(), "memory".into()], None, None);
        let b = smmu.register_chunk(2, 3, "B", vec!["rust".into(), "graph".into()], None, None);

        // From A, B should be reachable via entity edge (shared keyword "rust").
        let results = smmu.retrieve_relevant(a, 1, [0.0, 0.0, 0.0, 1.0]);
        let b_entry = results.iter().find(|(id, _)| *id == b);
        assert!(
            b_entry.is_some(),
            "B should be reachable from A via entity edge"
        );
        // Jaccard("rust","memory" vs "rust","graph") = 1/3
        let expected_jaccard = 1.0 / 3.0;
        assert!(
            (b_entry.unwrap().1 - expected_jaccard).abs() < 1e-5,
            "Entity edge weight should be Jaccard similarity"
        );
    }

    /// Beyond MAX_SEMANTIC_NEIGHBORS, old chunks should NOT receive new
    /// semantic/entity edges from the newly registered chunk.
    #[test]
    fn test_bounded_scan_skips_old_chunks() {
        let mut smmu = MultiViewMMU::new();

        // Register chunk 0 with a known embedding.
        let emb = vec![1.0, 0.0, 0.5];
        let first = smmu.register_chunk(
            0,
            1,
            "first",
            vec!["shared".into()],
            Some(emb.clone()),
            None,
        );

        // Fill up MAX_SEMANTIC_NEIGHBORS + 10 intermediate chunks (no embedding,
        // no shared keywords) to push chunk 0 outside the recency window.
        for i in 1..=(MAX_SEMANTIC_NEIGHBORS + 10) {
            smmu.register_chunk(
                (i * 2) as i64,
                (i * 2 + 1) as i64,
                &format!("filler-{i}"),
                vec![format!("unique-{i}")],
                None,
                None,
            );
        }

        // Now register a chunk with the SAME embedding and keyword as chunk 0.
        let last = smmu.register_chunk(
            999_000,
            999_001,
            "last",
            vec!["shared".into()],
            Some(emb.clone()),
            None,
        );

        // `last` should NOT have a direct semantic or entity edge to `first`
        // because `first` is outside the recency window.
        // We check by retrieving from `last` with 1 hop, semantic-only.
        let sem_results = smmu.retrieve_relevant(last, 1, [0.0, 1.0, 0.0, 0.0]);
        let first_sem = sem_results.iter().find(|(id, _)| *id == first);
        assert!(
            first_sem.is_none(),
            "Chunk 0 should be outside the recency window and have no direct semantic edge to last chunk"
        );

        // Same check for entity edges.
        let ent_results = smmu.retrieve_relevant(last, 1, [0.0, 0.0, 0.0, 1.0]);
        let first_ent = ent_results.iter().find(|(id, _)| *id == first);
        assert!(
            first_ent.is_none(),
            "Chunk 0 should be outside the recency window and have no direct entity edge to last chunk"
        );
    }

    /// Verify that the constant MAX_SEMANTIC_NEIGHBORS is accessible and
    /// reasonable.
    #[test]
    fn test_max_semantic_neighbors_constant() {
        assert!(
            MAX_SEMANTIC_NEIGHBORS >= 16,
            "MAX_SEMANTIC_NEIGHBORS should be at least 16"
        );
        assert!(
            MAX_SEMANTIC_NEIGHBORS <= 1024,
            "MAX_SEMANTIC_NEIGHBORS should not be excessively large"
        );
    }

    /// Performance regression guard: registering 500 chunks with embeddings
    /// should complete quickly because we only scan the most recent K, not all n.
    #[test]
    fn test_register_500_chunks_bounded_time() {
        let mut smmu = MultiViewMMU::new();
        let emb = vec![0.5_f32; 384]; // 384-dim like all-MiniLM-L6-v2

        let start = std::time::Instant::now();
        for i in 0..500 {
            smmu.register_chunk(
                (i * 100) as i64,
                (i * 100 + 50) as i64,
                &format!("chunk-{i}"),
                vec!["common".into(), format!("tag-{}", i % 10)],
                Some(emb.clone()),
                None,
            );
        }
        let elapsed = start.elapsed();

        assert_eq!(smmu.chunk_count(), 500);
        // With O(K) scan (K=128), 500 registrations should be well under 5s
        // even on slow CI. An O(n^2) scan of 500 * 384-dim vectors would be
        // noticeably slower.
        assert!(
            elapsed.as_secs() < 5,
            "500 registrations took {:?}, expected < 5s with bounded scan",
            elapsed
        );
    }
}
