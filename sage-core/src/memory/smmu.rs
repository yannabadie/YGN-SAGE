//! Multi-View Semantic Memory Management Unit (S-MMU)
//!
//! Replaces the old single-graph SemanticMMU with 4 orthogonal graphs:
//! - **Temporal**: chronological links between sequential chunks, weighted by time proximity
//! - **Semantic**: similarity links using cosine similarity on embeddings
//! - **Causal**: parent-child agent causality links
//! - **Entity**: shared keyword/entity links using Jaccard similarity

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};

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

/// Multi-View S-MMU: 4 orthogonal views stored in a single DiGraph.
///
/// Each edge carries a `MultiEdge` that identifies its view (temporal, semantic,
/// causal, entity) and its weight. This allows unified traversal while keeping
/// the views logically separate.
#[derive(Debug, Clone)]
pub struct MultiViewMMU {
    pub graph: DiGraph<ChunkMetadata, MultiEdge>,
    pub chunk_map: HashMap<usize, NodeIndex>,
    next_chunk_id: usize,
}

impl Default for MultiViewMMU {
    fn default() -> Self {
        Self {
            graph: DiGraph::new(),
            chunk_map: HashMap::new(),
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
        // Connect to any existing chunk whose embedding has cosine similarity > 0.5.
        if let Some(ref emb) = embedding {
            for (&cid, &nidx) in &self.chunk_map {
                if cid == id {
                    continue;
                }
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
        // Link to chunks sharing keywords (Jaccard similarity).
        if !keywords.is_empty() {
            let kw_set: HashSet<&str> = keywords.iter().map(|s| s.as_str()).collect();
            for (&cid, &nidx) in &self.chunk_map {
                if cid == id {
                    continue;
                }
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
}
