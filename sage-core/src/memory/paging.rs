//! Semantic Paging: identifies chunks that can be paged out (evicted)
//! from the working set based on their distance from the active chunk
//! in the multi-view graph.
//!
//! The algorithm walks the S-MMU graph from the active chunk and returns
//! the *least* relevant chunks — those furthest away — as page-out candidates.

use super::smmu::MultiViewMMU;

/// Return up to `budget` chunk IDs that are candidates for page-out (eviction).
///
/// Strategy: retrieve all reachable chunks ranked by relevance, then return
/// the *tail* (least relevant) ones. Chunks not reachable at all within
/// `max_hops` are also considered candidates.
///
/// This function takes `&MultiViewMMU` (immutable reference) — it never
/// mutates the graph.
pub fn page_out_candidates(
    smmu: &MultiViewMMU,
    active_chunk_id: &str,
    max_hops: usize,
    budget: usize,
) -> Vec<String> {
    if smmu.chunk_count() == 0 || budget == 0 {
        return Vec::new();
    }

    // Get relevance scores (descending by score).
    let relevant = smmu.retrieve_relevant(active_chunk_id, max_hops, [1.0, 1.0, 1.0, 1.0]);

    // All chunk IDs in the S-MMU except the active one.
    let all_ids: Vec<String> = smmu
        .chunk_map
        .keys()
        .filter(|&cid| cid != active_chunk_id)
        .cloned()
        .collect();

    // Chunks NOT reachable within max_hops are prime eviction candidates.
    let reachable_set: std::collections::HashSet<&str> =
        relevant.iter().map(|(cid, _)| cid.as_str()).collect();
    let mut unreachable: Vec<String> = all_ids
        .into_iter()
        .filter(|cid| !reachable_set.contains(cid.as_str()))
        .collect();

    // Start with unreachable, then append reachable in ascending relevance
    // (least relevant first).
    let mut candidates = Vec::with_capacity(budget);
    candidates.append(&mut unreachable);

    // Append reachable chunks in reverse order (least relevant first).
    for (cid, _) in relevant.into_iter().rev() {
        candidates.push(cid);
    }

    candidates.truncate(budget);
    candidates
}
