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
    active_chunk_id: usize,
    max_hops: usize,
    budget: usize,
) -> Vec<usize> {
    if smmu.chunk_count() == 0 || budget == 0 {
        return Vec::new();
    }

    // Get relevance scores (descending by score).
    let relevant = smmu.retrieve_relevant(active_chunk_id, max_hops, [1.0, 1.0, 1.0, 1.0]);

    // All chunk IDs in the S-MMU except the active one.
    let all_ids: Vec<usize> = smmu
        .chunk_map
        .keys()
        .copied()
        .filter(|&cid| cid != active_chunk_id)
        .collect();

    // Chunks NOT reachable within max_hops are prime eviction candidates.
    let reachable_set: std::collections::HashSet<usize> =
        relevant.iter().map(|&(cid, _)| cid).collect();
    let mut unreachable: Vec<usize> = all_ids
        .iter()
        .copied()
        .filter(|cid| !reachable_set.contains(cid))
        .collect();

    // Start with unreachable, then append reachable in ascending relevance
    // (least relevant first).
    let mut candidates = Vec::with_capacity(budget);
    candidates.append(&mut unreachable);

    // Append reachable chunks in reverse order (least relevant first).
    for &(cid, _) in relevant.iter().rev() {
        candidates.push(cid);
    }

    candidates.truncate(budget);
    candidates
}
