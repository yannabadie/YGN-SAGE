#!/usr/bin/env python3
"""Generate expert-quality topologies via Claude Opus 4.6 distillation.

These topologies are hand-crafted by the frontier model with deep knowledge
of the SAGE architecture. They target gaps in existing SFT data:
- Complex multi-node DAGs (5-7 nodes)
- Adaptive topologies with checkpoints
- Diverse task patterns (code, math, data, system design)
- Proper model_tier selection and provider hints
- State edges for dynamic information flow

Usage:
    python scripts/generate_expert_topologies.py
    python scripts/generate_expert_topologies.py --output data/expert_topologies.jsonl
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from sage_tool_schemas import wrap_toolcall, TOOLCALL_SYSTEM_PROMPT

# ═══════════════════════════════════════════════════════════════
# Expert topologies — distilled from Claude Opus 4.6's understanding
# of the SAGE architecture, BigCodeBench tasks, and multi-agent patterns
# ═══════════════════════════════════════════════════════════════

EXPERT_TOPOLOGIES = [
    # ── Complex: 5-node code generation with verification pipeline ──
    {
        "prompt": "Write a function that takes a list of file paths, reads each file, extracts all email addresses using regex, deduplicates them, and returns a sorted list. Handle encoding errors gracefully.",
        "topology": {
            "difficulty": "complex",
            "reasoning": "Multi-step task requiring regex expertise, file I/O error handling, and deduplication. A 5-node pipeline ensures correctness: planner decomposes, coder implements, tester writes tests, reviewer checks edge cases, synthesizer merges.",
            "nodes": [
                {"role": "planner", "model_tier": "reasoner", "prompt": "Decompose this file-processing task into subtasks: 1) file reading with encoding error handling, 2) regex email extraction, 3) deduplication and sorting. List edge cases: binary files, empty files, malformed emails, mixed encodings.", "provider_hint": "google"},
                {"role": "coder", "model_tier": "codex", "prompt": "Implement the function following the planner's decomposition. Use `re.findall(r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+', text)` for email extraction. Handle UnicodeDecodeError with fallback to latin-1. Return sorted(set(emails)).", "provider_hint": "openai"},
                {"role": "tester", "model_tier": "fast", "prompt": "Write pytest tests covering: normal files, empty files, binary files (should not crash), files with duplicate emails, files with no emails, mixed encoding files. Use tempfile for test fixtures.", "provider_hint": "google"},
                {"role": "reviewer", "model_tier": "reasoner", "prompt": "Review the implementation against the tests. Check: does the regex handle edge cases (emails with dots, hyphens, plus signs)? Is the encoding fallback correct? Are there any resource leaks (unclosed files)? Use context managers.", "provider_hint": "google"},
                {"role": "synthesizer", "model_tier": "fast", "prompt": "Combine the implementation with any fixes from the reviewer. Return ONLY the final, complete, self-contained Python code in a ```python block. Include imports. No explanation.", "provider_hint": "google"},
            ],
            "edges": [
                {"from_idx": 0, "to_idx": 1, "flow_type": "message"},
                {"from_idx": 1, "to_idx": 2, "flow_type": "message"},
                {"from_idx": 1, "to_idx": 3, "flow_type": "message"},
                {"from_idx": 2, "to_idx": 4, "flow_type": "message"},
                {"from_idx": 3, "to_idx": 4, "flow_type": "message"},
            ],
            "adaptation": {
                "checkpoints": [1, 3],
                "max_upgrades": 2,
                "max_reroutes": 1,
                "quality_threshold": 0.5,
            },
        },
    },
    # ── Complex: Data pipeline with parallel processing ──
    {
        "prompt": "Create a function that downloads JSON data from multiple URLs concurrently, validates each response against a schema, aggregates the results into a pandas DataFrame, and handles timeouts and invalid responses gracefully.",
        "topology": {
            "difficulty": "complex",
            "reasoning": "Concurrent I/O + schema validation + aggregation requires careful error handling. 5-node DAG with parallel reviewer path ensures robustness. The coder handles async/concurrent logic, validator checks schema, aggregator builds DataFrame.",
            "nodes": [
                {"role": "architect", "model_tier": "reasoner", "prompt": "Design the concurrent download architecture. Choose between asyncio+aiohttp or concurrent.futures.ThreadPoolExecutor. Define the JSON schema validation approach (jsonschema or manual). Plan error handling: per-URL timeouts, invalid JSON, schema violations. Output a clear design document.", "provider_hint": "google"},
                {"role": "coder", "model_tier": "codex", "prompt": "Implement using concurrent.futures.ThreadPoolExecutor for simplicity. Use requests with timeout parameter. Validate each response. Collect results in a list of dicts, then pd.DataFrame(results). Return (dataframe, list_of_errors).", "provider_hint": "openai"},
                {"role": "error-handler", "model_tier": "fast", "prompt": "Review the error handling paths. Ensure: 1) individual URL failures don't crash the batch, 2) timeouts are respected, 3) invalid JSON is caught, 4) schema violations produce clear error messages. Add retry logic (max 2 retries with backoff).", "provider_hint": "google"},
                {"role": "tester", "model_tier": "fast", "prompt": "Write tests using unittest.mock to mock HTTP responses. Test: successful batch, partial failures, all failures, timeout handling, invalid JSON, schema violations. Use @patch('requests.get') for mocking.", "provider_hint": "minimax"},
                {"role": "synthesizer", "model_tier": "fast", "prompt": "Merge the coder's implementation with the error-handler's improvements and ensure all tests would pass. Return ONLY the complete Python code in a ```python block.", "provider_hint": "google"},
            ],
            "edges": [
                {"from_idx": 0, "to_idx": 1, "flow_type": "message"},
                {"from_idx": 1, "to_idx": 2, "flow_type": "message"},
                {"from_idx": 1, "to_idx": 3, "flow_type": "message"},
                {"from_idx": 2, "to_idx": 4, "flow_type": "message"},
                {"from_idx": 3, "to_idx": 4, "flow_type": "message"},
            ],
            "adaptation": {
                "checkpoints": [1],
                "max_upgrades": 3,
                "max_reroutes": 2,
                "quality_threshold": 0.4,
            },
        },
    },
    # ── Complex: Mathematical computation with verification ──
    {
        "prompt": "Implement a function that computes the PageRank of nodes in a directed graph. The function takes an adjacency list and returns a dictionary of node scores. Use power iteration with damping factor 0.85.",
        "topology": {
            "difficulty": "complex",
            "reasoning": "PageRank requires correct matrix math and convergence logic. A math-expert + coder + verifier pipeline catches numerical errors. The verifier independently computes expected values for small graphs.",
            "nodes": [
                {"role": "math-expert", "model_tier": "reasoner", "prompt": "Derive the PageRank power iteration formula. M[i][j] = 1/out_degree(j) if edge j->i exists. PR(t+1) = (1-d)/N + d * M @ PR(t). Explain handling of dangling nodes (no outgoing edges). Specify convergence criterion: L1 norm < 1e-6.", "provider_hint": "google"},
                {"role": "coder", "model_tier": "codex", "prompt": "Implement PageRank using numpy. Build transition matrix from adjacency list. Handle dangling nodes by distributing their rank equally. Use power iteration with max_iter=100 and convergence threshold 1e-6. Return dict mapping node to score.", "provider_hint": "openai"},
                {"role": "verifier", "model_tier": "reasoner", "prompt": "Verify the implementation by manually computing PageRank for a 3-node triangle graph and a 4-node graph with a dangling node. Check that scores sum to 1.0. Check convergence behavior. Flag any numerical instability.", "provider_hint": "google"},
                {"role": "optimizer", "model_tier": "fast", "prompt": "Review for performance: is the matrix multiplication efficient? Could we use sparse matrices for large graphs? Add input validation (non-empty graph, valid adjacency list). Ensure the function signature matches the spec.", "provider_hint": "deepseek"},
                {"role": "synthesizer", "model_tier": "fast", "prompt": "Produce the final implementation incorporating all feedback. Ensure it passes the verifier's test cases. Return ONLY the code in a ```python block.", "provider_hint": "google"},
            ],
            "edges": [
                {"from_idx": 0, "to_idx": 1, "flow_type": "message"},
                {"from_idx": 1, "to_idx": 2, "flow_type": "message"},
                {"from_idx": 1, "to_idx": 3, "flow_type": "message"},
                {"from_idx": 2, "to_idx": 4, "flow_type": "message"},
                {"from_idx": 3, "to_idx": 4, "flow_type": "message"},
            ],
            "adaptation": {
                "checkpoints": [1, 2],
                "max_upgrades": 2,
                "max_reroutes": 1,
                "quality_threshold": 0.5,
            },
        },
    },
    # ── Moderate: 3-node with state edges ──
    {
        "prompt": "Write a function that parses a CSV file with headers, filters rows where a specified column value exceeds a threshold, and returns the filtered data as a list of dictionaries.",
        "topology": {
            "difficulty": "moderate",
            "reasoning": "Standard data processing task. 3 nodes: coder + reviewer + synthesizer. State edge carries the parsed schema from coder to reviewer for validation.",
            "nodes": [
                {"role": "coder", "model_tier": "fast", "prompt": "Implement using csv.DictReader. Accept filename, column_name, threshold as parameters. Handle: missing column (KeyError), non-numeric values (skip row), empty file. Return list of dicts.", "provider_hint": "google"},
                {"role": "reviewer", "model_tier": "budget", "prompt": "Check the implementation for: proper file handling (with statement), correct type conversion (float), edge cases (empty CSV, header-only, column not found). Suggest fixes if needed.", "provider_hint": "deepseek"},
                {"role": "synthesizer", "model_tier": "fast", "prompt": "Merge implementation and review. Return ONLY the final Python code in a ```python block.", "provider_hint": "google"},
            ],
            "edges": [
                {"from_idx": 0, "to_idx": 1, "flow_type": "message"},
                {"from_idx": 0, "to_idx": 1, "flow_type": "state"},
                {"from_idx": 1, "to_idx": 2, "flow_type": "message"},
            ],
            "adaptation": {
                "checkpoints": [0],
                "max_upgrades": 1,
                "max_reroutes": 0,
                "quality_threshold": 0.6,
            },
        },
    },
    # ── Simple: Efficient 2-node for easy tasks ──
    {
        "prompt": "Write a function that takes a string and returns True if it's a palindrome (ignoring case and non-alphanumeric characters), False otherwise.",
        "topology": {
            "difficulty": "simple",
            "reasoning": "Trivial string task. 2-node DAG is cost-optimal: one fast coder, one budget verifier. No adaptation needed — the task is too simple to fail.",
            "nodes": [
                {"role": "coder", "model_tier": "fast", "prompt": "Implement a palindrome checker. Clean the string: lowercase, keep only alphanumeric chars. Compare with reversed. One-liner possible: cleaned == cleaned[::-1].", "provider_hint": "google"},
                {"role": "verifier", "model_tier": "budget", "prompt": "Verify the solution handles: empty string, single char, spaces, punctuation, mixed case. Test with 'A man, a plan, a canal: Panama' (True) and 'race a car' (False). Return the final code.", "provider_hint": "deepseek"},
            ],
            "edges": [
                {"from_idx": 0, "to_idx": 1, "flow_type": "message"},
            ],
        },
    },
    # ── Complex: 6-node with parallel branches and adaptation ──
    {
        "prompt": "Create a REST API client class that supports GET, POST, PUT, DELETE with automatic retry, rate limiting, response caching, and comprehensive error handling. Include async support.",
        "topology": {
            "difficulty": "complex",
            "reasoning": "Multi-concern task (retry, rate-limit, cache, async) benefits from specialized agents. 6-node DAG with 2 parallel branches: one for sync implementation, one for async. Architect designs the class interface, specialists implement concerns.",
            "nodes": [
                {"role": "architect", "model_tier": "reasoner", "prompt": "Design the REST client class API. Define: __init__(base_url, max_retries=3, rate_limit=10/sec, cache_ttl=300), get/post/put/delete methods, retry with exponential backoff, token bucket rate limiter, LRU cache for GET responses. Separate sync and async interfaces.", "provider_hint": "google"},
                {"role": "sync-coder", "model_tier": "codex", "prompt": "Implement the synchronous REST client using requests. Include: session reuse, retry decorator with exponential backoff (tenacity or manual), thread-safe token bucket rate limiter, functools.lru_cache for GET caching, proper exception hierarchy.", "provider_hint": "openai"},
                {"role": "async-coder", "model_tier": "codex", "prompt": "Implement the async version using aiohttp. Include: session management (async context manager), async retry with backoff, asyncio-compatible rate limiter (semaphore-based), async LRU cache, proper async exception handling.", "provider_hint": "openai"},
                {"role": "tester", "model_tier": "fast", "prompt": "Write comprehensive tests for both sync and async clients. Use responses library for sync mocking, aioresponses for async. Test: successful requests, retries on 5xx, rate limiting behavior, cache hits/misses, timeout handling.", "provider_hint": "google"},
                {"role": "reviewer", "model_tier": "reasoner", "prompt": "Review both implementations for: thread safety (sync), coroutine safety (async), resource leaks (unclosed sessions), correct HTTP semantics, proper cache invalidation on POST/PUT/DELETE. Check that retry doesn't retry on 4xx.", "provider_hint": "google"},
                {"role": "synthesizer", "model_tier": "fast", "prompt": "Merge sync and async implementations into a single module. Apply reviewer's fixes. Ensure both classes share the retry/rate-limit logic (DRY). Return complete code in ```python block.", "provider_hint": "google"},
            ],
            "edges": [
                {"from_idx": 0, "to_idx": 1, "flow_type": "message"},
                {"from_idx": 0, "to_idx": 2, "flow_type": "message"},
                {"from_idx": 1, "to_idx": 4, "flow_type": "message"},
                {"from_idx": 2, "to_idx": 4, "flow_type": "message"},
                {"from_idx": 1, "to_idx": 3, "flow_type": "message"},
                {"from_idx": 2, "to_idx": 3, "flow_type": "message"},
                {"from_idx": 3, "to_idx": 5, "flow_type": "message"},
                {"from_idx": 4, "to_idx": 5, "flow_type": "message"},
            ],
            "adaptation": {
                "checkpoints": [1, 2],
                "max_upgrades": 3,
                "max_reroutes": 2,
                "quality_threshold": 0.4,
            },
        },
    },
    # ── Complex: 7-node deep pipeline for system design ──
    {
        "prompt": "Implement a simple in-memory key-value store with TTL expiration, LRU eviction when capacity is reached, and thread-safe operations. Support get, set, delete, and stats operations.",
        "topology": {
            "difficulty": "complex",
            "reasoning": "Concurrent data structure with multiple concerns (TTL, LRU, thread safety). 7-node pipeline: architect designs, specialist handles each concern, integration verifies thread safety under load.",
            "nodes": [
                {"role": "architect", "model_tier": "reasoner", "prompt": "Design the KV store. Data structures: OrderedDict for LRU ordering, dict for O(1) lookup, heap or lazy cleanup for TTL. Threading: use threading.Lock or RLock. API: get(key)->value, set(key, value, ttl=None), delete(key)->bool, stats()->{size, hits, misses}.", "provider_hint": "google"},
                {"role": "core-coder", "model_tier": "codex", "prompt": "Implement the core KV store following the architect's design. Use OrderedDict with move_to_end for LRU. Store entries as (value, expire_time). Lazy TTL cleanup on access. capacity parameter in __init__.", "provider_hint": "openai"},
                {"role": "threading-specialist", "model_tier": "codex", "prompt": "Add thread safety to the core implementation. Use threading.RLock (not Lock, to allow nested calls). Wrap every public method with the lock. Ensure no deadlocks. Add a background cleanup thread for expired entries (optional, daemon thread).", "provider_hint": "openai"},
                {"role": "tester", "model_tier": "fast", "prompt": "Write tests: basic get/set/delete, TTL expiration (use time.sleep or mock time), LRU eviction at capacity, stats tracking (hits/misses), thread safety (concurrent set/get from 10 threads). Use threading.Thread for concurrency tests.", "provider_hint": "google"},
                {"role": "load-tester", "model_tier": "fast", "prompt": "Write a stress test: 100 threads, 1000 operations each, mix of get/set/delete. Verify no exceptions, no data corruption, stats are consistent (hits + misses == total gets). Measure throughput.", "provider_hint": "deepseek"},
                {"role": "reviewer", "model_tier": "reasoner", "prompt": "Review for correctness: check LRU ordering after get (should refresh), TTL edge cases (expire exactly at boundary), eviction under concurrent access, memory leaks (entries never cleaned). Verify the lock granularity is correct.", "provider_hint": "google"},
                {"role": "synthesizer", "model_tier": "fast", "prompt": "Produce the final, production-quality implementation. Apply all reviewer fixes. Include docstrings. Return ONLY the code in a ```python block.", "provider_hint": "google"},
            ],
            "edges": [
                {"from_idx": 0, "to_idx": 1, "flow_type": "message"},
                {"from_idx": 1, "to_idx": 2, "flow_type": "message"},
                {"from_idx": 2, "to_idx": 3, "flow_type": "message"},
                {"from_idx": 2, "to_idx": 4, "flow_type": "message"},
                {"from_idx": 2, "to_idx": 5, "flow_type": "message"},
                {"from_idx": 3, "to_idx": 6, "flow_type": "message"},
                {"from_idx": 4, "to_idx": 6, "flow_type": "message"},
                {"from_idx": 5, "to_idx": 6, "flow_type": "message"},
            ],
            "adaptation": {
                "checkpoints": [1, 2, 5],
                "max_upgrades": 3,
                "max_reroutes": 2,
                "quality_threshold": 0.4,
            },
        },
    },
    # ── Moderate: Math with budget optimization ──
    {
        "prompt": "Write a function that computes the nth Fibonacci number using matrix exponentiation for O(log n) time complexity. Handle edge cases for n=0 and n=1.",
        "topology": {
            "difficulty": "moderate",
            "reasoning": "Algorithmic task requiring math knowledge. Budget-optimized: a reasoner derives the matrix formula, a budget coder implements it. No need for expensive codex tier.",
            "nodes": [
                {"role": "math-solver", "model_tier": "reasoner", "prompt": "Derive the matrix exponentiation formula for Fibonacci: [[1,1],[1,0]]^n gives F(n). Explain the fast matrix power algorithm (binary exponentiation). Handle n=0 returns 0, n=1 returns 1.", "provider_hint": "google"},
                {"role": "coder", "model_tier": "budget", "prompt": "Implement matrix multiplication for 2x2 matrices and matrix power using binary exponentiation. Apply to Fibonacci: multiply [[1,1],[1,0]] n times, return top-left element. Use integer arithmetic only (no floating point).", "provider_hint": "deepseek"},
                {"role": "synthesizer", "model_tier": "fast", "prompt": "Combine the math derivation with the implementation. Test mentally with n=10 (should be 55). Return ONLY the final code in ```python block.", "provider_hint": "google"},
            ],
            "edges": [
                {"from_idx": 0, "to_idx": 1, "flow_type": "message"},
                {"from_idx": 1, "to_idx": 2, "flow_type": "message"},
            ],
            "adaptation": {
                "checkpoints": [1],
                "max_upgrades": 1,
                "max_reroutes": 0,
                "quality_threshold": 0.6,
            },
        },
    },
]


def main():
    import argparse as _ap
    parser = _ap.ArgumentParser(description="Generate expert topologies (Claude Opus distillation)")
    parser.add_argument("--output", default="data/expert_topologies.jsonl")
    args = parser.parse_args()

    with open(args.output, "w", encoding="utf-8") as f:
        for i, item in enumerate(EXPERT_TOPOLOGIES):
            entry = {
                "task_id": f"expert/{i:03d}",
                "prompt": item["prompt"],
                "topology": item["topology"],
                "topology_toolcall": wrap_toolcall(item["topology"]),
                "system_prompt": TOOLCALL_SYSTEM_PROMPT,
                "node_count": len(item["topology"]["nodes"]),
                "edge_count": len(item["topology"]["edges"]),
                "difficulty": item["topology"]["difficulty"],
                "model": "claude-opus-4-6-distilled",
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Generated {len(EXPERT_TOPOLOGIES)} expert topologies")
    print(f"Output: {args.output}")

    by_diff = {}
    by_nodes = {}
    for t in EXPERT_TOPOLOGIES:
        d = t["topology"]["difficulty"]
        n = len(t["topology"]["nodes"])
        by_diff[d] = by_diff.get(d, 0) + 1
        by_nodes[n] = by_nodes.get(n, 0) + 1

    print(f"Difficulty: {by_diff}")
    print(f"Nodes: {by_nodes}")
    adapt = sum(1 for t in EXPERT_TOPOLOGIES if "adaptation" in t["topology"])
    print(f"With adaptation: {adapt}/{len(EXPERT_TOPOLOGIES)}")


if __name__ == "__main__":
    main()
