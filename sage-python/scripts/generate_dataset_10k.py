#!/usr/bin/env python3
"""Generate 10K+ synthetic topology training entries for YGN-SAGE.

Produces diverse coding task prompts paired with topology DAGs across:
- 20+ coding task types (algorithms, data structures, APIs, debugging, etc.)
- 8+ topology patterns (sequential, AVR, debate, parallel, diamond, funnel, expert_panel, pipeline)
- 3 difficulty levels: simple (30%), moderate (40%), complex (30%)
- Adaptation metadata on ~40% of entries
- Recovery scenarios on ~10% of entries
- Varied edge types: message, control, state
- Proper model_tiers: reasoner, fast, budget

Each entry is validated: YAML-parsable, valid node references, acyclic, roles present.

Usage:
    python scripts/generate_dataset_10k.py [--count 8500] [--seed 42]
    python scripts/generate_dataset_10k.py --count 10000 --output data/synthetic_topologies_10k.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("gen_10k")

# ---------------------------------------------------------------------------
# Task type definitions — 25 categories with prompt templates
# ---------------------------------------------------------------------------

TASK_TYPES: dict[str, list[str]] = {
    "sorting": [
        "Implement {algo} sort algorithm that handles arrays of up to {n} elements efficiently.",
        "Write a function that sorts a list of {obj} by {field} using {algo} sort.",
        "Create an in-place {algo} sort that works on a doubly-linked list.",
        "Implement a stable {algo} sort for a list of tuples, sorting by the {k}-th element.",
        "Write a sorting function that uses {algo} sort with a custom comparator for {obj}.",
        "Implement an external {algo} sort for files larger than available memory.",
        "Create a hybrid sort that switches from {algo} to insertion sort for partitions smaller than {threshold}.",
    ],
    "graph_algorithms": [
        "Implement Dijkstra's shortest path algorithm for a weighted graph with {n} vertices.",
        "Write a function to detect all cycles in a directed graph with up to {n} nodes.",
        "Create a solution for finding the minimum spanning tree using {algo} algorithm.",
        "Implement topological sort for a DAG with {n} nodes and dependency edges.",
        "Write a function that finds all strongly connected components in a directed graph.",
        "Implement the Bellman-Ford algorithm to detect negative weight cycles in a graph.",
        "Create a function to find all bridges and articulation points in an undirected graph.",
        "Implement A* pathfinding on a 2D grid with obstacles.",
        "Write a function to compute the maximum flow in a network using {algo} algorithm.",
        "Implement a solution to find the shortest path in a weighted graph with negative edges using SPFA.",
    ],
    "dynamic_programming": [
        "Solve the {problem} problem using dynamic programming with O({complexity}) time complexity.",
        "Implement a DP solution for the longest common subsequence of two strings of length up to {n}.",
        "Write a function that solves the 0/1 knapsack problem with {n} items and capacity {cap}.",
        "Create a DP solution for the edit distance between two strings.",
        "Implement the coin change problem: find minimum coins to make amount {n} from denominations {coins}.",
        "Solve the matrix chain multiplication problem for {n} matrices using DP.",
        "Write a DP solution for the longest increasing subsequence in an array of {n} elements.",
        "Implement a solution for the traveling salesman problem using bitmask DP for up to {n} cities.",
        "Create a DP solution for counting the number of distinct paths in a {rows}x{cols} grid with obstacles.",
        "Solve the palindrome partitioning problem: find minimum cuts to partition a string into palindromes.",
    ],
    "string_manipulation": [
        "Implement the KMP string matching algorithm for pattern matching.",
        "Write a function to find the longest palindromic substring in a string.",
        "Create a function that compresses a string using run-length encoding.",
        "Implement a regex matcher that supports '.' and '*' wildcards.",
        "Write a function to decode a string encoded as '{count}[{str}]' format recursively.",
        "Implement the Rabin-Karp algorithm for multi-pattern string matching.",
        "Create a function that finds all anagrams of a pattern in a text string.",
        "Write a suffix array construction algorithm for a string of length {n}.",
        "Implement Manacher's algorithm for finding all palindromic substrings.",
        "Create a function to convert between different string encodings (UTF-8, ASCII, base64).",
    ],
    "tree_traversal": [
        "Implement in-order, pre-order, and post-order traversal of a binary tree iteratively.",
        "Write a function to serialize and deserialize a binary tree to/from a string.",
        "Create a function that finds the lowest common ancestor in a binary search tree.",
        "Implement a balanced BST (AVL tree) with insert, delete, and search operations.",
        "Write a function that validates whether a binary tree is a valid BST.",
        "Create a function to find the diameter of a binary tree.",
        "Implement level-order traversal (BFS) of an N-ary tree with spiral order output.",
        "Write a function to convert a sorted array into a height-balanced BST.",
        "Implement a red-black tree with rebalancing after insertion.",
        "Create a function to find all paths from root to leaf that sum to a target value.",
    ],
    "api_design": [
        "Design and implement a RESTful API client for {service} with retry logic and rate limiting.",
        "Create a Python class that wraps the {service} API with authentication, caching, and error handling.",
        "Implement an API gateway that routes requests to multiple backend services with load balancing.",
        "Write an async HTTP client that handles pagination, retries, and concurrent requests to {service}.",
        "Design a webhook handler that validates signatures, parses payloads, and dispatches to handlers.",
        "Create an API middleware that implements JWT authentication with token refresh.",
        "Implement a GraphQL resolver layer with DataLoader batching for {service}.",
        "Write an API rate limiter using token bucket algorithm with per-endpoint configuration.",
        "Create a REST API wrapper with automatic schema validation and OpenAPI spec generation.",
        "Implement an API client SDK with connection pooling and circuit breaker pattern.",
    ],
    "database_queries": [
        "Write a function that efficiently queries a SQLite database for {query_type} with parameterized inputs.",
        "Implement a database migration system that handles schema changes with rollback support.",
        "Create an ORM-like query builder that generates SQL from Python method chains.",
        "Write a function that performs bulk upsert operations on a PostgreSQL table.",
        "Implement a connection pool manager with configurable max connections and timeout.",
        "Create a query optimizer that rewrites slow queries using index hints and join reordering.",
        "Write a function to implement full-text search with ranking across multiple database tables.",
        "Implement a database sharding strategy with consistent hashing.",
        "Create a caching layer with write-through and write-behind strategies for database queries.",
        "Write a function that generates database reports with aggregation, grouping, and pagination.",
    ],
    "web_scraping": [
        "Write a web scraper that extracts {data_type} from {site_type} pages with pagination handling.",
        "Implement an async web crawler that respects robots.txt and rate limits.",
        "Create a scraper that handles JavaScript-rendered content using headless browser automation.",
        "Write a function that scrapes and parses structured data from HTML tables.",
        "Implement a web scraper with automatic retry, proxy rotation, and CAPTCHA detection.",
        "Create a data extraction pipeline that scrapes, cleans, and stores data in structured format.",
        "Write a scraper that monitors {site_type} pages for changes and sends notifications.",
        "Implement a concurrent scraper using asyncio with connection pooling and error recovery.",
    ],
    "file_io": [
        "Write a function that reads a large CSV file in chunks and processes each chunk with {operation}.",
        "Implement a file watcher that monitors a directory for changes and triggers callbacks.",
        "Create a log file parser that extracts timestamps, levels, and messages into structured data.",
        "Write a function that merges multiple sorted files into one sorted output file.",
        "Implement a file-based key-value store with ACID-like guarantees.",
        "Create a function that converts between file formats: {format_from} to {format_to}.",
        "Write a concurrent file processor that handles multiple files in parallel with progress reporting.",
        "Implement a file archiver that creates compressed archives with integrity verification.",
    ],
    "math": [
        "Implement a function that computes the {k}-th prime number using the Sieve of Eratosthenes.",
        "Write a solution for modular exponentiation with large numbers (a^b mod m).",
        "Create a function that solves a system of linear equations using Gaussian elimination.",
        "Implement a function for computing large Fibonacci numbers using matrix exponentiation.",
        "Write a function that performs polynomial multiplication using FFT.",
        "Create a number theory function that computes Euler's totient for numbers up to {n}.",
        "Implement a function to find the greatest common divisor of {k} numbers using extended Euclidean algorithm.",
        "Write a function that computes combinatorics: C(n,k) mod p using Lucas' theorem.",
        "Create a function for arbitrary-precision arithmetic (big integer) with basic operations.",
        "Implement a function to solve Diophantine equations ax + by = c.",
    ],
    "cryptography": [
        "Implement AES encryption and decryption in CBC mode with PKCS7 padding.",
        "Write a function that generates and verifies HMAC-SHA256 signatures.",
        "Create a simple RSA key generation, encryption, and decryption implementation.",
        "Implement a password hashing function using bcrypt with configurable work factor.",
        "Write a function that implements the Diffie-Hellman key exchange protocol.",
        "Create a digital signature scheme using ECDSA with key management.",
        "Implement a secure random token generator with configurable entropy.",
        "Write a function to encrypt/decrypt files using a streaming cipher.",
    ],
    "ml_pipelines": [
        "Implement a data preprocessing pipeline for {data_type} with normalization, encoding, and splitting.",
        "Write a function that trains a {model} classifier with cross-validation and hyperparameter tuning.",
        "Create a feature engineering pipeline that extracts statistical features from time series data.",
        "Implement a mini-batch gradient descent optimizer with momentum and learning rate scheduling.",
        "Write an ensemble model that combines predictions from multiple classifiers using {strategy}.",
        "Create a data augmentation pipeline for {data_type} with configurable transformations.",
        "Implement a custom scikit-learn transformer with fit/transform interface for {operation}.",
        "Write a model evaluation function that computes precision, recall, F1, and AUC-ROC.",
    ],
    "testing": [
        "Write comprehensive unit tests for a {component} class using pytest with parametrize.",
        "Implement a property-based testing framework using Hypothesis for {component}.",
        "Create a test fixture system that manages database state for integration tests.",
        "Write a mock/stub factory that generates test doubles for external API dependencies.",
        "Implement a mutation testing tool that modifies source code and runs tests to measure coverage.",
        "Create a load testing script that simulates {n} concurrent users making API requests.",
        "Write a test data generator that creates realistic {data_type} datasets for testing.",
        "Implement a snapshot testing system for complex data structures.",
    ],
    "refactoring": [
        "Refactor a monolithic function that handles {task} into a clean class hierarchy with SOLID principles.",
        "Convert callback-based async code to use Python asyncio coroutines.",
        "Refactor a procedural script into a well-structured module with dependency injection.",
        "Extract a reusable library from duplicated code across {n} modules.",
        "Convert imperative data processing code to use functional programming with map/filter/reduce.",
        "Refactor a tightly-coupled class to use the strategy pattern for {behavior}.",
        "Transform a synchronous API client into an async one with connection pooling.",
        "Refactor a deeply nested conditional block into a clean state machine.",
    ],
    "debugging": [
        "Find and fix a race condition in a multithreaded producer-consumer implementation.",
        "Debug a memory leak in a long-running Python service that processes {data_type}.",
        "Identify and fix an off-by-one error in a binary search implementation.",
        "Debug a deadlock in a multi-threaded application with multiple shared resources.",
        "Find and fix a floating-point precision bug in a financial calculation module.",
        "Debug a performance regression in a data processing pipeline handling {n} records.",
        "Identify and fix a subtle bug in a recursive algorithm causing stack overflow.",
        "Debug an encoding issue causing data corruption in a file processing pipeline.",
    ],
    "optimization": [
        "Optimize a function that processes {n} records from O(n^2) to O(n log n) using {technique}.",
        "Write a cache-efficient matrix multiplication for {n}x{n} matrices.",
        "Optimize a recursive solution with memoization and bottom-up DP conversion.",
        "Create a memory-optimized version of {algorithm} using generators and itertools.",
        "Implement a profiling-driven optimization for a CPU-bound {task} function.",
        "Optimize database queries by implementing batch processing and connection pooling.",
        "Write a SIMD-friendly algorithm for {operation} on large arrays.",
        "Optimize an I/O-bound pipeline using async processing and buffered writes.",
    ],
    "concurrency": [
        "Implement a thread pool executor with task prioritization and cancellation support.",
        "Write an async event loop that handles multiple concurrent {operation} operations.",
        "Create a lock-free queue implementation using atomic operations.",
        "Implement a reader-writer lock with write preference and timeout support.",
        "Write a concurrent hash map with fine-grained locking for multi-threaded access.",
        "Create an actor model implementation with message passing and supervision.",
        "Implement a work-stealing scheduler for parallel task execution.",
        "Write an async pipeline that processes data through multiple stages concurrently.",
    ],
    "networking": [
        "Implement a TCP server that handles multiple concurrent connections with an event loop.",
        "Write a simple HTTP/1.1 server from scratch that handles GET and POST requests.",
        "Create a WebSocket server with ping/pong keepalive and message fragmentation.",
        "Implement a DNS resolver that performs recursive lookups with caching.",
        "Write a reverse proxy that load-balances requests across multiple backends.",
        "Create a simple peer-to-peer protocol for file sharing with discovery.",
        "Implement an RPC framework with serialization, timeout, and retry logic.",
        "Write a network packet sniffer that parses and analyzes TCP/IP headers.",
    ],
    "parsing": [
        "Implement a recursive descent parser for a simple {language} expression grammar.",
        "Write a JSON parser from scratch that handles all JSON data types.",
        "Create a tokenizer and parser for a SQL subset (SELECT, WHERE, JOIN).",
        "Implement a CSV parser that handles quoted fields, escaped characters, and multi-line values.",
        "Write a parser for mathematical expressions with operator precedence and parentheses.",
        "Create an XML/HTML parser that builds a DOM tree from raw text.",
        "Implement a YAML parser for a subset of the YAML spec (scalars, lists, maps).",
        "Write a command-line argument parser with subcommands, flags, and type validation.",
    ],
    "compilation": [
        "Implement a simple bytecode compiler for arithmetic expressions.",
        "Write a type checker for a simple statically-typed language.",
        "Create a code generator that translates an AST to Python bytecode.",
        "Implement a simple garbage collector using mark-and-sweep algorithm.",
        "Write a basic register allocator using graph coloring.",
        "Create a simple JIT compiler that compiles mathematical expressions to native code.",
        "Implement constant folding and dead code elimination optimization passes.",
        "Write a linker that resolves symbol references across multiple object files.",
    ],
    "data_structures": [
        "Implement a {ds} data structure with insert, delete, and search operations.",
        "Write a persistent (immutable) {ds} that supports efficient undo operations.",
        "Create a {ds} that supports both FIFO and LIFO operations efficiently.",
        "Implement a self-balancing BST using {algo} with guaranteed O(log n) operations.",
        "Write a union-find (disjoint set) data structure with path compression and union by rank.",
        "Create a bloom filter with configurable false positive rate and hash functions.",
        "Implement an LRU cache with O(1) get and put operations.",
        "Write a segment tree with lazy propagation for range queries and updates.",
        "Implement a trie with auto-complete functionality for a dictionary of {n} words.",
        "Create a skip list with probabilistic balancing and efficient range queries.",
    ],
    "system_design": [
        "Implement a simple in-memory message queue with pub/sub pattern and topic filtering.",
        "Write a distributed cache system with consistent hashing and replication.",
        "Create a simple task scheduler that handles cron-like scheduling with timezone support.",
        "Implement a circuit breaker pattern for service-to-service communication.",
        "Write a simple event sourcing system with event store and projection rebuilding.",
        "Create a rate limiter service supporting fixed window, sliding window, and token bucket.",
        "Implement a simple service registry with health checking and discovery.",
        "Write a configuration management system with hierarchical overrides and hot-reloading.",
    ],
    "geometry": [
        "Implement a function to compute the convex hull of a set of 2D points.",
        "Write a line segment intersection algorithm using sweep line technique.",
        "Create a function to find the closest pair of points in O(n log n).",
        "Implement a point-in-polygon test using ray casting algorithm.",
        "Write a function to compute the area of intersection of two rectangles.",
        "Create a Voronoi diagram generator using Fortune's algorithm.",
        "Implement a function to find the minimum enclosing circle of a set of points.",
        "Write a function to compute the Delaunay triangulation of a point set.",
    ],
    "functional_programming": [
        "Implement a lazy evaluation framework with memoized thunks and stream operations.",
        "Write a monadic parser combinator library supporting sequencing, alternation, and repetition.",
        "Create a pattern matching system similar to Rust's match for Python data classes.",
        "Implement a transducer library for composable data transformation pipelines.",
        "Write a functional reactive programming system with observables and operators.",
        "Create an immutable persistent data structure library with structural sharing.",
        "Implement a continuation-passing style transformer for async operations.",
        "Write a lens library for nested immutable data structure access and modification.",
    ],
    "security": [
        "Implement an input sanitization library that prevents SQL injection and XSS.",
        "Write a secure session management system with token rotation and invalidation.",
        "Create a role-based access control (RBAC) system with permission inheritance.",
        "Implement a CSRF protection middleware for a web framework.",
        "Write a secure file upload handler with content-type validation and virus scanning hooks.",
        "Create an OAuth 2.0 authorization server with PKCE flow support.",
        "Implement a certificate pinning verification for HTTPS connections.",
        "Write a secrets management module with encrypted storage and access auditing.",
    ],
}

# Template fill values
SORT_ALGOS = ["quicksort", "mergesort", "heapsort", "timsort", "radix", "counting", "shell"]
GRAPH_ALGOS = ["Kruskal's", "Prim's", "Edmonds-Karp", "Ford-Fulkerson", "Dinic's"]
DP_PROBLEMS = ["knapsack", "longest common subsequence", "edit distance", "matrix chain",
               "optimal BST", "rod cutting", "word break", "palindrome partitioning",
               "egg drop", "burst balloons"]
OBJECTS = ["students", "products", "transactions", "records", "events", "employees", "orders"]
FIELDS = ["name", "timestamp", "score", "priority", "price", "date", "id"]
SERVICES = ["GitHub", "Stripe", "AWS S3", "Slack", "Twitter", "OpenAI", "Elasticsearch", "Redis"]
DATA_TYPES = ["JSON", "CSV", "log files", "image metadata", "time-series", "user sessions"]
SITE_TYPES = ["e-commerce", "news", "social media", "job listing", "API documentation"]
FILE_FORMATS = [("CSV", "JSON"), ("JSON", "XML"), ("YAML", "JSON"), ("Parquet", "CSV"),
                ("CSV", "SQLite"), ("XML", "YAML")]
DS_TYPES = ["hash map", "priority queue", "deque", "B-tree", "red-black tree",
            "AVL tree", "splay tree", "treap"]
DS_ALGOS = ["AVL rotation", "red-black coloring", "splay operations", "treap rotation"]
LANGUAGES = ["arithmetic", "boolean", "Lisp-like", "JSON-like", "regex-like", "Markdown"]
MODELS = ["decision tree", "logistic regression", "random forest", "SVM", "k-NN",
          "gradient boosting", "neural network"]
ENSEMBLE_STRATEGIES = ["voting", "stacking", "bagging", "boosting"]
COMPONENTS = ["cache", "queue", "parser", "router", "scheduler", "pipeline"]
OPERATIONS = ["I/O", "computation", "network", "database", "transformation"]
ML_DATA_TYPES = ["tabular", "text", "time-series", "image features"]
ML_OPERATIONS = ["feature scaling", "missing value imputation", "encoding", "dimensionality reduction"]
QUERY_TYPES = ["aggregation", "join", "time-range", "full-text search", "hierarchical"]
TASKS_GENERAL = ["data processing", "report generation", "file management", "user authentication"]
BEHAVIORS = ["logging", "caching", "serialization", "validation"]
TECHNIQUES = ["sorting and two-pointers", "hash map lookup", "binary search", "heap/priority queue"]
ALGORITHMS = ["BFS traversal", "DFS traversal", "sliding window", "two-pointer scan"]


def _fill_template(template: str) -> str:
    """Fill template placeholders with random values."""
    replacements = {
        "{algo}": random.choice(SORT_ALGOS),
        "{n}": str(random.choice([100, 1000, 10000, 100000, 1000000])),
        "{k}": str(random.randint(1, 5)),
        "{obj}": random.choice(OBJECTS),
        "{field}": random.choice(FIELDS),
        "{threshold}": str(random.choice([8, 10, 16, 32])),
        "{problem}": random.choice(DP_PROBLEMS),
        "{complexity}": random.choice(["n^2", "n*W", "n*m", "2^n * n", "n log n"]),
        "{cap}": str(random.choice([50, 100, 500, 1000, 10000])),
        "{coins}": str(random.choice([[1, 5, 10, 25], [1, 3, 4], [2, 5, 7, 10]])),
        "{rows}": str(random.randint(3, 100)),
        "{cols}": str(random.randint(3, 100)),
        "{service}": random.choice(SERVICES),
        "{data_type}": random.choice(DATA_TYPES),
        "{site_type}": random.choice(SITE_TYPES),
        "{format_from}": "",
        "{format_to}": "",
        "{operation}": random.choice(OPERATIONS),
        "{ds}": random.choice(DS_TYPES),
        "{language}": random.choice(LANGUAGES),
        "{model}": random.choice(MODELS),
        "{strategy}": random.choice(ENSEMBLE_STRATEGIES),
        "{component}": random.choice(COMPONENTS),
        "{behavior}": random.choice(BEHAVIORS),
        "{technique}": random.choice(TECHNIQUES),
        "{algorithm}": random.choice(ALGORITHMS),
        "{task}": random.choice(TASKS_GENERAL),
        "{query_type}": random.choice(QUERY_TYPES),
    }
    # Handle file format pairs
    fmt = random.choice(FILE_FORMATS)
    replacements["{format_from}"] = fmt[0]
    replacements["{format_to}"] = fmt[1]

    result = template
    for key, val in replacements.items():
        result = result.replace(key, val)
    return result


# ---------------------------------------------------------------------------
# Topology patterns
# ---------------------------------------------------------------------------

ROLES = {
    "planner": "Analyze the task requirements and create a detailed implementation plan.",
    "architect": "Design the system architecture and component interfaces.",
    "coder": "Implement the solution following the plan.",
    "reviewer": "Review the code for correctness, edge cases, and best practices.",
    "tester": "Write comprehensive test cases and verify the solution.",
    "debugger": "Identify and fix bugs in the implementation.",
    "optimizer": "Optimize the solution for performance and memory usage.",
    "synthesizer": "Produce the final, complete, self-contained Python solution incorporating all feedback. Return ONLY the code inside a single ```python fenced block. No explanation, no commentary.",
    "analyst": "Derive and prove the mathematical or algorithmic approach.",
    "verifier": "Formally verify the correctness and edge-case coverage of the solution.",
}

# Role-specific prompt templates
ROLE_PROMPTS: dict[str, list[str]] = {
    "planner": [
        "Analyze and plan the solution for: {task}. Break down into clear implementation steps.",
        "Create a detailed implementation plan for: {task}. Identify edge cases and constraints.",
        "Design the algorithm and data structures needed for: {task}. Specify time/space complexity targets.",
    ],
    "architect": [
        "Design the component structure and interfaces for: {task}. Ensure modularity and testability.",
        "Plan the system architecture for: {task}. Define APIs, data flow, and error handling strategy.",
    ],
    "coder": [
        "Implement the solution following the plan. Task: {task}. Include type hints and handle edge cases.",
        "Write a self-contained Python function that solves: {task}. Include type hints and handle edge cases.",
        "Implement a production-quality solution for: {task}. Follow PEP-8, add docstrings.",
    ],
    "reviewer": [
        "Review the code for correctness, edge cases, and best practices. Suggest specific improvements.",
        "Critically review the implementation. Check: correctness, efficiency, error handling, edge cases.",
        "Review the code for bugs, style issues, and potential improvements. Be specific and actionable.",
    ],
    "tester": [
        "Write comprehensive test cases for the solution. Cover: normal, edge, error, and boundary cases.",
        "Create pytest test functions that verify all aspects of the implementation.",
        "Design test cases that would catch common bugs: off-by-one, empty input, overflow, type errors.",
    ],
    "debugger": [
        "Analyze the code for potential bugs and fix them. Focus on: boundary conditions, type errors, logic flaws.",
        "Debug the implementation. Check for race conditions, memory leaks, and incorrect assumptions.",
    ],
    "optimizer": [
        "Optimize the solution. Reduce time complexity where possible. Minimize memory allocation.",
        "Profile and optimize the implementation. Target: eliminate redundant computations, use efficient data structures.",
    ],
    "synthesizer": [
        "Produce the final, complete, self-contained Python solution incorporating all feedback. Return ONLY the code inside a single ```python fenced block. No explanation, no commentary.",
    ],
    "analyst": [
        "Derive and prove the algorithmic approach for: {task}. Establish correctness invariants.",
        "Analyze the mathematical properties needed for: {task}. Provide formal proof of correctness.",
    ],
    "verifier": [
        "Verify the solution's correctness. Check: invariants hold, edge cases covered, complexity claims valid.",
        "Formally verify the implementation against the specification. Identify any gaps or violations.",
    ],
}


def _role_prompt(role: str, task: str) -> str:
    """Generate a role-specific prompt for a task."""
    templates = ROLE_PROMPTS.get(role, [f"Perform the {role} role for: {{task}}."])
    template = random.choice(templates)
    return template.replace("{task}", task)


# Topology pattern definitions: each returns (nodes, edges)
def _pattern_sequential(task: str, difficulty: str) -> tuple[list[dict], list[dict]]:
    """Simple 2-node: coder -> synthesizer."""
    return (
        [
            {"role": "coder", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("coder", task)},
            {"role": "synthesizer", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("synthesizer", task)},
        ],
        [{"from_idx": 0, "to_idx": 1, "flow_type": "message", "gate": "open"}],
    )


def _pattern_sequential_3(task: str, difficulty: str) -> tuple[list[dict], list[dict]]:
    """3-node: planner -> coder -> synthesizer."""
    tier_coder = random.choice(["fast", "budget"])
    return (
        [
            {"role": "planner", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("planner", task)},
            {"role": "coder", "model_tier": tier_coder, "fallback_tier": "",
             "prompt": _role_prompt("coder", task)},
            {"role": "synthesizer", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("synthesizer", task)},
        ],
        [
            {"from_idx": 0, "to_idx": 1, "flow_type": "message", "gate": "open"},
            {"from_idx": 1, "to_idx": 2, "flow_type": "message", "gate": "open"},
        ],
    )


def _pattern_avr(task: str, difficulty: str) -> tuple[list[dict], list[dict]]:
    """AVR: planner -> coder -> reviewer -> synthesizer."""
    tier_coder = "fast" if difficulty == "moderate" else "reasoner"
    return (
        [
            {"role": "planner", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("planner", task)},
            {"role": "coder", "model_tier": tier_coder, "fallback_tier": "reasoner",
             "prompt": _role_prompt("coder", task)},
            {"role": "reviewer", "model_tier": "budget", "fallback_tier": "fast",
             "prompt": _role_prompt("reviewer", task)},
            {"role": "synthesizer", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("synthesizer", task)},
        ],
        [
            {"from_idx": 0, "to_idx": 1, "flow_type": "message", "gate": "open"},
            {"from_idx": 1, "to_idx": 2, "flow_type": "message", "gate": "conditional"},
            {"from_idx": 2, "to_idx": 3, "flow_type": "message", "gate": "open"},
        ],
    )


def _pattern_debate(task: str, difficulty: str) -> tuple[list[dict], list[dict]]:
    """Debate: planner -> coder_a + coder_b -> reviewer -> synthesizer."""
    return (
        [
            {"role": "planner", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("planner", task)},
            {"role": "coder", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("coder", task) + " Approach A: optimize for readability."},
            {"role": "coder", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("coder", task) + " Approach B: optimize for performance."},
            {"role": "reviewer", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": "Compare both implementations and select the better approach. " + _role_prompt("reviewer", task)},
            {"role": "synthesizer", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("synthesizer", task)},
        ],
        [
            {"from_idx": 0, "to_idx": 1, "flow_type": "message", "gate": "open"},
            {"from_idx": 0, "to_idx": 2, "flow_type": "message", "gate": "open"},
            {"from_idx": 1, "to_idx": 3, "flow_type": "message", "gate": "open"},
            {"from_idx": 2, "to_idx": 3, "flow_type": "message", "gate": "open"},
            {"from_idx": 3, "to_idx": 4, "flow_type": "message", "gate": "open"},
        ],
    )


def _pattern_parallel(task: str, difficulty: str) -> tuple[list[dict], list[dict]]:
    """Parallel: planner -> (coder + tester) -> synthesizer."""
    return (
        [
            {"role": "planner", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("planner", task)},
            {"role": "coder", "model_tier": "fast", "fallback_tier": "reasoner",
             "prompt": _role_prompt("coder", task)},
            {"role": "tester", "model_tier": "budget", "fallback_tier": "",
             "prompt": _role_prompt("tester", task)},
            {"role": "synthesizer", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("synthesizer", task)},
        ],
        [
            {"from_idx": 0, "to_idx": 1, "flow_type": "message", "gate": "open"},
            {"from_idx": 0, "to_idx": 2, "flow_type": "control", "gate": "open"},
            {"from_idx": 1, "to_idx": 3, "flow_type": "message", "gate": "open"},
            {"from_idx": 2, "to_idx": 3, "flow_type": "state", "gate": "open"},
        ],
    )


def _pattern_diamond(task: str, difficulty: str) -> tuple[list[dict], list[dict]]:
    """Diamond: planner -> (analyst + coder) -> reviewer -> synthesizer."""
    return (
        [
            {"role": "planner", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("planner", task)},
            {"role": "analyst", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("analyst", task)},
            {"role": "coder", "model_tier": "fast", "fallback_tier": "reasoner",
             "prompt": _role_prompt("coder", task)},
            {"role": "reviewer", "model_tier": "budget", "fallback_tier": "fast",
             "prompt": _role_prompt("reviewer", task)},
            {"role": "synthesizer", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("synthesizer", task)},
        ],
        [
            {"from_idx": 0, "to_idx": 1, "flow_type": "control", "gate": "open"},
            {"from_idx": 0, "to_idx": 2, "flow_type": "message", "gate": "open"},
            {"from_idx": 1, "to_idx": 3, "flow_type": "state", "gate": "open"},
            {"from_idx": 2, "to_idx": 3, "flow_type": "message", "gate": "open"},
            {"from_idx": 3, "to_idx": 4, "flow_type": "message", "gate": "open"},
        ],
    )


def _pattern_funnel(task: str, difficulty: str) -> tuple[list[dict], list[dict]]:
    """Funnel: planner -> coder -> reviewer -> optimizer -> synthesizer."""
    return (
        [
            {"role": "planner", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("planner", task)},
            {"role": "coder", "model_tier": "fast", "fallback_tier": "reasoner",
             "prompt": _role_prompt("coder", task)},
            {"role": "reviewer", "model_tier": "budget", "fallback_tier": "fast",
             "prompt": _role_prompt("reviewer", task)},
            {"role": "optimizer", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("optimizer", task)},
            {"role": "synthesizer", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("synthesizer", task)},
        ],
        [
            {"from_idx": 0, "to_idx": 1, "flow_type": "message", "gate": "open"},
            {"from_idx": 1, "to_idx": 2, "flow_type": "message", "gate": "conditional"},
            {"from_idx": 2, "to_idx": 3, "flow_type": "message", "gate": "open"},
            {"from_idx": 3, "to_idx": 4, "flow_type": "message", "gate": "open"},
        ],
    )


def _pattern_expert_panel(task: str, difficulty: str) -> tuple[list[dict], list[dict]]:
    """Expert panel: architect -> (analyst + coder + tester) -> reviewer -> synthesizer."""
    return (
        [
            {"role": "architect", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("architect", task)},
            {"role": "analyst", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("analyst", task)},
            {"role": "coder", "model_tier": "fast", "fallback_tier": "reasoner",
             "prompt": _role_prompt("coder", task)},
            {"role": "tester", "model_tier": "budget", "fallback_tier": "fast",
             "prompt": _role_prompt("tester", task)},
            {"role": "reviewer", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("reviewer", task)},
            {"role": "synthesizer", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("synthesizer", task)},
        ],
        [
            {"from_idx": 0, "to_idx": 1, "flow_type": "control", "gate": "open"},
            {"from_idx": 0, "to_idx": 2, "flow_type": "message", "gate": "open"},
            {"from_idx": 0, "to_idx": 3, "flow_type": "control", "gate": "open"},
            {"from_idx": 1, "to_idx": 4, "flow_type": "state", "gate": "open"},
            {"from_idx": 2, "to_idx": 4, "flow_type": "message", "gate": "open"},
            {"from_idx": 3, "to_idx": 4, "flow_type": "state", "gate": "open"},
            {"from_idx": 4, "to_idx": 5, "flow_type": "message", "gate": "open"},
        ],
    )


def _pattern_pipeline(task: str, difficulty: str) -> tuple[list[dict], list[dict]]:
    """Pipeline: planner -> coder -> tester -> debugger -> optimizer -> synthesizer."""
    return (
        [
            {"role": "planner", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("planner", task)},
            {"role": "coder", "model_tier": "fast", "fallback_tier": "reasoner",
             "prompt": _role_prompt("coder", task)},
            {"role": "tester", "model_tier": "budget", "fallback_tier": "",
             "prompt": _role_prompt("tester", task)},
            {"role": "debugger", "model_tier": "fast", "fallback_tier": "reasoner",
             "prompt": _role_prompt("debugger", task)},
            {"role": "optimizer", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("optimizer", task)},
            {"role": "synthesizer", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("synthesizer", task)},
        ],
        [
            {"from_idx": 0, "to_idx": 1, "flow_type": "message", "gate": "open"},
            {"from_idx": 1, "to_idx": 2, "flow_type": "message", "gate": "open"},
            {"from_idx": 2, "to_idx": 3, "flow_type": "message", "gate": "conditional"},
            {"from_idx": 3, "to_idx": 4, "flow_type": "message", "gate": "open"},
            {"from_idx": 4, "to_idx": 5, "flow_type": "message", "gate": "open"},
        ],
    )


def _pattern_verifier(task: str, difficulty: str) -> tuple[list[dict], list[dict]]:
    """Verifier pattern: planner -> coder -> verifier -> synthesizer."""
    return (
        [
            {"role": "planner", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("planner", task)},
            {"role": "coder", "model_tier": "fast", "fallback_tier": "reasoner",
             "prompt": _role_prompt("coder", task)},
            {"role": "verifier", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("verifier", task)},
            {"role": "synthesizer", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("synthesizer", task)},
        ],
        [
            {"from_idx": 0, "to_idx": 1, "flow_type": "message", "gate": "open"},
            {"from_idx": 1, "to_idx": 2, "flow_type": "message", "gate": "conditional"},
            {"from_idx": 2, "to_idx": 3, "flow_type": "message", "gate": "open"},
        ],
    )


def _pattern_double_review(task: str, difficulty: str) -> tuple[list[dict], list[dict]]:
    """Double review: planner -> coder -> reviewer_1 -> reviewer_2 -> synthesizer."""
    return (
        [
            {"role": "planner", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("planner", task)},
            {"role": "coder", "model_tier": "fast", "fallback_tier": "reasoner",
             "prompt": _role_prompt("coder", task)},
            {"role": "reviewer", "model_tier": "budget", "fallback_tier": "fast",
             "prompt": "First review pass: check correctness and edge cases. " + _role_prompt("reviewer", task)},
            {"role": "reviewer", "model_tier": "fast", "fallback_tier": "",
             "prompt": "Second review pass: check performance and code quality. " + _role_prompt("reviewer", task)},
            {"role": "synthesizer", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("synthesizer", task)},
        ],
        [
            {"from_idx": 0, "to_idx": 1, "flow_type": "message", "gate": "open"},
            {"from_idx": 1, "to_idx": 2, "flow_type": "message", "gate": "conditional"},
            {"from_idx": 2, "to_idx": 3, "flow_type": "message", "gate": "open"},
            {"from_idx": 3, "to_idx": 4, "flow_type": "message", "gate": "open"},
        ],
    )


def _pattern_analyst_coder_tester(task: str, difficulty: str) -> tuple[list[dict], list[dict]]:
    """Analyst + coder + tester: analyst -> coder -> tester -> synthesizer."""
    return (
        [
            {"role": "analyst", "model_tier": "reasoner", "fallback_tier": "",
             "prompt": _role_prompt("analyst", task)},
            {"role": "coder", "model_tier": "fast", "fallback_tier": "reasoner",
             "prompt": _role_prompt("coder", task)},
            {"role": "tester", "model_tier": "budget", "fallback_tier": "fast",
             "prompt": _role_prompt("tester", task)},
            {"role": "synthesizer", "model_tier": "fast", "fallback_tier": "",
             "prompt": _role_prompt("synthesizer", task)},
        ],
        [
            {"from_idx": 0, "to_idx": 1, "flow_type": "message", "gate": "open"},
            {"from_idx": 1, "to_idx": 2, "flow_type": "message", "gate": "open"},
            {"from_idx": 2, "to_idx": 3, "flow_type": "message", "gate": "open"},
        ],
    )


# Pattern registry: name -> (generator_func, min_nodes, difficulty_range)
PATTERNS: dict[str, tuple[Any, int, list[str]]] = {
    "sequential": (_pattern_sequential, 2, ["simple"]),
    "sequential_3": (_pattern_sequential_3, 3, ["simple", "moderate"]),
    "avr": (_pattern_avr, 4, ["moderate", "complex"]),
    "debate": (_pattern_debate, 5, ["moderate", "complex"]),
    "parallel": (_pattern_parallel, 4, ["moderate", "complex"]),
    "diamond": (_pattern_diamond, 5, ["moderate", "complex"]),
    "funnel": (_pattern_funnel, 5, ["complex"]),
    "expert_panel": (_pattern_expert_panel, 6, ["complex"]),
    "pipeline": (_pattern_pipeline, 6, ["complex"]),
    "verifier": (_pattern_verifier, 4, ["moderate", "complex"]),
    "double_review": (_pattern_double_review, 5, ["moderate", "complex"]),
    "analyst_coder_tester": (_pattern_analyst_coder_tester, 4, ["moderate"]),
}


# ---------------------------------------------------------------------------
# Reasoning templates per difficulty + pattern
# ---------------------------------------------------------------------------

REASONING_SIMPLE = [
    "Simple single-function task. A {n}-node topology ({roles}) is sufficient. "
    "No adaptation needed — the task is straightforward enough that a fast-tier model handles it reliably. "
    "Adding checkpoints or fallbacks would waste tokens without improving success rate.",
    "Straightforward coding task that requires only basic implementation. "
    "A minimal {n}-node ({roles}) topology avoids unnecessary overhead. "
    "Fast-tier models can handle this without review or planning stages.",
    "Direct implementation task. The {roles} pattern keeps it lean — "
    "no need for multi-step verification on a problem with clear specs and no ambiguity.",
    "This is a well-defined single-function problem. Using {n} nodes ({roles}) "
    "minimizes token cost while still ensuring clean code output.",
    "Low-complexity task that maps directly to a known pattern. "
    "{n}-node {roles} topology is the cost-optimal choice. "
    "No checkpoint needed as failure probability is very low.",
]

REASONING_MODERATE = [
    "Moderate complexity task requiring careful implementation. A {n}-node topology "
    "({roles}) provides the right balance between thoroughness and token efficiency. "
    "Checkpoint at the coder node allows model upgrade if initial implementation fails.",
    "This task has subtle edge cases that benefit from a planning phase. "
    "The {roles} pattern with {n} nodes separates concern: reasoning in the planner, "
    "implementation in the coder, verification in the reviewer.",
    "Multi-step problem requiring both algorithmic thinking and careful implementation. "
    "The {n}-node {pattern} topology uses {roles} — the planner handles algorithmic design "
    "while the coder focuses on correct implementation.",
    "Task with non-trivial correctness requirements. "
    "Using a {pattern} topology ({n} nodes: {roles}) ensures the solution is both planned and reviewed. "
    "Fallback tier on the coder enables recovery from initial implementation mistakes.",
    "This problem benefits from separating design and implementation concerns. "
    "The {pattern} pattern ({roles}) handles this naturally. "
    "Adaptation is enabled at the coder checkpoint for moderate-risk recovery.",
]

REASONING_COMPLEX = [
    "Complex problem requiring deep algorithmic reasoning and careful verification. "
    "The {n}-node {pattern} topology ({roles}) provides comprehensive coverage: "
    "planning, implementation, review, and optimization. Multiple checkpoints with fallback "
    "tiers ensure recovery from any stage failure.",
    "High-difficulty task with multiple interacting concerns. A {pattern} topology ({n} nodes) "
    "is needed: {roles}. The reasoner tier on planning nodes provides the deep thinking needed, "
    "while checkpoint-based adaptation catches implementation errors early.",
    "Competition-level problem requiring formal analysis and robust implementation. "
    "The {pattern} pattern with {n} nodes ({roles}) separates mathematical derivation from "
    "coding, with dedicated verification. Multiple fallback tiers ensure quality.",
    "This task combines algorithmic complexity with tricky implementation details. "
    "A {n}-node {pattern} topology handles this via {roles}. "
    "Edge-level state flow preserves context between stages. "
    "Adaptation checkpoints on critical nodes enable model upgrades if quality drops.",
    "Multi-faceted problem requiring parallel analysis and synthesis. "
    "The {pattern} topology ({roles}, {n} nodes) leverages specialized roles — "
    "each node focuses on one aspect while edge-level communication coordinates. "
    "Reasoner tier on analysis nodes, fast on implementation, budget on routine review.",
]


def _generate_reasoning(difficulty: str, pattern_name: str, nodes: list[dict]) -> str:
    """Generate contextual reasoning for a topology choice."""
    roles = " -> ".join(n["role"] for n in nodes)
    n_nodes = len(nodes)

    if difficulty == "simple":
        template = random.choice(REASONING_SIMPLE)
    elif difficulty == "moderate":
        template = random.choice(REASONING_MODERATE)
    else:
        template = random.choice(REASONING_COMPLEX)

    return template.format(
        n=n_nodes,
        roles=roles,
        pattern=pattern_name,
    )


# ---------------------------------------------------------------------------
# Adaptation metadata generation
# ---------------------------------------------------------------------------

def _generate_adaptation(nodes: list[dict], difficulty: str, include_adaptation: bool) -> dict:
    """Generate adaptation metadata for a topology."""
    if not include_adaptation or difficulty == "simple":
        return {
            "checkpoints": [],
            "max_upgrades": 0,
            "max_reroutes": 0,
            "quality_threshold": 0.5,
        }

    # Find nodes that could be checkpoints (coders, testers, debuggers)
    checkpoint_roles = {"coder", "tester", "debugger", "optimizer"}
    candidate_indices = [
        i for i, n in enumerate(nodes)
        if n["role"] in checkpoint_roles
    ]

    if not candidate_indices:
        return {
            "checkpoints": [],
            "max_upgrades": 0,
            "max_reroutes": 0,
            "quality_threshold": 0.5,
        }

    # Select 1-2 checkpoints
    n_checkpoints = min(len(candidate_indices), random.choice([1, 1, 1, 2]))
    checkpoints = sorted(random.sample(candidate_indices, n_checkpoints))

    # Set fallback_tier on checkpoint nodes
    for idx in checkpoints:
        node = nodes[idx]
        if node["model_tier"] == "budget":
            node["fallback_tier"] = "fast"
        elif node["model_tier"] == "fast":
            node["fallback_tier"] = "reasoner"
        # reasoner nodes already at max

    max_upgrades = random.choice([1, 1, 2]) if difficulty == "complex" else 1
    threshold = random.choice([0.4, 0.5, 0.5, 0.6])

    return {
        "checkpoints": checkpoints,
        "max_upgrades": max_upgrades,
        "max_reroutes": 0 if difficulty == "moderate" else random.choice([0, 0, 1]),
        "quality_threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Recovery scenario generation
# ---------------------------------------------------------------------------

FAILURE_TYPES = ["WRONG_ANSWER", "RUNTIME_ERROR", "TIMEOUT", "NO_CODE"]

FAILURE_DESCRIPTIONS: dict[str, list[str]] = {
    "WRONG_ANSWER": [
        "The coder used 0-based indexing where the problem expects 1-based output.",
        "The implementation returns incorrect results for empty input.",
        "Off-by-one error in the loop boundary causes wrong output for edge cases.",
        "The algorithm handles the base case incorrectly.",
        "Incorrect modular arithmetic causes wrong results for large inputs.",
        "The implementation misses a corner case when the input contains duplicates.",
        "The solution returns the wrong data type (list instead of tuple).",
        "The sorting is not stable, violating the output requirements.",
    ],
    "RUNTIME_ERROR": [
        "The coder tried to access index -1 on an empty list, causing IndexError.",
        "Division by zero when the input contains a zero-length sequence.",
        "Stack overflow from unbounded recursion on deeply nested input.",
        "KeyError when accessing a dictionary with an unexpected key.",
        "TypeError from mixing int and str types in comparison.",
        "AttributeError: NoneType has no attribute when optional value is missing.",
        "RecursionError: maximum recursion depth exceeded for input size > 1000.",
        "MemoryError: the solution creates an unnecessarily large intermediate list.",
    ],
    "TIMEOUT": [
        "The coder used O(n^2) nested loops where an O(n log n) approach was needed.",
        "The recursive solution has exponential time complexity without memoization.",
        "Redundant recomputation in the inner loop makes the solution too slow for n > 10000.",
        "The solution creates too many intermediate copies, causing excessive GC pressure.",
        "Unnecessary sorting in each iteration increases complexity from O(n) to O(n^2 log n).",
        "The regex pattern causes catastrophic backtracking on certain inputs.",
    ],
    "NO_CODE": [
        "The coder discussed the approach theoretically but never wrote the actual implementation.",
        "The output contained only pseudocode without any executable Python code.",
        "The coder generated comments and documentation but no actual function body.",
    ],
}


def _generate_recovery(task_id: str, prompt: str, nodes: list[dict], edges: list[dict],
                       difficulty: str, adaptation: dict) -> dict | None:
    """Generate a recovery scenario entry."""
    if not adaptation.get("checkpoints"):
        return None

    checkpoint_idx = adaptation["checkpoints"][0]
    node = nodes[checkpoint_idx]
    failure_type = random.choice(FAILURE_TYPES)
    failure_desc = random.choice(FAILURE_DESCRIPTIONS[failure_type])

    old_tier = node["model_tier"]
    # Determine upgrade path
    if old_tier == "budget":
        new_tier = random.choice(["fast", "reasoner"])
    elif old_tier == "fast":
        new_tier = "reasoner"
    else:
        new_tier = "reasoner"  # already at max, retry with same

    # Build recovered topology
    recovered_nodes = [dict(n) for n in nodes]
    recovered_nodes[checkpoint_idx]["model_tier"] = new_tier
    recovered_nodes[checkpoint_idx]["fallback_tier"] = new_tier

    recovered_adaptation = dict(adaptation)
    recovered_adaptation["max_upgrades"] = max(0, adaptation["max_upgrades"] - 1)

    initial_reasoning = f"Standard {difficulty} topology with adaptation enabled for: {prompt}"
    recovered_reasoning = (
        f"After {failure_type} at node {checkpoint_idx}, controller upgraded "
        f"{old_tier} -> {new_tier}. {failure_desc.rstrip('.')}."
    )

    return {
        "task_id": task_id,
        "prompt": prompt,
        "initial_topology": {
            "reasoning": initial_reasoning,
            "difficulty": difficulty,
            "adaptation": adaptation,
            "nodes": nodes,
            "edges": edges,
        },
        "failure_scenario": {
            "failed_node_idx": checkpoint_idx,
            "failure_type": failure_type,
            "failure_description": failure_desc,
        },
        "recovery_action": {
            "action": "upgrade_model",
            "target_node": checkpoint_idx,
            "old_tier": old_tier,
            "new_tier": new_tier,
            "description": (
                f"Controller detects quality < {adaptation['quality_threshold']} "
                f"at checkpoint node {checkpoint_idx} ({node['role']}). "
                f"Upgrades {node['role']} from {old_tier} to {new_tier}."
            ),
        },
        "recovered_topology": {
            "reasoning": recovered_reasoning,
            "difficulty": difficulty,
            "adaptation": recovered_adaptation,
            "nodes": recovered_nodes,
            "edges": edges,
        },
        "expected_outcome": (
            f"PASSED — the {new_tier} tier handles the failure mode "
            f"({failure_desc.lower().rstrip('.')}) through deeper reasoning."
        ),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "difficulty": difficulty,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_entry(entry: dict) -> list[str]:
    """Validate a topology entry. Returns list of error messages (empty = valid)."""
    errors = []

    topology = entry.get("topology", {})
    if not topology:
        errors.append("Missing topology")
        return errors

    # Check YAML-parsable
    try:
        yaml_str = yaml.dump(topology, default_flow_style=False)
        yaml.safe_load(yaml_str)
    except Exception as e:
        errors.append(f"YAML parse error: {e}")

    # Check nodes
    nodes = topology.get("nodes", [])
    if not nodes:
        errors.append("No nodes")
    for i, node in enumerate(nodes):
        if "role" not in node:
            errors.append(f"Node {i} missing role")
        if "model_tier" not in node:
            errors.append(f"Node {i} missing model_tier")
        if node.get("model_tier") not in ("reasoner", "fast", "budget"):
            errors.append(f"Node {i} invalid model_tier: {node.get('model_tier')}")

    # Check edges reference valid indices
    edges = topology.get("edges", [])
    n_nodes = len(nodes)
    for i, edge in enumerate(edges):
        fi = edge.get("from_idx")
        ti = edge.get("to_idx")
        if fi is None or ti is None:
            errors.append(f"Edge {i} missing from_idx or to_idx")
        elif fi >= n_nodes or ti >= n_nodes or fi < 0 or ti < 0:
            errors.append(f"Edge {i} references invalid node: {fi} -> {ti}")
        if edge.get("flow_type") not in ("message", "control", "state"):
            errors.append(f"Edge {i} invalid flow_type: {edge.get('flow_type')}")

    # Check acyclicity (simple DFS)
    if not errors:
        adj: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
        for edge in edges:
            adj[edge["from_idx"]].append(edge["to_idx"])

        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n_nodes
        has_cycle = False

        def dfs(u: int) -> bool:
            nonlocal has_cycle
            color[u] = GRAY
            for v in adj[u]:
                if color[v] == GRAY:
                    has_cycle = True
                    return True
                if color[v] == WHITE and dfs(v):
                    return True
            color[u] = BLACK
            return False

        for i in range(n_nodes):
            if color[i] == WHITE:
                dfs(i)
        if has_cycle:
            errors.append("Topology has a cycle")

    # Check reasoning
    if not topology.get("reasoning"):
        errors.append("Missing reasoning")

    # Check difficulty
    if topology.get("difficulty") not in ("simple", "moderate", "complex"):
        errors.append(f"Invalid difficulty: {topology.get('difficulty')}")

    # Check last node is synthesizer
    if nodes and nodes[-1].get("role") != "synthesizer":
        errors.append(f"Last node must be synthesizer, got: {nodes[-1].get('role')}")

    return errors


def _validate_recovery_entry(entry: dict) -> list[str]:
    """Validate a recovery entry."""
    errors = []
    for key in ("initial_topology", "recovered_topology"):
        topo = entry.get(key, {})
        fake = {"topology": topo, "task_id": entry.get("task_id"), "prompt": entry.get("prompt")}
        sub_errors = _validate_entry(fake)
        errors.extend([f"{key}: {e}" for e in sub_errors])
    return errors


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def _select_pattern(difficulty: str) -> tuple[str, Any]:
    """Select a topology pattern appropriate for the difficulty."""
    candidates = [
        (name, func)
        for name, (func, _, diffs) in PATTERNS.items()
        if difficulty in diffs
    ]
    name, func = random.choice(candidates)
    return name, func


def _generate_task_prompt(task_type: str) -> str:
    """Generate a random prompt for a given task type."""
    templates = TASK_TYPES[task_type]
    template = random.choice(templates)
    return _fill_template(template)


def _make_task_id(prefix: str, idx: int, task_type: str) -> str:
    """Generate a deterministic task ID."""
    return f"{prefix}/{task_type}_{idx:05d}"


def generate_entries(count: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Generate synthetic topology entries.

    Returns (standard_entries, recovery_entries).
    """
    random.seed(seed)

    # Difficulty distribution: 30% simple, 40% moderate, 30% complex
    n_simple = int(count * 0.30)
    n_moderate = int(count * 0.40)
    n_complex = count - n_simple - n_moderate

    difficulties = (
        ["simple"] * n_simple
        + ["moderate"] * n_moderate
        + ["complex"] * n_complex
    )
    random.shuffle(difficulties)

    task_type_list = list(TASK_TYPES.keys())
    standard_entries: list[dict] = []
    recovery_entries: list[dict] = []

    # Track prompt hashes to avoid exact duplicates
    seen_hashes: set[str] = set()

    for idx in range(count):
        difficulty = difficulties[idx]
        task_type = task_type_list[idx % len(task_type_list)]

        # Generate unique prompt
        for _attempt in range(10):
            prompt = _generate_task_prompt(task_type)
            h = hashlib.md5(prompt.encode()).hexdigest()[:12]
            if h not in seen_hashes:
                seen_hashes.add(h)
                break

        task_id = _make_task_id("Synthetic", idx, task_type)

        # Select pattern
        pattern_name, pattern_func = _select_pattern(difficulty)

        # Generate topology
        nodes, edges = pattern_func(prompt, difficulty)

        # Decide whether to include adaptation (40% of moderate/complex, never simple)
        include_adaptation = (
            difficulty != "simple"
            and random.random() < 0.55  # slightly above 40% to account for simple
        )

        adaptation = _generate_adaptation(nodes, difficulty, include_adaptation)

        # Generate reasoning
        reasoning = _generate_reasoning(difficulty, pattern_name, nodes)

        topology = {
            "reasoning": reasoning,
            "difficulty": difficulty,
            "adaptation": adaptation,
            "nodes": nodes,
            "edges": edges,
        }

        entry = {
            "task_id": task_id,
            "prompt": prompt,
            "topology": topology,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "difficulty": difficulty,
        }

        # Validate
        validation_errors = _validate_entry(entry)
        if validation_errors:
            log.warning("Validation failed for %s: %s", task_id, validation_errors)
            continue

        standard_entries.append(entry)

        # Recovery scenario (10% of entries with checkpoints)
        if (
            adaptation.get("checkpoints")
            and random.random() < 0.25  # ~10% overall since ~40% have adaptation
        ):
            recovery = _generate_recovery(
                task_id + "_recovery", prompt, nodes, edges, difficulty, adaptation
            )
            if recovery:
                rec_errors = _validate_recovery_entry(recovery)
                if rec_errors:
                    log.warning("Recovery validation failed for %s: %s", task_id, rec_errors)
                else:
                    recovery_entries.append(recovery)

    return standard_entries, recovery_entries


def main():
    parser = argparse.ArgumentParser(description="Generate 10K+ synthetic topology dataset")
    parser.add_argument("--count", type=int, default=8500,
                        help="Number of standard entries to generate (default: 8500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/synthetic_topologies_10k.jsonl")
    parser.add_argument("--output-recovery", default="data/synthetic_recovery_scenarios.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    recovery_path = Path(args.output_recovery)

    log.info("Generating %d synthetic topology entries (seed=%d)...", args.count, args.seed)
    standard, recovery = generate_entries(args.count, args.seed)

    # Write standard entries
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in standard:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("Wrote %d standard entries to %s", len(standard), output_path)

    # Write recovery entries
    if recovery:
        with open(recovery_path, "w", encoding="utf-8") as f:
            for entry in recovery:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.info("Wrote %d recovery entries to %s", len(recovery), recovery_path)

    # Stats
    difficulties = {}
    patterns_used: dict[str, int] = {}
    task_types_used: dict[str, int] = {}
    has_adaptation = 0
    total_nodes = 0
    total_edges = 0
    flow_types: dict[str, int] = {}

    for entry in standard:
        d = entry["difficulty"]
        difficulties[d] = difficulties.get(d, 0) + 1
        total_nodes += entry["node_count"]
        total_edges += entry["edge_count"]
        if entry["topology"]["adaptation"].get("checkpoints"):
            has_adaptation += 1
        for edge in entry["topology"]["edges"]:
            ft = edge["flow_type"]
            flow_types[ft] = flow_types.get(ft, 0) + 1

    # Extract task type from task_id
    for entry in standard:
        tid = entry["task_id"]
        parts = tid.split("/")
        if len(parts) > 1:
            tt = parts[1].rsplit("_", 1)[0] if "_" in parts[1] else parts[1]
            task_types_used[tt] = task_types_used.get(tt, 0) + 1

    log.info("=== Statistics ===")
    log.info("Total standard entries: %d", len(standard))
    log.info("Total recovery entries: %d", len(recovery))
    log.info("Combined entry count: %d", len(standard) + len(recovery) * 2)
    log.info("Difficulty distribution: %s", difficulties)
    log.info("Entries with adaptation: %d (%.1f%%)", has_adaptation, 100 * has_adaptation / max(len(standard), 1))
    log.info("Avg nodes: %.1f, Avg edges: %.1f", total_nodes / max(len(standard), 1), total_edges / max(len(standard), 1))
    log.info("Edge flow types: %s", flow_types)
    log.info("Task types covered: %d", len(task_types_used))
    log.info("Task type distribution: %s", dict(sorted(task_types_used.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
