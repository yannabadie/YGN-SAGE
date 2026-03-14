"""Centralized constants for YGN-SAGE.

Every numeric constant is named, documented, and sourced.
NO magic numbers in production code -- all values imported from here.

Sources:
- [BENCH] = calibrated from benchmark evidence (cite which)
- [PAPER] = from research paper (cite arXiv ID)
- [ENG]   = engineering decision (explain rationale)
"""
from __future__ import annotations

# -- Routing thresholds -------------------------------------------------------
# S1/S2/S3 cognitive system boundaries
# [BENCH] Calibrated on 50 human-labeled ground truth tasks (2026-03-11)
S1_COMPLEXITY_CEIL = 0.50       # Below this -> S1 (fast/intuitive)
S1_UNCERTAINTY_CEIL = 0.30      # Below this -> S1
S3_COMPLEXITY_FLOOR = 0.65      # Above this -> S3 (formal)
S3_UNCERTAINTY_FLOOR = 0.60     # Above this -> S3
SPECULATIVE_ZONE_MIN = 0.35     # [PAPER] AdaptOrch speculative zone
SPECULATIVE_ZONE_MAX = 0.55     # [PAPER] AdaptOrch speculative zone

# S2 sub-tier boundary: tasks above this get "reasoner", below get "mutator"
# [ENG] Midpoint between S1 ceiling (0.50) and S3 floor (0.65)
S2_REASONER_COMPLEXITY_FLOOR = 0.55

# S3 sub-tier boundary: tasks above this get "codex", below get "reasoner"
# [ENG] Very high complexity reserved for most powerful model
S3_CODEX_COMPLEXITY_FLOOR = 0.80

# CGRS self-braking (Convergence-Guided Resource Scaling)
# [ENG] Tuned on HumanEval+/MBPP+ benchmarks (2026-03-10)
BRAKE_WINDOW = 3                # Number of recent outputs to check
BRAKE_ENTROPY_THRESHOLD = 0.15  # Shannon entropy below this = converged
BRAKE_HISTORY_MAXLEN = 10       # Deque maxlen for entropy history

# Confidence thresholds for AdaptiveRouter stages
# [ENG] Stage 0 (structural) must be very confident to skip ONNX
ADAPTIVE_C0_THRESHOLD = 0.85    # Stage 0 confidence to accept routing
ADAPTIVE_C1_THRESHOLD = 0.70    # Stage 1 confidence to accept routing

# -- Quality estimation weights -----------------------------------------------
# [BENCH] Calibrated from 600 quality triples (DistilBERT training, 2026-03-12)
QUALITY_BASELINE = 0.30         # Non-empty response baseline (Signal 1)
QUALITY_LENGTH_WEIGHT = 0.20    # Adequate length signal (Signal 2)
QUALITY_CODE_WEIGHT = 0.20      # Code presence for code tasks (Signal 3)
QUALITY_ERROR_WEIGHT = 0.15     # Error/traceback absence (Signal 4)
QUALITY_AVR_WEIGHT = 0.15       # AVR convergence signal (Signal 5)

# Quality sub-thresholds
# [ENG] Non-code tasks get full length score if >= 1 word
QUALITY_NONCODE_BASELINE = 0.1  # Non-code task without code presence

# AVR convergence tiers
# [ENG] Fewer iterations = higher quality (converged faster)
QUALITY_AVR_FAST = 0.15         # <= 2 iterations
QUALITY_AVR_MEDIUM = 0.10       # <= 4 iterations
QUALITY_AVR_SLOW = 0.05         # > 4 iterations

# Length adequacy denominators
# [ENG] Short task prompts expect less output; long prompts expect more
QUALITY_LENGTH_DENOM_SHORT = 20   # Denominator for task_words < 10
QUALITY_LENGTH_DENOM_LONG = 50    # Denominator for task_words >= 10

# -- Agent loop limits --------------------------------------------------------
# [ENG] Tuned on HumanEval+/MBPP+ benchmarks (2026-03-10)
S2_AVR_MAX_ITERATIONS = 3       # Max AVR retry cycles for S2 code tasks
S2_MAX_RETRIES_BEFORE_ESCALATION = 2  # S2 failures before S3 escalation
S3_MAX_RETRIES = 2              # Max S3 CEGAR repair attempts
MAX_AGENT_MESSAGES = 40         # Context window protection
MAX_AGENT_STEPS = 20            # Agent loop step limit
STAGNATION_WINDOW = 3           # Consecutive identical outputs for detection

# -- Drift monitor ------------------------------------------------------------
# [PAPER] 3-signal behavioral drift detection (monitoring/drift.py)
DRIFT_CHECK_INTERVAL = 10       # Analyze drift every N events
DRIFT_WEIGHT_LATENCY = 0.40     # Latency trend weight
DRIFT_WEIGHT_ERRORS = 0.40      # Error rate weight
DRIFT_WEIGHT_COST = 0.20        # Cost trend weight
DRIFT_ACTION_CONTINUE = 0.40    # Below this -> CONTINUE
DRIFT_ACTION_SWITCH = 0.70      # Below this -> SWITCH_MODEL, above -> RESET_AGENT
DRIFT_CATASTROPHIC_FACTOR = 0.85  # Single catastrophic signal floor multiplier
# Latency trend: 3x increase maps to 1.0 via (ratio - 1) / LATENCY_RATIO_SCALE
DRIFT_LATENCY_RATIO_SCALE = 2.0
# Cost trend: 6x increase maps to 1.0 via (ratio - 1) / COST_RATIO_SCALE
DRIFT_COST_RATIO_SCALE = 5.0
# Minimum events for meaningful trend analysis
DRIFT_MIN_EVENTS_FOR_TREND = 3

# -- Shadow routing Phase 5 gate ----------------------------------------------
# [ENG] Evidence-based gate for Rust router promotion
SHADOW_SOFT_TRACES = 500        # Minimum traces for soft gate
SHADOW_SOFT_DIVERGENCE = 0.10   # Max divergence rate for soft gate
SHADOW_HARD_TRACES = 1000       # Minimum traces for hard gate
SHADOW_HARD_DIVERGENCE = 0.05   # Max divergence rate for hard gate
SHADOW_MAX_TRACE_BYTES = 10 * 1024 * 1024  # 10 MB rotation threshold

# -- Memory system ------------------------------------------------------------
# [ENG] Pressure-triggered compression (MEM1 pattern)
MEMORY_COMPRESSION_THRESHOLD = 20  # Events before compression trigger
MEMORY_KEEP_RECENT = 5             # Recent events preserved after compression
RELEVANCE_GATE_THRESHOLD = 0.30    # [BENCH] CRAG gate, Sprint 3 evidence (2026-03)

# -- Topology limits ----------------------------------------------------------
MAX_TOPOLOGY_AGENTS = 4         # [ENG] Max agents in LLM-synthesized topology
LLM_SYNTHESIS_MIN_SYSTEM = 2    # [ENG] Only attempt LLM topology for S2/S3

# -- Exploration budgets ------------------------------------------------------
# [ENG] Budget allocation per cognitive system
DEFAULT_BUDGET_USD = 10.0       # Default per-task budget
EXPLORATION_BUDGET_LOW = 0.30   # [ENG] Low exploration for S1/S2 tasks
EXPLORATION_BUDGET_HIGH = 0.50  # [ENG] Higher exploration for S3 tasks

# -- Guardrails ---------------------------------------------------------------
COST_GUARDRAIL_MAX_USD = 10.0   # [ENG] Default cost budget
OUTPUT_GUARDRAIL_MIN_LENGTH = 1 # [ENG] Minimum output chars (reject empty)

# -- kNN routing --------------------------------------------------------------
# [PAPER] arXiv 2505.12601: kNN on embeddings
KNN_K = 5                       # Number of nearest neighbors
KNN_DISTANCE_THRESHOLD = 0.30   # Minimum cosine similarity for valid match

# -- kNN-to-profile conversion ------------------------------------------------
# [ENG] Synthetic CognitiveProfile values for kNN tier results
# These must land in the correct routing zone for _route_from_profile()
KNN_S1_COMPLEXITY = 0.2         # Must be <= S1_COMPLEXITY_CEIL
KNN_S1_UNCERTAINTY = 0.1        # Must be <= S1_UNCERTAINTY_CEIL
KNN_S2_COMPLEXITY = 0.5         # Must be > S1 ceil but < S3 floor
KNN_S2_UNCERTAINTY = 0.4        # Must be > S1 ceil but < S3 floor
KNN_S3_COMPLEXITY = 0.8         # Must be > S3_COMPLEXITY_FLOOR
KNN_S3_UNCERTAINTY = 0.7        # Must be > S3_UNCERTAINTY_FLOOR

# -- Orchestrator quality thresholds ------------------------------------------
# [ENG] Quality-gated cascade (FrugalGPT pattern)
ORCHESTRATOR_S1_QUALITY = 0.40  # S1: accept lower quality, optimize cost
ORCHESTRATOR_S2_QUALITY = 0.60  # S2: balanced quality/cost
ORCHESTRATOR_S3_QUALITY = 0.80  # S3: high quality requirement
MAX_CASCADE_ATTEMPTS = 3        # FrugalGPT: max provider cascade retries

# -- Heuristic fallback -------------------------------------------------------
# [ENG] Degraded keyword-count heuristic (last resort, no ONNX/kNN)
HEURISTIC_COMPLEXITY_DENOM = 3.0  # hits / DENOM -> complexity score
HEURISTIC_QUESTION_UNCERTAINTY = 0.3  # Uncertainty when "?" present
HEURISTIC_DEFAULT_UNCERTAINTY = 0.2   # Uncertainty otherwise
HEURISTIC_FALLBACK_CONFIDENCE = 0.5   # Default confidence for heuristic route

# -- Entropy probe (AdaptiveRouter Stage 2) -----------------------------------
# [ENG] Thresholds for entropy-based routing adjustment
ENTROPY_LOW_THRESHOLD = 0.30    # Below -> high confidence (predictable)
ENTROPY_HIGH_THRESHOLD = 0.70   # Above -> lower confidence (unpredictable)
ENTROPY_LOW_CONFIDENCE = 0.75   # Confidence when entropy is low
ENTROPY_HIGH_CONFIDENCE = 0.65  # Confidence when entropy is high
ENTROPY_MID_CONFIDENCE = 0.60   # Confidence when entropy is mid-range

# -- Max tokens per routing tier ----------------------------------------------
# [ENG] Context window allocation per cognitive system
MAX_TOKENS_S1 = 2048            # S1: short responses
MAX_TOKENS_S2 = 4096            # S2: moderate responses
MAX_TOKENS_S3 = 8192            # S3: long formal reasoning

# -- Timeouts (seconds) -------------------------------------------------------
DEFAULT_HTTP_TIMEOUT = 60       # [ENG] Standard HTTP timeout
CODEX_CLI_TIMEOUT = 120         # [ENG] Codex CLI timeout
EVAL_TASK_TIMEOUT = 120.0       # [ENG] Per-task evaluation timeout

# -- Default cost fallback -----------------------------------------------------
# [ENG] When model not in cost table, use this per-1K-token rate
DEFAULT_COST_PER_1K = 0.001
