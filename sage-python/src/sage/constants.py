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

# -- Quality estimation weights (DELETED 2026-04-18) --------------------------
# The 5-signal heuristic (QUALITY_BASELINE / QUALITY_LENGTH_WEIGHT /
# QUALITY_CODE_WEIGHT / QUALITY_ERROR_WEIGHT / QUALITY_AVR_WEIGHT plus
# their sub-thresholds) is BANNED under .claude/rules/critical-directives.md §2
# and §6 of docs/heuristics-needing-ablation.md: the correlation with
# ground-truth quality was r=0.34 Pearson, mediocre compared to the Rust
# Z3 QualityLabeler (formal verification). No production code referenced
# these constants — only a `test_quality_weights_sum_to_one` regression
# test, now also deleted. Use `sage_core.QualityLabeler` instead.
#
# `test_quality_estimator_v2.py` keeps an assertion that these constants
# MUST NOT reappear in the QualityEstimator source — don't re-introduce
# them without a §2 carve-out.

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

# -- Composite write gate (arXiv 2603.15994) -----------------------------------
# [PAPER] Salience gate weights, subject to ablation
SALIENCE_WEIGHT_CONFIDENCE = 0.25   # Model confidence in the content
SALIENCE_WEIGHT_NOVELTY = 0.30      # 1 - max_similarity to existing entries
SALIENCE_WEIGHT_RELIABILITY = 0.20  # Source tier reputation score
SALIENCE_WEIGHT_RECENCY = 0.10      # Time decay since task start
SALIENCE_WEIGHT_RELEVANCE = 0.15    # Task-content overlap (RelevanceGate)
SALIENCE_NOVELTY_SIM_THRESHOLD = 0.90  # Near-duplicate if cosine > this
SALIENCE_DEFAULT_THRESHOLD = 0.35   # [PAPER] Composite gate threshold

# -- Online evolution gating (SA-3) --------------------------------------------
# [ENG] Calibrated initial values, subject to ablation
EVOLUTION_MIN_OUTCOMES = 5          # Minimum outcomes before first evolution
EVOLUTION_COOLDOWN_OUTCOMES = 3     # Min new outcomes between evolution runs
EVOLUTION_SATURATION_THRESHOLD = 0.80  # Stop evolving when archive is this full
EVOLUTION_ONLINE_POP_SIZE = 5       # Population size for online evolution pass
EVOLUTION_ONLINE_GENERATIONS = 2    # Generations per online evolution pass

# -- Evolution Memory (CORAL arXiv 2604.01658) ---------------------------------
# [PAPER] Persistent mutation/skill store. Skills extracted via SQL aggregation.
EVOLUTION_MEMORY_MIN_SAMPLES = 5            # Min mutations before skill extraction
EVOLUTION_MEMORY_SKILL_DECAY_HALF_LIFE_DAYS = 30.0  # Temporal decay on skills
EVOLUTION_MEMORY_SKILL_TOP_K = 3            # Max skills injected per mutation prompt

# -- Extended drift monitor / Agent Stability Index (arXiv 2601.04170) ----------
# [PAPER] 12-dimension ASI. The first 3 reuse the legacy signals via DriftMonitor.
# These 9 new dimensions are added by ExtendedDriftMonitor. Subject to ablation.
ASI_WEIGHT_SEMANTIC = 0.08         # Embedding distance between consecutive outputs
ASI_WEIGHT_BEHAVIORAL = 0.10       # Action sequence variance
ASI_WEIGHT_TOPIC = 0.05            # Keyword overlap between task and response
ASI_WEIGHT_REASONING_DEPTH = 0.04  # Chain-of-thought length trend
ASI_WEIGHT_MEMORY_UTIL = 0.03      # S-MMU retrieval hit rate trend
ASI_WEIGHT_TOOL_DIVERSITY = 0.03   # Shannon entropy of tool usage
ASI_WEIGHT_OUTPUT_STABILITY = 0.02 # Coefficient of variation of response lengths
ASI_WEIGHT_CONFIDENCE_TREND = 0.02 # Write gate confidence trend
ASI_WEIGHT_COORDINATION = 0.01     # Sub-agent spawn/complete ratio
ASI_BEHAVIORAL_WINDOW = 10         # Sliding window for behavior consistency

# -- Consolidation pipeline ----------------------------------------------------
# [ENG] Inter-tier memory consolidation
CONSOLIDATION_INTERVAL_STEPS = 10   # Consolidation runs every N agent loop steps
CONSOLIDATION_BATCH_SIZE = 20       # Max episodic entries to consolidate per pass
BANDIT_FLUSH_INTERVAL = 10          # [ENG] Persist bandit+archive state every N pipeline tasks
MAX_TOOL_CREATIONS_PER_RUN = 2      # [ENG] ToolForge: max tools created per pipeline run

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
