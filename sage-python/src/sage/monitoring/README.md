# Monitoring

Runtime monitoring for agent performance and behavior drift detection.

## Modules

### `drift.py` -- DriftMonitor (3-signal baseline)

Base drift detector with 3 signals: latency trend (40%), error rate (40%), cost trend (20%). Catastrophic factor (0.85) ensures a single bad signal is never masked. Actions: CONTINUE / SWITCH_MODEL / RESET_AGENT.

### `extended_drift.py` -- ExtendedDriftMonitor (12-dimension ASI)

Agent Stability Index (arXiv 2601.04170) extending the base 3 signals with 9 new dimensions:

| # | Signal | Weight | Source |
|---|--------|--------|--------|
| 4 | Semantic drift | 8% | Embedding distance between consecutive outputs |
| 5 | Behavioral consistency | 10% | Action sequence variance (Levenshtein) |
| 6 | Topic coherence | 5% | Keyword overlap between task and response |
| 7 | Reasoning depth | 4% | Chain-of-thought length trend |
| 8 | Memory utilization | 3% | S-MMU retrieval hit rate trend |
| 9 | Tool diversity | 3% | Shannon entropy of tool usage |
| 10 | Output stability | 2% | Coefficient of variation of response lengths |
| 11 | Confidence trend | 2% | Write gate confidence trend |
| 12 | Coordination | 1% | Sub-agent spawn/complete ratio |

All weights from paper, documented as "subject to ablation".

Includes **BehaviorTracker**: sliding window of action sequences with normalized Levenshtein distance. Paper 2602.11619 shows consistent agents (score > 0.7) achieve 80-92% accuracy vs 25-60% for inconsistent agents.

## Usage

DriftMonitor is instantiated during boot. ExtendedDriftMonitor wraps it and adds the 9 new dimensions.
