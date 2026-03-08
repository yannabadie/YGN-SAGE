# Strategy

Strategy pillar: cognitive routing and resource allocation for the agent system.

## Modules

### `metacognition.py` -- ComplexityRouter

Tripartite cognitive routing system (formerly MetacognitiveController; alias preserved for backward compatibility). Classifies tasks into three processing tiers:

- **S1 (Fast/Intuitive)** -- Simple lookups, formatting, direct answers. Routed to lightweight models (budget/fast tier).
- **S2 (Analytical/Reasoning)** -- Multi-step reasoning, code generation, analysis. Routed to reasoning-capable models.
- **S3 (Formal/Verification)** -- Tasks requiring formal proofs, Z3 constraints, or exhaustive verification. Routed to the most capable models.

**Key features:**

- **CGRS (Cognitive Growth Rate Stabilizer)** -- Self-braking mechanism that prevents runaway escalation from S1 to S3.
- **Speculative Zone (0.35-0.55)** -- When complexity score falls in the indecisive range, the router logs the ambiguity but proceeds with normal routing (no speculative dual-execution).

### `engine.py` -- StrategyEngine

High-level strategy orchestration. Maintains bounded history (`deque(maxlen=1000)`) of routing decisions and outcomes for feedback-driven adaptation.

### `allocator.py` -- Resource Allocator

Allocates compute budget and model resources based on task complexity and available quotas.

### `solvers.py` -- Strategy Solvers

Optimization solvers for resource allocation and scheduling decisions.
