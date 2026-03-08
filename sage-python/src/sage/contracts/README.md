# contracts

Contract IR and formal verification for agent task orchestration. Inspired by VeriMAP (2510.17109) and Plan-and-Act (2503.09572). Every subtask declares typed I/O schemas, capabilities, budgets, security labels, and failure policies. Execution proceeds through a verified DAG with pre/post checks, policy gates, and CEGAR repair.

## Modules

### `task_node.py` -- TaskNode IR

Core intermediate representation for a single subtask in a DAG.

- **Key exports**: `TaskNode`, `IOSchema`, `BudgetConstraint`, `SecurityLabel`, `FailurePolicy`
- `IOSchema` -- named typed fields with `validate(data)` method
- `SecurityLabel` -- enum: LOW, MEDIUM, HIGH, TOP (for info-flow enforcement)
- `BudgetConstraint` -- max tokens, max cost (USD), max wall time (seconds)

### `verification.py` -- Verification Functions

Pre-check and post-check functions bound to TaskNodes.

- **Key exports**: `VFResult`, `VerificationFn`, `pre_check()`, `post_check()`, `run_verification()`
- `VFResult` -- pass/fail with message and evidence dict

### `dag.py` -- TaskDAG

Directed acyclic graph of TaskNodes with topological scheduling.

- **Key exports**: `TaskDAG`, `CycleError`
- Kahn's algorithm for topological sort
- Cycle detection, I/O compatibility validation, `ready_nodes()` scheduler

### `z3_verify.py` -- Z3 SMT Verification

Formal property checking using Z3 solver (optional dependency).

- **Key exports**: `ContractVerdict`, `verify_capability_coverage()`, `verify_budget_feasibility()`, `verify_type_compatibility()`, `verify_provider_assignment()`
- Provider assignment is a genuine SAT problem: assigns providers to nodes respecting capability requirements and mutual exclusion constraints

### `policy.py` -- PolicyVerifier

Structural and security policy enforcement on a TaskDAG.

- **Key exports**: `PolicyVerifier`, `PolicyViolation`
- Info-flow: blocks data flowing from higher to lower security labels
- Budget: validates aggregate cost against total budget
- Fan-in/fan-out: limits excessive node degree

### `executor.py` -- DAGExecutor

Contract-driven execution engine for TaskDAGs.

- **Key exports**: `DAGExecutor`, `NodeResult`, `ExecutionReport`
- Executes nodes in topological order with pre/post VF checks
- Policy violations checked before execution begins
- Integrates `CostTracker` for budget enforcement

### `planner.py` -- TaskPlanner

Plan-and-Act decomposition from step specifications into a verified TaskDAG.

- **Key exports**: `TaskPlanner`
- Static planning from explicit step specs (`plan_static()`)
- Validates cycles and I/O compatibility

### `repair.py` -- RepairLoop (CEGAR)

Counterexample-guided abstraction refinement for failed verifications.

- **Key exports**: `RepairLoop`, `RepairAction`
- Feeds counterexamples back as constraints to the runner
- Hard fences: escalate after `max_retries`, abort after `2x max_retries`

### `cost_tracker.py` -- CostTracker

Cumulative per-node cost accounting with budget cap.

- **Key exports**: `CostTracker`
- `record(node_id, cost_usd)` -- additive cost tracking
- `is_over_budget()` -- checks against cap (0 = unlimited)
- `total_spent` -- aggregate across all nodes
