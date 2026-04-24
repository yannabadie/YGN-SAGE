# ADR-014 - Rename LtlVerifier to GraphPropertyChecker

**Status:** Accepted - 2026-04-24.
**Related:** AUDIT3 #8.

## Context

`LtlVerifier` lived in `sage-core/src/verification/ltl.rs`, but the
implementation did not perform temporal-logic model checking. Its checks are
graph-structural algorithms over `TopologyGraph`:

* reachability via BFS,
* safety by scanning graph edges for high-to-low information flow,
* liveness by checking entry-to-exit reachability,
* bounded liveness by depth-limited graph traversal.

The old name implied Linear Temporal Logic verification and made the API look
more formal than its actual behavior.

## Decision

Rename the public Rust/PyO3 class from `LtlVerifier` to
`GraphPropertyChecker`. The module path remains `verification::ltl` for this
change to keep the diff and downstream import churn small.

Keep `LtlVerifier` as a deprecated Rust type alias for one release:

```rust
#[deprecated(note = "Use GraphPropertyChecker; ADR-014")]
pub type LtlVerifier = GraphPropertyChecker;
```

## Consequences

* The exported PyO3 class name now reflects what the implementation actually
  does: graph property checking.
* Rust callers can migrate from `LtlVerifier` to `GraphPropertyChecker`
  incrementally during the deprecation window.
* The deprecated alias is temporary compatibility only; it should be removed
  after one release once downstream callers have had time to update.
