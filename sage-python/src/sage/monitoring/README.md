# Monitoring

Runtime monitoring for agent performance and behavior drift detection.

## Modules

### `drift.py` -- DriftMonitor

Detects performance degradation over time by tracking key metrics across agent runs:

- **Latency drift** -- Alerts when average response time increases beyond a configurable threshold.
- **Accuracy drift** -- Monitors pass rates and routing accuracy for sustained drops.
- **Cost drift** -- Tracks cost-per-task trends to catch unexpected spending increases.

The monitor maintains a sliding window of recent metrics and compares against baseline values established during benchmarking. When drift exceeds the configured sensitivity, events are emitted on the EventBus for dashboard display and alerting.

## Usage

DriftMonitor is instantiated during boot and subscribes to `BENCH_RESULT` and `ROUTING` events on the EventBus. No manual configuration is required for default thresholds.
