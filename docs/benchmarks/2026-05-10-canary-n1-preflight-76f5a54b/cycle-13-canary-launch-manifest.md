# Cycle-13 Canary Launch Manifest - N=1 preflight

**Status**: real generation preflight, ungraded until grader gate is ready.
**Date**: 2026-05-10.

## Frozen at launch

| Parameter | Value |
|-----------|-------|
| Commit SHA | `76f5a54be4293d8845b08ed0d95b72e0d43fcb27` |
| Dataset | SWE-bench Pro test split, stratified N=5 input, first task only |
| Seed | 42 |
| Provider allowlist | `google`, `deepseek` |
| Provider denylist | `openai` |
| OpenAI | **EXCLUDED** from execution by runtime provider policy |

## Budget and timeouts

| Parameter | Value |
|-----------|-------|
| Budget per task | $5.00 |
| Global budget | $5.00 |
| Timeout per task | 120s |

## Environment

```bash
SAGE_LLM_TIER=budget
SAGE_DIFF_VERIFIER_MODE=observe
SAGE_OTEL_EXPORTER=none
SAGE_BOOT_BYPASS_EPOCH_GUARD=1
SAGE_BOOT_BYPASS_REASON=cycle-13-canary-n1-preflight
SAGE_OPERATOR_ID=ygn-sage-arm-d-canary
SAGE_DANGEROUS_TOOLS=0
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

## Decision boundary

This launch can validate runtime wiring, provider policy, trace completeness,
budget behavior, and patch extraction. It must not be reported as official
SWE-bench Pro graded evidence while the grader preflight remains blocked.
