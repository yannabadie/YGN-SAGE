# GPT-5.4 Pro dataset generation report

Generated files in `data/`:
- `gpt54_error_correction.jsonl` — 20 entries
- `gpt54_preferences.jsonl` — 20 entries
- `gpt54_deep_reasoning.jsonl` — 20 entries
- `gpt54_simple_calibrated.jsonl` — 20 entries
- `gpt54_audit.jsonl` — 10 entries
- `topology_sft_gpt54_pro.jsonl` — concatenation of error-correction, deep-reasoning, and simple-calibrated files

Validation summary (preferences-aware validator):
error_correction: 20/20 valid
preferences: 20/20 valid
deep_reasoning: 20/20 valid
simple_calibrated: 20/20 valid
audit: 10/10 valid


Notes:
- The task prompts were sourced from real benchmark/problem families across Codeforces, LeetCode, and BigCodeBench.
- The local source file `data/topology_sft_v2_combined.jsonl` was not present in this container, so `gpt54_audit.jsonl` is a best-effort surrogate audit set built from 5 BigCodeBench tasks and 5 Codeforces tasks rather than from extracted local dataset rows.
- I used a corrected validator that explicitly checks `topology_a` and `topology_b` in the preferences file, because the provided generic snippet only inspects `topology`, `topology_v2`, or `improved`.
