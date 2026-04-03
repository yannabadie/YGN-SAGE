#!/usr/bin/env python3
"""Validate SAGE training data compatibility with Gemma4 tokenizer.

Runs locally without GPU — downloads only the tokenizer, not model weights.

Checks:
  1. Token length distribution (min/max/mean/median/p95/p99)
  2. How many entries exceed 1024 and 2048 tokens
  3. <tool_call> token handling (single token vs sub-token split)
  4. System role handling in chat template
  5. Pass/fail summary

Usage:
    python scripts/validate_gemma4_data.py
    python scripts/validate_gemma4_data.py --data data/v2_final.jsonl
    python scripts/validate_gemma4_data.py --model google/gemma-4-26B-A4B-it
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

# ── Import system prompt (same as training scripts) ──────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sage_tool_schemas import TOOLCALL_SYSTEM_PROMPT as SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SAGE training data against Gemma4 tokenizer",
    )
    parser.add_argument(
        "--data",
        default="data/v2_final.jsonl",
        help="Path to training data JSONL (default: data/v2_final.jsonl)",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-4-26B-A4B-it",
        help="HuggingFace model ID for tokenizer (default: google/gemma-4-26B-A4B-it)",
    )
    return parser.parse_args()


def load_entries(data_path: str) -> list[dict]:
    """Load all entries from JSONL file."""
    entries = []
    with open(data_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  WARNING: Skipping line {line_num} (invalid JSON): {exc}")
    return entries


def build_messages(entry: dict) -> list[dict]:
    """Build messages list from a training entry.

    Supports two formats:
    - v2_multiturn: has 'turns' list with role/content dicts
    - v1 flat: has 'prompt' and 'topology_toolcall'/'topology_json'/'topology_yaml'
    """
    sys_prompt = entry.get("system_prompt", SYSTEM_PROMPT)
    messages = [{"role": "system", "content": sys_prompt}]

    if "turns" in entry and isinstance(entry["turns"], list):
        # v2 multi-turn format
        for turn in entry["turns"]:
            messages.append({
                "role": turn["role"],
                "content": turn["content"],
            })
    else:
        # v1 flat format
        prompt = entry.get("prompt", "")
        topology_text = (
            entry.get("topology_toolcall")
            or entry.get("topology_json")
            or entry.get("topology_yaml", "")
        )
        if prompt and topology_text:
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": topology_text})

    return messages


def check_token_lengths(tokenizer, entries: list[dict]) -> dict:
    """Tokenize all entries and compute length statistics."""
    lengths = []
    errors = []

    for i, entry in enumerate(entries):
        messages = build_messages(entry)
        # Need at least system + one user + one assistant
        if len(messages) < 3:
            errors.append(f"Entry {i}: too few messages ({len(messages)})")
            continue

        try:
            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True,
            )
            lengths.append(len(token_ids))
        except Exception as exc:
            errors.append(f"Entry {i}: apply_chat_template failed: {exc}")

    if not lengths:
        return {"lengths": [], "errors": errors}

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    p95_idx = int(n * 0.95)
    p99_idx = int(n * 0.99)

    return {
        "lengths": lengths,
        "errors": errors,
        "total": n,
        "min": lengths_sorted[0],
        "max": lengths_sorted[-1],
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "p95": lengths_sorted[min(p95_idx, n - 1)],
        "p99": lengths_sorted[min(p99_idx, n - 1)],
        "exceed_1024": sum(1 for l in lengths if l > 1024),
        "exceed_2048": sum(1 for l in lengths if l > 2048),
    }


def check_tool_call_token(tokenizer) -> dict:
    """Check how <tool_call> is tokenized."""
    token_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    tokens_decoded = [tokenizer.decode([tid]) for tid in token_ids]

    return {
        "token_ids": token_ids,
        "num_tokens": len(token_ids),
        "decoded_pieces": tokens_decoded,
        "is_single_token": len(token_ids) == 1,
    }


def check_system_role(tokenizer) -> dict:
    """Verify that system role content appears in the formatted output."""
    sentinel = "SAGE_SYSTEM_ROLE_TEST_SENTINEL_12345"
    messages = [
        {"role": "system", "content": sentinel},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "World"},
    ]

    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False,
        )
        return {
            "formatted_output": formatted,
            "sentinel_present": sentinel in formatted,
            "error": None,
        }
    except Exception as exc:
        return {
            "formatted_output": None,
            "sentinel_present": False,
            "error": str(exc),
        }


def main():
    args = parse_args()

    print("=" * 70)
    print("  Gemma4 Tokenizer Compatibility Validation")
    print("=" * 70)
    print()

    # ── Step 1: Load tokenizer ───────────────────────────────────
    print(f"[1/5] Loading tokenizer: {args.model}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"       Vocab size: {tokenizer.vocab_size}")
    print(f"       Model max length: {getattr(tokenizer, 'model_max_length', 'N/A')}")
    print()

    # ── Step 2: Load data ────────────────────────────────────────
    print(f"[2/5] Loading data: {args.data}")
    entries = load_entries(args.data)
    print(f"       Entries loaded: {len(entries)}")
    if not entries:
        print("ERROR: No entries found. Aborting.")
        sys.exit(1)
    print()

    # ── Step 3: Token length statistics ──────────────────────────
    print("[3/5] Tokenizing all entries...")
    stats = check_token_lengths(tokenizer, entries)

    if not stats["lengths"]:
        print("ERROR: No entries could be tokenized. Aborting.")
        for err in stats["errors"][:10]:
            print(f"  {err}")
        sys.exit(1)

    if stats["errors"]:
        print(f"       Warnings: {len(stats['errors'])} entries had errors")
        for err in stats["errors"][:5]:
            print(f"         {err}")
        if len(stats["errors"]) > 5:
            print(f"         ... and {len(stats['errors']) - 5} more")
        print()

    print(f"       Entries tokenized: {stats['total']}")
    print(f"       Token lengths:")
    print(f"         Min:    {stats['min']}")
    print(f"         Max:    {stats['max']}")
    print(f"         Mean:   {stats['mean']:.1f}")
    print(f"         Median: {stats['median']:.1f}")
    print(f"         P95:    {stats['p95']}")
    print(f"         P99:    {stats['p99']}")
    print(f"       Exceed 1024 tokens: {stats['exceed_1024']} / {stats['total']} "
          f"({stats['exceed_1024'] / stats['total'] * 100:.1f}%)")
    print(f"       Exceed 2048 tokens: {stats['exceed_2048']} / {stats['total']} "
          f"({stats['exceed_2048'] / stats['total'] * 100:.1f}%)")
    print()

    # ── Step 4: <tool_call> token handling ───────────────────────
    print("[4/5] Checking <tool_call> token handling...")
    tc = check_tool_call_token(tokenizer)
    print(f"       Token IDs for '<tool_call>': {tc['token_ids']}")
    print(f"       Number of sub-tokens: {tc['num_tokens']}")
    print(f"       Decoded pieces: {tc['decoded_pieces']}")
    if tc["is_single_token"]:
        print("       Result: <tool_call> is a SINGLE token")
    else:
        print(f"       Result: <tool_call> is SPLIT into {tc['num_tokens']} sub-tokens")
    print()

    # ── Step 5: System role handling ─────────────────────────────
    print("[5/5] Checking system role handling...")
    sr = check_system_role(tokenizer)
    if sr["error"]:
        print(f"       ERROR: {sr['error']}")
    else:
        # Show a truncated preview of the formatted output
        preview = sr["formatted_output"][:300] if sr["formatted_output"] else ""
        print(f"       System content in output: {sr['sentinel_present']}")
        print(f"       Formatted preview (first 300 chars):")
        for line in preview.split("\n"):
            print(f"         {line}")
    print()

    # ── Summary ──────────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    fit_pct = (stats["total"] - stats["exceed_1024"]) / stats["total"] * 100
    check_fit = fit_pct > 95.0
    check_toolcall = tc["num_tokens"] <= 3
    check_system = sr["sentinel_present"]

    results = [
        ("Token fit (>95% in 1024)", check_fit,
         f"{fit_pct:.1f}% fit in 1024 tokens"),
        ("<tool_call> not split >3", check_toolcall,
         f"{tc['num_tokens']} sub-token(s)"),
        ("System role preserved", check_system,
         "content appears in formatted output" if check_system else
         (sr["error"] or "content MISSING from formatted output")),
    ]

    all_pass = True
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {detail}")

    print()
    if all_pass:
        print("  All checks PASSED. Data is compatible with Gemma4 tokenizer.")
    else:
        print("  Some checks FAILED. Review the details above.")
    print("=" * 70)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
