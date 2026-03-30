"""GAIA Level 1 evaluation — bare DeepSeek Chat model (no SAGE pipeline)."""
import asyncio
import json
import os
import re
import sys
import time

# Load .env
for line in open("/workspace/YGN-SAGE/.env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        os.environ[k] = v.strip('"')

from datasets import load_dataset
from openai import OpenAI


def gaia_exact_match(expected: str, response: str) -> bool:
    exp = expected.strip().lower()
    resp = response.strip().lower()
    if not exp:
        return False
    if resp == exp:
        return True
    pattern = r"(?<!\w)" + re.escape(exp) + r"(?!\w)"
    return bool(re.search(pattern, resp))


def main():
    ds = load_dataset(
        "gaia-benchmark/GAIA", "2023_all",
        split="validation", token=os.environ["HF_TOKEN"],
    )
    level1 = [item for item in ds if str(item.get("Level", "")) == "1"]
    # Skip file-dependent tasks
    level1_no_file = [item for item in level1 if not item.get("file_name")]
    print(f"GAIA Level 1: {len(level1)} total, {len(level1_no_file)} without files")

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
    )

    LIMIT = 20
    tasks = level1_no_file[:LIMIT]
    results = []
    passed = 0

    for i, item in enumerate(tasks):
        task_id = item.get("task_id", str(i))
        question = item["Question"]
        expected = item["Final answer"]
        level = item.get("Level", "?")

        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. Answer the question concisely. "
                            "End your response with 'FINAL ANSWER: <answer>' where <answer> "
                            "is your final answer to the question."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                max_tokens=512,
                timeout=60,
            )
            answer = r.choices[0].message.content or ""
            latency = time.time() - t0
            # Try usage-based cost (DeepSeek $0.28/$0.42 per 1M tokens)
            usage = r.usage
            cost = 0.0
            if usage:
                cost = (usage.prompt_tokens * 0.28 + usage.completion_tokens * 0.42) / 1_000_000
        except Exception as e:
            answer = ""
            latency = time.time() - t0
            cost = 0.0
            print(f"  [{i+1}] ERROR: {e}", flush=True)
            results.append({
                "task_id": task_id, "level": level, "passed": False,
                "expected": expected, "response": "", "error": str(e),
                "latency_s": latency, "cost_usd": cost,
            })
            continue

        ok = gaia_exact_match(expected, answer)
        if ok:
            passed += 1

        resp_preview = answer.replace("\n", " ")[:100]
        status = "PASS" if ok else "FAIL"
        print(
            f'  [{i+1}/{len(tasks)}] {status}  L{level}  '
            f'expected="{expected}"  got="{resp_preview}"',
            flush=True,
        )

        results.append({
            "task_id": task_id,
            "level": str(level),
            "passed": ok,
            "expected": expected,
            "response": answer[:500],
            "latency_s": round(latency, 2),
            "cost_usd": round(cost, 6),
        })

    total = len(tasks)
    rate = passed / total * 100 if total else 0
    print(f"\nBare DeepSeek Chat Level 1: {passed}/{total} ({rate:.0f}%)")
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    print(f"Total cost: ${total_cost:.4f}")

    return {
        "model": "deepseek-chat",
        "benchmark": "gaia-level1",
        "pipeline": "bare",
        "total": total,
        "passed": passed,
        "pass_rate": round(rate, 1),
        "total_cost_usd": round(total_cost, 4),
        "results": results,
    }


if __name__ == "__main__":
    report = main()
    # Save intermediate results
    os.makedirs("/workspace/YGN-SAGE/docs/benchmarks", exist_ok=True)
    with open("/workspace/YGN-SAGE/docs/benchmarks/gaia_bare_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to docs/benchmarks/gaia_bare_results.json")
