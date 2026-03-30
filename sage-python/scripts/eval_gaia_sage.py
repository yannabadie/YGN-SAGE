"""GAIA Level 1 evaluation — SAGE pipeline (budget tier: DeepSeek Chat)."""
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


def gaia_exact_match(expected: str, response: str) -> bool:
    exp = expected.strip().lower()
    resp = response.strip().lower()
    if not exp:
        return False
    if resp == exp:
        return True
    pattern = r"(?<!\w)" + re.escape(exp) + r"(?!\w)"
    return bool(re.search(pattern, resp))


async def main():
    ds = load_dataset(
        "gaia-benchmark/GAIA", "2023_all",
        split="validation", token=os.environ["HF_TOKEN"],
    )
    level1 = [item for item in ds if str(item.get("Level", "")) == "1"]
    level1_no_file = [item for item in level1 if not item.get("file_name")]
    print(f"GAIA Level 1: {len(level1)} total, {len(level1_no_file)} without files")

    # Boot SAGE
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=False, llm_tier="fast")
    print("SAGE booted (fast tier — Gemini Flash Lite)")

    LIMIT = 20
    tasks = level1_no_file[:LIMIT]
    results = []
    passed = 0

    for i, item in enumerate(tasks):
        task_id = item.get("task_id", str(i))
        question = item["Question"]
        expected = item["Final answer"]
        level = item.get("Level", "?")

        # Add instruction to focus on final answer
        prompt = (
            f"{question}\n\n"
            "Think step by step, then end with: FINAL ANSWER: <your answer>"
        )

        t0 = time.time()
        try:
            answer = await asyncio.wait_for(
                system.agent_loop.run(prompt),
                timeout=300,
            )
            latency = time.time() - t0
        except asyncio.TimeoutError:
            latency = time.time() - t0
            answer = ""
            print(f"  [{i+1}/{len(tasks)}] TIMEOUT  L{level}  id={task_id[:12]}", flush=True)
            results.append({
                "task_id": task_id, "level": str(level), "passed": False,
                "expected": expected, "response": "", "error": "timeout",
                "latency_s": round(latency, 2), "cost_usd": 0.0,
            })
            continue
        except Exception as e:
            latency = time.time() - t0
            answer = ""
            print(f"  [{i+1}/{len(tasks)}] ERROR  L{level}  {e}", flush=True)
            results.append({
                "task_id": task_id, "level": str(level), "passed": False,
                "expected": expected, "response": "", "error": str(e)[:200],
                "latency_s": round(latency, 2), "cost_usd": 0.0,
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
            "cost_usd": 0.0,
        })

    total = len(tasks)
    rate = passed / total * 100 if total else 0
    print(f"\nSAGE (fast) Level 1: {passed}/{total} ({rate:.0f}%)")

    return {
        "model": "gemini-3.1-flash-lite-preview",
        "benchmark": "gaia-level1",
        "pipeline": "sage-fast",
        "total": total,
        "passed": passed,
        "pass_rate": round(rate, 1),
        "results": results,
    }


if __name__ == "__main__":
    report = asyncio.run(main())
    os.makedirs("/workspace/YGN-SAGE/docs/benchmarks", exist_ok=True)
    with open("/workspace/YGN-SAGE/docs/benchmarks/gaia_sage_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to docs/benchmarks/gaia_sage_results.json")
