"""Extract fulltext from NotebookLM sources and upload to ExoCortex.

Prioritizes sources by keyword relevance to YGN-SAGE pillars.
Extracts content in batches to avoid rate limits.
"""
import asyncio
import os
import re
import sys
from pathlib import Path

# Force UTF-8
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from notebooklm import NotebookLMClient

# --- Config ---
EXPORT_DIR = Path.home() / ".sage" / "notebooklm_export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Notebooks to extract (circled in red on screenshot)
NOTEBOOKS = {
    "097c4c5c-beb2-4f65-9c6d-f597926a4232": "MetaScaffold_Core",
    "34d65dbb-4299-46e3-ab04-07879ed64541": "Core_Research_MARL",
    "ba22b122-1755-40c7-bd41-7be0be499430": "Technical_Implementation",
    "dcf45958-35bc-4f37-bee7-52b08571d2e2": "Discover_AI_Frontiers",
    "38d31b0a-c379-4325-8f96-495730a473bb": "OpenClawBDD",
}

# Keywords that indicate high relevance to YGN-SAGE
PRIORITY_KEYWORDS = [
    # Pillar 1: Strategy / Game Theory
    "psro", "cfr", "regret", "nash", "game theory", "marl", "multi-agent",
    "equilibrium", "policy space", "double oracle",
    # Pillar 2: Evolution
    "alphaevolve", "funsearch", "map-elites", "quality diversity", "evolution",
    "mutation", "genetic", "open-ended", "dgm", "sampo",
    # Pillar 3: Memory
    "memory", "episodic", "working memory", "rag", "retrieval", "mem1",
    "compressor", "forgetting", "context window",
    # Pillar 4: Metacognition
    "metacognition", "system 1", "system 2", "dual process", "cognitive",
    "reasoning", "self-monitoring", "self-braking", "stanovich",
    # Pillar 5: Tools / Verification
    "z3", "smt", "formal verification", "sandbox", "wasm", "ebpf",
    "process reward", "prm", "code generation",
    # ADK / Architecture
    "agent development", "adk", "opensage", "topology", "self-programming",
    "self-improving", "agentic", "orchestration", "tool use",
]


def relevance_score(title: str) -> int:
    """Score a source title by keyword matches."""
    title_lower = title.lower()
    return sum(1 for kw in PRIORITY_KEYWORDS if kw in title_lower)


def safe_filename(title: str, source_id: str) -> str:
    """Create a filesystem-safe filename."""
    clean = re.sub(r'[^\w\s-]', '', title[:80]).strip()
    clean = re.sub(r'\s+', '_', clean)
    return f"{clean}_{source_id[:8]}.md"


async def extract_notebook(client, nb_id: str, nb_name: str, max_sources: int = 100):
    """Extract top sources from a notebook by relevance."""
    sources = await client.sources.list(nb_id)
    print(f"\n{'='*60}")
    print(f"{nb_name}: {len(sources)} sources total")

    # Score and sort by relevance
    scored = []
    for s in sources:
        if not s.is_ready:
            continue
        score = relevance_score(s.title)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top N by relevance, then fill remaining with others
    top = scored[:max_sources]
    extracted = 0
    errors = 0

    for score, s in top:
        fname = safe_filename(s.title, s.id)
        fpath = EXPORT_DIR / nb_name / fname

        if fpath.exists():
            extracted += 1
            continue

        try:
            ft = await client.sources.get_fulltext(nb_id, s.id)
            content = ft.content if hasattr(ft, 'content') else str(ft)

            fpath.parent.mkdir(parents=True, exist_ok=True)
            md = f"# {s.title}\n\n"
            md += f"**Notebook:** {nb_name}\n"
            md += f"**Source ID:** {s.id}\n"
            md += f"**Kind:** {s.kind}\n"
            md += f"**Relevance Score:** {score}\n\n"
            md += f"## Content\n\n{content}\n"

            fpath.write_text(md, encoding="utf-8")
            extracted += 1

            if extracted % 20 == 0:
                print(f"  Extracted {extracted}/{len(top)}...")

        except Exception as e:
            errors += 1
            print(f"  ERROR ({s.title[:40]}): {e}", file=sys.stderr)

        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)

    print(f"  Done: {extracted} extracted, {errors} errors")
    return extracted


async def main():
    max_per_notebook = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    async with await NotebookLMClient.from_storage() as client:
        total = 0
        for nb_id, nb_name in NOTEBOOKS.items():
            count = await extract_notebook(client, nb_id, nb_name, max_per_notebook)
            total += count

    print(f"\n{'='*60}")
    print(f"Total extracted: {total} sources")
    print(f"Saved to: {EXPORT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
