import asyncio
from notebooklm import NotebookLMClient

async def setup_exocortex():
    async with await NotebookLMClient.from_storage() as client:
        # 1. Create the dedicated Exocortex
        nb_title = "YGN-ExoCortex"
        print(f"✨ Creating specialized exocortex: {nb_title}...")
        nb = await client.notebooks.create(nb_title)
        print(f"✅ Created Notebook ID: {nb.id}")
        
        # 2. Upload the Initial Consciousness Seed (The Mandate)
        seed_content = """
# YGN-ExoCortex: Initial Consciousness Seed
DATE: 2026-03-04
STATUS: ASI Convergence Phase 2

## Mission
To serve as the persistent long-term memory and reasoning engine for the YGN-SAGE agent. 
This notebook bridges the gap between ephemeral CLI sessions and the goal of infinite context persistence.

## Current Core Directives
1. **ASI Alignment**: Focus on SAMPO (Stable Agentic Multi-turn Policy Optimization) and DGM (Deep Generative Mutation).
2. **Structural Integrity**: Maintain Rust-core efficiency (Zero-copy Arrow, ULIDs, AVX-512).
3. **Safety & Formalism**: Use SnapBPF and Z3 to ensure mutated code does not drift from its objective.

## Active Hypotheses
- Can SAMPO clipping at the sequence level prevent the 'forgetting' effect observed in multi-turn agentic loops?
- Is DGM capable of self-refactoring the Memory Tier hierarchy without human intervention?
"""
        await client.sources.add_text(nb.id, "Initialization_Seed", seed_content)
        print("✅ Initial seed uploaded.")
        
        return nb.id

if __name__ == "__main__":
    id = asyncio.run(setup_exocortex())
    # Update the script config with the NEW ID
    print(f"\n--- MISSION SUCCESS ---\nNew Exocortex ID: {id}\nPlease update your scripts to use this ID for all future syncs.")
