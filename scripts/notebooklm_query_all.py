import asyncio
import os
import sys

# Ajouter le chemin vers notebooklm-py si besoin
sys.path.insert(0, os.path.abspath('sage-python'))

from notebooklm import NotebookLMClient

async def query_notebooks():
    try:
        # Initialiser le client
        async with await NotebookLMClient.from_storage() as client:
            
            notebooks = [
                ("ba22b122-1755-40c7-bd41-7be0be499430", "YGN-SAGE : Technical Implementation Details"),
                ("34d65dbb-4299-46e3-ab04-07879ed64541", "YGN-SAGE : Core Research & MARL"),
                ("dcf45958-35bc-4f37-bee7-52b08571d2e2", "Discover AI: Frontiers of Agentic Reasoning and Architecture")
            ]
            
            prompt = """
Nous sommes le 4 Mars 2026. Voici l'état actuel de YGN-SAGE (Self-Adaptive Generation Engine) :
- Architecture hybride Python/Rust (sage-python, sage-core).
- Backend de mémoire : Rust `WorkingMemory` avec export `apache-arrow` zero-copy vers Python.
- Tri SOTA : Implémentation de H96 (partitionnement vectorisé branchless) avec AVX-512 `vcompressps` en Rust.
- Stratégie / Théorie des Jeux : Implémentation exacte des mandats SOTA pour VAD-CFR (Volatility-Adaptive Discounted CFR) et SHOR-PSRO avec recuit simulé pour la sélection de stratégies multi-agents.
- Sandboxing : Micro-VMs `wasmtime` (latence de démarrage <0.05ms) avec fallback automatique vers Docker.
- Moteur d'évolution : Algorithme génétique avec MAP-Elites et mutateur LLM Pro pour des cycles d'amélioration continus.

Notre ambition est d'atteindre une Intelligence Artificielle Super-Humaine (ASI) ou au minimum de définir le SOTA absolu en ingénierie logicielle autonome.

Analyse cet état de l'art par rapport au contenu de ton carnet.
1. Quelles sont les failles ou les angles morts de notre architecture actuelle ?
2. Quelle est LA prochaine innovation de rupture (SOTA 2026+) que nous devons absolument implémenter dans les prochains mois (MARL, neuro-symbolique, eBPF pur, memory folding, etc.) ?
3. Donne un plan d'action extrêmement précis, pensé au millimètre, pour les prochaines étapes de notre roadmap (Conductor) afin d'assurer l'évolution vers l'ASI.
Soyez extrêmement technique et précis dans les algorithmes, les structures de données et les frameworks.
"""
            
            results = {}
            for nb_id, nb_title in notebooks:
                print(f"\\n{'='*50}\\nInterrogation de: {nb_title}\\n{'='*50}")
                try:
                    result = await client.chat.ask(nb_id, prompt)
                    print(result.answer)
                    results[nb_title] = result.answer
                except Exception as e:
                    print(f"Erreur pour le carnet {nb_title}: {e}")
                    results[nb_title] = f"Erreur de génération : {e}"
                
            with open("research_journal/sota_2026_plan_ideas.md", "w", encoding="utf-8") as f:
                for title, text in results.items():
                    f.write(f"# {title}\\n\\n{text}\\n\\n---\\n\\n")
                    
            print("✅ Réponses sauvegardées dans research_journal/sota_2026_plan_ideas.md")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(query_notebooks())
