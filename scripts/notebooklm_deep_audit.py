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
            
            # PROMPT ULTRA-TECHNIQUE ET CONTEXTUALISÉ
            prompt = """
Nous sommes le 4 Mars 2026. Je suis l'architecte de YGN-SAGE et j'ai audité la codebase actuelle. 
Voici les spécificités techniques exactes et les points de blocage identifiés :

1. MÉMOIRE (sage-core/memory.rs) :
- État actuel : `WorkingMemory` stocke un `Vec<MemoryEvent>`. L'export Arrow via `to_arrow` reconstruit des builders à chaque appel. C'est un goulot d'étranglement O(N).
- Problématique : Comment migrer vers un stockage "Arrow-native" (RecordBatches directs) interne en Rust tout en gardant une API mutable pour les agents Python ? Comment implémenter le S-MMU (Semantic MMU) pour paginer ces chunks Arrow via un graphe de connaissances petgraph ?

2. STRATÉGIE (sage-python/solvers.py) :
- État actuel : VAD-CFR et SHOR-PSRO sont implémentés avec les constantes exactes (alpha=1.5, beta=-0.1). Le recuit est statique sur 75 itérations.
- Problématique : Comment passer à SAMPO (Stable Agentic Multi-turn Policy Optimization) pour stabiliser les récompenses sur des trajectoires de mutation de code ? Comment évoluer les hyperparamètres du solveur lui-même dans le cycle MAP-Elites ?

3. ÉVOLUTION (sage-python/engine.py) :
- État actuel : MAP-Elites évolue des chaînes de caractères (code). Les "features" sont des descripteurs statiques.
- Problématique : Comment implémenter une Darwin Gödel Machine (DGM) où la population évolue la logique du mutateur ? Quelle structure de données pour le "World Model" (Transformer World Model) permettrait de simuler l'exécution avant de passer dans wasmtime ?

4. SANDBOXING (sage-core/sandbox) :
- État actuel : wasmtime et solana_rbpf sont là. Fallback Docker lent (cold start 1s).
- Problématique : Détaille l'implémentation de SnapBPF pour hooker `add_to_page_cache_lru()` et restaurer l'état mémoire des micro-VMs en <1ms. Comment intégrer Z3 (SMT solver) pour valider les invariants de l'AST du code évolué avant exécution ?

5. TOPOLOGIE :
- État actuel : Délégation Parent-Enfant simple.
- Problématique : Comment utiliser l'algorithme H96 (déjà implémenté pour le tri) comme base pour un routeur de topologie dynamique via MCTS (AFlow) ?

Analyse ces 5 points en profondeur en te basant sur tes connaissances SOTA 2026. Donne des structures de données Rust/Python, des snippets de logique algorithmique et une roadmap d'intégration millimétrée. Nous visons l'ASI.
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
                
            with open("research_journal/sota_2026_deep_audit_results.md", "w", encoding="utf-8") as f:
                for title, text in results.items():
                    f.write(f"# {title}\\n\\n{text}\\n\\n---\\n\\n")
                    
            print("✅ Réponses sauvegardées dans research_journal/sota_2026_deep_audit_results.md")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(query_notebooks())
