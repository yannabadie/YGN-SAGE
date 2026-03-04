import asyncio
import os
import json
import logging
from discover.knowledge import NotebookLMBridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NotebookLM-Sync")

async def sync_and_query_sota():
    logger.info("📡 Initialisation de la connexion NotebookLM...")
    
    # Récupération de l'ID depuis l'environnement ou les fichiers de config
    notebook_id = os.getenv("NOTEBOOKLM_DEFAULT_ID")
    if not notebook_id:
        logger.error("❌ NOTEBOOKLM_DEFAULT_ID non trouvé dans l'environnement.")
        return

    bridge = NotebookLMBridge(notebook_id=notebook_id)
    
    # Présentation détaillée de la codebase et du projet
    project_presentation = """
    PROJET: YGN-SAGE (Self-Adaptive Generation Engine) - Mars 2026
    
    ARCHITECTURE:
    - Rust Core (sage-core): Orchestrateur haute performance (ULID, Zero-copy Arrow).
    - Python SDK (sage-python): Interface agentique multi-LLM.
    - Sandbox: Wasmtime + eBPF (solana_rbpf) pour exécution <1ms.
    
    AVANCÉES SOTA RÉCENTES:
    - Tri H96: Implémentation Rust AVX-512 utilisant 'vcompressps' pour un partitionnement branchless.
    - Performance: AIO Ratio de 0.00% (Overhead d'infrastructure inexistant).
    - Grounding: Intégration native Google Search Retrieval dans le provider Gemini.
    
    OBJECTIF ACTUEL:
    Surpasser NumPy sur des tris de >1M d'éléments en optimisant le partitionnement in-place Rust
    et en intégrant des mécanismes de méta-stratégie (VAD-CFR) pour la sélection du pivot.
    """
    
    question = f"""
    En te basant sur tes sources (OpenSage, AlphaEvolve, VAD-CFR), analyse la présentation suivante du projet YGN-SAGE :
    {project_presentation}
    
    Questions critiques :
    1. L'utilisation de 'vcompressps' pour le partitionnement H96 est-elle suffisante pour battre les tris radix ou les tris par fusion SIMD sur Zen 5 ?
    2. Comment le mécanisme VAD-CFR (Multi-Agent Regret Minimization) peut-il être adapté pour stabiliser le choix du pivot dans un tri hautement volatil ?
    3. Identifie-tu des angles morts dans notre architecture 'Zero-Copy Arrow' vis-à-vis de l'ASI ?
    """
    
    logger.info(f"📤 Envoi de la présentation et des questions au notebook {notebook_id}...")
    response = await bridge.ask_research_question(question)
    
    print("\n" + "🚀 RÉPONSE DE NOTEBOOKLM (SOTA GROUNDING) " + "="*30)
    print(response)
    print("="*70)

if __name__ == "__main__":
    asyncio.run(sync_and_query_sota())
