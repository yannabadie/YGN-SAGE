Architecture SAGE — toutes les briques

                      ┌─── PIPELINE 5-STAGE ───┐
                      │                         │
  CLASSIFY ──► DECOMPOSE ──► TOPOLOGY ──► ASSIGN ──► EXECUTE ──► LEARN
     │            │            │            │          │           │
     ▼            ▼            ▼            ▼          ▼           ▼
  SystemRouter TaskPlanner TopologyEngine ModelAssigner TopologyRunner Bandit+
    (Rust)      (Python)     (Rust)        (Rust)      (Python)    MAP-Elites
     │                         │                         │
     ├─ kNN (Rust, 92%)        ├─ S-MMU hit             ├─ Wasm WASI sandbox
     ├─ StructuralFeatures     ├─ MAP-Elites archive    ├─ tree-sitter validator
     └─ ContextualBandit       ├─ LLM synthesis         ├─ subprocess fallback
        (Thompson sampling)    ├─ CMA-ME mutation       └─ ProviderPool (8 providers)
                               ├─ MCTS search               ├─ CircuitBreaker
                               ├─ Template fallback          └─ FrugalGPT cascade
                               └─ [Path 6: politique RL] ← EN COURS

  VÉRIFICATION                 MÉMOIRE                   QUALITÉ
     │                           │                          │
     ├─ OxiZ SmtVerifier        ├─ WorkingMemory (Arrow)   ├─ QualityLabeler (Z3)
     │  ├─ verify_arithmetic    ├─ S-MMU (4 vues)         │  ├─ syntax (tree-sitter)
     │  ├─ prove_memory_safety  │  ├─ Temporel             │  ├─ arithmetic (OxiZ)
     │  ├─ verify_invariant     │  ├─ Sémantique           │  └─ structural
     │  ├─ CEGAR synthesis      │  ├─ Causal               │
     │  └─ provider_assignment  │  └─ Entité               ├─ TopologyDensity (S_complex)
     │                          ├─ Episodic (SQLite)       │  └─ N_max bounds
     ├─ HybridVerifier          ├─ Semantic (entity graph) │
     │  ├─ 6 structural checks  ├─ ExoCortex (RAG)        └─ TopologyReward (dense)
     │  └─ 4 semantic checks    └─ RustEmbedder (ONNX)
     │
     └─ LtlVerifier
        ├─ reachability
        ├─ safety
        ├─ liveness
        └─ bounded liveness

  Les briques clés et leur rôle

  ┌──────────────────┬─────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────┐
  │      Brique      │                     Rôle dans le Self-Adaptive                      │                    Status                    │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ OxiZ/SmtVerifier │ Preuves formelles — reward dense pour GRPO, provider assignment SAT │ Actif                                        │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Wasm WASI        │ Sandbox sécurisée pour exécuter le code des agents                  │ Actif (Linux), subprocess fallback (Windows) │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ tree-sitter      │ Validation AST avant exécution — bloque code dangereux              │ Actif                                        │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ MAP-Elites       │ Archive quality-diversity des topologies                            │ Actif                                        │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ CMA-ME           │ Optimisation continue des paramètres de topologie                   │ Actif                                        │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ MCTS             │ Recherche Monte Carlo dans l'espace de mutations                    │ Actif                                        │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ S-MMU            │ Mémoire sémantique multi-vue — retrouve des topologies similaires   │ Actif                                        │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Arrow            │ Stockage zero-copy pour la working memory                           │ Actif                                        │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ ContextualBandit │ Thompson sampling pour choisir (modèle, topologie)                  │ Actif                                        │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ HybridVerifier   │ 10 checks structurels/sémantiques sur TopologyGraph                 │ Actif                                        │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ LtlVerifier      │ Model checking temporel (reachability, safety, liveness)            │ Actif                                        │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ S_complex/N_max  │ Métrique de coût topologique (AgentConductor)                       │ Actif                                        │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ TopologyReward   │ Reward dense multi-signal pour GRPO                                 │ Actif                                        │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ QualityLabeler   │ Scoring Z3 formel (pas heuristique)                                 │ Actif                                        │
  └──────────────────┴─────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────┘

  Ce qui est dormant (prêt mais pas câblé)

  ┌──────────────────────┬──────────────────────────────────┬───────────────────────┐
  │        Brique        │              Raison              │         Quand         │
  ├──────────────────────┼──────────────────────────────────┼───────────────────────┤
  │ eBPF                 │ solana_rbpf casse le CI Ubuntu   │ Phase 2               │
  ├──────────────────────┼──────────────────────────────────┼───────────────────────┤
  │ RagCache             │ Cache pour File Search résultats │ Phase C               │
  ├──────────────────────┼──────────────────────────────────┼───────────────────────┤
  │ RustEntityGraph      │ Mémoire causale Rust-native      │ Phase C               │
  ├──────────────────────┼──────────────────────────────────┼───────────────────────┤
  │ Persistence (SQLite) │ Bandit state cross-session       │ Quand bandit converge │
  └──────────────────────┴──────────────────────────────────┴───────────────────────┘