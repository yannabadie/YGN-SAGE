# YGN-SAGE — Audit Remediation Plan (Claude Code)

> **Audit date:** 2026-03-09 | **Auditor:** Claude Opus 4.6
> **Repository:** `github.com/yannabadie/YGN-SAGE` @ master (54 commits)
> **Rating:** 4.2 / 10
> **Codebase:** Python 31,526 LOC | Rust 2,862 LOC | Markdown 31,220 LOC

Ce document est conçu pour être fourni directement à Claude Code. Chaque finding a : un ID, une sévérité, le fichier + numéro de ligne exact, la preuve technique, et l'action corrective précise.

---

## TABLE DES SÉVÉRITÉS

| Sévérité | Count | Signification |
|----------|-------|---------------|
| CRITICAL | 7 | Cassé, dangereux, ou fondamentalement incorrect |
| HIGH | 8 | Significativement trompeur ou non sécurisé |
| MEDIUM | 8 | Sous-optimal, incohérent |
| LOW | 3 | Cosmétique, mineur |

---

## PHASE 0 — TRIAGE IMMÉDIAT (objectif : 0 CRITICAL restant)

### Z3-01 [CRITICAL] — z3_validator.rs : dead code non compilable

**Fichier:** `sage-core/src/sandbox/z3_validator.rs`
**Preuve:** Ligne 2 : `use z3::{Config, Context, Solver, SatResult, ast::{Int, Bool, Ast}};`
Le crate `z3` n'est PAS dans `sage-core/Cargo.toml` (vérifié : aucune entrée z3 dans `[dependencies]`).
Le fichier est exclu du module tree à `sage-core/src/sandbox/mod.rs:7` :
```
// z3_validator: moved to Python (sage.sandbox.z3_validator) using z3-solver package
```

**Impact:** 133 lignes de dead code présenté comme feature dans la structure du projet.

**Action:**
```bash
# Supprimer le fichier mort
rm sage-core/src/sandbox/z3_validator.rs
```
Mettre à jour `sage-core/src/sandbox/README.md` si il référence ce fichier.

---

### Z3-02 [CRITICAL] — check_loop_bound() retourne TOUJOURS False

**Fichiers affectés (2 implémentations identiques du même bug) :**
- `sage-python/src/sage/sandbox/z3_validator.py:55-67`
- `sage-python/src/sage/topology/kg_rlvr.py:81-89`

**Preuve mathématique :**
```python
# Code actuel (ligne 55-66 de z3_validator.py) :
def check_loop_bound(self, var_name: str, hard_cap: int) -> bool:
    s = z3.Solver()
    iters = z3.Int(var_name)     # Variable symbolique SANS CONTRAINTE
    cap = z3.IntVal(hard_cap)
    s.add(iters > cap)           # "Existe-t-il iters > cap ?"
    return s.check() == z3.unsat # Toujours sat car iters est libre
```

`z3.Int(var_name)` crée une variable symbolique sans borne. Le solver trouvera toujours un `iters = cap + 1` qui satisfait `iters > cap`. Donc `s.check()` retourne toujours `sat`, et la méthode retourne toujours `False` ("boucle non bornée").

**Conséquence :** Chaque boucle vérifiée est déclarée potentiellement infinie. Les appelants à `kg_rlvr.py:140` retournent un score de `-1.0` pour tout step contenant "assert loop".

**Fix :**
```python
# z3_validator.py — Remplacer lignes 55-67
def check_loop_bound(self, var_name: str, hard_cap: int) -> bool:
    """Check if a loop variable is provably bounded.

    Returns True only if, given iters >= 0, it's impossible for iters > cap.
    For an unconstrained symbolic var this returns False (correct: we can't prove it).
    The caller must supply additional constraints for meaningful proofs.
    """
    s = z3.Solver()
    iters = z3.Int(var_name)
    cap = z3.IntVal(hard_cap)
    # Contrainte : la variable de boucle est non-négative
    s.add(iters >= 0)
    # On cherche un contre-exemple : iters > cap malgré les contraintes
    s.add(iters > cap)
    return s.check() == z3.unsat
```

**ATTENTION :** Même avec ce fix, un `Int` avec seulement `>= 0` retournera encore `sat` (car `cap + 1 >= 0`). Ce fix rend la méthode **sémantiquement correcte mais inutile sans contraintes additionnelles**. La vraie solution est de passer les contraintes du programme analysé. En attendant, documenter que la méthode est un placeholder.

Appliquer le même fix dans `kg_rlvr.py:81-89`.

---

### Z3-03 [CRITICAL] — verify_arithmetic() retourne TOUJOURS False

**Fichier:** `sage-python/src/sage/topology/kg_rlvr.py:91-98`

**Preuve mathématique :**
```python
# Code actuel :
def verify_arithmetic(self, expr: str, expected: int, tolerance: int = 0) -> bool:
    solver = z3.Solver()
    result = z3.Int("result")  # Variable SANS CONNEXION à expr
    solver.add(z3.Or(result > expected + tolerance, result < expected - tolerance))
    return solver.check() == z3.unsat
```

Le paramètre `expr` est **complètement ignoré**. `result` est une variable symbolique libre. Le solver trouvera toujours une valeur hors tolérance. Retourne toujours `False`.

**Appelé par :** `kg_rlvr.py:151` — tout "assert arithmetic(...)" dans un reasoning step reçoit un score de `-1.0`.

**Fix — Option A (supprimer) :**
```python
def verify_arithmetic(self, expr: str, expected: int, tolerance: int = 0) -> bool:
    """STUB — Not implemented. Always returns True (fail-open).

    A correct implementation would parse `expr` into a Z3 arithmetic
    expression and check if its value is within [expected - tolerance, expected + tolerance].
    This requires an expression parser (e.g., sympy -> z3 bridge).
    """
    if not self.has_z3:
        return True
    # TODO: implement expr -> Z3 ArithRef conversion
    return True  # Fail-open until implemented
```

**Fix — Option B (implémenter correctement) :**
```python
def verify_arithmetic(self, expr: str, expected: int, tolerance: int = 0) -> bool:
    """Evaluate arithmetic expr and verify result is within tolerance."""
    if not self.has_z3:
        return True
    try:
        # Parse simple arithmetic safely
        import ast
        actual = ast.literal_eval(expr)
        return expected - tolerance <= actual <= expected + tolerance
    except (ValueError, SyntaxError):
        return False  # Fail-closed on unparseable
```

L'option B n'a plus besoin de Z3 — c'est honnête.

---

### Z3-05 [CRITICAL] — z3_topology.py : "Z3-based" est Kahn's algorithm + proof fabriquée

**Fichier:** `sage-python/src/sage/topology/z3_topology.py`

**Preuves :**

1. **Nom trompeur :** Le fichier s'appelle `z3_topology.py` mais Z3 n'est jamais appelé avec succès. La méthode `_z3_verify()` (lignes 131-142) fait `import sage_core` qui échoue, le `except Exception:` (ligne 139) avale l'erreur, et retourne `""`.

2. **Preuve fabriquée :** Lignes 75-83 :
```python
result.proof = (
    f"PROVED: Topology is a valid DAG with {len(spec.agents)} agents, "
    f"{len(spec.edges)} edges, max depth {depth}. "
    f"Terminates: sat. No cycles: sat."
)
```
Ce texte contient "PROVED" et "sat" mais provient uniquement de Kahn's algorithm (BFS topological sort, lignes 86-103), pas de Z3.

3. **Tests trompeurs :** `test_verify_returns_proof()` dans `sage-python/tests/test_z3_topology.py:40-47` :
```python
assert "sat" in result.proof.lower() or "proved" in result.proof.lower()
```
Ce test passe car le string est hardcodé, pas parce que Z3 a produit une preuve.

**Action :**
```bash
# Renommer le fichier honnêtement
mv sage-python/src/sage/topology/z3_topology.py sage-python/src/sage/topology/topology_verifier.py
```

Mettre à jour :
- `sage-python/src/sage/topology/__init__.py` (si import)
- `sage-python/tests/test_z3_topology.py` (renommer aussi)
- Tout `from sage.topology.z3_topology import` dans le projet :
```bash
grep -rn "z3_topology" sage-python/src/ sage-python/tests/ --include="*.py"
```

Supprimer le mot "Z3" du docstring et de la classe. Remplacer le proof string par :
```python
result.proof = (
    f"Graph analysis: DAG with {len(spec.agents)} agents, "
    f"{len(spec.edges)} edges, max depth {depth}. "
    f"Termination: verified (acyclic). Method: Kahn's algorithm."
)
```

Supprimer `_z3_verify()` entièrement ou la garder avec un `# TODO` explicite.

---

### Z3-06 [CRITICAL] — Evolution safety gate valide le config, pas le code muté

**Fichier:** `sage-python/src/sage/evolution/engine.py:167-170`

**Preuve :**
```python
# Ligne 162-163 : le code muté est dans new_code
new_code, features = await mutate_fn(parent.code, dgm_context=dgm_context)

# Ligne 167-168 : MAIS le Z3 gate valide self.config.z3_constraints (STATIQUE)
if self._z3 and self.config.z3_constraints:
    result = self._z3.validate_mutation(self.config.z3_constraints)
```

`self.config.z3_constraints` est défini à l'init (ligne 51 : `field(default_factory=list)`) et ne change JAMAIS. Il n'est pas dérivé de `new_code`. Le safety gate vérifie les mêmes contraintes statiques à chaque itération.

**Action :**
Option A — Supprimer le safety gate (honnête) :
```python
# Supprimer les lignes 167-170 et le champ z3_safety_gate du config
```

Option B — Implémenter un vrai gate (difficile mais correct) :
```python
# Extraire les contraintes du code muté via AST
if self._z3:
    constraints = self._extract_constraints_from_code(new_code)
    if constraints:
        result = self._z3.validate_mutation(constraints)
        if not result.safe:
            self.z3_rejections += 1
            continue
```
Cela nécessite une méthode `_extract_constraints_from_code()` qui analyse le code Python muté pour en extraire des bornes vérifiables (accès tableau, bornes de boucle, etc.).

---

### Z3-07 [CRITICAL] — Fallback silencieux : has_z3=False → return True (rubber stamp)

**Fichier:** `sage-python/src/sage/topology/kg_rlvr.py:65-107`

**Preuve :** Quand `z3-solver` n'est pas installé (optionnel dans pyproject.toml), toutes les méthodes de `FormalKnowledgeGraph` retournent `True` :
- Ligne 70-71 : `prove_memory_safety` → `return True`
- Ligne 82-83 : `check_loop_bound` → `return True`
- Ligne 93-94 : `verify_arithmetic` → `return True`
- Ligne 106-107 : `verify_invariant` → `return True`

**Impact :** Sans z3-solver, le système déclare que TOUT est formellement vérifié. Le warning à la ligne 67 (`logging.warning`) est facilement manqué.

**Action :** Changer le fallback de fail-open à fail-neutral :
```python
def prove_memory_safety(self, addr_expr: int, limit: int) -> bool:
    if not self.has_z3:
        return 0 <= addr_expr < limit  # Vérification triviale au lieu de rubber stamp
```

```python
def check_loop_bound(self, iterations_symbolic: str, hard_cap: int) -> bool:
    if not self.has_z3:
        return False  # Fail-closed : on ne peut pas prouver sans Z3
```

```python
def verify_arithmetic(self, expr: str, expected: int, tolerance: int = 0) -> bool:
    if not self.has_z3:
        return False  # Fail-closed
```

```python
def verify_invariant(self, pre: str, post: str) -> bool:
    if not self.has_z3:
        return False  # Fail-closed
```

Et remonter la sévérité du warning :
```python
logging.error("z3-solver not installed. ALL formal verification disabled — returning unverified.")
```

---

### SEC-01 [CRITICAL] — create_python_tool : exécution de code LLM arbitraire

**Fichier:** `sage-python/src/sage/tools/meta.py:40-72`

**Preuve du bypass de l'AST check :**

Le check (lignes 51-54) ne bloque que les appels directs `exec()` et `eval()` :
```python
for node in ast.walk(parsed_ast):
    if isinstance(node, ast.Call) and getattr(node.func, 'id', '') in ('exec', 'eval'):
        return "Security Error: ..."
```

**6 vecteurs de bypass connus :**
1. `__import__('os').system('rm -rf /')` — ast.Attribute, pas ast.Name
2. `getattr(__builtins__, 'exec')('malicious')` — exec via getattr
3. `compile('code', '', 'exec')` — compile n'est pas bloqué
4. `type('X', (), {'__init__': lambda self: os.system('cmd')})()` — metaclass
5. `(lambda: __import__('subprocess').call(['sh']))()` — lambda + import
6. `open('/etc/passwd').read()` — aucun FS access control

Ensuite `importlib.util.module_from_spec` (ligne 66) exécute le module complet.

**Action — Remplacement complet :**
```python
# Option A : RestrictedPython (pip install RestrictedPython)
from RestrictedPython import compile_restricted, safe_globals

@Tool.define(name="create_python_tool", ...)
async def create_python_tool(name: str, code: str, registry: ToolRegistry = None) -> str:
    try:
        byte_code = compile_restricted(code, filename=f"<tool:{name}>", mode="exec")
    except SyntaxError as e:
        return f"Syntax error: {e}"

    restricted_globals = safe_globals.copy()
    restricted_globals['__name__'] = name
    # Whitelist explicit safe modules
    restricted_globals['json'] = __import__('json')
    restricted_globals['re'] = __import__('re')
    restricted_globals['math'] = __import__('math')

    namespace = {}
    exec(byte_code, restricted_globals, namespace)  # RestrictedPython contrôle l'exécution
    # ... registration logic
```

```python
# Option B : Exécution Wasm (si le sandbox feature est compilé)
# Sérialiser le code dans un module Wasm et l'exécuter via WasmSandbox
```

```python
# Option C (minimum viable) : Supprimer create_python_tool entièrement
# et ne garder que des tools statiquement définis
```

---

### SEC-02 [HIGH] — create_bash_tool : injection shell

**Fichier:** `sage-python/src/sage/tools/meta.py:101-123`

**Preuve :** Ligne 118 :
```python
result = subprocess.run({repr(script)}, shell=True, capture_output=True, text=True, timeout=60)
```

Le paramètre `script` vient directement du LLM. `shell=True` permet l'injection de commandes shell arbitraires.

**Action :**
```python
# Remplacer shell=True par une exécution contrôlée
import shlex

# Dans le template généré (lignes 107-120) :
result = subprocess.run(
    shlex.split({repr(script)}),
    shell=False,
    capture_output=True,
    text=True,
    timeout=60,
    cwd="/tmp/sage-sandbox",  # Isoler le répertoire de travail
)
```

Ou supprimer `create_bash_tool` entièrement si le use case ne justifie pas le risque.

---

### SEC-03 [HIGH] — run_bash builtin : shell non sandboxé

**Fichier:** `sage-python/src/sage/tools/builtin.py:25`

**Preuve :**
```python
proc = await asyncio.create_subprocess_shell(
```

**Action :**
```python
# Remplacer par create_subprocess_exec
proc = await asyncio.create_subprocess_exec(
    "/bin/bash", "-c", command,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    # Optionnel : limiter les capabilities
    env={**os.environ, "PATH": "/usr/bin:/bin"},
    cwd="/tmp/sage-sandbox",
)
```

Et ajouter une validation d'entrée :
```python
BLOCKED_PATTERNS = re.compile(r'rm\s+-rf|mkfs|dd\s+if=|:(){ :|fork|/dev/sd')
if BLOCKED_PATTERNS.search(command):
    return "BLOCKED: Potentially destructive command"
```

---

## PHASE 1 — FIABILITÉ (objectif : tests significatifs, benchmarks publiés)

### SEC-04 [MEDIUM] — _safe_z3_eval : whitelist AST trop permissive

**Fichier:** `sage-python/src/sage/topology/kg_rlvr.py:26-59`

**Détail :** `ast.Call` est autorisé avec accès complet au module `z3` dans le namespace. `z3` contient des méthodes qui instancient des objets complexes. L'attaquant contrôle les strings `pre` et `post` dans `verify_invariant()`.

**Action :** Restreindre les fonctions z3 appelables à une whitelist explicite :
```python
_ALLOWED_Z3_FUNCTIONS = {"Int", "IntVal", "Bool", "BoolVal", "And", "Or", "Not", "Implies"}

# Dans la boucle de validation (ligne 49-52) :
if isinstance(node, ast.Call):
    if isinstance(node.func, ast.Attribute):
        if not (isinstance(node.func.value, ast.Name)
                and node.func.value.id == "z3"
                and node.func.attr in _ALLOWED_Z3_FUNCTIONS):
            raise ValueError(f"Only z3.{_ALLOWED_Z3_FUNCTIONS} allowed, got z3.{node.func.attr}")
```

---

### EVO-01 [HIGH] — SelfImprovementLoop : wrapper vide

**Fichier:** `sage-python/src/sage/evolution/self_improve.py`

**Preuve :** La classe accepte 3 callables externes (`benchmark_fn`, `diagnose_fn`, `evolve_fn`) mais ne fournit AUCUNE implémentation. Les paramètres `orchestrator`, `registry`, `evolution_engine` du constructeur (lignes 35-38) ne sont jamais utilisés dans `run_cycle()`.

**Action — Implémenter des defaults concrets :**
```python
# Ajouter après la classe SelfImprovementLoop :

async def default_diagnose_fn(failures: list) -> list[str]:
    """Analyse AST des échecs pour identifier les patterns communs."""
    import ast
    diagnosis = []
    for f in failures:
        if hasattr(f, 'generated_code') and f.generated_code:
            try:
                tree = ast.parse(f.generated_code)
                # Compter les anti-patterns
                for node in ast.walk(tree):
                    if isinstance(node, ast.Try) and not node.handlers:
                        diagnosis.append(f"Task {f.task_id}: bare except clause")
                    if isinstance(node, ast.Global):
                        diagnosis.append(f"Task {f.task_id}: global variable mutation")
            except SyntaxError:
                diagnosis.append(f"Task {f.task_id}: generated code has syntax errors")
        if hasattr(f, 'error') and f.error:
            diagnosis.append(f"Task {f.task_id}: {f.error[:200]}")
    return diagnosis
```

---

### EVO-02 [HIGH] — Evolution : aucune validation contre baseline

**Fichier:** `sage-python/src/sage/evolution/engine.py`

**Action :** Ajouter un smoke test de convergence dans les tests :
```python
# tests/test_evolution_convergence.py
import pytest
from sage.evolution.engine import EvolutionEngine, EvolutionConfig
from sage.evolution.population import Individual

@pytest.mark.asyncio
async def test_evolution_improves_over_random():
    """Verify evolution produces better solutions than random sampling."""
    config = EvolutionConfig(population_size=20, max_generations=10, mutations_per_generation=5)
    engine = EvolutionEngine(config=config)

    # Seed with random individuals
    import random
    seeds = [Individual(code=f"x={random.random()}", score=random.random(), features=(0, 0))
             for _ in range(10)]
    engine.seed(seeds)

    initial_best = engine.best_solution().score

    # Run 10 generations with a trivial mutator
    async def trivial_mutate(code, dgm_context=None):
        # Randomly perturb the score-producing code
        return f"x={random.random()}", (random.randint(0,9), random.randint(0,9))

    for _ in range(10):
        await engine.evolve_step(trivial_mutate)

    final_best = engine.best_solution().score
    # MAP-Elites should retain the best regardless
    assert final_best >= initial_best, "Evolution must not regress best-known solution"
```

---

### EVO-03 [MEDIUM] — Naming DGM/Gödel Machine trompeur

**Fichier:** `sage-python/src/sage/evolution/engine.py`

**Preuve :** Variables `_dgm_solver`, `dgm_action`, `DGM_ACTION_DESCRIPTIONS` (lignes 28, 99, 140). Le docstring (ligne 62) dit "this is NOT a Gödel Machine" mais les noms persistent.

**Action :**
```bash
cd sage-python/src/sage/evolution
# Renommer dans engine.py :
sed -i 's/_dgm_solver/_sampo_solver/g; s/dgm_action/sampo_action/g; s/DGM_ACTION_DESCRIPTIONS/SAMPO_ACTION_DESCRIPTIONS/g; s/dgm_context/sampo_context/g; s/dgm_entropy/sampo_entropy/g' engine.py
```

Vérifier les tests :
```bash
grep -rn "dgm" sage-python/tests/ --include="*.py"
```

---

### RTG-01 [HIGH] — Routing benchmark circulaire

**Fichier:** `sage-python/src/sage/bench/routing.py`

**Preuve :** Le benchmark "30/30 self-consistency" compare les décisions du router contre des labels dérivés du même router. C'est un test d'idempotence, pas de qualité.

**Action :** Ajouter un benchmark non-circulaire :
```python
# sage-python/src/sage/bench/routing_quality.py
"""Routing quality benchmark: measures cost/accuracy tradeoff."""

GROUND_TRUTH = [
    # (task, expected_minimum_system, rationale)
    ("What is 2+2?", 1, "trivial arithmetic"),
    ("Write a bubble sort in Python", 2, "simple algorithm"),
    ("Debug a race condition in async Rust with deadlock on Arc<Mutex>", 3, "complex concurrent debug"),
    ("What's the capital of France?", 1, "trivial factual"),
    ("Implement a B+ tree with concurrent insert/delete", 3, "complex data structure"),
    # ... ajouter 50+ cas manuellement étiquetés
]

async def run_routing_quality():
    router = ComplexityRouter()
    correct = 0
    over_routed = 0  # S3 quand S1 suffisait (gaspillage)
    under_routed = 0  # S1 quand S3 nécessaire (risque)

    for task, min_system, _ in GROUND_TRUTH:
        profile = router.assess_complexity(task)
        decision = router.route(profile)
        if decision.system >= min_system:
            correct += 1
        if decision.system > min_system:
            over_routed += 1
        if decision.system < min_system:
            under_routed += 1

    return {
        "accuracy": correct / len(GROUND_TRUTH),
        "over_routing_rate": over_routed / len(GROUND_TRUTH),
        "under_routing_rate": under_routed / len(GROUND_TRUTH),
    }
```

---

### RTG-02 [HIGH] — Orchestrateur : fallback silencieux vers legacy

**Fichier:** `sage-python/src/sage/boot.py:100-108`

**Preuve :** L'orchestrateur EST wired (contrairement aux audits précédents) mais avec un `except Exception as e` (ligne 105) qui fallback vers le legacy routing sans interruption. En mode mock (ligne 95), il est entièrement bypassé.

**Action :** Logger clairement le fallback et compter les occurrences :
```python
# boot.py, après ligne 105 :
except Exception as e:
    _log.warning(
        "Orchestrator failed (%s), falling back to legacy routing — "
        "this means multi-provider routing is NOT active",
        e,
    )
    self._orchestrator_fallback_count = getattr(self, '_orchestrator_fallback_count', 0) + 1
```

Ajouter une métrique visible dans le dashboard.

---

### RTG-03 [MEDIUM] — Seuils de routing modifiés sans justification

**Fichier:** `sage-python/src/sage/strategy/metacognition.py:88-91`

**Preuve :** Valeurs actuelles vs audits précédents (ARCHITECTURE.md lignes 25-26) :

| Seuil | Ancienne valeur | Valeur actuelle | Delta |
|-------|----------------|-----------------|-------|
| s1_complexity_ceil | 0.35 | 0.50 | +43% |
| s1_uncertainty_ceil | 0.30 | 0.30 | 0% |
| s3_complexity_floor | 0.70 | 0.65 | -7% |
| s3_uncertainty_floor | 0.60 | 0.60 | 0% |

**Action :** Ajouter un ADR (Architecture Decision Record) :
```bash
# Créer docs/ADR-routing-thresholds.md avec :
# - Justification du changement
# - Données de test qui ont motivé les nouvelles valeurs
# - Impact mesuré sur le benchmark de routing
```

Mettre à jour ARCHITECTURE.md pour refléter les vrais seuils.

---

### MEM-01 [HIGH] — S-MMU Rust non exposée directement à Python

**Fichier:** `sage-core/src/memory/smmu.rs`

**Preuve :** `grep -c "pyclass\|pymethods\|pyfunction" sage-core/src/memory/smmu.rs` → 0 hits.

L'accès se fait via `WorkingMemory` (Rust, `sage-core/src/memory/mod.rs`) qui est `#[pyclass]`. Mais en mode mock Python (`sage-python/src/sage/memory/working.py:74`), `smmu_chunk_count()` retourne toujours 0 → le read path dans `agent_loop.py:326-330` ne produit jamais de contexte S-MMU.

**Action — Exposer MultiViewMMU directement :**

Ajouter dans `sage-core/src/memory/smmu.rs` :
```rust
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyMultiViewMMU {
    inner: MultiViewMMU,
}

#[pymethods]
impl PyMultiViewMMU {
    #[new]
    fn new() -> Self {
        Self { inner: MultiViewMMU::new() }
    }

    fn register_chunk(
        &mut self,
        start_time: i64, end_time: i64,
        summary: &str, keywords: Vec<String>,
        embedding: Option<Vec<f32>>, parent_chunk_id: Option<usize>,
    ) -> usize {
        self.inner.register_chunk(start_time, end_time, summary, keywords, embedding, parent_chunk_id)
    }

    fn chunk_count(&self) -> usize {
        self.inner.chunk_count()
    }

    fn retrieve_relevant(&self, chunk_id: usize, max_hops: usize, weights: [f32; 4]) -> Vec<(usize, f32)> {
        self.inner.retrieve_relevant(chunk_id, max_hops, weights)
    }
}
```

Ajouter dans `sage-core/src/lib.rs` :
```rust
m.add_class::<memory::smmu::PyMultiViewMMU>()?;
```

---

### MEM-02 [MEDIUM] — CausalMemory sans persistence

**Fichier:** `sage-python/src/sage/memory/causal.py` (si existant) ou dans le module mémoire.

**Action :** Appliquer le même pattern SQLite que SemanticMemory (qui a déjà été fixé).

---

### MEM-03 [MEDIUM] — ExoCortex hardcode DEFAULT_STORE

**Fichier:** `sage-python/src/sage/memory/` (backend RAG)

**Action :** Paramétrer le store ID via variable d'environnement :
```python
EXOCORTEX_STORE = os.environ.get("SAGE_EXOCORTEX_STORE", "DEFAULT_STORE_ID")
```

---

### MEM-04 [LOW] — Aucune preuve que 4-tier memory > long-context baseline

**Action :** Créer un benchmark A/B :
```python
# tests/bench_memory_vs_longcontext.py
# 1. Exécuter 20 tâches multi-turn avec 4-tier memory
# 2. Exécuter les mêmes 20 tâches en injectant tout l'historique dans le prompt
# 3. Comparer : accuracy, cost, latency
```

---

## PHASE 2 — TESTS & CI

### TST-01 [HIGH] — 70% mock ratio dans les tests

**Preuve :** `grep -c "mock\|Mock\|MagicMock\|patch\|monkeypatch" sage-python/tests/*.py` → 598 sur 847 test functions.

**Action :**
1. Identifier les tests 100% mock (aucune logique réelle testée) :
```bash
grep -l "def test_" sage-python/tests/*.py | while read f; do
  total=$(grep -c "def test_" "$f")
  mocks=$(grep -c "mock\|Mock\|patch" "$f")
  ratio=$((mocks * 100 / total))
  echo "$f: $total tests, $mocks mock refs, ${ratio}% mock"
done | sort -t'%' -k1 -rn
```
2. Pour les fichiers > 80% mock, réécrire avec des fixtures déterministes ou des implémentations in-memory.
3. Objectif : < 30% mock ratio.

---

### TST-02 [MEDIUM] — Tests Z3 passent car Z3 n'est jamais appelé

**Fichier:** `sage-python/tests/test_z3_topology.py`

**Action :** Ajouter un guard dans les tests :
```python
import pytest
z3 = pytest.importorskip("z3", reason="z3-solver required for formal verification tests")
```

Et séparer les tests en deux groupes :
- `test_topology_graph.py` — tests du BFS/Kahn's algorithm (toujours exécutés)
- `test_topology_z3.py` — tests Z3 (skip si z3 absent, avec marker explicite)

---

### TST-03 [MEDIUM] — CI ne teste pas sandbox ni ONNX

**Fichier:** `.github/workflows/ci.yml:30`

**Preuve :** `cargo test --no-default-features` → sandbox et onnx exclus.

**Action :**
```yaml
# .github/workflows/ci.yml — Ajouter un job supplémentaire :
  rust-features:
    name: Rust (sandbox + ONNX features)
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: sage-core
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install maturin
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --features sandbox,cranelift
      - name: ONNX compile check
        run: cargo check --features onnx
```

---

### TST-04 [LOW] — Badge README périmé

**Fichier:** `README.md:13`

**Preuve :** Badge dit "730 passed", 847 test functions existent.

**Action :**
```markdown
<!-- Remplacer ligne 13 : -->
<img src="https://img.shields.io/badge/tests-847%20passed-brightgreen?style=flat-square" alt="Tests">
```

Ou mieux : lier le badge au CI pour mise à jour automatique.

---

## PHASE 3 — BUILD & DOCS

### BLD-01 [MEDIUM] — Rust edition 2021, pas de lints workspace

**Fichier:** `Cargo.toml`

**Action :**
```toml
[workspace]
members = ["sage-core"]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2024"  # ou au minimum 2021 avec lints
license = "MIT"
rust-version = "1.85"  # MSRV

[workspace.lints.rust]
unsafe_code = "deny"

[workspace.lints.clippy]
all = "warn"
pedantic = "warn"
```

Dans `sage-core/Cargo.toml` ajouter :
```toml
[lints]
workspace = true
```

---

### BLD-02 [LOW] — ca-bundle.pem dans le repo

**Fichier:** `Cert/ca-bundle.pem` (79KB)

**Action :**
```bash
echo "Cert/" >> .gitignore
git rm -r --cached Cert/
```

Documenter dans README :
```markdown
# Si derrière un proxy corporate :
export SSL_CERT_FILE=/path/to/your/ca-bundle.pem
```

---

### DOC-01 [MEDIUM] — Ratio doc/code ~ 1:1

**Preuve :** `docs/plans/` = 832K à lui seul. 31K lignes de markdown vs 34K lignes de code.

**Action :** Archiver les plans obsolètes :
```bash
mkdir -p docs/archive
mv docs/plans/*sprint*obsolete* docs/archive/
```

Garder uniquement : ARCHITECTURE.md, ADRs, README. Tout le reste dans `docs/archive/`.

---

## RÉSUMÉ DES PRIORITÉS

| Priorité | Actions | Effort estimé |
|----------|---------|---------------|
| P0 (immédiat) | Z3-01, Z3-02, Z3-03, Z3-05, Z3-06, Z3-07, SEC-01, SEC-02, SEC-03 | 2-3 jours |
| P1 (semaine 1-2) | SEC-04, EVO-01, EVO-03, RTG-01, RTG-02, TST-01, TST-02, TST-03 | 1-2 semaines |
| P2 (semaine 2-4) | MEM-01, RTG-03, EVO-02, MEM-02, MEM-03, BLD-01 | 2-3 semaines |
| P3 (mois 2+) | MEM-04, TST-04, BLD-02, DOC-01 | continu |

---

## COMMANDE DE VÉRIFICATION POST-FIX

Après application des fixes, exécuter :

```bash
# 1. Vérifier que z3_validator.rs est supprimé
test ! -f sage-core/src/sandbox/z3_validator.rs && echo "OK: dead code removed"

# 2. Vérifier que check_loop_bound a une contrainte >= 0
grep -A5 "def check_loop_bound" sage-python/src/sage/sandbox/z3_validator.py | grep ">= 0" && echo "OK: loop bound constrained"
grep -A5 "def check_loop_bound" sage-python/src/sage/topology/kg_rlvr.py | grep ">= 0\|return False" && echo "OK: kg_rlvr loop bound fixed"

# 3. Vérifier que verify_arithmetic ne crée plus de variable libre
grep -A10 "def verify_arithmetic" sage-python/src/sage/topology/kg_rlvr.py | grep -v "z3.Int" && echo "OK: no unconstrained symbolic var"

# 4. Vérifier que z3_topology.py est renommé
test ! -f sage-python/src/sage/topology/z3_topology.py && echo "OK: z3_topology renamed"

# 5. Vérifier que shell=True est supprimé de meta.py
grep -c "shell=True" sage-python/src/sage/tools/meta.py | grep "^0$" && echo "OK: no shell=True"

# 6. Vérifier que create_subprocess_shell est remplacé
grep -c "create_subprocess_shell" sage-python/src/sage/tools/builtin.py | grep "^0$" && echo "OK: no subprocess_shell"

# 7. Vérifier fallback Z3 fail-closed
grep -A2 "if not self.has_z3:" sage-python/src/sage/topology/kg_rlvr.py | grep "return False\|return 0 <=" && echo "OK: Z3 fallback fail-closed"

# 8. Run full test suite
cd sage-python && python -m pytest tests/ -v --tb=short 2>&1 | tail -5
cd ../sage-core && cargo test --no-default-features 2>&1 | tail -3
cd ../sage-core && cargo clippy --no-default-features -- -D warnings 2>&1 | tail -3
```
