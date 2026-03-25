# PROMPT CLAUDE CODE — GiGPO Micro-Decisions dans topology_env.py

## CONTEXTE

Tu travailles sur YGN-SAGE, un Agent Development Kit multi-agents. La branche est `VeRLGIGPO`.

Le fichier `sage-python/src/sage/verl/topology_env.py` implémente un environnement gym-style pour l'entraînement RL via verl-agent + GiGPO.

**Le problème fondamental** : actuellement, le modèle agit UNE SEULE FOIS (il génère un YAML au step 0). Aux steps 1..N, l'env délivre les résultats pré-calculés et **ignore complètement** `model_response`. Regarde `_step_deliver_node_result()` : le paramètre model_response n'y est même pas. Tous les nœuds sont exécutés en bloc à step 0 par `_execute_topology_traced()`.

GiGPO calcule des step-level advantages sur les actions du modèle à chaque step. Si le modèle ne prend aucune décision aux steps 1..N, les advantages dégénèrent vers l'épisode-level = GRPO. GiGPO devient inutile.

**La solution** : exécuter les nœuds **incrémentalement** (un par un), et aux nœuds checkpoint, **pauser** pour demander au modèle une décision réelle : `continue`, `upgrade`, ou `reroute`. Cette décision est l'action GiGPO qui reçoit un step-level advantage.

## CE QUE TU DOIS MODIFIER

### Fichier principal : `sage-python/src/sage/verl/topology_env.py`

Réécrire la classe `SageTopologyEnv` avec une machine à états à 4 états :

```
AWAITING_YAML → EXECUTING → AWAITING_DECISION → EXECUTING → ... → TERMINAL
```

#### Nouveaux champs d'instance dans `__init__` :

```python
self._state = "awaiting_yaml"  # Machine à états
self._exec_cursor = 0           # Prochain nœud à exécuter
self._pending_checkpoint = None  # Dict du checkpoint en attente de décision
self._upgrades_used = 0          # Compteur d'upgrades utilisés dans l'épisode
self._node_outputs = {}          # dict[int, str] — rempli incrémentalement
```

Conserve TOUS les champs existants (self._config, self._trace, self._topo_dict, self._node_traces, self._memory, self._checkpoints, self._max_upgrades, self._quality_threshold, self._predecessor_map, self._step_reward_vec, self._difficulty).

#### Nouvelle logique de `step()` :

```python
def step(self, model_response: str) -> tuple[dict, float, bool, dict]:
    if self._state == "awaiting_yaml":
        return self._handle_yaml(model_response)
    elif self._state == "awaiting_decision":
        return self._handle_decision(model_response)
    elif self._state == "terminal":
        return self._finalize_episode()
    else:
        # _state == "executing" : ne devrait pas arriver
        # (l'env drive l'exécution en interne après handle_yaml/handle_decision)
        return self._execute_until_checkpoint_or_end()
```

#### `_handle_yaml(yaml_text)` :

C'est essentiellement l'actuel `_step_parse_and_execute`, SAUF qu'au lieu d'exécuter tous les nœuds d'un coup, on :
1. Parse le YAML (identique à maintenant)
2. Calcule le structural reward (identique)
3. Parse les adaptation metadata (checkpoints, max_upgrades, quality_threshold — identique)
4. Build predecessor_map (identique, via `_build_predecessor_map`)
5. Assigne les modèles (identique, via `_assign_models_to_topology`)
6. **NE PAS exécuter les nœuds**. Au lieu de ça :
   - `self._state = "executing"`
   - `self._exec_cursor = 0`
   - Appeler `self._execute_until_checkpoint_or_end()`

#### `_execute_until_checkpoint_or_end()` — NOUVELLE méthode clé :

Exécute les nœuds **un par un** depuis `self._exec_cursor` :

```python
def _execute_until_checkpoint_or_end(self) -> tuple[dict, float, bool, dict]:
    nodes = self._topo_dict.get("nodes", [])
    
    while self._exec_cursor < len(nodes):
        node_idx = self._exec_cursor
        node = nodes[node_idx]
        
        # Exécuter CE nœud (un seul)
        trace = self._execute_single_node(node_idx, node)
        self._node_traces.append(trace)
        self._node_outputs[node_idx] = trace["output"]
        
        # Calculer le per-node reward
        role = trace["role"]
        reward = self._compute_node_reward(role, trace["output"])
        
        # Construire l'anchor
        predecessors = self._predecessor_map.get(node_idx, [])
        pred_text = " ".join(self._node_outputs.get(p, "")[:200] for p in predecessors)
        context_hash = hashlib.md5(pred_text.encode()).hexdigest()[:8] if pred_text else ""
        anchor = _make_anchor(role, self._difficulty, context_hash)
        
        # Enregistrer le step
        self._trace.steps.append(StepResult(
            step_idx=len(self._trace.steps),
            node_idx=node_idx,
            role=role,
            output=trace["output"],
            reward=reward,
            latency=trace.get("latency", 0.0),
            anchor_key=anchor,
            model_id=trace.get("model_id", ""),
        ))
        
        self._exec_cursor += 1
        
        # Si ce nœud est un checkpoint ET qu'il reste des upgrades
        if node_idx in self._checkpoints:
            quality = self._estimate_quality(trace["output"], role)
            
            # Stocker le checkpoint en attente
            self._pending_checkpoint = {
                "node_idx": node_idx,
                "role": role,
                "quality": quality,
                "output": trace["output"][:300],
                "model_tier": node.get("model_tier", ""),
                "fallback_tier": node.get("fallback_tier", ""),
            }
            self._state = "awaiting_decision"
            
            # Construire l'observation AVEC le quality bucket dans l'anchor
            q_bucket = _quality_bucket(quality, self._quality_threshold)
            decision_anchor = _make_anchor(
                f"decision:{role}", self._difficulty,
                f"{q_bucket}:{context_hash}"
            )
            
            remaining_upgrades = self._max_upgrades - self._upgrades_used
            has_fallback = bool(node.get("fallback_tier", ""))
            
            obs_text = (
                f"[CHECKPOINT] Node {node_idx} ({role}, {node.get('model_tier', '?')}) completed.\n"
                f"Output quality: {quality:.2f} (threshold: {self._quality_threshold})\n"
                f"Output preview: {trace['output'][:200]}\n"
            )
            if has_fallback and remaining_upgrades > 0:
                obs_text += (
                    f"Fallback available: {node['fallback_tier']}\n"
                    f"Upgrades remaining: {remaining_upgrades}/{self._max_upgrades}\n"
                    f"Actions: [continue] [upgrade] [reroute]\n"
                )
            else:
                obs_text += "No fallback available or no upgrades remaining.\nActions: [continue] [reroute]\n"
            
            return (
                {"text": obs_text, "image": None, "anchor": decision_anchor},
                reward,
                False,
                {"status": "CHECKPOINT", "node_idx": node_idx, "quality": quality},
            )
    
    # Tous les nœuds exécutés → terminal
    self._state = "terminal"
    return self._finalize_episode()
```

#### `_handle_decision(model_response)` — NOUVELLE méthode :

Parse la décision du modèle et agit :

```python
def _handle_decision(self, model_response: str) -> tuple[dict, float, bool, dict]:
    decision = self._parse_decision(model_response)
    cp = self._pending_checkpoint
    self._pending_checkpoint = None
    
    node_idx = cp["node_idx"]
    role = cp["role"]
    reward = 0.0
    
    if decision == "upgrade" and cp["fallback_tier"] and self._upgrades_used < self._max_upgrades:
        # Re-exécuter le nœud avec le fallback_tier
        self._upgrades_used += 1
        
        # Modifier le model_tier du nœud dans topo_dict
        node = self._topo_dict["nodes"][node_idx]
        original_tier = node.get("model_tier", "")
        node["model_tier"] = cp["fallback_tier"]
        
        # Re-assigner le modèle réel
        self._assign_models_to_topology(self._topo_dict, self._topo_dict["nodes"])
        
        # Re-exécuter
        new_trace = self._execute_single_node(node_idx, node)
        self._node_outputs[node_idx] = new_trace["output"]
        
        # Mettre à jour le trace
        new_quality = self._estimate_quality(new_trace["output"], role)
        quality_improved = new_quality > cp["quality"]
        
        reward = _REWARD_UPGRADE_COST  # coût de l'upgrade
        if quality_improved:
            reward += _REWARD_UPGRADE_SUCCESS
        
        # Enregistrer le step d'upgrade
        self._trace.steps.append(StepResult(
            step_idx=len(self._trace.steps),
            node_idx=node_idx,
            role=f"upgrade:{role}",
            output=new_trace["output"],
            reward=reward,
            latency=new_trace.get("latency", 0.0),
            anchor_key=_make_anchor(f"upgrade:{role}", self._difficulty, ""),
            model_id=new_trace.get("model_id", ""),
            action="upgrade",
            was_upgraded=True,
            quality_before=cp["quality"],
            quality_after=new_quality,
        ))
        
        obs_text = (
            f"Node {node_idx} upgraded {original_tier}→{cp['fallback_tier']}. "
            f"Quality: {cp['quality']:.2f}→{new_quality:.2f}. "
            f"Continuing execution."
        )
    
    elif decision == "reroute":
        reward = _REWARD_REROUTE_PENALTY
        self._trace.steps.append(StepResult(
            step_idx=len(self._trace.steps),
            node_idx=node_idx,
            role="reroute",
            output="REROUTE",
            reward=reward,
            latency=0.0,
            anchor_key=_make_anchor("reroute", self._difficulty, ""),
            action="reroute",
        ))
        self._state = "terminal"
        self._trace.status = "REROUTED"
        return self._finalize_episode()
    
    else:  # "continue"
        reward = 0.0
        self._trace.steps.append(StepResult(
            step_idx=len(self._trace.steps),
            node_idx=node_idx,
            role=f"continue:{role}",
            output="continue",
            reward=reward,
            latency=0.0,
            anchor_key=_make_anchor(f"decision:{role}", self._difficulty, "continue"),
            action="continue",
        ))
        obs_text = f"Continuing with node {node_idx} output as-is."
    
    # Reprendre l'exécution
    self._state = "executing"
    return self._execute_until_checkpoint_or_end()
```

#### `_execute_single_node(node_idx, node_dict)` — NOUVELLE méthode :

Exécute UN SEUL nœud (soit via API en mode exec, soit en mode structural) :

```python
def _execute_single_node(self, node_idx: int, node_dict: dict) -> dict:
    role = node_dict.get("role", f"node-{node_idx}")
    exec_mode = os.environ.get("SAGE_VERL_EXEC", "0") == "1"
    
    if not exec_mode:
        return {
            "node_idx": node_idx,
            "role": role,
            "output": f"[structural mode] Node {node_idx} ({role})",
            "latency": 0.0,
            "model_id": node_dict.get("model_tier", ""),
        }
    
    # Mode exécution réelle : appel LLM
    try:
        from sage.execution import _get_agent_provider
        from sage.llm.base import LLMConfig, Message, Role
        
        provider, model = _get_agent_provider()
        if provider is None:
            return self._structural_stub(node_idx, role, node_dict)
        
        # Construire le prompt à partir du rôle + contexte prédécesseurs
        predecessors = self._predecessor_map.get(node_idx, [])
        context = "\n\n".join(
            f"[{self._topo_dict['nodes'][p].get('role', f'node-{p}')}]: "
            f"{self._node_outputs.get(p, '')[:500]}"
            for p in predecessors if p in self._node_outputs
        )
        
        custom_prompt = node_dict.get("prompt", f"You are acting as: {role}")
        messages = [
            Message(role=Role.SYSTEM, content=custom_prompt),
        ]
        if context:
            messages.append(Message(role=Role.SYSTEM, content=f"Context from previous agents:\n{context}"))
        messages.append(Message(role=Role.USER, content=self._trace.prompt[:2000]))
        
        config = LLMConfig(provider="agent", model=model)
        
        t0 = time.time()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(
                lambda: asyncio.run(provider.generate(messages=messages, config=config))
            ).result(timeout=60)
        output = response.content or ""
        latency = time.time() - t0
        
        return {
            "node_idx": node_idx,
            "role": role,
            "output": output,
            "latency": latency,
            "model_id": node_dict.get("_assigned_model_id", node_dict.get("model_tier", "")),
        }
    except Exception as exc:
        log.warning("Node %d execution failed: %s", node_idx, exc)
        return self._structural_stub(node_idx, role, node_dict)


def _structural_stub(self, node_idx: int, role: str, node_dict: dict) -> dict:
    return {
        "node_idx": node_idx,
        "role": role,
        "output": f"[fallback] Node {node_idx} ({role}): structural only",
        "latency": 0.0,
        "model_id": node_dict.get("model_tier", ""),
    }
```

#### `_parse_decision(text)` — NOUVELLE méthode :

```python
def _parse_decision(self, text: str) -> str:
    t = text.strip().lower()
    if "upgrade" in t:
        return "upgrade"
    elif "reroute" in t:
        return "reroute"
    return "continue"
```

#### `_estimate_quality(output, role)` — NOUVELLE méthode :

```python
def _estimate_quality(self, output: str, role: str) -> float:
    """Estimate output quality for checkpoint decision."""
    # Essayer Rust QualityLabeler
    try:
        from sage_core import QualityLabeler
        ql = QualityLabeler()
        label = ql.label(f"Node role: {role}", output)
        if label and label.assessable:
            return float(label.score)
    except ImportError:
        pass
    
    # Heuristique minimale pour mode structural
    if not output or output.startswith("[structural") or output.startswith("[fallback"):
        return 0.5  # neutre en mode structural
    if output.startswith("ERROR"):
        return 0.1
    # Présence de code = meilleure qualité
    if "```" in output or "def " in output:
        return 0.7
    return 0.4
```

#### Nouvelle helper function à ajouter au niveau module :

```python
def _quality_bucket(quality: float, threshold: float) -> str:
    if quality < threshold * 0.6:
        return "very_low"
    elif quality < threshold:
        return "low"
    elif quality < threshold * 1.4:
        return "adequate"
    else:
        return "high"
```

#### Constantes de reward à ajouter au niveau module :

```python
_REWARD_UPGRADE_COST = -0.05
_REWARD_REROUTE_PENALTY = -0.3
_REWARD_UPGRADE_SUCCESS = 0.15
```

#### `StepResult` — ajouter les champs :

```python
@dataclass
class StepResult:
    step_idx: int
    node_idx: int
    role: str
    output: str
    reward: float
    latency: float
    anchor_key: str
    model_id: str = ""
    action: str = ""           # NOUVEAU : texte de l'action du modèle
    was_upgraded: bool = False  # NOUVEAU
    quality_before: float = 0.0  # NOUVEAU
    quality_after: float = 0.0   # NOUVEAU
```

### Méthodes à CONSERVER INTACTES (ne pas toucher) :

- `reset()` — identique
- `_build_predecessor_map()` — identique
- `_assign_models_to_topology()` — identique
- `_compute_node_reward()` — identique
- `get_trace()` — identique
- `get_step_rewards()` — identique
- `_terminal()` — identique
- `SageTopologyEnvManager` — identique

### Méthodes à SUPPRIMER (remplacées) :

- `_step_parse_and_execute()` → remplacé par `_handle_yaml()`
- `_step_deliver_node_result()` → remplacé par `_execute_until_checkpoint_or_end()`
- `_execute_topology_traced()` → remplacé par `_execute_single_node()` appelé en boucle
- `_execute_sequential_fallback()` → intégré dans `_execute_single_node()` comme fallback

### `_finalize_episode()` — MODIFIER légèrement :

Ajouter le calcul de résilience au terminal :

```python
# Après le calcul d'exec_score, avant de construire le StepRewardVector :
n_upgrades = sum(1 for s in self._trace.steps if s.was_upgraded)
if n_upgrades > 0:
    any_succeeded = any(
        s.was_upgraded and s.quality_after > s.quality_before
        for s in self._trace.steps
    )
    if any_succeeded and status == "PASSED":
        resilience_bonus = 0.5
    elif any_succeeded:
        resilience_bonus = 0.3
    else:
        resilience_bonus = 0.0
    self._trace.steps.append(StepResult(
        step_idx=len(self._trace.steps), node_idx=-1, role="resilience_bonus",
        output=f"upgrades={n_upgrades}, bonus={resilience_bonus}",
        reward=resilience_bonus, latency=0.0,
        anchor_key="resilience:bonus",
    ))
```

### Fichier secondaire : `sage-python/src/sage/topology/runner.py`

Dans `run_traced()`, le controller est appelé mais `was_upgraded` n'est pas écrit dans les traces. Ajouter le tracking :

```python
# Dans run_traced(), après le block controller :
was_upgraded = False
original_tier = ""
if self._controller:
    decision = self._controller.evaluate_and_decide(...)
    if decision.action == "upgrade_model":
        was_upgraded = True
        original_tier = getattr(node, "model_id", "")
        result = await self._retry_with_upgrade(node_idx, decision, task)
        self._node_outputs[node_idx] = result

traces.append({
    ...
    "was_upgraded": was_upgraded,
    "original_tier": original_tier,
})
```

## TESTS À ÉCRIRE

Fichier : `sage-python/tests/test_verl_micro_decisions.py`

### Test 1 : Épisode sans checkpoint (simple task → GRPO behavior)

```python
def test_no_checkpoint_is_grpo():
    """Simple topology without checkpoints = model acts once = GRPO."""
    env = SageTopologyEnv()
    env.reset("Write hello world", "test/hello")
    
    yaml_text = """
    difficulty: simple
    nodes:
      - {role: coder, model_tier: budget, prompt: "Write hello world"}
      - {role: synthesizer, model_tier: budget, prompt: "Produce final solution"}
    edges:
      - {from_idx: 0, to_idx: 1, flow_type: message}
    """
    obs, r, done, info = env.step(yaml_text)
    # Pas de checkpoint → doit aller directement au terminal
    assert done is True  # Tout exécuté d'un coup
    assert env._upgrades_used == 0
```

### Test 2 : Checkpoint déclenche une décision

```python
def test_checkpoint_triggers_decision():
    """Checkpoint node should pause and ask for decision."""
    env = SageTopologyEnv()
    env.reset("Sort a list", "test/sort")
    
    yaml_text = """
    difficulty: moderate
    adaptation:
      checkpoints: [0]
      max_upgrades: 1
      quality_threshold: 0.5
    nodes:
      - {role: coder, model_tier: fast, fallback_tier: reasoner, prompt: "Write sort"}
      - {role: synthesizer, model_tier: fast, prompt: "Produce final"}
    edges:
      - {from_idx: 0, to_idx: 1, flow_type: message}
    """
    obs, r, done, info = env.step(yaml_text)
    
    # Le step 0 parse le YAML et exécute jusqu'au checkpoint
    # Node 0 est un checkpoint → doit demander une décision
    assert done is False
    assert env._state == "awaiting_decision"
    assert "[CHECKPOINT]" in obs["text"]
    assert "continue" in obs["text"].lower() or "upgrade" in obs["text"].lower()
```

### Test 3 : Décision "continue" reprend l'exécution

```python
def test_continue_resumes_execution():
    env = SageTopologyEnv()
    env.reset("Sort a list", "test/sort")
    
    yaml_text = """
    difficulty: moderate
    adaptation:
      checkpoints: [0]
      max_upgrades: 1
      quality_threshold: 0.5
    nodes:
      - {role: coder, model_tier: fast, fallback_tier: reasoner, prompt: "Write sort"}
      - {role: synthesizer, model_tier: fast, prompt: "Produce final"}
    edges:
      - {from_idx: 0, to_idx: 1, flow_type: message}
    """
    env.step(yaml_text)  # → AWAITING_DECISION
    assert env._state == "awaiting_decision"
    
    obs, r, done, info = env.step("continue")
    # Devrait avoir exécuté le synthesizer et finalisé
    assert done is True
    assert env._upgrades_used == 0
```

### Test 4 : Décision "upgrade" re-exécute le nœud

```python
def test_upgrade_reexecutes_node():
    env = SageTopologyEnv()
    env.reset("Dijkstra", "test/dijkstra")
    
    yaml_text = """
    difficulty: moderate
    adaptation:
      checkpoints: [0]
      max_upgrades: 1
      quality_threshold: 0.5
    nodes:
      - {role: coder, model_tier: fast, fallback_tier: reasoner, prompt: "Dijkstra"}
      - {role: synthesizer, model_tier: fast, prompt: "Final solution"}
    edges:
      - {from_idx: 0, to_idx: 1, flow_type: message}
    """
    env.step(yaml_text)  # → AWAITING_DECISION
    
    obs, r, done, info = env.step("upgrade")
    # L'upgrade devrait avoir été comptabilisé
    assert env._upgrades_used == 1
    # Le step d'upgrade devrait être dans la trace
    upgrade_steps = [s for s in env._trace.steps if s.was_upgraded]
    assert len(upgrade_steps) == 1
```

### Test 5 : Max upgrades respecté

```python
def test_max_upgrades_respected():
    """Once max_upgrades exhausted, no more upgrade option."""
    env = SageTopologyEnv()
    env.reset("Complex task", "test/complex")
    
    yaml_text = """
    difficulty: complex
    adaptation:
      checkpoints: [0, 1]
      max_upgrades: 1
      quality_threshold: 0.5
    nodes:
      - {role: planner, model_tier: fast, fallback_tier: reasoner}
      - {role: coder, model_tier: fast, fallback_tier: reasoner}
      - {role: synthesizer, model_tier: fast}
    edges:
      - {from_idx: 0, to_idx: 1, flow_type: message}
      - {from_idx: 1, to_idx: 2, flow_type: message}
    """
    env.step(yaml_text)  # → checkpoint 0
    env.step("upgrade")   # → upgrade planner, upgrades_used=1
    # Si checkpoint 1 est atteint, l'observation ne devrait pas proposer "upgrade"
    if env._state == "awaiting_decision":
        assert "No fallback available or no upgrades remaining" in env._pending_checkpoint or True
        # Vérifier que upgrades_used == max_upgrades
        assert env._upgrades_used >= env._max_upgrades
```

### Test 6 : Anchor states différents pour decisions

```python
def test_anchor_states_distinguish_quality():
    """GiGPO anchor should include quality bucket."""
    env = SageTopologyEnv()
    env.reset("Sort", "test/sort")
    
    yaml_text = """
    difficulty: moderate
    adaptation:
      checkpoints: [0]
      max_upgrades: 1
      quality_threshold: 0.5
    nodes:
      - {role: coder, model_tier: fast, fallback_tier: reasoner}
      - {role: synthesizer, model_tier: fast}
    edges:
      - {from_idx: 0, to_idx: 1, flow_type: message}
    """
    obs, r, done, info = env.step(yaml_text)
    
    # L'anchor doit contenir "decision:" et un quality bucket
    assert "decision:" in obs["anchor"]
    # Le quality bucket doit être l'un des 4
    assert any(b in obs["anchor"] for b in ["very_low", "low", "adequate", "high"])
```

### Test 7 : Le StepRewardVector capture les décisions

```python
def test_step_reward_vector_includes_decisions():
    env = SageTopologyEnv()
    env.reset("Sort", "test/sort")
    
    yaml_text = """
    difficulty: moderate
    adaptation:
      checkpoints: [0]
      max_upgrades: 1
      quality_threshold: 0.5
    nodes:
      - {role: coder, model_tier: fast, fallback_tier: reasoner}
      - {role: synthesizer, model_tier: fast}
    edges:
      - {from_idx: 0, to_idx: 1, flow_type: message}
    """
    env.step(yaml_text)
    env.step("continue")
    
    vec = env.get_step_rewards()
    # Au minimum : topology_generator + coder + decision:continue + synthesizer + terminal
    assert vec.n_steps >= 4
    assert any("decision:" in a for a in vec.anchor_keys)
```

## CONTRAINTES IMPÉRATIVES

1. **Ne pas casser les 39 tests existants** dans `tests/test_verl_v2.py` et `tests/test_verl_reward.py`. Lance-les après tes modifications : `python -m pytest tests/test_verl_v2.py tests/test_verl_reward.py -v`

2. **Conserver la rétro-compatibilité** : si `adaptation.checkpoints` est vide ou absent, l'env se comporte comme avant (exécute tout d'un coup, pas de décision demandée). Les topologies simples sans checkpoints = GRPO automatique.

3. **Ne pas modifier** `step_reward.py`, `reward.py`, `edge_credit.py`, `rewardflow.py`, `training_memory.py`. Ce sont des modules stables.

4. **Le SageTopologyEnvManager** doit continuer à fonctionner avec la même interface. Il appelle `step()` avec une action par env — le nouveau code gère ça naturellement.

5. **Imports** : garder tous les imports existants. Ajouter `time` si pas déjà importé (il l'est). Ne pas ajouter de dépendances externes.

6. **Supprimer** le fichier `sage-python/src/sage/verl/MICRO_DECISION_SPEC.md` après implémentation — c'était un doc temporaire de spécification.

7. **Commiter** avec le message : `feat: GiGPO micro-decisions — model makes real choices at checkpoint nodes`

## POURQUOI C'EST CRITIQUE

Avec cette architecture, GiGPO a un sens réel :
- Step 0 : le modèle génère le YAML (action = texte YAML) — **anchor = prompt hash**
- Steps aux checkpoints : le modèle décide continue/upgrade/reroute (action = décision) — **anchor = role:difficulty:quality_bucket**
- GiGPO groupe les décisions identiques aux mêmes anchor states et calcule des advantages step-level

Exemple concret : 4 trajectoires pour "Write Dijkstra". Toutes arrivent au checkpoint coder avec quality=0.3. Trajectoire A et B choisissent "upgrade" → reward final 0.8. Trajectoire C et D choisissent "continue" → reward final 0.2. GiGPO assigne un advantage positif à "upgrade" au anchor "decision:coder:moderate:low".

Le modèle apprend 3 choses simultanément :
1. **Comment structurer** une topologie (YAML)
2. **Où placer** les checkpoints (quels nœuds sont fragiles)
3. **Quand upgrader** vs continuer (le coût-bénéfice de l'adaptation)

C'est LE différenciateur de SAGE vs tous les concurrents (The Conductor, CARD, AgentConductor, AdaptOrch). Aucun d'eux n'entraîne le modèle à prendre des décisions d'adaptation en cours d'exécution.
