"""Knowledge Graph - Reinforcement Learning from Verifiable Rewards (KG-RLVR).

Implements Process Reward Models (PRM) to evaluate internal reasoning paths 
(<think> tags) using a Knowledge Graph, shifting from Outcome-based to 
Process-based rewards (System 3 AI).
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Any, Tuple

class SimpleKnowledgeGraph:
    """A lightweight, mockable Knowledge Graph for verifiable reasoning."""
    def __init__(self):
        # Format: (Subject, Predicate, Object)
        self.triples = set([
            ("YGN-SAGE", "uses", "SAMPO"),
            ("SAMPO", "prevents", "Amnesia"),
            ("Docker", "has", "HighLatency"),
            ("eBPF", "has", "LowLatency"),
            ("YGN-SAGE", "uses", "eBPF"),
            ("DGM", "mutates", "Hyperparameters"),
            ("System3", "requires", "ProcessRewards")
        ])

    def verify_step(self, step: str) -> float:
        """Score a reasoning step based on its alignment with the KG."""
        step_lower = step.lower()
        score = 0.0
        matched = False
        
        for subj, pred, obj in self.triples:
            # Simple heuristic: if both subject and object are mentioned, it's a verifiable step
            if subj.lower() in step_lower and obj.lower() in step_lower:
                score += 1.0
                matched = True
                
        # If it makes a logical deduction but isn't explicitly in the graph, we give partial credit
        # In a real system, we'd use Graph-RFT or MCTS to validate the traversal path.
        if not matched and len(step) > 10:
            score += 0.1
            
        return min(1.0, score)


class ProcessRewardModel:
    """Evaluates agent reasoning paths using verifiable KG rewards."""
    
    def __init__(self, kg: SimpleKnowledgeGraph = None):
        self.kg = kg or SimpleKnowledgeGraph()
        self.logger = logging.getLogger(__name__)

    def extract_reasoning_steps(self, content: str) -> List[str]:
        """Extracts text inside <think>...</think> tags and splits into steps."""
        pattern = r"<think>(.*?)</think>"
        matches = re.findall(pattern, content, re.DOTALL)
        
        steps = []
        for match in matches:
            # Split by common step indicators or just newlines
            raw_steps = [s.strip() for s in match.split('\n') if s.strip()]
            steps.extend(raw_steps)
            
        return steps

    def calculate_r_path(self, content: str) -> Tuple[float, Dict[str, Any]]:
        """Calculate the R_path (Process Reward) for a given generation."""
        steps = self.extract_reasoning_steps(content)
        
        if not steps:
            # Penalty for not reasoning (System 1 instead of System 3)
            return -1.0, {"error": "No <think> blocks found. System 3 reasoning required."}
            
        step_scores = []
        for step in steps:
            score = self.kg.verify_step(step)
            step_scores.append(score)
            
        # Overall R_path is the average of verifiable steps
        # This prevents hallucination chains where only the final outcome is correct
        r_path = sum(step_scores) / len(step_scores)
        
        details = {
            "total_steps": len(steps),
            "step_scores": step_scores,
            "verifiable_ratio": sum(1 for s in step_scores if s >= 1.0) / len(steps)
        }
        
        return r_path, details
