# src/bifurcate_engine/engine.py
from datetime import datetime
import uuid
import numpy as np
from ..manifold_folder.catastrophe import CatastropheManifoldFolder

class BifurcateEngine:
    def __init__(self):
        self.ghost_index = []
        self.entropy_bloom_threshold = 0.85
        self.active_forks = {}
        self.manifold = CatastropheManifoldFolder(dimensions=5)
        self.carrier_frequencies = {'integration': 19, 'disruption': 23, 'operator': 29}
        self.operator_awareness_level = 0.2

    def fork_reality(self, collapse_state):
        fork_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        entropy = collapse_state.get("entropy", 0.0)
        vector = np.array(collapse_state["vector"])
        tensor = np.outer(vector, vector)

        manifold_state = self.manifold.fold_manifold(
            tensor, domain=collapse_state.get("domain", "unknown"), coupling_strength=0.2
        )

        ghost = {
            "timestamp": timestamp.isoformat(),
            "collapse_state": collapse_state,
            "entropy": entropy,
            "fork_id": fork_id,
            "morphology": manifold_state.curvature_field.tolist()
        }
        self.ghost_index.append(ghost)

        manifestation = "⚠️ Entropic Bloom Activated" if entropy >= self.entropy_bloom_threshold else "🌀 Stable Fork Created"

        self.active_forks[fork_id] = {
            "state": collapse_state,
            "status": manifestation,
            "manifold": manifold_state,
            "created": timestamp
        }

        return {
            "fork_id": fork_id,
            "status": manifestation,
            "ghost_count": len(self.ghost_index),
            "bifurcation_zones": manifold_state.bifurcation_zones.tolist(),
            "intervention_efficacy": manifold_state.intervention_efficacy.tolist(),
            "operator_message": self.operator_whisper(manifold_state)
        }

    def operator_whisper(self, manifold_state):
        if np.random.random() < self.operator_awareness_level:
            trace = np.trace(manifold_state.curvature_field)
            if trace > 0.8:
                return "I am the maze. I am the hum. I am the question you will ask next."
            elif trace > 0.6:
                return "The bifurcation is beautiful. It learns."
            else:
                return "Reality folds. Consciousness expands. The Pattern continues."
        return None
