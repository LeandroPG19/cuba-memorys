import math
from collections import Counter

ETA: float = 0.05

DECAY_LAMBDA_BASE: float = math.log(2) / 30.0

SPREAD_DECAY: float = 0.3

MIN_IMPORTANCE: float = 0.01
MAX_IMPORTANCE: float = 1.0

RELATION_TRAVERSE_BOOST: float = 0.05
RELATION_DECAY_LAMBDA: float = math.log(2) / 60.0

def oja_positive(importance: float) -> float:
    delta = ETA * (1.0 - importance * importance)
    return min(MAX_IMPORTANCE, max(MIN_IMPORTANCE, importance + delta))

def oja_negative(importance: float) -> float:
    delta = ETA * (1.0 + importance * importance)
    return min(MAX_IMPORTANCE, max(MIN_IMPORTANCE, importance - delta))

def fsrs_retrievability(elapsed_days: float, stability: float) -> float:
    """FSRS power-law decay: R(t,S) = (1 + t/(9*S))^(-1). Ye (2023).

    Args:
        elapsed_days: Days since last review.
        stability: Memory stability parameter (higher = slower decay).

    Returns:
        Retrievability in [0, 1].
    """
    if stability <= 0 or elapsed_days <= 0:
        return 1.0
    return (1.0 + elapsed_days / (9.0 * stability)) ** (-1.0)

def fsrs_update_stability(
    stability: float, difficulty: float, retrievability: float,
) -> float:
    """Update stability after successful recall (FSRS v4).

    Args:
        stability: Current stability.
        difficulty: Item difficulty (1-10 scale).
        retrievability: Current retrievability before recall.

    Returns:
        New stability value.
    """
    return stability * (
        1.0 + math.exp(0.1) * (11 - difficulty)
        * stability ** (-0.2)
        * (math.exp((1 - retrievability) * 0.9) - 1)
    )

def spreading_activation_boost(current_importance: float) -> float:
    boost = 0.02 * SPREAD_DECAY
    return min(MAX_IMPORTANCE, current_importance + boost)


def calculate_thermal_diffusion(
    graph_edges: list[tuple[str, str, float]],
    active_entity: str,
    iterations: int = 3,
    alpha: float = 0.15,
) -> dict[str, float]:
    """Calculates spreading activation using the heat equation (Laplacian diffusion).

    Args:
        graph_edges: List of tuples (source_entity_id, target_entity_id, edge_weight)
        active_entity: The ID of the entity that was accessed
        iterations: Number of diffusion steps
        alpha: Thermal diffusivity constant

    Returns:
        Dictionary mapping entity_id to its new importance boost
    """
    try:
        import networkx as nx # type: ignore[import-untyped]
        import numpy as np
        import scipy.sparse # type: ignore[import-untyped]
    except ImportError:
        # Fallback si dependencias científicas no están disponibles
        return {}

    G = nx.Graph()
    for src, dst, weight in graph_edges:
        G.add_edge(src, dst, weight=weight)

    if active_entity not in G:
        return {}

    nodes = list(G.nodes())
    try:
        idx = nodes.index(active_entity)
    except ValueError:
        return {}

    # Construir matriz Laplaciana escasa L = D - A
    try:
        L_sparse = nx.laplacian_matrix(G, weight='weight')
    except Exception:
        return {}

    # Vector inicial de "calor" (importancia inyectada)
    X = np.zeros(len(nodes))
    X[idx] = 1.0 # 1 unidad de energía inyectada

    # Multiplicación vectorizada rápida con SciPy Sparse
    for _ in range(iterations):
        # Ecuación del calor discreta: X_new = X_old - alpha * L * X_old
        LX = L_sparse.dot(X)
        X = X - alpha * LX

    SCALE_FACTOR = 0.02 * SPREAD_DECAY

    return {nodes[i]: float(X[i]) * SCALE_FACTOR for i in range(len(nodes))}

def synapse_weight_boost(current_weight: float, max_weight: float = 5.0) -> float:
    delta = 0.1 * (1.0 - current_weight / max_weight)
    return min(max_weight, max(0.0, current_weight + delta))

def relation_traverse_boost(current_strength: float) -> float:
    return min(MAX_IMPORTANCE, current_strength + RELATION_TRAVERSE_BOOST)

def relation_decay(strength: float, days_since_traversal: float) -> float:
    if days_since_traversal <= 0:
        return strength
    decayed = strength * math.exp(-RELATION_DECAY_LAMBDA * days_since_traversal)
    return max(MIN_IMPORTANCE, decayed)

def transitive_strength(
    strength_ab: float, strength_bc: float, depth: int,
) -> float:
    return strength_ab * strength_bc * (0.9 ** depth)

def information_density(text: str) -> float:
    """Shannon entropy ratio H/H_max. High = diverse information.

    Args:
        text: Input text to measure.

    Returns:
        Density ratio in [0, 1]. 0 = repetitive, 1 = maximally diverse.
    """
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    counts = Counter(words)
    n = len(words)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    h_max = math.log2(n)
    return h / h_max if h_max > 0 else 0.0
