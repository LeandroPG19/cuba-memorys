import math

ETA: float = 0.05

DECAY_LAMBDA_BASE: float = math.log(2) / 30.0

SPREAD_DECAY: float = 0.3

MIN_IMPORTANCE: float = 0.01
MAX_IMPORTANCE: float = 1.0

SM2_EF_MIN: float = 1.3
SM2_EF_MAX: float = 2.5
SM2_EF_BASE: float = 2.5

RELATION_TRAVERSE_BOOST: float = 0.05
RELATION_DECAY_LAMBDA: float = math.log(2) / 60.0

def oja_positive(importance: float) -> float:
    delta = ETA * (1.0 - importance * importance)
    return min(MAX_IMPORTANCE, max(MIN_IMPORTANCE, importance + delta))

def oja_negative(importance: float) -> float:
    delta = ETA * (1.0 + importance * importance)
    return min(MAX_IMPORTANCE, max(MIN_IMPORTANCE, importance - delta))

def sm2_easiness_factor(access_count: int) -> float:
    if access_count <= 0:
        quality = 1
    elif access_count <= 2:
        quality = 2
    elif access_count <= 5:
        quality = 3
    elif access_count <= 10:
        quality = 4
    else:
        quality = 5

    ef = SM2_EF_BASE + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    return max(SM2_EF_MIN, min(SM2_EF_MAX, ef))

def sm2_decay(importance: float, days_elapsed: float, access_count: int) -> float:
    if days_elapsed <= 0:
        return importance

    ef = sm2_easiness_factor(access_count)
    adaptive_lambda = DECAY_LAMBDA_BASE / ef
    decayed = importance * math.exp(-adaptive_lambda * days_elapsed)
    return max(MIN_IMPORTANCE, decayed)

def ebbinghaus_decay(importance: float, days_elapsed: float) -> float:
    if days_elapsed <= 0:
        return importance
    decayed = importance * math.exp(-DECAY_LAMBDA_BASE * days_elapsed)
    return max(MIN_IMPORTANCE, decayed)

def spreading_activation_boost(current_importance: float) -> float:
    boost = 0.02 * SPREAD_DECAY
    return min(MAX_IMPORTANCE, current_importance + boost)

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
