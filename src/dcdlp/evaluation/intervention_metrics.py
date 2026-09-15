from __future__ import annotations

import numpy as np


def controlled_effect(plus_scores, minus_scores) -> float:
    return float(np.mean(np.asarray(plus_scores) - np.asarray(minus_scores)))


def sensitivity(original_scores, counterfactual_scores) -> float:
    return float(np.mean(np.abs(np.asarray(counterfactual_scores) - np.asarray(original_scores))))


def cross_sensitivity(original_representation, counterfactual_representation) -> float:
    difference = np.asarray(counterfactual_representation) - np.asarray(original_representation)
    return float(np.linalg.norm(difference, axis=-1).mean())


def routing_selectivity(target_delta, *other_deltas, epsilon: float = 1e-8) -> float:
    target = np.asarray(target_delta)
    denominator = target.copy()
    for value in other_deltas:
        denominator = denominator + np.asarray(value)
    return float(np.mean(target / (denominator + epsilon)))


def dose_response_monotonicity(doses, scores) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(np.asarray(doses), np.asarray(scores)).statistic)

