from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(values, seed: int = 0, samples: int = 10_000, confidence: float = 0.95):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        estimates[index] = rng.choice(values, size=len(values), replace=True).mean()
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))


def cliffs_delta(first, second) -> float:
    first, second = np.asarray(first), np.asarray(second)
    comparisons = np.sign(first[:, None] - second[None, :])
    return float(comparisons.mean())


def holm_bonferroni(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted

