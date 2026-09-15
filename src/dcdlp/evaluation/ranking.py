from __future__ import annotations

import numpy as np


def reciprocal_ranks(positive_scores, negative_scores) -> np.ndarray:
    positive = np.asarray(positive_scores, dtype=float).reshape(-1, 1)
    negative = np.asarray(negative_scores, dtype=float)
    if negative.ndim == 1:
        negative = np.broadcast_to(negative.reshape(1, -1), (len(positive), len(negative)))
    if negative.shape[0] != positive.shape[0]:
        raise ValueError("negative_scores must have one row per positive")
    optimistic = 1 + (negative > positive).sum(axis=1)
    pessimistic = 1 + (negative >= positive).sum(axis=1)
    ranks = 0.5 * (optimistic + pessimistic)
    return 1.0 / ranks


def ranking_metrics(positive_scores, negative_scores, ks=(10, 20, 50, 100)) -> dict[str, float]:
    rr = reciprocal_ranks(positive_scores, negative_scores)
    output = {"mrr": float(rr.mean())}
    ranks = 1.0 / rr
    output.update({f"hits{k}": float((ranks <= k).mean()) for k in ks})
    output["mean_positive_rank"] = float(ranks.mean())
    return output

