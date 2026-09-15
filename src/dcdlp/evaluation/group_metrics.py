from __future__ import annotations

import numpy as np

from .ranking import reciprocal_ranks


def group_ranking_metrics(positive_scores, negative_scores, groups) -> dict:
    rr = reciprocal_ranks(positive_scores, negative_scores)
    groups = np.asarray(groups)
    per_group = {}
    for group in ("HH", "HL", "LH", "LL"):
        mask = groups == group
        per_group[group] = {"mrr": float(rr[mask].mean()) if mask.any() else float("nan"), "count": int(mask.sum())}
    values = np.asarray([item["mrr"] for item in per_group.values() if np.isfinite(item["mrr"])])
    return {
        "groups": per_group,
        "macro_mrr": float(values.mean()),
        "worst_group_mrr": float(values.min()),
        "group_gap": float(values.max() - values.min()),
    }

