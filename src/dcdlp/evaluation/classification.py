from __future__ import annotations

import numpy as np


def classification_metrics(labels, scores) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return {"auc": float("nan"), "ap": float("nan")}
    return {"auc": float(roc_auc_score(labels, scores)), "ap": float(average_precision_score(labels, scores))}

