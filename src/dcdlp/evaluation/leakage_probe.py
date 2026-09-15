from __future__ import annotations

import numpy as np


def run_posthoc_probe(train_x, train_y, test_x, test_y, task: str = "classification", seed: int = 0):
    if task == "classification":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        model = LogisticRegression(max_iter=1000, random_state=seed).fit(train_x, train_y)
        predicted = model.predict(test_x)
        return {"accuracy": float(accuracy_score(test_y, predicted)),
                "macro_f1": float(f1_score(test_y, predicted, average="macro"))}
    if task == "regression":
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, r2_score
        model = Ridge(alpha=1.0).fit(train_x, train_y)
        predicted = model.predict(test_x)
        return {"r2": float(r2_score(test_y, predicted)), "mae": float(mean_absolute_error(test_y, predicted))}
    raise ValueError("task must be classification or regression")

