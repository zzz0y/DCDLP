from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def degree_bin_boundaries(train_values: Iterable[float], bins: int = 10) -> np.ndarray:
    values = np.asarray(list(train_values), dtype=float)
    if len(values) == 0:
        raise ValueError("Training values are empty")
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    return np.unique(np.quantile(values, quantiles))


def apply_degree_bins(values: Iterable[float], boundaries: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(list(values), dtype=float), boundaries).astype(int)


def _classification_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    correlation = spearmanr(y_true, y_pred).statistic
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
    }


@dataclass
class ProbeResult:
    task: str
    selected_hyperparameter: float
    validation_metric: float
    test_metrics: dict[str, float]
    test_predictions: np.ndarray
    architecture: str
    hyperparameter_candidates: tuple[float, ...]
    seed: int


def classification_probe(
    train_x,
    train_y,
    valid_x,
    valid_y,
    test_x,
    test_y,
    *,
    seed: int = 0,
    candidates: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
) -> ProbeResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    best_c, best_metric = None, -np.inf
    for candidate in candidates:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=candidate, max_iter=2_000, random_state=seed,
                class_weight="balanced",
            ),
        )
        model.fit(train_x, train_y)
        metric = _classification_metrics(valid_y, model.predict(valid_x))["macro_f1"]
        if metric > best_metric + 1e-12:
            best_c, best_metric = candidate, metric
    combined_x = np.concatenate([np.asarray(train_x), np.asarray(valid_x)])
    combined_y = np.concatenate([np.asarray(train_y), np.asarray(valid_y)])
    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(best_c), max_iter=2_000, random_state=seed,
            class_weight="balanced",
        ),
    ).fit(combined_x, combined_y)
    predictions = final_model.predict(test_x)
    return ProbeResult(
        task="classification",
        selected_hyperparameter=float(best_c),
        validation_metric=float(best_metric),
        test_metrics=_classification_metrics(test_y, predictions),
        test_predictions=np.asarray(predictions),
        architecture="StandardScaler+LogisticRegression",
        hyperparameter_candidates=tuple(map(float, candidates)),
        seed=int(seed),
    )


def regression_probe(
    train_x,
    train_y,
    valid_x,
    valid_y,
    test_x,
    test_y,
    *,
    seed: int = 0,
    candidates: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> ProbeResult:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    best_alpha, best_mae = None, np.inf
    for candidate in candidates:
        model = make_pipeline(StandardScaler(), Ridge(alpha=candidate))
        model.fit(train_x, train_y)
        metric = _regression_metrics(valid_y, model.predict(valid_x))["mae"]
        if metric < best_mae - 1e-12:
            best_alpha, best_mae = candidate, metric
    combined_x = np.concatenate([np.asarray(train_x), np.asarray(valid_x)])
    combined_y = np.concatenate([np.asarray(train_y), np.asarray(valid_y)])
    final_model = make_pipeline(
        StandardScaler(), Ridge(alpha=float(best_alpha))
    ).fit(combined_x, combined_y)
    predictions = final_model.predict(test_x)
    return ProbeResult(
        task="regression",
        selected_hyperparameter=float(best_alpha),
        validation_metric=float(best_mae),
        test_metrics=_regression_metrics(test_y, predictions),
        test_predictions=np.asarray(predictions),
        architecture="StandardScaler+Ridge",
        hyperparameter_candidates=tuple(map(float, candidates)),
        seed=int(seed),
    )
