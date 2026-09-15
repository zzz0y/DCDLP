from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import spearmanr


def dose_response_metrics(
    doses: Iterable[float],
    target_changes: Iterable[float],
    non_target_changes: Iterable[float] | None = None,
) -> dict[str, float | int | str]:
    dose = np.asarray(list(doses), dtype=float)
    target = np.asarray(list(target_changes), dtype=float)
    finite = np.isfinite(dose) & np.isfinite(target)
    dose, target = dose[finite], target[finite]
    if len(dose) < 2 or len(np.unique(dose)) < 2:
        return {
            "status": "MISSING", "n": int(len(dose)), "spearman": np.nan,
            "monotonic_consistency": np.nan, "sign_correct_rate": np.nan,
            "non_target_dose_sensitivity": np.nan,
        }
    order = np.argsort(dose, kind="stable")
    ordered_dose, ordered_target = dose[order], target[order]
    dose_steps = np.diff(ordered_dose)
    target_steps = np.diff(ordered_target)
    comparable = dose_steps != 0
    monotonic = (
        float(np.mean(target_steps[comparable] * dose_steps[comparable] >= 0))
        if comparable.any() else np.nan
    )
    nonzero = dose != 0
    sign_rate = (
        float(np.mean(np.sign(target[nonzero]) == np.sign(dose[nonzero])))
        if nonzero.any() else np.nan
    )
    correlation = spearmanr(dose, target).statistic
    non_target_sensitivity = np.nan
    if non_target_changes is not None:
        other = np.asarray(list(non_target_changes), dtype=float)[finite]
        if np.isfinite(other).sum() >= 2:
            value = spearmanr(np.abs(dose), np.abs(other)).statistic
            non_target_sensitivity = float(value) if np.isfinite(value) else 0.0
    return {
        "status": "AVAILABLE",
        "n": int(len(dose)),
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        "monotonic_consistency": monotonic,
        "sign_correct_rate": sign_rate,
        "non_target_dose_sensitivity": non_target_sensitivity,
    }

