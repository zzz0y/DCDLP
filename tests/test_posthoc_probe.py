import numpy as np

from dcdlp.evaluation.posthoc_probe import (
    apply_degree_bins,
    classification_probe,
    degree_bin_boundaries,
    regression_probe,
)


def test_degree_bins_are_fitted_on_training_values_only():
    boundaries = degree_bin_boundaries([0, 1, 2, 3], bins=2)
    np.testing.assert_array_equal(boundaries, [1.5])
    np.testing.assert_array_equal(apply_degree_bins([-100, 1, 2, 100], boundaries), [0, 0, 1, 1])


def test_probes_select_hyperparameter_on_validation_and_score_test():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 4))
    classification_y = (x[:, 0] > 0).astype(int)
    regression_y = 2 * x[:, 0] - x[:, 1]
    split = (slice(0, 60), slice(60, 90), slice(90, 120))
    classified = classification_probe(
        x[split[0]], classification_y[split[0]],
        x[split[1]], classification_y[split[1]],
        x[split[2]], classification_y[split[2]],
        seed=9,
    )
    regressed = regression_probe(
        x[split[0]], regression_y[split[0]],
        x[split[1]], regression_y[split[1]],
        x[split[2]], regression_y[split[2]],
    )
    assert classified.test_metrics["accuracy"] > 0.8
    assert regressed.test_metrics["r2"] > 0.9
    assert len(classified.test_predictions) == 30
    assert len(regressed.test_predictions) == 30
