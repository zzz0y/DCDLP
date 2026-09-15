import numpy as np
import torch

from dcdlp.evaluation.classification import classification_metrics
from dcdlp.evaluation.group_metrics import group_ranking_metrics
from dcdlp.evaluation.intervention_metrics import controlled_effect, sensitivity
from dcdlp.evaluation.ranking import ranking_metrics
from dcdlp.models.losses import intervention_losses
from dcdlp.train import grouped_official_negatives


def test_ranking_metrics_on_hand_example():
    positives = np.array([0.9, 0.4])
    negatives = np.array([[0.1, 0.2], [0.8, 0.3]])
    metrics = ranking_metrics(positives, negatives, ks=(1, 2))
    assert metrics["mrr"] == 0.75
    assert metrics["hits1"] == 0.5
    assert metrics["hits2"] == 1.0


def test_classification_group_and_intervention_metrics():
    classification = classification_metrics([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert classification == {"auc": 1.0, "ap": 1.0}
    groups = group_ranking_metrics([0.9, 0.4], [[0.1], [0.8]], ["HH", "LL"])
    assert groups["worst_group_mrr"] == 0.5
    assert controlled_effect([3, 4], [1, 2]) == 2.0
    assert sensitivity([1, 1], [2, 0]) == 1.0


def test_intervention_loss_has_finite_gradient_at_zero_distance():
    representation = torch.zeros(2, 4, requires_grad=True)
    original = {
        "logit": torch.zeros(2, requires_grad=True),
        "z_degree": representation,
        "z_cn": representation,
        "z_residual": representation,
    }
    counterfactual = {
        "logit": torch.zeros(2, requires_grad=True),
        "z_degree": representation.clone(),
        "z_cn": representation.clone(),
        "z_residual": representation.clone(),
    }
    invariant, route, robust = intervention_losses(original, counterfactual, "degree")
    (invariant + route + robust).backward()
    assert torch.isfinite(representation.grad).all()


def test_official_candidates_are_used_without_uniform_fallback():
    positives = np.asarray([[0, 1], [2, 3]])
    official = np.asarray([
        [[0, 2], [0, 3]],
        [[1, 2], [1, 3]],
    ])
    grouped = grouped_official_negatives(
        positives, official, all_positive=positives
    )
    np.testing.assert_array_equal(grouped, official)


def test_official_candidates_missing_or_target_conflict_fail_loudly():
    positives = np.asarray([[0, 1]])
    with np.testing.assert_raises_regex(
        RuntimeError, "refusing to substitute Uniform"
    ):
        grouped_official_negatives(positives, None, positives)
    with np.testing.assert_raises_regex(
        RuntimeError, "corresponding target"
    ):
        grouped_official_negatives(
            positives,
            np.asarray([[[0, 1], [0, 2]]]),
            positives,
        )
