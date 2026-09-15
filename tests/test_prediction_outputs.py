import numpy as np
import torch

from dcdlp.data.synthetic import make_smoke_dataset
from dcdlp.evaluate import _intervention_evaluation
from dcdlp.models.dcdlp import DCDLP
from dcdlp.train import (
    TrainConfig,
    edge_index_from_graph,
    evaluate_split,
    fit_training_cn_residualizer,
)


def _model_and_data(seed=30):
    dataset = make_smoke_dataset(seed)
    config = TrainConfig(
        dataset="smoke", seed=seed, cn_feature_mode="residual"
    )
    regressor, _ = fit_training_cn_residualizer(dataset, config)
    model = DCDLP(
        dataset.features.shape[1],
        hidden_dim=16,
        branch_dim=8,
        dropout=0.0,
        cn_feature_mode="residual",
        cn_regressor=regressor,
        interaction_mode="audited",
    ).eval()
    x = torch.as_tensor(dataset.features)
    edges = edge_index_from_graph(dataset.train_graph(), torch.device("cpu"))
    return model, dataset, x, edges


def test_prediction_rows_contain_selective_audit_schema():
    model, dataset, x, edges = _model_and_data()
    _, rows = evaluate_split(
        model,
        dataset,
        dataset.test_pos,
        31,
        5,
        x,
        edges,
    )
    required = {
        "score_total",
        "score_degree",
        "score_cn",
        "score_residual",
        "score_interaction",
        "interaction_share",
        "cn_raw",
        "cn_expected",
        "cn_residual",
        "degree_u",
        "degree_v",
    }
    assert required <= set(rows[0])
    denominator = sum(
        abs(rows[0][f"score_{name}"])
        for name in ("degree", "cn", "residual", "interaction")
    )
    expected_share = abs(rows[0]["score_interaction"]) / (
        denominator + 1e-8
    )
    assert np.isclose(rows[0]["interaction_share"], expected_share)


def test_intervention_rows_include_before_after_delta_and_coverage():
    model, dataset, x, edges = _model_and_data(31)
    audit = _intervention_evaluation(
        model, dataset, x, edges, seed=32, max_pairs=3
    )
    assert "coverage_cn" in audit and "coverage_degree" in audit
    available = [
        row for row in audit["records"] if row["status"] == "AVAILABLE"
    ]
    assert available
    required = {
        "score_total_before",
        "score_total_after",
        "delta_score_total",
        "score_interaction_before",
        "score_interaction_after",
        "delta_score_interaction",
        "target_response",
        "cross_response",
        "routing_selectivity",
    }
    assert required <= set(available[0])

