import numpy as np
import pytest
import torch

from dcdlp.data.pair_statistics import ConditionalCNRegressor, pair_features
from dcdlp.data.synthetic import make_smoke_dataset
from dcdlp.models.dcdlp import DCDLP


def test_branch_shapes_and_undirected_symmetry():
    dataset = make_smoke_dataset(0)
    edges = torch.as_tensor(dataset.train_pos.T, dtype=torch.long)
    x = torch.as_tensor(dataset.features)
    pairs = torch.as_tensor(dataset.test_pos[:3], dtype=torch.long)
    model = DCDLP(x.shape[1], hidden_dim=16, branch_dim=8, dropout=0.0).eval()
    forward = model(x, edges, pairs)
    reverse = model(x, edges, pairs.flip(1))
    assert forward["logit"].shape == (3,)
    assert forward["z_degree"].shape == (3, 8)
    assert forward["z_cn"].shape == (3, 8)
    assert forward["z_residual"].shape == (3, 8)
    torch.testing.assert_close(forward["logit"], reverse["logit"], atol=1e-6, rtol=0)


def test_raw_plus_residual_features_are_explicit_and_symmetric():
    dataset = make_smoke_dataset(3)
    graph = dataset.train_graph()
    calibration_pairs = dataset.train_pos
    regressor = ConditionalCNRegressor(3, symmetric=True).fit(
        pair_features(graph, calibration_pairs)
    )
    edges = torch.as_tensor(dataset.train_pos.T, dtype=torch.long)
    x = torch.as_tensor(dataset.features)
    pairs = torch.as_tensor(dataset.test_pos[:3], dtype=torch.long)
    model = DCDLP(
        x.shape[1],
        hidden_dim=16,
        branch_dim=8,
        dropout=0.0,
        cn_feature_mode="raw_plus_residual",
        cn_regressor=regressor,
        interaction_mode="audited",
    ).eval()
    forward = model(x, edges, pairs)
    reverse = model(x, edges, pairs.flip(1))
    for key in (
        "cn_raw",
        "cn_log_raw",
        "cn_expected",
        "cn_residual_feature",
        "cn_normalized",
    ):
        assert forward[key].shape == (3,)
        assert torch.isfinite(forward[key]).all()
    torch.testing.assert_close(
        forward["logit"], reverse["logit"], atol=1e-6, rtol=0
    )


def test_residual_mode_requires_training_split_regressor():
    dataset = make_smoke_dataset(4)
    edges = torch.as_tensor(dataset.train_pos.T, dtype=torch.long)
    x = torch.as_tensor(dataset.features)
    pairs = torch.as_tensor(dataset.test_pos[:1], dtype=torch.long)
    model = DCDLP(
        x.shape[1],
        hidden_dim=16,
        branch_dim=8,
        dropout=0.0,
        cn_feature_mode="residual",
    ).eval()
    with pytest.raises(RuntimeError, match="requires a fitted"):
        model(x, edges, pairs)


def test_interaction_modes_preserve_audit_and_disable_exactly():
    dataset = make_smoke_dataset(5)
    edges = torch.as_tensor(dataset.train_pos.T, dtype=torch.long)
    x = torch.as_tensor(dataset.features)
    pairs = torch.as_tensor(dataset.test_pos[:3], dtype=torch.long)
    unrestricted = DCDLP(
        x.shape[1], hidden_dim=16, branch_dim=8, dropout=0.0
    ).eval()
    audited = DCDLP(
        x.shape[1],
        hidden_dim=16,
        branch_dim=8,
        dropout=0.0,
        interaction_mode="audited",
    ).eval()
    disabled = DCDLP(
        x.shape[1],
        hidden_dim=16,
        branch_dim=8,
        dropout=0.0,
        interaction_mode="disabled",
    ).eval()
    audited.load_state_dict(unrestricted.state_dict())
    disabled.load_state_dict(unrestricted.state_dict())
    base = unrestricted(x, edges, pairs)
    audit = audited(x, edges, pairs)
    no_interaction = disabled(x, edges, pairs)
    torch.testing.assert_close(base["logit"], audit["logit"])
    torch.testing.assert_close(
        base["score_interaction"], audit["score_interaction"]
    )
    torch.testing.assert_close(
        no_interaction["score_interaction"],
        torch.zeros_like(no_interaction["score_interaction"]),
    )
    torch.testing.assert_close(
        no_interaction["logit"],
        base["logit"] - base["score_interaction"],
        atol=1e-6,
        rtol=0,
    )
