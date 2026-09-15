import numpy as np
import pytest
import torch

import dcdlp.train as training
from dcdlp.data.pair_statistics import ConditionalCNRegressor, pair_features
from dcdlp.data.synthetic import make_smoke_dataset
from dcdlp.models.dcdlp import DCDLP
from dcdlp.train import TrainConfig, fit_training_cn_residualizer


def _regressor(dataset, seed=0):
    return ConditionalCNRegressor(seed, symmetric=True).fit(
        pair_features(dataset.train_graph(), dataset.train_pos),
        metadata={"data_source": "unit-test-train-only"},
    )


def test_selective_cn_modes_have_equal_parameter_budget_and_valid_shapes():
    dataset = make_smoke_dataset(10)
    regressor = _regressor(dataset, 10)
    x = torch.as_tensor(dataset.features)
    edges = torch.as_tensor(dataset.train_pos.T, dtype=torch.long)
    pairs = torch.as_tensor(dataset.test_pos[:3], dtype=torch.long)
    counts = []
    for mode in ("raw", "residual", "raw_plus_residual"):
        model = DCDLP(
            x.shape[1],
            hidden_dim=16,
            branch_dim=8,
            dropout=0.0,
            cn_feature_mode=mode,
            cn_regressor=regressor if mode != "raw" else None,
            cn_input_schema="selective_v1",
        ).eval()
        output = model(x, edges, pairs)
        assert output["z_cn"].shape == (3, 8)
        assert output["score_cn"].shape == (3,)
        counts.append(sum(value.numel() for value in model.cn_branch.parameters()))
    assert len(set(counts)) == 1


def test_residual_selective_input_excludes_raw_and_normalized_cn():
    dataset = make_smoke_dataset(11)
    regressor = _regressor(dataset, 11)
    x = torch.as_tensor(dataset.features)
    edges = torch.as_tensor(dataset.train_pos.T, dtype=torch.long)
    pairs = torch.as_tensor(dataset.test_pos[:3], dtype=torch.long)
    model = DCDLP(
        x.shape[1],
        hidden_dim=16,
        branch_dim=8,
        dropout=0.0,
        cn_feature_mode="residual",
        cn_regressor=regressor,
        cn_input_schema="selective_v1",
    ).eval()
    captured = []
    handle = model.cn_branch.encoder[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    output = model(x, edges, pairs)
    handle.remove()
    explicit = captured[0][:, -2:]
    torch.testing.assert_close(explicit[:, 0], output["cn_residual_feature"])
    torch.testing.assert_close(explicit[:, 1], torch.zeros_like(explicit[:, 1]))


def test_training_residualizer_metadata_is_train_only_and_hashed():
    dataset = make_smoke_dataset(12)
    config = TrainConfig(
        dataset="smoke", seed=12, cn_feature_mode="residual"
    )
    regressor, metadata = fit_training_cn_residualizer(dataset, config)
    assert regressor is not None
    assert metadata["uses_valid_or_test"] is False
    assert metadata["data_source"] == "train_positive+fixed_train_negative_epoch0"
    assert metadata["fit_sample_count"] == 2 * len(dataset.train_pos)
    assert len(metadata["data_hash"]) == 64
    assert metadata == regressor.fit_metadata


def test_training_residualizer_rejects_held_out_pair_in_calibration(monkeypatch):
    dataset = make_smoke_dataset(13)
    config = TrainConfig(
        dataset="smoke", seed=13, cn_feature_mode="residual"
    )
    monkeypatch.setattr(
        training,
        "_sample_train_negatives",
        lambda *_args, **_kwargs: dataset.valid_pos[:1],
    )
    with pytest.raises(RuntimeError, match="held-out valid/test"):
        fit_training_cn_residualizer(dataset, config)

