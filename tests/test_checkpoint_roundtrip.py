from pathlib import Path

import pytest
import torch

from dcdlp.data.synthetic import make_smoke_dataset
from dcdlp.evaluate import evaluate_checkpoint
from dcdlp.train import TrainConfig, train_model


def test_checkpoint_restores_residualizer_and_mechanism_config(tmp_path):
    dataset = make_smoke_dataset(21)
    config = TrainConfig(
        dataset="smoke",
        seed=21,
        hidden_dim=16,
        branch_dim=8,
        batch_size=256,
        pretrain_epochs=1,
        disentangle_epochs=0,
        negatives_per_positive_eval=5,
        ablation="A5",
        cn_feature_mode="residual",
        cn_input_schema="selective_v1",
        interaction_mode="audited",
        interaction_share_cap=0.25,
        lambda_interaction_share=0.01,
        score_routing_enabled=True,
        lambda_score_route=0.02,
        lambda_target_response=0.03,
        routing_finetune_encoder="frozen",
        output_dir=str(tmp_path),
    )
    _, result = train_model(dataset, config)
    checkpoint = Path(result["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["cn_regressor"] is not None
    assert payload["cn_regressor"].fit_metadata["uses_valid_or_test"] is False
    assert payload["config"]["cn_feature_mode"] == "residual"
    assert payload["config"]["cn_input_schema"] == "selective_v1"
    assert payload["config"]["interaction_mode"] == "audited"
    assert payload["config"]["interaction_share_cap"] == 0.25
    assert payload["config"]["score_routing_enabled"] is True
    assert payload["config"]["lambda_score_route"] == 0.02
    assert payload["config"]["lambda_target_response"] == 0.03
    assert payload["config"]["routing_finetune_encoder"] == "frozen"

    # The smoke dataset is deterministic and needs no on-disk prepared data.
    evaluated = evaluate_checkpoint(checkpoint, tmp_path / "data", ["standard"])
    assert "mrr" in evaluated["standard"]
    with pytest.raises(RuntimeError, match="refusing to substitute Uniform"):
        evaluate_checkpoint(checkpoint, tmp_path / "data", ["heart"])
