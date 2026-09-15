import argparse

from scripts.aggregate_results import flatten
from scripts.run_suite import build_train_command


def test_suite_config_forwards_phase2_mechanism_options():
    args = argparse.Namespace(
        pretrain_epochs=None,
        disentangle_epochs=None,
        hidden_dim=None,
        branch_dim=None,
        num_layers=None,
        dropout=None,
        backbone=None,
        lr=None,
        weight_decay=None,
        batch_size=None,
        intervention_ratio=None,
        max_interventions_per_batch=None,
        max_interventions_per_epoch=None,
        cn_feature_mode=None,
        interaction_mode=None,
    )
    command = build_train_command(
        "cora", "heart", "A5", 1, "uniform",
        {
            "pretrain_epochs": 20,
            "disentangle_epochs": 20,
            "hidden_dim": 128,
            "branch_dim": 64,
            "num_layers": 2,
            "dropout": 0.3,
            "backbone": "gcn",
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 1024,
            "intervention_ratio": 0.25,
            "cn_feature_mode": "raw_plus_residual",
            "interaction_mode": "audited",
        },
        args,
    )
    assert "--ablation" in command and command[command.index("--ablation") + 1] == "A5"
    assert "--cn-feature-mode" in command
    assert command[command.index("--cn-feature-mode") + 1] == "raw_plus_residual"
    assert "--interaction-mode" in command
    assert command[command.index("--interaction-mode") + 1] == "audited"
    assert command[command.index("--pretrain-epochs") + 1] == "20"
    assert command[command.index("--hidden-dim") + 1] == "128"


def test_cli_override_wins_over_suite_config():
    args = argparse.Namespace(
        pretrain_epochs=3,
        disentangle_epochs=None,
        hidden_dim=None,
        branch_dim=None,
        num_layers=None,
        dropout=None,
        backbone=None,
        lr=None,
        weight_decay=None,
        batch_size=None,
        intervention_ratio=None,
        max_interventions_per_batch=None,
        max_interventions_per_epoch=None,
        cn_feature_mode="raw",
        interaction_mode=None,
    )
    command = build_train_command(
        "cora", "heart", "A5", 0, "uniform",
        {"pretrain_epochs": 20, "cn_feature_mode": "raw_plus_residual"},
        args,
    )
    assert command[command.index("--pretrain-epochs") + 1] == "3"
    assert command[command.index("--cn-feature-mode") + 1] == "raw"


def test_legacy_results_are_separated_from_phase2_aggregates():
    row = flatten({"dataset": "cora", "metrics": {"mrr": 0.5}, "runtime": {}})
    assert row["cn_feature_mode"] == "legacy_unknown"
    assert row["interaction_mode"] == "legacy_unknown"
