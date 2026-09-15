from pathlib import Path

from dcdlp.data.synthetic import make_smoke_dataset
from dcdlp.train import TrainConfig, train_model
from scripts.run_posthoc_leakage_audit import (
    PROBE_SPECS,
    run_posthoc_leakage_audit,
)


def test_independent_frozen_checkpoint_audit_runs_all_required_probes(tmp_path):
    dataset = make_smoke_dataset(41)
    _, train_result = train_model(
        dataset,
        TrainConfig(
            dataset="smoke",
            seed=41,
            hidden_dim=16,
            branch_dim=8,
            pretrain_epochs=0,
            disentangle_epochs=0,
            negatives_per_positive_eval=5,
            cn_feature_mode="residual",
            interaction_mode="audited",
            output_dir=str(tmp_path / "training"),
        ),
    )
    output_dir = tmp_path / "probe"
    result = run_posthoc_leakage_audit(
        Path(train_result["checkpoint"]),
        tmp_path / "data",
        output_dir,
        probe_seeds=(3, 4),
        max_pairs_per_split=32,
        pair_sample_seed=9,
    )
    assert {row["probe"] for row in result["rows"]} == set(PROBE_SPECS)
    assert len(result["rows"]) == 2 * len(PROBE_SPECS)
    assert result["metadata"]["model_frozen"] is True
    assert result["metadata"]["test_not_used_for_selection"] is True
    assert (output_dir / "posthoc_probe.csv").exists()
    assert (output_dir / "posthoc_probe_summary.csv").exists()
    assert (output_dir / "posthoc_probe_metadata.json").exists()

