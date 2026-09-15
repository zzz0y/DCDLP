"""Independent frozen-checkpoint leakage probes for selective DCDLP."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from dcdlp.data.loaders import load_dataset
from dcdlp.data.pair_statistics import pair_features
from dcdlp.evaluate import load_checkpoint_model
from dcdlp.evaluation.posthoc_probe import regression_probe
from dcdlp.train import (
    TrainConfig,
    edge_index_from_graph,
    fit_training_cn_residualizer,
    score_pairs,
)
from dcdlp.utils import array_hash, write_json


PROBE_SPECS = {
    "z_cn_to_degree": ("z_cn", "degree", "cross_leakage"),
    "z_residual_to_degree": (
        "z_residual", "degree", "cross_leakage"
    ),
    "z_degree_to_cn_residual": (
        "z_degree", "cn_residual", "cross_leakage"
    ),
    "z_degree_to_degree": ("z_degree", "degree", "target_retention"),
    "z_cn_to_cn_residual": (
        "z_cn", "cn_residual", "target_retention"
    ),
}


def _fixed_subset(pairs: np.ndarray, limit: int, seed: int) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    if limit <= 0 or len(pairs) <= limit:
        return pairs
    rng = np.random.default_rng(seed)
    return pairs[np.sort(rng.choice(len(pairs), size=limit, replace=False))]


def _config_from_payload(payload: dict) -> TrainConfig:
    known = TrainConfig.__dataclass_fields__
    return TrainConfig(**{
        key: value for key, value in payload["config"].items()
        if key in known
    })


def run_posthoc_leakage_audit(
    checkpoint: Path,
    data_root: Path,
    output_dir: Path,
    *,
    probe_seeds: tuple[int, ...] = (0, 1, 2),
    max_pairs_per_split: int = 0,
    pair_sample_seed: int = 0,
) -> dict:
    model, payload = load_checkpoint_model(checkpoint, "cpu")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = payload["config"]
    dataset = load_dataset(
        config["dataset"],
        data_root,
        config.get("protocol_eval", "standard"),
        config["seed"],
    )
    train_config = _config_from_payload(payload)
    residualizer = payload.get("cn_regressor")
    residualizer_metadata = payload.get("cn_feature_metadata", {})
    if residualizer is None:
        residualizer, residualizer_metadata = fit_training_cn_residualizer(
            dataset, train_config
        )
    if residualizer is None:
        raise RuntimeError("A train-only CN residualizer is required for probes")

    import torch

    x = torch.as_tensor(dataset.features, dtype=torch.float32)
    edges = edge_index_from_graph(dataset.train_graph(), torch.device("cpu"))
    split_pairs: dict[str, np.ndarray] = {}
    representations: dict[str, dict[str, np.ndarray]] = {}
    targets: dict[str, dict[str, np.ndarray]] = {}
    graph = dataset.train_graph()
    for offset, split in enumerate(("train", "valid", "test")):
        pairs = _fixed_subset(
            getattr(dataset, f"{split}_pos"),
            max_pairs_per_split,
            pair_sample_seed + offset,
        )
        split_pairs[split] = pairs
        scored = score_pairs(model, x, edges, pairs, batch_size=1024)
        stats = pair_features(graph, pairs, dataset.features)
        _, residual = residualizer.residual(stats)
        representations[split] = {
            key: np.asarray(scored[key])
            for key in ("z_degree", "z_cn", "z_residual")
        }
        targets[split] = {
            "degree": np.asarray(stats["degree_score"], dtype=float),
            "cn_residual": np.asarray(residual, dtype=float),
        }

    rows: list[dict] = []
    prediction_rows: list[dict] = []
    for probe_seed in probe_seeds:
        for probe_name, (representation, target, role) in PROBE_SPECS.items():
            result = regression_probe(
                representations["train"][representation],
                targets["train"][target],
                representations["valid"][representation],
                targets["valid"][target],
                representations["test"][representation],
                targets["test"][target],
                seed=probe_seed,
            )
            base = {
                "probe": probe_name,
                "role": role,
                "representation": representation,
                "target": target,
                "probe_seed": probe_seed,
                "architecture": result.architecture,
                "hyperparameter_candidates": ";".join(
                    map(str, result.hyperparameter_candidates)
                ),
                "selected_hyperparameter": result.selected_hyperparameter,
                "validation_metric": result.validation_metric,
                "train_count": len(split_pairs["train"]),
                "valid_count": len(split_pairs["valid"]),
                "test_count": len(split_pairs["test"]),
            }
            rows.append({**base, **result.test_metrics})
            for index, prediction in enumerate(result.test_predictions):
                prediction_rows.append({
                    "probe": probe_name,
                    "probe_seed": probe_seed,
                    "pair_id": f"test-{index}",
                    "target": float(targets["test"][target][index]),
                    "prediction": float(prediction),
                })

    metric_names = ("r2", "mae", "spearman")
    summary_rows: list[dict] = []
    for probe_name in PROBE_SPECS:
        local = [row for row in rows if row["probe"] == probe_name]
        for metric in metric_names:
            values = np.asarray([row[metric] for row in local], dtype=float)
            summary_rows.append({
                "probe": probe_name,
                "role": local[0]["role"],
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "count": len(values),
                "seeds": ";".join(map(str, probe_seeds)),
            })

    output_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(name: str, values: list[dict]) -> None:
        if not values:
            return
        with (output_dir / name).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    write_csv("posthoc_probe.csv", rows)
    write_csv("posthoc_probe_summary.csv", summary_rows)
    write_csv("posthoc_probe_predictions.csv", prediction_rows)
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "dataset": dataset.name,
        "model_seed": int(config["seed"]),
        "probe_seeds": list(probe_seeds),
        "pair_sample_seed": int(pair_sample_seed),
        "pair_split_hashes": {
            split: array_hash(pairs) for split, pairs in split_pairs.items()
        },
        "residualizer": residualizer_metadata,
        "model_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "test_not_used_for_selection": True,
        "files": {
            "per_seed": str(output_dir / "posthoc_probe.csv"),
            "summary": str(output_dir / "posthoc_probe_summary.csv"),
            "predictions": str(output_dir / "posthoc_probe_predictions.csv"),
        },
    }
    write_json(output_dir / "posthoc_probe_metadata.json", metadata)
    return {"rows": rows, "summary": summary_rows, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-seeds", default="0,1,2")
    parser.add_argument("--max-pairs-per-split", type=int, default=0)
    parser.add_argument("--pair-sample-seed", type=int, default=0)
    args = parser.parse_args()
    result = run_posthoc_leakage_audit(
        Path(args.checkpoint),
        Path(args.data_root),
        Path(args.output_dir),
        probe_seeds=tuple(
            int(value) for value in args.probe_seeds.split(",") if value
        ),
        max_pairs_per_split=args.max_pairs_per_split,
        pair_sample_seed=args.pair_sample_seed,
    )
    print(result["metadata"]["files"]["summary"])


if __name__ == "__main__":
    main()

