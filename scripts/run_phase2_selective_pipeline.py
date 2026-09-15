"""Run the isolated Smoke or Cora selective-routing matrix."""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from dcdlp.data.loaders import load_dataset
from dcdlp.evaluate import evaluate_checkpoint
from dcdlp.train import TrainConfig, train_model
from dcdlp.utils import stable_hash, write_json
try:
    from scripts.aggregate_results import aggregate_result_root
    from scripts.run_posthoc_leakage_audit import run_posthoc_leakage_audit
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from aggregate_results import aggregate_result_root
    from run_posthoc_leakage_audit import run_posthoc_leakage_audit


MODEL_MATRIX = {
    "M0": {"cn_feature_mode": "raw", "interaction_mode": "audited"},
    "M1": {"cn_feature_mode": "residual", "interaction_mode": "audited"},
    "M2": {
        "cn_feature_mode": "raw_plus_residual",
        "interaction_mode": "audited",
    },
    "M3": {"cn_feature_mode": "residual", "interaction_mode": "disabled"},
    "M4": {
        "cn_feature_mode": "residual",
        "interaction_mode": "unrestricted",
    },
}


def _append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = [
        "timestamp", "model_label", "dataset", "seed", "status",
        "run_id", "result_file", "checkpoint", "evaluation_file",
        "probe_file", "log_file", "failure_code", "message",
    ]
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def _run_id(dataset_name: str, config: TrainConfig) -> str:
    return (
        f"{dataset_name}_{config.protocol_train}_seed{config.seed}_"
        f"{stable_hash(asdict(config))}"
    )


def run_pipeline(
    *,
    dataset_name: str,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    data_root: Path,
    output_root: Path,
    pretrain_epochs: int,
    routing_epochs: int,
    device: str,
    resume: bool = False,
    max_probe_pairs: int = 0,
) -> list[dict]:
    output_root = output_root.resolve()
    for directory in (
        "checkpoints", "raw", "predictions", "mechanism_audit",
        "probe", "aggregate", "logs",
    ):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.csv"
    statuses: list[dict] = []
    protocol_eval = "standard" if dataset_name == "smoke" else "heart"
    for model_label in models:
        if model_label not in MODEL_MATRIX:
            raise ValueError(f"Unknown model label {model_label}")
        for seed in seeds:
            mode = MODEL_MATRIX[model_label]
            config = TrainConfig(
                dataset=dataset_name,
                protocol_train="uniform",
                protocol_eval=protocol_eval,
                seed=seed,
                hidden_dim=16 if dataset_name == "smoke" else 128,
                branch_dim=8 if dataset_name == "smoke" else 64,
                num_layers=2,
                dropout=0.1 if dataset_name == "smoke" else 0.3,
                lr=1e-3,
                weight_decay=1e-4,
                batch_size=256 if dataset_name == "smoke" else 1024,
                pretrain_epochs=pretrain_epochs,
                disentangle_epochs=routing_epochs,
                negatives_per_positive_eval=10 if dataset_name == "smoke" else 20,
                intervention_ratio=0.05 if dataset_name == "smoke" else 0.25,
                # Fixed resource caps are shared by every compared model and
                # are frozen before the Cora runs; they are not selected from
                # test metrics.
                max_interventions_per_batch=8 if dataset_name == "smoke" else 16,
                max_interventions_per_epoch=16 if dataset_name == "smoke" else 64,
                lambda_inv=0.0,
                lambda_route=0.0,
                lambda_adv=0.0,
                lambda_orth=0.0,
                lambda_degrob=0.0,
                output_dir=str(output_root),
                prediction_subdir="predictions",
                device=device,
                ablation="A5",
                cn_feature_mode=mode["cn_feature_mode"],
                cn_input_schema="selective_v1",
                interaction_mode=mode["interaction_mode"],
                interaction_share_cap=0.20,
                interaction_delta_cap=0.20,
                lambda_interaction_share=0.01,
                lambda_interaction_delta=0.01,
                lambda_link=1.0,
                score_routing_enabled=True,
                lambda_score_route=0.03,
                lambda_target_response=0.03,
                routing_margin_cn=0.0,
                routing_margin_degree=0.0,
                target_floor_cn=0.05,
                target_floor_degree=0.05,
                interaction_route_weight=1.0,
                routing_finetune_encoder="frozen",
                encoder_lr_multiplier=0.1,
            )
            run_id = _run_id(dataset_name, config)
            result_path = output_root / "raw" / f"{run_id}.json"
            log_path = output_root / "logs" / f"{model_label}_{run_id}.log"
            row = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "model_label": model_label,
                "dataset": dataset_name,
                "seed": seed,
                "run_id": run_id,
                "result_file": str(result_path),
                "log_file": str(log_path),
            }
            try:
                if result_path.exists():
                    if not resume:
                        raise FileExistsError(
                            f"Refusing to overwrite existing run {result_path}"
                        )
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    dataset = load_dataset(
                        dataset_name, data_root, protocol_eval, seed
                    )
                    _, result = train_model(dataset, config)
                checkpoint = Path(result["checkpoint"])
                mechanism_dir = output_root / "mechanism_audit" / run_id
                probe_dir = output_root / "probe" / run_id
                evaluation_file = mechanism_dir / "evaluation.json"
                probe_file = probe_dir / "posthoc_probe.csv"
                if not evaluation_file.exists():
                    evaluate_checkpoint(
                        checkpoint,
                        data_root,
                        [protocol_eval, "intervention"],
                        mechanism_dir,
                    )
                if not probe_file.exists():
                    run_posthoc_leakage_audit(
                        checkpoint,
                        data_root,
                        probe_dir,
                        probe_seeds=(0, 1, 2),
                        max_pairs_per_split=max_probe_pairs,
                        pair_sample_seed=seed,
                    )
                result.update({
                    "pipeline_model": model_label,
                    "mechanism_predictions": str(
                        mechanism_dir / "intervention_predictions.csv"
                    ),
                    "probe_results": str(probe_file),
                    "evaluation_file": str(evaluation_file),
                })
                write_json(result_path, result)
                row.update({
                    "status": "COMPLETED",
                    "checkpoint": str(checkpoint),
                    "evaluation_file": str(evaluation_file),
                    "probe_file": str(probe_file),
                    "failure_code": "",
                    "message": "",
                })
                log_path.write_text(
                    json.dumps(row, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:
                row.update({
                    "status": "FAILED",
                    "failure_code": type(exc).__name__,
                    "message": str(exc),
                })
                failure_log = (
                    json.dumps(row, ensure_ascii=False, indent=2)
                    + "\n\n" + traceback.format_exc()
                )
                if log_path.exists():
                    log_path = log_path.with_name(
                        f"{log_path.stem}_{int(time.time())}{log_path.suffix}"
                    )
                    row["log_file"] = str(log_path)
                log_path.write_text(failure_log, encoding="utf-8")
            _append_manifest(manifest, row)
            statuses.append(row)
    if any(row["status"] != "COMPLETED" for row in statuses):
        return statuses
    aggregate_result_root(output_root)
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["smoke", "cora"], default="smoke")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--models", default="M0,M1,M3")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="results/phase2_selective")
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--routing-epochs", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-probe-pairs", type=int, default=0)
    args = parser.parse_args()
    statuses = run_pipeline(
        dataset_name=args.dataset,
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        models=tuple(value for value in args.models.split(",") if value),
        data_root=Path(args.data_root),
        output_root=Path(args.output_dir),
        pretrain_epochs=args.pretrain_epochs,
        routing_epochs=args.routing_epochs,
        device=args.device,
        resume=args.resume,
        max_probe_pairs=args.max_probe_pairs,
    )
    print(json.dumps(statuses, ensure_ascii=False, indent=2))
    if any(row["status"] != "COMPLETED" for row in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
