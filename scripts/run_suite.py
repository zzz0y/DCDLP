from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


UNSUPPORTED_ABLATIONS = {
    "A8": "Matching-pair cache and adapter are not implemented.",
    "A9": "Edit-count-matched random-rewire cache and adapter are not implemented.",
}

SELECTIVE_MODEL_OVERRIDES = {
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


# These options are intentionally read from the suite YAML as well as the
# command line.  The original runner only forwarded the smoke-suite knobs,
# which meant a new model mode could be present in the config but never reach
# the training process.
TRAIN_OPTION_SPECS = (
    ("pretrain_epochs", "--pretrain-epochs"),
    ("disentangle_epochs", "--disentangle-epochs"),
    ("hidden_dim", "--hidden-dim"),
    ("branch_dim", "--branch-dim"),
    ("num_layers", "--num-layers"),
    ("dropout", "--dropout"),
    ("backbone", "--backbone"),
    ("lr", "--lr"),
    ("weight_decay", "--weight-decay"),
    ("batch_size", "--batch-size"),
    ("intervention_ratio", "--intervention-ratio"),
    ("max_interventions_per_batch", "--max-interventions-per-batch"),
    ("max_interventions_per_epoch", "--max-interventions-per-epoch"),
    ("cn_feature_mode", "--cn-feature-mode"),
    ("cn_input_schema", "--cn-input-schema"),
    ("interaction_mode", "--interaction-mode"),
    ("interaction_share_cap", "--interaction-share-cap"),
    ("interaction_delta_cap", "--interaction-delta-cap"),
    ("lambda_interaction_share", "--lambda-interaction-share"),
    ("lambda_interaction_delta", "--lambda-interaction-delta"),
    ("lambda_link", "--lambda-link"),
    ("score_routing_enabled", "--score-routing-enabled"),
    ("lambda_score_route", "--lambda-score-route"),
    ("lambda_target_response", "--lambda-target-response"),
    ("routing_margin_cn", "--routing-margin-cn"),
    ("routing_margin_degree", "--routing-margin-degree"),
    ("target_floor_cn", "--target-floor-cn"),
    ("target_floor_degree", "--target-floor-degree"),
    ("interaction_route_weight", "--interaction-route-weight"),
    ("routing_finetune_encoder", "--routing-finetune-encoder"),
    ("encoder_lr_multiplier", "--encoder-lr-multiplier"),
    ("output_dir", "--output-dir"),
    ("prediction_subdir", "--prediction-subdir"),
)


def build_train_command(
    dataset: str,
    protocol_eval: str,
    model: str,
    seed: int,
    protocol_train: str,
    config: dict,
    args: argparse.Namespace,
) -> list[str]:
    """Build one reproducible train command from CLI and suite settings."""
    command = [
        sys.executable, "-m", "dcdlp.cli", "train",
        "--dataset", dataset,
        "--protocol", protocol_eval,
        "--seed", str(seed),
        "--protocol-train", protocol_train,
    ]
    config = {**config, **SELECTIVE_MODEL_OVERRIDES.get(model, {})}
    is_ablation = model.startswith("A") and model[1:].isdigit()
    if is_ablation:
        command += ["--ablation", model]
    elif model in SELECTIVE_MODEL_OVERRIDES:
        command += ["--ablation", "A5"]

    for key, option in TRAIN_OPTION_SPECS:
        value = getattr(args, key, None)
        if value is None:
            value = config.get(key)
        if value is not None:
            command += [option, str(value)]
    return command


def append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "timestamp", "suite", "dataset", "model", "seed", "protocol_train",
            "status", "return_code", "result_file", "log_file", "message",
        ])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def execute(
    root: Path, command: list[str], log_file: Path,
    environment_overrides: dict[str, str] | None = None,
) -> tuple[int, str]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment.update(environment_overrides or {})
        process = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                                 text=True, env=environment)
    text = log_file.read_text(encoding="utf-8", errors="replace")
    result_file = ""
    for line in reversed(text.splitlines()):
        if '"result_file"' in line:
            try:
                # Pretty JSON may place the field on its own line.
                result_file = line.split(":", 1)[1].strip().strip('",')
            except IndexError:
                pass
            break
    return process.returncode, result_file


def terminal_jobs(path: Path) -> set[tuple[str, str, str, str, str]]:
    if not path.exists():
        return set()
    output = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") in {"COMPLETED", "NOT_RUN"}:
                output.add((row.get("suite", ""), row.get("dataset", ""), row.get("model", ""),
                            row.get("seed", ""), row.get("protocol_train", "")))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--datasets", default="",
                        help="Optional comma-separated dataset subset.")
    parser.add_argument("--seeds", default="",
                        help="Optional comma-separated seed subset.")
    parser.add_argument("--protocols", default="",
                        help="Optional comma-separated training-protocol subset.")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("DCDLP_JOBS", "1")),
                        help="Concurrent experiment processes for small and medium datasets.")
    parser.add_argument("--large-jobs", type=int, default=int(os.environ.get("DCDLP_LARGE_JOBS", "1")),
                        help="Concurrent experiment processes for OGB datasets.")
    parser.add_argument("--threads-per-job", type=int,
                        default=int(os.environ.get("DCDLP_THREADS_PER_JOB", "8")),
                        help="CPU math-library threads assigned to each experiment process.")
    parser.add_argument("--large-batch-size", type=int,
                        default=int(os.environ.get("DCDLP_LARGE_BATCH_SIZE", "0")),
                        help="Batch-size override used only for OGB datasets; zero preserves the model default.")
    parser.add_argument("--large-max-interventions-per-batch", type=int,
                        default=int(os.environ.get("DCDLP_LARGE_MAX_INTERVENTIONS_PER_BATCH", "0")),
                        help="Per-batch intervention cap used only for OGB; zero is unlimited.")
    parser.add_argument("--large-max-interventions-per-epoch", type=int,
                        default=int(os.environ.get("DCDLP_LARGE_MAX_INTERVENTIONS_PER_EPOCH", "0")),
                        help="Per-epoch intervention cap used only for OGB; zero is unlimited.")
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--disentangle-epochs", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--branch-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--backbone", choices=["gcn", "sage"])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--intervention-ratio", type=float)
    parser.add_argument(
        "--cn-feature-mode",
        choices=["raw", "residual", "raw_plus_residual"],
    )
    parser.add_argument(
        "--cn-input-schema", choices=["legacy", "selective_v1"]
    )
    parser.add_argument(
        "--interaction-mode",
        choices=["unrestricted", "audited", "disabled"],
    )
    parser.add_argument("--interaction-share-cap", type=float)
    parser.add_argument("--interaction-delta-cap", type=float)
    parser.add_argument("--lambda-interaction-share", type=float)
    parser.add_argument("--lambda-interaction-delta", type=float)
    parser.add_argument("--lambda-link", type=float)
    parser.add_argument("--score-routing-enabled", type=str)
    parser.add_argument("--lambda-score-route", type=float)
    parser.add_argument("--lambda-target-response", type=float)
    parser.add_argument("--routing-margin-cn", type=float)
    parser.add_argument("--routing-margin-degree", type=float)
    parser.add_argument("--target-floor-cn", type=float)
    parser.add_argument("--target-floor-degree", type=float)
    parser.add_argument("--interaction-route-weight", type=float)
    parser.add_argument(
        "--routing-finetune-encoder", choices=["frozen", "low_lr"]
    )
    parser.add_argument("--encoder-lr-multiplier", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--prediction-subdir", choices=["raw", "predictions"]
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip jobs already recorded as COMPLETED or NOT_RUN.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count jobs without executing them or changing the manifest.")
    args = parser.parse_args()
    if args.jobs < 1 or args.large_jobs < 1 or args.threads_per_job < 1:
        parser.error("--jobs, --large-jobs, and --threads-per-job must be positive")
    if (args.large_batch_size < 0 or args.large_max_interventions_per_batch < 0 or
            args.large_max_interventions_per_epoch < 0):
        parser.error("large-dataset resource overrides must be non-negative")
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "suite" / f"{args.suite}.yaml").read_text(encoding="utf-8"))
    result_root = root / config.get("output_dir", "results")
    manifest = result_root / "manifest.csv"
    previous = terminal_jobs(manifest) if args.resume else set()
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    models = ([item.strip() for item in args.models.split(",") if item.strip()]
              if args.models else config.get("models", ["dcdlp"]))
    if args.suite == "smoke":
        datasets, seeds, protocols = ["smoke"], [0], ["uniform"]
    else:
        datasets = config.get("datasets", [])
        protocols = config.get("protocol_train", ["uniform"])
        seeds = None
    if args.datasets:
        requested_datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
        unknown = set(requested_datasets) - set(datasets)
        if unknown:
            parser.error(f"datasets are not in suite {args.suite}: {sorted(unknown)}")
        datasets = requested_datasets
    if args.protocols:
        requested_protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]
        unknown = set(requested_protocols) - set(protocols)
        if unknown:
            parser.error(f"protocols are not in suite {args.suite}: {sorted(unknown)}")
        protocols = requested_protocols
    requested_seeds = ({int(item.strip()) for item in args.seeds.split(",") if item.strip()}
                       if args.seeds else None)
    failed = 0
    counts = {"runnable": 0, "not_run": 0, "skipped": 0}
    runnable_index = 0
    jobs_by_dataset: dict[str, list[dict]] = {dataset: [] for dataset in datasets}
    for dataset in datasets:
        local_seeds = seeds if seeds is not None else (config.get("ogb_seeds") if dataset.startswith("ogbl-") else config.get("small_seeds"))
        if requested_seeds is not None:
            local_seeds = [seed for seed in local_seeds if seed in requested_seeds]
            if not local_seeds:
                parser.error(f"none of the requested seeds apply to dataset {dataset}")
        protocol_eval = ("ogb" if dataset.startswith("ogbl-") else
                         "standard" if dataset.startswith("synthetic-") or dataset == "smoke" else "heart")
        for model, seed, protocol_train in itertools.product(models, local_seeds, protocols):
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            log_file = result_root / "logs" / f"{args.suite}_{dataset}_{model}_{protocol_train}_seed{seed}.log"
            is_ablation = model.startswith("A") and model[1:].isdigit()
            identity = (args.suite, dataset, model, str(seed), protocol_train)
            if identity in previous:
                counts["skipped"] += 1
                if not args.dry_run:
                    print(f"[SKIPPED] {dataset} {model} {protocol_train} seed={seed}")
                continue
            unsupported_message = UNSUPPORTED_ABLATIONS.get(model)
            if (model != "dcdlp" and not is_ablation
                    and model not in SELECTIVE_MODEL_OVERRIDES):
                unsupported_message = (
                    "Official baseline is intentionally isolated from the DCDLP implementation. "
                    "Run scripts/download_data.py --third-party and the matching official adapter; no synthetic score was emitted."
                )
            if unsupported_message:
                counts["not_run"] += 1
                if args.dry_run:
                    continue
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_file.write_text(unsupported_message + "\n", encoding="utf-8")
                append_manifest(manifest, {"timestamp": timestamp, "suite": args.suite, "dataset": dataset,
                    "model": model, "seed": seed, "protocol_train": protocol_train, "status": "NOT_RUN",
                    "return_code": "", "result_file": "", "log_file": log_file, "message": unsupported_message})
                continue
            counts["runnable"] += 1
            if args.dry_run:
                continue
            command = build_train_command(
                dataset, protocol_eval, model, seed, protocol_train, config, args
            )
            if dataset.startswith("ogbl-"):
                if args.large_batch_size:
                    command += ["--batch-size", str(args.large_batch_size)]
                if args.large_max_interventions_per_batch:
                    command += ["--max-interventions-per-batch",
                                str(args.large_max_interventions_per_batch)]
                if args.large_max_interventions_per_epoch:
                    command += ["--max-interventions-per-epoch",
                                str(args.large_max_interventions_per_epoch)]
            environment = {
                "OMP_NUM_THREADS": str(args.threads_per_job),
                "MKL_NUM_THREADS": str(args.threads_per_job),
                "OPENBLAS_NUM_THREADS": str(args.threads_per_job),
                "NUMEXPR_NUM_THREADS": str(args.threads_per_job),
            }
            if gpu_ids:
                gpu_id = gpu_ids[runnable_index % len(gpu_ids)]
                environment["CUDA_VISIBLE_DEVICES"] = gpu_id
                command += ["--device", "cuda"]
            runnable_index += 1
            jobs_by_dataset[dataset].append({
                "timestamp": timestamp, "dataset": dataset, "model": model, "seed": seed,
                "protocol_train": protocol_train, "command": command, "log_file": log_file,
                "environment": environment,
            })
    if args.dry_run:
        print(json.dumps({"suite": args.suite, **counts}, ensure_ascii=False))
        return

    def run_job(job: dict) -> tuple[dict, int, str]:
        return_code, result_file = execute(
            root, job["command"], job["log_file"], job["environment"]
        )
        return job, return_code, result_file

    for dataset in datasets:
        pending_jobs = jobs_by_dataset[dataset]
        if not pending_jobs:
            continue
        requested_workers = args.large_jobs if dataset.startswith("ogbl-") else args.jobs
        worker_count = min(requested_workers, len(pending_jobs))
        print(
            f"[SCHEDULER] dataset={dataset} jobs={len(pending_jobs)} "
            f"concurrency={worker_count} threads_per_job={args.threads_per_job}"
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(run_job, job) for job in pending_jobs]
            for future in as_completed(futures):
                job, return_code, result_file = future.result()
                status = "COMPLETED" if return_code == 0 else "FAILED"
                failed += int(return_code != 0)
                append_manifest(manifest, {
                    "timestamp": job["timestamp"], "suite": args.suite,
                    "dataset": job["dataset"], "model": job["model"], "seed": job["seed"],
                    "protocol_train": job["protocol_train"], "status": status,
                    "return_code": return_code, "result_file": result_file,
                    "log_file": job["log_file"], "message": "",
                })
                print(
                    f"[{status}] {job['dataset']} {job['model']} "
                    f"{job['protocol_train']} seed={job['seed']}"
                )
        if failed and not args.continue_on_error:
            raise SystemExit(1)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
