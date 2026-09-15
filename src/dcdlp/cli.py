from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data.loaders import load_dataset, save_dataset
from .data.negative_sampling import degree_corrected_sampling, uniform_negative_sampling
from .data.pair_statistics import ConditionalCNRegressor, assign_quadrants, pair_features
from .evaluate import evaluate_checkpoint
from .interventions.cache import build_intervention_cache
from .train import TrainConfig, train_model
from .utils import json_safe, project_root


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _print_json(value, indent=None) -> None:
    print(json.dumps(json_safe(value), indent=indent, ensure_ascii=False, allow_nan=False, default=str))


def _data_root(args) -> Path:
    return Path(args.data_root) if getattr(args, "data_root", None) else project_root() / "data"


def prepare_data(args) -> None:
    dataset = load_dataset(args.dataset, _data_root(args), args.protocol, args.seed)
    output = _data_root(args) / "processed" / f"{args.dataset}_{args.protocol}_seed{args.seed}.npz"
    save_dataset(dataset, output)
    _print_json({"dataset": dataset.name, "nodes": dataset.num_nodes,
                 "train_edges": len(dataset.train_pos), "output": str(output)})


def build_negatives(args) -> None:
    dataset = load_dataset(args.dataset, _data_root(args), args.protocol, args.seed)
    positives = getattr(dataset, f"{args.split}_pos")
    count = args.count or len(positives) * args.per_positive
    if args.method == "degree_corrected":
        negatives = degree_corrected_sampling(dataset.train_graph(), dataset.all_positive, count, args.seed)
    else:
        negatives = uniform_negative_sampling(dataset.num_nodes, dataset.all_positive, count, args.seed)
    output = _data_root(args) / "negatives" / f"{args.dataset}_{args.method}_{args.split}_seed{args.seed}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, negatives=negatives)
    _print_json({"count": len(negatives), "output": str(output)})


def build_pair_stats(args) -> None:
    import pandas as pd

    dataset = load_dataset(args.dataset, _data_root(args), args.protocol, args.seed)
    graph = dataset.train_graph()
    train_neg = uniform_negative_sampling(dataset.num_nodes, dataset.all_positive, len(dataset.train_pos), args.seed)
    train_pairs = np.vstack([dataset.train_pos, train_neg])
    train_stats = pair_features(graph, train_pairs, dataset.features)
    regressor = ConditionalCNRegressor(args.seed).fit(train_stats)
    degree_threshold = float(np.median(train_stats["degree_score"]))
    frames = []
    for split in ("train", "valid", "test"):
        positives = getattr(dataset, f"{split}_pos")
        stats = pair_features(graph, positives, dataset.features)
        expected, residual = regressor.residual(stats)
        frame = pd.DataFrame(stats)
        frame.insert(0, "v", positives[:, 1])
        frame.insert(0, "u", positives[:, 0])
        frame.insert(0, "pair_id", [f"{split}-{i}" for i in range(len(positives))])
        frame["label"] = 1
        frame["cn_expected"] = expected
        frame["cn_residual"] = residual
        frame["quadrant"] = assign_quadrants(stats["degree_score"], residual, degree_threshold)
        frame["split"] = split
        frames.append(frame)
    output = project_root() / "cache" / "pair_stats" / f"{args.dataset}_{args.protocol}_seed{args.seed}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    _print_json({"output": str(output), "rows": sum(map(len, frames))})


def build_interventions(args) -> None:
    dataset = load_dataset(args.dataset, _data_root(args), args.protocol, args.seed)
    pairs = getattr(dataset, f"{args.split}_pos")[:args.max_pairs]
    forbidden = {tuple(item) for item in np.vstack([dataset.valid_pos, dataset.test_pos]).tolist()}
    output = project_root() / "cache" / "interventions" / f"{args.dataset}_{args.split}_seed{args.seed}.parquet"
    frame = build_intervention_cache(dataset.train_graph(), pairs, args.types.split(","), output, args.seed, forbidden)
    _print_json({"output": str(output), "valid": int(frame.get("valid", []).sum()),
                 "total": len(frame)})


def train(args) -> None:
    values = vars(args).copy()
    overrides = values.pop("overrides", [])
    values["protocol_eval"] = values.get("protocol", "standard")
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = {"protocol": "protocol_eval"}.get(key, key)
        values[key] = value
    fields = TrainConfig.__dataclass_fields__
    config_values = {}
    for key in fields:
        if key not in values or values[key] is None:
            continue
        target_type = fields[key].type
        value = values[key]
        if target_type in (int, "int"):
            value = int(value)
        elif target_type in (float, "float"):
            value = float(value)
        elif target_type in (bool, "bool"):
            value = _parse_bool(value)
        config_values[key] = value
    config = TrainConfig(**config_values)
    dataset = load_dataset(config.dataset, _data_root(args), config.protocol_eval, config.seed)
    _, result = train_model(dataset, config)
    _print_json(result, indent=2)


def evaluate(args) -> None:
    result = evaluate_checkpoint(
        Path(args.checkpoint),
        _data_root(args),
        args.protocols.split(","),
        Path(args.output_dir) if args.output_dir else None,
    )
    _print_json(result, indent=2)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dcdlp", description="DCDLP experiment CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", default="smoke")
    common.add_argument("--protocol", default="standard")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--data-root")

    prepare = sub.add_parser("prepare-data", parents=[common])
    prepare.set_defaults(func=prepare_data)
    negative = sub.add_parser("build-negatives", parents=[common])
    negative.add_argument("--method", choices=["uniform", "degree_corrected"], default="degree_corrected")
    negative.add_argument("--split", choices=["train", "valid", "test"], default="test")
    negative.add_argument("--count", type=int)
    negative.add_argument("--per-positive", type=int, default=50)
    negative.set_defaults(func=build_negatives)
    stats = sub.add_parser("build-pair-stats", parents=[common])
    stats.set_defaults(func=build_pair_stats)
    intervention = sub.add_parser("build-interventions", parents=[common])
    intervention.add_argument("--split", choices=["train", "valid", "test"], default="train")
    intervention.add_argument("--types", default="cn_plus,cn_minus,degree_u_plus,degree_u_minus")
    intervention.add_argument("--max-pairs", type=int, default=100000)
    intervention.set_defaults(func=build_interventions)

    train_parser = sub.add_parser("train", parents=[common])
    train_parser.add_argument("overrides", nargs="*")
    train_parser.add_argument("--protocol-train", choices=["uniform", "degree_corrected", "mix"])
    train_parser.add_argument("--pretrain-epochs", type=int)
    train_parser.add_argument("--disentangle-epochs", type=int)
    train_parser.add_argument("--hidden-dim", type=int)
    train_parser.add_argument("--branch-dim", type=int)
    train_parser.add_argument("--num-layers", type=int)
    train_parser.add_argument("--dropout", type=float)
    train_parser.add_argument("--backbone", choices=["gcn", "sage"])
    train_parser.add_argument("--lr", type=float)
    train_parser.add_argument("--weight-decay", type=float)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--intervention-ratio", type=float)
    train_parser.add_argument("--max-interventions-per-batch", type=int)
    train_parser.add_argument("--max-interventions-per-epoch", type=int)
    train_parser.add_argument("--device")
    train_parser.add_argument("--ablation")
    train_parser.add_argument(
        "--cn-feature-mode",
        choices=["raw", "residual", "raw_plus_residual"],
    )
    train_parser.add_argument(
        "--cn-input-schema", choices=["legacy", "selective_v1"]
    )
    train_parser.add_argument(
        "--interaction-mode",
        choices=["unrestricted", "audited", "disabled"],
    )
    train_parser.add_argument("--interaction-share-cap", type=float)
    train_parser.add_argument("--interaction-delta-cap", type=float)
    train_parser.add_argument("--lambda-interaction-share", type=float)
    train_parser.add_argument("--lambda-interaction-delta", type=float)
    train_parser.add_argument("--lambda-link", type=float)
    train_parser.add_argument("--score-routing-enabled", type=_parse_bool)
    train_parser.add_argument("--lambda-score-route", type=float)
    train_parser.add_argument("--lambda-target-response", type=float)
    train_parser.add_argument("--routing-margin-cn", type=float)
    train_parser.add_argument("--routing-margin-degree", type=float)
    train_parser.add_argument("--target-floor-cn", type=float)
    train_parser.add_argument("--target-floor-degree", type=float)
    train_parser.add_argument("--interaction-route-weight", type=float)
    train_parser.add_argument(
        "--routing-finetune-encoder", choices=["frozen", "low_lr"]
    )
    train_parser.add_argument("--encoder-lr-multiplier", type=float)
    train_parser.add_argument("--output-dir")
    train_parser.add_argument(
        "--prediction-subdir", choices=["raw", "predictions"]
    )
    train_parser.set_defaults(func=train)
    evaluation = sub.add_parser("evaluate", parents=[common])
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--protocols", default="standard,heart,degree_corrected,intervention")
    evaluation.add_argument("--output-dir")
    evaluation.set_defaults(func=evaluate)
    return parser


def main(argv=None) -> None:
    args = make_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
