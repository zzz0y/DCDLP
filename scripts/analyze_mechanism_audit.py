from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch

from dcdlp.data.loaders import GraphDataset, load_dataset
from dcdlp.data.pair_statistics import (
    ConditionalCNRegressor,
    assign_quadrants,
    pair_features,
)
from dcdlp.evaluation.dose_response import dose_response_metrics
from dcdlp.evaluation.posthoc_probe import (
    apply_degree_bins,
    classification_probe,
    degree_bin_boundaries,
    regression_probe,
)
from dcdlp.evaluation.routing_audit import (
    branch_response,
    paired_effect,
    random_rewire_like,
    summarize_values,
)
from dcdlp.evaluation.statistics import holm_bonferroni
from dcdlp.interventions import InterventionError, intervene_cn, intervene_degree
from dcdlp.interventions.validator import (
    EditLog,
    common_neighbor_set,
    edge,
    validate_intervention,
)
from dcdlp.models.dcdlp import DCDLP
from dcdlp.train import ablation_profile, edge_index_from_graph


MODEL_A5 = "A5"
MODEL_A14 = "A14-w010"
EXPECTED_W010 = {
    "lambda_inv": 0.003,
    "lambda_route": 0.003,
    "lambda_degrob": 0.001,
}
PERFORMANCE_PARITY_TOLERANCE = 0.01
PAIR_KEYS = [
    "dataset", "seed", "protocol_train", "protocol_eval",
    "pair_index", "u", "v", "quadrant", "intervention_kind",
    "requested_magnitude", "control",
]


@dataclass
class RunArtifact:
    model_label: str
    dataset: str
    seed: int
    protocol_train: str
    protocol_eval: str
    config_hash: str
    result_json: Path
    checkpoint: Path
    predictions: Path | None
    config: dict[str, Any]
    metrics: dict[str, Any]
    runtime: dict[str, Any]

    @property
    def key(self) -> tuple[str, int, str, str]:
        return self.dataset, self.seed, self.protocol_train, self.protocol_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen-checkpoint A5/A14-w010 mechanism audit"
    )
    parser.add_argument("--models", default="A5,A14-w010")
    parser.add_argument("--use-existing-checkpoints", action="store_true")
    parser.add_argument("--output-dir", default="results/mechanism_audit")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-dir", default="cache/interventions")
    parser.add_argument("--datasets", default="cora,citeseer")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--protocol-train", default="uniform")
    parser.add_argument("--protocol-eval", default="heart")
    parser.add_argument("--max-intervention-pairs", type=int, default=24)
    parser.add_argument("--max-probe-pairs", type=int, default=512)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    return value


def resolve_artifact_path(
    raw: str | None,
    result_path: Path,
    results_root: Path,
    category: str,
) -> Path | None:
    if not raw:
        return None
    supplied = Path(str(raw))
    basename = supplied.name
    candidates = [
        supplied,
        result_path.parent / basename,
        result_path.parent.parent / category / basename,
        results_root / category / basename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = list(results_root.rglob(basename))
    return matches[0].resolve() if len(matches) == 1 else None


def checkpoint_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def classify_model(config: dict[str, Any], result_path: Path) -> str | None:
    if config.get("ablation") == "A5":
        return MODEL_A5
    if config.get("ablation") != "A14":
        return None
    exact = all(
        math.isclose(float(config.get(key, math.inf)), value, rel_tol=0, abs_tol=1e-12)
        for key, value in EXPECTED_W010.items()
    )
    # Directory naming is supporting evidence only; exact checkpoint config
    # remains mandatory so default A14 can never be silently substituted.
    if exact and "w010" in str(result_path).lower():
        return MODEL_A14
    if exact:
        return MODEL_A14
    return None


def discover_runs(results_root: Path, output_dir: Path) -> tuple[list[RunArtifact], list[dict[str, Any]]]:
    artifacts: list[RunArtifact] = []
    inventory: list[dict[str, Any]] = []
    for result_path in sorted(results_root.rglob("*.json")):
        if output_dir.resolve() in result_path.resolve().parents:
            continue
        try:
            record = json_load(result_path)
        except (OSError, json.JSONDecodeError) as exc:
            inventory.append({
                "status": "INVALID", "result_json": str(result_path), "reason": str(exc),
            })
            continue
        if not {"dataset", "seed", "ablation", "checkpoint"} <= set(record):
            continue
        checkpoint = resolve_artifact_path(
            record.get("checkpoint"), result_path, results_root, "checkpoints"
        )
        if checkpoint is None:
            inventory.append({
                "status": "MISSING", "result_json": str(result_path),
                "reason": "checkpoint path could not be resolved",
            })
            continue
        try:
            payload = checkpoint_payload(checkpoint)
            config = dict(payload.get("config", {}))
        except Exception as exc:
            inventory.append({
                "status": "INVALID", "result_json": str(result_path),
                "checkpoint": str(checkpoint), "reason": f"checkpoint load failed: {exc}",
            })
            continue
        label = classify_model(config, result_path)
        inventory.append({
            "status": "AVAILABLE", "model_label": label or "OUT_OF_SCOPE",
            "dataset": record.get("dataset"), "seed": record.get("seed"),
            "protocol_train": record.get("protocol_train"),
            "protocol_eval": record.get("protocol_eval"),
            "ablation": record.get("ablation"),
            "config_hash": record.get("config_hash"),
            "result_json": str(result_path.resolve()),
            "checkpoint": str(checkpoint),
        })
        if label is None:
            continue
        predictions = resolve_artifact_path(
            record.get("predictions"), result_path, results_root, "raw"
        )
        artifacts.append(RunArtifact(
            model_label=label,
            dataset=str(record["dataset"]),
            seed=int(record["seed"]),
            protocol_train=str(record.get("protocol_train", config.get("protocol_train", ""))),
            protocol_eval=str(record.get("protocol_eval", config.get("protocol_eval", ""))),
            config_hash=str(record.get("config_hash", "")),
            result_json=result_path.resolve(),
            checkpoint=checkpoint,
            predictions=predictions,
            config=config,
            metrics=dict(record.get("metrics", {})),
            runtime=dict(record.get("runtime", {})),
        ))
    return artifacts, inventory


def expected_keys(args: argparse.Namespace, artifacts: list[RunArtifact]) -> list[tuple[str, int, str, str]]:
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    keys = {
        (dataset, seed, args.protocol_train, args.protocol_eval)
        for dataset in datasets for seed in seeds
    }
    keys.update(item.key for item in artifacts)
    return sorted(keys)


def pair_runs(
    args: argparse.Namespace,
    artifacts: list[RunArtifact],
) -> tuple[pd.DataFrame, list[tuple[RunArtifact, RunArtifact]]]:
    rows: list[dict[str, Any]] = []
    available: list[tuple[RunArtifact, RunArtifact]] = []
    for key in expected_keys(args, artifacts):
        dataset, seed, protocol_train, protocol_eval = key
        found = {
            label: [
                item for item in artifacts
                if item.key == key and item.model_label == label
            ]
            for label in (MODEL_A5, MODEL_A14)
        }
        status = "AVAILABLE"
        reason = ""
        if any(len(found[label]) == 0 for label in found):
            status = "MISSING"
            reason = ", ".join(
                f"{label} missing" for label in found if len(found[label]) == 0
            )
        elif any(len(found[label]) > 1 for label in found):
            status = "MISSING"
            reason = ", ".join(
                f"{label} ambiguous ({len(found[label])} records)"
                for label in found if len(found[label]) > 1
            )
        a5 = found[MODEL_A5][0] if len(found[MODEL_A5]) == 1 else None
        a14 = found[MODEL_A14][0] if len(found[MODEL_A14]) == 1 else None
        prediction_status = (
            "AVAILABLE" if a5 and a14 and a5.predictions and a14.predictions
            else "MISSING"
        )
        row = {
            "dataset": dataset, "seed": seed, "protocol_train": protocol_train,
            "protocol_eval": protocol_eval, "status": status, "reason": reason,
            "prediction_pair_status": prediction_status,
            "a5_config_hash": a5.config_hash if a5 else "MISSING",
            "a14_w010_config_hash": a14.config_hash if a14 else "MISSING",
            "a5_checkpoint": str(a5.checkpoint) if a5 else "MISSING",
            "a14_w010_checkpoint": str(a14.checkpoint) if a14 else "MISSING",
            "a5_predictions": str(a5.predictions) if a5 and a5.predictions else "MISSING",
            "a14_w010_predictions": (
                str(a14.predictions) if a14 and a14.predictions else "MISSING"
            ),
            "a5_mrr": a5.metrics.get("mrr", np.nan) if a5 else np.nan,
            "a14_w010_mrr": a14.metrics.get("mrr", np.nan) if a14 else np.nan,
        }
        rows.append(row)
        if status == "AVAILABLE" and a5 and a14:
            available.append((a5, a14))
    return pd.DataFrame(rows), available


def load_model(artifact: RunArtifact, device: torch.device) -> tuple[DCDLP, dict[str, Any]]:
    payload = checkpoint_payload(artifact.checkpoint)
    config = dict(payload["config"])
    profile = ablation_profile(config["ablation"])
    if profile["model"] != "dcdlp":
        raise ValueError("Mechanism audit only supports DCDLP branch checkpoints")
    model = DCDLP(
        payload["input_dim"], config["hidden_dim"], config["branch_dim"],
        config["num_layers"], config["dropout"], config["backbone"],
        use_interaction=profile["interaction"],
        active_branches=profile["active"], decoder_mode=profile["decoder"],
        cn_feature_mode=config.get("cn_feature_mode", "raw"),
        cn_regressor=payload.get("cn_regressor"),
        interaction_mode=config.get("interaction_mode", "unrestricted"),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config


def output_numpy(output: dict[str, torch.Tensor], index: int = 0) -> dict[str, np.ndarray]:
    return {
        key: value[index:index + 1].detach().cpu().numpy()
        for key, value in output.items()
    }


@torch.inference_mode()
def score_one(
    model: DCDLP,
    x: torch.Tensor,
    graph: nx.Graph,
    pair: tuple[int, int],
    device: torch.device,
) -> dict[str, np.ndarray]:
    edges = edge_index_from_graph(graph, device)
    pairs = torch.as_tensor([pair], dtype=torch.long, device=device)
    return output_numpy(model(x, edges, pairs, remove_target_edges=False))


def vector_json(output: dict[str, np.ndarray], key: str) -> str:
    return json.dumps(np.asarray(output[key]).reshape(-1).astype(float).tolist())


def read_cache_files(cache_root: Path, dataset: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not cache_root.exists():
        return rows
    names = [
        path for path in cache_root.rglob("*")
        if path.is_file() and dataset.lower() in path.name.lower()
        and f"seed{seed}" in path.name.lower()
    ]
    for path in names:
        try:
            if path.suffix == ".parquet":
                rows.extend(pd.read_parquet(path).to_dict("records"))
            elif path.suffix == ".jsonl":
                rows.extend(
                    json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            elif path.suffix == ".csv":
                rows.extend(pd.read_csv(path).to_dict("records"))
        except Exception:
            continue
    return rows


def parse_edges(value: Any) -> list[tuple[int, int]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return [edge(*item) for item in value]


def cached_counterfactual(
    graph: nx.Graph,
    pair: tuple[int, int],
    kind: str,
    magnitude: int,
    cache_rows: list[dict[str, Any]],
    forbidden: set[tuple[int, int]],
) -> tuple[nx.Graph, EditLog] | None:
    u, v = pair
    desired_prefix = "cn_" if kind == "cn" else "degree_u_"
    desired_suffix = "plus" if magnitude > 0 else "minus"
    for row in cache_rows:
        if not bool(row.get("valid", False)):
            continue
        if edge(int(row.get("u", -1)), int(row.get("v", -1))) != edge(u, v):
            continue
        intervention_type = str(row.get("intervention_type", ""))
        requested = int(float(row.get("requested_delta", 0)))
        if (
            intervention_type.startswith(desired_prefix)
            and intervention_type.endswith(desired_suffix)
            and requested == magnitude
        ):
            removed, added = parse_edges(row.get("removed_edges")), parse_edges(row.get("added_edges"))
            candidate = graph.copy()
            candidate.remove_edges_from(removed)
            candidate.add_edges_from(added)
            log = EditLog(
                str(row.get("pair_id", "")), u, v, intervention_type, magnitude,
                removed, added, True, "", int(row.get("seed", 0)),
            )
            validate_intervention(graph, candidate, log, forbidden)
            return candidate, log
    return None


def constrained_counterfactual(
    graph: nx.Graph,
    pair: tuple[int, int],
    kind: str,
    magnitude: int,
    *,
    seed: int,
    forbidden: set[tuple[int, int]],
    cache_rows: list[dict[str, Any]],
) -> tuple[nx.Graph, EditLog, str]:
    cached = cached_counterfactual(
        graph, pair, kind, magnitude, cache_rows, forbidden
    )
    if cached is not None:
        return cached[0], cached[1], "EXISTING_CACHE"
    u, v = pair
    if kind == "degree":
        cf_graph, log = intervene_degree(
            graph, u, v, "u", magnitude, seed, forbidden
        )
    else:
        cf_graph, log = intervene_cn(
            graph, u, v, magnitude, seed, forbidden
        )
    return cf_graph, log, "GENERATED_EVALUATION"


def select_rows(values: np.ndarray, limit: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if limit <= 0 or len(values) <= limit:
        return values.copy()
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(values), size=limit, replace=False))
    return values[indices]


def pair_quadrants(dataset: GraphDataset) -> tuple[np.ndarray, ConditionalCNRegressor, float]:
    graph = dataset.train_graph()
    train_stats = pair_features(graph, dataset.train_pos, dataset.features)
    regressor = ConditionalCNRegressor(0).fit(train_stats)
    threshold = float(np.median(train_stats["degree_score"]))
    test_stats = pair_features(graph, dataset.test_pos, dataset.features)
    _, residual = regressor.residual(test_stats)
    return assign_quadrants(test_stats["degree_score"], residual, threshold), regressor, threshold


def audit_interventions(
    args: argparse.Namespace,
    a5: RunArtifact,
    a14: RunArtifact,
    device: torch.device,
    cache_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = load_dataset(a5.dataset, Path(args.data_root), a5.protocol_eval, a5.seed)
    graph = dataset.train_graph()
    forbidden = {edge(*item) for item in np.vstack([dataset.valid_pos, dataset.test_pos])}
    quadrants, _, _ = pair_quadrants(dataset)
    model_a5, _ = load_model(a5, device)
    model_a14, _ = load_model(a14, device)
    models = {MODEL_A5: model_a5, MODEL_A14: model_a14}
    x = torch.as_tensor(dataset.features, dtype=torch.float32, device=device)
    selected_indices = np.arange(min(args.max_intervention_pairs, len(dataset.test_pos)))
    cache_rows = read_cache_files(cache_root, a5.dataset, a5.seed)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    generation_seconds = 0.0
    for pair_index in selected_indices:
        pair = tuple(map(int, dataset.test_pos[pair_index]))
        originals = {
            label: score_one(model, x, graph, pair, device)
            for label, model in models.items()
        }
        for kind in ("degree", "cn"):
            for magnitude in (-2, -1, 1, 2):
                intervention_seed = stable_seed(
                    args.seed, a5.dataset, a5.seed, pair_index, kind, magnitude
                )
                generation_started = time.perf_counter()
                try:
                    cf_graph, log, source = constrained_counterfactual(
                        graph, pair, kind, magnitude, seed=intervention_seed,
                        forbidden=forbidden, cache_rows=cache_rows,
                    )
                    generation_seconds += time.perf_counter() - generation_started
                except (InterventionError, RuntimeError, ValueError) as exc:
                    generation_seconds += time.perf_counter() - generation_started
                    failures.append({
                        "dataset": a5.dataset, "seed": a5.seed,
                        "protocol_train": a5.protocol_train,
                        "protocol_eval": a5.protocol_eval,
                        "pair_index": int(pair_index), "u": pair[0], "v": pair[1],
                        "intervention_kind": kind,
                        "requested_magnitude": magnitude,
                        "status": "MISSING",
                        "reason": str(exc),
                    })
                    continue
                controls: dict[str, tuple[nx.Graph, str]] = {
                    "constrained": (cf_graph, source),
                    "matching_only": (graph, "MATCHED_NO_GRAPH_EDIT"),
                }
                try:
                    random_control = random_rewire_like(
                        graph, pair[0], pair[1], len(log.removed_edges),
                        seed=stable_seed(intervention_seed, "random"),
                        forbidden_edges=forbidden,
                    )
                    controls["random_rewire"] = (
                        random_control.graph, "GENERATED_RANDOM_CONTROL"
                    )
                except RuntimeError as exc:
                    failures.append({
                        "dataset": a5.dataset, "seed": a5.seed,
                        "protocol_train": a5.protocol_train,
                        "protocol_eval": a5.protocol_eval,
                        "pair_index": int(pair_index), "u": pair[0], "v": pair[1],
                        "intervention_kind": kind,
                        "requested_magnitude": magnitude,
                        "status": "MISSING", "reason": str(exc),
                        "control": "random_rewire",
                    })
                actual_magnitude = (
                    cf_graph.degree(pair[0]) - graph.degree(pair[0])
                    if kind == "degree"
                    else len(common_neighbor_set(cf_graph, *pair))
                    - len(common_neighbor_set(graph, *pair))
                )
                for control, (control_graph, provenance) in controls.items():
                    for label, model in models.items():
                        counterfactual = score_one(
                            model, x, control_graph, pair, device
                        )
                        response = branch_response(
                            originals[label], counterfactual, kind
                        )
                        row = {
                            "status": "AVAILABLE",
                            "dataset": a5.dataset, "seed": a5.seed,
                            "protocol_train": a5.protocol_train,
                            "protocol_eval": a5.protocol_eval,
                            "model": label,
                            "config_hash": (
                                a5.config_hash if label == MODEL_A5
                                else a14.config_hash
                            ),
                            "checkpoint": str(
                                a5.checkpoint if label == MODEL_A5
                                else a14.checkpoint
                            ),
                            "pair_index": int(pair_index), "u": pair[0], "v": pair[1],
                            "quadrant": str(quadrants[pair_index]),
                            "intervention_kind": kind,
                            "requested_magnitude": magnitude,
                            "actual_magnitude": (
                                actual_magnitude if control == "constrained" else np.nan
                            ),
                            "control": control, "provenance": provenance,
                            "edit_count": (
                                len(log.removed_edges) if control != "matching_only" else 0
                            ),
                        }
                        row.update(response)
                        for score_name, output_name in (
                            ("total", "logit"), ("degree", "score_degree"),
                            ("cn", "score_cn"), ("residual", "score_residual"),
                            ("interaction", "score_interaction"),
                        ):
                            row[f"score_{score_name}_original"] = float(
                                np.asarray(originals[label][output_name]).reshape(-1)[0]
                            )
                            row[f"score_{score_name}_counterfactual"] = float(
                                np.asarray(counterfactual[output_name]).reshape(-1)[0]
                            )
                        for branch in ("degree", "cn", "residual"):
                            row[f"z_{branch}_original"] = vector_json(
                                originals[label], f"z_{branch}"
                            )
                            row[f"z_{branch}_counterfactual"] = vector_json(
                                counterfactual, f"z_{branch}"
                            )
                        rows.append(row)
    elapsed = time.perf_counter() - started
    for artifact in (a5, a14):
        failures.append({
            "status": "TIMING", "dataset": artifact.dataset, "seed": artifact.seed,
            "model": artifact.model_label, "evaluation_seconds": elapsed / 2.0,
            "intervention_generation_seconds": generation_seconds,
        })
    return rows, failures


@torch.inference_mode()
def extract_representations(
    model: DCDLP,
    dataset: GraphDataset,
    pairs: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    x = torch.as_tensor(dataset.features, dtype=torch.float32, device=device)
    base_edges = edge_index_from_graph(dataset.train_graph(), device)
    output: dict[str, list[np.ndarray]] = {
        "z_degree": [], "z_cn": [], "z_residual": [],
    }
    for start in range(0, len(pairs), batch_size):
        batch = torch.as_tensor(
            pairs[start:start + batch_size], dtype=torch.long, device=device
        )
        scored = model(x, base_edges, batch, remove_target_edges=True)
        for key in output:
            output[key].append(scored[key].detach().cpu().numpy())
    return {
        key: np.concatenate(value, axis=0) if value else np.empty((0, 0))
        for key, value in output.items()
    }


def probe_targets(
    dataset: GraphDataset,
    split_pairs: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    graph = dataset.train_graph()
    stats = {
        split: pair_features(graph, pairs, dataset.features)
        for split, pairs in split_pairs.items()
    }
    regressor = ConditionalCNRegressor(0).fit(stats["train"])
    degree_boundaries = degree_bin_boundaries(stats["train"]["degree_score"])
    output: dict[str, dict[str, np.ndarray]] = {}
    for split, values in stats.items():
        _, residual = regressor.residual(values)
        output[split] = {
            "degree_bin": apply_degree_bins(values["degree_score"], degree_boundaries),
            "conditional_cn_residual": residual,
            "raw_cn": np.asarray(values["cn"], dtype=float),
            "pair_degree": np.asarray(values["degree_score"], dtype=float),
        }
    return output


def audit_probes(
    args: argparse.Namespace,
    a5: RunArtifact,
    a14: RunArtifact,
    device: torch.device,
) -> list[dict[str, Any]]:
    dataset = load_dataset(a5.dataset, Path(args.data_root), a5.protocol_eval, a5.seed)
    split_pairs = {
        "train": select_rows(
            dataset.train_pos, args.max_probe_pairs,
            stable_seed(args.seed, a5.dataset, a5.seed, "probe-train"),
        ),
        "valid": select_rows(
            dataset.valid_pos, args.max_probe_pairs,
            stable_seed(args.seed, a5.dataset, a5.seed, "probe-valid"),
        ),
        "test": select_rows(
            dataset.test_pos, args.max_probe_pairs,
            stable_seed(args.seed, a5.dataset, a5.seed, "probe-test"),
        ),
    }
    targets = probe_targets(dataset, split_pairs)
    tasks = [
        ("z_cn", "degree_bin", "classification", "cross_leakage"),
        ("z_residual", "degree_bin", "classification", "cross_leakage"),
        ("z_degree", "conditional_cn_residual", "regression", "cross_leakage"),
        ("z_residual", "conditional_cn_residual", "regression", "cross_leakage"),
        ("z_degree", "raw_cn", "regression", "cross_leakage"),
        ("z_cn", "pair_degree", "regression", "cross_leakage"),
        ("z_degree", "degree_bin", "classification", "target_retention"),
        ("z_cn", "conditional_cn_residual", "regression", "target_retention"),
    ]
    rows: list[dict[str, Any]] = []
    for artifact in (a5, a14):
        model, _ = load_model(artifact, device)
        representations = {
            split: extract_representations(
                model, dataset, pairs, device, args.probe_batch_size
            )
            for split, pairs in split_pairs.items()
        }
        for representation, target, task, role in tasks:
            base = {
                "status": "AVAILABLE", "dataset": artifact.dataset,
                "seed": artifact.seed, "protocol_train": artifact.protocol_train,
                "protocol_eval": artifact.protocol_eval,
                "model": artifact.model_label, "config_hash": artifact.config_hash,
                "checkpoint": str(artifact.checkpoint),
                "representation": representation, "target": target,
                "task": task, "role": role,
                "train_n": len(split_pairs["train"]),
                "valid_n": len(split_pairs["valid"]),
                "test_n": len(split_pairs["test"]),
            }
            try:
                values = [
                    representations[split][representation]
                    for split in ("train", "valid", "test")
                ]
                labels = [
                    targets[split][target]
                    for split in ("train", "valid", "test")
                ]
                if task == "classification":
                    result = classification_probe(
                        *sum(zip(values, labels), ()), seed=args.seed
                    )
                else:
                    result = regression_probe(*sum(zip(values, labels), ()))
                base.update({
                    "selected_hyperparameter": result.selected_hyperparameter,
                    "validation_metric": result.validation_metric,
                    "test_targets_json": json.dumps(
                        np.asarray(labels[-1]).astype(float).tolist()
                    ),
                    "test_predictions_json": json.dumps(
                        np.asarray(result.test_predictions).astype(float).tolist()
                    ),
                    **result.test_metrics,
                })
            except Exception as exc:
                base.update({"status": "MISSING", "reason": str(exc)})
            rows.append(base)
    return rows


def add_probe_comparisons(
    probes: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if probes.empty or not {
        "dataset", "seed", "representation", "target", "task", "role", "model",
        "status", "test_targets_json", "test_predictions_json",
    } <= set(probes.columns):
        return probes
    comparison_rows: list[dict[str, Any]] = []
    groups = [
        "dataset", "seed", "protocol_train", "protocol_eval",
        "representation", "target", "task", "role",
    ]
    available = probes[probes["status"] == "AVAILABLE"]
    for group, local in available.groupby(groups, dropna=False):
        indexed = {row["model"]: row for _, row in local.iterrows()}
        if not {MODEL_A5, MODEL_A14} <= set(indexed):
            comparison_rows.append({
                **dict(zip(groups, group)), "model": "PAIRED_A14_MINUS_A5",
                "status": "MISSING", "reason": "strict probe pair unavailable",
            })
            continue
        first, second = indexed[MODEL_A5], indexed[MODEL_A14]
        y_first = np.asarray(json.loads(first["test_targets_json"]), dtype=float)
        y_second = np.asarray(json.loads(second["test_targets_json"]), dtype=float)
        if y_first.shape != y_second.shape or not np.array_equal(y_first, y_second):
            comparison_rows.append({
                **dict(zip(groups, group)), "model": "PAIRED_A14_MINUS_A5",
                "status": "MISSING", "reason": "probe test targets differ",
            })
            continue
        pred_a5 = np.asarray(json.loads(first["test_predictions_json"]), dtype=float)
        pred_a14 = np.asarray(json.loads(second["test_predictions_json"]), dtype=float)
        if group[6] == "classification":
            contribution_a5 = (pred_a5 == y_first).astype(float)
            contribution_a14 = (pred_a14 == y_first).astype(float)
            primary_metric = "accuracy_contribution"
            desired_for_cross_leakage = "LOWER"
        else:
            contribution_a5 = np.abs(pred_a5 - y_first)
            contribution_a14 = np.abs(pred_a14 - y_first)
            primary_metric = "absolute_error"
            desired_for_cross_leakage = "HIGHER"
        effect = paired_effect(
            contribution_a14, contribution_a5,
            seed=stable_seed(args.seed, *group, "probe"),
            bootstrap_samples=args.bootstrap_samples,
        )
        comparison_rows.append({
            **dict(zip(groups, group)), "model": "PAIRED_A14_MINUS_A5",
            **effect, "primary_metric": primary_metric,
            "desired_for_cross_leakage": desired_for_cross_leakage,
        })
    if comparison_rows:
        return pd.concat([probes, pd.DataFrame(comparison_rows)], ignore_index=True)
    return probes


def summarize_long(
    frame: pd.DataFrame,
    metrics: list[str],
    group_columns: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame([{"status": "MISSING", "reason": "no available rows"}])
    for group, local in frame.groupby(group_columns, dropna=False):
        group = group if isinstance(group, tuple) else (group,)
        base = dict(zip(group_columns, group))
        for metric in metrics:
            summary = summarize_values(
                local[metric], seed=stable_seed(args.seed, *group, metric),
                bootstrap_samples=args.bootstrap_samples,
            )
            rows.append({**base, "metric": metric, **summary})
    return pd.DataFrame(rows)


def paired_statistical_tests(
    per_pair: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if per_pair.empty or "model" not in per_pair:
        return pd.DataFrame([{"status": "MISSING", "reason": "no paired interventions"}])
    valid = per_pair[
        (per_pair["status"] == "AVAILABLE")
        & (per_pair["control"].isin([
            "constrained", "matching_only", "random_rewire",
        ]))
    ]
    metrics = [
        "routing_selectivity_main_branches",
        "routing_selectivity_with_interaction",
        "target_abs_delta", "non_target_abs_delta",
        "interaction_share", "abs_delta_total",
        "abs_delta_degree", "abs_delta_cn", "abs_delta_residual",
    ]
    rows: list[dict[str, Any]] = []
    strata = [("ALL", valid)]
    strata += [(f"dataset={name}", local) for name, local in valid.groupby("dataset")]
    strata += [
        (f"dataset={dataset},seed={seed}", local)
        for (dataset, seed), local in valid.groupby(["dataset", "seed"])
    ]
    strata += [(f"kind={name}", local) for name, local in valid.groupby("intervention_kind")]
    strata += [(f"group={name}", local) for name, local in valid.groupby("quadrant")]
    for stratum, local in strata:
        for control in ("constrained", "matching_only", "random_rewire"):
            subset = local[local["control"] == control]
            for metric in metrics:
                pivot = subset.pivot_table(
                    index=PAIR_KEYS, columns="model", values=metric, aggfunc="first"
                )
                if not {MODEL_A5, MODEL_A14} <= set(pivot.columns):
                    rows.append({
                        "status": "MISSING", "stratum": stratum,
                        "control": control, "metric": metric,
                        "reason": "strict A5/A14 pair unavailable",
                    })
                    continue
                pivot = pivot.dropna(subset=[MODEL_A5, MODEL_A14])
                effect = paired_effect(
                    pivot[MODEL_A14], pivot[MODEL_A5],
                    seed=stable_seed(args.seed, stratum, control, metric),
                    bootstrap_samples=args.bootstrap_samples,
                )
                p_value = np.nan
                if len(pivot) and np.any(
                    np.abs(pivot[MODEL_A14] - pivot[MODEL_A5]) > 1e-12
                ):
                    try:
                        p_value = float(wilcoxon(
                            pivot[MODEL_A14], pivot[MODEL_A5],
                            zero_method="zsplit",
                        ).pvalue)
                    except ValueError:
                        pass
                rows.append({
                    **effect, "stratum": stratum, "control": control,
                    "metric": metric, "p_value": p_value,
                })
    frame = pd.DataFrame(rows)
    finite = frame["p_value"].notna() if "p_value" in frame else pd.Series(dtype=bool)
    frame["p_value_holm"] = np.nan
    if finite.any():
        frame.loc[finite, "p_value_holm"] = holm_bonferroni(
            frame.loc[finite, "p_value"].to_numpy()
        )
    return frame


def dose_response_table(per_pair: pd.DataFrame) -> pd.DataFrame:
    if per_pair.empty:
        return pd.DataFrame([{"status": "MISSING", "reason": "no interventions"}])
    constrained = per_pair[
        (per_pair["status"] == "AVAILABLE")
        & (per_pair["control"] == "constrained")
    ]
    rows: list[dict[str, Any]] = []
    groups = [
        "dataset", "seed", "model", "pair_index", "u", "v",
        "quadrant", "intervention_kind",
    ]
    for group, local in constrained.groupby(groups):
        kind = group[-1]
        target = local[f"delta_{kind}"]
        non_target = local["non_target_abs_delta"]
        rows.append({
            **dict(zip(groups, group)),
            **dose_response_metrics(
                local["actual_magnitude"], target, non_target
            ),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame([
        {"status": "MISSING", "reason": "fewer than two legal magnitudes per pair"}
    ])


def prediction_pair_tests(
    pairs: list[tuple[RunArtifact, RunArtifact]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a5, a14 in pairs:
        if not a5.predictions or not a14.predictions:
            rows.append({
                "status": "MISSING", "dataset": a5.dataset, "seed": a5.seed,
                "metric": "reciprocal_rank", "reason": "prediction CSV missing",
            })
            continue
        first, second = pd.read_csv(a5.predictions), pd.read_csv(a14.predictions)
        first = first[first["label"] == 1].copy()
        second = second[second["label"] == 1].copy()
        merged = first.merge(second, on=["u", "v", "label"], suffixes=("_a5", "_a14"))
        if len(merged) != len(first) or len(merged) != len(second):
            rows.append({
                "status": "MISSING", "dataset": a5.dataset, "seed": a5.seed,
                "metric": "reciprocal_rank",
                "reason": "positive-edge prediction rows are not strictly paired",
            })
            continue
        effect = paired_effect(
            1.0 / merged["rank_a14"], 1.0 / merged["rank_a5"],
            seed=stable_seed(args.seed, a5.dataset, a5.seed, "rr"),
            bootstrap_samples=args.bootstrap_samples,
        )
        rows.append({
            **effect, "dataset": a5.dataset, "seed": a5.seed,
            "protocol_train": a5.protocol_train,
            "metric": "reciprocal_rank", "stratum": "ALL_POSITIVE_EDGES",
        })
        for quadrant, local in merged.groupby("quadrant_a5"):
            if not np.all(local["quadrant_a5"] == local["quadrant_a14"]):
                rows.append({
                    "status": "MISSING", "dataset": a5.dataset, "seed": a5.seed,
                    "metric": "reciprocal_rank", "stratum": f"group={quadrant}",
                    "reason": "quadrant definitions differ across predictions",
                })
                continue
            effect = paired_effect(
                1.0 / local["rank_a14"], 1.0 / local["rank_a5"],
                seed=stable_seed(args.seed, a5.dataset, a5.seed, quadrant),
                bootstrap_samples=args.bootstrap_samples,
            )
            rows.append({
                **effect, "dataset": a5.dataset, "seed": a5.seed,
                "protocol_train": a5.protocol_train,
                "metric": "reciprocal_rank", "stratum": f"group={quadrant}",
            })
    return rows


def efficiency_rows(
    pairs: list[tuple[RunArtifact, RunArtifact]],
    per_pair: pd.DataFrame,
    timings: list[dict[str, Any]],
    device: torch.device,
) -> pd.DataFrame:
    timing_index = {
        (item.get("dataset"), item.get("seed"), item.get("model")): item
        for item in timings if item.get("status") == "TIMING"
    }
    rows: list[dict[str, Any]] = []
    for a5, a14 in pairs:
        for artifact in (a5, a14):
            model, _ = load_model(artifact, device)
            local = per_pair[
                (per_pair.get("dataset") == artifact.dataset)
                & (per_pair.get("seed") == artifact.seed)
                & (per_pair.get("model") == artifact.model_label)
                & (per_pair.get("control") == "constrained")
            ] if not per_pair.empty else pd.DataFrame()
            timing = timing_index.get(
                (artifact.dataset, artifact.seed, artifact.model_label), {}
            )
            rows.append({
                "status": "AVAILABLE", "dataset": artifact.dataset,
                "seed": artifact.seed, "protocol_train": artifact.protocol_train,
                "model": artifact.model_label, "config_hash": artifact.config_hash,
                "checkpoint": str(artifact.checkpoint),
                "training_time": artifact.runtime.get("train_seconds", np.nan),
                "evaluation_time": timing.get("evaluation_seconds", np.nan),
                "intervention_generation_time": timing.get(
                    "intervention_generation_seconds", np.nan
                ),
                "peak_gpu_memory": artifact.runtime.get("peak_gpu_mb", np.nan),
                "peak_cpu_memory": "MISSING",
                "parameter_count": sum(value.numel() for value in model.parameters()),
                "mrr": artifact.metrics.get("mrr", np.nan),
                "worst_group_mrr": artifact.metrics.get("worst_group_mrr", np.nan),
                "group_gap": artifact.metrics.get("group_gap", np.nan),
                "routing_selectivity": (
                    local["routing_selectivity_main_branches"].mean()
                    if len(local) else np.nan
                ),
                "cross_sensitivity": (
                    local["non_target_abs_delta"].mean() if len(local) else np.nan
                ),
            })
    return pd.DataFrame(rows)


def ensure_csv(frame: pd.DataFrame, path: Path, reason: str = "no available data") -> None:
    if frame.empty:
        frame = pd.DataFrame([{"status": "MISSING", "reason": reason}])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def dataframe_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| status |\n|---|\n| MISSING |"
    columns = list(frame.columns)
    def clean(value: Any) -> str:
        if pd.isna(value):
            return "MISSING"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(map(str, columns)) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    lines.extend(
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def plot_pareto(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    stem: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(5.2, 3.8))
    usable = (
        frame.dropna(subset=[x, y])
        if x in frame and y in frame else pd.DataFrame()
    )
    if usable.empty:
        axis.text(0.5, 0.5, "MISSING", ha="center", va="center", fontsize=16)
        axis.set_axis_off()
    else:
        for model, local in usable.groupby("model"):
            axis.scatter(local[x], local[y], label=model, alpha=0.8)
        axis.set_xlabel(x.replace("_", " "))
        axis.set_ylabel(y.replace("_", " "))
        axis.legend()
        axis.grid(alpha=0.25)
    axis.set_title(title)
    fig.tight_layout()
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def decision_summary(
    paired_runs: pd.DataFrame,
    statistics: pd.DataFrame,
    probes: pd.DataFrame,
    interaction: pd.DataFrame,
) -> dict[str, Any]:
    missing_pairs = int((paired_runs["status"] != "AVAILABLE").sum())
    criteria: dict[str, dict[str, Any]] = {}
    available_runs = paired_runs[paired_runs["status"] == "AVAILABLE"]
    if len(available_runs):
        mrr_delta = (
            available_runs["a14_w010_mrr"] - available_runs["a5_mrr"]
        ).mean()
        criteria["performance_parity"] = {
            "met": bool(mrr_delta >= -PERFORMANCE_PARITY_TOLERANCE),
            "mean_mrr_delta": float(mrr_delta),
            "predeclared_tolerance": PERFORMANCE_PARITY_TOLERANCE,
        }
    else:
        criteria["performance_parity"] = {"met": False, "status": "MISSING"}

    all_constrained = (
        statistics[
            (statistics["stratum"] == "ALL")
            & (statistics["control"] == "constrained")
        ]
        if {"stratum", "control"} <= set(statistics.columns)
        else pd.DataFrame()
    )
    def stat_delta(stratum: str, metric: str, control: str = "constrained") -> float | None:
        if all_constrained.empty and stratum == "ALL":
            return None
        source = statistics[
            (statistics.get("stratum") == stratum)
            & (statistics.get("control") == control)
            & (statistics.get("metric") == metric)
            & (statistics.get("status") == "AVAILABLE")
        ] if {"stratum", "control", "metric", "status"} <= set(statistics.columns) else pd.DataFrame()
        return float(source.iloc[0]["mean_delta"]) if len(source) else None

    route = {
        kind: stat_delta(
            f"kind={kind}", "routing_selectivity_main_branches"
        )
        for kind in ("degree", "cn")
    }
    route_available = all(value is not None for value in route.values())
    criteria["routing"] = {
        "met": bool(
            route_available
            and max(route.values()) > 0
            and min(route.values()) >= -0.01
        ),
        "status": "AVAILABLE" if route_available else "MISSING",
        "mean_deltas": route,
    }
    cross_metrics = {
        "degree_to_cn": stat_delta("kind=degree", "abs_delta_cn"),
        "degree_to_residual": stat_delta("kind=degree", "abs_delta_residual"),
        "cn_to_degree": stat_delta("kind=cn", "abs_delta_degree"),
        "cn_to_residual": stat_delta("kind=cn", "abs_delta_residual"),
    }
    cross_available = all(value is not None for value in cross_metrics.values())
    criteria["cross_sensitivity"] = {
        "met": bool(
            cross_available
            and sum(value < 0 for value in cross_metrics.values()) >= 2
        ),
        "status": "AVAILABLE" if cross_available else "MISSING",
        "mean_deltas": cross_metrics,
    }
    probe_pairs = (
        probes[
            (probes["model"] == "PAIRED_A14_MINUS_A5")
            & (probes["status"] == "AVAILABLE")
        ]
        if {"model", "status"} <= set(probes.columns)
        else pd.DataFrame()
    )
    cross_probes = probe_pairs[probe_pairs["role"] == "cross_leakage"] if len(probe_pairs) else pd.DataFrame()
    probe_improvements = []
    for _, row in cross_probes.iterrows():
        if row.get("desired_for_cross_leakage") == "LOWER":
            probe_improvements.append(float(row["mean_delta"]) < 0)
        else:
            probe_improvements.append(float(row["mean_delta"]) > 0)
    criteria["posthoc_leakage"] = {
        "met": bool(probe_improvements and any(probe_improvements)),
        "status": "AVAILABLE" if len(cross_probes) else "MISSING",
        "improved_probe_count": int(sum(probe_improvements)),
        "paired_probe_count": int(len(probe_improvements)),
    }
    target_probes = probe_pairs[probe_pairs["role"] == "target_retention"] if len(probe_pairs) else pd.DataFrame()
    retained = []
    for _, row in target_probes.iterrows():
        delta = float(row["mean_delta"])
        retained.append(
            delta >= -0.02 if row["primary_metric"] == "accuracy_contribution"
            else delta <= 0.02
        )
    criteria["target_retention"] = {
        "met": bool(retained and all(retained)),
        "status": "AVAILABLE" if len(target_probes) else "MISSING",
        "retained_count": int(sum(retained)),
        "paired_probe_count": int(len(retained)),
    }
    interaction_available = (
        interaction[
            (interaction["status"] == "AVAILABLE")
            & (interaction["metric"] == "interaction_share")
            & (interaction["control"] == "constrained")
        ]
        if {"status", "metric", "control"} <= set(interaction.columns) else pd.DataFrame()
    )
    criteria["interaction_bypass"] = {
        "met": bool(
            len(interaction_available)
            and interaction_available["mean"].max() < 0.5
            and (
                stat_delta("ALL", "interaction_share") is not None
                and stat_delta("ALL", "interaction_share") <= 0.05
            )
        ),
        "status": "AVAILABLE" if len(interaction_available) else "MISSING",
        "max_interaction_share": (
            float(interaction_available["mean"].max())
            if len(interaction_available) else "MISSING"
        ),
        "paired_mean_delta": stat_delta("ALL", "interaction_share"),
    }
    constrained_route = stat_delta("ALL", "routing_selectivity_main_branches")
    random_route = stat_delta(
        "ALL", "routing_selectivity_main_branches", "random_rewire"
    )
    criteria["random_rewire_control"] = {
        "met": bool(
            constrained_route is not None and random_route is not None
            and constrained_route > random_route
        ),
        "status": (
            "AVAILABLE" if constrained_route is not None and random_route is not None
            else "MISSING"
        ),
        "a14_advantage_constrained": constrained_route,
        "a14_advantage_random_rewire": random_route,
    }
    seed_rows = statistics[
        statistics["stratum"].astype(str).str.startswith("dataset=")
        & statistics["stratum"].astype(str).str.contains(",seed=")
        & (statistics["control"] == "constrained")
        & (statistics["metric"] == "routing_selectivity_main_branches")
        & (statistics["status"] == "AVAILABLE")
    ] if {"stratum", "control", "metric", "status"} <= set(statistics.columns) else pd.DataFrame()
    criteria["cross_dataset_seed_stability"] = {
        "met": bool(
            len(seed_rows)
            and (seed_rows["mean_delta"] >= 0).mean() >= 2 / 3
        ),
        "status": "AVAILABLE" if len(seed_rows) else "MISSING",
        "nonnegative_fraction": (
            float((seed_rows["mean_delta"] >= 0).mean())
            if len(seed_rows) else "MISSING"
        ),
    }
    complete = missing_pairs == 0 and all(
        item.get("status") != "MISSING" for item in criteria.values()
    )
    supported = complete and all(item.get("met", False) for item in criteria.values())
    return {
        "evidence_status": "COMPLETE" if complete else "INCOMPLETE_MISSING",
        "missing_run_pairs": missing_pairs,
        "decision": (
            "A14_MECHANISM_SUPPORTED_WITH_PERFORMANCE_PARITY"
            if supported else "A14_MECHANISM_NOT_SUPPORTED"
        ),
        "criteria": criteria,
        "test_set_tuning_warning": (
            "A14-w010 was selected after inspecting test-set MRR in exploratory "
            "runs; this phase is diagnostic, not independent confirmatory evidence."
        ),
    }


def markdown_report(
    output: Path,
    summary: dict[str, Any],
    paired_runs: pd.DataFrame,
    inventory: list[dict[str, Any]],
    cache_root: Path,
) -> None:
    available = int((paired_runs["status"] == "AVAILABLE").sum())
    missing = int((paired_runs["status"] != "AVAILABLE").sum())
    reusable = [
        "DCDLP.forward 已输出 total/degree/CN/residual/interaction 分数及三类表示。",
        "现有 degree/CN 受约束干预、编辑日志和 validator 可直接复用。",
        "ConditionalCNRegressor、HH/HL/LH/LL 分组、bootstrap 与 Holm 校正已有基础实现。",
        "现有逐边预测 CSV 提供正边 rank 和三主分支分数。",
    ]
    missing_features = [
        "原统一分析入口、score-level routing、interaction bypass、dose-response 未接入。",
        "原 post-hoc probe 没有 validation 选参、标准化、Spearman 或严格模型配对。",
        "原 cliffs_delta 是非配对笛卡尔比较，不适合本任务的逐边配对效应。",
        "A8/A9 matching/random-rewire 对照无可复用模型；本实现仅生成冻结模型评测对照。",
    ]
    risks = [
        "A14-w010 来源于查看测试 MRR 后的探索性权重选择，不能作为独立确认性结论。",
        "HeaRT 固定划分对不同 model seed 复用 seed-0 数据；seed 是训练随机性而非独立数据划分。",
        "缓存 pair_id 若仅为位置索引，脱离 dataset/split/seed 后可能错配；审计同时核验 u/v/type/magnitude。",
        "逐边比较只允许相同 dataset/seed/protocol/u/v/intervention/control；任何缺项均标 MISSING。",
        "官方 HeaRT/OGB 候选不可用时不得用 uniform 候选替代；机制审计不会生成替代排名。",
    ]
    answers = [
        ("Degree 路由选择性是否提高？", summary["criteria"].get("routing", {})),
        ("CN 路由选择性是否提高？", summary["criteria"].get("routing", {})),
        ("交叉分支敏感性是否降低？", summary["criteria"].get("cross_sensitivity", {})),
        ("post-hoc 泄漏是否降低？", summary["criteria"].get("posthoc_leakage", {})),
        ("目标机制信息是否保持？", "见 posthoc_probe.csv 的 target_retention；缺失则为 MISSING"),
        ("interaction 是否成为绕行通道？", summary["criteria"].get("interaction_bypass", {})),
        ("是否优于 random-rewire？", "见 statistical_tests.csv；缺失配对不推断"),
        ("是否跨 dataset/seed 稳定？", f"可用严格运行配对={available}，缺失={missing}"),
        ("运行成本是否有合理回报？", "见 efficiency.csv 与 Pareto 图；机制条件未满足则无合理回报"),
        ("最终建议", summary["decision"]),
    ]
    lines = [
        "# DCDLP A5 / A14-w010 机制审计报告",
        "",
        f"- 证据状态：`{summary['evidence_status']}`",
        f"- 严格配对：AVAILABLE={available}, MISSING={missing}",
        f"- 决策：`{summary['decision']}`",
        f"- 干预缓存目录：`{cache_root}`（不存在或不匹配时显式记录 GENERATED_EVALUATION/MISSING）",
        "",
        "## 1. 可直接复用的功能",
        *[f"- {item}" for item in reusable],
        "",
        "## 2. 原先尚未实现或未接入的功能",
        *[f"- {item}" for item in missing_features],
        "",
        "## 3. 本阶段新增/修改文件",
        "- `scripts/analyze_mechanism_audit.py`",
        "- `src/dcdlp/evaluation/routing_audit.py`",
        "- `src/dcdlp/evaluation/posthoc_probe.py`",
        "- `src/dcdlp/evaluation/dose_response.py`",
        "- `tests/test_routing_audit.py`",
        "- `tests/test_posthoc_probe.py`",
        "",
        "## 4. 数据与统计风险",
        *[f"- {item}" for item in risks],
        "",
        "## 5. 运行配对",
        "",
        dataframe_markdown(paired_runs),
        "",
        "## 6. 验收问题",
        "",
    ]
    for index, (question, answer) in enumerate(answers, 1):
        lines.extend([
            f"{index}. **{question}**",
            f"   - `{json.dumps(finite_or_none(answer), ensure_ascii=False, default=str)}`",
        ])
    lines.extend([
        "",
        "## 7. 决策依据",
        "",
        "```json",
        json.dumps(finite_or_none(summary["criteria"]), indent=2, ensure_ascii=False),
        "```",
        "",
        "若证据状态为 `INCOMPLETE_MISSING`，报告不会用默认 A14、替代分数或未配对样本填补缺口。",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.use_existing_checkpoints:
        raise SystemExit("--use-existing-checkpoints is required; training is forbidden")
    requested = {item.strip() for item in args.models.split(",") if item.strip()}
    if requested != {MODEL_A5, MODEL_A14}:
        raise SystemExit("Phase one requires exactly --models A5,A14-w010")
    root = Path(__file__).resolve().parents[1]
    results_root = (root / args.results_dir).resolve()
    output = (root / args.output_dir).resolve()
    data_root = (root / args.data_root).resolve()
    cache_root = (root / args.cache_dir).resolve()
    args.data_root = str(data_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    artifacts, inventory = discover_runs(results_root, output)
    paired_runs, pairs = pair_runs(args, artifacts)
    print(
        f"[DISCOVERY] scoped_artifacts={len(artifacts)} "
        f"available_pairs={len(pairs)} "
        f"missing_pairs={(paired_runs['status'] != 'AVAILABLE').sum()}",
        flush=True,
    )
    ensure_csv(paired_runs, output / "paired_runs.csv")
    ensure_csv(pd.DataFrame(inventory), output / "artifact_inventory.csv")

    per_pair_rows: list[dict[str, Any]] = []
    timing_and_failure_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    for a5, a14 in pairs:
        print(
            f"[PAIR_START] dataset={a5.dataset} seed={a5.seed} "
            f"protocol_train={a5.protocol_train}",
            flush=True,
        )
        try:
            rows, timing = audit_interventions(
                args, a5, a14, device, cache_root
            )
            per_pair_rows.extend(rows)
            timing_and_failure_rows.extend(timing)
            print(
                f"[INTERVENTIONS_DONE] dataset={a5.dataset} seed={a5.seed} "
                f"rows={len(rows)}",
                flush=True,
            )
        except Exception as exc:
            timing_and_failure_rows.append({
                "status": "MISSING", "dataset": a5.dataset, "seed": a5.seed,
                "reason": f"intervention audit failed: {exc}",
            })
        try:
            probe_rows.extend(audit_probes(args, a5, a14, device))
            print(
                f"[PROBES_DONE] dataset={a5.dataset} seed={a5.seed}",
                flush=True,
            )
        except Exception as exc:
            probe_rows.append({
                "status": "MISSING", "dataset": a5.dataset, "seed": a5.seed,
                "reason": f"post-hoc probe audit failed: {exc}",
            })
        print(
            f"[PAIR_DONE] dataset={a5.dataset} seed={a5.seed}",
            flush=True,
        )

    per_pair = pd.DataFrame(per_pair_rows)
    ensure_csv(per_pair, output / "per_pair_branch_deltas.csv")
    ensure_csv(
        pd.DataFrame(timing_and_failure_rows),
        output / "audit_failures_and_timings.csv",
    )

    if not per_pair.empty:
        valid_pair_rows = per_pair[per_pair["status"] == "AVAILABLE"]
        constrained = valid_pair_rows[
            valid_pair_rows["control"] == "constrained"
        ]
    else:
        valid_pair_rows = pd.DataFrame()
        constrained = pd.DataFrame()
    routing = summarize_long(
        valid_pair_rows,
        ["routing_selectivity_main_branches", "routing_selectivity_with_interaction"],
        ["dataset", "seed", "model", "intervention_kind", "quadrant", "control"],
        args,
    )
    cross = summarize_long(
        valid_pair_rows,
        [
            "abs_delta_degree", "abs_delta_cn", "abs_delta_residual",
            "abs_delta_interaction", "target_abs_delta",
            "non_target_abs_delta", "abs_delta_total",
        ],
        ["dataset", "seed", "model", "intervention_kind", "control"],
        args,
    )
    interaction = summarize_long(
        valid_pair_rows, ["interaction_share", "abs_delta_interaction"],
        ["dataset", "seed", "model", "intervention_kind", "control"], args,
    )
    dose = dose_response_table(per_pair)
    probes = add_probe_comparisons(pd.DataFrame(probe_rows), args)
    statistics = paired_statistical_tests(per_pair, args)
    prediction_statistics = pd.DataFrame(prediction_pair_tests(pairs, args))
    if not prediction_statistics.empty:
        statistics = pd.concat([statistics, prediction_statistics], ignore_index=True)
    timings = [
        row for row in timing_and_failure_rows if row.get("status") == "TIMING"
    ]
    efficiency = efficiency_rows(pairs, per_pair, timings, device)

    ensure_csv(routing, output / "routing_selectivity.csv")
    ensure_csv(cross, output / "cross_sensitivity.csv")
    ensure_csv(probes, output / "posthoc_probe.csv")
    ensure_csv(dose, output / "dose_response.csv")
    ensure_csv(interaction, output / "interaction_audit.csv")
    ensure_csv(statistics, output / "statistical_tests.csv")
    ensure_csv(efficiency, output / "efficiency.csv")

    # A single probe is used only as a plotting coordinate, never for tuning.
    if not efficiency.empty and not probes.empty:
        leakage = probes[
            (probes.get("representation") == "z_cn")
            & (probes.get("target") == "degree_bin")
            & (probes.get("status") == "AVAILABLE")
        ]
        leakage = leakage[["dataset", "seed", "model", "macro_f1"]].rename(
            columns={"macro_f1": "leakage_probe_score"}
        )
        efficiency = efficiency.merge(
            leakage, on=["dataset", "seed", "model"], how="left"
        )
        ensure_csv(efficiency, output / "efficiency.csv")

    figures = output / "figures"
    plot_pareto(
        efficiency, "mrr", "routing_selectivity",
        "MRR vs Routing Selectivity", figures / "mrr_vs_routing",
    )
    if "leakage_probe_score" not in efficiency:
        efficiency["leakage_probe_score"] = np.nan
    plot_pareto(
        efficiency, "mrr", "leakage_probe_score",
        "MRR vs Leakage Probe Score", figures / "mrr_vs_leakage",
    )
    plot_pareto(
        efficiency, "training_time", "routing_selectivity",
        "Runtime vs Routing Selectivity", figures / "runtime_vs_routing",
    )
    plot_pareto(
        efficiency, "training_time", "worst_group_mrr",
        "Runtime vs Worst Group MRR", figures / "runtime_vs_worst_group",
    )

    summary = decision_summary(paired_runs, statistics, probes, interaction)
    summary.update({
        "audit_seed": args.seed,
        "requested_models": sorted(requested),
        "available_pair_count": len(pairs),
        "artifact_count": len(artifacts),
        "cache_files": [str(path) for path in cache_root.rglob("*") if path.is_file()]
        if cache_root.exists() else [],
        "output_dir": str(output),
    })
    json_dump(output / "summary.json", finite_or_none(summary))
    markdown_report(
        output / "MECHANISM_AUDIT_REPORT.md",
        summary, paired_runs, inventory, cache_root,
    )
    print(json.dumps({
        "output_dir": str(output),
        "available_pairs": len(pairs),
        "missing_pairs": int((paired_runs["status"] != "AVAILABLE").sum()),
        "decision": summary["decision"],
        "evidence_status": summary["evidence_status"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
