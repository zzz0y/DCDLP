from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .data.loaders import load_dataset
from .baselines.gae import GAE
from .evaluation.intervention_metrics import cross_sensitivity, routing_selectivity, sensitivity
from .evaluation.routing_audit import branch_response
from .interventions import InterventionError, intervene_cn, intervene_degree
from .models.dcdlp import DCDLP
from .train import ablation_profile, edge_index_from_graph, evaluate_split, score_pairs
from .utils import write_json


def _one(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _prediction_interaction_share(output: dict) -> float:
    if "interaction_share" in output:
        return _one(output["interaction_share"])
    components = [
        abs(_one(output[f"score_{name}"]))
        for name in ("degree", "cn", "residual", "interaction")
    ]
    return components[-1] / (sum(components) + 1e-8)


def _intervention_evaluation(
    model,
    dataset,
    x,
    base_edges,
    seed: int,
    max_pairs: int = 100,
    interaction_share_cap: float = 0.2,
):
    graph = dataset.train_graph()
    forbidden = {tuple(edge) for edge in np.vstack([dataset.valid_pos, dataset.test_pos]).tolist()}
    records = {"degree": [], "cn": []}
    rows: list[dict] = []
    failures: dict[str, int] = {}
    for index, pair in enumerate(dataset.test_pos[:max_pairs]):
        u, v = map(int, pair)
        original = score_pairs(model, x, base_edges, np.asarray([[u, v]]))
        for kind in ("degree", "cn"):
            try:
                if kind == "degree":
                    cf_graph, log = intervene_degree(graph, u, v, "u", 1, seed + index * 17, forbidden)
                else:
                    cf_graph, log = intervene_cn(graph, u, v, 1, seed + index * 17, forbidden)
                cf_edges = edge_index_from_graph(cf_graph, x.device)
                counterfactual = score_pairs(model, x, cf_edges, np.asarray([[u, v]]))
                records[kind].append((original, counterfactual))
                response = branch_response(original, counterfactual, kind)
                target = kind
                cross_names = [
                    name for name in ("degree", "cn", "residual")
                    if name != target
                ]
                cross_response = (
                    sum(response[f"abs_delta_{name}"] for name in cross_names)
                    + response["abs_delta_interaction"]
                )
                target_response = response[f"abs_delta_{target}"]
                row = {
                    "pair_id": str(index),
                    "u": u,
                    "v": v,
                    "intervention_kind": kind,
                    "intervention_type": log.intervention_type,
                    "requested_delta": log.requested_delta,
                    "status": "AVAILABLE",
                    "failure_code": "",
                    "target_response": target_response,
                    "cross_response": cross_response,
                    "routing_selectivity": target_response
                    / (target_response + cross_response + 1e-8),
                    "cross_sensitivity": cross_response,
                    "routing_ratio": target_response
                    / (cross_response + 1e-8),
                    "interaction_share_before": _prediction_interaction_share(
                        original
                    ),
                    "interaction_share_after": _prediction_interaction_share(
                        counterfactual
                    ),
                    "interaction_share_cap": interaction_share_cap,
                    "interaction_share_cap_violation_before": int(
                        _prediction_interaction_share(original)
                        > interaction_share_cap
                    ),
                    "pre_degree_u": graph.degree(u),
                    "pre_degree_v": graph.degree(v),
                    "post_degree_u": cf_graph.degree(u),
                    "post_degree_v": cf_graph.degree(v),
                }
                for name, output_name in (
                    ("total", "logit"),
                    ("degree", "score_degree"),
                    ("cn", "score_cn"),
                    ("residual", "score_residual"),
                    ("interaction", "score_interaction"),
                ):
                    before = _one(original[output_name])
                    after = _one(counterfactual[output_name])
                    row[f"score_{name}_before"] = before
                    row[f"score_{name}_after"] = after
                    row[f"delta_score_{name}"] = after - before
                    row[f"abs_delta_score_{name}"] = abs(after - before)
                for name in ("cn_raw", "cn_expected", "cn_residual_feature"):
                    if name in original and name in counterfactual:
                        before = _one(original[name])
                        after = _one(counterfactual[name])
                        row[f"{name}_before"] = before
                        row[f"{name}_after"] = after
                        row[f"delta_{name}"] = after - before
                rows.append(row)
            except InterventionError as exc:
                failures[exc.code] = failures.get(exc.code, 0) + 1
                rows.append({
                    "pair_id": str(index),
                    "u": u,
                    "v": v,
                    "intervention_kind": kind,
                    "intervention_type": kind,
                    "status": "MISSING",
                    "failure_code": exc.code,
                })
    attempted_pairs = min(max_pairs, len(dataset.test_pos))
    output = {
        "status": "COMPLETED",
        "attempted_pairs": attempted_pairs,
        "failures": failures,
        "records": rows,
    }
    for kind, values in records.items():
        output[f"valid_{kind}"] = len(values)
        kind_rows = [
            row for row in rows
            if row["intervention_kind"] == kind
            and row["status"] == "AVAILABLE"
        ]
        missing_rows = [
            row for row in rows
            if row["intervention_kind"] == kind
            and row["status"] == "MISSING"
        ]
        output[f"coverage_{kind}"] = {
            "attempted": attempted_pairs,
            "valid": len(kind_rows),
            "missing": len(missing_rows),
            "valid_rate": len(kind_rows) / attempted_pairs
            if attempted_pairs else 0.0,
            "failure_codes": {
                code: sum(row["failure_code"] == code for row in missing_rows)
                for code in sorted({row["failure_code"] for row in missing_rows})
            },
        }
        if not values:
            continue
        original_scores = np.asarray([item[0]["logit"][0] for item in values])
        cf_scores = np.asarray([item[1]["logit"][0] for item in values])
        output["dsi" if kind == "degree" else "csi"] = sensitivity(original_scores, cf_scores)
        target = "degree" if kind == "degree" else "cn"
        others = [name for name in ("degree", "cn", "residual") if name != target]
        target_delta = np.asarray([
            np.linalg.norm(item[1][f"z_{target}"] - item[0][f"z_{target}"]) for item in values
        ])
        other_deltas = [np.asarray([
            np.linalg.norm(item[1][f"z_{name}"] - item[0][f"z_{name}"]) for item in values
        ]) for name in others]
        output[f"routing_{kind}"] = routing_selectivity(target_delta, *other_deltas)
        wrong = "cn" if kind == "degree" else "degree"
        output[f"cross_{kind}_to_{wrong}"] = cross_sensitivity(
            np.vstack([item[0][f"z_{wrong}"] for item in values]),
            np.vstack([item[1][f"z_{wrong}"] for item in values]),
        )
        for name in ("degree", "cn", "residual", "interaction"):
            output[f"mean_abs_delta_score_{name}_{kind}"] = float(
                np.mean([
                    row[f"abs_delta_score_{name}"] for row in kind_rows
                ])
            )
        for metric in (
            "target_response",
            "cross_response",
            "routing_selectivity",
            "cross_sensitivity",
            "routing_ratio",
        ):
            output[f"{metric}_{kind}"] = float(
                np.mean([row[metric] for row in kind_rows])
            )
        shares = np.asarray([
            row["interaction_share_before"] for row in kind_rows
        ])
        output[f"interaction_{kind}"] = {
            "mean_share": float(shares.mean()),
            "median_share": float(np.median(shares)),
            "p90_share": float(np.quantile(shares, 0.9)),
            "mean_abs_delta": float(np.mean([
                row["abs_delta_score_interaction"] for row in kind_rows
            ])),
            "share_cap_violation_rate": float(np.mean([
                row["interaction_share_cap_violation_before"]
                for row in kind_rows
            ])),
        }
    return output


def load_checkpoint_model(
    checkpoint: Path,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(checkpoint, map_location="cpu")
    config = payload["config"]
    profile = ablation_profile(config.get("ablation", "A14"))
    if profile["model"] == "gae":
        model = GAE(payload["input_dim"], config["hidden_dim"], config["backbone"], config["branch_dim"])
    else:
        model = DCDLP(
            payload["input_dim"], config["hidden_dim"], config["branch_dim"], config["num_layers"],
            config["dropout"], config["backbone"], use_interaction=profile["interaction"],
            active_branches=profile["active"], decoder_mode=profile["decoder"],
            cn_feature_mode=config.get("cn_feature_mode", "raw"),
            cn_regressor=payload.get("cn_regressor"),
            interaction_mode=config.get("interaction_mode", "unrestricted"),
            # Checkpoints written before the selective-CN revision used
            # mode-dependent encoder widths.  Preserve that exact state-dict
            # shape while all newly trained checkpoints record selective_v1.
            cn_input_schema=config.get("cn_input_schema", "legacy"),
        )
    model.load_state_dict(payload["model"])
    model.to(torch.device(device))
    model.eval()
    return model, payload


def evaluate_checkpoint(
    checkpoint: Path,
    data_root: Path,
    protocols: list[str],
    output_dir: Path | None = None,
) -> dict:
    model, payload = load_checkpoint_model(checkpoint)
    config = payload["config"]
    dataset = load_dataset(config["dataset"], data_root, config.get("protocol_eval", "standard"), config["seed"])
    x = torch.as_tensor(dataset.features, dtype=torch.float32)
    edges = edge_index_from_graph(dataset.train_graph(), torch.device("cpu"))
    output = {}
    for index, protocol in enumerate(protocols):
        if protocol == "standard":
            metrics, _ = evaluate_split(model, dataset, dataset.test_pos, config["seed"] + 5000 + index,
                                        config["negatives_per_positive_eval"], x, edges, "uniform")
            output[protocol] = metrics
        elif protocol == "degree_corrected":
            metrics, _ = evaluate_split(model, dataset, dataset.test_pos, config["seed"] + 5000 + index,
                                        config["negatives_per_positive_eval"], x, edges, "degree_corrected")
            output[protocol] = metrics
        elif protocol == "intervention":
            output[protocol] = _intervention_evaluation(
                model,
                dataset,
                x,
                edges,
                config["seed"] + 5000 + index,
                interaction_share_cap=float(
                    config.get("interaction_share_cap", 0.2)
                ),
            )
        elif protocol in {"heart", "ogb"}:
            if dataset.test_neg is None:
                raise RuntimeError(
                    "Official grouped candidates are absent from the loaded "
                    "dataset; refusing to substitute Uniform negatives"
                )
            else:
                metrics, _ = evaluate_split(
                    model,
                    dataset,
                    dataset.test_pos,
                    config["seed"] + 5000 + index,
                    config["negatives_per_positive_eval"],
                    x,
                    edges,
                    "official",
                    dataset.test_neg,
                )
                output[protocol] = {
                    "status": "AVAILABLE",
                    "candidate_source": "dataset_official_grouped",
                    **metrics,
                }
        else:
            output[protocol] = {"status": "NOT_AVAILABLE", "reason": f"Unknown protocol {protocol}"}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        intervention_rows = output.get("intervention", {}).get("records", [])
        if intervention_rows:
            fieldnames = sorted({
                key for row in intervention_rows for key in row
            })
            with (output_dir / "intervention_predictions.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(intervention_rows)
        write_json(output_dir / "evaluation.json", output)
    return output
