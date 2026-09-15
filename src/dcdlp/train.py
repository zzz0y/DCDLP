from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data.loaders import GraphDataset
from .data.negative_sampling import degree_corrected_sampling, grouped_negatives, uniform_negative_sampling
from .data.pair_statistics import ConditionalCNRegressor, assign_quadrants, pair_features
from .evaluation.classification import classification_metrics
from .evaluation.group_metrics import group_ranking_metrics
from .evaluation.ranking import ranking_metrics
from .baselines.gae import GAE
from .interventions import InterventionError, intervene_cn, intervene_degree
from .models.adversary import AdversarialProbe
from .models.dcdlp import DCDLP
from .models.losses import (
    interaction_regularization,
    intervention_losses,
    link_prediction_loss,
    orthogonal_loss,
    score_routing_losses,
)
from .utils import (
    array_hash,
    git_commit,
    project_root,
    seed_everything,
    stable_hash,
    write_json,
)


@dataclass
class TrainConfig:
    dataset: str = "smoke"
    protocol_train: str = "uniform"
    protocol_eval: str = "standard"
    seed: int = 0
    hidden_dim: int = 64
    branch_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    backbone: str = "gcn"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    pretrain_epochs: int = 2
    disentangle_epochs: int = 2
    negatives_per_positive_eval: int = 20
    intervention_ratio: float = 0.25
    max_interventions_per_batch: int = 0
    max_interventions_per_epoch: int = 0
    lambda_inv: float = 0.03
    lambda_route: float = 0.03
    lambda_adv: float = 0.01
    lambda_orth: float = 0.001
    lambda_degrob: float = 0.01
    output_dir: str = "results"
    prediction_subdir: str = "raw"
    device: str = "auto"
    ablation: str = "A14"
    cn_feature_mode: str = "raw"
    cn_input_schema: str = "selective_v1"
    interaction_mode: str = "unrestricted"
    interaction_share_cap: float = 0.2
    interaction_delta_cap: float | None = None
    lambda_interaction_share: float = 0.0
    lambda_interaction_delta: float = 0.0
    lambda_link: float = 1.0
    score_routing_enabled: bool = False
    lambda_score_route: float = 0.0
    lambda_target_response: float = 0.0
    routing_margin_cn: float = 0.0
    routing_margin_degree: float = 0.0
    target_floor_cn: float = 0.0
    target_floor_degree: float = 0.0
    interaction_route_weight: float = 1.0
    routing_finetune_encoder: str = "frozen"
    encoder_lr_multiplier: float = 0.1


def ablation_profile(ablation: str) -> dict:
    profiles = {
        "A0": {"model": "gae", "active": ("residual",), "interaction": False, "decoder": "additive", "losses": set(), "interventions": "none"},
        "A1": {"model": "dcdlp", "active": ("degree",), "interaction": False, "decoder": "additive", "losses": set(), "interventions": "none"},
        "A2": {"model": "dcdlp", "active": ("cn",), "interaction": False, "decoder": "additive", "losses": set(), "interventions": "none"},
        "A3": {"model": "dcdlp", "active": ("residual",), "interaction": False, "decoder": "additive", "losses": set(), "interventions": "none"},
        "A4": {"model": "dcdlp", "active": ("degree", "cn"), "interaction": False, "decoder": "concat", "losses": set(), "interventions": "none"},
        "A5": {"model": "dcdlp", "active": ("degree", "cn", "residual"), "interaction": True, "decoder": "additive", "losses": set(), "interventions": "none"},
        "A6": {"model": "dcdlp", "active": ("degree", "cn", "residual"), "interaction": True, "decoder": "additive", "losses": {"orth"}, "interventions": "none"},
        "A7": {"model": "dcdlp", "active": ("degree", "cn", "residual"), "interaction": True, "decoder": "additive", "losses": {"adv"}, "interventions": "none"},
        "A8": {"unsupported": "Matching-pair cache is required; run build-pair-stats and the matching adapter."},
        "A9": {"unsupported": "Random-rewire audit requires a cached edit-count-matched intervention set."},
        "A10": {"model": "dcdlp", "active": ("degree", "cn", "residual"), "interaction": True, "decoder": "additive", "losses": {"orth", "adv", "intervention"}, "interventions": "cn"},
        "A11": {"model": "dcdlp", "active": ("degree", "cn", "residual"), "interaction": True, "decoder": "additive", "losses": {"orth", "adv", "intervention"}, "interventions": "degree"},
        "A12": {"model": "dcdlp", "active": ("degree", "cn", "residual"), "interaction": True, "decoder": "additive", "losses": {"orth", "adv", "intervention"}, "interventions": "both", "no_degrob": True},
        "A13": {"model": "dcdlp", "active": ("degree", "cn", "residual"), "interaction": False, "decoder": "additive", "losses": {"orth", "adv", "intervention"}, "interventions": "both"},
        "A14": {"model": "dcdlp", "active": ("degree", "cn", "residual"), "interaction": True, "decoder": "additive", "losses": {"orth", "adv", "intervention"}, "interventions": "both"},
    }
    if ablation not in profiles:
        raise ValueError(f"Unknown ablation {ablation}")
    return profiles[ablation]


def edge_index_from_graph(graph, device: torch.device) -> torch.Tensor:
    edges = np.asarray(list(graph.edges()), dtype=np.int64).reshape(-1, 2)
    return torch.as_tensor(edges.T, dtype=torch.long, device=device)


def _sample_train_negatives(dataset: GraphDataset, config: TrainConfig, epoch: int) -> np.ndarray:
    graph = dataset.train_graph()
    count = len(dataset.train_pos)
    local_seed = config.seed * 10_000 + epoch
    if config.protocol_train == "uniform":
        return uniform_negative_sampling(dataset.num_nodes, dataset.all_positive, count, local_seed)
    if config.protocol_train == "degree_corrected":
        return degree_corrected_sampling(graph, dataset.all_positive, count, local_seed)
    if config.protocol_train == "mix":
        first = count // 2
        uniform = uniform_negative_sampling(dataset.num_nodes, dataset.all_positive, first, local_seed)
        dc = degree_corrected_sampling(graph, np.vstack([dataset.all_positive, uniform]), count - first, local_seed + 1)
        return np.vstack([uniform, dc])
    raise ValueError(f"Unknown training negative protocol: {config.protocol_train}")


def fit_training_cn_residualizer(
    dataset: GraphDataset,
    config: TrainConfig,
) -> tuple[ConditionalCNRegressor | None, dict[str, object]]:
    """Fit CN calibration strictly on train positives and fixed train negatives."""
    metadata: dict[str, object] = {
        "cn_feature_mode": config.cn_feature_mode,
        "cn_input_schema": config.cn_input_schema,
        "fit_split": "train",
        "uses_valid_or_test": False,
        "data_source": "train_positive+fixed_train_negative_epoch0",
        "fit_seed": int(config.seed),
        "fit_sample_count": 0,
        "data_hash": "NOT_USED",
        "config_hash": stable_hash({
            "dataset": config.dataset,
            "protocol_train": config.protocol_train,
            "seed": config.seed,
            "cn_feature_mode": config.cn_feature_mode,
            "cn_input_schema": config.cn_input_schema,
        }),
        "residualizer_type": "NOT_USED",
        "residualizer_parameters": {},
        "used_as_model_input": config.cn_feature_mode != "raw",
    }
    fixed_train_negative = _sample_train_negatives(dataset, config, 0)
    calibration_pairs = np.vstack([
        dataset.train_pos,
        fixed_train_negative,
    ]).astype(np.int64, copy=False)
    held_out = {
        tuple(sorted(map(int, pair)))
        for pair in np.vstack([dataset.valid_pos, dataset.test_pos])
    }
    leaked = [
        tuple(sorted(map(int, pair)))
        for pair in calibration_pairs
        if tuple(sorted(map(int, pair))) in held_out
    ]
    if leaked:
        raise RuntimeError(
            "CN residualizer calibration contains held-out valid/test pairs"
        )
    graph = dataset.train_graph()
    calibration_stats = pair_features(
        graph, calibration_pairs, dataset.features
    )
    fit_context = {
        **metadata,
        "fit_sample_count": int(len(calibration_pairs)),
        "data_hash": array_hash(calibration_pairs),
    }
    regressor = ConditionalCNRegressor(
        config.seed, symmetric=True
    ).fit(calibration_stats, metadata=fit_context)
    return regressor, dict(regressor.fit_metadata)


def _quantile_bins(values: torch.Tensor, bins: int = 10) -> torch.Tensor:
    if len(values) == 1:
        return torch.zeros(1, dtype=torch.long, device=values.device)
    order = values.argsort().argsort()
    return torch.clamp(order * bins // len(values), max=bins - 1).long()


def _adversarial_loss(outputs: dict, graph, pairs: torch.Tensor, probes: nn.ModuleDict) -> torch.Tensor:
    degrees = torch.as_tensor([graph.degree(i) for i in range(graph.number_of_nodes())],
                              dtype=torch.float32, device=pairs.device)
    degree_values = torch.log1p(degrees[pairs[:, 0]]) + torch.log1p(degrees[pairs[:, 1]])
    cn_values = torch.as_tensor([
        len(set(graph.neighbors(int(u))) & set(graph.neighbors(int(v))))
        for u, v in pairs.detach().cpu().tolist()
    ], dtype=torch.float32, device=pairs.device)
    degree_bins, cn_bins = _quantile_bins(degree_values), _quantile_bins(cn_values)
    return sum([
        F.cross_entropy(probes["cn_to_degree"](outputs["z_cn"]), degree_bins),
        F.cross_entropy(probes["residual_to_degree"](outputs["z_residual"]), degree_bins),
        F.cross_entropy(probes["degree_to_cn"](outputs["z_degree"]), cn_bins),
        F.cross_entropy(probes["residual_to_cn"](outputs["z_residual"]), cn_bins),
    ])


def _build_optimizer(
    model: nn.Module,
    probes: nn.Module,
    config: TrainConfig,
) -> torch.optim.Optimizer:
    if isinstance(model, DCDLP):
        encoder_parameters = list(model.node_encoder.parameters())
        encoder_ids = {id(value) for value in encoder_parameters}
        head_parameters = [
            value for value in model.parameters()
            if id(value) not in encoder_ids
        ]
        groups = [
            {"params": encoder_parameters, "lr": config.lr, "name": "encoder"},
            {"params": head_parameters, "lr": config.lr, "name": "heads"},
            {"params": list(probes.parameters()), "lr": config.lr, "name": "probes"},
        ]
    else:
        groups = [{
            "params": list(model.parameters()) + list(probes.parameters()),
            "lr": config.lr,
            "name": "model_and_probes",
        }]
    return torch.optim.AdamW(
        groups, lr=config.lr, weight_decay=config.weight_decay
    )


def _enter_routing_finetune_stage(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> None:
    if not isinstance(model, DCDLP) or not config.score_routing_enabled:
        return
    if config.routing_finetune_encoder not in {"frozen", "low_lr"}:
        raise ValueError(
            "routing_finetune_encoder must be frozen or low_lr"
        )
    if config.encoder_lr_multiplier < 0:
        raise ValueError("encoder_lr_multiplier must be non-negative")
    frozen = config.routing_finetune_encoder == "frozen"
    for parameter in model.node_encoder.parameters():
        parameter.requires_grad_(not frozen)
    for group in optimizer.param_groups:
        if group.get("name") == "encoder":
            group["lr"] = (
                0.0
                if frozen
                else config.lr * config.encoder_lr_multiplier
            )


@torch.no_grad()
def score_pairs(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    pairs: np.ndarray,
    batch_size: int = 0,
) -> dict[str, np.ndarray]:
    model.eval()
    pair_array = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if batch_size <= 0 or len(pair_array) <= batch_size:
        tensor_pairs = torch.as_tensor(
            pair_array, dtype=torch.long, device=x.device
        )
        outputs = model(
            x, edge_index, tensor_pairs, remove_target_edges=True
        )
        return {
            key: value.detach().cpu().numpy()
            for key, value in outputs.items()
        }
    chunks: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(pair_array), batch_size):
        tensor_pairs = torch.as_tensor(
            pair_array[start:start + batch_size],
            dtype=torch.long,
            device=x.device,
        )
        outputs = model(
            x, edge_index, tensor_pairs, remove_target_edges=True
        )
        for key, value in outputs.items():
            chunks.setdefault(key, []).append(value.detach().cpu().numpy())
    return {
        key: np.concatenate(values, axis=0)
        for key, values in chunks.items()
    }


def grouped_official_negatives(
    positives: np.ndarray,
    negatives: np.ndarray | None,
    all_positive: np.ndarray | None = None,
) -> np.ndarray:
    if negatives is None:
        raise RuntimeError(
            "Official evaluation candidates are missing; refusing to "
            "substitute Uniform negatives"
        )
    array = np.asarray(negatives)
    if (
        array.ndim != 3
        or array.shape[0] != len(positives)
        or array.shape[-1] != 2
    ):
        raise RuntimeError(
            "Official grouped candidates must have shape "
            f"[num_positives, num_negatives, 2], got {array.shape}; "
            "refusing to substitute Uniform negatives"
        )
    grouped = array.astype(np.int64, copy=False)
    normalized = np.sort(grouped, axis=-1)
    normalized_positives = np.sort(
        np.asarray(positives, dtype=np.int64), axis=1
    )
    if np.any(
        np.all(
            normalized
            == normalized_positives[:, None, :],
            axis=-1,
        )
    ):
        raise RuntimeError(
            "Official grouped candidates contain their corresponding target "
            "positive edge"
        )
    # Keep the official candidate tensor byte-for-byte apart from dtype
    # normalization.  Some benchmark candidate sets deliberately retain
    # edges observed in another split; filtering them here would silently
    # change the benchmark protocol.
    return grouped


def evaluate_split(
    model: DCDLP, dataset: GraphDataset, positives: np.ndarray, seed: int,
    negatives_per_positive: int, x: torch.Tensor, edge_index: torch.Tensor,
    negative_method: str = "uniform",
    official_negatives: np.ndarray | None = None,
    candidate_metadata: dict | None = None,
) -> tuple[dict, list[dict]]:
    if negative_method == "official":
        negatives = grouped_official_negatives(
            positives, official_negatives, dataset.all_positive
        )
    else:
        pool_count = max(
            negatives_per_positive * len(positives),
            negatives_per_positive,
        )
    if negative_method == "uniform":
        negative_pool = uniform_negative_sampling(dataset.num_nodes, dataset.all_positive, pool_count, seed)
    elif negative_method == "degree_corrected":
        negative_pool = degree_corrected_sampling(dataset.train_graph(), dataset.all_positive, pool_count, seed)
    elif negative_method != "official":
        raise ValueError(f"Unknown evaluation negative method: {negative_method}")
    if negative_method != "official":
        negatives = grouped_negatives(
            positives, negative_pool, negatives_per_positive, seed + 1
        )
    if candidate_metadata is not None:
        candidate_metadata.update({
            "source": (
                "dataset_official_grouped"
                if negative_method == "official"
                else f"generated_{negative_method}"
            ),
            "hash": array_hash(negatives),
            "shape": list(np.asarray(negatives).shape),
            "seed": None if negative_method == "official" else int(seed),
        })
    positive_output = score_pairs(model, x, edge_index, positives)
    flat_negatives = negatives.reshape(-1, 2)
    negative_output = score_pairs(
        model,
        x,
        edge_index,
        flat_negatives,
        batch_size=8192 if negative_method == "official" else 0,
    )
    neg_scores = negative_output["logit"].reshape(
        len(positives), negatives.shape[1]
    )
    metrics = ranking_metrics(positive_output["logit"], neg_scores)
    labels = np.concatenate([np.ones(len(positives)), np.zeros(len(flat_negatives))])
    all_scores = np.concatenate([positive_output["logit"], negative_output["logit"]])
    metrics.update(classification_metrics(labels, all_scores))

    graph = dataset.train_graph()
    train_pairs = np.vstack([dataset.train_pos, _sample_train_negatives(dataset, TrainConfig(seed=seed), 0)])
    train_stats = pair_features(graph, train_pairs, dataset.features)
    current_stats = pair_features(graph, positives, dataset.features)
    try:
        regressor = getattr(
            getattr(model, "cn_branch", None), "cn_regressor", None
        )
        if regressor is None:
            regressor = ConditionalCNRegressor(seed, symmetric=True).fit(
                train_stats,
                metadata={
                    "data_source": "legacy-evaluation-train-only",
                    "uses_valid_or_test": False,
                },
            )
        _, train_residual = regressor.residual(train_stats)
        expected, residual = regressor.residual(current_stats)
    except (ImportError, ValueError):
        expected = np.full(len(positives), np.log1p(train_stats["cn"]).mean())
        residual = np.log1p(current_stats["cn"]) - expected
        train_residual = np.log1p(train_stats["cn"]) - np.log1p(train_stats["cn"]).mean()
    degree_threshold = float(np.median(train_stats["degree_score"]))
    groups = assign_quadrants(current_stats["degree_score"], residual, degree_threshold)
    metrics.update(group_ranking_metrics(positive_output["logit"], neg_scores, groups))
    reciprocal = 1.0 / (1 + (neg_scores > positive_output["logit"][:, None]).sum(axis=1))
    rows = []
    for index, (u, v) in enumerate(positives):
        model_cn_expected = positive_output.get("cn_expected")
        model_cn_residual = positive_output.get("cn_residual_feature")
        model_cn_normalized = positive_output.get("cn_normalized")
        rows.append({
            "u": int(u), "v": int(v), "label": 1,
            "score_total": float(positive_output["logit"][index]),
            "score_degree": float(positive_output["score_degree"][index]),
            "score_cn": float(positive_output["score_cn"][index]),
            "score_residual": float(positive_output["score_residual"][index]),
            "score_interaction": float(
                positive_output["score_interaction"][index]
            ),
            "interaction_share": float(
                positive_output.get(
                    "interaction_share",
                    np.zeros(len(positives), dtype=float),
                )[index]
            ),
            "degree_u": int(current_stats["degree_u"][index]),
            "degree_v": int(current_stats["degree_v"][index]),
            "degree_score": float(current_stats["degree_score"][index]),
            "cn_raw": int(current_stats["cn"][index]),
            # Keep the old alias for historical table consumers.
            "cn": int(current_stats["cn"][index]),
            "cn_expected": float(expected[index]),
            "cn_residual": float(residual[index]), "quadrant": str(groups[index]),
            "model_cn_expected": (
                float(model_cn_expected[index])
                if model_cn_expected is not None else np.nan
            ),
            "model_cn_residual": (
                float(model_cn_residual[index])
                if model_cn_residual is not None else np.nan
            ),
            "model_cn_normalized": (
                float(model_cn_normalized[index])
                if model_cn_normalized is not None else np.nan
            ),
            "cn_feature_mode": getattr(model, "cn_feature_mode", "raw"),
            "interaction_mode": getattr(
                model, "interaction_mode", "disabled"
            ),
            "rank": float(1.0 / reciprocal[index]),
        })
    return metrics, rows


def train_model(dataset: GraphDataset, config: TrainConfig) -> tuple[nn.Module, dict]:
    seed_everything(config.seed)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else
                          "cpu" if config.device == "auto" else config.device)
    x = torch.as_tensor(dataset.features, dtype=torch.float32, device=device)
    graph = dataset.train_graph()
    edge_index = edge_index_from_graph(graph, device)
    profile = ablation_profile(config.ablation)
    if "unsupported" in profile:
        raise RuntimeError(profile["unsupported"])
    if config.prediction_subdir not in {"raw", "predictions"}:
        raise ValueError("prediction_subdir must be raw or predictions")
    cn_regressor, cn_feature_metadata = fit_training_cn_residualizer(
        dataset, config
    )
    if profile["model"] == "gae":
        model = GAE(x.shape[1], config.hidden_dim, config.backbone, config.branch_dim).to(device)
    else:
        model = DCDLP(
            x.shape[1], config.hidden_dim, config.branch_dim, config.num_layers,
            config.dropout, config.backbone, use_interaction=profile["interaction"],
            active_branches=profile["active"], decoder_mode=profile["decoder"],
            cn_feature_mode=config.cn_feature_mode,
            cn_regressor=cn_regressor,
            interaction_mode=config.interaction_mode,
            cn_input_schema=config.cn_input_schema,
        ).to(device)
    probes = nn.ModuleDict({
        "cn_to_degree": AdversarialProbe(config.branch_dim),
        "residual_to_degree": AdversarialProbe(config.branch_dim),
        "degree_to_cn": AdversarialProbe(config.branch_dim),
        "residual_to_cn": AdversarialProbe(config.branch_dim),
    }).to(device)
    if config.lambda_link <= 0:
        raise ValueError("lambda_link must remain positive")
    optimizer = _build_optimizer(model, probes, config)
    best_state, best_mrr, best_epoch = None, -1.0, -1
    started = time.perf_counter()
    total_epochs = config.pretrain_epochs + config.disentangle_epochs
    forbidden = {tuple(map(int, item)) for item in np.vstack([dataset.valid_pos, dataset.test_pos]).tolist()}
    rng = np.random.default_rng(config.seed)
    failure_counts: dict[str, int] = {}
    evaluation_negative_method = (
        "official"
        if config.protocol_eval in {"heart", "ogb"}
        else "uniform"
    )
    for epoch in range(total_epochs):
        if epoch == config.pretrain_epochs:
            _enter_routing_finetune_stage(model, optimizer, config)
        model.train()
        probes.train()
        interventions_this_epoch = 0
        negative = _sample_train_negatives(dataset, config, epoch)
        pairs_np = np.vstack([dataset.train_pos, negative])
        labels_np = np.concatenate([np.ones(len(dataset.train_pos)), np.zeros(len(negative))])
        order = rng.permutation(len(pairs_np))
        for start in range(0, len(order), config.batch_size):
            selected = order[start:start + config.batch_size]
            pairs = torch.as_tensor(pairs_np[selected], dtype=torch.long, device=device)
            labels = torch.as_tensor(labels_np[selected], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, edge_index, pairs, remove_target_edges=True)
            loss = config.lambda_link * link_prediction_loss(
                outputs["logit"], labels
            )
            if epoch >= config.pretrain_epochs and "orth" in profile["losses"]:
                loss = loss + config.lambda_orth * orthogonal_loss(
                    outputs["z_degree"], outputs["z_cn"], outputs["z_residual"]
                )
            if epoch >= config.pretrain_epochs and "adv" in profile["losses"]:
                loss = loss + config.lambda_adv * _adversarial_loss(outputs, graph, pairs, probes)
            if epoch >= config.pretrain_epochs:
                interaction_share_loss, _ = interaction_regularization(
                    outputs,
                    mode=config.interaction_mode,
                    share_cap=config.interaction_share_cap,
                    delta_cap=config.interaction_delta_cap,
                )
                loss = (
                    loss
                    + config.lambda_interaction_share
                    * interaction_share_loss
                )
            uses_interventions = (
                "intervention" in profile["losses"]
                or config.score_routing_enabled
            )
            if epoch >= config.pretrain_epochs and uses_interventions:
                intervention_count = max(1, int(len(pairs) * config.intervention_ratio))
                if config.max_interventions_per_batch > 0:
                    intervention_count = min(intervention_count, config.max_interventions_per_batch)
                if config.max_interventions_per_epoch > 0:
                    remaining = config.max_interventions_per_epoch - interventions_this_epoch
                    intervention_count = min(intervention_count, max(remaining, 0))
                if intervention_count > 0:
                    intervention_indices = rng.choice(
                        len(pairs), size=min(intervention_count, len(pairs)), replace=False
                    )
                    interventions_this_epoch += len(intervention_indices)
                    intervention_terms = []
                    for local_index in intervention_indices:
                        u, v = map(int, pairs[local_index].detach().cpu().tolist())
                        mode = profile["interventions"]
                        kind = mode if mode in {"cn", "degree"} else ("cn" if local_index % 2 == 0 else "degree")
                        try:
                            if kind == "cn":
                                cf_graph, _ = intervene_cn(graph, u, v, 1, config.seed + epoch * 1009 + int(local_index), forbidden)
                            else:
                                cf_graph, _ = intervene_degree(graph, u, v, "u", 1, config.seed + epoch * 1009 + int(local_index), forbidden)
                            one_pair = pairs[local_index:local_index + 1]
                            original_output = {key: value[local_index:local_index + 1] for key, value in outputs.items()}
                            cf_output = model(x, edge_index_from_graph(cf_graph, device), one_pair, remove_target_edges=False)
                            legacy_term = loss.new_tensor(0.0)
                            if "intervention" in profile["losses"]:
                                invariant, route, robust = intervention_losses(
                                    original_output, cf_output, kind
                                )
                                robust_weight = (
                                    0.0
                                    if profile.get("no_degrob")
                                    else config.lambda_degrob
                                )
                                legacy_term = (
                                    config.lambda_inv * invariant
                                    + config.lambda_route * route
                                    + robust_weight * robust
                                )
                            score_route_term = loss.new_tensor(0.0)
                            if config.score_routing_enabled:
                                route_score, target_response = score_routing_losses(
                                    original_output,
                                    cf_output,
                                    kind,
                                    margin_cn=config.routing_margin_cn,
                                    margin_degree=config.routing_margin_degree,
                                    target_floor_cn=config.target_floor_cn,
                                    target_floor_degree=config.target_floor_degree,
                                    interaction_route_weight=(
                                        config.interaction_route_weight
                                    ),
                                )
                                score_route_term = (
                                    config.lambda_score_route * route_score
                                    + config.lambda_target_response
                                    * target_response
                                )
                            _, interaction_delta_loss = interaction_regularization(
                                original_output,
                                cf_output,
                                mode=config.interaction_mode,
                                share_cap=config.interaction_share_cap,
                                delta_cap=config.interaction_delta_cap,
                            )
                            intervention_terms.append(
                                legacy_term
                                + score_route_term
                                + config.lambda_interaction_delta
                                * interaction_delta_loss
                            )
                        except InterventionError as exc:
                            failure_counts[exc.code] = failure_counts.get(exc.code, 0) + 1
                    if intervention_terms:
                        loss = loss + torch.stack(intervention_terms).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        valid_metrics, _ = evaluate_split(
            model, dataset, dataset.valid_pos, config.seed + 777,
            config.negatives_per_positive_eval, x, edge_index,
            negative_method=evaluation_negative_method,
            official_negatives=dataset.valid_neg,
        )
        if valid_metrics["mrr"] > best_mrr:
            best_mrr, best_epoch = valid_metrics["mrr"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    runtime = time.perf_counter() - started
    test_candidate_metadata: dict[str, object] = {}
    test_metrics, prediction_rows = evaluate_split(
        model, dataset, dataset.test_pos, config.seed + 999,
        config.negatives_per_positive_eval, x, edge_index,
        negative_method=evaluation_negative_method,
        official_negatives=dataset.test_neg,
        candidate_metadata=test_candidate_metadata,
    )
    output_root = project_root() / config.output_dir
    run_id = f"{dataset.name}_{config.protocol_train}_seed{config.seed}_{stable_hash(asdict(config))}"
    checkpoint = output_root / "checkpoints" / f"{run_id}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": asdict(config),
        "input_dim": x.shape[1],
        "cn_regressor": (
            model.cn_branch.cn_regressor
            if isinstance(model, DCDLP) else None
        ),
        "cn_feature_metadata": cn_feature_metadata,
    }, checkpoint)
    predictions = (
        output_root / config.prediction_subdir
        / f"{run_id}_predictions.csv"
    )
    predictions.parent.mkdir(parents=True, exist_ok=True)
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]) if prediction_rows else [])
        if prediction_rows:
            writer.writeheader()
            writer.writerows(prediction_rows)
    result = {
        "dataset": dataset.name, "protocol_train": config.protocol_train,
        "protocol_eval": config.protocol_eval, "model": "gae" if config.ablation == "A0" else "dcdlp",
        "ablation": config.ablation, "seed": config.seed,
        # Keep the mechanism configuration at the top level so aggregation
        # and remote audits cannot accidentally combine raw/residual or
        # audited/disabled runs into one mean.
        "cn_feature_mode": config.cn_feature_mode,
        "interaction_mode": config.interaction_mode,
        "score_routing_enabled": config.score_routing_enabled,
        "routing_finetune_encoder": config.routing_finetune_encoder,
        "config_hash": stable_hash(asdict(config)), "git_commit": git_commit(project_root()),
        "best_epoch": best_epoch, "metrics": test_metrics,
        "runtime": {"train_seconds": runtime, "inference_seconds": 0.0,
                    "peak_gpu_mb": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0},
        "checkpoint": str(checkpoint), "predictions": str(predictions),
        "intervention_failures": failure_counts,
        "cn_feature_metadata": cn_feature_metadata,
        "evaluation_candidates": {
            **test_candidate_metadata,
            "protocol": config.protocol_eval,
            "negatives_per_positive": int(
                np.asarray(dataset.test_neg).shape[1]
                if evaluation_negative_method == "official"
                else config.negatives_per_positive_eval
            ),
        },
        "data_integrity": {
            "split_hash": stable_hash({
                "train": array_hash(dataset.train_pos),
                "valid": array_hash(dataset.valid_pos),
                "test": array_hash(dataset.test_pos),
            }),
            "train_positive_hash": array_hash(dataset.train_pos),
            "valid_positive_hash": array_hash(dataset.valid_pos),
            "test_positive_hash": array_hash(dataset.test_pos),
            "candidate_hash": test_candidate_metadata.get("hash"),
        },
        "intervention_cache": {
            "source": "deterministic_online_generation",
            "id": stable_hash({
                "test_positive_hash": array_hash(dataset.test_pos),
                "seed": config.seed,
                "types": ["cn_plus_1", "degree_u_plus_1"],
                "validator": "strict",
            }),
        },
        "parameter_count": {
            "total": int(sum(value.numel() for value in model.parameters())),
            "trainable": int(sum(
                value.numel() for value in model.parameters()
                if value.requires_grad
            )),
            "cn_branch": int(sum(
                value.numel()
                for value in getattr(model, "cn_branch", nn.Module()).parameters()
            )),
        },
        "loss_weights": {
            "lambda_link": config.lambda_link,
            "lambda_score_route": config.lambda_score_route,
            "lambda_target_response": config.lambda_target_response,
            "lambda_interaction_share": config.lambda_interaction_share,
            "lambda_interaction_delta": config.lambda_interaction_delta,
            "lambda_orthogonal_legacy": config.lambda_orth,
            "lambda_adversarial_legacy": config.lambda_adv,
        },
        "device": str(device),
    }
    result_path = output_root / "raw" / f"{run_id}.json"
    write_json(result_path, result)
    result["result_file"] = str(result_path)
    return model, result
