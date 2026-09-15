from __future__ import annotations

import torch
from torch.nn import functional as F


def link_prediction_loss(logits: torch.Tensor, labels: torch.Tensor, pos_weight: float | None = None):
    weight = None if pos_weight is None else logits.new_tensor(pos_weight)
    return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=weight)


def orthogonal_loss(*representations: torch.Tensor) -> torch.Tensor:
    loss = representations[0].new_tensor(0.0)
    normalized = []
    for value in representations:
        value = value - value.mean(0, keepdim=True)
        value = value / value.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
        normalized.append(value)
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            covariance = normalized[i].t() @ normalized[j] / max(normalized[i].shape[0] - 1, 1)
            loss = loss + covariance.square().mean()
    return loss


def interaction_share(
    score_degree: torch.Tensor,
    score_cn: torch.Tensor,
    score_residual: torch.Tensor,
    score_interaction: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return the per-example share of the additive interaction score."""
    denominator = (
        score_degree.abs()
        + score_cn.abs()
        + score_residual.abs()
        + score_interaction.abs()
    )
    return score_interaction.abs() / (denominator + epsilon)


def interaction_regularization(
    original: dict[str, torch.Tensor],
    counterfactual: dict[str, torch.Tensor] | None = None,
    *,
    mode: str,
    share_cap: float = 0.2,
    delta_cap: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return audited share and intervention-delta penalties.

    Unrestricted interaction remains an unregularized performance control;
    disabled interaction is exactly zero in the model and is not regularized.
    """
    if mode not in {"disabled", "audited", "unrestricted"}:
        raise ValueError(
            "mode must be disabled, audited, or unrestricted"
        )
    zero = original["logit"].new_tensor(0.0)
    if mode != "audited":
        return zero, zero
    share = original.get("interaction_share")
    if share is None:
        share = interaction_share(
            original["score_degree"],
            original["score_cn"],
            original["score_residual"],
            original["score_interaction"],
        )
    share_loss = F.relu(share - float(share_cap)).mean()
    if counterfactual is None or delta_cap is None:
        delta_loss = zero
    else:
        delta = (
            counterfactual["score_interaction"]
            - original["score_interaction"]
        ).abs()
        delta_loss = F.relu(delta - float(delta_cap)).mean()
    return share_loss, delta_loss


def score_routing_losses(
    original: dict[str, torch.Tensor],
    counterfactual: dict[str, torch.Tensor],
    kind: str,
    *,
    margin_cn: float = 0.0,
    margin_degree: float = 0.0,
    target_floor_cn: float = 0.0,
    target_floor_degree: float = 0.0,
    interaction_route_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directly supervise selective branch-score response to an intervention."""
    if kind not in {"cn", "degree"}:
        raise ValueError("kind must be cn or degree")
    deltas = {
        name: (
            counterfactual[f"score_{name}"]
            - original[f"score_{name}"]
        ).abs()
        for name in ("degree", "cn", "residual", "interaction")
    }
    if kind == "cn":
        target = deltas["cn"]
        cross = (
            deltas["degree"]
            + deltas["residual"]
            + float(interaction_route_weight) * deltas["interaction"]
        )
        margin = margin_cn
        floor = target_floor_cn
    else:
        target = deltas["degree"]
        cross = (
            deltas["cn"]
            + deltas["residual"]
            + float(interaction_route_weight) * deltas["interaction"]
        )
        margin = margin_degree
        floor = target_floor_degree
    route_loss = F.relu(float(margin) + cross - target).mean()
    target_loss = F.relu(float(floor) - target).mean()
    return route_loss, target_loss


def intervention_losses(original: dict, counterfactual: dict, kind: str, margin: float = 0.2):
    distances = {
        # torch.linalg.vector_norm defines a finite zero gradient when the two
        # representations are identical.  The equivalent hand-written
        # sqrt(sum(x ** 2)) has an undefined derivative at zero and poisoned
        # the following optimizer step with NaNs.
        key: torch.linalg.vector_norm(
            original[f"z_{key}"] - counterfactual[f"z_{key}"], dim=-1
        ).mean()
        for key in ("degree", "cn", "residual")
    }
    if kind == "cn":
        invariant = distances["degree"].square() + distances["residual"].square()
        route = F.relu(original["logit"].new_tensor(margin) - distances["cn"] + distances["degree"] + distances["residual"])
        degree_robustness = original["logit"].new_tensor(0.0)
    elif kind == "degree":
        invariant = distances["cn"].square() + distances["residual"].square()
        route = F.relu(original["logit"].new_tensor(margin) - distances["degree"] + distances["cn"] + distances["residual"])
        degree_robustness = F.smooth_l1_loss(original["logit"], counterfactual["logit"])
    else:
        raise ValueError("kind must be cn or degree")
    return invariant, route, degree_robustness
