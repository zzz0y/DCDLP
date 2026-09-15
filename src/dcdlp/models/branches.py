from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


CN_FEATURE_MODES = {"raw", "residual", "raw_plus_residual"}
CN_INPUT_SCHEMAS = {"legacy", "selective_v1"}


def mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim), nn.ReLU(),
    )


class DegreeBranch(nn.Module):
    def __init__(self, output_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = mlp(4, output_dim, output_dim, dropout)
        self.scorer = nn.Linear(output_dim, 1)

    def forward(self, degrees: torch.Tensor, pairs: torch.Tensor):
        du, dv = degrees[pairs[:, 0]], degrees[pairs[:, 1]]
        small, large = torch.minimum(du, dv), torch.maximum(du, dv)
        features = torch.stack([
            torch.log1p(small), torch.log1p(large),
            torch.log1p(small * large), torch.abs(torch.log1p(large) - torch.log1p(small)),
        ], dim=-1)
        z = self.encoder(features)
        return z, self.scorer(z).squeeze(-1)


class CNBranch(nn.Module):
    def __init__(
        self,
        node_dim: int,
        output_dim: int = 64,
        dropout: float = 0.3,
        use_aa_ra: bool = False,
        cn_feature_mode: str = "raw",
        cn_regressor: object | None = None,
        cn_input_schema: str = "selective_v1",
    ) -> None:
        super().__init__()
        if cn_feature_mode not in CN_FEATURE_MODES:
            raise ValueError(
                f"Unknown cn_feature_mode {cn_feature_mode!r}; "
                f"expected one of {sorted(CN_FEATURE_MODES)}"
            )
        self.node_mlp = mlp(node_dim, output_dim, output_dim, dropout)
        self.empty_token = nn.Parameter(torch.zeros(output_dim))
        self.use_aa_ra = use_aa_ra
        self.cn_feature_mode = cn_feature_mode
        self.cn_regressor = cn_regressor
        if cn_input_schema not in CN_INPUT_SCHEMAS:
            raise ValueError(
                f"Unknown cn_input_schema {cn_input_schema!r}; "
                f"expected one of {sorted(CN_INPUT_SCHEMAS)}"
            )
        self.cn_input_schema = cn_input_schema
        explicit_feature_count = (
            {
                "raw": 1,
                "residual": 2,
                "raw_plus_residual": 3,
            }[cn_feature_mode]
            if cn_input_schema == "legacy"
            else 2
        )
        self.encoder = mlp(
            output_dim * 2 + explicit_feature_count + (2 if use_aa_ra else 0),
            output_dim,
            output_dim,
            dropout,
        )
        self.scorer = nn.Linear(output_dim, 1)

    def set_cn_regressor(self, regressor: object | None) -> None:
        self.cn_regressor = regressor

    def forward(self, h: torch.Tensor, pairs: torch.Tensor, neighbors: list[set[int]], degrees: torch.Tensor):
        node_messages = self.node_mlp(h)
        pooled_rows = []
        raw_counts: list[int] = []
        endpoint_degrees: list[tuple[float, float]] = []
        heuristic_rows: list[torch.Tensor] = []
        for raw_u, raw_v in pairs.detach().cpu().tolist():
            u, v = int(raw_u), int(raw_v)
            common = sorted(neighbors[u] & neighbors[v])
            if common:
                indices = torch.as_tensor(common, device=h.device)
                messages = node_messages[indices]
                pooled_sum = messages.sum(0) / math.sqrt(len(common) + 1)
                pooled_max = messages.max(0).values
            else:
                pooled_sum = h.new_zeros(node_messages.shape[1])
                pooled_max = self.empty_token
            pooled_rows.append(torch.cat([pooled_sum, pooled_max]))
            raw_counts.append(len(common))
            endpoint_degrees.append(
                (float(degrees[u].detach().cpu()), float(degrees[v].detach().cpu()))
            )
            heuristics: list[torch.Tensor] = []
            if self.use_aa_ra:
                if common:
                    degree_common = degrees[torch.as_tensor(common, device=h.device)].clamp_min(1)
                    aa = (1.0 / degree_common.clamp_min(2).log()).sum()
                    ra = (1.0 / degree_common).sum()
                else:
                    aa = ra = h.new_tensor(0.0)
                heuristics.append(torch.stack([aa, ra]).reshape(-1))
            heuristic_rows.append(
                torch.cat(heuristics) if heuristics else h.new_empty(0)
            )

        raw_cn = h.new_tensor(raw_counts)
        raw_log_cn = torch.log1p(raw_cn)
        degree_array = np.asarray(endpoint_degrees, dtype=float)
        degree_product = h.new_tensor(degree_array[:, 0] * degree_array[:, 1])
        normalized = raw_cn / degree_product.clamp_min(1.0).sqrt()
        if self.cn_regressor is None:
            if self.cn_feature_mode != "raw":
                raise RuntimeError(
                    f"cn_feature_mode={self.cn_feature_mode!r} requires a fitted "
                    "training-split ConditionalCNRegressor"
                )
            expected = torch.full_like(raw_log_cn, float("nan"))
            residual = torch.full_like(raw_log_cn, float("nan"))
        else:
            expected_np = self.cn_regressor.predict(
                {
                    "degree_u": degree_array[:, 0],
                    "degree_v": degree_array[:, 1],
                    "cn": np.asarray(raw_counts, dtype=float),
                }
            )
            expected = h.new_tensor(np.asarray(expected_np, dtype=float))
            residual = raw_log_cn - expected

        if self.cn_input_schema == "legacy":
            explicit = {
                "raw": raw_log_cn[:, None],
                "residual": torch.stack([residual, normalized], dim=-1),
                "raw_plus_residual": torch.stack(
                    [raw_log_cn, residual, normalized], dim=-1
                ),
            }[self.cn_feature_mode]
        else:
            zeros = torch.zeros_like(raw_log_cn)
            # All selective modes use the same explicit width and therefore
            # the same CN-branch parameter budget.  In residual mode the raw
            # and degree-normalized raw counts are deliberately absent.
            explicit = {
                "raw": torch.stack([raw_log_cn, zeros], dim=-1),
                "residual": torch.stack([residual, zeros], dim=-1),
                "raw_plus_residual": torch.stack(
                    [raw_log_cn, residual], dim=-1
                ),
            }[self.cn_feature_mode]
        rows = [
            torch.cat([pooled, explicit_row, heuristics])
            for pooled, explicit_row, heuristics in zip(
                pooled_rows, explicit, heuristic_rows
            )
        ]
        z = self.encoder(torch.stack(rows))
        statistics = {
            "cn_raw": raw_cn,
            "cn_log_raw": raw_log_cn,
            "cn_expected": expected,
            "cn_residual_feature": residual,
            "cn_normalized": normalized,
        }
        return z, self.scorer(z).squeeze(-1), statistics


class ResidualBranch(nn.Module):
    def __init__(self, node_dim: int, output_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = mlp(node_dim * 3, output_dim, output_dim, dropout)
        self.scorer = nn.Linear(output_dim, 1)

    def forward(self, h: torch.Tensor, pairs: torch.Tensor):
        hu, hv = h[pairs[:, 0]], h[pairs[:, 1]]
        features = torch.cat([hu * hv, torch.abs(hu - hv), hu + hv], dim=-1)
        z = self.encoder(features)
        return z, self.scorer(z).squeeze(-1)
