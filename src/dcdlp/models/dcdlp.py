from __future__ import annotations

import torch
from torch import nn

from .branches import CNBranch, DegreeBranch, ResidualBranch
from .losses import interaction_share
from .node_encoder import NodeEncoder, mask_pair_edges


class DCDLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 128, branch_dim: int = 64,
        num_layers: int = 2, dropout: float = 0.3, backbone: str = "gcn",
        use_interaction: bool = True, use_aa_ra: bool = False,
        active_branches: tuple[str, ...] = ("degree", "cn", "residual"),
        decoder_mode: str = "additive",
        cn_feature_mode: str = "raw",
        cn_regressor: object | None = None,
        interaction_mode: str = "unrestricted",
        cn_input_schema: str = "selective_v1",
    ) -> None:
        super().__init__()
        if interaction_mode not in {"unrestricted", "audited", "disabled"}:
            raise ValueError(
                "interaction_mode must be unrestricted, audited, or disabled"
            )
        self.node_encoder = NodeEncoder(input_dim, hidden_dim, num_layers, dropout, backbone)
        self.degree_branch = DegreeBranch(branch_dim, dropout)
        self.cn_branch = CNBranch(
            hidden_dim,
            branch_dim,
            dropout,
            use_aa_ra,
            cn_feature_mode=cn_feature_mode,
            cn_regressor=cn_regressor,
            cn_input_schema=cn_input_schema,
        )
        self.residual_branch = ResidualBranch(hidden_dim, branch_dim, dropout)
        self.active_branches = set(active_branches)
        self.decoder_mode = decoder_mode
        self.cn_feature_mode = cn_feature_mode
        self.cn_input_schema = cn_input_schema
        self.interaction_mode = (
            interaction_mode if use_interaction else "disabled"
        )
        self.concat_decoder = nn.Linear(branch_dim * 3, 1)
        self.interaction = nn.Parameter(torch.empty(branch_dim, branch_dim))
        nn.init.xavier_uniform_(self.interaction)
        interaction_enabled = self.interaction_mode != "disabled"
        self.interaction_scale = nn.Parameter(
            torch.tensor(0.1 if interaction_enabled else 0.0),
            requires_grad=interaction_enabled,
        )
        self.interaction.requires_grad_(interaction_enabled)
        self.bias = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _structure(edge_index: torch.Tensor, num_nodes: int):
        neighbors = [set() for _ in range(num_nodes)]
        for raw_u, raw_v in edge_index.detach().cpu().t().tolist():
            u, v = int(raw_u), int(raw_v)
            neighbors[u].add(v)
            neighbors[v].add(u)
        degrees = torch.as_tensor([len(item) for item in neighbors], dtype=torch.float32, device=edge_index.device)
        return neighbors, degrees

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, pairs: torch.Tensor,
        remove_target_edges: bool = True,
    ) -> dict[str, torch.Tensor]:
        message_edges = mask_pair_edges(edge_index, pairs) if remove_target_edges else edge_index
        h = self.node_encoder(x, message_edges)
        neighbors, degrees = self._structure(message_edges, x.shape[0])
        z_degree, score_degree = self.degree_branch(degrees, pairs)
        z_cn, score_cn, cn_statistics = self.cn_branch(
            h, pairs, neighbors, degrees
        )
        z_residual, score_residual = self.residual_branch(h, pairs)
        if "degree" not in self.active_branches:
            z_degree, score_degree = torch.zeros_like(z_degree), torch.zeros_like(score_degree)
        if "cn" not in self.active_branches:
            z_cn, score_cn = torch.zeros_like(z_cn), torch.zeros_like(score_cn)
        if "residual" not in self.active_branches:
            z_residual, score_residual = torch.zeros_like(z_residual), torch.zeros_like(score_residual)
        projected = z_degree @ self.interaction
        raw_interaction = (
            self.interaction_scale
            * (projected * z_cn).sum(-1)
            / z_cn.shape[-1] ** 0.5
        )
        score_interaction = (
            torch.zeros_like(raw_interaction)
            if self.interaction_mode == "disabled"
            else raw_interaction
        )
        if self.decoder_mode == "concat":
            logit = self.concat_decoder(torch.cat([z_degree, z_cn, z_residual], dim=-1)).squeeze(-1) + self.bias
        else:
            logit = score_degree + score_cn + score_residual + score_interaction + self.bias
        score_interaction_share = interaction_share(
            score_degree, score_cn, score_residual, score_interaction
        )
        return {
            "logit": logit, "score_degree": score_degree, "score_cn": score_cn,
            "score_residual": score_residual, "score_interaction": score_interaction,
            "interaction_share": score_interaction_share,
            "z_degree": z_degree, "z_cn": z_cn, "z_residual": z_residual,
            **cn_statistics,
        }
