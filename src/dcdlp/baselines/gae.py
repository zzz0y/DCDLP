from __future__ import annotations

import torch
from torch import nn

from dcdlp.models.node_encoder import NodeEncoder, mask_pair_edges


class GAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, backbone: str = "gcn", branch_dim: int = 64) -> None:
        super().__init__()
        self.encoder = NodeEncoder(input_dim, hidden_dim, 2, 0.3, backbone)
        self.branch_dim = branch_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pairs: torch.Tensor, remove_target_edges: bool = True):
        message_edges = mask_pair_edges(edge_index, pairs) if remove_target_edges else edge_index
        h = self.encoder(x, message_edges)
        product = h[pairs[:, 0]] * h[pairs[:, 1]]
        logit = product.sum(-1)
        if product.shape[1] >= self.branch_dim:
            residual = product[:, :self.branch_dim]
        else:
            residual = torch.nn.functional.pad(product, (0, self.branch_dim - product.shape[1]))
        zeros = torch.zeros_like(residual)
        zero_scores = torch.zeros_like(logit)
        return {
            "logit": logit, "score_degree": zero_scores, "score_cn": zero_scores,
            "score_residual": logit, "score_interaction": zero_scores,
            "z_degree": zeros, "z_cn": zeros, "z_residual": residual,
        }
