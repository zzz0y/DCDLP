from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def undirected_with_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    reverse = edge_index.flip(0)
    loops = torch.arange(num_nodes, device=edge_index.device).repeat(2, 1)
    return torch.cat([edge_index, reverse, loops], dim=1)


def mask_pair_edges(edge_index: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Remove every undirected target pair from a message-passing edge list."""
    if pairs.numel() == 0 or edge_index.numel() == 0:
        return edge_index
    low = torch.minimum(edge_index[0], edge_index[1])
    high = torch.maximum(edge_index[0], edge_index[1])
    pair_low = torch.minimum(pairs[:, 0], pairs[:, 1])
    pair_high = torch.maximum(pairs[:, 0], pairs[:, 1])
    # Encoding an undirected pair as one integer changes the old O(E * B)
    # Python loop into a vectorized O(E + B) membership test.  The resulting
    # edge mask is identical, including for reversed or duplicate pairs.
    num_nodes = torch.maximum(high.max(), pair_high.max()) + 1
    edge_keys = low * num_nodes + high
    pair_keys = pair_low * num_nodes + pair_high
    return edge_index[:, ~torch.isin(edge_keys, pair_keys)]


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edges = undirected_with_loops(edge_index, x.shape[0])
        src, dst = edges
        degree = torch.bincount(dst, minlength=x.shape[0]).to(x.dtype).clamp_min(1)
        norm = degree[src].rsqrt() * degree[dst].rsqrt()
        messages = self.linear(x)[src] * norm.unsqueeze(-1)
        output = x.new_zeros((x.shape[0], messages.shape[1]))
        output.index_add_(0, dst, messages)
        return output + self.bias


class SAGEConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edges = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        src, dst = edges
        aggregate = x.new_zeros(x.shape)
        aggregate.index_add_(0, dst, x[src])
        counts = torch.bincount(dst, minlength=x.shape[0]).to(x.dtype).clamp_min(1).unsqueeze(-1)
        return self.linear(torch.cat([x, aggregate / counts], dim=-1))


class NodeEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
        dropout: float = 0.3, backbone: str = "gcn",
    ) -> None:
        super().__init__()
        if backbone not in {"gcn", "sage"}:
            raise ValueError("backbone must be gcn or sage")
        layer_type = GCNLayer if backbone == "gcn" else SAGEConv
        dims = [input_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([layer_type(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            x = norm(layer(x, edge_index))
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
