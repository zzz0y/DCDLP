from __future__ import annotations

import torch
from torch import nn


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float):
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.scale * gradient, None


def gradient_reverse(value: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(value, scale)


class AdversarialProbe(nn.Module):
    def __init__(self, input_dim: int = 64, bins: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, bins))

    def forward(self, representation: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        return self.network(gradient_reverse(representation, scale))

