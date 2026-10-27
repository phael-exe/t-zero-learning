"""Rede Q totalmente conectada."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for size in hidden_sizes:
            layers += [nn.Linear(last, size), nn.ReLU()]
            last = size
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
