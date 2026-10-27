"""Replay buffer circular que guarda terminated e truncated separadamente."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], seed: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.truncated = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        # next_obs precisa ser a observação FINAL real do episódio,
        # nunca a observação de reset (ver scripts/sanity_checks.py).
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.terminated[i] = float(terminated)
        self.truncated[i] = float(truncated)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return Batch(
            obs=torch.from_numpy(self.obs[idx]),
            actions=torch.from_numpy(self.actions[idx]),
            rewards=torch.from_numpy(self.rewards[idx]),
            next_obs=torch.from_numpy(self.next_obs[idx]),
            terminated=torch.from_numpy(self.terminated[idx]),
            truncated=torch.from_numpy(self.truncated[idx]),
        )

    def __len__(self) -> int:
        return self.size
