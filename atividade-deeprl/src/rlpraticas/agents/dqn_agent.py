"""Agente DQN (com Double DQN opcional) escrito para leitura em aula."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rlpraticas.models.q_network import QNetwork
from rlpraticas.utils.replay_buffer import Batch


def compute_td_target(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    bootstrap_on_truncation: bool = True,
) -> torch.Tensor:
    """Alvo de Bellman y = r + γ · (1 − done) · V(s').

    terminated = 1: s' é terminal de verdade, então V(s') = 0 e não há bootstrap.
    truncated  = 1: o episódio foi cortado por limite de tempo; s' não é
                    terminal e o bootstrap continua.

    Com bootstrap_on_truncation=False reproduzimos o bug clássico do Gym antigo,
    em que `done = terminated or truncated` corta o bootstrap também no truncamento.
    """
    if bootstrap_on_truncation:
        done = terminated
    else:
        done = torch.maximum(terminated, truncated)
    return rewards + gamma * (1.0 - done) * next_values


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, hp: dict, seed: int):
        self.n_actions = n_actions
        self.gamma = float(hp["gamma"])
        self.double_dqn = bool(hp.get("double_dqn", True))
        self.bootstrap_on_truncation = bool(hp.get("bootstrap_on_truncation", True))
        self.max_grad_norm = float(hp.get("max_grad_norm", 10.0))

        self.q_net = QNetwork(obs_dim, n_actions, hp["hidden_sizes"])
        self.target_net = QNetwork(obs_dim, n_actions, hp["hidden_sizes"])
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(hp["learning_rate"]))
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return self.q_net(x).squeeze(0).numpy()

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """ε-greedy. Com epsilon=0 a política é determinística (avaliação)."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(obs)))

    def update(self, batch: Batch) -> dict[str, float]:
        with torch.no_grad():
            next_q_target = self.target_net(batch.next_obs)
            if self.double_dqn:
                # Double DQN: a rede online escolhe a ação, a rede alvo avalia.
                best = self.q_net(batch.next_obs).argmax(dim=1, keepdim=True)
                next_values = next_q_target.gather(1, best).squeeze(1)
            else:
                next_values = next_q_target.max(dim=1).values
            targets = compute_td_target(
                batch.rewards,
                next_values,
                batch.terminated,
                batch.truncated,
                self.gamma,
                self.bootstrap_on_truncation,
            )

        q_all = self.q_net(batch.obs)
        q_taken = q_all.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_taken, targets)  # Huber

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "q_mean": q_taken.mean().item(),
            "q_max": q_all.max().item(),
            "target_mean": targets.mean().item(),
            "grad_norm": float(grad_norm),
            "truncated_frac": batch.truncated.mean().item(),
        }

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: Path) -> None:
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.q_net.load_state_dict(state)
        self.target_net.load_state_dict(state)
