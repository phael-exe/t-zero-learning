"""Protocolo de avaliação: ambiente separado, sementes fixas, sem exploração."""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np


def evaluate(
    policy: Callable[[np.ndarray], int],
    env: gym.Env,
    n_episodes: int,
    seed: int,
    gamma: float,
    value_fn: Callable[[np.ndarray], float] | None = None,
) -> dict[str, float]:
    """Roda `n_episodes` episódios completos e devolve estatísticas.

    As sementes dos episódios são sempre seed, seed+1, ..., então duas
    avaliações em momentos diferentes do treino enfrentam os mesmos
    estados iniciais. Isso reduz a variância da curva de avaliação.

    Se `value_fn` for dada (ex.: max_a Q(s0, a) do DQN), também comparamos
    o valor previsto no estado inicial com o retorno descontado observado:
    a diferença é uma medida direta de superestimação.
    """
    returns, lengths, disc_returns, predicted = [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        if value_fn is not None:
            predicted.append(value_fn(obs))
        total, discounted, discount, length = 0.0, 0.0, 1.0, 0
        done = False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += float(reward)
            discounted += discount * float(reward)
            discount *= gamma
            length += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(length)
        disc_returns.append(discounted)

    stats = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_discounted_return": float(np.mean(disc_returns)),
    }
    if predicted:
        stats["mean_predicted_value"] = float(np.mean(predicted))
        stats["overestimation"] = stats["mean_predicted_value"] - stats["mean_discounted_return"]
    return stats
