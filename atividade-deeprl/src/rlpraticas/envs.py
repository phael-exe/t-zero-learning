"""Criação de ambientes Gymnasium de forma consistente entre treino e avaliação."""

from __future__ import annotations

import gymnasium as gym


def make_env(cfg: dict, seed: int, render_mode: str | None = None) -> gym.Env:
    """Mesmo construtor para treino e avaliação: wrappers idênticos nos dois."""
    kwargs = {}
    if cfg.get("max_episode_steps"):
        kwargs["max_episode_steps"] = int(cfg["max_episode_steps"])
    env = gym.make(cfg["env_id"], render_mode=render_mode, **kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env
