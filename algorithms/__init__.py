"""Algorithm implementations.

Provides :class:`PPO` for continuous action spaces (canonical, single
optimizer) and a split actor/critic optimizer variant in
:mod:`algorithms.ppo_continuous_action_split_optim`.
New algorithms should inherit from :class:`Algorithm`.
"""

from core.base_config import RunConfig
from algorithms.base import Algorithm
from algorithms.ppo_continuous_action import PPOConfig, Args, PPO

__all__ = ["Algorithm", "RunConfig", "PPOConfig", "Args", "PPO"]
