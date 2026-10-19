"""SAC networks: a tanh-squashed Gaussian actor and twin soft Q-networks.

CleanRL's ``Actor`` and ``SoftQNetwork`` (``sac_continuous_action.py``),
grouped in one module so a config's ``network:`` key builds the whole agent
and ``model.pt`` holds all of it. The target Q-networks are not part of this
module: the algorithm makes them as copies.

Adapted from CleanRL (https://github.com/vwxyzjn/cleanrl),
Copyright (c) 2019 CleanRL developers, MIT License (see LICENSE).
"""
import numpy as np
import torch
import torch.nn as nn

LOG_STD_MAX = 2
LOG_STD_MIN = -5


def _mlp(in_dim: int, out_dim: int, hidden: int, act) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), act(),
        nn.Linear(hidden, hidden), act(),
        nn.Linear(hidden, out_dim),
    )


class SoftQNetwork(nn.Module):
    """Q(s, a): the observation and the action are concatenated at the input."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, act):
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, 1, hidden, act)

    def forward(self, x, a):
        return self.net(torch.cat([x, a], 1))


class Actor(nn.Module):
    """Outputs the mean and log-std of a Gaussian *before* the tanh squashing.

    ``log_std`` is squashed into ``[LOG_STD_MIN, LOG_STD_MAX]`` with a tanh
    (SpinUp / Denis Yarats). ``action_scale`` and ``action_bias`` map tanh's
    ``(-1, 1)`` onto the env's ``[low, high]``.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, act, low, high):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), act(), nn.Linear(hidden, hidden), act())
        self.fc_mean = nn.Linear(hidden, act_dim)
        self.fc_logstd = nn.Linear(hidden, act_dim)
        self.register_buffer("action_scale", torch.tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = self.trunk(x)
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std


class SACAgent(nn.Module):
    """Actor + twin soft Q-networks for SAC (continuous ``Box`` actions).

    Constructor args (set through a config's ``network_kwargs``):
        activation: hidden activation, an attribute name on ``torch.nn``.
        hidden_layers_size: width of the two hidden layers of every MLP.
    """

    def __init__(self, envs, activation: str = "ReLU", hidden_layers_size: int = 256):
        super().__init__()
        h = int(hidden_layers_size)
        act = getattr(nn, activation)
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        low = np.asarray(envs.single_action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(envs.single_action_space.high, dtype=np.float32).reshape(-1)
        self.actor = Actor(obs_dim, act_dim, h, act, low, high)
        self.qf1 = SoftQNetwork(obs_dim, act_dim, h, act)
        self.qf2 = SoftQNetwork(obs_dim, act_dim, h, act)

    def act(self, x, deterministic: bool = False):
        """Action selection for evaluation/inference: the squashed mean, or a
        squashed sample. The uniform policy interface used by
        ``evaluate_checkpoint``."""
        mean, log_std = self.actor(x)
        pre_tanh = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        return torch.tanh(pre_tanh) * self.actor.action_scale + self.actor.action_bias
