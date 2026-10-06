"""Categorical actor-critic MLP for discrete actions (A2C assignment network).

ASSIGNMENT (Part 3): complete ``DiscreteActorCritic.get_action_and_value``.
Check your work with:   python -m pytest tests/test_a2c.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from networks.actor_critic_network import layer_init


class DiscreteActorCritic(nn.Module):
    """Separate actor (logits over actions) and critic (V(s)) MLPs.

    The actor outputs one **logit** per discrete action; the policy is
    ``Categorical(logits=...)``. The critic outputs a scalar state value.
    Observation normalization is not supported (raw observations, like DQN).

    Constructor args (set through a config's ``network_kwargs``):
        activation: hidden activation, an attribute name on ``torch.nn``
            (e.g. Tanh, ReLU, GELU, SiLU).
        hidden_layers_size: width of each hidden layer in the actor and critic MLPs.
    """

    def __init__(self, envs, activation: str = "Tanh", hidden_layers_size: int = 64):
        super().__init__()
        h = int(hidden_layers_size)
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        n_actions = int(envs.single_action_space.n)
        act = getattr(nn, activation)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h)),
            act(),
            layer_init(nn.Linear(h, h)),
            act(),
            layer_init(nn.Linear(h, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h)),
            act(),
            layer_init(nn.Linear(h, h)),
            act(),
            layer_init(nn.Linear(h, n_actions), std=0.01),
        )

    def get_value(self, x):
        """State value V(s), shape (B, 1)."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic: bool = False):
        """Policy distribution + value for a batch of observations.

        Args:
            x: observations, shape (B, obs_dim).
            action: if given (shape (B,), int64), evaluate log_prob of *these*
                actions instead of sampling new ones.
            deterministic: when sampling, take the most probable action
                (argmax of the logits) instead of drawing from the distribution.

        Returns ``(action, log_prob, entropy, value)`` with shapes
        (B,) int64, (B,), (B,), (B, 1).

        Hint: ``torch.distributions.Categorical(logits=...)`` gives you
        ``sample()``, ``log_prob(action)`` and ``entropy()``.
        """
        logits = self.actor(x)
        dist = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.get_value(x)

        return action, log_prob, entropy, value
        
    def act(self, x, deterministic: bool = False):
        """Action selection for evaluation (uniform policy interface)."""
        action, _, _, _ = self.get_action_and_value(x, deterministic=deterministic)
        return action
