import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """State-action value MLP for discrete actions (CleanRL-style DQN network).

    Outputs one Q-value per discrete action; obs normalization is not
    supported (DQN here follows CleanRL and learns from raw observations).

    Constructor args (set through a config's ``network_kwargs``):
        activation: hidden activation, an attribute name on ``torch.nn``
            (e.g. ReLU, Tanh, GELU, SiLU).
        hidden_layers_size: width of each of the two hidden layers.
    """

    def __init__(self, envs, activation: str = "ReLU", hidden_layers_size: int = 120):
        super().__init__()
        h = int(hidden_layers_size)
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act = getattr(nn, activation)
        self.network = nn.Sequential(
            nn.Linear(obs_dim, h),
            act(),
            nn.Linear(h, h),
            act(),
            nn.Linear(h, int(envs.single_action_space.n)),
        )

    def forward(self, x):
        return self.network(x)

    def act(self, x, deterministic: bool = True):
        """Greedy action selection (uniform eval policy interface).

        The greedy policy is inherently deterministic; the flag is accepted
        for interface compatibility and ignored.
        """
        return torch.argmax(self.network(x), dim=1)
