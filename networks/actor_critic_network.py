import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from networks.normalization import ObsNormalizer


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# TODO: make obs normalization agent-gnostic so we do not need to repeat it.

class ContinuousActorCritic(nn.Module):
    """Gaussian actor–critic MLP for continuous actions (CleanRL-style PPO network).

    When ``use_obs_norm`` is true the agent owns a running observation
    normalizer (:class:`networks.normalization.ObsNormalizer`); its statistics
    are buffers in this module's ``state_dict``, so they are saved and restored
    with the weights.  The **algorithm** must drive it during rollout — call
    :meth:`update_norm` on each collected obs and act on normalized inputs
    (see docs/adding-a-new-algorithm.md).  Evaluation never updates the stats.

    Constructor args (set through a config's ``network_kwargs``):
        activation: hidden activation, an attribute name on ``torch.nn``
            (e.g. Tanh, ReLU, GELU, SiLU).
        hidden_layers_size: width of each hidden layer in the actor and critic MLPs.
        use_obs_norm: opt-in running mean/var observation normalization, meant
            for flat continuous observations; both PPO variants drive it.
        obs_norm_epsilon: numerical-stability epsilon in the normalization
            denominator (only used when ``use_obs_norm`` is true).
    """

    def __init__(
        self,
        envs,
        activation: str = "Tanh",
        hidden_layers_size: int = 64,
        use_obs_norm: bool = False,
        obs_norm_epsilon: float = 1e-8,
    ):
        super().__init__()
        h = int(hidden_layers_size)
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        act = getattr(nn, activation)
        self.use_obs_norm = bool(use_obs_norm)
        # Instantiated only when enabled: the state_dict of a norm-free agent
        # stays identical to the pre-normalization format.
        self.obs_normalizer = (
            ObsNormalizer(obs_dim=obs_dim, epsilon=obs_norm_epsilon)
            if self.use_obs_norm
            else None
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h)),
            act(),
            layer_init(nn.Linear(h, h)),
            act(),
            layer_init(nn.Linear(h, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h)),
            act(),
            layer_init(nn.Linear(h, h)),
            act(),
            layer_init(nn.Linear(h, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    # --- observation normalization (no-ops when use_obs_norm is false) ---

    @torch.no_grad()
    def update_norm(self, x):
        """Fold a batch of raw observations into the running statistics."""
        if self.use_obs_norm:
            self.obs_normalizer.update(x)

    def normalize_obs(self, x):
        """Normalize (and clip) raw observations; identity when disabled."""
        if not self.use_obs_norm:
            return x
        return self.obs_normalizer.normalize(x)

    # --- policy / value heads ---

    def get_value(self, x, input_is_normalized: bool = False):
        if not input_is_normalized:
            x = self.normalize_obs(x)
        return self.critic(x)

    def get_action_and_value(
        self, x, action=None, input_is_normalized: bool = False, deterministic: bool = False
    ):
        if not input_is_normalized:
            x = self.normalize_obs(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean if deterministic else probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def act(self, x, deterministic: bool = False):
        """Action selection for evaluation/inference (normalizes obs internally).

        The uniform policy interface used by ``evaluate_checkpoint`` — every
        agent network exposes ``act`` regardless of family (actor-critic,
        Q-network, ...).
        """
        action, _, _, _ = self.get_action_and_value(x, deterministic=deterministic)
        return action
