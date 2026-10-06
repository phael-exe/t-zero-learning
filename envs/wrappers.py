"""Preprocessing wrapper stacks applied between ``gym.make`` and the agent.

A wrapper stack is a callable ``(env, env_id, gamma) -> env`` (*gamma* is the
algorithm's discount factor — needed by reward normalization; stacks that
don't normalize rewards just ignore it).  Which stack an env gets is resolved
per env in :func:`envs.factory.make_env`:

1. the env family's adapter override (``EnvAdapter.apply_wrappers``), if any
   — a permanent fact about the env family (rare);
2. otherwise the ``env_wrappers:`` config field, a name looked up in
   :data:`WRAPPER_STACKS` below — a per-run experimental choice;
3. otherwise the algorithm's ``default_wrappers`` — the input contract its
   standard agent assumes.

There is deliberately no further fallback: if none of the three is set, env
construction fails loudly.

Note that :func:`continuous_control_wrappers` is **not** env-agnostic: it is
the input contract of the flat-vector MLP agents used by
``ppo_continuous_action`` (flatten, clip actions, normalize + clip rewards).
Observation normalization deliberately does **not** live here: it is owned by
the agent (``network_kwargs.use_obs_norm`` on ``ContinuousActorCritic``), so its
statistics are checkpointed with the weights — see :mod:`networks.normalization`.  A new
algorithm family (pixel observations / CNN encoders, discrete actions) should
declare its own stack here, add it to :data:`WRAPPER_STACKS`, and set it as
its ``default_wrappers`` — do not extend an existing stack with special cases.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.adapters import get_adapter


def _validate_continuous_control_spaces(env: gym.Env, env_id: str) -> None:
    """Fail loudly when an env does not fit this stack's assumptions.

    Without this check, image observations would be silently flattened into a
    meaningless vector and discrete action spaces would crash deep inside
    ``ClipAction`` with an unhelpful message.
    """
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) >= 2:
        raise TypeError(
            f"{env_id}: observation space {obs_space} looks like image data. "
            "The continuous-control preprocessing stack (flatten to a vector) "
            "would corrupt it. Pixel envs need a CNN agent and "
            "their own wrapper stack — see docs/adding-a-new-environment.md."
        )
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError(
            f"{env_id}: action space {env.action_space} is not a continuous "
            "Box. This stack (and ppo_continuous_action's Gaussian policy) "
            "assumes continuous actions. Discrete envs need their own "
            "algorithm and wrapper stack — see "
            "docs/adding-a-new-environment.md."
        )


def continuous_control_wrappers(env: gym.Env, env_id: str, gamma: float) -> gym.Env:
    """Preprocessing stack for flat-observation continuous-control agents.

    Pipeline (in order):
        1. FlattenObservation — handles Dict observation spaces (e.g. dm_control)
        2. RecordEpisodeStatistics — skipped for envs that apply it internally;
           sits *before* the reward transforms so episodic returns stay raw
        3. ClipAction — clips actions to the action space bounds
        4. NormalizeReward — scales rewards by a running std of the discounted
           return (CleanRL's training-signal shaping; *gamma* is the
           algorithm's discount factor). Its running statistic is env state
           and is **not** checkpointed — resumed runs rebuild it from scratch
        5. TransformReward — clips the normalized reward to [-10, 10]

    Observations are *not* clipped here: obs normalization + clipping is the
    agent's job when ``network_kwargs.use_obs_norm`` is enabled (the clip belongs
    after normalization, and the stats belong in the checkpoint).
    """
    _validate_continuous_control_spaces(env, env_id)

    # TODO: remove FlattenObservation once dm_control Dict obs is no longer used
    env = gym.wrappers.FlattenObservation(env)

    if not get_adapter(env_id).skip_episode_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)

    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env


def no_wrappers(env: gym.Env, env_id: str, gamma: float) -> gym.Env:
    """Apply no preprocessing at all — the agent sees the raw env.

    Note: without ``RecordEpisodeStatistics`` the training loop will not see
    episodic returns unless the env records them itself.
    """
    return env


def discrete_control_wrappers(env: gym.Env, env_id: str, gamma: float) -> gym.Env:
    """Preprocessing stack for flat-observation discrete-action agents (e.g. DQN).

    Pipeline: FlattenObservation + RecordEpisodeStatistics. No reward
    transforms — value-based methods bootstrap on raw rewards (*gamma* is
    unused). Fails loudly on non-Discrete action spaces and image obs.
    """
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) >= 2:
        raise TypeError(
            f"{env_id}: observation space {obs_space} looks like image data. "
            "This stack flattens obs to a vector; pixel envs need a CNN agent "
            "and their own stack — see docs/adding-a-new-environment.md."
        )
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError(
            f"{env_id}: action space {env.action_space} is not Discrete. "
            "This stack is for value-based discrete-action agents (DQN)."
        )

    env = gym.wrappers.FlattenObservation(env)
    if not get_adapter(env_id).skip_episode_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


# ---------------------------------------------------------------------------
# Named stacks, selectable per run via the ``env_wrappers:`` config field.
# New algorithm families register their stack here.
# ---------------------------------------------------------------------------
WRAPPER_STACKS = {
    "continuous_control": continuous_control_wrappers,
    "discrete_control": discrete_control_wrappers,
    "none": no_wrappers,
}


def resolve_wrapper_stack(name, default):
    """Return the stack for config *name*, or *default* when *name* is unset.

    Raises :class:`ValueError` for unknown names, listing the options.
    """
    if not name:
        return default
    try:
        return WRAPPER_STACKS[name]
    except KeyError:
        raise ValueError(
            f"unknown env_wrappers name {name!r}; available: "
            f"{sorted(WRAPPER_STACKS)} (see envs/wrappers.py)"
        ) from None
