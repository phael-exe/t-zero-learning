"""Wrapper stacks: validation, reward normalization/clipping, and resolution order.

The reward tests use a deliberately out-of-range toy env — real envs
rarely produce rewards ≫ 10, so passing them through the stack would not
prove the normalize + clip pipeline is wired up.  Observations are *not*
clipped by the stack: obs normalization/clipping is agent-side (opt-in via
``network_kwargs.use_obs_norm`` — see tests/test_normalization.py).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from envs.factory import make_env
from envs.wrappers import (
    WRAPPER_STACKS,
    continuous_control_wrappers,
    resolve_wrapper_stack,
)


class _OutOfRangeEnv(gym.Env):
    """Returns observations/rewards far outside [-10, 10]."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([100.0, -100.0, 0.5]), {}

    def step(self, action):
        return np.array([50.0, -50.0, 1.0]), 1000.0, False, False, {}


class _ImageObsEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)


class _DiscreteActionEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float64)
    action_space = gym.spaces.Discrete(3)


# ---------------------------------------------------------------------------
# continuous_control_wrappers semantics
# ---------------------------------------------------------------------------

def test_rewards_normalized_and_clipped_obs_untouched():
    env = continuous_control_wrappers(_OutOfRangeEnv(), "Toy-v0", gamma=0.99)
    obs, _ = env.reset(seed=0)
    # Observations pass through raw — normalization/clipping is the agent's job.
    assert obs == pytest.approx([100.0, -100.0, 0.5])

    obs, reward, _, _, _ = env.step(env.action_space.sample())
    # NormalizeReward rescales the raw 1000.0; TransformReward clips to ±10.
    assert reward != pytest.approx(1000.0)
    assert abs(reward) <= 10.0


def test_reward_normalization_uses_gamma():
    """gamma reaches NormalizeReward (different discounts → different scaling)."""

    def rewards_for(gamma: float) -> list[float]:
        env = continuous_control_wrappers(_OutOfRangeEnv(), "Toy-v0", gamma=gamma)
        env.reset(seed=0)
        out = []
        for _ in range(20):
            _, r, _, _, _ = env.step(env.action_space.sample())
            out.append(float(r))
        return out

    assert rewards_for(0.0) != pytest.approx(rewards_for(0.99))


def test_image_observations_rejected():
    with pytest.raises(TypeError, match="image"):
        continuous_control_wrappers(_ImageObsEnv(), "Toy-v0", gamma=0.99)


def test_discrete_actions_rejected():
    with pytest.raises(TypeError, match="not a continuous"):
        continuous_control_wrappers(_DiscreteActionEnv(), "Toy-v0", gamma=0.99)


# ---------------------------------------------------------------------------
# Generic contract: every named stack must produce a working env
# (parametrized over the registry — a new stack is covered automatically)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stack_name", sorted(WRAPPER_STACKS))
def test_stack_produces_working_env(stack_name):
    stack = WRAPPER_STACKS[stack_name]
    # Each stack declares an action-space contract; test with a matching env.
    env_id = "CartPole-v1" if stack_name == "discrete_control" else "Pendulum-v1"
    env = stack(gym.make(env_id), env_id, 0.99)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    obs, reward, _, _, _ = env.step(env.action_space.sample())
    assert np.all(np.isfinite(obs)) and np.isfinite(reward)
    env.close()


def test_discrete_stack_rejects_continuous_actions():
    from envs.wrappers import discrete_control_wrappers

    with pytest.raises(TypeError, match="not Discrete"):
        discrete_control_wrappers(gym.make("Pendulum-v1"), "Pendulum-v1", 0.99)


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------

def test_resolve_named_stack():
    assert resolve_wrapper_stack("continuous_control", None) is continuous_control_wrappers


def test_resolve_empty_name_falls_back_to_default():
    sentinel = object()
    assert resolve_wrapper_stack("", sentinel) is sentinel
    assert resolve_wrapper_stack(None, sentinel) is sentinel


def test_resolve_unknown_name_raises_with_options():
    with pytest.raises(ValueError, match="continuous_control"):
        resolve_wrapper_stack("does_not_exist", None)


def test_make_env_without_stack_fails_loudly():
    """No adapter override + no wrappers argument must be an error, not a
    silently-raw env (an unclipped env trains subtly differently)."""
    thunk = make_env(
        "Pendulum-v1", idx=0, capture_video=False, run_name="t", gamma=0.99,
        wrappers=None,
    )
    with pytest.raises(ValueError, match="no preprocessing wrapper stack"):
        thunk()
