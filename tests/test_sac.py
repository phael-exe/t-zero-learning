# Unit tests for the SAC assignment. Run with:  python -m pytest tests/test_sac.py
# (Instructor: verify against the solution with
#  SAC_MODULE=algorithms.sac_solution python -m pytest tests/test_sac.py)
import importlib
import os
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform

from networks import get_network

sac = importlib.import_module(os.environ.get("SAC_MODULE", "algorithms.sac"))


def _params(B=5, A=3, seed=0):
    torch.manual_seed(seed)
    mean = torch.randn(B, A) * 0.5
    log_std = torch.full((B, A), -0.5) + 0.1 * torch.randn(B, A)
    scale, bias = torch.ones(A), torch.zeros(A)
    return mean, log_std, scale, bias


# ------------------------------------------------- Part 1: squashed Gaussian


def test_sample_shapes():
    mean, log_std, scale, bias = _params(B=7, A=3)
    action, log_prob, mean_action = sac.squashed_gaussian_sample(mean, log_std, scale, bias)
    assert action.shape == (7, 3) and mean_action.shape == (7, 3)
    assert log_prob.shape == (7, 1), "log_prob is summed over action dims, keepdim=True: shape (B, 1)"


def test_actions_stay_inside_the_bounds():
    torch.manual_seed(1)
    mean = torch.randn(1000, 2) * 10  # far outside tanh's linear range
    log_std = torch.full((1000, 2), 1.0)
    low, high = torch.tensor([-1.0, -3.0]), torch.tensor([2.0, 1.0])
    scale, bias = (high - low) / 2, (high + low) / 2
    action, _, mean_action = sac.squashed_gaussian_sample(mean, log_std, scale, bias)
    for a in (action, mean_action):
        assert torch.all(a >= low) and torch.all(a <= high)
    # and the whole range is used, not just a centered slice of it
    assert action[:, 0].min() < -0.9 and action[:, 0].max() > 1.9


def test_mean_action_is_the_squashed_mean():
    mean, log_std, _, _ = _params()
    scale, bias = torch.tensor([2.0, 0.5, 1.0]), torch.tensor([1.0, 0.0, -0.5])
    _, _, mean_action = sac.squashed_gaussian_sample(mean, log_std, scale, bias)
    assert torch.allclose(mean_action, torch.tanh(mean) * scale + bias, atol=1e-6)


def test_tiny_std_samples_the_mean_action():
    mean, _, scale, bias = _params()
    log_std = torch.full_like(mean, -20.0)
    action, _, mean_action = sac.squashed_gaussian_sample(mean, log_std, scale, bias)
    assert torch.allclose(action, mean_action, atol=1e-5)


def _reference_log_prob(action, mean, log_std, scale, bias):
    """log pi(action) from torch's own change-of-variables machinery."""
    dist = TransformedDistribution(
        Normal(mean, log_std.exp()), [TanhTransform(), AffineTransform(loc=bias, scale=scale)]
    )
    return dist.log_prob(action).sum(1, keepdim=True)


@pytest.mark.parametrize("scale_bias", [((1.0, 1.0), (0.0, 0.0)), ((2.0, 0.4), (1.0, -0.3))])
def test_log_prob_includes_the_tanh_and_scale_correction(scale_bias):
    torch.manual_seed(2)
    mean = torch.randn(200, 2) * 0.5
    log_std = torch.full((200, 2), -1.0)
    scale, bias = torch.tensor(scale_bias[0]), torch.tensor(scale_bias[1])
    action, log_prob, _ = sac.squashed_gaussian_sample(mean, log_std, scale, bias)
    expected = _reference_log_prob(action.detach(), mean, log_std, scale, bias)
    assert torch.allclose(log_prob, expected, atol=1e-3), (
        "log_prob must be the density of the *squashed, rescaled* action: "
        "the Gaussian log-density of the pre-tanh sample minus "
        "log(action_scale * (1 - tanh^2) + 1e-6), summed over action dims"
    )


def test_log_prob_is_not_the_plain_gaussian_density():
    # with action_scale = 1, tanh squeezes the mass into (-1, 1): the density
    # of the action is higher than the Gaussian density of the pre-tanh sample
    mean, log_std, scale, bias = _params(B=100, A=2, seed=3)
    action, log_prob, _ = sac.squashed_gaussian_sample(mean, log_std, scale, bias)
    pre_tanh = torch.atanh(action.detach().clamp(-1 + 1e-6, 1 - 1e-6))
    plain = Normal(mean, log_std.exp()).log_prob(pre_tanh).sum(1, keepdim=True)
    assert torch.all(log_prob >= plain - 1e-4)
    assert (log_prob - plain).mean() > 0.05, "the tanh correction is missing"


def test_gradient_flows_through_the_sample():
    # reparameterization: action = tanh(mean + std * eps) depends on mean and log_std
    mean, log_std, scale, bias = _params()
    mean.requires_grad_(True)
    log_std.requires_grad_(True)
    action, _, _ = sac.squashed_gaussian_sample(mean, log_std, scale, bias)
    action.sum().backward()
    assert mean.grad is not None and mean.grad.abs().sum() > 0, "use rsample(), not sample()"
    assert log_std.grad is not None and log_std.grad.abs().sum() > 0, "use rsample(), not sample()"


def test_log_prob_has_gradient_for_the_entropy_term():
    mean, log_std, scale, bias = _params()
    log_std.requires_grad_(True)
    _, log_prob, _ = sac.squashed_gaussian_sample(mean, log_std, scale, bias)
    log_prob.sum().backward()
    assert log_std.grad is not None and log_std.grad.abs().sum() > 0


def test_samples_are_random():
    mean, log_std, scale, bias = _params()
    a1, _, _ = sac.squashed_gaussian_sample(mean, log_std, scale, bias)
    a2, _, _ = sac.squashed_gaussian_sample(mean, log_std, scale, bias)
    assert not torch.allclose(a1, a2)


# ------------------------------------------------- Part 2: soft Q target


def test_soft_target_hand_computed():
    rewards = torch.tensor([1.0, 0.5])
    dones = torch.tensor([0.0, 0.0])
    next_q1 = torch.tensor([10.0, 4.0])
    next_q2 = torch.tensor([8.0, 6.0])
    next_log_pi = torch.tensor([-2.0, 1.0])
    target = sac.compute_soft_q_target(rewards, dones, next_q1, next_q2, next_log_pi, alpha=0.5, gamma=0.9)
    # 1   + 0.9 * (min(10, 8) - 0.5 * -2) = 1   + 0.9 * 9   = 9.1
    # 0.5 + 0.9 * (min(4, 6)  - 0.5 *  1) = 0.5 + 0.9 * 3.5 = 3.65
    assert target.shape == (2,)
    assert torch.allclose(target, torch.tensor([9.1, 3.65]), atol=1e-5)


def test_soft_target_uses_the_min_of_the_two_critics():
    args = dict(rewards=torch.zeros(3), dones=torch.zeros(3), next_log_pi=torch.zeros(3), alpha=0.0, gamma=1.0)
    q_lo, q_hi = torch.tensor([1.0, -2.0, 3.0]), torch.tensor([5.0, 0.0, 3.5])
    t1 = sac.compute_soft_q_target(next_q1=q_lo, next_q2=q_hi, **args)
    t2 = sac.compute_soft_q_target(next_q1=q_hi, next_q2=q_lo, **args)
    assert torch.allclose(t1, q_lo) and torch.allclose(t2, q_lo)


def test_soft_target_terminal_does_not_bootstrap():
    rewards = torch.tensor([1.5, -2.0])
    target = sac.compute_soft_q_target(
        rewards, torch.ones(2), torch.tensor([100.0, 100.0]), torch.tensor([100.0, 100.0]),
        torch.tensor([-5.0, 5.0]), alpha=0.2, gamma=0.99,
    )
    assert torch.allclose(target, rewards)


def test_soft_target_alpha_zero_is_the_clipped_double_q_target():
    torch.manual_seed(4)
    r, q1, q2, logp = torch.randn(8), torch.randn(8), torch.randn(8), torch.randn(8)
    d = (torch.rand(8) < 0.3).float()
    target = sac.compute_soft_q_target(r, d, q1, q2, logp, alpha=0.0, gamma=0.95)
    assert torch.allclose(target, r + 0.95 * (1 - d) * torch.min(q1, q2), atol=1e-6)


def test_soft_target_rewards_entropy():
    # a less random next policy (higher log pi) is worth less
    common = dict(rewards=torch.zeros(1), dones=torch.zeros(1), next_q1=torch.ones(1), next_q2=torch.ones(1),
                  alpha=0.2, gamma=0.99)
    low_entropy = sac.compute_soft_q_target(next_log_pi=torch.tensor([2.0]), **common)
    high_entropy = sac.compute_soft_q_target(next_log_pi=torch.tensor([-2.0]), **common)
    assert high_entropy.item() > low_entropy.item()


# ------------------------------------------------- the harness uses Part 1


def test_network_act_agrees_with_part_1():
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1, 1, (4,), np.float32),
        single_action_space=gym.spaces.Box(np.array([-1.0, 0.0], np.float32), np.array([3.0, 0.5], np.float32)),
    )
    torch.manual_seed(5)
    agent = get_network(sac.Args.network)(envs)
    obs = torch.randn(6, 4)
    with torch.no_grad():
        action, log_prob, mean_action = sac.sample_action(agent.actor, obs)
        deterministic = agent.act(obs, deterministic=True)
    assert action.shape == (6, 2) and log_prob.shape == (6, 1)
    assert torch.allclose(mean_action, deterministic, atol=1e-6)
