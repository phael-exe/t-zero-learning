# Unit tests for the PPO assignment. Run with:  python -m pytest tests/test_ppo.py
# (Instructor: verify against the solution with
#  PPO_MODULE=algorithms.ppo_solution python -m pytest tests/test_ppo.py)
#
# PPO reuses the actor-critic network from the A2C assignment
# (networks/discrete_actor_critic.py); the last test checks it is in place.
import importlib
import os
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from networks import get_network

ppo = importlib.import_module(os.environ.get("PPO_MODULE", "algorithms.ppo"))


def a2c_n_step_returns(rewards, dones_after, next_value, gamma):
    """A2C's bootstrapped n-step return (A2C Part 1), with A2C's done convention:
    ``dones_after[t]`` is the done flag returned *by* step t."""
    returns = torch.zeros_like(rewards)
    running = next_value
    for t in reversed(range(rewards.shape[0])):
        running = rewards[t] + gamma * (1.0 - dones_after[t]) * running
        returns[t] = running
    return returns


# ------------------------------------------------------------------ Part 1: GAE


def test_gae_shapes_and_returns_are_advantages_plus_values():
    torch.manual_seed(0)
    rewards, values = torch.randn(5, 3), torch.randn(5, 3)
    dones = torch.zeros(5, 3)
    adv, ret = ppo.compute_gae(rewards, values, dones, torch.randn(3), torch.zeros(3), 0.99, 0.95)
    assert adv.shape == (5, 3) and ret.shape == (5, 3)
    assert torch.allclose(ret, adv + values)


def test_gae_hand_computed_no_done():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [1.0]])
    dones = torch.zeros(2, 1)
    next_value, next_done = torch.tensor([4.0]), torch.tensor([0.0])
    adv, _ = ppo.compute_gae(rewards, values, dones, next_value, next_done, gamma=0.5, gae_lambda=0.5)
    # delta1 = 2 + 0.5*4 - 1 = 3                 ; A1 = 3
    # delta0 = 1 + 0.5*1 - 0.5 = 1               ; A0 = 1 + 0.5*0.5*3 = 1.75
    assert torch.allclose(adv, torch.tensor([[1.75], [3.0]]))


def test_gae_lambda_zero_is_one_step_td_error():
    torch.manual_seed(1)
    rewards, values = torch.randn(4, 2), torch.randn(4, 2)
    dones = torch.zeros(4, 2)
    next_value = torch.randn(2)
    adv, _ = ppo.compute_gae(rewards, values, dones, next_value, torch.zeros(2), gamma=0.9, gae_lambda=0.0)
    next_values = torch.cat([values[1:], next_value[None]], dim=0)
    assert torch.allclose(adv, rewards + 0.9 * next_values - values, atol=1e-6)


def test_gae_lambda_one_matches_a2c_n_step_return():
    # The link to A2C: with lambda = 1, GAE's return target is exactly the
    # bootstrapped n-step return from A2C Part 1.
    torch.manual_seed(2)
    T, N = 6, 3
    rewards, values = torch.randn(T, N), torch.randn(T, N)
    dones = torch.zeros(T, N)
    dones[2, 0] = 1.0  # an episode ended at step 1 in env 0
    dones[4, 2] = 1.0  # ... and at step 3 in env 2
    next_value, next_done = torch.randn(N), torch.tensor([0.0, 1.0, 0.0])
    adv, ret = ppo.compute_gae(rewards, values, dones, next_value, next_done, gamma=0.9, gae_lambda=1.0)
    # A2C's convention: the done flag returned *by* step t
    dones_after = torch.cat([dones[1:], next_done[None]], dim=0)
    expected = a2c_n_step_returns(rewards, dones_after, next_value, 0.9)
    assert torch.allclose(ret, expected, atol=1e-5)
    assert torch.allclose(adv, expected - values, atol=1e-5)


def test_gae_done_cuts_propagation():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.zeros(3, 1)
    dones = torch.tensor([[0.0], [0.0], [1.0]])  # the episode ended at step 1
    next_value, next_done = torch.tensor([100.0]), torch.tensor([0.0])
    adv, _ = ppo.compute_gae(rewards, values, dones, next_value, next_done, gamma=0.5, gae_lambda=1.0)
    # step 1 is terminal: nothing from steps 2+ may leak into steps 0 and 1
    assert adv[1, 0].item() == pytest.approx(1.0)
    assert adv[0, 0].item() == pytest.approx(1.5)
    assert adv[2, 0].item() == pytest.approx(51.0)


def test_gae_next_done_ignores_next_value():
    rewards = torch.tensor([[2.0], [3.0]])
    values = torch.zeros(2, 1)
    dones = torch.zeros(2, 1)
    adv, _ = ppo.compute_gae(rewards, values, dones, torch.tensor([1000.0]), torch.tensor([1.0]), 0.9, 0.95)
    assert adv[1, 0].item() == pytest.approx(3.0)
    # delta0 = 2 + 0.9 * V(s1) - V(s0) = 2 ; A0 = 2 + 0.9 * 0.95 * 3
    assert adv[0, 0].item() == pytest.approx(2.0 + 0.9 * 0.95 * 3.0)


def test_gae_envs_are_independent():
    torch.manual_seed(3)
    rewards, values = torch.randn(4, 2), torch.randn(4, 2)
    dones = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    next_value, next_done = torch.randn(2), torch.zeros(2)
    adv, _ = ppo.compute_gae(rewards, values, dones, next_value, next_done, 0.99, 0.95)
    for n in range(2):
        adv_n, _ = ppo.compute_gae(rewards[:, n:n + 1], values[:, n:n + 1], dones[:, n:n + 1],
                                   next_value[n:n + 1], next_done[n:n + 1], 0.99, 0.95)
        assert torch.allclose(adv[:, n], adv_n[:, 0], atol=1e-6)


# --------------------------------------------- Part 2: clipped surrogate loss


def test_clipped_loss_at_ratio_one_equals_vanilla_policy_gradient_loss():
    logp = torch.tensor([-1.0, -2.0, -0.5])
    adv = torch.tensor([2.0, -1.0, 0.5])
    loss = ppo.compute_clipped_policy_loss(logp, logp.clone(), adv, clip_coef=0.2)
    # ratio = 1 everywhere: -mean(ratio * A) = -mean(A)
    assert loss.shape == ()
    assert loss.item() == pytest.approx(-adv.mean().item())


def test_clipped_loss_hand_computed():
    old = torch.zeros(4)
    new = torch.log(torch.tensor([1.5, 1.5, 0.5, 0.5]))  # ratios
    adv = torch.tensor([1.0, -1.0, 1.0, -1.0])
    loss = ppo.compute_clipped_policy_loss(new, old, adv, clip_coef=0.2)
    # objective per sample = min(r*A, clip(r)*A):
    #   r=1.5,A=+1 -> min(1.5, 1.2) = 1.2     r=1.5,A=-1 -> min(-1.5, -1.2) = -1.5
    #   r=0.5,A=+1 -> min(0.5, 0.8) = 0.5     r=0.5,A=-1 -> min(-0.5, -0.8) = -0.8
    assert loss.item() == pytest.approx(-(1.2 - 1.5 + 0.5 - 0.8) / 4)


def _grad_wrt_newlogprob(ratio, adv, clip_coef=0.2):
    new = torch.log(torch.tensor([ratio])).requires_grad_(True)
    loss = ppo.compute_clipped_policy_loss(new, torch.zeros(1), torch.tensor([adv]), clip_coef)
    loss.backward()
    return new.grad.item()


def test_clipped_loss_no_gradient_once_positive_advantage_is_clipped():
    assert _grad_wrt_newlogprob(1.5, +1.0) == 0.0


def test_clipped_loss_no_gradient_once_negative_advantage_is_clipped():
    assert _grad_wrt_newlogprob(0.5, -1.0) == 0.0


def test_clipped_loss_keeps_gradient_when_moving_back_toward_the_band():
    # ratio already too high but A < 0 (or too low and A > 0): the unclipped
    # term is the pessimistic one, so the gradient must still flow.
    assert _grad_wrt_newlogprob(1.5, -1.0) != 0.0
    assert _grad_wrt_newlogprob(0.5, +1.0) != 0.0


def test_clipped_loss_inside_band_has_vanilla_gradient():
    # d/dlogp of -(ratio * A) = -ratio * A
    assert _grad_wrt_newlogprob(1.1, 2.0) == pytest.approx(-1.1 * 2.0, rel=1e-5)


def test_clipped_loss_is_never_more_optimistic_than_unclipped():
    torch.manual_seed(4)
    new, old, adv = torch.randn(1000) * 0.5, torch.randn(1000) * 0.5, torch.randn(1000)
    loss = ppo.compute_clipped_policy_loss(new, old, adv, clip_coef=0.2)
    unclipped = -(torch.exp(new - old) * adv).mean()
    assert loss.item() >= unclipped.item() - 1e-6


def test_clipped_loss_old_logprob_and_advantages_are_constants():
    new = torch.tensor([-1.0, -0.5], requires_grad=True)
    old = torch.tensor([-1.2, -0.4], requires_grad=True)
    adv = torch.tensor([1.0, -1.0], requires_grad=True)
    ppo.compute_clipped_policy_loss(new, old, adv, clip_coef=0.2).backward()
    assert new.grad is not None
    for t in (old, adv):
        assert t.grad is None or torch.all(t.grad == 0), "old log-probs and advantages must not get gradients"


# ---------------------------------------------- Part 3: approx KL and clipfrac


def test_kl_and_clipfrac_zero_for_identical_policies():
    logp = torch.tensor([-1.0, -0.3, -2.0])
    kl, cf = ppo.approx_kl_and_clipfrac(logp, logp.clone(), clip_coef=0.2)
    assert kl.item() == pytest.approx(0.0, abs=1e-7)
    assert cf.item() == pytest.approx(0.0)


def test_kl_and_clipfrac_hand_computed():
    old = torch.zeros(4)
    ratios = torch.tensor([1.5, 0.5, 1.1, 1.0])
    kl, cf = ppo.approx_kl_and_clipfrac(torch.log(ratios), old, clip_coef=0.2)
    expected_kl = ((ratios - 1) - torch.log(ratios)).mean()
    assert kl.item() == pytest.approx(expected_kl.item(), rel=1e-5)
    assert cf.item() == pytest.approx(0.5)  # 1.5 and 0.5 are outside [0.8, 1.2]


def test_kl_is_non_negative_and_carries_no_gradient():
    torch.manual_seed(5)
    new = (torch.randn(500) * 0.3).requires_grad_(True)
    old = torch.randn(500) * 0.3
    kl, cf = ppo.approx_kl_and_clipfrac(new, old, clip_coef=0.2)
    assert kl.item() >= 0.0
    assert not kl.requires_grad and not cf.requires_grad
    assert kl.shape == () and cf.shape == ()


def test_kl_estimates_the_categorical_kl():
    # actions drawn from the old policy: the estimator should approach KL(old || new)
    torch.manual_seed(6)
    old_p = torch.tensor([0.5, 0.3, 0.2])
    new_p = torch.tensor([0.45, 0.33, 0.22])
    actions = torch.multinomial(old_p, 200_000, replacement=True)
    kl, _ = ppo.approx_kl_and_clipfrac(torch.log(new_p)[actions], torch.log(old_p)[actions], clip_coef=0.2)
    exact = (old_p * (old_p.log() - new_p.log())).sum()
    assert kl.item() == pytest.approx(exact.item(), rel=0.1)


# ------------------------------------------------ the network from A2C (Part 3)


def test_a2c_network_is_in_place():
    torch.manual_seed(0)
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1, 1, (8,)),
        single_action_space=gym.spaces.Discrete(4),
    )
    agent = get_network(ppo.Args.network)(envs)
    try:
        action, logprob, entropy, value = agent.get_action_and_value(torch.randn(5, 8))
    except NotImplementedError:
        pytest.fail("networks/discrete_actor_critic.py still has the A2C blank: "
                    "copy in your completed get_action_and_value from the A2C assignment")
    assert action.shape == (5,) and logprob.shape == (5,) and entropy.shape == (5,) and value.shape == (5, 1)
