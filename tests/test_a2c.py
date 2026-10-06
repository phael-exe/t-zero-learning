# Unit tests for the A2C assignment. Run with:  python -m pytest tests/test_a2c.py
# (Instructor: verify against the solution with
#  A2C_MODULE=algorithms.a2c_solution python -m pytest tests/test_a2c.py)
#
# Part 3 lives in networks/discrete_actor_critic.py; it is exercised here
# through the algorithm module's default network (``Args.network``).
import importlib
import os
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from networks import get_network

a2c = importlib.import_module(os.environ.get("A2C_MODULE", "algorithms.a2c"))
DiscreteActorCritic = get_network(a2c.Args.network)


# ------------------------------------------------------- Part 1: n-step returns


def test_returns_shape_and_gamma_zero_is_reward():
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    dones = torch.zeros(3, 2)
    next_value = torch.tensor([10.0, 20.0])
    returns = a2c.compute_n_step_returns(rewards, dones, next_value, gamma=0.0)
    assert returns.shape == (3, 2)
    assert torch.allclose(returns, rewards)


def test_returns_bootstrap_from_next_value_when_no_done():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    dones = torch.zeros(3, 1)
    next_value = torch.tensor([10.0])
    returns = a2c.compute_n_step_returns(rewards, dones, next_value, gamma=0.5)
    # R2 = 1 + 0.5*10 = 6 ; R1 = 1 + 0.5*6 = 4 ; R0 = 1 + 0.5*4 = 3
    assert torch.allclose(returns, torch.tensor([[3.0], [4.0], [6.0]]))


def test_returns_done_cuts_bootstrapping():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])  # episode ends after step 1
    next_value = torch.tensor([100.0])
    returns = a2c.compute_n_step_returns(rewards, dones, next_value, gamma=0.5)
    # R2 = 1 + 0.5*100 = 51 ; R1 = 1 (terminal) ; R0 = 1 + 0.5*1 = 1.5
    assert torch.allclose(returns, torch.tensor([[1.5], [1.0], [51.0]]))


def test_returns_done_at_last_step_ignores_next_value():
    rewards = torch.tensor([[2.0], [3.0]])
    dones = torch.tensor([[0.0], [1.0]])
    next_value = torch.tensor([1000.0])
    returns = a2c.compute_n_step_returns(rewards, dones, next_value, gamma=0.9)
    assert torch.allclose(returns, torch.tensor([[2.0 + 0.9 * 3.0], [3.0]]))


def test_returns_envs_are_independent():
    # env 0 terminates at step 0, env 1 never does — columns must not mix
    rewards = torch.ones(3, 2)
    dones = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    next_value = torch.tensor([0.0, 8.0])
    returns = a2c.compute_n_step_returns(rewards, dones, next_value, gamma=0.5)
    expected = torch.tensor([[1.0, 1.0 + 0.5 * (1.0 + 0.5 * (1.0 + 0.5 * 8.0))],
                             [1.0 + 0.5 * 1.0, 1.0 + 0.5 * (1.0 + 0.5 * 8.0)],
                             [1.0, 1.0 + 0.5 * 8.0]])
    assert torch.allclose(returns, expected)


# ----------------------------------------------------------- Part 2: policy loss


def test_policy_loss_value_with_baseline():
    logprobs = torch.tensor([-1.0, -2.0])
    returns = torch.tensor([5.0, 1.0])
    values = torch.tensor([3.0, 3.0])
    loss = a2c.compute_policy_loss(logprobs, returns, values, use_baseline=True)
    # advantages = [2, -2]; loss = -mean([-1*2, -2*-2]) = -mean([-2, 4]) = -1
    assert loss.shape == ()
    assert loss.item() == pytest.approx(-1.0)


def test_policy_loss_without_baseline_uses_raw_returns():
    logprobs = torch.tensor([-1.0, -2.0])
    returns = torch.tensor([5.0, 1.0])
    values = torch.tensor([3.0, 3.0])
    loss = a2c.compute_policy_loss(logprobs, returns, values, use_baseline=False)
    # weights = [5, 1]; loss = -mean([-5, -2]) = 3.5
    assert loss.item() == pytest.approx(3.5)


def test_policy_loss_gradient_increases_logprob_of_positive_advantage():
    logits = torch.zeros(2, requires_grad=True)
    logprobs = torch.log_softmax(logits, dim=0)
    returns = torch.tensor([2.0, 0.0])
    values = torch.tensor([1.0, 1.0])  # advantages = [+1, -1]
    loss = a2c.compute_policy_loss(logprobs, returns, values)
    loss.backward()
    # gradient descent on `loss` must raise logit 0 and lower logit 1
    assert logits.grad[0] < 0 and logits.grad[1] > 0


def test_policy_loss_does_not_backprop_into_critic():
    logprobs = torch.tensor([-1.0, -2.0], requires_grad=True)
    values = torch.tensor([3.0, 3.0], requires_grad=True)
    returns = torch.tensor([5.0, 1.0])
    loss = a2c.compute_policy_loss(logprobs, returns, values)
    loss.backward()
    assert logprobs.grad is not None
    assert values.grad is None or torch.all(values.grad == 0), (
        "the advantage must be a constant for the policy gradient (detach it)"
    )


# --------------------------------------------- Part 3: categorical actor-critic


def make_agent(n_actions=3, obs_dim=4, seed=0):
    torch.manual_seed(seed)
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1, 1, (obs_dim,)),
        single_action_space=gym.spaces.Discrete(n_actions),
    )
    return DiscreteActorCritic(envs, activation="Tanh", hidden_layers_size=16)


def test_agent_output_shapes_and_dtypes():
    agent = make_agent()
    x = torch.randn(5, 4)
    action, logprob, entropy, value = agent.get_action_and_value(x)
    assert action.shape == (5,) and action.dtype == torch.int64
    assert logprob.shape == (5,)
    assert entropy.shape == (5,)
    assert value.shape == (5, 1)
    assert torch.all((action >= 0) & (action < 3))


def test_agent_logprob_matches_log_softmax_of_logits():
    agent = make_agent()
    x = torch.randn(6, 4)
    given = torch.tensor([0, 1, 2, 0, 1, 2])
    action, logprob, _, _ = agent.get_action_and_value(x, action=given)
    assert torch.equal(action, given), "when an action is given, it must be returned unchanged"
    expected = torch.log_softmax(agent.actor(x), dim=1).gather(1, given[:, None]).squeeze(1)
    assert torch.allclose(logprob, expected, atol=1e-6)


def test_agent_entropy_of_uniform_policy_is_log_n():
    agent = make_agent(n_actions=4)
    # zero the last actor layer -> uniform logits
    with torch.no_grad():
        agent.actor[-1].weight.zero_()
        agent.actor[-1].bias.zero_()
    _, _, entropy, _ = agent.get_action_and_value(torch.randn(3, 4))
    assert torch.allclose(entropy, torch.full((3,), float(torch.log(torch.tensor(4.0)))), atol=1e-6)


def test_agent_deterministic_is_argmax_and_sampling_is_stochastic():
    agent = make_agent(n_actions=3)
    with torch.no_grad():  # make action 2 clearly, but not overwhelmingly, preferred
        agent.actor[-1].weight.zero_()
        agent.actor[-1].bias.copy_(torch.tensor([0.0, 0.0, 1.0]))
    x = torch.randn(200, 4)
    det, _, _, _ = agent.get_action_and_value(x, deterministic=True)
    assert torch.all(det == 2)
    sampled, _, _, _ = agent.get_action_and_value(x)
    assert len(torch.unique(sampled)) > 1, "sampling must draw from the distribution, not argmax"


def test_agent_value_matches_get_value():
    agent = make_agent()
    x = torch.randn(5, 4)
    _, _, _, value = agent.get_action_and_value(x)
    assert torch.allclose(value, agent.get_value(x))
