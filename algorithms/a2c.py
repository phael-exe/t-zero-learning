"""A2C (synchronous advantage actor-critic, discrete actions) — ``initialize`` / ``train`` lifecycle.

ASSIGNMENT: complete the three blocks marked "YOUR CODE HERE":
    Part 1: compute_n_step_returns              (this file)
    Part 2: compute_policy_loss                 (this file)
    Part 3: DiscreteActorCritic.get_action_and_value
            (networks/discrete_actor_critic.py)
Check your work with:   python -m pytest tests/test_a2c.py
Then train with:        python train.py --config a2c_cartpole

Everything else is a working training harness — you should not need to modify
it, but you are encouraged to read it: the rollout, the update and the logging
live in this one file; the base class only provides run plumbing (seeding, run
dir, wandb, checkpoint scheduling).

A2C is A3C (Mnih et al., 2016) with the asynchronous actors replaced by
``num_envs`` environments stepped in lockstep: every ``num_steps`` steps the
whole batch of transitions produces one gradient step on
``policy_loss + vf_coef * value_loss - ent_coef * entropy``.

Adapted from CleanRL (https://github.com/vwxyzjn/cleanrl),
Copyright (c) 2019 CleanRL developers, MIT License (see LICENSE).
"""
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

import envs.custom_envs  # noqa: F401 — Gym registration side effects
from envs.custom_envs.envs_utils import episode_completions_from_vector_infos
from envs import build_vector_envs, resolve_training_video_schedule
from envs.wrappers import discrete_control_wrappers
from networks.discrete_actor_critic import DiscreteActorCritic
from core.checkpoint import save_checkpoint
from core.base_config import AgentConfig, RunConfig
from algorithms.base import Algorithm


@dataclass
class A2CConfig:
    """A2C algorithm hyperparameters for discrete actions."""
    learning_rate: float = 7e-4
    """the learning rate of the optimizer"""
    num_steps: int = 5
    """steps per environment per update (the n of the n-step return; A3C's t_max)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    vf_coef: float = 0.5
    """coefficient of the value loss"""
    ent_coef: float = 0.01
    """coefficient of the entropy bonus"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    use_baseline: bool = True
    """subtract V(s) from the n-step return in the policy gradient (False = REINFORCE-style weights)"""


@dataclass
class Args(RunConfig):
    """Full configuration for A2C with discrete actions.

    Inherits run-level fields from :class:`RunConfig` and composes
    :class:`AgentConfig` (network architecture) and :class:`A2CConfig`
    (algorithm hyperparameters).
    """
    algorithm: str = "a2c"
    env_id: str = "CartPole-v1"
    """the gymnasium environment id (must have a Discrete action space)"""
    num_envs: int = 8
    """parallel environments stepped in lockstep (A3C's actors)"""

    # Nested configs
    agent: AgentConfig = field(default_factory=AgentConfig)
    """network architecture configuration"""
    a2c: A2CConfig = field(default_factory=A2CConfig)
    """A2C hyperparameters — the field name deliberately equals the algorithm
    name, so the YAML section, saved run configs, and CLI override paths all
    use one spelling"""

    @property
    def algo(self) -> A2CConfig:
        """Shorthand access to algorithm hyperparameters."""
        return self.a2c


# ---------------------------------------------------------------------------
# The algorithmic core — module-level so tests/test_a2c.py can verify it in
# isolation, before any training run.
# ---------------------------------------------------------------------------


def compute_n_step_returns(
    rewards: torch.Tensor, dones: torch.Tensor, next_value: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Bootstrapped n-step returns for a rollout of shape (T, N).

    Args:
        rewards: (T, N) reward received after the action at step t.
        dones: (T, N) 1.0 if the episode ended at step t (no bootstrapping
            through it), else 0.0.
        next_value: (N,) critic estimate V(s_T) of the observation *after*
            the last step, used to bootstrap unfinished episodes.
        gamma: discount factor.

    Returns:
        (T, N) returns R_t = r_t + gamma * (1 - done_t) * R_{t+1}, with
        R_T = next_value.
    """
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    R = next_value

    for t in reversed(range(T)):
        R = rewards[t] + gamma * (1.0 - dones[t]) * R
        returns[t] = R

    return returns

def compute_policy_loss(
    logprobs: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, use_baseline: bool = True
) -> torch.Tensor:
    """Policy-gradient loss for a flat batch (shape (B,) each).

    The weight of each log-probability is the advantage
    ``returns - values`` (or just ``returns`` when ``use_baseline`` is
    False). The weight is a *constant* for the policy gradient — no gradient
    may flow through it into the critic.

    Returns a scalar whose gradient *descent* performs policy gradient
    *ascent*.
    """
    if use_baseline:
        advantages = returns - values.detach()
    else:
        advantages = returns

    return -(logprobs * advantages).mean()

class A2C(Algorithm):
    """Synchronous advantage actor-critic for discrete action spaces.

    Usage::

        a2c = A2C(args)
        a2c.initialize()          # envs, actor-critic, optimizer, rollout buffers
        a2c.train()               # n-step rollout → returns → one gradient step
    """

    default_wrappers = staticmethod(discrete_control_wrappers)
    """Flat-vector obs, episode statistics, raw rewards."""

    # ------------------------------------------------------------------
    # initialize — everything before the first training step
    # ------------------------------------------------------------------

    def initialize(self):
        super().initialize()  # run dir, seeding, device, wrapper stack
        args = self.args

        video_length_steps = resolve_training_video_schedule(
            env_id=self.env_id,
            env_kwargs=self.env_kwargs,
            capture_video=args.capture_video,
            video_every_global_steps=int(args.video_every_global_steps),
            total_timesteps=int(args.total_timesteps),
            video_length_seconds=float(args.video_length_seconds),
        )
        self.envs, effective_num_envs = build_vector_envs(
            env_id=self.env_id,
            env_kwargs=self.env_kwargs,
            seed=args.seed,
            num_envs=int(args.num_envs),
            capture_video=args.capture_video,
            run_name=self.run_name,
            gamma=args.algo.gamma,
            experiment_dir=self.experiment_dir,
            video_every_global_steps=int(args.video_every_global_steps),
            video_length_steps=int(video_length_steps),
            wrappers=self.wrappers,
        )
        args.num_envs = effective_num_envs

        args.batch_size = int(args.num_envs * args.algo.num_steps)
        args.num_iterations = int(args.total_timesteps) // args.batch_size

        self._setup_logging_and_checkpoints()

        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        self.agent = DiscreteActorCritic(
            self.envs, args.agent.activation, args.agent.hidden_layers_size
        ).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=args.algo.learning_rate, eps=1e-5)

        self.last_iteration_resume = 0
        if self.resuming:
            self._resume_from_checkpoint()

        # Rollout storage: (num_steps, num_envs)
        T, N = args.algo.num_steps, args.num_envs
        obs_shape = self.envs.single_observation_space.shape
        self.obs = torch.zeros((T, N) + obs_shape, device=self.device)
        self.actions = torch.zeros((T, N), dtype=torch.int64, device=self.device)
        self.logprobs = torch.zeros((T, N), device=self.device)
        self.entropies = torch.zeros((T, N), device=self.device)
        self.rewards = torch.zeros((T, N), device=self.device)
        self.dones = torch.zeros((T, N), device=self.device)
        self.values = torch.zeros((T, N), device=self.device)
        # 0 on the vector env's autoreset step: that (obs -> next_obs) pair
        # crosses an episode boundary and carries no learning signal.
        self.valid = torch.zeros((T, N), device=self.device)

        # Log ~every 1000 env steps (one iteration is only num_envs*num_steps)
        self.log_every_iters = max(1, 1000 // args.batch_size)

        self.start_time = time.time()
        if not self.resuming:
            self.global_step = 0
            self.global_ep_counter = 0
            self.recent_ep_returns = deque(maxlen=100)
            self.recent_ep_lengths = deque(maxlen=100)
        # Env simulator state is not checkpointed: (re)start from fresh episodes.
        self.next_obs, _ = self.envs.reset(seed=args.seed + self.global_step)
        self.next_obs = torch.as_tensor(self.next_obs, dtype=torch.float32, device=self.device)
        self.autoreset = torch.zeros(N, device=self.device)

    # ------------------------------------------------------------------
    # train — rollout → n-step returns → single gradient step
    # ------------------------------------------------------------------

    def train(self):
        args = self.args
        cfg = args.algo
        agent = self.agent
        envs = self.envs
        device = self.device

        iter_start = self.last_iteration_resume + 1
        if self.resuming and self.last_iteration_resume >= args.num_iterations:
            print(
                f"Training already complete (last_iteration={self.last_iteration_resume} >= num_iterations={args.num_iterations})."
            )
            envs.close()
            return

        with tqdm(range(iter_start, args.num_iterations + 1), desc="A2C iterations", unit="iter") as pbar:
            for iteration in pbar:
                self.iteration = iteration

                # ---------------- rollout: num_steps in every env ----------------
                for step in range(cfg.num_steps):
                    self.global_step += args.num_envs
                    self.obs[step] = self.next_obs
                    self.valid[step] = 1.0 - self.autoreset

                    # Gradients are recomputed in the update below; the rollout
                    # only needs the sampled actions.
                    with torch.no_grad():
                        action, _, _, _ = agent.get_action_and_value(self.next_obs)
                    self.actions[step] = action

                    next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                    done = np.logical_or(terminations, truncations)
                    self.rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
                    self.dones[step] = torch.as_tensor(done, dtype=torch.float32, device=device)
                    self.next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                    self.autoreset = self.dones[step]

                    completed_returns, completed_lengths, _ = episode_completions_from_vector_infos(
                        terminations, truncations, infos
                    )
                    if completed_returns:
                        self.recent_ep_returns.extend(completed_returns)
                        self.recent_ep_lengths.extend(completed_lengths)
                        self.global_ep_counter += len(completed_returns)

                # ---------------- targets: bootstrapped n-step returns ----------------
                with torch.no_grad():
                    next_value = agent.get_value(self.next_obs).reshape(-1)
                    returns = compute_n_step_returns(self.rewards, self.dones, next_value, cfg.gamma)

                # ---------------- update: one gradient step on the whole batch ----------------
                keep = self.valid.reshape(-1).bool()
                b_obs = self.obs.reshape((-1,) + envs.single_observation_space.shape)[keep]
                b_actions = self.actions.reshape(-1)[keep]
                b_returns = returns.reshape(-1)[keep]

                _, logprobs, entropy, values = agent.get_action_and_value(b_obs, b_actions)
                values = values.reshape(-1)

                pg_loss = compute_policy_loss(logprobs, b_returns, values, cfg.use_baseline)
                v_loss = 0.5 * ((values - b_returns) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                if iteration % self.log_every_iters == 0:
                    self._log_metrics(
                        pbar,
                        pg_loss=pg_loss,
                        v_loss=v_loss,
                        entropy_loss=entropy_loss,
                        grad_norm=grad_norm,
                        b_values=values.detach(),
                        b_returns=b_returns,
                    )

                if self.checkpoint_every > 0 and self.global_step >= self.next_checkpoint_step:
                    while self.next_checkpoint_step <= self.global_step:
                        self.next_checkpoint_step += self.checkpoint_every
                    save_checkpoint(
                        self.run_dir, self.global_step,
                        self.checkpoint_state_dict(), self.checkpoints_keep,
                    )

        self._post_training_eval()
        envs.close()

    # ------------------------------------------------------------------
    # Logging — periodic diagnostics (tqdm postfix + wandb)
    # ------------------------------------------------------------------

    def _log_metrics(self, pbar, *, pg_loss, v_loss, entropy_loss, grad_norm, b_values, b_returns) -> None:
        """Pure diagnostics — nothing here affects training."""
        args = self.args
        sps = int(self.global_step / (time.time() - self.start_time)) if self.global_step else 0
        postfix: dict = {"sps": sps, "gs": self.global_step, "ent": round(float(entropy_loss), 3)}
        if len(self.recent_ep_returns) > 0:
            postfix["r_last100"] = round(float(np.mean(self.recent_ep_returns)), 1)
        pbar.set_postfix(postfix, refresh=False)

        if not args.track:
            return
        import wandb

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        # The weights the policy gradient actually used this update
        weights = (b_returns - b_values) if args.algo.use_baseline else b_returns

        metrics_dict = {
            "losses/policy_loss": pg_loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/explained_variance": explained_var,
            "losses/grad_norm": float(grad_norm),
            "charts/advantage_mean": weights.mean().item(),
            "charts/advantage_std": weights.std().item() if weights.numel() > 1 else 0.0,
            "charts/SPS": sps,
            "global_step": self.global_step,
        }
        if len(self.recent_ep_returns) > 0:
            metrics_dict.update(
                {
                    "charts/episodic_return_mean_last100": float(np.mean(self.recent_ep_returns)),
                    "charts/episodic_length_mean_last100": float(np.mean(self.recent_ep_lengths)),
                    "charts/num_episodes": self.global_ep_counter,
                }
            )
        wandb.log(metrics_dict, step=self.global_step)

    # ------------------------------------------------------------------
    # Evaluation — the framework loop rebuilds the agent from these kwargs
    # and drives it through DiscreteActorCritic.act
    # ------------------------------------------------------------------

    def eval_model_kwargs(self) -> dict:
        return dict(
            activation=self.args.agent.activation,
            hidden_layers_size=self.args.agent.hidden_layers_size,
        )

    # ------------------------------------------------------------------
    # Checkpoint contract — on-policy, so only network/optimizer/counters
    # ------------------------------------------------------------------

    def checkpoint_state_dict(self) -> dict:
        return {
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "last_iteration": int(self.iteration),
            "next_special_log_step": int(self.next_special_log_step),
            "next_checkpoint_step": int(self.next_checkpoint_step),
            "special_log_every": int(self.special_log_every),
            "global_ep_counter": int(self.global_ep_counter),
            "recent_ep_returns": list(self.recent_ep_returns),
            "recent_ep_lengths": list(self.recent_ep_lengths),
        }

    def load_checkpoint_state_dict(self, state: dict) -> None:
        self.agent.load_state_dict(state["agent_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = int(state.get("global_step", 0))
        self.last_iteration_resume = int(state["last_iteration"])
        self.next_special_log_step = int(state["next_special_log_step"])
        self.next_checkpoint_step = int(state["next_checkpoint_step"])
        self.special_log_every = int(state["special_log_every"])
        self.global_ep_counter = int(state["global_ep_counter"])
        self.recent_ep_returns = deque(state["recent_ep_returns"], maxlen=100)
        self.recent_ep_lengths = deque(state["recent_ep_lengths"], maxlen=100)


def main(args: Args, resume_run_dir: Path | None = None):
    """Entry point — dispatched by ``train.py``."""
    a2c = A2C(args, resume_run_dir)
    a2c.initialize()
    a2c.train()
