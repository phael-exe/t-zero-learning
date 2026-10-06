"""PPO (continuous actions) — algorithm class with ``initialize`` / ``train`` lifecycle.

This is the canonical PPO reference in this repo. A variant with split
actor/critic optimizers lives in ``ppo_continuous_action_split_optim.py``;
fixes to the shared parts should be mirrored there.

Adapted from CleanRL (https://github.com/vwxyzjn/cleanrl),
Copyright (c) 2019 CleanRL developers, MIT License (see LICENSE).
"""
import time
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from collections import deque

import envs.custom_envs  # noqa: F401 — Gym registration side effects
from envs.custom_envs.envs_utils import episode_completions_from_vector_infos
from envs import (
    build_vector_envs,
    continuous_control_wrappers,
    make_env,
    resolve_training_video_schedule,
)
from networks import get_network
from core.checkpoint import save_checkpoint
from core.base_config import RunConfig
from algorithms.base import Algorithm


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters for continuous actions."""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""


@dataclass
class Args(RunConfig):
    """Full configuration for PPO with continuous actions.

    Inherits run-level fields from :class:`RunConfig` and composes
    :class:`PPOConfig` (algorithm hyperparameters); the network is
    ``network`` + ``network_kwargs`` (see :func:`networks.get_network`).
    """
    algorithm: str = "ppo_continuous_action"

    # Nested configs
    network: str = "ContinuousActorCritic"
    """network class to build (see ``RunConfig.network``); any class with
    the interface of ``ContinuousActorCritic`` works"""
    ppo_continuous_action: PPOConfig = field(default_factory=PPOConfig)
    """PPO hyperparameters — the field name deliberately equals the algorithm
    name, so the YAML section, saved run configs (``asdict``), and CLI
    override paths all use one spelling"""

    @property
    def algo(self) -> PPOConfig:
        """Shorthand access to algorithm hyperparameters."""
        return self.ppo_continuous_action


class PPO(Algorithm):
    """Proximal Policy Optimization for continuous action spaces.

    Usage::

        ppo = PPO(args)
        ppo.initialize()          # envs, agent, optimizers, buffers, optional resume
        ppo.train()               # rollout → GAE → clipped update loop
    """

    default_wrappers = staticmethod(continuous_control_wrappers)
    """Input contract of the flat-vector MLP agent: flatten, clip actions,
    normalize + clip rewards (obs normalization is agent-side, opt-in)."""

    def __init__(self, args: Args, resume_run_dir: Path | None = None):
        super().__init__(args, resume_run_dir)

    # ------------------------------------------------------------------
    # initialize — everything before the first training iteration
    # ------------------------------------------------------------------

    def initialize(self):
        super().initialize()  # task parsing, run dir, seeding, device
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

        # Rollout / iteration sizes (depend on `num_envs` for vector MT benchmarks)
        args.batch_size = int(args.num_envs * args.algo.num_steps)
        args.minibatch_size = int(args.batch_size // args.algo.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size

        self._setup_logging_and_checkpoints()

        assert isinstance(self.envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        self.agent = get_network(args.network)(self.envs, **args.network_kwargs).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=args.algo.learning_rate, eps=1e-5)

        self.last_iteration_resume = 0
        if self.resuming:
            self._resume_from_checkpoint()

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((args.algo.num_steps, args.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        self.actions = torch.zeros((args.algo.num_steps, args.num_envs) + self.envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((args.algo.num_steps, args.num_envs)).to(self.device)
        self.rewards = torch.zeros((args.algo.num_steps, args.num_envs)).to(self.device)
        self.dones = torch.zeros((args.algo.num_steps, args.num_envs)).to(self.device)
        self.values = torch.zeros((args.algo.num_steps, args.num_envs)).to(self.device)

        # Initial state (unchanged from CleanRL): reset envs, zero counters
        self.start_time = time.time()
        if not self.resuming:
            self.global_step = 0
            self.next_obs, _ = self.envs.reset(seed=args.seed)
            self.next_obs = torch.Tensor(self.next_obs).to(self.device)
            self.next_done = torch.zeros(args.num_envs).to(self.device)
            self.global_ep_counter = 0
            self.recent_ep_returns = deque(maxlen=100)
            self.recent_ep_lengths = deque(maxlen=100)
        else:
            # Env simulator state is not checkpointed, so the freshly built
            # envs must be reset before stepping. This overrides the
            # checkpointed next_obs/next_done: resumed training continues from
            # new episodes (offset seed avoids replaying the initial ones).
            self.next_obs, _ = self.envs.reset(seed=args.seed + self.global_step)
            self.next_obs = torch.Tensor(self.next_obs).to(self.device)
            self.next_done = torch.zeros(args.num_envs).to(self.device)

    # ------------------------------------------------------------------
    # train — the PPO iteration loop (rollout → GAE → clipped update)
    # ------------------------------------------------------------------

    def train(self):
        # Read-only shorthands. All mutable run state lives on `self` so that
        # checkpointing and logging always see the current values.
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

        with tqdm(range(iter_start, args.num_iterations + 1), desc="PPO iterations", unit="iter") as pbar:
            for iteration in pbar:
                self.iteration = iteration  # persisted in checkpoints as `last_iteration`

                # Annealing the rate if instructed to do so.
                if cfg.anneal_lr:
                    frac = 1.0 - (iteration - 1.0) / args.num_iterations
                    lrnow = frac * cfg.learning_rate
                    self.optimizer.param_groups[0]["lr"] = lrnow

                rollout_ep_returns: list[float] = []
                rollout_ep_successes: list[float] = []
                for step in range(0, cfg.num_steps):
                    self.global_step += args.num_envs
                    # Obs normalization (no-op when use_obs_norm is off): fold
                    # the raw obs into the running stats, then store/act on the
                    # normalized version. The buffer holds obs normalized with
                    # the stats *at collection time*, so the update phase stays
                    # consistent with the logprobs collected here.
                    agent.update_norm(self.next_obs)
                    obs_input = agent.normalize_obs(self.next_obs)
                    self.obs[step] = obs_input
                    self.dones[step] = self.next_done

                    # ALGO LOGIC: action logic
                    with torch.no_grad():
                        action, logprob, _, value = agent.get_action_and_value(obs_input, input_is_normalized=True)
                        self.values[step] = value.flatten()
                    self.actions[step] = action
                    self.logprobs[step] = logprob

                    # Execute the game and log data (unchanged from CleanRL).
                    next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                    next_done = np.logical_or(terminations, truncations)
                    self.rewards[step] = torch.tensor(reward).to(device).view(-1)
                    self.next_obs = torch.Tensor(next_obs).to(device)
                    self.next_done = torch.Tensor(next_done).to(device)

                    completed_returns, completed_lengths, completed_successes = (
                        episode_completions_from_vector_infos(terminations, truncations, infos)
                    )
                    if completed_returns:
                        self.recent_ep_returns.extend(completed_returns)
                        self.recent_ep_lengths.extend(completed_lengths)
                        self.global_ep_counter += len(completed_returns)
                        rollout_ep_returns.extend(completed_returns)
                        rollout_ep_successes.extend(completed_successes)

                # bootstrap value if not done
                with torch.no_grad():
                    # next_obs is raw; get_value normalizes internally (stats
                    # are updated when this obs opens the next rollout).
                    next_value = agent.get_value(self.next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(self.rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(cfg.num_steps)):
                        if t == cfg.num_steps - 1:
                            nextnonterminal = 1.0 - self.next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]
                            nextvalues = self.values[t + 1]
                        delta = self.rewards[t] + cfg.gamma * nextvalues * nextnonterminal - self.values[t]
                        advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + self.values

                # flatten the batch
                b_obs = self.obs.reshape((-1,) + envs.single_observation_space.shape)
                b_logprobs = self.logprobs.reshape(-1)
                b_actions = self.actions.reshape((-1,) + envs.single_action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = self.values.reshape(-1)

                # Optimizing the policy and value network
                b_inds = np.arange(args.batch_size)
                clipfracs = []
                grad_norms: list[float] = []
                for epoch in range(cfg.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, args.batch_size, args.minibatch_size):
                        end = start + args.minibatch_size
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                            b_obs[mb_inds], b_actions[mb_inds], input_is_normalized=True
                        )
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]

                        mb_advantages = b_advantages[mb_inds]
                        if cfg.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        if cfg.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -cfg.clip_coef,
                                cfg.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef
                        self.optimizer.zero_grad()
                        loss.backward()
                        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                        # clip_grad_norm_ returns a (possibly CUDA) scalar tensor; .item() is device-safe
                        grad_norms.append(float(grad_norm.detach().item()))
                        self.optimizer.step()

                    if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                        break

                # Diagnostics only — nothing here affects training.
                self._log_iteration_metrics(
                    pbar,
                    pg_loss=pg_loss,
                    v_loss=v_loss,
                    entropy_loss=entropy_loss,
                    old_approx_kl=old_approx_kl,
                    approx_kl=approx_kl,
                    clipfracs=clipfracs,
                    grad_norms=grad_norms,
                    b_values=b_values,
                    b_returns=b_returns,
                    rollout_ep_returns=rollout_ep_returns,
                    rollout_ep_successes=rollout_ep_successes,
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
    # Logging — per-iteration diagnostics (tqdm postfix + wandb)
    # ------------------------------------------------------------------

    def _log_iteration_metrics(
        self,
        pbar,
        *,
        pg_loss,
        v_loss,
        entropy_loss,
        old_approx_kl,
        approx_kl,
        clipfracs,
        grad_norms,
        b_values,
        b_returns,
        rollout_ep_returns,
        rollout_ep_successes,
    ) -> None:
        """Update the tqdm postfix and, when tracking, log metrics to wandb.

        Takes the per-iteration temporaries from :meth:`train`; persistent run
        state (``global_step``, episode stats, log schedules) is read from
        ``self``. Pure diagnostics — nothing here affects training.
        """
        args = self.args

        # Explained variance: how well the value function predicts returns.
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Calculate SPS (Steps Per Second)
        sps = int(self.global_step / (time.time() - self.start_time))
        postfix: dict = {
            "sps": sps,
            "gs": self.global_step,
        }
        postfix["lr"] = float(self.optimizer.param_groups[0]["lr"])
        if rollout_ep_returns:
            postfix["r_roll"] = float(np.mean(rollout_ep_returns))
        if rollout_ep_successes:
            postfix["succ_roll"] = float(np.mean(rollout_ep_successes))
        pbar.set_postfix(postfix, refresh=False)

        if not args.track:
            return
        import wandb

        metrics_dict = {
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/SPS": sps,
            "global_step": self.global_step,
        }
        metrics_dict["charts/learning_rate"] = self.optimizer.param_groups[0]["lr"]
        metrics_dict["losses/grad_norm"] = float(np.mean(grad_norms)) if len(grad_norms) else 0.0

        # Episode metrics: log ONCE per PPO iteration (running mean over last 100 completed episodes).
        if len(self.recent_ep_returns) > 0:
            metrics_dict.update(
                {
                    "charts/episodic_return_mean_last100": float(np.mean(self.recent_ep_returns)),
                    "charts/episodic_length_mean_last100": float(np.mean(self.recent_ep_lengths)) if len(self.recent_ep_lengths) > 0 else 0.0,
                    "charts/num_episodes": self.global_ep_counter,
                }
            )
        if rollout_ep_returns:
            metrics_dict["rollout/mean_episode_return"] = float(np.mean(rollout_ep_returns))
            metrics_dict["rollout/episodes_completed"] = int(len(rollout_ep_returns))
        if rollout_ep_successes:
            metrics_dict["rollout/mean_success_rate"] = float(np.mean(rollout_ep_successes))

        # Special logging (e.g., weight histograms) every X global steps.
        # Use a threshold (not modulo) because global_step increments by num_envs.
        if self.global_step >= self.next_special_log_step:
            for name, param in self.agent.named_parameters():
                metrics_dict[f"weights/{name}"] = wandb.Histogram(param.detach().clone().cpu().numpy())
            # Advance schedule; keep incrementing in case we skipped over multiple thresholds.
            while self.next_special_log_step <= self.global_step:
                self.next_special_log_step += self.special_log_every

        # Log all metrics at once
        wandb.log(metrics_dict, step=self.global_step)

    # ------------------------------------------------------------------
    # Checkpoint contract — algorithm-specific state
    # ------------------------------------------------------------------

    def checkpoint_state_dict(self) -> dict:
        """Build the PPO-specific payload for a checkpoint.

        Everything is read from ``self`` — the training loop keeps all
        resumable state there.  The generic ``checkpoint.save_checkpoint``
        wraps this inside ``{"algorithm": ...}`` alongside RNG and
        ``global_step``.
        """
        state: dict = {
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "last_iteration": int(self.iteration),
            "next_special_log_step": int(self.next_special_log_step),
            "next_checkpoint_step": int(self.next_checkpoint_step),
            "special_log_every": int(self.special_log_every),
            "next_obs": self.next_obs.detach().cpu(),
            "next_done": self.next_done.detach().cpu(),
            "global_ep_counter": int(self.global_ep_counter),
            "recent_ep_returns": list(self.recent_ep_returns),
            "recent_ep_lengths": list(self.recent_ep_lengths),
        }
        return state

    def load_checkpoint_state_dict(self, state: dict) -> None:
        """Restore PPO-specific state from a checkpoint ``algorithm`` payload."""
        if state.get("optimizer_state_dict") is None:
            self.envs.close()
            raise ValueError(
                "checkpoint has no single-optimizer state — it was produced by "
                "a different algorithm (use ppo_continuous_action_split_optim "
                "to resume it)"
            )
        self.agent.load_state_dict(state["agent_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = int(state.get("global_step", 0))
        self.last_iteration_resume = int(state["last_iteration"])
        self.next_special_log_step = int(state["next_special_log_step"])
        self.next_checkpoint_step = int(state["next_checkpoint_step"])
        self.special_log_every = int(state["special_log_every"])
        self.next_obs = state["next_obs"].to(self.device)
        self.next_done = state["next_done"].to(self.device)
        self.global_ep_counter = int(state["global_ep_counter"])
        self.recent_ep_returns = deque(state["recent_ep_returns"], maxlen=100)
        self.recent_ep_lengths = deque(state["recent_ep_lengths"], maxlen=100)


def main(args: Args, resume_run_dir: Path | None = None):
    """Entry point — kept for backward compatibility with ``train.py``."""
    ppo = PPO(args, resume_run_dir)
    ppo.initialize()
    ppo.train()
