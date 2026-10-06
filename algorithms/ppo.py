"""PPO (discrete actions) — ``initialize`` / ``train`` lifecycle.

ASSIGNMENT: complete the three blocks marked "YOUR CODE HERE":
    Part 1: compute_gae                   (this file)
    Part 2: compute_clipped_policy_loss   (this file)
    Part 3: approx_kl_and_clipfrac        (this file)
The network is the one you completed in the A2C assignment
(networks/discrete_actor_critic.py) — keep your version of it.
Check your work with:   python -m pytest tests/test_ppo.py
Then train with:        python train.py --config ppo_lunarlander

Everything else is a working training harness — you should not need to modify
it, but you are encouraged to read it and compare it with algorithms/a2c.py:
the rollout, the update and the logging live in this one file.

The training loop is CleanRL's ``ppo.py`` (Huang et al., 2022; see "The 37
Implementation Details of PPO"), with the A2C assignment's harness around it.
One deliberate difference from CleanRL: gymnasium >= 1.0 resets a finished
env on the *next* ``step`` call (whose action is ignored), so the rollout
holds one transition per episode that jumps from the terminal observation to
the reset observation. CleanRL trains on it; here it is masked out of the
update (``valid``), exactly as in ``algorithms/a2c.py``.

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
from networks import get_network
from core.checkpoint import save_checkpoint
from core.base_config import RunConfig
from algorithms.base import Algorithm


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters for discrete actions (defaults = CleanRL ``ppo.py``)."""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""


@dataclass
class Args(RunConfig):
    """Full configuration for PPO with discrete actions.

    Inherits run-level fields from :class:`RunConfig` and composes
    :class:`PPOConfig` (algorithm hyperparameters); the network is
    ``network`` + ``network_kwargs`` (see :func:`networks.get_network`).
    """
    algorithm: str = "ppo"
    env_id: str = "LunarLander-v3"
    """the gymnasium environment id (must have a Discrete action space)"""
    num_envs: int = 4
    """the number of parallel game environments"""

    # Nested configs
    network: str = "DiscreteActorCritic"
    """network class to build (see ``RunConfig.network``); any class with
    ``get_value``, ``get_action_and_value`` and ``act`` works"""
    ppo: PPOConfig = field(default_factory=PPOConfig)
    """PPO hyperparameters — the field name deliberately equals the algorithm
    name, so the YAML section, saved run configs, and CLI override paths all
    use one spelling"""

    @property
    def algo(self) -> PPOConfig:
        """Shorthand access to algorithm hyperparameters."""
        return self.ppo


# ---------------------------------------------------------------------------
# The algorithmic core — module-level so tests/test_ppo.py can verify it in
# isolation, before any training run.
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation for a rollout of shape (T, N).

    Uses CleanRL's convention for ``dones``: ``dones[t]`` is the done flag
    *returned by the previous step* — 1.0 means an episode ended right
    before step t, so nothing may bootstrap from step t into step t-1.

    Args:
        rewards: (T, N) reward received after the action at step t.
        values: (T, N) critic estimates V(s_t) recorded during the rollout.
        dones: (T, N) done flag of the step before t (see above).
        next_value: (N,) V of the observation after the last step.
        next_done: (N,) done flag returned by the last step.
        gamma: discount factor.
        gae_lambda: the GAE lambda.

    Returns:
        ``(advantages, returns)``, both (T, N): ``advantages[t] = delta_t +
        gamma * lambda * nonterminal * advantages[t+1]`` with ``delta_t =
        r_t + gamma * nonterminal * V(s_{t+1}) - V(s_t)``, and ``returns =
        advantages + values`` (the critic's regression target).
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 1.0 - next_done
            next_values = next_value
        else:
            nextnonterminal = 1.0 - dones[ t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * nextnonterminal * next_values - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


def compute_clipped_policy_loss(
    newlogprob: torch.Tensor, oldlogprob: torch.Tensor, advantages: torch.Tensor, clip_coef: float
) -> torch.Tensor:
    """PPO's clipped surrogate loss for a minibatch (shape (B,) each).

    ``ratio = pi_new(a|s) / pi_old(a|s)``. The objective is the pessimistic
    minimum of ``ratio * A`` and ``clip(ratio, 1 - clip_coef, 1 + clip_coef)
    * A``; ``oldlogprob`` and ``advantages`` are constants.

    Returns a scalar whose gradient *descent* performs ascent on that
    objective.
    """
    advantages = advantages.detach()
    ratio = torch.exp(newlogprob - oldlogprob.detach())
    surr_unclipped = ratio * advantages
    surr_clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
    return -torch.min(surr_unclipped, surr_clipped).mean()

def approx_kl_and_clipfrac(
    newlogprob: torch.Tensor, oldlogprob: torch.Tensor, clip_coef: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagnostics of how far the policy moved from the one that collected the data.

    Returns ``(approx_kl, clipfrac)``, two scalars that carry no gradient:
    ``approx_kl`` estimates KL(pi_old || pi_new) as the mean of
    ``(ratio - 1) - log(ratio)`` (http://joschu.net/blog/kl-approx.html);
    ``clipfrac`` is the fraction of samples with ``|ratio - 1| > clip_coef``.
    """
    # ===================== YOUR CODE HERE (Part 3) =====================
    raise NotImplementedError("Implement approx_kl_and_clipfrac")
    # ===================================================================


class PPO(Algorithm):
    """Proximal Policy Optimization for discrete action spaces.

    Usage::

        ppo = PPO(args)
        ppo.initialize()          # envs, actor-critic, optimizer, rollout buffers
        ppo.train()               # rollout → GAE → K epochs of clipped minibatch updates
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
        args.minibatch_size = int(args.batch_size // args.algo.num_minibatches)
        args.num_iterations = int(args.total_timesteps) // args.batch_size

        self._setup_logging_and_checkpoints()

        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        self.agent = get_network(args.network)(self.envs, **args.network_kwargs).to(self.device)
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
        self.rewards = torch.zeros((T, N), device=self.device)
        self.dones = torch.zeros((T, N), device=self.device)
        self.values = torch.zeros((T, N), device=self.device)

        # Log ~every 1000 env steps
        self.log_every_iters = max(1, 1000 // args.batch_size)

        self.start_time = time.time()
        if not self.resuming:
            self.global_step = 0
            self.num_updates = 0
            self.global_ep_counter = 0
            self.recent_ep_returns = deque(maxlen=100)
            self.recent_ep_lengths = deque(maxlen=100)
        # Env simulator state is not checkpointed: (re)start from fresh episodes.
        self.next_obs, _ = self.envs.reset(seed=args.seed + self.global_step)
        self.next_obs = torch.as_tensor(self.next_obs, dtype=torch.float32, device=self.device)
        self.next_done = torch.zeros(N, device=self.device)

    # ------------------------------------------------------------------
    # train — rollout → GAE → K epochs of clipped minibatch updates
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

        with tqdm(range(iter_start, args.num_iterations + 1), desc="PPO iterations", unit="iter") as pbar:
            for iteration in pbar:
                self.iteration = iteration

                # Annealing the rate if instructed to do so.
                if cfg.anneal_lr:
                    frac = 1.0 - (iteration - 1.0) / args.num_iterations
                    self.optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

                # ---------------- rollout: num_steps in every env ----------------
                for step in range(cfg.num_steps):
                    self.global_step += args.num_envs
                    self.obs[step] = self.next_obs
                    self.dones[step] = self.next_done

                    # Unlike A2C, PPO keeps the log-probability and value of
                    # the *collecting* policy: the update compares against them.
                    with torch.no_grad():
                        action, logprob, _, value = agent.get_action_and_value(self.next_obs)
                        self.values[step] = value.flatten()
                    self.actions[step] = action
                    self.logprobs[step] = logprob

                    next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                    next_done = np.logical_or(terminations, truncations)
                    self.rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
                    self.next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                    self.next_done = torch.as_tensor(next_done, dtype=torch.float32, device=device)

                    completed_returns, completed_lengths, _ = episode_completions_from_vector_infos(
                        terminations, truncations, infos
                    )
                    if completed_returns:
                        self.recent_ep_returns.extend(completed_returns)
                        self.recent_ep_lengths.extend(completed_lengths)
                        self.global_ep_counter += len(completed_returns)

                # ---------------- targets: GAE advantages and returns ----------------
                with torch.no_grad():
                    next_value = agent.get_value(self.next_obs).reshape(-1)
                    advantages, returns = compute_gae(
                        self.rewards, self.values, self.dones, next_value, self.next_done, cfg.gamma, cfg.gae_lambda
                    )

                # flatten the batch, dropping the autoreset steps: with
                # gymnasium's next-step autoreset, a step whose previous step
                # ended an episode (dones == 1) only jumps terminal obs ->
                # reset obs, its action is ignored and it carries no signal.
                keep = self.dones.reshape(-1) == 0
                b_obs = self.obs.reshape((-1,) + envs.single_observation_space.shape)
                b_logprobs = self.logprobs.reshape(-1)
                b_actions = self.actions.reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = self.values.reshape(-1)

                # ---------------- update: K epochs over shuffled minibatches ----------------
                b_inds = np.flatnonzero(keep.cpu().numpy())
                clipfracs = []
                for epoch in range(cfg.update_epochs):
                    np.random.shuffle(b_inds)
                    for mb_inds in np.array_split(b_inds, cfg.num_minibatches):
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                            b_obs[mb_inds], b_actions[mb_inds]
                        )
                        approx_kl, clipfrac = approx_kl_and_clipfrac(
                            newlogprob, b_logprobs[mb_inds], cfg.clip_coef
                        )
                        with torch.no_grad():
                            old_approx_kl = (b_logprobs[mb_inds] - newlogprob).mean()
                        clipfracs.append(clipfrac.item())

                        mb_advantages = b_advantages[mb_inds]
                        if cfg.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss = compute_clipped_policy_loss(
                            newlogprob, b_logprobs[mb_inds], mb_advantages, cfg.clip_coef
                        )

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
                        self.optimizer.step()
                        self.num_updates += 1

                    if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                        break

                if iteration % self.log_every_iters == 0:
                    self._log_metrics(
                        pbar,
                        pg_loss=pg_loss,
                        v_loss=v_loss,
                        entropy_loss=entropy_loss,
                        old_approx_kl=old_approx_kl,
                        approx_kl=approx_kl,
                        clipfracs=clipfracs,
                        grad_norm=grad_norm,
                        epochs_run=epoch + 1,
                        b_values=b_values[keep],
                        b_returns=b_returns[keep],
                        b_advantages=b_advantages[keep],
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

    def _log_metrics(
        self, pbar, *, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, grad_norm,
        epochs_run, b_values, b_returns, b_advantages,
    ) -> None:
        """Pure diagnostics — nothing here affects training."""
        args = self.args
        sps = int(self.global_step / (time.time() - self.start_time)) if self.global_step else 0
        postfix: dict = {"sps": sps, "gs": self.global_step, "kl": round(float(approx_kl), 4)}
        if len(self.recent_ep_returns) > 0:
            postfix["r_last100"] = round(float(np.mean(self.recent_ep_returns)), 1)
        pbar.set_postfix(postfix, refresh=False)

        if not args.track:
            return
        import wandb

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        metrics_dict = {
            "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            "losses/policy_loss": pg_loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": float(np.mean(clipfracs)),
            "losses/explained_variance": explained_var,
            "losses/grad_norm": float(grad_norm),
            # GAE advantages before the per-minibatch normalization
            "charts/advantage_mean": b_advantages.mean().item(),
            "charts/advantage_std": b_advantages.std().item() if b_advantages.numel() > 1 else 0.0,
            "charts/epochs_run": epochs_run,
            "charts/num_updates": self.num_updates,
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
    # Checkpoint contract — on-policy, so only network/optimizer/counters
    # ------------------------------------------------------------------

    def checkpoint_state_dict(self) -> dict:
        return {
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "last_iteration": int(self.iteration),
            "num_updates": int(self.num_updates),
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
        self.num_updates = int(state["num_updates"])
        self.next_special_log_step = int(state["next_special_log_step"])
        self.next_checkpoint_step = int(state["next_checkpoint_step"])
        self.special_log_every = int(state["special_log_every"])
        self.global_ep_counter = int(state["global_ep_counter"])
        self.recent_ep_returns = deque(state["recent_ep_returns"], maxlen=100)
        self.recent_ep_lengths = deque(state["recent_ep_lengths"], maxlen=100)


def main(args: Args, resume_run_dir: Path | None = None):
    """Entry point — dispatched by ``train.py``."""
    ppo = PPO(args, resume_run_dir)
    ppo.initialize()
    ppo.train()
