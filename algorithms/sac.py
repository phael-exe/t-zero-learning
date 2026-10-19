"""SAC (continuous actions) — ``initialize`` / ``train`` lifecycle.

ASSIGNMENT: complete the two blocks marked "YOUR CODE HERE":
    Part 1: squashed_gaussian_sample   (this file)
    Part 2: compute_soft_q_target      (this file)
Check your work with:   python -m pytest tests/test_sac.py
Then train with:        python train.py --config sac_pendulum

Everything else is a working training harness: the replay buffer, the actor
and temperature losses, the target networks, the update schedule and the
logging all live in this one file, and you are encouraged to read it.

The training loop is CleanRL's ``sac_continuous_action.py`` (Haarnoja et al.,
2018, with automatic entropy tuning), with the DQN assignment's harness
around it. Two knobs CleanRL doesn't have, both off by default
(``updates_per_step=1``, ``clipped_double_q=True`` reproduce CleanRL):
``updates_per_step`` (gradient steps per env step) and ``clipped_double_q``
(``false`` uses a single Q-network in the target and in the actor loss).

One deliberate difference from CleanRL, which targets gymnasium 0.29 and
reads ``infos["final_observation"]`` on truncation: gymnasium >= 1.0 resets a
finished env on the *next* ``step`` call. So ``next_obs`` returned with a
termination or truncation already *is* the real final observation, and the
step after it (action ignored, final obs -> reset obs) is kept out of the
buffer, exactly as in ``algorithms/dqn.py``. ``dones`` in the buffer are
``terminations`` only: a time-limit truncation still bootstraps.

Adapted from CleanRL (https://github.com/vwxyzjn/cleanrl),
Copyright (c) 2019 CleanRL developers, MIT License (see LICENSE).
"""
import copy
import time
from collections import deque, namedtuple
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from tqdm.auto import tqdm

import envs.custom_envs  # noqa: F401 — Gym registration side effects
from envs.custom_envs.envs_utils import episode_completions_from_vector_infos
from envs import build_vector_envs, resolve_training_video_schedule
from envs.wrappers import continuous_control_raw_wrappers
from networks import get_network
from core.base_config import RunConfig
from core.checkpoint import delete_checkpoints, save_checkpoint
from algorithms.base import Algorithm


@dataclass
class SACConfig:
    """SAC hyperparameters (defaults = CleanRL ``sac_continuous_action.py``)."""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient"""
    batch_size: int = 256
    """the batch size of samples from the replay memory"""
    learning_starts: int = 5000
    """timestep to start learning (uniform random actions before it)"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network optimizer (also used for alpha)"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed), in Q updates"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks, in Q updates"""
    alpha: float = 0.2
    """entropy regularization coefficient (used when autotune is false)"""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    updates_per_step: int = 1
    """gradient updates (Q, and the delayed actor) per environment step"""
    clipped_double_q: bool = True
    """use min(Q1, Q2) in the target and the actor loss; false uses Q1 alone"""


@dataclass
class Args(RunConfig):
    """Full configuration for SAC with continuous actions.

    Inherits run-level fields from :class:`RunConfig` and composes
    :class:`SACConfig` (algorithm hyperparameters); the network is
    ``network`` + ``network_kwargs`` (see :func:`networks.get_network`).
    """
    algorithm: str = "sac"
    env_id: str = "Hopper-v5"
    """the gymnasium environment id (must have a continuous Box action space)"""
    num_envs: int = 1
    """the number of parallel game environments (SAC here supports 1)"""

    # Nested configs
    network: str = "SACAgent"
    """network class to build (see ``RunConfig.network``); any class with
    ``actor`` (returning mean, log_std; with ``action_scale``/``action_bias``),
    ``qf1``, ``qf2`` and ``act`` works"""
    sac: SACConfig = field(default_factory=SACConfig)
    """SAC hyperparameters — the field name deliberately equals the algorithm
    name, so the YAML section, saved run configs, and CLI override paths all
    use one spelling"""

    @property
    def algo(self) -> SACConfig:
        """Shorthand access to algorithm hyperparameters."""
        return self.sac


# ---------------------------------------------------------------------------
# The algorithmic core — module-level so tests/test_sac.py can verify it in
# isolation, before any training run.
# ---------------------------------------------------------------------------


def squashed_gaussian_sample(
    mean: torch.Tensor, log_std: torch.Tensor, action_scale: torch.Tensor, action_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample an action from a tanh-squashed Gaussian, with its log-probability.

    ``u ~ N(mean, exp(log_std))`` is drawn with the reparameterization trick
    (``mean + std * eps``), so gradients flow from the action back into
    ``mean`` and ``log_std``. Then ``y = tanh(u)`` and
    ``action = y * action_scale + action_bias``. The change of variables gives

        log pi(action) = sum_i [ log N(u_i) - log(action_scale_i * (1 - y_i^2) + 1e-6) ]

    (the 1e-6 keeps the log finite when ``|y|`` reaches 1).

    Args:
        mean, log_std: (B, A) Gaussian parameters *before* the squashing.
        action_scale, action_bias: (A,) maps tanh's (-1, 1) onto [low, high].

    Returns:
        ``(action, log_prob, mean_action)``: the sampled action (B, A),
        its log-probability (B, 1), and the deterministic action
        ``tanh(mean) * action_scale + action_bias`` (B, A).
    """
    # ===================== YOUR CODE HERE (Part 1) =====================
    raise NotImplementedError("Implement squashed_gaussian_sample")
    # ===================================================================


def compute_soft_q_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_q1: torch.Tensor,
    next_q2: torch.Tensor,
    next_log_pi: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """The regression target of both soft Q-networks for a batch (shape (B,) each).

    ``next_q1``/``next_q2`` are the *target* networks' Q(s', a') and
    ``next_log_pi`` is log pi(a'|s'), with a' freshly sampled from the
    current policy at s'. The soft value of s' is the pessimistic Q minus
    the entropy penalty, and a terminal transition does not bootstrap:

        target = r + gamma * (1 - done) * (min(Q1', Q2') - alpha * log pi(a'|s'))

    Returns a (B,) tensor.
    """
    # ===================== YOUR CODE HERE (Part 2) =====================
    raise NotImplementedError("Implement compute_soft_q_target")
    # ===================================================================


def sample_action(actor, obs: torch.Tensor):
    """Run the actor and sample from its squashed Gaussian (Part 1)."""
    mean, log_std = actor(obs)
    return squashed_gaussian_sample(mean, log_std, actor.action_scale, actor.action_bias)


def compute_actor_loss(log_pi: torch.Tensor, q1_pi: torch.Tensor, q2_pi: torch.Tensor, alpha: float) -> torch.Tensor:
    """``mean(alpha * log pi(a|s) - min(Q1(s, a), Q2(s, a)))`` with ``a`` the
    reparameterized sample: minimizing it pushes the actor towards actions
    the critics value, while keeping its entropy up. The gradient reaches the
    actor *through* the Q-networks (only the actor's optimizer steps)."""
    return (alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()


def compute_alpha_loss(log_alpha: torch.Tensor, log_pi: torch.Tensor, target_entropy: float) -> torch.Tensor:
    """Temperature loss ``mean(-alpha * (log pi + target_entropy))``.

    ``-log_pi`` estimates the policy's entropy. When it is below
    ``target_entropy`` the gradient step raises alpha (more weight on
    entropy), and when it is above, it lowers alpha. ``log_pi`` is a
    constant here: only alpha moves.
    """
    return (-log_alpha.exp() * (log_pi.detach() + target_entropy)).mean()


Batch = namedtuple("Batch", ["observations", "actions", "next_observations", "rewards", "dones"])


class ReplayBuffer:
    """FIFO experience replay for continuous actions (same contract as DQN's).

    Stores at most ``capacity`` transitions, overwriting the oldest first;
    ``sample`` draws uniformly with replacement from the stored ones.
    """

    def __init__(self, capacity: int, obs_shape: tuple, act_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, next_obs, action, reward: float, done: float):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        as_t = lambda a: torch.as_tensor(a[idx], device=self.device)  # noqa: E731
        return Batch(as_t(self.observations), as_t(self.actions), as_t(self.next_observations),
                     as_t(self.rewards), as_t(self.dones))

    # Checkpointing: only the filled part is saved, in storage order.
    def state_dict(self) -> dict:
        n = self.size
        return {
            "pos": self.pos, "size": n,
            "observations": self.observations[:n].copy(),
            "next_observations": self.next_observations[:n].copy(),
            "actions": self.actions[:n].copy(),
            "rewards": self.rewards[:n].copy(),
            "dones": self.dones[:n].copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        n = int(state["size"])
        if n > self.capacity:
            raise ValueError(f"checkpointed buffer holds {n} transitions, capacity is {self.capacity}")
        for key in ("observations", "next_observations", "actions", "rewards", "dones"):
            getattr(self, key)[:n] = state[key]
        self.size = n
        self.pos = int(state["pos"]) % self.capacity


class SAC(Algorithm):
    """Soft Actor-Critic for continuous action spaces.

    Usage::

        sac = SAC(args)
        sac.initialize()          # env, actor + twin Q (+ targets), optimizers, buffer
        sac.train()               # act → store → sample → Q / actor / alpha updates
    """

    default_wrappers = staticmethod(continuous_control_raw_wrappers)
    """Flat-vector obs, episode statistics, raw rewards."""

    # ------------------------------------------------------------------
    # initialize — everything before the first training step
    # ------------------------------------------------------------------

    def initialize(self):
        super().initialize()  # run dir, seeding, device, wrapper stack
        args = self.args
        cfg = args.algo
        assert int(args.num_envs) == 1, "sac supports num_envs=1 only"

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
            gamma=cfg.gamma,
            experiment_dir=self.experiment_dir,
            video_every_global_steps=int(args.video_every_global_steps),
            video_length_steps=int(video_length_steps),
            wrappers=self.wrappers,
        )
        args.num_envs = effective_num_envs

        # SAC updates on sampled minibatches; batch_size doubles as the
        # logging-cadence unit expected by _setup_logging_and_checkpoints.
        args.batch_size = int(cfg.batch_size)
        args.num_iterations = int(args.total_timesteps)

        self._setup_logging_and_checkpoints()

        assert isinstance(self.envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
        # the random warm-up actions come from this space: seed it, or runs
        # with the same seed diverge from the first step (CleanRL does this
        # in make_env)
        self.envs.single_action_space.seed(args.seed)

        self.agent = get_network(args.network)(self.envs, **args.network_kwargs).to(self.device)
        self.qf1_target = copy.deepcopy(self.agent.qf1)
        self.qf2_target = copy.deepcopy(self.agent.qf2)
        for p in list(self.qf1_target.parameters()) + list(self.qf2_target.parameters()):
            p.requires_grad_(False)
        self.q_optimizer = optim.Adam(
            list(self.agent.qf1.parameters()) + list(self.agent.qf2.parameters()), lr=cfg.q_lr
        )
        self.actor_optimizer = optim.Adam(list(self.agent.actor.parameters()), lr=cfg.policy_lr)

        # Automatic entropy tuning: target entropy -|A|
        act_dim = int(np.prod(self.envs.single_action_space.shape))
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.a_optimizer = optim.Adam([self.log_alpha], lr=cfg.q_lr)
        self.alpha = self.log_alpha.exp().item() if cfg.autotune else float(cfg.alpha)

        self.rb = ReplayBuffer(
            int(cfg.buffer_size), self.envs.single_observation_space.shape, act_dim, self.device
        )

        self.start_time = time.time()
        self.global_step = 0
        self.num_updates = 0
        self.global_ep_counter = 0
        self.recent_ep_returns = deque(maxlen=100)
        self.recent_ep_lengths = deque(maxlen=100)
        if self.resuming:
            self._resume_from_checkpoint()

    # ------------------------------------------------------------------
    # train — act, store, then Q / actor / alpha updates every env step
    # ------------------------------------------------------------------

    def train(self):
        args = self.args
        cfg = args.algo
        envs = self.envs
        device = self.device
        agent = self.agent
        # single-Q ablation: feed Q1 where the min of the two would go
        pick_q2 = (lambda q1, q2: q2) if cfg.clipped_double_q else (lambda q1, q2: q1)

        if self.global_step >= args.total_timesteps:
            print(f"Training already complete (global_step={self.global_step} >= {args.total_timesteps}).")
            envs.close()
            return

        # Env simulator state is not checkpointed: (re)start from fresh episodes.
        obs, _ = envs.reset(seed=args.seed + self.global_step)
        # Gymnasium vector envs autoreset on the step AFTER termination: that
        # step's (obs -> next_obs) pair crosses an episode boundary and must
        # not enter the replay buffer.
        autoreset = np.zeros(args.num_envs, dtype=bool)
        # Q-estimate tracking (charts/q_minus_return): Q of the first
        # (state, action) of an episode vs the soft return it then collected.
        self._episode_tracking_reset()

        logs = None
        with tqdm(range(self.global_step, int(args.total_timesteps)), desc="SAC steps", unit="step") as pbar:
            for global_step in pbar:
                # ALGO LOGIC: uniform random actions until learning starts
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                policy_log_pi = None
                if global_step < cfg.learning_starts:
                    actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
                else:
                    with torch.no_grad():
                        actions_t, policy_log_pi, _ = sample_action(agent.actor, obs_t)
                    actions = actions_t.cpu().numpy()

                if not autoreset[0]:
                    self._episode_tracking_step(obs_t, actions, policy_log_pi, cfg)

                next_obs, rewards, terminations, truncations, infos = envs.step(actions)
                self.global_step = global_step + 1

                completed_returns, completed_lengths, _ = episode_completions_from_vector_infos(
                    terminations, truncations, infos
                )
                if completed_returns:
                    self.recent_ep_returns.extend(completed_returns)
                    self.recent_ep_lengths.extend(completed_lengths)
                    self.global_ep_counter += len(completed_returns)

                # Store transitions (not the autoreset step). Only true
                # terminations stop bootstrapping, not time-limit truncations.
                for i in range(args.num_envs):
                    if not autoreset[i]:
                        self.rb.add(obs[i], next_obs[i], actions[i], float(rewards[i]), float(terminations[i]))
                        self._episode_tracking_reward(float(rewards[i]), cfg)
                ended = np.logical_or(terminations, truncations)
                if ended[0]:
                    self._episode_tracking_end()
                autoreset = ended
                obs = next_obs

                # ALGO LOGIC: training
                if global_step > cfg.learning_starts:
                    for _ in range(cfg.updates_per_step):
                        logs = self._update(cfg, pick_q2)

                    if global_step % 100 == 0 and logs is not None:
                        self._log_metrics(pbar, logs)

                if self.checkpoint_every > 0 and self.global_step >= self.next_checkpoint_step:
                    while self.next_checkpoint_step <= self.global_step:
                        self.next_checkpoint_step += self.checkpoint_every
                    save_checkpoint(
                        self.run_dir, self.global_step,
                        self.checkpoint_state_dict(), self.checkpoints_keep,
                    )

        self._post_training_eval()
        envs.close()

    def _update(self, cfg, pick_q2) -> dict:
        """One Q update; plus, every ``policy_frequency`` Q updates, as many
        actor (and alpha) updates to compensate for the delay; plus the Polyak
        update of the targets. Returns the Q values to log (the actor's are
        kept in ``self._last_actor_logs``)."""
        agent = self.agent
        data = self.rb.sample(cfg.batch_size)

        with torch.no_grad():
            next_actions, next_log_pi, _ = sample_action(agent.actor, data.next_observations)
            qf1_next = self.qf1_target(data.next_observations, next_actions).view(-1)
            qf2_next = self.qf2_target(data.next_observations, next_actions).view(-1)
            next_q_value = compute_soft_q_target(
                data.rewards, data.dones, qf1_next, pick_q2(qf1_next, qf2_next),
                next_log_pi.view(-1), self.alpha, cfg.gamma,
            )

        qf1_a_values = agent.qf1(data.observations, data.actions).view(-1)
        qf2_a_values = agent.qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()
        self.num_updates += 1

        logs = {
            "losses/qf1_values": qf1_a_values.mean().item(),
            "losses/qf2_values": qf2_a_values.mean().item(),
            "losses/qf1_loss": qf1_loss.item(),
            "losses/qf2_loss": qf2_loss.item(),
            "losses/qf_loss": qf_loss.item() / 2.0,
        }

        if self.num_updates % cfg.policy_frequency == 0:  # TD3 delayed update support
            for _ in range(cfg.policy_frequency):  # compensate for the delay
                pi, log_pi, _ = sample_action(agent.actor, data.observations)
                qf1_pi = agent.qf1(data.observations, pi)
                qf2_pi = agent.qf2(data.observations, pi)
                actor_loss = compute_actor_loss(log_pi, qf1_pi, pick_q2(qf1_pi, qf2_pi), self.alpha)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if cfg.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = sample_action(agent.actor, data.observations)
                    alpha_loss = compute_alpha_loss(self.log_alpha, log_pi, self.target_entropy)

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()
            self._last_actor_logs = {"losses/actor_loss": actor_loss.item(), "losses/entropy": -log_pi.mean().item()}
            if cfg.autotune:
                self._last_actor_logs["losses/alpha_loss"] = alpha_loss.item()

        # update the target networks
        if self.num_updates % cfg.target_network_frequency == 0:
            with torch.no_grad():
                for net, target in ((agent.qf1, self.qf1_target), (agent.qf2, self.qf2_target)):
                    for param, target_param in zip(net.parameters(), target.parameters()):
                        target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

        return logs

    # ------------------------------------------------------------------
    # Q-estimate tracking — pure diagnostics, nothing here affects training
    # ------------------------------------------------------------------
    #
    # For each episode whose first action came from the policy, record the
    # Q-estimate the target uses (min(Q1, Q2), or Q1 alone in the single-Q
    # ablation) of that first (s0, a0), and accumulate what the episode then
    # actually collected, in the same *soft* units the Q-functions learn:
    #     G = sum_t gamma^t r_t  -  alpha * sum_{t>=1} gamma^t log pi(a_t|s_t)
    # At the episode end, Q(s0, a0) - G is logged: positive = overestimation.
    # G ignores what would have followed a time-limit truncation, which is
    # negligible only when episodes are long compared to 1 / (1 - gamma)
    # (Hopper: 1000 steps, gamma^1000 ~ 4e-5; not Pendulum: 200 steps).

    def _episode_tracking_reset(self):
        self._ep = None

    def _episode_tracking_step(self, obs_t, actions, log_pi, cfg):
        if log_pi is None:  # random-action phase: nothing to compare
            return
        if self._ep is None:  # first policy step of the episode
            with torch.no_grad():
                a = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
                q1 = self.agent.qf1(obs_t, a).item()
                q = min(q1, self.agent.qf2(obs_t, a).item()) if cfg.clipped_double_q else q1
            self._ep = {"q0": q, "ret": 0.0, "disc": 1.0}
        else:
            self._ep["ret"] -= self._ep["disc"] * self.alpha * float(log_pi.sum())

    def _episode_tracking_reward(self, reward, cfg):
        if self._ep:
            self._ep["ret"] += self._ep["disc"] * reward
            self._ep["disc"] *= cfg.gamma

    def _episode_tracking_end(self):
        if self._ep:
            self._q_tracking = {
                "charts/q_estimate_ep_start": self._ep["q0"],
                "charts/soft_return_ep_start": self._ep["ret"],
                "charts/q_minus_return": self._ep["q0"] - self._ep["ret"],
            }
        self._ep = None

    # ------------------------------------------------------------------
    # Logging — periodic diagnostics (tqdm postfix + wandb)
    # ------------------------------------------------------------------

    def _log_metrics(self, pbar, logs: dict) -> None:
        """Pure diagnostics — nothing here affects training."""
        args = self.args
        sps = int(self.global_step / (time.time() - self.start_time)) if self.global_step else 0
        postfix: dict = {"sps": sps, "alpha": round(self.alpha, 4)}
        if len(self.recent_ep_returns) > 0:
            postfix["r_last100"] = round(float(np.mean(self.recent_ep_returns)), 1)
        pbar.set_postfix(postfix, refresh=False)

        if not args.track:
            return
        import wandb

        metrics_dict = dict(logs)
        metrics_dict.update(getattr(self, "_last_actor_logs", {}))
        metrics_dict.update(getattr(self, "_q_tracking", {}))
        self._q_tracking = {}
        metrics_dict.update({
            "losses/alpha": self.alpha,
            "charts/target_entropy": self.target_entropy,
            "charts/num_updates": self.num_updates,
            "charts/buffer_size": self.rb.size,
            "charts/SPS": sps,
            "global_step": self.global_step,
        })
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
    # Evaluation — SAC is reported with the deterministic (mean) action
    # ------------------------------------------------------------------

    def _post_training_eval(self):
        """Save the final model; evaluate it with the mean action
        (``eval/mean_return``) and with sampled actions
        (``eval/mean_return_sampled``)."""
        args = self.args
        if not args.save_model:
            return

        model_path = str(self.run_dir / "model.pt")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        delete_checkpoints(self.run_dir)

        metrics = dict(self.evaluate(model_path, deterministic=True)["metrics"])
        capture_video, args.capture_video = args.capture_video, False  # one eval video is enough
        try:
            sampled = self.evaluate(model_path, deterministic=False)["metrics"]
        finally:
            args.capture_video = capture_video
        metrics.update({f"{k}_sampled": v for k, v in sampled.items()})
        print(", ".join(f"{k}={v:.1f}" for k, v in metrics.items()))

        if args.track:
            import wandb

            wandb.log(metrics, step=self.global_step)

    # ------------------------------------------------------------------
    # Checkpoint contract — off-policy, so the replay buffer goes in too
    # ------------------------------------------------------------------

    def checkpoint_state_dict(self) -> dict:
        return {
            "agent_state_dict": self.agent.state_dict(),
            "qf1_target_state_dict": self.qf1_target.state_dict(),
            "qf2_target_state_dict": self.qf2_target.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "a_optimizer_state_dict": self.a_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha": float(self.alpha),
            "replay_buffer": self.rb.state_dict(),
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
        self.qf1_target.load_state_dict(state["qf1_target_state_dict"])
        self.qf2_target.load_state_dict(state["qf2_target_state_dict"])
        self.q_optimizer.load_state_dict(state["q_optimizer_state_dict"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
        self.a_optimizer.load_state_dict(state["a_optimizer_state_dict"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.alpha = float(state["alpha"])
        self.rb.load_state_dict(state["replay_buffer"])
        self.global_step = int(state.get("global_step", 0))
        self.num_updates = int(state["num_updates"])
        self.next_special_log_step = int(state["next_special_log_step"])
        self.next_checkpoint_step = int(state["next_checkpoint_step"])
        self.special_log_every = int(state["special_log_every"])
        self.global_ep_counter = int(state["global_ep_counter"])
        self.recent_ep_returns = deque(state["recent_ep_returns"], maxlen=100)
        self.recent_ep_lengths = deque(state["recent_ep_lengths"], maxlen=100)


def main(args: Args, resume_run_dir: Path | None = None):
    """Entry point — dispatched by ``train.py``."""
    sac = SAC(args, resume_run_dir)
    sac.initialize()
    sac.train()
