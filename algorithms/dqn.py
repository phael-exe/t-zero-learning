"""DQN (discrete actions) — algorithm class with ``initialize`` / ``train`` lifecycle.

ASSIGNMENT: complete the three blocks marked "YOUR CODE HERE":
    Part 1a: ReplayBuffer.add
    Part 1b: ReplayBuffer.sample
    Part 2:  compute_td_targets
Check your work with:   python -m pytest tests/test_dqn.py
Then train with:        python train.py --config dqn_cartpole

Everything else is a working training harness — you should not need to modify
it, but you are encouraged to read it: the whole algorithm (buffer, TD target,
epsilon-greedy rollout, target network) lives in this one file; the base class
only provides run plumbing (seeding, run dir, wandb, checkpoint scheduling).

Adapted from CleanRL (https://github.com/vwxyzjn/cleanrl),
Copyright (c) 2019 CleanRL developers, MIT License (see LICENSE).
"""
import time
from collections import deque, namedtuple
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

import envs.custom_envs  # noqa: F401 — Gym registration side effects
from envs.custom_envs.envs_utils import episode_completions_from_vector_infos
from envs import build_vector_envs, resolve_training_video_schedule
from envs.wrappers import discrete_control_wrappers
from networks import QNetwork
from core.base_config import AgentConfig, RunConfig
from algorithms.base import Algorithm


@dataclass
class DQNConfig:
    """DQN algorithm hyperparameters for discrete actions."""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate (1.0 = hard copy)"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of samples from the replay memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total_timesteps` it takes from start_e to end_e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


@dataclass
class Args(RunConfig):
    """Full configuration for DQN with discrete actions.

    Inherits run-level fields from :class:`RunConfig` and composes
    :class:`AgentConfig` (network architecture) and :class:`DQNConfig`
    (algorithm hyperparameters).
    """
    algorithm: str = "dqn"
    env_id: str = "CartPole-v1"
    """the gymnasium environment id (must have a Discrete action space)"""

    # Nested configs
    agent: AgentConfig = field(default_factory=lambda: AgentConfig(activation="ReLU", hidden_layers_size=120))
    """network architecture configuration"""
    dqn: DQNConfig = field(default_factory=DQNConfig)
    """DQN hyperparameters — the field name deliberately equals the algorithm
    name, so the YAML section, saved run configs, and CLI override paths all
    use one spelling"""

    @property
    def algo(self) -> DQNConfig:
        """Shorthand access to algorithm hyperparameters."""
        return self.dqn


# ---------------------------------------------------------------------------
# The algorithmic core — module-level so tests/test_dqn.py can verify it in
# isolation, before any training run.
# ---------------------------------------------------------------------------

Batch = namedtuple("Batch", ["observations", "actions", "next_observations", "rewards", "dones"])


class ReplayBuffer:
    """Minimal FIFO experience replay buffer for off-policy learning.

    Contract:
    - Stores at most ``capacity`` transitions. Once full, ``add`` overwrites the
      oldest transition first (circular / FIFO order).
    - ``sample`` draws ``batch_size`` transitions uniformly at random (with
      replacement) from the transitions currently stored — never from empty slots.
    """

    def __init__(self, capacity: int, obs_shape: tuple, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.pos = 0  # index of the next slot to write
        self.size = 0  # number of valid transitions currently stored

    def add(self, obs, next_obs, action: int, reward: float, done: float):
        """Store one transition, overwriting the oldest one when full.

        """
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.pos + 1, self.capacity)
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Sample ``batch_size`` stored transitions uniformly at random.

        Returns a ``Batch`` of torch tensors on ``self.device`` with shapes:
        observations (B, *obs_shape), actions (B, 1) int64,
        next_observations (B, *obs_shape), rewards (B, 1), dones (B, 1).

        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return Batch(
            observations = torch.as_tensor(self.observations[indices], device=self.device),
            next_observations = torch.as_tensor(self.observations[indices], device=self.device),
            next_observations = torch.as_tensor(self.next_observations[indices], device=self.device),
            actions = torch.as_tensor(self.actions[indices], device=self.device),
            rewards = torch.as_tensor(self.rewards[indices], device=self.device),
            dones = torch.as_tensor(self.dones[indices], device=self.device)
        )


def compute_td_targets(target_network, batch: Batch, gamma: float) -> torch.Tensor:
    """Compute the one-step TD target for a batch of transitions.

    """
    with torch.no_grad():
        next_q_values = target_network(batch.next_observations)
        max_next_q = next_q_values.max(dim=1).values
        td_targets = batch.rewards.squeeze(1) + gamma * (1 - batch.dones.squeeze(1)) * max_next_q
    return td_targets


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQN(Algorithm):
    """Deep Q-Network for discrete action spaces.

    Usage::

        dqn = DQN(args)
        dqn.initialize()          # envs, q/target networks, optimizer, buffer
        dqn.train()               # epsilon-greedy rollout → replay → TD update
    """

    default_wrappers = staticmethod(discrete_control_wrappers)
    """Input contract of the flat-vector Q-network: flatten obs, record
    episode statistics, raw rewards (value-based bootstrapping)."""

    def __init__(self, args: Args, resume_run_dir: Path | None = None):
        if resume_run_dir is not None:
            raise NotImplementedError(
                "dqn does not support --resume (the replay buffer is not "
                "checkpointed); start a fresh run instead"
            )
        super().__init__(args, resume_run_dir)

    # ------------------------------------------------------------------
    # initialize — everything before the first training step
    # ------------------------------------------------------------------

    def initialize(self):
        super().initialize()  # run dir, seeding, device, wrapper stack
        args = self.args
        assert int(args.num_envs) == 1, "dqn supports num_envs=1 only"

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

        # DQN updates on sampled minibatches; batch_size doubles as the
        # logging-cadence unit expected by _setup_logging_and_checkpoints.
        args.batch_size = int(args.algo.batch_size)
        args.num_iterations = int(args.total_timesteps)

        self._setup_logging_and_checkpoints()

        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        self.q_network = QNetwork(
            self.envs, args.agent.activation, args.agent.hidden_layers_size
        ).to(self.device)
        self.agent = self.q_network  # base-class hooks (model saving) expect self.agent
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.algo.learning_rate)
        self.target_network = QNetwork(
            self.envs, args.agent.activation, args.agent.hidden_layers_size
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.rb = ReplayBuffer(
            int(args.algo.buffer_size),
            self.envs.single_observation_space.shape,
            self.device,
        )

        self.start_time = time.time()
        self.global_step = 0
        self.global_ep_counter = 0
        self.recent_ep_returns = deque(maxlen=100)
        self.recent_ep_lengths = deque(maxlen=100)

    # ------------------------------------------------------------------
    # train — epsilon-greedy rollout, replay sampling, TD updates
    # ------------------------------------------------------------------

    def train(self):
        args = self.args
        cfg = args.algo
        envs = self.envs
        device = self.device

        obs, _ = envs.reset(seed=args.seed)
        # Gymnasium vector envs autoreset on the step AFTER termination: that
        # step's (obs -> next_obs) pair crosses an episode boundary and must
        # not enter the replay buffer.
        autoreset = np.zeros(args.num_envs, dtype=bool)

        with tqdm(range(int(args.total_timesteps)), desc="DQN steps", unit="step") as pbar:
            for global_step in pbar:
                self.global_step = global_step

                # ALGO LOGIC: epsilon-greedy action selection
                epsilon = linear_schedule(
                    cfg.start_e, cfg.end_e, cfg.exploration_fraction * args.total_timesteps, global_step
                )
                if np.random.random() < epsilon:
                    actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
                else:
                    with torch.no_grad():
                        q_values = self.q_network(torch.as_tensor(obs, dtype=torch.float32, device=device))
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()

                next_obs, rewards, terminations, truncations, infos = envs.step(actions)

                completed_returns, completed_lengths, _ = episode_completions_from_vector_infos(
                    terminations, truncations, infos
                )
                if completed_returns:
                    self.recent_ep_returns.extend(completed_returns)
                    self.recent_ep_lengths.extend(completed_lengths)
                    self.global_ep_counter += len(completed_returns)

                # Store transitions; bootstrapping stops only at true terminations,
                # not truncations (see compute_td_targets).
                for i in range(args.num_envs):
                    if not autoreset[i]:
                        self.rb.add(
                            np.asarray(obs[i], dtype=np.float32),
                            np.asarray(next_obs[i], dtype=np.float32),
                            int(actions[i]),
                            float(rewards[i]),
                            float(terminations[i]),
                        )
                autoreset = np.logical_or(terminations, truncations)
                obs = next_obs

                # ALGO LOGIC: training
                if global_step > cfg.learning_starts and global_step % cfg.train_frequency == 0:
                    data = self.rb.sample(cfg.batch_size)
                    with torch.no_grad():
                        td_target = compute_td_targets(self.target_network, data, cfg.gamma)
                    old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if global_step % 100 == 0:
                        self._log_metrics(pbar, epsilon=epsilon, loss=loss, old_val=old_val)

                # update target network
                if global_step > cfg.learning_starts and global_step % cfg.target_network_frequency == 0:
                    for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                        target_param.data.copy_(cfg.tau * q_param.data + (1.0 - cfg.tau) * target_param.data)

        self._post_training_eval()
        envs.close()

    # ------------------------------------------------------------------
    # Logging — periodic diagnostics (tqdm postfix + wandb)
    # ------------------------------------------------------------------

    def _log_metrics(self, pbar, *, epsilon, loss, old_val) -> None:
        """Pure diagnostics — nothing here affects training."""
        args = self.args
        sps = int(self.global_step / (time.time() - self.start_time)) if self.global_step else 0
        postfix: dict = {"sps": sps, "eps": round(float(epsilon), 3)}
        if len(self.recent_ep_returns) > 0:
            postfix["r_last100"] = round(float(np.mean(self.recent_ep_returns)), 1)
        pbar.set_postfix(postfix, refresh=False)

        if not args.track:
            return
        import wandb

        metrics_dict = {
            "losses/td_loss": loss.item(),
            "losses/q_values": old_val.mean().item(),
            "charts/epsilon": float(epsilon),
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
    # and drives it through QNetwork.act (greedy policy)
    # ------------------------------------------------------------------

    def eval_model_kwargs(self) -> dict:
        return dict(
            activation=self.args.agent.activation,
            hidden_layers_size=self.args.agent.hidden_layers_size,
        )

    # ------------------------------------------------------------------
    # Checkpoint contract — not supported (replay buffer is not persisted)
    # ------------------------------------------------------------------

    def checkpoint_state_dict(self) -> dict:
        raise NotImplementedError(
            "dqn does not support checkpoint/resume; set checkpoint_every: 0"
        )

    def load_checkpoint_state_dict(self, state: dict) -> None:
        raise NotImplementedError("dqn does not support checkpoint/resume")


def main(args: Args, resume_run_dir: Path | None = None):
    """Entry point — dispatched by ``train.py``."""
    dqn = DQN(args, resume_run_dir)
    dqn.initialize()
    dqn.train()
