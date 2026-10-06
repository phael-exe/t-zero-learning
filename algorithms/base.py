"""Abstract base for training algorithms.

Provides shared setup (seeding, device, run directory, checkpointing,
wandb) so that algorithm files focus on the training loop itself.

Lifecycle: ``__init__`` → ``initialize()`` → ``train()``.
"""

import os
import random
import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from core.base_config import RunConfig
from core.run_naming import string_to_id
from core.checkpoint import (
    apply_rng_state,
    delete_checkpoints,
    latest_checkpoint_path,
    load_checkpoint,
    write_run_config_yaml,
)


# ---------------------------------------------------------------------------
# Standalone checkpoint evaluation (used by scripts and Algorithm.evaluate)
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    model_path: str,
    env_id: str,
    Model: type,
    device: torch.device = torch.device("cpu"),
    eval_episodes: int = 10,
    capture_video: bool = True,
    gamma: float = 0.99,
    experiment_dir: str | None = None,
    run_name: str = "eval",
    env_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
    deterministic: bool = False,
    wrappers=None,
) -> dict:
    """Evaluate a saved model checkpoint.

    *wrappers* is the preprocessing stack the model was trained with (see
    :func:`envs.make_env`); required unless the env has an adapter override.
    *model_kwargs* are the constructor kwargs to rebuild the agent
    (``Model(envs, **model_kwargs)``) and must match training — e.g. for
    actor-critic agents, ``use_obs_norm`` restores frozen normalization
    statistics from the state_dict, never updated here.  The model must
    expose ``act(obs, deterministic=...)`` (the uniform policy interface).

    Returns a dict with ``episodic_returns`` (list[float]) and ``metrics``
    (dict suitable for wandb logging).
    """
    from envs import make_env

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id, 0, capture_video, run_name, gamma,
                experiment_dir, env_kwargs, name_prefix="eval",
                wrappers=wrappers,
            )
        ]
    )
    agent = Model(envs, **(model_kwargs or {})).to(device)
    agent.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns: list[float] = []
    while len(episodic_returns) < eval_episodes:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions = agent.act(obs_t, deterministic=deterministic)
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "_episode" in infos:
            for i, done in enumerate(infos["_episode"]):
                if done:
                    ret = float(infos["episode"]["r"][i])
                    print(
                        f"eval_episode={len(episodic_returns)}, "
                        f"episodic_return={ret}"
                    )
                    episodic_returns.append(ret)
        obs = next_obs

    envs.close()

    mean_ret = float(np.mean(episodic_returns))
    std_ret = float(np.std(episodic_returns))
    return {
        "episodic_returns": episodic_returns,
        "metrics": {
            "eval/mean_return": mean_ret,
            "eval/std_return": std_ret,
        },
    }


class Algorithm(ABC):
    """Lifecycle and checkpoint contract for training algorithms.

    Subclasses override :meth:`initialize` (calling ``super().initialize()``
    first), :meth:`train`, :meth:`checkpoint_state_dict`, and
    :meth:`load_checkpoint_state_dict`.

    ``super().initialize()`` provides:

    - ``self.resuming``, ``self.device``
    - ``self.env_id``, ``self.env_kwargs`` (from the config)
    - ``self.run_dir``, ``self.run_name``, ``self.experiment_dir``
    - Seeded RNGs (random, numpy, torch)

    After building environments and computing ``args.batch_size``, call
    :meth:`_setup_logging_and_checkpoints`.  After creating the agent and
    optimizers, call :meth:`_resume_from_checkpoint` (when resuming).

    Usage::

        algo = MyAlgorithm(args)
        algo.initialize()
        algo.train()
    """

    default_wrappers = None
    """Preprocessing stack ``(env, env_id) -> env`` this algorithm assumes.

    Every algorithm must declare the wrapper stack matching its agent's input
    contract (see :mod:`envs.wrappers`).  Per-env adapter overrides take
    precedence.  Leaving this ``None`` makes env construction fail loudly.
    """

    def __init__(self, args: RunConfig, resume_run_dir: Path | None = None):
        self.args = args
        self.resume_run_dir = resume_run_dir

    # ------------------------------------------------------------------
    # initialize — shared setup (subclass calls super first)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Set up the run directory, seed RNGs, and select the device.

        Subclasses should call ``super().initialize()`` at the start of
        their own ``initialize()`` and then build environments, networks,
        optimizers, and rollout buffers.
        """
        args = self.args

        # Environment
        self.env_id = args.env_id
        self.env_kwargs = dict(args.env_kwargs or {})
        # Preprocessing stack: config-named stack, or this algorithm's default
        # (per-env adapter overrides are applied later, inside make_env).
        from envs.wrappers import resolve_wrapper_stack

        self.wrappers = resolve_wrapper_stack(
            getattr(args, "env_wrappers", ""), self.default_wrappers
        )
        # Short stable hash of the kwargs for run names; the full kwargs
        # are preserved in the saved config YAML.
        kwargs_json = (
            json.dumps(self.env_kwargs, sort_keys=True) if self.env_kwargs else ""
        )
        args.env_kwargs_id = string_to_id(kwargs_json)

        # Run directory
        self.resuming = self.resume_run_dir is not None
        if self.resuming:
            self.run_dir = Path(self.resume_run_dir).resolve()
            self.run_name = self.run_dir.name
            self.experiment_dir = str(self.run_dir.parent)
        else:
            self.experiment_dir = f"runs/{args.exp_name}"
            os.makedirs(self.experiment_dir, exist_ok=True)
            # Env ids may contain '/' (e.g. "Meta-World/MT10"); flatten so
            # the run name stays a single directory level.
            env_name = self.env_id.replace("/", "-")
            self.run_name = (
                f"{env_name}__{args.env_kwargs_id}__{args.seed}"
                f"__{time.strftime('%Y%m%d_%H%M%S')}"
            )
            self.run_dir = Path(self.experiment_dir) / self.run_name

        # Seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )

    # ------------------------------------------------------------------
    # Helpers for subclass initialize()
    # ------------------------------------------------------------------

    def _setup_logging_and_checkpoints(self) -> None:
        """Checkpoint scheduling, config YAML write, and wandb init.

        Call after ``args.batch_size`` is set.
        """
        args = self.args

        if not self.resuming and (args.save_model or int(args.checkpoint_every) > 0):
            write_run_config_yaml(self.experiment_dir, self.run_name, args)

        self.checkpoint_every = int(args.checkpoint_every)
        self.next_checkpoint_step = (
            self.checkpoint_every if self.checkpoint_every > 0 else 0
        )
        self.checkpoints_keep = (
            max(1, int(args.checkpoints_to_keep))
            if self.checkpoint_every > 0
            else 0
        )

        # Special logging cadence (at least one full rollout/update apart)
        self.special_log_every = max(
            int(args.special_log_every), int(args.batch_size)
        )
        self.next_special_log_step = self.special_log_every

        if args.track:
            import wandb
            wandb.init(
                project=os.environ.get("WANDB_PROJECT"),
                entity=os.environ.get("WANDB_ENTITY"),
                config=asdict(args),
                name=self.run_name,
                save_code=True,
            )

    def _resume_from_checkpoint(self) -> None:
        """Load the latest checkpoint, restore algorithm state and RNG.

        Call after agent and optimizers are created so that
        :meth:`load_checkpoint_state_dict` can populate them.

        Raises :class:`FileNotFoundError` if no checkpoint is found.
        """
        args = self.args
        ckpt_path = latest_checkpoint_path(self.run_dir)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"no checkpoint_gs*.pt found under {self.run_dir}"
            )
        print(f"Resuming from {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        algo_state = ckpt["algorithm"]
        # global_step lives in the envelope, not the algorithm payload
        algo_state["global_step"] = ckpt["global_step"]
        self.load_checkpoint_state_dict(algo_state)
        apply_rng_state(ckpt["rng"], cuda=torch.cuda.is_available() and args.cuda)

    # ------------------------------------------------------------------
    # Evaluation — override for env-specific protocols
    # ------------------------------------------------------------------

    def evaluate(self, model_path, eval_episodes=10, deterministic=False):
        """Evaluate a saved model.

        Envs that declare a custom evaluation protocol (see :mod:`envs.adapters`)
        use it; otherwise the standard single-env evaluation loop runs.
        """
        from envs.adapters import get_adapter

        adapter = get_adapter(self.env_id)
        if adapter.evaluate is not None:
            return adapter.evaluate(self, model_path, eval_episodes, deterministic)

        args = self.args
        return evaluate_checkpoint(
            model_path=model_path,
            env_id=self.env_id,
            Model=type(self.agent),
            device=self.device,
            eval_episodes=eval_episodes,
            capture_video=args.capture_video,
            gamma=args.algo.gamma,
            experiment_dir=self.experiment_dir,
            run_name=self.run_name,
            env_kwargs=self.env_kwargs,
            model_kwargs=args.network_kwargs,
            deterministic=deterministic,
            wrappers=self.wrappers,
        )

    def _post_training_eval(self):
        """Save the final model, run evaluation, and log metrics."""
        args = self.args
        if not args.save_model:
            return

        model_path = str(self.run_dir / "model.pt")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        delete_checkpoints(self.run_dir)

        results = self.evaluate(model_path)

        if args.track and results.get("metrics"):
            import wandb

            wandb.log(results["metrics"], step=self.global_step)

    # ------------------------------------------------------------------
    # Abstract methods — subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self) -> None:
        """Run the training loop to completion and clean up environments."""
        ...

    @abstractmethod
    def checkpoint_state_dict(self) -> dict:
        """Build the algorithm-specific payload for a checkpoint.

        The generic :mod:`checkpoint` machinery wraps this inside
        ``{"algorithm": ...}`` alongside RNG state and ``global_step``.
        All resumable state should live on ``self`` so no arguments are
        needed.
        """
        ...

    @abstractmethod
    def load_checkpoint_state_dict(self, state: dict) -> None:
        """Restore algorithm-specific state from a checkpoint payload."""
        ...
