"""Shared helpers for smoke tests.

Smoke tests are driven by the ``ALGORITHMS`` registry: every registered
algorithm must have an entry in :data:`SMOKE_SETTINGS` below (a guard test
fails otherwise).  When you add an algorithm, add its cheapest-possible
settings here and the train/resume smoke tests pick it up automatically.

Rules for a smoke entry:
- pick the cheapest env with the right space types (Box/Box here; a future
  DQN would use a discrete-action env like CartPole-v1)
- keep ``total_timesteps`` an exact multiple of the algorithm's rollout /
  update size so the final checkpoint lands exactly on ``total_timesteps``
- configs are built programmatically — never commit smoke configs under
  ``configs/``
"""
from __future__ import annotations

from typing import Any

from core.config_loader import _import_algorithm

# algorithm name -> smoke-run settings
#   env_id:          cheap env compatible with the algorithm's action space
#   run_overrides:   fields on Args (RunConfig level)
#   algo_overrides:  fields on args.algo (the algorithm's own config section)
SMOKE_SETTINGS: dict[str, dict[str, Any]] = {
    "ppo_continuous_action": {
        "env_id": "Pendulum-v1",
        "run_overrides": {"total_timesteps": 256, "num_envs": 2},
        "algo_overrides": {"num_steps": 32, "num_minibatches": 4, "update_epochs": 2},
    },
    "ppo_continuous_action_split_optim": {
        "env_id": "Pendulum-v1",
        "run_overrides": {"total_timesteps": 256, "num_envs": 2},
        "algo_overrides": {"num_steps": 32, "num_minibatches": 4, "update_epochs": 2},
    },
    "a2c": {
        "env_id": "CartPole-v1",
        "run_overrides": {"total_timesteps": 256, "num_envs": 2},
        "algo_overrides": {"num_steps": 8},
    },
    "ppo": {
        "env_id": "CartPole-v1",
        "run_overrides": {"total_timesteps": 256, "num_envs": 2},
        "algo_overrides": {"num_steps": 32, "num_minibatches": 4, "update_epochs": 2},
    },
}


def make_smoke_args(algo_name: str, **overrides: Any):
    """Return (args, main_fn) for *algo_name* with tiny smoke-run settings."""
    settings = SMOKE_SETTINGS[algo_name]
    ArgsClass, main = _import_algorithm(algo_name)

    args = ArgsClass()
    args.algorithm = algo_name
    args.exp_name = "smoke"
    args.env_id = settings["env_id"]
    args.seed = 1
    args.cuda = False
    args.track = False
    args.capture_video = False
    args.save_model = False
    args.checkpoint_every = 64
    args.checkpoints_to_keep = 2
    for key, value in settings.get("run_overrides", {}).items():
        setattr(args, key, value)
    for key, value in settings.get("algo_overrides", {}).items():
        setattr(args.algo, key, value)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args, main


def only_run_dir(tmp_path):
    """Return the single run directory created under ``tmp_path/runs/smoke``."""
    exp_dir = tmp_path / "runs" / "smoke"
    run_dirs = [p for p in exp_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, f"expected exactly one run dir, got {run_dirs}"
    return run_dirs[0]
