#!/usr/bin/env python3
"""
Evaluate saved checkpoints (any registered algorithm) and record eval videos.

Pass either:
  - a **single run directory** (the folder that directly holds ``model.pt``
    and ``config.yml``);
  - an **experiment directory** (e.g. ``runs/ppo_antdir_goal_v4_8dir``): every immediate
    child folder that contains those files is processed.

Everything needed to rebuild the eval env (env id, kwargs, wrapper stack, network
architecture) is read from the run's saved ``config.yml``; the run's
``algorithm`` supplies the default network and wrapper stack when the config
leaves them unset, exactly as at training time.

Examples (headless-friendly)::

  MUJOCO_GL=egl python evaluate.py runs/my_exp/EnvId__hash__seed__ts --deterministic
  MUJOCO_GL=egl python evaluate.py runs/my_exp --eval-episodes 5 --no-video

Envs whose adapter declares a custom evaluation protocol (e.g. Meta-World vector
benchmarks MT10/MT25/MT50) are not supported here yet and are rejected — use
``scripts/eval_metaworld.py`` for Meta-World runs instead.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
from pathlib import Path

import torch
import yaml

from algorithms.base import evaluate_checkpoint
from envs import make_env  # noqa: F401 — registers custom envs/adapters on import
from envs.adapters import get_adapter
from envs.wrappers import resolve_wrapper_stack
from core.config_loader import ALGORITHMS, translate_legacy_agent_section
from networks import get_network

MODEL_FILE = "model.pt"
CONFIG_FILE = "config.yml"
LEGACY_ALGORITHM = "ppo_continuous_action"
"""assumed for saved configs written before the ``algorithm`` key existed"""


def iter_checkpoint_run_dirs(run_path: Path) -> list[Path]:
    """Single run folder, or experiment root containing multiple run subfolders."""
    run_path = run_path.resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"not a directory: {run_path}")
    if (run_path / MODEL_FILE).is_file() and (run_path / CONFIG_FILE).is_file():
        return [run_path]
    found: list[Path] = []
    for child in sorted(run_path.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if (child / MODEL_FILE).is_file() and (child / CONFIG_FILE).is_file():
            found.append(child)
    return found


def algorithm_defaults(algo_name: str) -> tuple[str, object]:
    """Return ``(default network name, default wrapper stack)`` for *algo_name*.

    Read from the registered algorithm class and its ``Args`` dataclass, so a
    run evaluates with the same defaults it was trained with.
    """
    if algo_name not in ALGORITHMS:
        raise ValueError(
            f"unknown algorithm {algo_name!r}; available: {', '.join(ALGORITHMS)}"
        )
    module_path, cls_name, args_cls_name = ALGORITHMS[algo_name]
    mod = importlib.import_module(module_path)
    network_field = next(
        f for f in dataclasses.fields(getattr(mod, args_cls_name)) if f.name == "network"
    )
    return network_field.default, getattr(mod, cls_name).default_wrappers


def parse_run_config(cfg: dict) -> dict | None:
    """Extract eval-relevant settings from a saved run ``config.yml``.

    Handles the current format (``network`` + ``network_kwargs``, a section
    named after the algorithm, ``env_id`` + ``env_kwargs``) and falls back to
    legacy formats (``agent:`` section; pre-rename ``ppo:`` section; flat
    top-level ``activation`` / ``gamma``, ``task: "EnvId=>json"``).
    Returns None when no env id can be determined.
    """
    env_id = cfg.get("env_id")
    env_kwargs = dict(cfg.get("env_kwargs") or {})
    if not env_id and "task" in cfg:  # legacy: task: "EnvId=>{json kwargs}"
        task = str(cfg["task"])
        env_id, _, kwargs_json = task.partition("=>")
        env_id = env_id.strip()
        if kwargs_json.strip():
            env_kwargs = json.loads(kwargs_json)
    if not env_id:
        return None

    if "agent" not in cfg and "network_kwargs" not in cfg:  # legacy: flat top-level keys
        cfg = {**cfg, "agent": {k: cfg[k] for k in ("activation", "hidden_layers_size") if k in cfg}}
    algo_name = cfg.get("algorithm") or LEGACY_ALGORITHM
    default_network, default_wrappers = algorithm_defaults(algo_name)
    cfg = translate_legacy_agent_section(cfg, default_network)
    algo = cfg.get(algo_name) or cfg.get("ppo") or {}
    return {
        "env_id": env_id,
        "env_kwargs": env_kwargs,
        "Model": get_network(cfg.get("network") or default_network),
        "model_kwargs": dict(cfg.get("network_kwargs") or {}),
        "gamma": float(algo.get("gamma", cfg.get("gamma", 0.99))),
        "wrappers": resolve_wrapper_stack(
            cfg.get("env_wrappers", ""), default_wrappers
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_path",
        type=Path,
        help="a single run directory, or an experiment root (runs/...) holding several",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu | cuda (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1,
        help="number of eval episodes per run (default: 1)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use policy mean actions (no sampling) for smoother behaviour/videos",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="skip video recording, only print episodic returns",
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_dirs = iter_checkpoint_run_dirs(args.run_path)
    if not run_dirs:
        print(
            f"No runs with {MODEL_FILE} and {CONFIG_FILE} under {args.run_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    for run_dir in run_dirs:
        with open(run_dir / CONFIG_FILE, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        run_cfg = parse_run_config(cfg)
        if run_cfg is None:
            print(f"skip {run_dir}: no 'env_id' in config", file=sys.stderr)
            continue

        if get_adapter(run_cfg["env_id"]).evaluate is not None:
            print(
                f"error: {run_dir.name}: env {run_cfg['env_id']!r} declares a custom "
                "evaluation protocol (env adapter 'evaluate'), which evaluate.py "
                "does not support yet (for Meta-World use scripts/eval_metaworld.py)",
                file=sys.stderr,
            )
            sys.exit(1)

        env_kwargs = run_cfg["env_kwargs"]

        print(
            f"evaluating: {run_dir.name} (env={run_cfg['env_id']}, device={device}, "
            f"episodes={args.eval_episodes}, deterministic={args.deterministic}, "
            f"video={not args.no_video})"
        )
        results = evaluate_checkpoint(
            str(run_dir / MODEL_FILE),
            run_cfg["env_id"],
            run_cfg["Model"],
            device=device,
            eval_episodes=int(args.eval_episodes),
            capture_video=not args.no_video,
            gamma=run_cfg["gamma"],
            experiment_dir=str(run_dir.parent),
            run_name=run_dir.name,
            env_kwargs=env_kwargs,
            model_kwargs=run_cfg["model_kwargs"],
            deterministic=bool(args.deterministic),
            wrappers=run_cfg["wrappers"],
        )
        print(f"  {run_dir.name}: {results['metrics']}")


if __name__ == "__main__":
    main()
