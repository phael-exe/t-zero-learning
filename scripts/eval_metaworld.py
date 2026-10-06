#!/usr/bin/env python3
"""
Post-training evaluation for Meta-World vector benchmarks (MT10, MT25, MT50).

Loads a PPO checkpoint trained on ``gym.make_vec("Meta-World/MT10", ...)`` and evaluates
the same policy on each task using ``Meta-World/MT1`` (one task at a time) with matching
one-hot task IDs. Per-task eval videos are written under ``<run_dir>/videos/`` when
``capture_video`` is True (same pattern as ``evaluate_checkpoint``).

Usage from repo root::

  PYTHONNOUSERSITE=1 MUJOCO_GL=egl python scripts/eval_metaworld.py runs/ppo_metaworld_mt10/.../run_name --deterministic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Repo root (parent of scripts/)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Ensure custom envs + metaworld registrations
import envs.custom_envs  # noqa: F401
try:
    import metaworld  # noqa: F401
    from metaworld import env_dict as _mw
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        f"metaworld is required. Install: pip install metaworld. Import error: {e!r}"
    ) from e

import gymnasium as gym

from core.config_loader import translate_legacy_agent_section
from networks import ContinuousActorCritic, get_network
from envs.custom_envs.envs_utils import episode_completions_from_vector_infos

MODEL_FILE = "model.pt"
CONFIG_FILE = "config.yml"

# Task order must match `make_mt_envs` (OrderedDict key order) for correct one-hot `env_id`.
BENCHMARK_TO_TASKS: dict[str, list[str]] = {
    "Meta-World/MT10": list(_mw.MT10_V3.keys()),
    "Meta-World/MT25": list(_mw.MT25_V3.keys()),
    "Meta-World/MT50": list(_mw.MT50_V3.keys()),
}


def iter_checkpoint_run_dirs(run_path: Path) -> list[Path]:
    run_path = run_path.resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"not a directory: {run_path}")
    model = run_path / MODEL_FILE
    cfg = run_path / CONFIG_FILE
    if model.is_file() and cfg.is_file():
        return [run_path]
    found: list[Path] = []
    for child in sorted(run_path.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if (child / MODEL_FILE).is_file() and (child / CONFIG_FILE).is_file():
            found.append(child)
    return found


# Keys that apply to ``gym.make("Meta-World/MT1", ...)`` but not the vector make_vec.
_MT1_EXCLUDE = frozenset(
    {
        "vector_strategy",
    }
)


def _mt1_env_kwargs(
    base: dict[str, Any],
    env_name: str,
    task_index: int,
    num_tasks: int,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "env_name": env_name,
        "use_one_hot": base.get("use_one_hot", True),
        "num_tasks": num_tasks,
        "env_id": task_index,
        "seed": seed,
    }
    for k, v in base.items():
        if k in _MT1_EXCLUDE:
            continue
        if k in out:
            continue
        if k in ("use_one_hot", "env_name", "num_tasks", "env_id", "seed"):
            continue
        out[k] = v
    return out


def _raw_mt1_thunk(
    capture_video: bool,
    experiment_dir: str,
    run_name: str,
    name_prefix: str,
    env_kwargs: dict[str, Any],
):
    """
    Build a single ``Meta-World/MT1`` env without the extra ``make_env`` stack (obs/reward
    clipping, etc.) so the observation distribution matches training on
    ``gym.make_vec("Meta-World/MT10", ...)``.
    """
    def thunk():
        kw = dict(env_kwargs)
        if capture_video:
            kw["render_mode"] = "rgb_array"
        env = gym.make("Meta-World/MT1", **kw)
        if capture_video:
            video_dir = f"{experiment_dir}/{run_name}/videos/"
            os.makedirs(video_dir, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_dir,
                episode_trigger=lambda episode_id: episode_id == 0,
                name_prefix=name_prefix,
            )
        return env

    return thunk


def _run_eval_episodes(
    envs: gym.vector.VectorEnv,
    agent: torch.nn.Module,
    device: torch.device,
    eval_episodes: int,
    deterministic: bool,
) -> tuple[list[float], list[float]]:
    returns: list[float] = []
    successes: list[float] = []
    obs, _ = envs.reset()
    while len(returns) < eval_episodes:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            # get_action_and_value normalizes obs internally when the agent
            # was trained with use_obs_norm (frozen stats from the state_dict).
            actions, _, _, _ = agent.get_action_and_value(  # type: ignore[operator]
                obs_t, deterministic=deterministic
            )
        next_obs, _, terminations, truncations, infos = envs.step(actions.cpu().numpy())
        rets, _lens, sucs = episode_completions_from_vector_infos(
            terminations, truncations, infos
        )
        for j, r in enumerate(rets):
            if len(returns) >= eval_episodes:
                break
            returns.append(r)
            if j < len(sucs):
                successes.append(sucs[j])
            else:
                successes.append(float("nan"))
        obs = next_obs
    return returns, successes


def _eval_one_task(
    model_path: str,
    num_tasks: int,
    task_index: int,
    env_name: str,
    task_json: dict[str, Any],
    Model: type,
    model_kwargs: dict[str, Any],
    device: torch.device,
    capture_video: bool,
    experiment_dir: str,
    run_name: str,
    eval_episodes: int,
    seed: int,
    deterministic: bool,
) -> tuple[list[float], list[float]]:
    safe = env_name.replace("/", "-")
    env_kwargs = _mt1_env_kwargs(
        task_json, env_name, task_index, num_tasks, seed=seed
    )
    envs = gym.vector.SyncVectorEnv(
        [
            _raw_mt1_thunk(
                capture_video,
                experiment_dir,
                run_name,
                f"eval-{safe}",
                env_kwargs,
            )
        ]
    )
    try:
        agent = Model(envs, **model_kwargs).to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()
        return _run_eval_episodes(
            envs, agent, device, eval_episodes, deterministic
        )
    finally:
        envs.close()


def _nan_to_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(x) for x in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def evaluate_metaworld(
    model_path: str | Path,
    env_id: str,
    env_kwargs: dict[str, Any] | None,
    device: torch.device,
    experiment_dir: str,
    run_name: str,
    eval_episodes: int = 10,
    capture_video: bool = True,
    deterministic: bool = False,
    base_seed: int = 0,
    save_json: bool = True,
    Model: type = ContinuousActorCritic,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Evaluate a checkpoint trained on MT10/MT25/MT50 on every constituent task via ``Meta-World/MT1``.

    The agent is rebuilt as ``Model(envs, **model_kwargs)`` — the run's
    ``network`` class and ``network_kwargs``.

    Returns a summary dict, or None if *env_id* is not a vector MT benchmark.
    """
    model_path = Path(model_path)
    task_json = dict(env_kwargs or {})
    if env_id not in BENCHMARK_TO_TASKS:
        print(
            f"eval_metaworld: env_id {env_id!r} is not a vector benchmark "
            f"{sorted(BENCHMARK_TO_TASKS)}; skip."
        )
        return None

    task_list = BENCHMARK_TO_TASKS[env_id]
    num_tasks = len(task_list)
    if num_tasks == 0:
        return None

    per_task_return: dict[str, float] = {}
    per_task_success: dict[str, float] = {}
    per_task_all_returns: dict[str, list[float]] = {}
    all_returns: list[float] = []
    all_successes: list[float] = []

    run_dir = Path(experiment_dir) / run_name

    for i, tname in enumerate(task_list):
        print(f"eval: task {i + 1}/{num_tasks} {tname!r} ...", flush=True)
        rets, sucs = _eval_one_task(
            str(model_path),
            num_tasks,
            i,
            tname,
            task_json,
            Model=Model,
            model_kwargs=dict(model_kwargs or {}),
            device=device,
            capture_video=capture_video,
            experiment_dir=experiment_dir,
            run_name=run_name,
            eval_episodes=eval_episodes,
            seed=base_seed + 1000 * i,
            deterministic=deterministic,
        )
        per_task_all_returns[tname] = rets
        per_task_return[tname] = float(np.mean(rets))
        ok = [s for s in sucs if not np.isnan(s)]
        per_task_success[tname] = float(np.mean(ok)) if ok else float("nan")
        all_returns.extend(rets)
        all_successes.extend(s for s in sucs if not np.isnan(s))

    out: dict[str, Any] = {
        "env_id": env_id,
        "per_task_return": per_task_return,
        "per_task_success": per_task_success,
        "per_task_all_returns": per_task_all_returns,
        "mean_return": float(np.mean(all_returns)) if all_returns else float("nan"),
        "mean_success_rate": float(np.mean(all_successes)) if all_successes else float("nan"),
    }

    if save_json:
        p = run_dir / "metaworld_eval.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(_nan_to_none(out), f, indent=2)
        print(f"Wrote {p}", flush=True)

    print("--- Meta-World eval summary ---", flush=True)
    for t in task_list:
        pr = per_task_return[t]
        ps = per_task_success[t]
        ps_s = f"{ps:.3f}" if not (isinstance(ps, float) and np.isnan(ps)) else "nan"
        print(
            f"  {t}: return={pr:.3f} success={ps_s}",
            flush=True,
        )
    print(
        f"  mean_return={out['mean_return']:.3f} mean_success_rate={out['mean_success_rate']:.3f}",
        flush=True,
    )
    return out


def _load_ppo_config(run_dir: Path) -> dict[str, Any]:
    p = run_dir / CONFIG_FILE
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_path",
        type=Path,
        help="Experiment root or single run directory with model and config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu | cuda (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10, help="episodes per task (default: 10)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use policy mean (deterministic) actions",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="do not record videos (faster)"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="do not write metaworld_eval.json under the run folder",
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_dirs = iter_checkpoint_run_dirs(args.run_path)
    if not run_dirs:
        print(
            f"No runs with {MODEL_FILE} and {CONFIG_FILE} under {args.run_path!s}",
            file=sys.stderr,
        )
        sys.exit(1)

    for run_dir in run_dirs:
        model_f = run_dir / MODEL_FILE
        cfg = _load_ppo_config(run_dir)
        env_id = cfg.get("env_id")
        if not env_id:
            print(f"skip {run_dir}: no env_id in config", file=sys.stderr)
            continue
        env_kwargs = cfg.get("env_kwargs") or {}
        if "agent" not in cfg and "network_kwargs" not in cfg:  # legacy: flat top-level keys
            cfg["agent"] = {k: cfg[k] for k in ("activation", "hidden_layers_size") if k in cfg}
        cfg = translate_legacy_agent_section(cfg, "ContinuousActorCritic")
        seed = int(cfg.get("seed", 0))
        experiment_dir = str(run_dir.parent)
        run_name = run_dir.name
        ev = evaluate_metaworld(
            str(model_f),
            env_id=str(env_id),
            env_kwargs=env_kwargs,
            device=device,
            experiment_dir=experiment_dir,
            run_name=run_name,
            eval_episodes=int(args.eval_episodes),
            capture_video=not args.no_video,
            deterministic=bool(args.deterministic),
            base_seed=seed,
            save_json=not args.no_json,
            Model=get_network(cfg.get("network") or "ContinuousActorCritic"),
            model_kwargs=cfg.get("network_kwargs") or {},
        )
        if ev is None:
            print(
                f"skip {run_dir.name}: env_id {env_id!r} is not a vector Meta-World benchmark"
            )


if __name__ == "__main__":
    main()
