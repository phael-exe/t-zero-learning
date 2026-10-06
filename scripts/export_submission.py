#!/usr/bin/env python3
"""Export a trained t-zero GFootball run as an arena submission folder.

    python scripts/export_submission.py runs/<exp>/<run_dir> --out submissions/my_team --team "My Team"
    python -m arena.check submissions/my_team

Works for any discrete-action network with the uniform policy interface
(``act(obs, deterministic=True)``), trained on a ``GFootball/*`` env with the
``simple115v2`` representation and one controlled player. The network is
rebuilt from the run's ``config.yml``, as training built it:
``get_network(network)(envs, **network_kwargs)``. The output is self-contained — it does not import
t-zero:

    agent.py           raw obs -> simple115v2 -> policy -> action
    weights/policy.pt  the greedy policy as TorchScript (network + any
                       normalization buffers baked in)
    weights/config.yml the run's config, for provenance
    manifest.yml       team name, controlled_players: 1
    probe.pkl          actions of the *original* model on 5_vs_5 states, recorded
                       through the framework's own env + wrapper path, so
                       arena.check can prove the export behaves identically

Anything else (other representations, frame stacks, several players, your own
wrappers) is a hand-written ``agent.py`` — start from ``arena/template/``.
The exporter refuses runs whose wrapper stack changes observations, because it
cannot reproduce them.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.probe import PROBE_FILE, ProbeRecorder  # noqa: E402
from arena.rules import RULES  # noqa: E402
from core.config_loader import ALGORITHMS, load_config  # noqa: E402
from networks import get_network  # noqa: E402

PROBE_ENV_ID = f"GFootball/{RULES.scenario}-v0"

AGENT_PY = '''"""Exported by scripts/export_submission.py from t-zero run {run_name!r}.

Algorithm: {algorithm}. Trained on: {env_id}.
Pipeline: raw observation -> simple115v2 (115 floats) -> TorchScript greedy
policy (weights/policy.pt) -> action. Edit freely; re-run arena.check after.
"""
from pathlib import Path

import torch
from gfootball.env.wrappers import Simple115StateWrapper


class Agent:
    def __init__(self, path):
        self.policy = torch.jit.load(str(Path(path) / "weights" / "policy.pt"), map_location="cpu")
        self.policy.eval()

    def reset(self):
        pass

    def act(self, observations):
        x = Simple115StateWrapper.convert_observation(observations, True)
        with torch.no_grad():
            return self.policy(torch.as_tensor(x, dtype=torch.float32)).tolist()
'''


class _GreedyPolicy(torch.nn.Module):
    """``obs batch -> action batch``, the model's deterministic ``act``."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.act(x, deterministic=True)


def _fail(message: str) -> None:
    print(f"export_submission: {message}", file=sys.stderr)
    sys.exit(1)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_trained_model(run_dir: Path):
    """``(args, network class, wrapper stack)`` for a finished run."""
    from envs.wrappers import resolve_wrapper_stack

    for name in ("model.pt", "config.yml"):
        if not (run_dir / name).is_file():
            _fail(f"{run_dir / name} not found (train with save_model: true)")
    args, algo_name = load_config(str(run_dir / "config.yml"))
    module_path, cls_name, _ = ALGORITHMS[algo_name]
    AlgoCls = getattr(importlib.import_module(module_path), cls_name)
    if not str(args.env_id).startswith("GFootball/"):
        _fail(f"run was trained on {args.env_id}, not a GFootball env")
    kwargs = dict(args.env_kwargs or {})
    if kwargs.get("representation", "simple115v2") != "simple115v2":
        _fail(f"representation {kwargs['representation']!r}: the generic export only handles "
              "simple115v2 — write your own agent.py (arena/template)")
    if int(kwargs.get("controlled_players", 1)) != 1:
        _fail("controlled_players > 1: write your own agent.py (arena/template)")
    wrappers = resolve_wrapper_stack(getattr(args, "env_wrappers", ""), AlgoCls.default_wrappers)
    try:
        Network = get_network(args.network)
    except (ImportError, AttributeError, ValueError) as exc:
        _fail(f"network {args.network!r} from config.yml: {exc}")
    return args, Network, wrappers


def main(argv=None) -> int:
    import gymnasium as gym

    from envs import make_env

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="run folder holding model.pt and config.yml")
    parser.add_argument("--out", type=Path, required=True, help="submission folder to create")
    parser.add_argument("--team", default=None, help="team name for manifest.yml (default: run name)")
    parser.add_argument("--probe-episodes", type=int, default=2)
    parser.add_argument("--probe-steps", type=int, default=300, help="steps recorded per probe episode")
    parser.add_argument("--probe-random", type=float, default=0.3,
                        help="fraction of probe steps that execute a random action (state diversity)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite a non-empty --out folder")
    args_cli = parser.parse_args(argv)

    run_dir = args_cli.run_dir.resolve()
    out = args_cli.out
    if out.exists() and any(out.iterdir()) and not args_cli.force:
        _fail(f"{out} is not empty (use --force to overwrite)")
    args, Network, wrappers = load_trained_model(run_dir)

    # Probe env: the arena scenario through the framework's own env + wrapper path.
    probe_kwargs = {"representation": "simple115v2", "rewards": "scoring"}
    env = make_env(PROBE_ENV_ID, 0, False, "export", float(args.algo.gamma), None,
                   probe_kwargs, wrappers=wrappers)()
    if not isinstance(env.action_space, gym.spaces.Discrete):
        _fail(f"action space {env.action_space} is not Discrete")
    stub = SimpleNamespace(single_observation_space=env.observation_space,
                           single_action_space=env.action_space)
    model = Network(stub, **args.network_kwargs)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()

    from gfootball.env.wrappers import Simple115StateWrapper

    rng = np.random.default_rng(args_cli.seed)
    recorder = ProbeRecorder(controlled_players=1)
    inputs = []
    for episode in range(args_cli.probe_episodes):
        obs, _ = env.reset(seed=args_cli.seed + episode)
        recorder.new_episode()
        for _ in range(args_cli.probe_steps):
            raw = env.unwrapped.raw_observations()
            converted = Simple115StateWrapper.convert_observation(raw, True)[0]
            if not np.allclose(converted, obs, atol=1e-6):
                _fail("the run's wrapper stack changes observations (max diff "
                      f"{np.abs(converted - obs).max():.3g}); the generic export cannot reproduce "
                      "that — write your own agent.py (arena/template)")
            x = torch.as_tensor(obs, dtype=torch.float32)[None]
            with torch.no_grad():
                action = int(model.act(x, deterministic=True)[0])
            recorder.add(raw, [action])
            inputs.append(obs)
            executed = int(rng.integers(env.action_space.n)) if rng.random() < args_cli.probe_random else action
            obs, _, terminated, truncated, _ = env.step(executed)
            if terminated or truncated:
                break
    env.close()

    batch = torch.as_tensor(np.stack(inputs), dtype=torch.float32)
    with torch.no_grad():
        policy = torch.jit.trace(_GreedyPolicy(model), batch[:1])
        eager = model.act(batch, deterministic=True)
        traced = policy(batch)
    if not torch.equal(eager, traced):
        _fail(f"TorchScript policy disagrees with the model on {(eager != traced).sum().item()} probe "
              "states — this network cannot be traced as-is; write your own agent.py")

    if out.exists() and args_cli.force:
        shutil.rmtree(out)
    (out / "weights").mkdir(parents=True)
    policy.save(str(out / "weights" / "policy.pt"))
    shutil.copy(run_dir / "config.yml", out / "weights" / "config.yml")
    (out / "agent.py").write_text(AGENT_PY.format(run_name=run_dir.name, algorithm=args.algorithm,
                                                  env_id=args.env_id))
    manifest = {
        "team": args_cli.team or run_dir.name,
        "members": [],
        "controlled_players": 1,
        "deterministic": True,
        "description": f"{args.algorithm} on {args.env_id}, exported from {run_dir.name} "
                       f"(t-zero {_git_commit()})",
    }
    with open(out / "manifest.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    recorder.save(out / PROBE_FILE)
    print(f"wrote {out}/ ({recorder.num_steps} probe steps)")
    print(f"next: python -m arena.check {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
