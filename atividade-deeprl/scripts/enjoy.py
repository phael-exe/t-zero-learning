"""Grava um GIF do agente treinado. Olhar o comportamento revela problemas que as curvas escondem.

    python scripts/enjoy.py --run results/runs/dqn_acrobot/padrao/seed0
    (o GIF é salvo dentro da pasta da execução)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # renderização sem monitor (Docker)
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import _paths  # noqa: F401
import yaml
from PIL import Image

from evaluate import build_policy
from rlpraticas.envs import make_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--estocastico", action="store_true")
    parser.add_argument("--sem-normalizacao", action="store_true")
    args = parser.parse_args()

    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    policy = build_policy(cfg, run, args)

    env = make_env(cfg, args.seed, render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)
    frames, total = [Image.fromarray(env.render())], 0.0
    for _ in range(args.max_steps):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += float(reward)
        frames.append(Image.fromarray(env.render()))
        if terminated or truncated:
            break
    env.close()

    out = run / "agente.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=33, loop=0)
    print(f"{len(frames)} quadros, retorno {total:.1f} → {out}")


if __name__ == "__main__":
    main()
