"""Treina UMA execução (uma semente) a partir de um arquivo de configuração.

Exemplos:
    python scripts/train.py --config configs/dqn_acrobot.yaml --seed 0
    python scripts/train.py --config configs/dqn_acrobot.yaml --seed 0 \
        --set variant=sem_bootstrap --set hyperparameters.bootstrap_on_truncation=false
"""

from __future__ import annotations

import argparse
import json
import time

import _paths  # noqa: F401

from rlpraticas.config import load_config, set_by_path
from rlpraticas.utils.reproducibility import make_run_dir, save_run_metadata, set_global_seeds


def train_one(cfg: dict, overwrite: bool = False) -> dict:
    set_global_seeds(int(cfg["seed"]))
    run_dir = make_run_dir(cfg, overwrite=overwrite)
    save_run_metadata(cfg, run_dir)

    if cfg["algo"] == "dqn":
        from rlpraticas.trainers.dqn import train
    elif cfg["algo"] == "ppo":
        from rlpraticas.trainers.ppo_sb3 import train
    else:
        raise ValueError(f"Algoritmo desconhecido: {cfg['algo']}")

    start = time.time()
    final = train(cfg, run_dir)
    final["wall_time_s"] = round(time.time() - start, 1)
    with open(run_dir / "final.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"Concluído em {final['wall_time_s']} s → {run_dir}  |  retorno final = {final['mean_return']:.1f}")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        set_by_path(cfg, "seed", args.seed)
    train_one(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
