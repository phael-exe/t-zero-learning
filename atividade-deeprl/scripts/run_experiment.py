"""Roda a mesma configuração em várias sementes, em sequência.

Exemplo (5 sementes, padrão da atividade):
    python scripts/run_experiment.py --config configs/dqn_acrobot.yaml --seeds 0 1 2 3 4

Para uma variante:
    python scripts/run_experiment.py --config configs/dqn_acrobot.yaml --seeds 0 1 2 3 4 \
        --set variant=sem_bootstrap --set hyperparameters.bootstrap_on_truncation=false

Execuções já concluídas (com final.json) são puladas, então dá para
interromper e retomar sem perder trabalho.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _paths  # noqa: F401

from rlpraticas.config import load_config, set_by_path
from train import train_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR")
    args = parser.parse_args()

    for seed in args.seeds:
        cfg = load_config(args.config, args.set)
        set_by_path(cfg, "seed", seed)
        run_dir = Path(cfg["logging"]["root_dir"]) / cfg["experiment"] / str(cfg["variant"]) / f"seed{seed}"
        if (run_dir / "final.json").exists():
            print(f"Pulando {run_dir} (já concluída)")
            continue
        # Execução interrompida no meio: recomeça do zero.
        train_one(cfg, overwrite=run_dir.exists())


if __name__ == "__main__":
    main()
