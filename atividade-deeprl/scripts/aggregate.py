"""Agrega as sementes de um experimento: tabela com IQM + IC 95% e curvas de avaliação.

Exemplo:
    python scripts/aggregate.py --experiment results/runs/dqn_acrobot
    python scripts/aggregate.py --experiment results/runs/dqn_acrobot --metric overestimation

Gera, dentro da pasta do experimento:
    summary.md            tabela por variante (retorno final de cada semente)
    curves_<métrica>.png  curva de avaliação: média entre sementes com IC 95% por bootstrap
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _paths  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rlpraticas.utils.stats import summarize


def read_curve(path: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or metric not in rows[0]:
        return np.array([]), np.array([])
    return (
        np.array([int(r["timestep"]) for r in rows]),
        np.array([float(r[metric]) for r in rows]),
    )


def curve_band(curves: list[np.ndarray], n_resamples: int = 1000, seed: int = 0):
    """Média por passo e IC 95% reamostrando sementes."""
    data = np.stack(curves)  # sementes x pontos
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, data.shape[0], size=(n_resamples, data.shape[0]))
    boot = data[idx].mean(axis=1)  # reamostras x pontos
    return data.mean(axis=0), np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="pasta results/runs/<experimento>")
    parser.add_argument("--metric", default="mean_return", help="coluna do eval.csv a plotar")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="usar só estas sementes (Parte E)")
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    variants = sorted(p for p in exp_dir.iterdir() if p.is_dir())
    if not variants:
        raise SystemExit(f"Nenhuma variante encontrada em {exp_dir}")

    lines = [
        f"# {exp_dir.name}: retorno final (avaliação determinística)",
        "",
        "| variante | sementes | média [IC 95%] | IQM [IC 95%] | mín | máx |",
        "|---|---|---|---|---|---|",
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    min_seeds = None

    for variant in variants:
        finals, curves, steps_ref = [], [], None
        for seed_dir in sorted(variant.glob("seed*")):
            if args.seeds is not None and int(seed_dir.name[4:]) not in args.seeds:
                continue
            final_path = seed_dir / "final.json"
            if not final_path.exists():
                print(f"Aviso: {seed_dir} sem final.json (execução incompleta), ignorada")
                continue
            finals.append(json.loads(final_path.read_text())["mean_return"])
            steps, values = read_curve(seed_dir / "eval.csv", args.metric)
            if len(values):
                if steps_ref is None or len(steps) < len(steps_ref):
                    steps_ref = steps
                curves.append(values)

        if not finals:
            continue
        s = summarize(np.array(finals))
        min_seeds = s["n_seeds"] if min_seeds is None else min(min_seeds, s["n_seeds"])
        lines.append(
            f"| {variant.name} | {s['n_seeds']} "
            f"| {s['mean']:.1f} [{s['mean_ci_low']:.1f}, {s['mean_ci_high']:.1f}] "
            f"| {s['iqm']:.1f} [{s['iqm_ci_low']:.1f}, {s['iqm_ci_high']:.1f}] "
            f"| {s['min']:.1f} | {s['max']:.1f} |"
        )

        if curves:
            n = len(steps_ref)
            mean, low, high = curve_band([c[:n] for c in curves])
            ax.plot(steps_ref, mean, label=f"{variant.name} (n={len(curves)})")
            ax.fill_between(steps_ref, low, high, alpha=0.25)

    if min_seeds is not None and min_seeds < 5:
        lines += ["", "> Atenção: menos de 5 sementes. Os intervalos de confiança são pouco informativos."]

    suffix = "" if args.seeds is None else "_seeds" + "-".join(map(str, args.seeds))
    (exp_dir / f"summary{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    ax.set_xlabel("passos de ambiente")
    ax.set_ylabel(args.metric)
    ax.set_title(f"{exp_dir.name}: {args.metric} na avaliação (média e IC 95% entre sementes)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = exp_dir / f"curves_{args.metric}{suffix}.png"
    fig.savefig(out, dpi=150)
    print(f"\nCurvas salvas em {out}")


if __name__ == "__main__":
    main()
