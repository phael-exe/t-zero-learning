#!/usr/bin/env python3
"""Pull the SAC-assignment runs from wandb and build the report's tables and figures.

Usage:
    python scripts/sac_report.py table                # Part 3 summary per exp_name
    python scripts/sac_report.py fig OUT.png EXP [EXP ...] [--keys k1,k2] [--x _step|_runtime] [--smooth N]
                                                      [--sps exp=steps_per_s,...]

Runs are grouped by ``config.exp_name`` (set with ``--override exp_name=...``
or by the config file's name). ``--smooth N`` applies a rolling mean of N
points to the per-episode ``charts/q_*`` metrics. Reads WANDB_PROJECT / WANDB_ENTITY from .env.
"""
import os
import sys
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv

load_dotenv(".env")
import wandb  # noqa: E402

PROJECT = os.environ.get("WANDB_PROJECT", "sac-assignment")
ENTITY = os.environ.get("WANDB_ENTITY")
RET = "charts/episodic_return_mean_last100"
LABELS = {
    "sac_hopper": "SAC inicial",
    "sac_hopper_utd2": "SAC final (utd2)",
    "sac_hopper_utd2_fixedalpha": "alpha fixo = 0,1",
    "sac_hopper_utd2_singleq": "um crítico só",
    "ppo_hopper": "PPO",
}


def runs_by_exp(env_id="Hopper-v5"):
    api = wandb.Api()
    path = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT
    groups = defaultdict(list)
    for r in api.runs(path):
        if r.state != "finished" or r.config.get("env_id") != env_id:
            continue
        groups[r.config.get("exp_name", "?")].append(r)
    for runs in groups.values():
        runs.sort(key=lambda r: r.config.get("seed", 0))
    return groups


def _fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{np.mean(vals):.0f} ({min(vals):.0f}–{max(vals):.0f})"


def table():
    for exp, runs in sorted(runs_by_exp().items()):
        ev = [r.summary.get("eval/mean_return") for r in runs]
        ret = [r.summary.get(RET) for r in runs]
        rt = [r.summary.get("_runtime", 0) / 60 for r in runs]
        seeds = [r.config.get("seed") for r in runs]
        print(f"{exp:32s} seeds={seeds} eval={_fmt(ev):22s} last100={_fmt(ret):22s} "
              f"time={np.mean(rt):.0f}min ids={[r.id for r in runs]}")


def history(run, keys, x):
    """``x`` is ``_step`` (== global_step, the step every metric is logged at) or ``_runtime``."""
    df = run.history(keys=keys, samples=2000, pandas=True)
    return df.sort_values(x) if x in df else df


def fig(out, exps, keys=(RET,), x="_step", smooth=1, sps=None, seeds=None):
    """One panel per key. With ``sps`` ({exp: (steps_per_second, warmup_steps)})
    the x-axis is the estimated wall-clock time in minutes instead: the first
    ``warmup_steps`` (random actions, no updates) are taken as free, the rest
    run at ``steps_per_second``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = runs_by_exp()
    fig_, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.2), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, key in zip(axes[0], keys):
        for c, exp in zip(colors, exps):
            runs = [r for r in groups.get(exp, []) if seeds is None or r.config.get("seed") in seeds]
            for i, run in enumerate(runs):
                df = history(run, [key], x)
                if key not in df:
                    continue
                df = df.dropna(subset=[key])
                if sps:
                    rate, warmup = sps[exp]
                    xs = (df["_step"] - warmup).clip(lower=0) / rate / 60
                else:
                    xs = df[x] / (60 if x == "_runtime" else 1)
                ys = df[key].rolling(smooth, min_periods=1).mean() if key.startswith("charts/q_") else df[key]
                ax.plot(xs, ys, color=c, alpha=0.85, lw=1.1,
                        label=LABELS.get(exp, exp) if i == 0 else None)
            if key == "losses/entropy":
                ax.axhline(groups[exps[0]][0].summary.get("charts/target_entropy", -3), color="k", ls=":", lw=1)
        ax.set_title(key, fontsize=9)
        xlabel = "tempo estimado (min)" if sps else ("minutos" if x == "_runtime" else "passos de ambiente")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig_.tight_layout()
    fig_.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table")
    f = sub.add_parser("fig")
    f.add_argument("out")
    f.add_argument("exps", nargs="+")
    f.add_argument("--keys", default=RET, help="comma-separated metric names, one panel each")
    f.add_argument("--x", default="_step", choices=["_step", "_runtime"])
    f.add_argument("--smooth", type=int, default=1)
    f.add_argument("--sps", help="exp=steps_per_second[@warmup_steps],... -> x-axis = estimated minutes")
    f.add_argument("--seeds", help="comma-separated seeds to plot (default: all)")
    a = ap.parse_args()
    if a.cmd == "table":
        table()
    else:
        sps = None
        if a.sps:
            sps = {}
            for kv in a.sps.split(","):
                k, v = kv.split("=")
                rate, _, warm = v.partition("@")
                sps[k] = (float(rate), float(warm or 0))
        seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else None
        fig(a.out, a.exps, a.keys.split(","), a.x, a.smooth, sps, seeds)
