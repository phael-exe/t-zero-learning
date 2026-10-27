"""Sementes, diretório de execução e metadados para reprodutibilidade."""

from __future__ import annotations

import json
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def set_global_seeds(seed: int) -> None:
    """Fixa as fontes de aleatoriedade de Python, NumPy e PyTorch.

    O ambiente é semeado separadamente, em env.reset(seed=...) e
    env.action_space.seed(...). Em CPU com uma thread o resultado é
    reprodutível; em GPU ainda pode haver não-determinismo residual.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def make_run_dir(cfg: dict, overwrite: bool = False) -> Path:
    run_dir = (
        Path(cfg["logging"]["root_dir"])
        / cfg["experiment"]
        / str(cfg.get("variant", "padrao"))
        / f"seed{cfg['seed']}"
    )
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{run_dir} já existe. Use --overwrite ou outro --set variant=..."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def save_run_metadata(cfg: dict, run_dir: Path) -> None:
    """Salva a config efetiva e as versões usadas, junto com os resultados."""
    import gymnasium

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
    }
    try:
        import stable_baselines3

        versions["stable_baselines3"] = stable_baselines3.__version__
    except ImportError:
        pass

    meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "versions": versions,
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
