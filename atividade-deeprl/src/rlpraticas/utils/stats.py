"""Estatísticas robustas para comparar execuções com várias sementes.

Segue a recomendação de Agarwal et al. (2021), "Deep RL at the Edge of the
Statistical Precipice": reportar a média interquartil (IQM) com intervalo de
confiança por bootstrap, em vez de média ± desvio de poucas execuções.
A biblioteca `rliable` implementa a versão completa (com várias tarefas);
aqui há uma versão mínima para uma tarefa, sem dependências extras.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def iqm(scores: np.ndarray) -> float:
    """Média interquartil: descarta os 25% piores e os 25% melhores e tira a média.

    Mesmo critério de scipy.stats.trim_mean(x, 0.25).
    """
    x = np.sort(np.asarray(scores, dtype=float))
    cut = int(0.25 * len(x))
    trimmed = x[cut : len(x) - cut] if len(x) - 2 * cut > 0 else x
    return float(trimmed.mean())


def bootstrap_ci(
    scores: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Intervalo de confiança percentil por bootstrap (reamostrando sementes)."""
    x = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    stats = np.array([statistic(x[row]) for row in idx])
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


def summarize(scores: np.ndarray) -> dict[str, float]:
    x = np.asarray(scores, dtype=float)
    mean_lo, mean_hi = bootstrap_ci(x, lambda v: float(np.mean(v)))
    iqm_lo, iqm_hi = bootstrap_ci(x, iqm)
    return {
        "n_seeds": int(len(x)),
        "mean": float(np.mean(x)),
        "mean_ci_low": mean_lo,
        "mean_ci_high": mean_hi,
        "iqm": iqm(x),
        "iqm_ci_low": iqm_lo,
        "iqm_ci_high": iqm_hi,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
