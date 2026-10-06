"""Standings and Bradley–Terry ratings from a list of match results.

Ratings: Bradley–Terry fitted with Hunter's MM algorithm, a draw counting as
half a win for each side. A light prior (one virtual draw against an average
opponent) keeps unbeaten or winless entrants finite. Reported on the Elo
scale (400 · log10), centred on 1000. Intervals come from a bootstrap over
matches, so "A is ahead of B" can be read off whether intervals overlap.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

POINTS_WIN, POINTS_DRAW = 3, 1


def standings(results: list[dict]) -> list[dict]:
    """Football table rows sorted by points, goal difference, goals for."""
    table = defaultdict(lambda: dict(played=0, w=0, d=0, l=0, gf=0, ga=0, forfeits=0))
    for r in results:
        for me, other, gf, ga in ((r["left"], r["right"], r["goals_left"], r["goals_right"]),
                                  (r["right"], r["left"], r["goals_right"], r["goals_left"])):
            row = table[me]
            row["played"] += 1
            row["gf"] += gf
            row["ga"] += ga
            row["w" if gf > ga else "d" if gf == ga else "l"] += 1
            row["forfeits"] += int(r.get("forfeit") == me)
    rows = []
    for name, row in table.items():
        row = dict(name=name, **row)
        row["gd"] = row["gf"] - row["ga"]
        row["points"] = POINTS_WIN * row["w"] + POINTS_DRAW * row["d"]
        rows.append(row)
    rows.sort(key=lambda r: (-r["points"], -r["gd"], -r["gf"], r["name"]))
    return rows


def _fit(names: list[str], left: np.ndarray, right: np.ndarray, score_left: np.ndarray,
         iterations: int = 500, prior: float = 1.0) -> np.ndarray:
    """Bradley–Terry strengths (MM). ``score_left`` is 1 / 0.5 / 0 per match."""
    k = len(names)
    wins = np.full(k, 0.5 * prior)  # prior: `prior` virtual draws vs strength 1
    games = np.zeros((k, k))
    np.add.at(wins, left, score_left)
    np.add.at(wins, right, 1.0 - score_left)
    np.add.at(games, (left, right), 1.0)
    games = games + games.T
    p = np.ones(k)
    for _ in range(iterations):
        denom = (games / (p[:, None] + p[None, :])).sum(axis=1) + prior / (p + 1.0)
        new = wins / denom
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(new - p)) < 1e-10:
            p = new
            break
        p = new
    return p


def bradley_terry(results: list[dict], bootstrap: int = 200, seed: int = 0) -> list[dict]:
    """Rows ``{name, rating, low, high}`` (Elo scale, 95 % bootstrap interval), best first."""
    if not results:
        return []
    names = sorted({r["left"] for r in results} | {r["right"] for r in results})
    index = {n: i for i, n in enumerate(names)}
    left = np.array([index[r["left"]] for r in results])
    right = np.array([index[r["right"]] for r in results])
    diff = np.array([r["goals_left"] - r["goals_right"] for r in results])
    score = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))

    def to_elo(p):
        e = 400.0 * np.log10(p)
        return 1000.0 + e - e.mean()

    point = to_elo(_fit(names, left, right, score))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap):
        idx = rng.integers(0, len(results), size=len(results))
        samples.append(to_elo(_fit(names, left[idx], right[idx], score[idx])))
    samples = np.array(samples) if samples else point[None, :]
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    rows = [dict(name=n, rating=round(float(point[i]), 1), low=round(float(low[i]), 1),
                 high=round(float(high[i]), 1)) for i, n in enumerate(names)]
    rows.sort(key=lambda r: -r["rating"])
    return rows


def pairwise(results: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Names, mean goal difference matrix (row vs column), and match counts."""
    names = sorted({r["left"] for r in results} | {r["right"] for r in results})
    index = {n: i for i, n in enumerate(names)}
    gd = np.zeros((len(names), len(names)))
    n = np.zeros_like(gd)
    for r in results:
        i, j = index[r["left"]], index[r["right"]]
        d = r["goals_left"] - r["goals_right"]
        gd[i, j] += d
        gd[j, i] -= d
        n[i, j] += 1
        n[j, i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, gd / n, np.nan)
    return names, mean, n
