"""Round-robin tournament between submissions (and the anchors).

    python -m arena.tournament submissions/ [more folders ...] \\
        --out runs/arena/final -n 10 --workers 12 [--anchors random,builtin]

Every pair of entrants plays ``n`` matches; sides alternate. Matches run in
parallel, each in a fresh engine with fresh agent processes. Every finished
match is appended to ``<out>/matches.jsonl`` immediately, so:

- an interrupted tournament resumes by re-running the same command;
- raising ``-n`` later plays only the missing matches (for pairs whose
  rating intervals still overlap, or everyone);
- ``--report-only`` rebuilds the tables from the file without playing.

Before any match, each submission gets a pre-flight (load + a few steps vs
``builtin``). Entrants that fail it are excluded and listed in
``<out>/excluded.json``, and the tournament goes on without them.

Outputs in ``<out>/``: ``standings.csv`` (3/1/0 points, goal difference),
``ratings.csv`` (Bradley–Terry, Elo scale, 95 % bootstrap interval) and
``pairwise_gd.csv`` (mean goal difference, row vs column).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from arena.agents import ANCHORS, AgentError, BuiltinAI, RemoteAgent, read_manifest
from arena.match import new_seed, play_match, play_specs
from arena.ratings import bradley_terry, pairwise, standings

MATCHES_FILE = "matches.jsonl"


def discover(paths: list[Path]) -> list[Path]:
    """Submission folders: each path itself if it has agent.py, else its children that do."""
    found = []
    for path in paths:
        path = Path(path).resolve()
        if (path / "agent.py").is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(p for p in path.iterdir() if (p / "agent.py").is_file()))
        else:
            raise FileNotFoundError(path)
    return found


def preflight(folder: Path, steps: int = 5) -> str | None:
    """None if the submission loads and plays a few steps; else the reason."""
    try:
        read_manifest(folder)
        with RemoteAgent(folder, stderr=None) as agent:
            result = play_match(agent, BuiltinAI(), max_steps=steps)
        if result.forfeit:
            return result.forfeit_reason
    except (AgentError, ValueError) as exc:
        return str(exc)
    return None


def _play_task(task: dict) -> dict:
    """Worker process: one match. Agents' stderr is discarded (forfeits keep the traceback)."""
    import subprocess

    result = play_specs(task["left"], task["right"], seed=task["seed"],
                        max_steps=task["max_steps"], agent_stderr=subprocess.DEVNULL)
    row = asdict(result)
    # Record entrant ids (unique) rather than whatever names the handles used.
    row.update(left=task["left_id"], right=task["right_id"], pair=task["pair"], index=task["index"])
    if result.forfeit:
        row["forfeit"] = task["left_id"] if result.forfeit == result.left else task["right_id"]
    return row


def load_results(out: Path) -> list[dict]:
    path = out / MATCHES_FILE
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_reports(out: Path, results: list[dict], bootstrap: int = 200) -> None:
    table = standings(results)
    ratings = {r["name"]: r for r in bradley_terry(results, bootstrap=bootstrap)}
    names, gd, counts = pairwise(results)

    with open(out / "standings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "name", "points", "played", "w", "d", "l",
                                               "gf", "ga", "gd", "forfeits", "rating", "low", "high"])
        writer.writeheader()
        for rank, row in enumerate(table, 1):
            r = ratings.get(row["name"], {})
            writer.writerow(dict(rank=rank, **row, rating=r.get("rating"), low=r.get("low"), high=r.get("high")))
    with open(out / "ratings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "rating", "low", "high"])
        writer.writeheader()
        writer.writerows(sorted(ratings.values(), key=lambda r: -r["rating"]))
    with open(out / "pairwise_gd.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_vs_col"] + names)
        for i, name in enumerate(names):
            writer.writerow([name] + ["" if np.isnan(v) else round(float(v), 2) for v in gd[i]])

    width = max([len(r["name"]) for r in table] + [4])
    print(f"\n{'#':>3}  {'team':<{width}}  {'pts':>4} {'P':>4} {'W':>4} {'D':>4} {'L':>4} "
          f"{'GD':>5}  {'rating [95% CI]':>22}  forfeits")
    for rank, row in enumerate(table, 1):
        r = ratings[row["name"]]
        print(f"{rank:>3}  {row['name']:<{width}}  {row['points']:>4} {row['played']:>4} {row['w']:>4} "
              f"{row['d']:>4} {row['l']:>4} {row['gd']:>+5}  "
              f"{r['rating']:>7.0f} [{r['low']:>5.0f}, {r['high']:>5.0f}]  {row['forfeits']}")
    print(f"\n{len(results)} matches; tables written to {out}/")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entrants", nargs="*", type=Path,
                        help="submission folders, or folders containing submission folders")
    parser.add_argument("--out", type=Path, required=True, help="results folder (reused to resume)")
    parser.add_argument("-n", "--matches-per-pair", type=int, default=10)
    parser.add_argument("--anchors", default="random,builtin",
                        help="comma-separated anchors to include (random, builtin; '' for none)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--max-steps", type=int, default=None, help="cut matches short (testing only)")
    parser.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples for rating intervals")
    parser.add_argument("--report-only", action="store_true", help="rebuild tables from matches.jsonl")
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        write_reports(out, load_results(out), args.bootstrap)
        return 0

    # --- entrants -----------------------------------------------------------
    entrants: dict[str, str] = {}  # id -> spec
    excluded = {}
    for folder in discover(args.entrants):
        reason = preflight(folder)
        team = folder.name
        try:
            team = read_manifest(folder).team
        except ValueError:
            pass
        if reason:
            excluded[str(folder)] = reason
            print(f"EXCLUDED {folder}: {reason.strip().splitlines()[-1] if reason.strip() else reason}")
            continue
        if team in entrants:
            excluded[str(folder)] = f"duplicate team name {team!r} (also {entrants[team]})"
            print(f"EXCLUDED {folder}: duplicate team name {team!r}")
            continue
        entrants[team] = str(folder)
    for anchor in filter(None, (a.strip() for a in args.anchors.split(","))):
        if anchor not in ANCHORS:
            parser.error(f"unknown anchor {anchor!r} (choose from {ANCHORS})")
        entrants[f"[{anchor}]"] = anchor
    (out / "excluded.json").write_text(json.dumps(excluded, indent=2))
    (out / "entrants.json").write_text(json.dumps(entrants, indent=2))
    if len(entrants) < 2:
        print("need at least two entrants", file=sys.stderr)
        return 1

    # --- schedule (skipping matches already on disk) ------------------------
    done = {(r["pair"], r["index"]) for r in load_results(out)}
    tasks = []
    for a, b in itertools.combinations(sorted(entrants), 2):
        pair = f"{a} vs {b}"
        for k in range(args.matches_per_pair):
            if (pair, k) in done:
                continue
            left, right = (a, b) if k % 2 == 0 else (b, a)
            tasks.append(dict(left=entrants[left], right=entrants[right], left_id=left, right_id=right,
                              pair=pair, index=k, seed=new_seed(), max_steps=args.max_steps))
    print(f"{len(entrants)} entrants, {len(tasks)} matches to play "
          f"({len(done)} already in {out / MATCHES_FILE}), {args.workers} workers")

    # --- play ---------------------------------------------------------------
    ctx = multiprocessing.get_context("spawn")
    with open(out / MATCHES_FILE, "a", encoding="utf-8") as log, \
            ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = [pool.submit(_play_task, t) for t in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            log.write(json.dumps(row) + "\n")
            log.flush()
            forfeit = f"  FORFEIT {row['forfeit']}" if row["forfeit"] else ""
            print(f"[{i}/{len(tasks)}] {row['left']} {row['goals_left']}-{row['goals_right']} "
                  f"{row['right']}{forfeit}", flush=True)

    results = [r for r in load_results(out) if r["left"] in entrants and r["right"] in entrants]
    write_reports(out, results, args.bootstrap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
