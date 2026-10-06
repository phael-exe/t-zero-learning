# GFootball arena

The arena ([arena/](../arena/)) plays GFootball agents against each other:
a self-check for one submission, head-to-head matches, and a round-robin
tournament. The code is separate from training. It never imports `train.py` or
`algorithms/`, and the only framework code that imports it is the GFootball env
shim, which uses it to load a submission as a training opponent.

What a submission folder must contain (`agent.py`, `manifest.yml`, the probe)
is in the student handout, [assignments/gfootball.md](../assignments/gfootball.md).
This page covers what the arena does with it.

All commands run inside the gfootball image:

```bash
docker compose run --rm gfootball python -m arena.<command> ...
```

## A match

| | |
|---|---|
| Scenario | `5_vs_5`: 4 outfield players plus a keeper per side |
| Length | 3000 engine steps, the full scenario; the match does not stop on a goal |
| Result | the final score; a draw is possible |
| Controlled players | 1 to 4 per side, from the submission's `manifest.yml`; the rest of the team and always the keeper are played by the game AI |
| Observations | gfootball's raw dict, one per controlled player, from the side's own point of view (always attacking towards x = +1) |
| Actions | gfootball's default set, 19 actions |
| Seed | a fresh random engine seed per match, printed and stored with the result |
| Sides | swap every match: A is left in even-numbered matches |

Each submission runs in its own process, with only its folder and the image's
packages on the import path and one CPU thread. The limits it must respect
(load time, `act` time, folder size, ...) are the frozen `Rules` object in
[arena/rules.py](../arena/rules.py). The checker, matches and tournaments all
read that one object, so changing a value there changes the handout.

**Forfeits.** A crash, a timeout or an invalid action ends the match. It is
recorded as a 0–3 loss for the side that broke the rules
(`Rules.forfeit_goals`), together with the reason and the score at that
point. Failing to load counts the same way.

Matches are not expected to replay bit for bit across machines. Play enough of
them to see a trend.

## Anchors

Two built-in opponents can be used anywhere a submission folder can:

| Name | What it is |
|---|---|
| `random` | one controlled player taking uniformly random actions |
| `builtin` | no controlled players: the whole team is the engine's own AI |

## `arena.check`: self-check

```bash
python -m arena.check my_team [--matches 1] [--max-steps N] [--json report.json]
```

The same check the grader runs. In order: layout and manifest, folder size,
loading `Agent` in an isolated process within the time limit, the probe replay
(if `probe.pkl` exists), then full matches against `random` and `builtin`,
each in a fresh process. It reports the scores and `act` timings, and warns
when one action dominates.

| Option | Effect |
|---|---|
| `--matches N` | matches per anchor (default 1; sides alternate) |
| `--max-steps N` | cut matches short, for quick iteration only |
| `--json FILE` | also write the report as JSON |

Exit code 0 is PASS (warnings allowed), 1 is FAIL.

## `arena.match`: head to head

```bash
python -m arena.match A B [-n 2] [--max-steps N] [--video DIR] [--dump DIR]
```

`A` and `B` are submission folders or anchor names. It prints each result and
a W/D/L tally with total goals.

| Option | Effect |
|---|---|
| `-n N` | number of matches (default 2; sides swap each match) |
| `--max-steps N` | cut matches short (default: the full 3000 steps) |
| `--video DIR` | one mp4 per match, rendered with software GL at about 10 steps/s (slow) |
| `--dump DIR` | gfootball `.dump` replays, viewable with `python -m gfootball.replay` |

## `arena.tournament`: round robin

```bash
python -m arena.tournament teams/ [more folders ...] --out results/ \
    [-n 10] [--workers N] [--anchors random,builtin]
```

Entrants are submission folders, or folders containing submission folders.
Every pair plays `n` matches with sides alternating. Matches run in parallel,
each with a fresh engine and fresh agent processes.

Before any match, each submission gets a pre-flight: it loads and plays a few
steps against `builtin`. Entrants that fail are excluded and listed in
`excluded.json`, as are duplicate team names. The tournament goes on without
them.

| Option | Effect |
|---|---|
| `-n N` | matches per pair (default 10) |
| `--workers N` | parallel matches (default: CPU count) |
| `--anchors LIST` | anchors to include (default `random,builtin`; `''` for none) |
| `--max-steps N` | cut matches short, for testing only |
| `--bootstrap N` | bootstrap resamples for the rating intervals (default 200) |
| `--report-only` | rebuild the tables from `matches.jsonl` without playing |

Every finished match is appended to `<out>/matches.jsonl` immediately. So
re-running the same command resumes an interrupted tournament, and raising
`-n` later plays only the missing matches.

Outputs in `<out>/`:

| File | Contents |
|---|---|
| `standings.csv` | football table: 3/1/0 points, W/D/L, goals, goal difference, forfeits, rating |
| `ratings.csv` | Bradley–Terry ratings on the Elo scale (centred on 1000) with a 95 % bootstrap interval |
| `pairwise_gd.csv` | mean goal difference, row vs column |
| `matches.jsonl` | one line per match: seed, score, steps, `act` timings, action counts, forfeit reason |
| `entrants.json`, `excluded.json` | who played, and who was left out and why |

A draw counts as half a win for each side in the ratings. Overlapping intervals
mean the data does not separate those two entrants yet. Play more matches for
those pairs.

## Submissions as training opponents

The env shim can load a submission folder as the opposing team
(`opponent: <folder>` in `env_kwargs`, or `env.unwrapped.set_opponent(...)`).
That path uses the arena's `InProcessAgent`: the submission is imported into
the training process and sees the mirrored view exactly as in a match. There
are no timeouts or isolation, and its helper modules share `sys.modules` with
yours. See [gfootball-environment.md](gfootball-environment.md) for the
training side.
