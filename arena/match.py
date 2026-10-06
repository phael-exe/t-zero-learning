"""Play arena matches: one engine, both teams driven by agent handles.

Library entry point: :func:`play_match`.  Command line::

    python -m arena.match <A> <B> [-n 10] [--video DIR] [--dump DIR]

``A`` / ``B`` are submission folders or the anchors ``random`` / ``builtin``.
Sides swap every match (A is left in even-numbered matches).  Each match gets a
fresh random engine seed, printed and stored with the result.  Matches are not
expected to replay bit-for-bit across machines; play many of them instead.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from arena.agents import AgentError, BaseAgent, make_agent, spec_name
from arena.rules import RULES

RENDER_RESOLUTION = (640, 360)


@dataclass
class MatchResult:
    left: str
    right: str
    seed: int
    goals_left: int
    goals_right: int
    steps: int
    forfeit: str = ""
    """name of the side that broke the rules ("" when the match was completed)."""
    forfeit_reason: str = ""
    score_at_forfeit: str = ""
    wall_seconds: float = 0.0
    act_ms_mean: dict = field(default_factory=dict)
    act_ms_max: dict = field(default_factory=dict)
    action_counts: dict = field(default_factory=dict)

    @property
    def winner(self) -> str:
        if self.goals_left == self.goals_right:
            return ""
        return self.left if self.goals_left > self.goals_right else self.right

    def summary(self) -> str:
        text = f"{self.left} {self.goals_left}-{self.goals_right} {self.right}  (seed {self.seed}, {self.steps} steps)"
        if self.forfeit:
            text += f"  FORFEIT by {self.forfeit}: {self.forfeit_reason.splitlines()[-1] if self.forfeit_reason else ''}"
        return text


def new_seed() -> int:
    return random.SystemRandom().randrange(0, 2**31 - 1)


def _make_engine(n_left: int, n_right: int, seed: int, scenario: str,
                 render: bool, dump_dir: str | None):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from gfootball.env import create_environment

    other = {"game_engine_random_seed": int(seed)}
    if render:
        other.update(render_resolution_x=RENDER_RESOLUTION[0],
                     render_resolution_y=RENDER_RESOLUTION[1])
    return create_environment(
        env_name=scenario,
        representation="raw",
        rewards="scoring",
        number_of_left_players_agent_controls=n_left,
        number_of_right_players_agent_controls=n_right,
        render=render,
        write_full_episode_dumps=dump_dir is not None,
        logdir=dump_dir or "",
        other_config_options=other,
    )


def play_match(
    left: BaseAgent,
    right: BaseAgent,
    seed: int | None = None,
    scenario: str = RULES.scenario,
    max_steps: int | None = None,
    video_path: str | Path | None = None,
    dump_dir: str | Path | None = None,
) -> MatchResult:
    """Play one match; agent-rule violations end it as a forfeit, never raise.

    *max_steps* cuts the match short (tests, quick checks); the score at that
    point stands.  *video_path* renders an mp4 (software GL, ~10 steps/s);
    *dump_dir* writes a gfootball ``.dump`` for ``python -m gfootball.replay``.
    """
    seed = new_seed() if seed is None else int(seed)
    n_left, n_right = left.controlled_players, right.controlled_players
    env = _make_engine(n_left, n_right, seed, scenario, video_path is not None,
                       str(dump_dir) if dump_dir is not None else None)
    core = env.unwrapped._env  # unmirrored global state: the real score
    writer = None
    if video_path is not None:
        import imageio

        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video_path), fps=10, macro_block_size=1)

    if left.name == right.name:
        right.name = f"{right.name} (2)"
    sides = (left, right)
    counts = (Counter(), Counter())
    timings = ([], [])
    forfeit, reason = "", ""
    steps = 0
    t_start = time.perf_counter()
    try:
        obs = env.reset()
        for side in (left, right):
            try:
                side.reset()
            except AgentError as err:
                forfeit, reason = side.name, f"{err.kind}: {err.detail}"
                raise
        done = False
        while not done and (max_steps is None or steps < max_steps):
            actions = []
            for i, side_obs in enumerate((obs[:n_left], obs[n_left:n_left + n_right])):
                side = sides[i]
                if side.controlled_players == 0:
                    continue
                try:
                    t0 = time.perf_counter()
                    side_actions = side.act(side_obs)
                    timings[i].append(time.perf_counter() - t0)
                except AgentError as err:
                    forfeit, reason = side.name, f"{err.kind}: {err.detail}"
                    raise
                counts[i].update(side_actions)
                actions.extend(side_actions)
            obs, _, done, _ = env.step(actions)
            steps += 1
            if writer is not None:
                writer.append_data(np.asarray(env.render(mode="rgb_array"), dtype=np.uint8))
    except AgentError:
        pass
    finally:
        score = [int(g) for g in core.observation()["score"]]
        env.close()
        if writer is not None:
            writer.close()

    result = MatchResult(
        left=left.name, right=right.name, seed=seed,
        goals_left=score[0], goals_right=score[1], steps=steps,
        wall_seconds=round(time.perf_counter() - t_start, 2),
    )
    for i, side in enumerate(sides):
        # Host-side compute time when available (RemoteAgent), else round-trip.
        secs = getattr(side, "act_seconds", None) or timings[i]
        if secs:
            result.act_ms_mean[side.name] = round(1000 * float(np.mean(secs)), 3)
            result.act_ms_max[side.name] = round(1000 * float(np.max(secs)), 3)
        result.action_counts[side.name] = dict(sorted(counts[i].items()))
    if forfeit:
        result.forfeit, result.forfeit_reason = forfeit, reason
        result.score_at_forfeit = f"{score[0]}-{score[1]}"
        lost, won = RULES.forfeit_goals
        if forfeit == left.name:
            result.goals_left, result.goals_right = lost, won
        else:
            result.goals_left, result.goals_right = won, lost
    return result


def play_specs(spec_left, spec_right, seed=None, max_steps=None, video_path=None,
               dump_dir=None, agent_stderr=None) -> MatchResult:
    """Start both agents from specs, play one match, shut them down.

    Failing to *load* also counts as a forfeit (the other side wins).
    """
    handles = []
    try:
        for spec in (spec_left, spec_right):
            try:
                handles.append(make_agent(spec, stderr=agent_stderr, seed=seed))
            except AgentError as err:
                names = [spec_name(spec_left), spec_name(spec_right)]
                lost, won = RULES.forfeit_goals
                gl, gr = (lost, won) if spec is spec_left else (won, lost)
                return MatchResult(left=names[0], right=names[1], seed=seed or 0,
                                   goals_left=gl, goals_right=gr, steps=0,
                                   forfeit=err.agent, forfeit_reason=f"{err.kind}: {err.detail}")
        return play_match(handles[0], handles[1], seed=seed, max_steps=max_steps,
                          video_path=video_path, dump_dir=dump_dir)
    finally:
        for h in handles:
            h.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", help="submission folder, or random / builtin")
    parser.add_argument("b", help="submission folder, or random / builtin")
    parser.add_argument("-n", "--matches", type=int, default=2, help="number of matches (sides swap each match)")
    parser.add_argument("--max-steps", type=int, default=None, help="cut matches short (default: full 3000 steps)")
    parser.add_argument("--video", type=Path, default=None, help="write one mp4 per match into this folder (slow)")
    parser.add_argument("--dump", type=Path, default=None, help="write gfootball .dump replays into this folder")
    args = parser.parse_args(argv)

    name_a, name_b = spec_name(args.a), spec_name(args.b)
    tally = Counter()
    goals = Counter()
    for i in range(args.matches):
        left, right = (args.a, args.b) if i % 2 == 0 else (args.b, args.a)
        seed = new_seed()
        video = args.video / f"match{i:03d}_{seed}.mp4" if args.video else None
        r = play_specs(left, right, seed=seed, max_steps=args.max_steps,
                       video_path=video, dump_dir=args.dump)
        print(f"[{i + 1}/{args.matches}] {r.summary()}")
        tally[r.winner or "draw"] += 1
        goals[r.left] += r.goals_left
        goals[r.right] += r.goals_right
    print(f"\n{name_a}: {tally[name_a]} W  {tally['draw']} D  {tally[name_b]} L   "
          f"goals {goals[name_a]}-{goals[name_b]}")


if __name__ == "__main__":
    sys.exit(main())
