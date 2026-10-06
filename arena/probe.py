"""Probe files: recorded ``(observations, actions)`` pairs that pin down a submission.

A probe is recorded where the agent is *known* to behave — the student's own
evaluation in their training pipeline — and replayed by ``arena.check``
through the submitted ``Agent``. If preprocessing, normalization statistics
or weights differ between the two, the actions stop matching, and the check
says so before a tournament does.

Recording (any pipeline; with the t-zero shim, ``env.unwrapped.raw_observations()``
returns exactly the observations the arena would give the agent)::

    rec = ProbeRecorder(controlled_players=1)
    rec.new_episode()                     # at every env reset
    rec.add(raw_observations, actions)    # every step
    rec.save("my_team/probe.pkl")

Episodes matter for stateful agents: ``arena.check`` calls ``reset()`` at the
start of each recorded episode and replays its steps in order.
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path

PROBE_FILE = "probe.pkl"
PROBE_FORMAT = 1


class ProbeRecorder:
    def __init__(self, controlled_players: int):
        self.controlled_players = int(controlled_players)
        self.episodes: list[list[dict]] = []

    def new_episode(self) -> None:
        self.episodes.append([])

    def add(self, observations: list[dict], actions) -> None:
        if not self.episodes:
            self.new_episode()
        observations = list(observations)
        actions = [int(a) for a in (actions if hasattr(actions, "__len__") else [actions])]
        if len(observations) != self.controlled_players or len(actions) != self.controlled_players:
            raise ValueError(
                f"expected {self.controlled_players} observation(s) and action(s), "
                f"got {len(observations)} / {len(actions)}"
            )
        self.episodes[-1].append({"obs": copy.deepcopy(observations), "actions": actions})

    @property
    def num_steps(self) -> int:
        return sum(len(e) for e in self.episodes)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "format": PROBE_FORMAT,
                    "controlled_players": self.controlled_players,
                    "episodes": [e for e in self.episodes if e],
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return path


def load_probe(path: str | Path) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or data.get("format") != PROBE_FORMAT:
        raise ValueError(f"{path}: not a probe file (format {PROBE_FORMAT})")
    return data
