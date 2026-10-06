"""Submission template — a hand-written policy, no learning, no weights.

Copy this folder, keep the ``Agent`` interface, replace the policy::

    cp -r arena/template my_team
    python -m arena.check my_team

The contract (``assignments/gfootball.md``):

- ``Agent(path)``: ``path`` is this folder. Load weights from it
  (``path / "weights" / "model.pt"``). Do not import from the t-zero checkout:
  the arena runs this file with only this folder and site-packages on the
  import path.
- ``reset()``: called at every kick-off. Clear frame stacks / RNN state here.
- ``act(observations) -> list[int]``: one raw gfootball observation dict per
  controlled player (``controlled_players`` in manifest.yml), one action in
  ``[0, 19)`` per observation, same order. Observations are always from your
  team's point of view: you are ``left_team`` and attack towards x = +1,
  whichever side you actually play on.

Which player an observation belongs to is ``obs["active"]`` (index into
``obs["left_team"]``) — it can change during the match, so never assume a
fixed slot-to-player mapping.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# gfootball default action set (gfootball/env/football_action_set.py).
IDLE, LEFT, TOP_LEFT, TOP, TOP_RIGHT, RIGHT, BOTTOM_RIGHT, BOTTOM, BOTTOM_LEFT = range(9)
LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT, SPRINT = 9, 10, 11, 12, 13
RELEASE_DIRECTION, RELEASE_SPRINT, SLIDING, DRIBBLE, RELEASE_DRIBBLE = 14, 15, 16, 17, 18

# Eight movement actions, counter-clockwise from "right" in field coordinates
# (y grows downwards in gfootball, so "top" is negative y).
_DIRECTIONS = [RIGHT, TOP_RIGHT, TOP, TOP_LEFT, LEFT, BOTTOM_LEFT, BOTTOM, BOTTOM_RIGHT]
OPPONENT_GOAL = np.array([1.0, 0.0])


def move_towards(position: np.ndarray, target: np.ndarray) -> int:
    dx, dy = target - position
    angle = np.arctan2(-dy, dx)  # flip y so angles are counter-clockwise
    return _DIRECTIONS[int(np.round(angle / (np.pi / 4))) % 8]


class Agent:
    def __init__(self, path: Path):
        self.path = Path(path)
        # A learned agent would load its weights here, e.g.
        #   self.policy = torch.jit.load(str(self.path / "weights" / "policy.pt"))

    def reset(self) -> None:
        pass

    def act(self, observations: list[dict]) -> list[int]:
        return [self._act_one(obs) for obs in observations]

    def _act_one(self, obs: dict) -> int:
        me = obs["active"]
        position = np.asarray(obs["left_team"][me])
        ball = np.asarray(obs["ball"][:2])
        we_have_ball = obs["ball_owned_team"] == 0 and obs["ball_owned_player"] == me
        if not we_have_ball:
            return move_towards(position, ball)
        if np.linalg.norm(OPPONENT_GOAL - position) < 0.3:
            return SHOT
        return move_towards(position, OPPONENT_GOAL)
