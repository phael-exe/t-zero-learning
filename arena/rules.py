"""The arena rules: one frozen object, shared by the checker, matches and tournaments.

Changing a value here changes what ``arena.check`` enforces for every student,
so treat edits as a handout change.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    scenario: str = "5_vs_5"
    """gfootball scenario every arena match is played on (3000 steps, AI keepers)."""
    max_controlled_players: int = 4
    """outfield players per side in ``5_vs_5`` (keepers are never controllable)."""
    num_actions: int = 19
    """size of gfootball's default action set."""
    max_submission_mb: float = 100.0
    load_timeout_s: float = 30.0
    """``Agent.__init__`` wall-time limit."""
    first_act_timeout_s: float = 10.0
    """first ``act`` call of a process (lazy imports, JIT warm-up)."""
    act_timeout_s: float = 1.0
    """any later single ``act`` call; exceeding it forfeits the match."""
    mean_act_ms: float = 20.0
    """mean ``act`` compute time over a match (all controlled slots together)."""
    reset_timeout_s: float = 5.0
    forfeit_goals: tuple[int, int] = (0, 3)
    """(forfeiting side, opponent) goals recorded for a forfeited match."""
    agent_threads: int = 1
    """CPU threads each agent process may use (OMP/MKL/torch)."""


RULES = Rules()
