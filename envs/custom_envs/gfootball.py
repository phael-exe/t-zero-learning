"""Google Research Football (``gfootball``) as a Gymnasium environment.

gfootball (github.com/google-research/football, archived 2026) targets the
legacy ``gym`` API: ``reset() -> obs``, ``step() -> (obs, r, done, info)``,
``render(mode)``, no ``seed`` in ``reset``.  :class:`GFootballEnv` adapts one
scenario to the Gymnasium 1.x contract so the rest of the framework can treat
it like any other env.  Ids are registered in ``envs/custom_envs/__init__.py``
(only when gfootball is importable) as ``GFootball/<scenario>-v0``.

Engine facts this shim relies on (verified against gfootball 2.10.3; the
full reference is docs/gfootball-environment.md):

- ``representation="simple115v2"`` gives a flat ``(115,) float32`` vector and
  the default action set is ``Discrete(19)`` — the input contract of the
  ``discrete_control`` wrapper stack / DQN.
- Engine randomness is read from ``config["game_engine_random_seed"]`` at
  every ``reset()`` (a fresh random value if the key is absent).  The shim
  seeds ``self.np_random`` from ``reset(seed=...)`` and writes one engine seed
  per episode, so a run seed gives a reproducible *sequence* of episodes and
  a deterministic policy still sees different episodes.  Note the seed only
  matters where the engine makes random decisions (built-in AI players);
  ``academy_empty_goal_close`` is fully deterministic given the actions.
- The engine only reports ``done``.  The raw observation's ``steps_left``
  tells us whether the scenario clock ran out (``truncated``) or the scenario
  ended on its own terms — goal, ball out, possession change (``terminated``).
- Rendering is software GL and slow (~10 steps/s); it is enabled lazily on the
  first ``render()`` call and switched off again on the next ``step()`` that
  nobody rendered, so a ``render_mode="rgb_array"`` env that is not being
  recorded runs at full speed (the framework builds worker 0 with
  ``render_mode="rgb_array"`` whenever ``capture_video`` is on).

Beyond one player vs the built-in AI (hooks for student extensions — no
multi-agent or self-play algorithm ships with t-zero):

- ``controlled_players=N`` (> 1): the learner controls N left players. Obs
  become ``(N, ...)``, the action space ``MultiDiscrete([19] * N)``, and the
  scalar reward is the *mean* over the N players (a goal is still +1). Per-player
  rewards are in ``info["player_rewards"]``. Which player each row is can change
  during play (``raw_observations()[i]["active"]``). The ``discrete_control``
  stack refuses these envs (its 2-D observation guard fires first): a
  multi-player learner needs its own stack and algorithm.
- ``opponent=``: who plays the right team. ``"builtin"`` (default) is the
  engine AI, ``"random"`` gives random actions, and a path to an arena
  submission folder (``arena/template``, ``python -m arena.check``) loads its
  ``Agent`` in-process. Its players are agent-controlled, and it sees mirrored
  raw observations exactly as in the arena. ``set_opponent(spec)`` swaps it at
  the next ``reset()``. With vector envs, use
  ``envs.call("set_opponent", path)``. Together these are the scaffolding for
  self-play (point it at an exported snapshot of yourself) and league /
  curriculum training.
- ``raw_observations()``: the learner's raw per-player observation dicts, in
  exactly the format an arena ``Agent.act`` receives (see ``arena/probe.py``).
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np

# Football Academy scenarios (gfootball/scenarios/academy_*.py), roughly by difficulty.
ACADEMY_SCENARIOS = (
    "academy_empty_goal_close",
    "academy_empty_goal",
    "academy_run_to_score",
    "academy_run_to_score_with_keeper",
    "academy_pass_and_shoot_with_keeper",
    "academy_run_pass_and_shoot_with_keeper",
    "academy_3_vs_1_with_keeper",
    "academy_corner",
    "academy_counterattack_easy",
    "academy_counterattack_hard",
    "academy_single_goal_versus_lazy",
)

# Full-game scenarios (gfootball/scenarios/*.py). 5_vs_5 is the arena scenario.
GAME_SCENARIOS = (
    "1_vs_1_easy",
    "5_vs_5",
    "11_vs_11_easy_stochastic",
    "11_vs_11_stochastic",
    "11_vs_11_hard_stochastic",
)


class GFootballEnv(gym.Env):
    """One gfootball scenario, single controlled player, Gymnasium API.

    Parameters map onto ``gfootball.env.create_environment``:

    - ``scenario``: gfootball level name (``env_name``), e.g.
      ``"academy_empty_goal_close"``.
    - ``representation``: observation encoding; ``"simple115v2"`` (flat
      vector) is the one the flat-MLP agents expect.
    - ``rewards``: ``"scoring"`` (±1 per goal) or ``"scoring,checkpoints"``
      (adds +0.1 shaping for each of 10 zones approached with the ball).
    - ``render_resolution``: ``(width, height)`` of rendered frames.
    - ``controlled_players``: left players the learner controls (module docstring).
    - ``opponent``: ``"builtin"`` | ``"random"`` | submission folder | an object
      with ``controlled_players``, ``reset()`` and ``act(obs_list)``.
    - ``**create_kwargs``: forwarded verbatim (e.g. ``stacked``,
      ``other_config_options``).
    """

    # One agent action = 100 ms of game time.
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scenario: str,
        representation: str = "simple115v2",
        rewards: str = "scoring,checkpoints",
        render_mode: str | None = None,
        render_resolution: tuple[int, int] = (640, 360),
        controlled_players: int = 1,
        opponent=None,
        **create_kwargs,
    ):
        if render_mode not in (None, "rgb_array"):
            raise ValueError(
                f"GFootball: unsupported render_mode {render_mode!r}; "
                "use None or 'rgb_array'."
            )
        self.render_mode = render_mode
        self._scenario = scenario
        self._representation = representation
        self._rewards = rewards
        self._render_resolution = tuple(int(v) for v in render_resolution)
        self._n_left = int(controlled_players)
        if self._n_left < 1:
            raise ValueError("GFootball: controlled_players must be >= 1")
        self._create_kwargs = dict(create_kwargs)
        self._user_pins_team_order = "reverse_team_processing" in (
            self._create_kwargs.get("other_config_options") or {}
        )

        self._env = None
        self._render_requested = False
        self._opponent = None
        self._pending_opponent = None
        self._opponent_obs: list = []
        self._load_opponent(opponent)
        self._build_engine()

        obs_space = self._env.observation_space
        low = self._learner_rows(np.asarray(obs_space.low, dtype=np.float32))
        high = self._learner_rows(np.asarray(obs_space.high, dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        engine_actions = self._env.action_space
        n_actions = int(getattr(engine_actions, "n", 0) or engine_actions.nvec[0])
        self.action_space = (
            gym.spaces.Discrete(n_actions)
            if self._n_left == 1
            else gym.spaces.MultiDiscrete([n_actions] * self._n_left)
        )

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def _build_engine(self) -> None:
        # Deferred import: keeps `import envs` cheap and banner-free when
        # gfootball is installed but unused.
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        from gfootball.env import create_environment

        other = dict(self._create_kwargs.get("other_config_options") or {})
        # Resolution is read when the (rendering) engine is created, so it
        # must be passed up front even though rendering starts off.
        width, height = self._render_resolution
        other.setdefault("render_resolution_x", width)
        other.setdefault("render_resolution_y", height)

        kwargs = {k: v for k, v in self._create_kwargs.items() if k != "other_config_options"}
        self._env = create_environment(
            env_name=self._scenario,
            representation=self._representation,
            rewards=self._rewards,
            number_of_left_players_agent_controls=self._n_left,
            number_of_right_players_agent_controls=self._n_right,
            render=False,
            other_config_options=other,
            **kwargs,
        )
        self._rendering_on = False

    @property
    def _n_right(self) -> int:
        return 0 if self._opponent is None else int(self._opponent.controlled_players)

    def _load_opponent(self, spec) -> None:
        if self._opponent is not None and hasattr(self._opponent, "close"):
            self._opponent.close()
        self._opponent = None
        if spec is None or spec == "builtin":
            return
        if spec == "random":
            from arena.agents import RandomAgent

            self._opponent = RandomAgent(1, seed=int(self.np_random.integers(2**31 - 1)))
        elif isinstance(spec, (str, os.PathLike)):
            from arena.agents import InProcessAgent

            self._opponent = InProcessAgent(spec)
        else:  # an agent object
            self._opponent = spec

    def set_opponent(self, spec) -> None:
        """Swap the right team's controller from the next ``reset()`` on."""
        self._pending_opponent = ("set", spec)

    def _learner_rows(self, array: np.ndarray) -> np.ndarray:
        """Engine output (one row per agent-controlled player) -> learner's view."""
        if self._n_left + self._n_right == 1:
            return array  # engine already squeezed the single player
        rows = array[: self._n_left]
        return rows[0] if self._n_left == 1 else rows

    def _raw_all(self) -> list:
        return self._env.unwrapped.observation()

    def _raw_observation(self) -> dict:
        return self._raw_all()[0]

    def raw_observations(self) -> list[dict]:
        """Raw per-player observations of the learner's players (arena ``act`` format)."""
        return self._raw_all()[: self._n_left]

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self._pending_opponent is not None:
            n_right_before = self._n_right
            self._load_opponent(self._pending_opponent[1])
            self._pending_opponent = None
            if self._n_right != n_right_before:
                self._env.close()
                self._build_engine()
        # Both keys are read by the scenario builder inside the engine's reset.
        # reverse_team_processing mirrors the processing order; upstream derives
        # it from the seed's parity once per env, we do it once per episode.
        engine_seed = int(self.np_random.integers(0, 2**31 - 1))
        config = self._env.unwrapped._config
        config["game_engine_random_seed"] = engine_seed
        if not self._user_pins_team_order:
            config["reverse_team_processing"] = bool(engine_seed % 2)
        obs = self._env.reset()
        if self._opponent is not None:
            self._opponent.reset()
            self._opponent_obs = self._raw_all()[self._n_left:]
        return self._learner_rows(np.asarray(obs, dtype=np.float32)), {}

    def step(self, action):
        self._maybe_disable_render()
        if self._n_left == 1:
            actions = [int(action)]
        else:
            actions = [int(a) for a in np.asarray(action).reshape(-1)]
        if self._opponent is not None:
            actions += [int(a) for a in self._opponent.act(self._opponent_obs)]
        engine_action = actions[0] if len(actions) == 1 else actions
        obs, reward, done, info = self._env.step(engine_action)
        info = dict(info)
        if len(actions) > 1:
            player_rewards = np.asarray(reward, dtype=np.float32)[: self._n_left]
            if self._n_left > 1:
                info["player_rewards"] = player_rewards
            reward = float(np.mean(player_rewards))
        if self._opponent is not None and not done:
            self._opponent_obs = self._raw_all()[self._n_left:]
        if done:
            raw = self._raw_observation()
            steps_left = int(raw["steps_left"])
            info["steps_left"] = steps_left
            scored = int(info.get("score_reward", 0)) != 0
            truncated = steps_left <= 0 and not scored
            terminated = not truncated
        else:
            terminated = truncated = False
        obs = self._learner_rows(np.asarray(obs, dtype=np.float32))
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        self._render_requested = True
        self._rendering_on = True
        frame = self._env.render(mode="rgb_array")
        return np.asarray(frame, dtype=np.uint8)

    def _maybe_disable_render(self) -> None:
        # Rendering was on but nobody asked for a frame since the last step:
        # stop paying for it (RecordVideo has finished its episode).
        if self._rendering_on and not self._render_requested:
            self._env.unwrapped.disable_render()
            self._rendering_on = False
        self._render_requested = False

    def close(self):
        if self._opponent is not None and hasattr(self._opponent, "close"):
            self._opponent.close()
        if self._env is not None:
            self._env.close()
            self._env = None


def make_gfootball_env(scenario: str, render_mode: str | None = None, **kwargs) -> gym.Env:
    """Registry entry point (``gym.make`` forwards ``env_kwargs`` here)."""
    return GFootballEnv(scenario, render_mode=render_mode, **kwargs)
