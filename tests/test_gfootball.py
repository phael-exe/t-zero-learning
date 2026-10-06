"""Google Research Football shim: Gymnasium contract + the engine quirks it hides.

Skipped when gfootball isn't installed (it needs a compiled C++ engine — see
docker/Dockerfile.gfootball).  The generic custom-env contract tests in
tests/test_custom_envs.py also pick up the ``GFootball/*`` ids automatically
once the package is present; this file covers what is specific to the shim:
seeding, terminated/truncated, lazy rendering, and the wrapper stack / factory
path DQN uses.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("gfootball")

import envs.custom_envs  # noqa: E402,F401 — registration side effects
from envs.adapters import get_adapter  # noqa: E402
from envs.custom_envs.gfootball import ACADEMY_SCENARIOS, GAME_SCENARIOS, GFootballEnv  # noqa: E402
from envs.factory import make_env  # noqa: E402
from envs.wrappers import discrete_control_wrappers  # noqa: E402

ENV_ID = "GFootball/academy_empty_goal_close-v0"


def _run_episode(env, policy, max_steps=500):
    obs, _ = env.reset()
    total, n = 0.0, 0
    for _ in range(max_steps):
        obs, r, terminated, truncated, info = env.step(policy(obs))
        total += r
        n += 1
        if terminated or truncated:
            return total, n, terminated, truncated, info
    raise AssertionError("episode did not end")


def test_all_academy_and_game_scenarios_registered():
    for scenario in ACADEMY_SCENARIOS + GAME_SCENARIOS:
        assert f"GFootball/{scenario}-v0" in gym.registry


def test_spaces_match_discrete_control_contract():
    env = gym.make(ENV_ID)
    assert env.observation_space == gym.spaces.Box(-np.inf, np.inf, (115,), np.float32)
    assert env.action_space == gym.spaces.Discrete(19)
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32 and obs.shape == (115,)
    assert info == {}
    env.close()


def test_step_returns_gymnasium_5_tuple_with_python_scalars():
    env = GFootballEnv("academy_empty_goal_close")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (115,) and obs.dtype == np.float32
    assert type(reward) is float
    assert type(terminated) is bool and type(truncated) is bool
    assert "score_reward" in info
    env.close()


def test_seed_makes_trajectory_reproducible():
    rng = np.random.default_rng(0)
    actions = rng.integers(19, size=40)

    def rollout(seed):
        env = GFootballEnv("academy_run_to_score_with_keeper")
        env.reset(seed=seed)
        out = []
        for a in actions:
            obs, _, term, trunc, _ = env.step(int(a))
            out.append(obs)
            if term or trunc:
                break
        env.close()
        return np.stack(out)

    a, b, c = rollout(7), rollout(7), rollout(8)
    np.testing.assert_array_equal(a, b)
    assert a.shape != c.shape or not np.array_equal(a, c)


def test_episodes_differ_within_a_run_but_repeat_across_seeded_runs():
    """One run seed -> a reproducible *sequence* of distinct engine seeds, so a
    deterministic policy is evaluated on different episodes, reproducibly.
    Uses a scenario with a built-in-AI keeper: that is where the engine RNG
    actually influences the dynamics."""
    actions = np.random.default_rng(1).integers(19, size=60)

    def two_episodes(seed):
        env = GFootballEnv("academy_run_to_score_with_keeper")
        out = []
        for i in range(2):
            env.reset(seed=seed if i == 0 else None)
            obs = []
            for a in actions:
                o, _, term, trunc, _ = env.step(int(a))
                obs.append(o)
                if term or trunc:
                    break
            out.append(np.stack(obs))
        env.close()
        return out

    a1, a2 = two_episodes(5)
    b1, b2 = two_episodes(5)
    assert a1.shape != a2.shape or not np.array_equal(a1, a2)
    np.testing.assert_array_equal(a1, b1)
    np.testing.assert_array_equal(a2, b2)


def test_scenario_end_is_terminated_not_truncated():
    """Idling: the keeper collects the ball -> possession change ends the
    scenario well before the 400-step clock, so this is a true terminal."""
    env = GFootballEnv("academy_empty_goal_close")
    _, n, terminated, truncated, info = _run_episode(env, lambda obs: 0)
    assert terminated and not truncated
    assert info["steps_left"] > 0 and n < 400
    env.close()


def test_goal_gives_score_reward_and_terminates():
    """Sprint straight at the goal (action 5 = right, 13 = sprint is sticky)."""
    env = GFootballEnv("academy_empty_goal_close")
    env.reset(seed=0)
    env.step(13)  # sprint on
    total, n, terminated, truncated, info = _run_episode(env, lambda obs: 5)
    assert info["score_reward"] == 1
    assert terminated and not truncated
    env.close()


class _FakeCore:
    """Stand-in for the legacy gfootball env: one step, then done."""

    def __init__(self, steps_left, score_reward):
        self._steps_left, self._score_reward = steps_left, score_reward

    def step(self, action):
        return np.zeros(115, np.float32), np.float32(self._score_reward), True, {"score_reward": self._score_reward}

    unwrapped = property(lambda self: self)

    def observation(self):
        return [{"steps_left": self._steps_left}]

    def disable_render(self):
        pass

    def close(self):
        pass


@pytest.mark.parametrize(
    "steps_left, score_reward, expected",
    [
        (0, 0, "truncated"),    # clock ran out
        (0, 1, "terminated"),   # goal on the last tick counts as a goal
        (250, 0, "terminated"), # scenario rule (out of play / possession change)
    ],
)
def test_done_split_from_steps_left(steps_left, score_reward, expected):
    env = GFootballEnv("academy_empty_goal_close")
    env.reset(seed=0)
    env._env.close()
    env._env = _FakeCore(steps_left, score_reward)
    _, _, terminated, truncated, info = env.step(0)
    assert (terminated, truncated) == ((expected == "terminated"), (expected == "truncated"))
    assert info["steps_left"] == steps_left


def test_rendering_is_lazy_and_switches_off():
    env = GFootballEnv("academy_empty_goal_close", render_mode="rgb_array", render_resolution=(320, 180))
    env.reset(seed=0)
    assert not env._rendering_on
    env.step(0)
    frame = env.render()
    assert frame.shape == (180, 320, 3) and frame.dtype == np.uint8
    assert env._rendering_on
    env.step(0)  # rendered frame was consumed by the render() above -> stays on
    assert env._rendering_on
    env.step(0)  # nobody asked for a frame during the previous step -> off
    assert not env._rendering_on
    env.close()


def test_render_mode_none_returns_nothing():
    env = GFootballEnv("academy_empty_goal_close")
    env.reset(seed=0)
    assert env.render() is None
    env.close()


def test_adapter_disables_training_video_only():
    adapter = get_adapter(ENV_ID)
    assert adapter.supports_training_video is False
    assert adapter.skip_episode_stats is False
    assert adapter.apply_wrappers is None and adapter.make_vector_env is None


def test_discrete_control_stack_and_factory_thunk():
    thunk = make_env(
        ENV_ID, idx=0, capture_video=False, run_name="t", gamma=0.99,
        env_kwargs={"rewards": "scoring"}, wrappers=discrete_control_wrappers,
    )
    env = thunk()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (115,)
    total, _, _, _, info = _run_episode(env, lambda obs: 0)
    assert "episode" in info  # RecordEpisodeStatistics applied by the stack
    assert float(info["episode"]["r"]) == pytest.approx(total)
    env.close()


def test_env_kwargs_reward_variant():
    """'scoring' only: an idle episode has exactly zero return."""
    env = gym.make(ENV_ID, rewards="scoring")
    total, *_ = _run_episode(env, lambda obs: 0)
    assert total == 0.0
    env.close()


# ---------------------------------------------------------------------------
# Hooks for student extensions: several players, opponents, raw observations
# ---------------------------------------------------------------------------

TEMPLATE_SUBMISSION = str(Path(__file__).resolve().parents[1] / "arena" / "template")


class _RecordingOpponent:
    """Opponent object: records what it is shown, always idles."""

    controlled_players = 1

    def __init__(self):
        self.resets = 0
        self.seen = []

    def reset(self):
        self.resets += 1

    def act(self, observations):
        self.seen.append(observations)
        return [0] * len(observations)


def test_raw_observations_are_the_arena_format_and_match_the_vector():
    from gfootball.env.wrappers import Simple115StateWrapper

    env = GFootballEnv("5_vs_5", rewards="scoring")
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, *_ = env.step(5)
    raw = env.raw_observations()
    assert len(raw) == 1 and {"active", "left_team", "sticky_actions", "designated"} <= set(raw[0])
    np.testing.assert_array_equal(Simple115StateWrapper.convert_observation(raw, True)[0], obs)
    env.close()


def test_several_controlled_players():
    env = GFootballEnv("5_vs_5", controlled_players=2)
    assert env.observation_space.shape == (2, 115)
    assert env.action_space == gym.spaces.MultiDiscrete([19, 19])
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2, 115)
    obs, reward, terminated, truncated, info = env.step(np.array([5, 3]))
    assert obs.shape == (2, 115) and type(reward) is float
    assert info["player_rewards"].shape == (2,)
    assert len(env.raw_observations()) == 2
    env.close()


def test_discrete_control_stack_rejects_several_players():
    """Loud failure: (N, 115) obs trip the stack's image guard before the action check."""
    env = gym.make("GFootball/5_vs_5-v0", controlled_players=2)
    with pytest.raises(TypeError):
        discrete_control_wrappers(env, "GFootball/5_vs_5-v0", 0.99)
    env.close()


@pytest.mark.parametrize("opponent", ["random", TEMPLATE_SUBMISSION])
def test_opponent_specs_keep_the_single_agent_contract(opponent):
    env = gym.make("GFootball/5_vs_5-v0", opponent=opponent)
    assert env.observation_space.shape == (115,)
    assert env.action_space == gym.spaces.Discrete(19)
    obs, _ = env.reset(seed=0)
    for _ in range(20):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert obs.shape == (115,) and type(reward) is float
    env.close()


def test_opponent_sees_its_own_mirrored_view_and_is_reset():
    opponent = _RecordingOpponent()
    env = GFootballEnv("5_vs_5", opponent=opponent)
    env.reset(seed=0)
    env.step(0)
    assert opponent.resets == 1
    kickoff = opponent.seen[0][0]
    # Right team, but it sees itself as left_team on its own (negative-x) half.
    assert np.all(np.asarray(kickoff["left_team"])[:, 0] <= 0.01)
    env.reset(seed=1)
    assert opponent.resets == 2
    env.close()


def test_set_opponent_applies_at_next_reset_and_rebuilds_when_needed():
    env = GFootballEnv("5_vs_5")
    env.reset(seed=0)
    assert env._n_right == 0
    opponent = _RecordingOpponent()
    env.set_opponent(opponent)
    env.step(0)
    assert opponent.seen == []          # not before the next reset
    env.reset(seed=1)
    assert env._n_right == 1
    env.step(0)
    assert len(opponent.seen) == 1
    env.set_opponent("builtin")
    env.reset(seed=2)
    assert env._n_right == 0
    env.step(0)
    env.close()
