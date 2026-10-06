"""Arena: submission contract, isolated agent host, matches, check, tournament, export.

Skipped without gfootball (run inside the image:
``docker compose run --rm gfootball python -m pytest tests/test_arena.py``).
Matches are cut to a few dozen steps; the full-length behaviour is the same
code path.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gfootball")

from arena import tournament  # noqa: E402
from arena.agents import (  # noqa: E402
    AgentError, BuiltinAI, InProcessAgent, RandomAgent, RemoteAgent, read_manifest,
)
from arena.check import run_check  # noqa: E402
from arena.match import play_match, play_specs  # noqa: E402
from arena.probe import PROBE_FILE, ProbeRecorder, load_probe  # noqa: E402
from arena.ratings import bradley_terry, standings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "arena" / "template"

CONSTANT_AGENT = """
class Agent:
    def __init__(self, path):
        pass
    def act(self, observations):
        return [{action}] * len(observations)
"""


def make_submission(root: Path, name: str, code: str, controlled_players: int = 1,
                    deterministic: bool = True, files: dict | None = None) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "agent.py").write_text(textwrap.dedent(code))
    (folder / "manifest.yml").write_text(
        f"team: {name}\ncontrolled_players: {controlled_players}\n"
        f"deterministic: {str(deterministic).lower()}\n"
    )
    for rel, content in (files or {}).items():
        (folder / rel).write_text(textwrap.dedent(content))
    return folder


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("body, message", [
    ("controlled_players: 1\n", "'team' is required"),
    ("team: x\n", "controlled_players"),
    ("team: x\ncontrolled_players: 5\n", "1..4"),
    ("team: x\ncontrolled_players: true\n", "controlled_players"),
    ("team: x\ncontrolled_players: 1\ndeterministic: maybe\n", "deterministic"),
])
def test_manifest_validation(tmp_path, body, message):
    (tmp_path / "manifest.yml").write_text(body)
    with pytest.raises(ValueError, match=message):
        read_manifest(tmp_path)


def test_manifest_reports_unknown_keys(tmp_path):
    (tmp_path / "manifest.yml").write_text("team: x\ncontrolled_players: 2\ncolour: red\n")
    m = read_manifest(tmp_path)
    assert (m.team, m.controlled_players, m.deterministic, m.unknown_keys) == ("x", 2, True, ["colour"])


# ---------------------------------------------------------------------------
# Agent host: isolation and failure modes
# ---------------------------------------------------------------------------

def _one_obs():
    from arena.match import _make_engine

    env = _make_engine(1, 0, seed=0, scenario="5_vs_5", render=False, dump_dir=None)
    obs = env.reset()
    env.close()
    return obs


def test_template_agent_runs_in_host():
    obs = _one_obs()
    with RemoteAgent(TEMPLATE) as agent:
        agent.reset()
        actions = agent.act(obs)
    assert len(actions) == 1 and 0 <= actions[0] < 19
    assert agent.load_seconds is not None


@pytest.mark.parametrize("code, kind, text", [
    ("raise RuntimeError('boom at import')\n", "crash", "boom at import"),
    ("X = 1\n", "crash", "does not define a class named Agent"),
    ("class Agent:\n    def __init__(self, path):\n        raise ValueError('bad weights')\n",
     "crash", "bad weights"),
])
def test_load_failures_are_agent_errors(tmp_path, code, kind, text):
    folder = make_submission(tmp_path, "broken", code)
    with pytest.raises(AgentError) as info:
        RemoteAgent(folder)
    assert info.value.kind == kind and text in info.value.detail


@pytest.mark.parametrize("returned, kind, text", [
    ("[]", "invalid", "returned 0 action(s)"),
    ("[19]", "invalid", "outside [0, 19)"),
    ("[1.5]", "invalid", "must return a list"),
    ("'shoot'", "invalid", "must return a list"),
    ("1 / 0", "crash", "ZeroDivisionError"),
])
def test_act_failures_are_agent_errors(tmp_path, returned, kind, text):
    code = f"""
    class Agent:
        def __init__(self, path):
            pass
        def act(self, observations):
            return {returned}
    """
    folder = make_submission(tmp_path, "bad_act", code)
    with RemoteAgent(folder) as agent:
        with pytest.raises(AgentError) as info:
            agent.act(_one_obs())
    assert info.value.kind == kind and text in info.value.detail


def test_numpy_and_bare_int_returns_are_accepted(tmp_path):
    code = """
    import numpy as np
    class Agent:
        def __init__(self, path):
            self.calls = 0
        def act(self, observations):
            self.calls += 1
            return np.array([5], dtype=np.int64) if self.calls == 1 else 12
    """
    obs = _one_obs()
    with RemoteAgent(make_submission(tmp_path, "np", code)) as agent:
        assert agent.act(obs) == [5]
        assert agent.act(obs) == [12]


def test_slow_act_times_out(tmp_path):
    code = """
    import time
    class Agent:
        def __init__(self, path):
            self.calls = 0
        def act(self, observations):
            self.calls += 1
            if self.calls > 1:      # the first call gets a warm-up allowance
                time.sleep(3)
            return [0]
    """
    obs = _one_obs()
    with RemoteAgent(make_submission(tmp_path, "slow", code)) as agent:
        agent.act(obs)
        with pytest.raises(AgentError) as info:
            agent.act(obs)
    assert info.value.kind == "timeout"


def test_prints_do_not_corrupt_the_protocol(tmp_path):
    code = """
    import os
    print("hello from import")
    class Agent:
        def __init__(self, path):
            print("loading")
        def act(self, observations):
            print("acting", len(observations))
            os.write(1, b"raw fd write\\n")
            return [3]
    """
    with RemoteAgent(make_submission(tmp_path, "chatty", code)) as agent:
        assert agent.act(_one_obs()) == [3]


def test_submission_cannot_import_the_framework_but_can_import_its_own_modules(tmp_path):
    uses_framework = make_submission(tmp_path, "uses_framework", "import networks\n" + CONSTANT_AGENT.format(action=0))
    with pytest.raises(AgentError, match="No module named 'networks'"):
        RemoteAgent(uses_framework)

    own_module = make_submission(
        tmp_path, "own_module", "from helpers import ACTION\n" + CONSTANT_AGENT.format(action="ACTION"),
        files={"helpers.py": "ACTION = 7\n"},
    )
    with RemoteAgent(own_module) as agent:
        assert agent.act(_one_obs()) == [7]


def test_in_process_agent_matches_remote_contract():
    obs = _one_obs()
    agent = InProcessAgent(TEMPLATE)
    agent.reset()
    with RemoteAgent(TEMPLATE) as remote:
        assert agent.act(obs) == remote.act(obs)


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------

# Asserts the arena's promise: whatever side it plays, an agent sees its own
# team as left_team, on its own half at kick-off, and attacking towards +x.
OWN_VIEW_AGENT = """
import numpy as np
class Agent:
    def __init__(self, path):
        self.first = True
    def reset(self):
        self.first = True
    def act(self, observations):
        if self.first:
            self.first = False
            for obs in observations:
                me = obs["active"]
                assert 0 <= me < len(obs["left_team"]), obs["active"]
                assert np.all(np.asarray(obs["left_team"])[:, 0] <= 0.01), obs["left_team"]
                assert np.all(np.asarray(obs["right_team"])[:, 0] >= -0.05), obs["right_team"]
        return [5] * len(observations)   # 5 = "right" = towards the opponent goal
"""


@pytest.mark.parametrize("our_side", ["left", "right"])
def test_both_sides_see_their_own_view(tmp_path, our_side):
    folder = make_submission(tmp_path, "view", OWN_VIEW_AGENT, controlled_players=2)
    with RemoteAgent(folder) as ours:
        other = RandomAgent(1, seed=0)
        left, right = (ours, other) if our_side == "left" else (other, ours)
        result = play_match(left, right, seed=1, max_steps=40)
    assert result.forfeit == "", result.forfeit_reason
    assert result.steps == 40
    assert result.action_counts["view"] == {5: 80}   # 2 players x 40 steps


def test_forfeit_on_crash_awards_the_match_to_the_opponent(tmp_path):
    code = """
    class Agent:
        def __init__(self, path):
            self.n = 0
        def act(self, observations):
            self.n += 1
            if self.n == 10:
                raise RuntimeError("mid-match crash")
            return [0]
    """
    with RemoteAgent(make_submission(tmp_path, "crasher", code)) as crasher:
        result = play_match(RandomAgent(1, seed=0), crasher, max_steps=50)
    assert result.forfeit == "crasher"
    assert "mid-match crash" in result.forfeit_reason
    assert (result.goals_left, result.goals_right) == (3, 0)
    assert result.winner == "random"


def test_load_failure_is_a_forfeit_in_play_specs(tmp_path):
    folder = make_submission(tmp_path, "noload", "raise ImportError('missing torch')\n")
    result = play_specs("builtin", str(folder), seed=3, max_steps=10)
    assert result.forfeit == "noload" and (result.goals_left, result.goals_right) == (3, 0)


def test_builtin_vs_submission_and_same_name_sides():
    result = play_specs("builtin", str(TEMPLATE), seed=4, max_steps=20)
    assert result.forfeit == "" and result.steps == 20
    result = play_specs(str(TEMPLATE), str(TEMPLATE), seed=5, max_steps=20)
    assert result.left != result.right


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def _levels(report):
    return {(f.check, f.level) for f in report.findings}


def test_template_passes_check_with_probe_warning():
    report = run_check(TEMPLATE, matches=1, max_steps=40)
    assert report.passed, report.findings
    assert ("probe", "WARN") in _levels(report)
    assert ("speed", "PASS") in _levels(report)


def test_check_fails_crashing_agent_and_warns_on_constant_actions(tmp_path):
    bad = make_submission(tmp_path, "bad", "raise SystemExit(3)\n")
    assert not run_check(bad, matches=0).passed

    constant = make_submission(tmp_path, "constant", CONSTANT_AGENT.format(action=0))
    report = run_check(constant, matches=1, max_steps=30)
    assert report.passed
    assert ("behaviour", "WARN") in _levels(report)


def test_check_fails_oversized_submission(tmp_path, monkeypatch):
    import arena.check as check
    from arena.rules import Rules

    folder = make_submission(tmp_path, "big", CONSTANT_AGENT.format(action=0),
                             files={"blob.bin": "x" * 20000})
    monkeypatch.setattr(check, "RULES", Rules(max_submission_mb=0.01))
    assert ("size", "FAIL") in _levels(check.run_check(folder, matches=0))


def _record_probe(folder: Path, agent, steps=30):
    from arena.match import _make_engine

    rec = ProbeRecorder(controlled_players=1)
    env = _make_engine(1, 0, seed=9, scenario="5_vs_5", render=False, dump_dir=None)
    obs = env.reset()
    rec.new_episode()
    rng = np.random.default_rng(0)
    for _ in range(steps):
        rec.add(obs, agent.act(obs))
        obs, *_ = env.step([int(rng.integers(19))])
    env.close()
    rec.save(folder / PROBE_FILE)


def test_probe_replay_passes_for_same_agent_and_fails_for_a_different_one(tmp_path):
    good = tmp_path / "good"
    import shutil

    shutil.copytree(TEMPLATE, good)
    _record_probe(good, InProcessAgent(TEMPLATE))
    assert load_probe(good / PROBE_FILE)["controlled_players"] == 1
    assert ("probe", "PASS") in _levels(run_check(good, matches=0))

    impostor = make_submission(tmp_path, "impostor", CONSTANT_AGENT.format(action=0))
    shutil.copy(good / PROBE_FILE, impostor / PROBE_FILE)
    assert ("probe", "FAIL") in _levels(run_check(impostor, matches=0))


# ---------------------------------------------------------------------------
# Ratings and tournament
# ---------------------------------------------------------------------------

def _r(left, right, gl, gr, forfeit=""):
    return dict(left=left, right=right, goals_left=gl, goals_right=gr, forfeit=forfeit)


def test_standings_and_bradley_terry_order():
    results = []
    for _ in range(6):
        results += [_r("A", "B", 2, 0), _r("B", "C", 1, 0), _r("C", "A", 0, 3), _r("B", "A", 1, 1)]
    table = standings(results)
    assert [row["name"] for row in table] == ["A", "B", "C"]
    a = table[0]
    assert (a["w"], a["d"], a["l"], a["points"]) == (12, 6, 0, 42)
    ratings = bradley_terry(results, bootstrap=50)
    assert [r["name"] for r in ratings] == ["A", "B", "C"]
    assert all(r["low"] <= r["rating"] <= r["high"] for r in ratings)
    assert abs(np.mean([r["rating"] for r in ratings]) - 1000) < 1e-6


def test_tournament_runs_resumes_and_extends(tmp_path, capsys):
    subs = tmp_path / "subs"
    import shutil

    shutil.copytree(TEMPLATE, subs / "scripted")
    make_submission(subs, "idle", CONSTANT_AGENT.format(action=0))
    make_submission(subs, "broken", "raise RuntimeError('nope')\n")
    out = tmp_path / "out"
    common = [str(subs), "--out", str(out), "--anchors", "builtin", "--workers", "2",
              "--max-steps", "20", "--bootstrap", "10"]

    assert tournament.main(common + ["-n", "2"]) == 0
    rows = tournament.load_results(out)
    assert len(rows) == 3 * 2  # 3 entrants (broken excluded) -> 3 pairs x 2
    assert "broken" in json.loads((out / "excluded.json").read_text()).popitem()[0]
    assert {r["left"] for r in rows} | {r["right"] for r in rows} == {"template-scripted", "idle", "[builtin]"}
    for name in ("standings.csv", "ratings.csv", "pairwise_gd.csv"):
        assert (out / name).is_file()

    capsys.readouterr()
    tournament.main(common + ["-n", "2"])
    assert "0 matches to play" in capsys.readouterr().out
    tournament.main(common + ["-n", "3"])
    assert len(tournament.load_results(out)) == 3 * 3


# ---------------------------------------------------------------------------
# Export: a framework run becomes a self-contained submission
# ---------------------------------------------------------------------------

def _fake_dqn_run(tmp_path):
    """A run folder as DQN training leaves it: config.yml and model.pt."""
    import gymnasium as gym
    import torch
    from types import SimpleNamespace

    from algorithms.dqn import Args
    from core.checkpoint import write_run_config_yaml
    from networks import QNetwork

    args = Args()
    args.env_id = "GFootball/academy_empty_goal_close-v0"
    args.env_kwargs = {"representation": "simple115v2", "rewards": "scoring,checkpoints"}
    args.network_kwargs = {"activation": "ReLU", "hidden_layers_size": 32}
    run = tmp_path / "runs" / "exp"
    write_run_config_yaml(str(run), "run1", args)
    torch.manual_seed(0)
    stub = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (115,), np.float32),
                           single_action_space=gym.spaces.Discrete(19))
    torch.save(QNetwork(stub, **args.network_kwargs).state_dict(), run / "run1" / "model.pt")
    return run / "run1"


def test_export_roundtrip_passes_check(tmp_path):
    from scripts import export_submission

    run = _fake_dqn_run(tmp_path)
    out = tmp_path / "sub"
    assert export_submission.main([str(run), "--out", str(out), "--team", "dqn-test",
                                   "--probe-episodes", "1", "--probe-steps", "40"]) == 0
    assert read_manifest(out).team == "dqn-test"
    report = run_check(out, matches=1, max_steps=30)
    assert report.passed, report.findings
    probe = [f for f in report.findings if f.check == "probe"][0]
    assert probe.level == "PASS" and "40/40" in probe.message

