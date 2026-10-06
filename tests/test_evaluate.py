"""evaluate.py: rebuilding eval settings from a saved run ``config.yml``.

The run's ``algorithm`` must supply the default network and wrapper stack, so
non-PPO runs are not evaluated with PPO's continuous-control defaults.
"""
from __future__ import annotations

import sys

import pytest
import yaml

import evaluate
from envs.wrappers import continuous_control_wrappers, discrete_control_wrappers
from networks import get_network


def test_dqn_run_uses_dqn_defaults():
    cfg = {"algorithm": "dqn", "env_id": "CartPole-v1", "env_wrappers": "", "dqn": {"gamma": 0.9}}
    run_cfg = evaluate.parse_run_config(cfg)
    assert run_cfg["Model"] is get_network("QNetwork")
    assert run_cfg["wrappers"] is discrete_control_wrappers
    assert run_cfg["gamma"] == 0.9


def test_legacy_dqn_agent_section_builds_qnetwork():
    # Pre-network_kwargs configs carried an ``agent:`` section and no ``network:``.
    cfg = {
        "algorithm": "dqn",
        "env_id": "CartPole-v1",
        "agent": {"hidden_layers_size": 32, "use_obs_norm": False},
    }
    run_cfg = evaluate.parse_run_config(cfg)
    assert run_cfg["Model"] is get_network("QNetwork")
    assert run_cfg["wrappers"] is discrete_control_wrappers


def test_ppo_run_uses_ppo_defaults():
    cfg = {"algorithm": "ppo_continuous_action", "env_id": "Pendulum-v1"}
    run_cfg = evaluate.parse_run_config(cfg)
    assert run_cfg["Model"] is get_network("ContinuousActorCritic")
    assert run_cfg["wrappers"] is continuous_control_wrappers


def test_config_without_algorithm_key_falls_back_to_ppo():
    run_cfg = evaluate.parse_run_config({"env_id": "Pendulum-v1", "activation": "tanh"})
    assert run_cfg["Model"] is get_network("ContinuousActorCritic")
    assert run_cfg["wrappers"] is continuous_control_wrappers


def test_explicit_network_and_wrappers_win_over_defaults():
    cfg = {
        "algorithm": "dqn",
        "env_id": "CartPole-v1",
        "network": "networks.discrete_actor_critic.DiscreteActorCritic",
        "env_wrappers": "continuous_control",
    }
    run_cfg = evaluate.parse_run_config(cfg)
    assert run_cfg["Model"] is get_network("networks.discrete_actor_critic.DiscreteActorCritic")
    assert run_cfg["wrappers"] is continuous_control_wrappers


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError, match="unknown algorithm"):
        evaluate.parse_run_config({"algorithm": "nope", "env_id": "CartPole-v1"})


def test_env_with_custom_eval_protocol_is_rejected(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / evaluate.MODEL_FILE).write_bytes(b"")
    cfg = {"algorithm": "ppo_continuous_action", "env_id": "Meta-World/MT10"}
    (run_dir / evaluate.CONFIG_FILE).write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", str(run_dir), "--no-video"])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 1
    assert "custom evaluation protocol" in capsys.readouterr().err
