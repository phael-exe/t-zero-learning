"""Every committed config must load into a valid Args instance.

Parametrized over ``configs/*.yml`` — adding a config automatically adds a
test; a stale config (unknown algorithm, wrong structure) fails CI.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.config_loader import ALGORITHMS, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = sorted((REPO_ROOT / "configs").glob("*.yml")) + sorted(
    (REPO_ROOT / "configs").glob("*.yaml")
)


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_config_loads(config_path):
    args, algo_name = load_config(str(config_path))

    assert algo_name in ALGORITHMS
    assert args.algorithm == algo_name
    assert args.env_id, f"{config_path.name}: env_id must be set"
    assert args.total_timesteps > 0
    assert isinstance(args.env_kwargs, dict)
    # Nested sections resolved to dataclasses, not left as raw dicts
    assert isinstance(args.network_kwargs, dict)
    assert not isinstance(args.algo, dict)


def test_config_without_algorithm_key_exits(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("env_id: Pendulum-v1\nseed: 1\n")
    with pytest.raises(SystemExit):
        load_config(str(bad))


def test_config_with_unknown_algorithm_exits(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("algorithm: does_not_exist\nenv_id: Pendulum-v1\n")
    with pytest.raises(SystemExit):
        load_config(str(bad))
