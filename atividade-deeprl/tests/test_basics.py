"""Testes rápidos (segundos). Rode com:  pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpraticas.agents.dqn_agent import compute_td_target  # noqa: E402
from rlpraticas.config import load_config  # noqa: E402
from rlpraticas.utils.stats import bootstrap_ci, iqm  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _target(terminated, truncated, bootstrap_on_truncation=True):
    t = lambda v: torch.tensor([float(v)])  # noqa: E731
    return compute_td_target(
        rewards=t(-1.0),
        next_values=t(-50.0),
        terminated=t(terminated),
        truncated=t(truncated),
        gamma=0.99,
        bootstrap_on_truncation=bootstrap_on_truncation,
    ).item()


def test_transicao_normal_faz_bootstrap():
    assert _target(0, 0) == pytest.approx(-1.0 + 0.99 * -50.0)


def test_terminal_nao_faz_bootstrap():
    assert _target(1, 0) == pytest.approx(-1.0)


def test_truncado_faz_bootstrap():
    assert _target(0, 1) == pytest.approx(-1.0 + 0.99 * -50.0)


def test_bug_classico_corta_bootstrap_no_truncamento():
    assert _target(0, 1, bootstrap_on_truncation=False) == pytest.approx(-1.0)


def test_terminated_tem_prioridade_sobre_truncated():
    assert _target(1, 1) == pytest.approx(-1.0)


def test_iqm_ignora_extremos():
    assert iqm(np.array([-500, -100, -90, -80, 1000])) == pytest.approx(-90.0)


def test_bootstrap_ci_contem_a_media():
    x = np.random.default_rng(0).normal(10, 1, size=20)
    low, high = bootstrap_ci(x, np.mean)
    assert low < x.mean() < high


def test_configs_carregam_e_aceitam_overrides():
    cfg = load_config(CONFIGS / "dqn_acrobot.yaml", ["hyperparameters.gamma=0.95", "variant=teste"])
    assert cfg["hyperparameters"]["gamma"] == 0.95
    assert cfg["variant"] == "teste"
    assert cfg["evaluation"]["n_eval_episodes"] == 10  # herdado de base.yaml
    load_config(CONFIGS / "ppo_cartpole.yaml")
