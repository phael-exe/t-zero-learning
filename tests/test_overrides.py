"""CLI override semantics (``apply_overrides``).

The override machinery is algorithm-agnostic (it walks dataclass attributes),
so these tests use the canonical PPO ``Args`` as a representative instance —
they do not need to be repeated per algorithm.

Pins the current behavior: the algorithm section is addressed by its field
name, which equals the YAML section name (``ppo_continuous_action.``), with
``algo.`` as an algorithm-agnostic shorthand — and malformed or unknown
overrides are hard errors (SystemExit), never silently ignored.
"""
from __future__ import annotations

import pytest

from algorithms.ppo_continuous_action import Args
from core.config_loader import apply_overrides


@pytest.fixture
def args() -> Args:
    return Args()


def test_top_level_int(args):
    apply_overrides(args, ["seed=123"])
    assert args.seed == 123


def test_dotted_key_inside_dict_field(args):
    # One network constructor arg at a time; values are parsed as YAML, and
    # the other keys already in the dict are kept.
    args.network_kwargs = {"activation": "Tanh"}
    apply_overrides(args, ["network_kwargs.hidden_layers_size=128", "network_kwargs.use_obs_norm=true"])
    assert args.network_kwargs == {"activation": "Tanh", "hidden_layers_size": 128, "use_obs_norm": True}


def test_dotted_nested_algo_section_name(args):
    # The field name equals the YAML section name — the same spelling works
    # in configs and in overrides.
    apply_overrides(args, ["ppo_continuous_action.gamma=0.98"])
    assert args.algo.gamma == 0.98


def test_algo_shorthand_alias(args):
    # `algo.` traverses the Args.algo property — an algorithm-agnostic alias.
    apply_overrides(args, ["algo.num_steps=512"])
    assert args.ppo_continuous_action.num_steps == 512


def test_bool_coercion(args):
    for raw, expected in [("true", True), ("1", True), ("yes", True),
                          ("false", False), ("0", False), ("nope", False)]:
        apply_overrides(args, [f"track={raw}"])
        assert args.track is expected, f"track={raw!r}"


def test_float_coercion(args):
    apply_overrides(args, ["algo.learning_rate=0.001"])
    assert args.algo.learning_rate == pytest.approx(1e-3)


def test_dict_field_parsed_as_json(args):
    apply_overrides(args, ['env_kwargs={"target_vel": 0.5}'])
    assert args.env_kwargs == {"target_vel": 0.5}


def test_unknown_key_is_fatal(args, capsys):
    with pytest.raises(SystemExit):
        apply_overrides(args, ["not_a_real_key=1"])
    assert "Error" in capsys.readouterr().out


def test_unknown_dotted_path_is_fatal(args, capsys):
    with pytest.raises(SystemExit):
        apply_overrides(args, ["not_a_section.gamma=0.5"])
    out = capsys.readouterr().out
    assert "Error" in out
    assert "Valid keys" in out  # error lists the valid keys of the target


def test_missing_equals_sign_is_fatal(args, capsys):
    with pytest.raises(SystemExit):
        apply_overrides(args, ["seed"])
    assert "Error" in capsys.readouterr().out


def test_none_overrides_is_noop(args):
    apply_overrides(args, None)
