"""Meta-World support, isolated as an "added environment".

This is the worked example of extending the framework to a quirky environment
*without* touching the core files.  Everything Meta-World-specific lives here:

- which multi-task benchmarks need ``gym.make_vec`` (MT10/MT25/MT50),
- how to build those vector envs,
- that Meta-World records its own episode statistics,
- that training-time video isn't supported for the vector benchmarks,
- how to evaluate a multi-task checkpoint per-task.

The core framework never imports this module directly; it is imported for its
registration side effects from ``envs/adapters/__init__.py`` (the "bundled
adapters" block).  If Meta-World isn't installed, registration is skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym

from envs.adapters.base import EnvAdapter, register_adapter

if TYPE_CHECKING:
    from algorithms.base import Algorithm

# Meta-World multi-task benchmarks expose a ``vector_entry_point`` and must be
# built with ``gym.make_vec`` rather than the standard ``AsyncVectorEnv`` path.
_VECTOR_BENCHMARKS = ("Meta-World/MT10", "Meta-World/MT25", "Meta-World/MT50")


# ---------------------------------------------------------------------------
# Custom vectoriser
# ---------------------------------------------------------------------------


def _make_vector_env(env_id: str, env_kwargs: dict, seed: int) -> gym.vector.VectorEnv:
    """Build MT10/MT25/MT50 via a single ``gym.make_vec`` call."""
    kw = dict(env_kwargs) if env_kwargs else {}
    use_one_hot = kw.pop("use_one_hot", True)
    vector_strategy = kw.pop("vector_strategy", "async")
    terminate_on_success = kw.pop("terminate_on_success", False)
    return gym.make_vec(  # type: ignore[call-overload]
        env_id,
        seed=seed,
        use_one_hot=use_one_hot,
        vector_strategy=vector_strategy,
        terminate_on_success=terminate_on_success,
        **kw,
    )


# ---------------------------------------------------------------------------
# Custom evaluation protocol
# ---------------------------------------------------------------------------


def _evaluate(
    algo: "Algorithm",
    model_path: str,
    eval_episodes: int,
    deterministic: bool,
) -> dict:
    """Evaluate a multi-task checkpoint per-task and shape results for logging.

    Delegates the heavy lifting to ``scripts/eval_metaworld.py`` (also usable as
    a standalone CLI), then flattens the per-task summary into the framework's
    standard ``{"episodic_returns", "metrics"}`` contract.
    """
    from scripts.eval_metaworld import evaluate_metaworld

    args = algo.args
    result = evaluate_metaworld(
        model_path=model_path,
        env_id=algo.env_id,
        env_kwargs=algo.env_kwargs,
        device=algo.device,
        experiment_dir=algo.experiment_dir,
        run_name=algo.run_name,
        eval_episodes=eval_episodes,
        capture_video=args.capture_video,
        deterministic=deterministic,
        base_seed=int(args.seed),
        save_json=True,
        Model=type(algo.agent),
        model_kwargs=args.network_kwargs,
    )
    if result is None:
        # Not a vector benchmark (e.g. single-task MT1) — fall back to standard.
        from algorithms.base import evaluate_checkpoint

        return evaluate_checkpoint(
            model_path=model_path,
            env_id=algo.env_id,
            Model=type(algo.agent),
            device=algo.device,
            eval_episodes=eval_episodes,
            capture_video=args.capture_video,
            gamma=args.algo.gamma,
            experiment_dir=algo.experiment_dir,
            run_name=algo.run_name,
            env_kwargs=algo.env_kwargs,
            model_kwargs=args.network_kwargs,
            deterministic=deterministic,
            wrappers=algo.wrappers,
        )

    metrics = {
        "eval/mean_return": float(result["mean_return"]),
        "eval/mean_success_rate": float(result["mean_success_rate"]),
    }
    for task, value in result["per_task_return"].items():
        metrics[f"eval/return_{task.replace('-', '_')}"] = float(value)
    for task, value in result["per_task_success"].items():
        metrics[f"eval/success_{task.replace('-', '_')}"] = float(value)

    all_returns: list[float] = []
    for rets in result.get("per_task_all_returns", {}).values():
        all_returns.extend(rets)

    return {"episodic_returns": all_returns, "metrics": metrics}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register() -> None:
    # Vector benchmarks: custom vectoriser + per-task eval, no training video.
    for env_id in _VECTOR_BENCHMARKS:
        register_adapter(
            env_id,
            EnvAdapter(
                make_vector_env=_make_vector_env,
                skip_episode_stats=True,
                supports_training_video=False,
                evaluate=_evaluate,
            ),
        )
    # Any other Meta-World env (e.g. single-task ``Meta-World/MT1``) still
    # records its own episode statistics, but is otherwise standard.
    register_adapter("Meta-World/", EnvAdapter(skip_episode_stats=True))


try:
    import metaworld  # noqa: F401 — registers Meta-World/* env ids with gymnasium

    _register()
except ImportError:
    # Meta-World not installed — nothing registers, framework stays generic.
    pass
