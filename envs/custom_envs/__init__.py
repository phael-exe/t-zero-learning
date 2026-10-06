import importlib.util

from gymnasium.envs.registration import register
register(
    id="HalfCheetahVel-v1",
    entry_point="envs.custom_envs.halfcheetahvel:make_target_vel_env",
    kwargs={"target_vel": 1.0, "render_mode":"rgb_array"}
)
register(
    id="HalfCheetahVelGoal-v1",
    entry_point="envs.custom_envs.halfcheetahvel_goal:make_halfcheetahvel_goal_env",
    kwargs={
        "schedule_mode": "fixed",
        "velocities": [1.0],
        "switch_after_steps": [],
        "render_mode": "rgb_array",
    },
)
register(
    id="AntDir-v1",
    entry_point="envs.custom_envs.antdir:make_target_dir_env",
    kwargs={"target_dir": [0.0, 1.0], "render_mode":"rgb_array"}
)
register(
    id="AntDirGoal-v1",
    entry_point="envs.custom_envs.antdir_goal:make_antdir_goal_env",
    kwargs={
        "schedule_mode": "fixed",
        "directions": [[1.0, 0.0], [-1.0, 0.0]],
        "switch_after_steps": [500],
        "render_mode": "rgb_array",
    },
)

# Google Research Football — optional dependency (C++ engine, see
# docker/Dockerfile.gfootball). Registered only when the package is present so
# the rest of the framework never notices its absence.
if importlib.util.find_spec("gfootball") is not None:
    from envs.custom_envs.gfootball import ACADEMY_SCENARIOS, GAME_SCENARIOS

    for _scenario in ACADEMY_SCENARIOS + GAME_SCENARIOS:
        register(
            id=f"GFootball/{_scenario}-v0",
            entry_point="envs.custom_envs.gfootball:make_gfootball_env",
            kwargs={
                "scenario": _scenario,
                "representation": "simple115v2",
                "rewards": "scoring,checkpoints",
            },
        )
