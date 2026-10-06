# Adding a New Environment

This guide explains how environments flow through the framework and how to add
a new one — from a stock Gymnasium env (zero code) to a fully custom
environment with its own vectoriser and evaluation protocol.

## How an environment travels through the code

Everything starts from two config fields and ends in a vectorised Gymnasium env:

```
configs/my_config.yml          train.py              algorithms/ppo_continuous_action.py
┌────────────────────┐    ┌───────────────┐    ┌─────────────────────────────────┐
│ env_id: MyEnv-v1   │ →  │ loads YAML,   │ →  │ initialize() calls              │
│ env_kwargs: {...}  │    │ builds args   │    │ envs.build_vector_envs(...)     │
└────────────────────┘    └───────────────┘    └─────────────────────────────────┘
                                                          │
                                  ┌───────────────────────┴───────────────────┐
                                  │ envs/factory.py::build_vector_envs        │
                                  │  ├─ adapter.make_vector_env set?          │
                                  │  │   yes → env's own vectoriser           │
                                  │  │        (e.g. gym.make_vec, Meta-World) │
                                  │  └─ no  → AsyncVectorEnv of make_env      │
                                  │           thunks, one per worker          │
                                  └───────────────────────┬───────────────────┘
                                                          │ (per worker)
                                  ┌───────────────────────┴───────────────────┐
                                  │ envs/factory.py::make_env thunk           │
                                  │  1. gym.make(env_id, **env_kwargs)        │
                                  │  2. video wrappers (worker 0 only)        │
                                  │  3. wrapper stack: adapter override, or   │
                                  │     the algorithm's default (see below)   │
                                  └───────────────────────────────────────────┘
```

The key design rule: **the core framework never names a specific environment**.
Anything env-specific is declared in one of two places:

| Concern | Where it lives |
|---|---|
| The environment itself (dynamics, reward, obs) | `envs/custom_envs/` + Gym registration |
| Behavioural quirks (custom vectoriser, eval protocol, wrapper skips) | `envs/adapters/` |

Files you will touch, depending on the case:

- [envs/custom_envs/__init__.py](../envs/custom_envs/__init__.py) — Gym registrations for packaged envs
- [envs/custom_envs/](../envs/custom_envs/) — env implementations (wrappers or full envs)
- [envs/adapters/__init__.py](../envs/adapters/__init__.py) — adapter imports ("bundled adapters" block)
- [envs/adapters/](../envs/adapters/) — one adapter module per quirky env
- [configs/](../configs/) — one YAML config per experiment

You should **not** need to touch `envs/factory.py`, `envs/wrappers.py`,
`train.py`, or the algorithm files.

---

## Case 1 — The env is already registered with Gymnasium

Example: `Hopper-v5`, `Humanoid-v5`, or any env registered by an installed
package (`import metaworld` registers `Meta-World/*`, etc.).

**No code needed.** Create a config in `configs/`:

```yaml
algorithm: ppo_continuous_action
seed: 42
env_id: Hopper-v5
env_kwargs: {}            # forwarded verbatim to gym.make(env_id, **env_kwargs)
total_timesteps: 1000000
num_envs: 8
# ... agent / ppo_continuous_action sections (copy from an existing config)
```

Run it:

```bash
python train.py --config my_config          # name without .yml, from configs/
```

`env_kwargs` can be overridden from the CLI as JSON:

```bash
python train.py --config my_config --override 'env_kwargs={"forward_reward_weight": 2.0}'
```

If the package registers the env but the env has quirks (records its own
episode statistics, needs `gym.make_vec`, ...), see Case 4.

## Case 2 — New task on top of an existing Gym env (reward/obs wrapper)

This is the most common case for research: reuse an existing env's dynamics
but change the reward or observation. Pattern:
[envs/custom_envs/halfcheetahvel.py](../envs/custom_envs/halfcheetahvel.py).

**Step 1.** Create `envs/custom_envs/myenv.py` with a `gym.Wrapper` and a
factory function:

```python
import gymnasium as gym
import numpy as np


class MyTaskWrapper(gym.Wrapper):
    def __init__(self, env, my_param=1.0):
        super().__init__(env)
        self.my_param = my_param

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = ...  # your task reward, typically from info fields
        return obs, reward, terminated, truncated, info


def make_my_env(render_mode="rgb_array", **kwargs):
    base_env = gym.make("HalfCheetah-v5", render_mode=render_mode)
    return MyTaskWrapper(base_env, **kwargs)
```

Notes:

- Accept and forward `render_mode` — the factory passes
  `render_mode="rgb_array"` when the run records video.
- Everything in the config's `env_kwargs` arrives in `**kwargs` here. Defaults
  in the registration below act as documentation of the available knobs.
- If you extend the observation (e.g. append goal features, as
  [envs/custom_envs/antdir_goal.py](../envs/custom_envs/antdir_goal.py) does),
  use `gym.ObservationWrapper` or update `self.observation_space` accordingly —
  the agent's input size is read from `envs.single_observation_space`.

**Step 2.** Register it in
[envs/custom_envs/__init__.py](../envs/custom_envs/__init__.py):

```python
register(
    id="MyEnv-v1",
    entry_point="envs.custom_envs.myenv:make_my_env",
    kwargs={"my_param": 1.0, "render_mode": "rgb_array"},
)
```

That module is imported for its side effects by `envs/factory.py`, so
registration is automatic everywhere (training, evaluation, scripts).

**Step 3.** Write a config with `env_id: MyEnv-v1` and your `env_kwargs`
(Case 1). Done.

Version the id (`-v1`, `-v2`, ...) instead of changing an env's semantics in
place — old run directories store the config used, and comparability across
runs depends on the id meaning one thing.

## Case 3 — The env is not in Gym format at all

Wrap it in a `gymnasium.Env` subclass so the rest of the framework never
knows the difference:

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MySimEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, **kwargs):
        self.render_mode = render_mode
        self.sim = ThirdPartySim(**kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)          # seeds self.np_random
        obs = self.sim.reset(seed=seed)
        return np.asarray(obs, dtype=np.float64), {}

    def step(self, action):
        obs, reward, done = self.sim.step(action)
        terminated = done                  # true environment termination
        truncated = False                  # time-limit cutoff (see below)
        return np.asarray(obs, dtype=np.float64), float(reward), terminated, truncated, {}

    def render(self):
        return self.sim.render_rgb()       # HxWx3 uint8, required only for video
```

Contract checklist (Gymnasium ≥ 1.0 API — the framework assumes all of this):

- `reset()` returns `(obs, info)`; `step()` returns the **5-tuple**
  `(obs, reward, terminated, truncated, info)`. If your source env uses the
  old 4-tuple `done` API, split it: `terminated` = real end state (affects
  bootstrapping in GAE), `truncated` = artificial cutoff (time limit).
- Seeding goes through `reset(seed=...)` — there is no `env.seed()`.
- Observations must be a `Box` (or a `Dict`, which the standard wrapper stack
  flattens) — the agent is an MLP over a flat float vector.
- Actions must be a continuous `Box` — `ppo_continuous_action` samples from a
  Gaussian and relies on `ClipAction`. Discrete action spaces (and image
  observations) are rejected with a `TypeError` at env-build time; they need a
  new algorithm with its own `default_wrappers`, not a new env.
- Wrap with a time limit if episodes can run forever: either register with
  `max_episode_steps=...` (Gymnasium adds `TimeLimit` for you) or wrap
  manually in your factory function.
- For video support, implement `render()` returning an RGB array and set
  `metadata["render_fps"]` (used to convert `video_length_seconds` to steps).
  If the env cannot render, see `supports_training_video` in Case 4.

Then register it exactly as in Case 2 (entry point can be the class itself:
`entry_point="envs.custom_envs.mysim:MySimEnv"`) and write a config.

## Case 4 — The env has quirks: use an adapter

Some envs don't fit the standard path: they ship their own vectoriser, record
episode statistics internally, can't render during training, or need a
non-trivial evaluation protocol (e.g. per-task success rates). Instead of
`if env_id == ...` checks in the core, declare the quirks in **one file** via
an [EnvAdapter](../envs/adapters/base.py):

| Field | Default | Set it when... |
|---|---|---|
| `make_vector_env` | `None` (AsyncVectorEnv of thunks) | the env has its own vector entry point (`gym.make_vec`), e.g. multi-task benchmarks |
| `apply_wrappers` | `None` (algorithm's default stack) | the env family needs different preprocessing than the algorithm assumes — e.g. Atari frame preprocessing, or no wrappers at all (`lambda env, env_id, gamma: env`). Replaces the whole stack; call `continuous_control_wrappers` yourself for partial reuse |
| `skip_episode_stats` | `False` | the env already applies `RecordEpisodeStatistics` internally (double-wrapping asserts on reset) |
| `supports_training_video` | `True` | the env can't `render()` per-step during training (video is then disabled with a note, instead of crashing) |
| `evaluate` | `None` (standard eval loop) | evaluation needs a custom protocol — per-task metrics, success rates, custom logging |

**Step 1.** Create `envs/adapters/myenv.py`:

```python
from envs.adapters.base import EnvAdapter, register_adapter

register_adapter(
    "MyEnv",                      # env-id *prefix*; longest match wins
    EnvAdapter(
        skip_episode_stats=True,
        supports_training_video=False,
    ),
)
```

**Step 2.** Add the import to the "bundled adapters" block at the bottom of
[envs/adapters/__init__.py](../envs/adapters/__init__.py):

```python
from envs.adapters import myenv  # noqa: F401
```

The worked, fully-commented example is
[envs/adapters/metaworld.py](../envs/adapters/metaworld.py): it registers a
custom vectoriser and per-task evaluation for `Meta-World/MT10|MT25|MT50`, a
plain `skip_episode_stats` adapter for every other `Meta-World/` env, and
skips registration entirely when the package isn't installed (guard your
adapter with `try: import <package>` the same way if the env comes from an
optional dependency).

A custom `evaluate` must return
`{"episodic_returns": list[float], "metrics": dict}` — the metrics dict is
logged to wandb as-is. See `Algorithm.evaluate` in
[algorithms/base.py](../algorithms/base.py) for how it is dispatched.

### Worked example: Google Research Football (Case 3 + Case 4)

[envs/custom_envs/gfootball.py](../envs/custom_envs/gfootball.py) wraps the
legacy-`gym` gfootball engine in a `gymnasium.Env` (Case 3): it splits `done`
into `terminated`/`truncated` using the engine's `steps_left`, maps
`reset(seed=...)` onto the engine's construction-time seed, and turns the slow
software renderer on only while `RecordVideo` is asking for frames.
[envs/custom_envs/__init__.py](../envs/custom_envs/__init__.py) registers one
id per Football Academy scenario (`GFootball/academy_empty_goal_close-v0`, …)
— guarded by `importlib.util.find_spec("gfootball")`, so nothing changes when
the package is absent. [envs/adapters/gfootball.py](../envs/adapters/gfootball.py)
is the Case 4 part: a one-liner `supports_training_video=False`.

The engine is a compiled C++ dependency; use
[docker/Dockerfile.gfootball](../docker/Dockerfile.gfootball) rather than
installing it natively. Observations are a flat 115-vector and actions are
`Discrete(19)`, so it runs on the `discrete_control` stack (DQN) unchanged —
see [configs/dqn_gfootball_empty_goal.yml](../configs/dqn_gfootball_empty_goal.yml).

---

## What the framework does to your env (know before debugging)

Between `gym.make` and the agent, every env gets a **preprocessing wrapper
stack** — a callable `(env, env_id, gamma) -> env` resolved per env (*gamma*
is the algorithm's discount factor, consumed by reward normalization; ignore
it in stacks that don't normalize rewards):

1. the env family's adapter override (`EnvAdapter.apply_wrappers`), if any —
   a permanent fact about the env family (rare);
2. otherwise the `env_wrappers:` config field — a per-run choice, naming a
   stack from `WRAPPER_STACKS` in [envs/wrappers.py](../envs/wrappers.py);
3. otherwise the **algorithm's** `default_wrappers` — the input contract its
   standard agent assumes.

If none of the three is set, env construction raises immediately with
instructions. Both 1 and 2 are customizable, you can write your own adapters inside envs/adapters/, and you can write your own stacks in  envs.wrapppers.py

### When to use each:

- Use adapters when the environment does not work well with gym wrappers and custom transformations need to be applied
- Use config and custom stacks when the environment works well with gym wrappers and you wish to test different ones in your experiments.

---

Other conventions worth knowing:

- **`env_kwargs` are part of the run identity.** The run directory is named
  `{env_id}__{hash(env_kwargs)}__{seed}__{timestamp}`, so two runs with
  different kwargs never collide. Keep kwargs JSON-serialisable (lists, not
  tuples; plain numbers/strings).
- **`/` in env ids is fine** (e.g. `Meta-World/MT10`) — it is flattened to
  `-` in run names.
- **Video**: only vector-worker 0 records. Eval records the first episode;
  training records fixed-length clips every `video_every_global_steps`. Both
  require `render_mode="rgb_array"` to work, which the factory passes to
  `gym.make` — your entry point must accept it.
- **Adapters use prefix matching**: `register_adapter("Meta-World/", ...)`
  covers every Meta-World env; a longer prefix
  (`"Meta-World/MT10"`) takes precedence for specific ids.

## Checklist and smoke test

1. [ ] Env registered (by its package, or in `envs/custom_envs/__init__.py`).
2. [ ] Entry point accepts `render_mode` and your `env_kwargs`.
3. [ ] Adapter created + imported, if the env has quirks (else skip).
4. [ ] Config in `configs/` with `env_id`, `env_kwargs`, and algorithm sections.
5. [ ] API check and short training run:

```bash
# API conformance (catches wrong step tuples, space mismatches, seeding bugs)
python - <<'EOF'
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import envs  # registers custom envs and adapters

check_env(gym.make("MyEnv-v1").unwrapped)
print("env check OK")
EOF

# 100k-step smoke run: no wandb, no video, small batch
python train.py --config my_config --override \
    track=false capture_video=false total_timesteps=100000 num_envs=4
```

Watch the smoke run for: episodic returns appearing in the logs (wrapper stack
working), no NaNs, and a `runs/<exp_name>/<run_name>/` directory with the
saved config YAML.

6. [ ] Run `pytest` — registering your env automatically opted it into the
   generic contract tests (spaces, finite obs/rewards, seed determinism).
   Then add a small **semantic test** for what your env actually does
   (reward math, observation features, scheduling) to
   `tests/test_custom_envs.py` — see [testing.md](testing.md) for the
   distinction and templates to copy.
