# Quickstart — Training

From a fresh clone to a trained, evaluated policy. Every command here was run
as written; total time is a few minutes on CPU.

## 1. Install

Python 3.10+ with a virtual environment (conda or venv):

```bash
pip install -r requirements.txt
```

Alternatively, use the containerized setup (GPU, headless rendering
preconfigured):

```bash
docker compose run t-zero   # builds from docker/Dockerfile, mounts the repo
```

Google Research Football (`GFootball/*` envs) needs a compiled game engine and
has its own image (CPU-only by default, ~4 GB; GPU build documented in
[docker/Dockerfile.gfootball](../docker/Dockerfile.gfootball)):

```bash
docker compose run --rm gfootball python train.py --config dqn_gfootball_empty_goal
```

Two environment variables matter:

- `MUJOCO_GL=egl` — required for MuJoCo rendering on headless machines
  (already set inside the container).
- Weights & Biases credentials — only if you want online experiment tracking:
  `cp .env.example .env` and fill in your values (`train.py` loads `.env`
  automatically; it is gitignored).

## 2. Train

```bash
python train.py --config ppo_pendulum
```

`--config` takes the **name of a file in `configs/`** (no path, no
extension). [configs/ppo_pendulum.yml](../configs/ppo_pendulum.yml) is the
commented quickstart config — PPO on `Pendulum-v1`, 500k steps, ~3 minutes
on CPU, ending around −200 mean return (near-optimal episodes score ≈ −1).

You'll see a settings banner, then a progress bar:

```
PPO iterations: 100%|██████████| 24/24 [..., sps=2794, gs=98304, r_roll=-224]
model saved to runs/ppo_pendulum/Pendulum-v1__empty__1__20260707_084551/model.pt
eval_episode=0, episodic_return=-233.5
...
```

`sps` = env steps per second, `gs` = global step, `r_roll` = rolling mean
episodic return (last 100 episodes). Because the config sets
`save_model: true`, training ends with a final evaluation automatically.

## 3. What lands on disk

```
runs/
└── ppo_pendulum/                                  # experiment = config name
    └── Pendulum-v1__empty__1__20260707_084551/    # run = env__kwargs-hash__seed__timestamp
        ├── config.yml            # full resolved config — provenance + resume + eval
        ├── model.pt              # final weights (state_dict)
        ├── checkpoint_gs*.pt     # rotating resume bundles while training
        │                         #   (deleted once the final model is saved)
        └── videos/               # only when capture_video: true
```

One config → one experiment folder; each seed/launch gets its own run folder
under it. `__empty__` is the hash slot for `env_kwargs` (this config has
none). Console + Weights & Biases (`track: true`) are the only logging
sinks — there is no TensorBoard writer.

## 4. Change things without editing the config

Any field can be overridden from the CLI with dotted paths:

```bash
# run-level fields
python train.py --config ppo_pendulum --override seed=7 total_timesteps=200000

# nested keys: network_kwargs.<arg> and <algorithm_name>.<field>
python train.py --config ppo_pendulum --override network_kwargs.activation=ReLU \
    ppo_continuous_action.learning_rate=3e-4

# algo. is an algorithm-agnostic alias for the algorithm's own section
python train.py --config ppo_pendulum --override algo.gamma=0.95

# dict fields (env_kwargs) take JSON — note the quotes
python train.py --config ppo_antdir_goal --override 'env_kwargs={"direction_pool": [[1.0, 0.0]]}'
```

Typos are safe: an unknown key (in the YAML or in an override) aborts at
startup with the list of valid keys — it can never silently train with
defaults.

## 5. Interrupt and resume

With `checkpoint_every > 0` the run periodically writes resume bundles.
Kill training at any point (Ctrl-C, preemption, crash) and continue with:

```bash
python train.py --resume runs/ppo_pendulum/Pendulum-v1__empty__1__20260707_084551
```

Everything is read from the run folder's `config.yml` and its latest
`checkpoint_gs*.pt`. See [checkpoint-semantics.md](checkpoint-semantics.md)
for exactly what is (and deliberately isn't) restored.

## 6. Evaluate and record videos

```bash
# one run
MUJOCO_GL=egl python evaluate.py runs/ppo_pendulum/Pendulum-v1__empty__1__20260707_084551

# or a whole experiment (every run under it)
MUJOCO_GL=egl python evaluate.py runs/ppo_pendulum --eval-episodes 5 --deterministic
```

Evaluation rebuilds the env and network from the run's saved `config.yml` —
no flags to repeat. `--deterministic` uses the policy mean (smoother videos),
`--no-video` prints returns only. Meta-World benchmark runs (MT10/…) have a
per-task protocol — use `scripts/eval_metaworld.py` for those.

## 7. Where to go next

| I want to… | Read |
|---|---|
| Train on my own environment | [adding-a-new-environment.md](adding-a-new-environment.md) |
| Add an algorithm (DQN, SAC, …) | [adding-a-new-algorithm.md](adding-a-new-algorithm.md) |
| Understand checkpoints/resume | [checkpoint-semantics.md](checkpoint-semantics.md) |
| Run the test suite | [testing.md](testing.md) |
| Launch many seeds / SLURM | `scripts/` and `cluster/` |
