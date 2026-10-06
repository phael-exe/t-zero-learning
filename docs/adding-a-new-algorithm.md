# Adding a New Algorithm

This guide explains how an algorithm plugs into the framework and walks
through adding a new one (e.g. DQN, SAC). The canonical reference
implementation to read alongside this guide is
[algorithms/ppo_continuous_action.py](../algorithms/ppo_continuous_action.py).

## How an algorithm travels through the code

Everything starts from one config key and ends in a training loop:

```
configs/my_config.yml       train.py                    core/config_loader.py
┌────────────────────┐   ┌──────────────────────┐   ┌──────────────────────────┐
│ algorithm: sac     │ → │ load_config()        │ → │ ALGORITHMS registry:     │
│ sac:               │   │ apply_overrides()    │   │  name → (module, class,  │
│   learning_rate: … │   │ algo_main(args, …)   │   │          args class)     │
└────────────────────┘   └──────────────────────┘   └──────────────────────────┘
                                                                │
                                     ┌──────────────────────────┴──────────────┐
                                     │ algorithms/sac.py::main(args, resume…)  │
                                     │   algo = SAC(args, resume_run_dir)      │
                                     │   algo.initialize()   # envs, nets, …   │
                                     │   algo.train()        # the loop        │
                                     └─────────────────────────────────────────┘
```

Note that the name `sac` appears both as the `algorithm:` value and as the
YAML section name — that repetition is deliberate, and Step 1 explains it
("the one-spelling rule").

Files you will touch:

- [algorithms/](../algorithms/) — one new self-contained module
- [core/config_loader.py](../core/config_loader.py) — one line in `ALGORITHMS`
- [tests/helpers.py](../tests/helpers.py) — one entry in `SMOKE_SETTINGS`
- [configs/](../configs/) — one YAML config per experiment
- [networks/](../networks/) — only if your agent needs a new architecture: a new file, exported from `networks/__init__.py`, then named by `network:` in the config

You should **not** need to touch `train.py`, `evaluate.py`, `core/checkpoint.py`,
or `envs/` (unless the algorithm needs a new wrapper stack — see
[envs/wrappers.py](../envs/wrappers.py)).

## Design philosophy

Algorithm files are **self-contained on purpose** (CleanRL style): the whole
training loop lives in one readable file, even at the cost of some duplication
between variants (`ppo_continuous_action.py` and
`ppo_continuous_action_split_optim.py` deliberately mirror each other).
The `Algorithm` base class only owns shared mechanics — run directories,
seeding, checkpoint plumbing, wandb — never the training logic.

### What the base class actually provides

Everything below lives in one file, [algorithms/base.py](../algorithms/base.py)
(~400 lines — worth reading in full once). Nothing else is inherited; if a
name is not in this table, it belongs to your algorithm file.

| Member of `Algorithm` | What it does |
|---|---|
| `initialize()` | Sets `self.env_id` / `self.env_kwargs` from the config, resolves `self.wrappers`, builds the run directory name (`env__kwargs-hash__seed__timestamp`), seeds `random`/`numpy`/`torch`, selects `self.device`. You override it and call it via `super().initialize()` first. |
| `_setup_logging_and_checkpoints()` | Writes the run's `config.yml` in the run folder, computes the checkpoint cadence (`self.next_checkpoint_step`, `self.checkpoints_keep`) and special-log cadence, and calls `wandb.init` when `track: true`. Needs `args.batch_size`, so call it after envs are built. |
| `_resume_from_checkpoint()` | Finds the latest `checkpoint_gs*.pt` in the run dir, calls **your** `load_checkpoint_state_dict()` with the algorithm payload, and restores RNG state. Call it last in `initialize()`, after agent and optimizers exist. |
| `evaluate(model_path, …)` | Post-training evaluation. Uses the env's adapter protocol if one is declared ([envs/adapters/](../envs/adapters/)), otherwise the generic single-env loop in `evaluate_checkpoint()` (a module-level function at the top of the same file). |
| `_post_training_eval()` | If `save_model: true`: saves the final `model.pt`, deletes the rotating checkpoints, runs `self.evaluate()`, and logs its metrics to wandb. Call it at the end of your `train()`. |
| `default_wrappers` (class attr) | The preprocessing stack your agent assumes — you must declare it (Step 2). |
| `train()`, `checkpoint_state_dict()`, `load_checkpoint_state_dict()` | Abstract — you implement them (Steps 2–3). |

After `super().initialize()` returns, these attributes exist on `self`:
`args`, `resuming`, `device`, `env_id`, `env_kwargs`, `wrappers`,
`run_dir`, `run_name`, `experiment_dir`.

---

## Step 1 — Config dataclasses

Create `algorithms/my_algo.py` with two dataclasses:

```python
from dataclasses import dataclass, field
from core.base_config import RunConfig


@dataclass
class MyAlgoConfig:
    """MyAlgo hyperparameters."""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    # ... every hyperparameter, each with a docstring


@dataclass
class Args(RunConfig):
    algorithm: str = "my_algo"
    network: str = "ContinuousActorCritic"
    """network class to build (see ``RunConfig.network``)"""

    my_algo: MyAlgoConfig = field(default_factory=MyAlgoConfig)
    """field name == algorithm name (the one-spelling rule)"""

    @property
    def algo(self) -> MyAlgoConfig:
        return self.my_algo
```

Notes:

- `RunConfig` ([core/base_config.py](../core/base_config.py)) provides all
  run-level fields (`seed`, `env_id`, `env_kwargs`, `total_timesteps`,
  `track`, `checkpoint_every`, …). Never redeclare those, except to give
  `network` your algorithm's default.
- The network is chosen by config, not by the algorithm: `network` names a
  class in [networks/__init__.py](../networks/__init__.py) and
  `network_kwargs` go straight to its constructor. Any network with the
  interface your training loop calls works. Its constructor signature
  documents the kwargs, and evaluation and export rebuild it from the saved
  `config.yml` the same way.
- Give **every field a docstring** — they are the source of truth for the
  config reference documentation.
- The `algo` property is what shared code (base class, tests, overrides)
  uses to reach your section without knowing its name.
- Unknown YAML keys are hard startup errors, so any field you add is
  immediately configurable and typo-safe.

### The one-spelling rule

Name the section field (`my_algo` above) **exactly** like the algorithm.
The same name then appears in four places, spelled identically:

| Place | Example |
|---|---|
| `Args` dataclass field holding the section | `my_algo: MyAlgoConfig = field(...)` |
| YAML hyperparameter section | `my_algo:` |
| `algorithm:` key in configs | `algorithm: my_algo` |
| `ALGORITHMS` registry key (Step 5) | `"my_algo"` |

This is not just convention. Saved run configs are produced by `asdict(args)`,
so the section in a run's `config.yml` is literally the field name; CLI
override paths address fields (`--override my_algo.learning_rate=1e-3`); and
resume re-loads the saved config through the same loader. One spelling keeps
all of these round-trips exact. (`algo.` also works in overrides as an
algorithm-agnostic shorthand, via the `algo` property.)

## Step 2 — The Algorithm subclass

Subclass [Algorithm](../algorithms/base.py) and follow its lifecycle:

```python
from algorithms.base import Algorithm
from envs import build_vector_envs, continuous_control_wrappers
from networks import get_network


class MyAlgo(Algorithm):

    default_wrappers = staticmethod(continuous_control_wrappers)
    """The preprocessing stack your agent's input contract assumes.
    Required — env construction fails loudly if this is None and the
    config doesn't name a stack. See envs/wrappers.py::WRAPPER_STACKS."""

    def initialize(self):
        super().initialize()   # run dir, seeding, device, env_id/env_kwargs
        args = self.args

        # 1. Build envs (self.env_id / self.env_kwargs / self.wrappers are set)
        self.envs, num_envs = build_vector_envs(...)

        # 2. Derived sizes, then logging/checkpoint scheduling
        args.batch_size = ...                 # must exist before the next call
        self._setup_logging_and_checkpoints() # config.yml, wandb, cadences

        # 3. Agent, optimizers, replay/rollout buffers — all on self
        self.agent = get_network(args.network)(self.envs, **args.network_kwargs).to(self.device)
        self.optimizer = ...
        self.global_step = 0

        # 4. Resume last (agent/optimizers must already exist)
        if self.resuming:
            self._resume_from_checkpoint()

    def train(self):
        ...  # the whole loop, in one place
        # final model save + evaluation (base class, see table above)
        self._post_training_eval()
```

Each numbered step calls into the base-class members described in the table
above — when in doubt about what one of them does, read it in
[algorithms/base.py](../algorithms/base.py); each is a short, documented
method.

### Agent-side observation normalization is an algorithm contract

`network_kwargs.use_obs_norm` (a [ContinuousActorCritic](../networks/actor_critic_network.py) argument) enables
a running mean/var observation normalizer that lives **inside the agent**
([networks/normalization.py](../networks/normalization.py)) — its statistics
are `state_dict` buffers, so checkpoints and `model.pt` carry them
automatically. But the agent only *holds* the statistics; **the algorithm's
rollout loop must drive them**. A config with `use_obs_norm: true` does
nothing useful under an algorithm that ignores this contract — the stats stay
at their init values and the agent silently trains on (approximately) raw
observations.

To support it, your rollout loop must do what both PPO variants do
(see the rollout loop in
[algorithms/ppo_continuous_action.py](../algorithms/ppo_continuous_action.py)):

```python
agent.update_norm(next_obs)                 # fold raw obs into running stats
obs_input = agent.normalize_obs(next_obs)   # normalize + clip to ±10
buffer_obs[step] = obs_input                # store the NORMALIZED obs
action, logprob, ... = agent.get_action_and_value(
    obs_input, input_is_normalized=True
)
```

Two rules matter:

1. **Store observations normalized with the stats at collection time** and
   pass `input_is_normalized=True` in the update phase. Re-normalizing stored
   raw obs later (with newer stats) would be inconsistent with the logprobs
   collected during rollout.
2. **Never call `update_norm` outside training rollout.** Evaluation and
   value bootstrapping use the frozen stats — `get_value(raw_obs)` /
   `get_action_and_value(raw_obs)` normalize internally by default.

All three calls are no-ops / identity when `use_obs_norm` is false, so the
loop needs no branching.

**Where variables live**: every training variable that must survive a resume
lives on `self` (`self.global_step`, `self.next_obs`, buffers, schedule
counters, …). Locals inside `train()` are only for read-only shorthands
(`args`, `cfg`, `device`). This is what makes the checkpoint contract below
trivial.

## Step 3 — The checkpoint contract

Implement the two abstract methods (see
[docs/checkpoint-semantics.md](checkpoint-semantics.md) for the envelope
format and what is deliberately *not* saved):

```python
def checkpoint_state_dict(self) -> dict:
    """Read everything from self; no arguments."""
    return {
        "agent_state_dict": self.agent.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        # ... every self.* field needed to continue training
    }

def load_checkpoint_state_dict(self, state: dict) -> None:
    self.agent.load_state_dict(state["agent_state_dict"])
    self.optimizer.load_state_dict(state["optimizer_state_dict"])
    self.global_step = int(state.get("global_step", 0))
    # ... restore the same fields
```

Inside `train()`, write checkpoints with
`core.checkpoint.save_checkpoint(...)` at your scheduled cadence
(`self.next_checkpoint_step`, `self.checkpoint_every` are set up by
`_setup_logging_and_checkpoints`). RNG state and `global_step` are handled by
the generic machinery — never put them in your payload.

If a checkpoint from a *different* algorithm could be structurally similar to
yours, it would be good practice to detect it in `load_checkpoint_state_dict` and raise a `ValueError` telling the user which algorithm to resume with (see the guard in `ppo_continuous_action.py`).

## Step 4 — The `main` entry point

`train.py` dispatches to a module-level `main`:

```python
def main(args: Args, resume_run_dir: Path | None = None):
    algo = MyAlgo(args, resume_run_dir)
    algo.initialize()
    algo.train()
```

## Step 5 — Register it

Add one line to `ALGORITHMS` in
[core/config_loader.py](../core/config_loader.py):

```python
ALGORITHMS = {
    ...
    "my_algo": ("algorithms.my_algo", "MyAlgo", "Args"),
}
```

## Step 6 — Smoke settings (the tests enforce this)

The train/resume smoke tests are parametrized over `ALGORITHMS`, and a guard
test fails until every registered algorithm has an entry in `SMOKE_SETTINGS`
in [tests/helpers.py](../tests/helpers.py):

```python
"my_algo": {
    "env_id": "Pendulum-v1",   # cheapest env with the right space types
    "run_overrides": {"total_timesteps": 256, "num_envs": 2},
    "algo_overrides": {...},   # smallest values that still complete an update
},
```

Rules: pick the cheapest compatible env (a discrete-action algorithm would use
`CartPole-v1`), and keep `total_timesteps` an exact multiple of the rollout /
update size so the final checkpoint lands exactly on `total_timesteps`.

Then run:

```bash
python -m pytest tests/ -q
```

You get, for free: config load/round-trip tests, override tests, an
end-to-end training smoke test, and a kill-and-resume determinism test.

## Step 7 — A config

Add `configs/my_algo_pendulum.yml`:

```yaml
algorithm: my_algo
env_id: Pendulum-v1
seed: 1
total_timesteps: 1000000

network_kwargs:
  activation: Tanh
  hidden_layers_size: 64

my_algo:
  learning_rate: 3.0e-4
  gamma: 0.99
```

and train:

```bash
python train.py --config my_algo_pendulum
```

---

## Checklist

- [ ] `MyAlgoConfig` dataclass — every field has a docstring
- [ ] `Args(RunConfig)` — section field named exactly like the algorithm; `algo` property
- [ ] `MyAlgo(Algorithm)` — `default_wrappers` declared; `initialize()` calls
      `super().initialize()`, then `_setup_logging_and_checkpoints()` after
      `args.batch_size`, then `_resume_from_checkpoint()` last
- [ ] All resumable training variables on `self`; `checkpoint_state_dict()` /
      `load_checkpoint_state_dict()` mirror each other
- [ ] `main(args, resume_run_dir=None)` at module level
- [ ] One line in `ALGORITHMS`
- [ ] One entry in `SMOKE_SETTINGS`; `pytest tests/ -q` green
- [ ] One config under `configs/`
