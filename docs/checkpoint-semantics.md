# Checkpoint Semantics

What gets saved during training, what a resume actually restores, and — just
as important — what is deliberately *not* restored. The generic machinery
lives in [core/checkpoint.py](../core/checkpoint.py); each algorithm supplies
its own payload (see [adding-a-new-algorithm.md](adding-a-new-algorithm.md),
Step 3).

## The three artifacts in a run directory

| File | Written | Purpose |
|---|---|---|
| `config.yml` | once, at run start | Full resolved config (`asdict(args)`). Provenance, and the input for both `--resume` and `evaluate.py`. |
| `checkpoint_gs{step}.pt` | every `checkpoint_every` env steps | **Resume bundle** — everything needed to continue training. Rotated: at most `checkpoints_to_keep` newest are kept. Deleted once the final model is saved. |
| `model.pt` | once, at the end (`save_model: true`) | **Final weights only** (a plain `state_dict`, loadable with `weights_only=True`). For evaluation and downstream analysis — cannot resume training. |

## Anatomy of a resume bundle

```python
{
    "format_version": 2,      # loader refuses any other version
    "global_step": 123456,    # env steps completed so far
    "rng": {                  # main-process RNG, captured at save time
        "python":     ...,    # random.getstate()
        "numpy":      ...,    # np.random.get_state()
        "torch_cpu":  ...,    # torch.get_rng_state()
        "torch_cuda": ...,    # per-device states (None on CPU-only runs)
    },
    "algorithm": { ... },     # the algorithm's own payload, opaque to core/
}
```

The envelope (everything except `"algorithm"`) is owned by
`core/checkpoint.py` and is identical for every algorithm. The `"algorithm"`
payload is produced by the algorithm's `checkpoint_state_dict()` and consumed
by its `load_checkpoint_state_dict()` — for PPO that means agent and optimizer
`state_dict`s, the iteration counter, schedule counters (next checkpoint /
special-log step), the last observation batch, and the rolling episode-return
statistics.

`format_version` is bumped only if the envelope structure changes; the loader
raises on a mismatch rather than guessing.

## When checkpoints are written

`checkpoint_every` counts **global env steps** (summed over all parallel
envs), but a checkpoint can only be written between update iterations. So
with `num_envs: 4`, `num_steps: 1024` (batch 4096) and
`checkpoint_every: 10000`, bundles land at steps 12288, 20480, 32768, … —
the end of the first iteration that crosses each threshold. Missed
thresholds are skipped, not queued.

Rotation happens at save time: after writing the new file, the oldest
`checkpoint_gs*.pt` beyond `checkpoints_to_keep` are deleted. When training
finishes and `save_model: true`, the final `model.pt` is written
and **all** resume bundles are removed — a completed run keeps only config,
model, and videos.

## What resume restores — and what it doesn't

```bash
python train.py --resume runs/<exp>/<run_dir>
```

reads `config.yml`, finds the newest `checkpoint_gs*.pt`, and continues
training **in the same run directory** (no new folder is created).

Restored exactly:

- agent weights and optimizer state (Adam moments, LR schedule position)
- observation-normalization statistics (`network_kwargs.use_obs_norm`) — they are
  buffers inside the agent's `state_dict`, so they ride along in both
  `checkpoint_gs*.pt` and `model.pt` with no special handling
- `global_step`, iteration counter, checkpoint / logging cadence counters
- rolling episode statistics (the `r_roll` progress-bar number)
- main-process RNG streams (Python, NumPy, torch CPU + CUDA)

Deliberately **not** restored:

- **Environment simulator state.** MuJoCo/Gymnasium internals are not
  serialized. On resume the envs are freshly reset with
  `seed = args.seed + global_step` — the offset avoids replaying the same
  initial episodes — and the checkpointed last observation is overwritten by
  the reset. The in-flight episodes at kill time are simply abandoned.
- **Reward-normalization statistics.** `NormalizeReward` (part of the
  `continuous_control` wrapper stack) keeps a running std of the discounted
  return *inside the env*, which falls under "environment state" above: a
  resumed run rebuilds it from scratch, so the reward scale jolts briefly
  after resume. Training-signal shaping only — evaluation returns are
  unaffected.
- **wandb run identity.** With `track: true`, resuming starts a *new* wandb
  run logging from the restored `global_step` onward.

**Consequence**: a killed-and-resumed run is *not* bit-identical to an
uninterrupted one — the trajectory stream differs from the resume point
onward. Learning dynamics are unaffected in distribution; if you need
bit-exact reproducibility, don't interrupt the run.

## Extending or re-budgeting a run

`--resume` accepts the same `--override` flags as fresh training, so a run
that already hit its budget can be extended:

```bash
python train.py --resume runs/<exp>/<run_dir> --override total_timesteps=100000000
```

Resuming a run whose `global_step` already satisfies `total_timesteps`
prints "Training already complete" and exits without training.

## Failure modes (all loud, none silent)

| Situation | What happens |
|---|---|
| `--resume` dir without `config.yml` or without any `checkpoint_gs*.pt` | startup error naming the missing file |
| Checkpoint written by a different `format_version` | `ValueError` from the loader |
| Checkpoint from a structurally different algorithm (e.g. resuming a single-optimizer PPO run with the split-optimizer variant) | friendly `ValueError` naming the algorithm to use instead |

These behaviours are pinned by the test suite
([tests/test_resume_smoke.py](../tests/test_resume_smoke.py),
[tests/test_checkpoint.py](../tests/test_checkpoint.py) — see
[testing.md](testing.md)): resume continues to the new budget, stays in the
same run directory, and the cross-variant guard raises its specific error.
