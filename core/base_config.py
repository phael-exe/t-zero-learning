"""Shared configuration dataclasses used by all algorithms."""

from dataclasses import dataclass, field


@dataclass
class RunConfig:
    """Run-level configuration shared by every algorithm.

    Algorithm-specific ``Args`` dataclasses inherit from this so that
    ``args.seed``, ``args.env_id``, etc. work without nesting.
    """
    exp_name: str = "experiment"
    """the name of this experiment (typically set by train.py from the config name)"""
    algorithm: str = ""
    """algorithm identifier — selects the YAML section and the code entry point"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """sets `torch.backends.cudnn.deterministic` to this value"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    capture_video: bool = False
    """master switch for video recording: enables the final evaluation video, and
    (only together with `video_every_global_steps` > 0) periodic training clips"""
    video_every_global_steps: int = 0
    """record a training video about every N **global env steps** (requires known episode
    horizon); <= 0 means no training clips even when `capture_video` is true"""
    video_length_seconds: float = 30.0
    """training clip length in seconds when using `video_every_global_steps` (converted to steps using env `render_fps`)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    checkpoint_every: int = 0
    """if > 0, save ``checkpoint_gs{step}.pt`` resume bundles every this many env timesteps"""
    checkpoints_to_keep: int = 3
    """when ``checkpoint_every`` > 0, keep at most this many checkpoint files (oldest deleted)"""
    special_log_every: int = 100000
    """how often (in global env steps) to do special logging (e.g., weight histograms)"""

    # Network
    network: str = ""
    """network class the algorithm builds: a name exported from ``networks/__init__.py``
    (e.g. 'QNetwork') or a dotted import path; each algorithm sets its own default"""
    network_kwargs: dict = field(default_factory=dict)
    """keyword arguments forwarded to the network constructor,
    ``network(envs, **network_kwargs)``; see its signature for what it accepts
    (e.g. activation, hidden_layers_size). Unset keys keep the constructor defaults"""

    # Environment
    env_id: str = "HalfCheetah-v4"
    """the gymnasium environment id (e.g. 'HalfCheetah-v4', 'Meta-World/MT10')"""
    env_kwargs: dict = field(default_factory=dict)
    """keyword arguments forwarded to ``gym.make(env_id, **env_kwargs)``"""
    env_wrappers: str = ""
    """named preprocessing stack from ``envs.wrappers.WRAPPER_STACKS`` (e.g.
    'continuous_control', 'none'); empty means the algorithm's default stack.
    Per-env adapter overrides (``EnvAdapter.apply_wrappers``) take precedence."""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""

    # Runtime-computed (populated by the algorithm's initialize())
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    env_kwargs_id: str = ""
    """short hash of env_kwargs used in run names (computed in runtime)"""
