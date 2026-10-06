#!/usr/bin/env python3
"""Training entry point — loads a YAML config, resolves the algorithm, and dispatches."""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

from core.checkpoint import latest_checkpoint_path
from core.config_loader import (
    load_config,
    get_config_path,
    get_experiment_name,
    apply_overrides,
    _import_algorithm,
)


def main():
    parser = argparse.ArgumentParser(description="Train with YAML configuration")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=str,
        help="Name of config file in configs folder (without extension)",
    )
    group.add_argument(
        "--resume",
        type=str,
        metavar="RUN_DIR",
        help="Path to a run directory containing config.yml and at least one checkpoint_gs*.pt",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="Override config values (format: key=value, section.key=value or network_kwargs.key=value)",
    )

    cli_args = parser.parse_args()
    load_dotenv()

    if cli_args.resume:
        run_dir = Path(cli_args.resume).expanduser().resolve()
        config_path = run_dir / "config.yml"
        if not config_path.is_file():
            print(f"Error: {config_path} not found")
            sys.exit(1)
        if latest_checkpoint_path(run_dir) is None:
            print(f"Error: no checkpoint_gs*.pt files under {run_dir}")
            sys.exit(1)
        print(f"Loading configuration from {config_path}")
        args, algo_name = load_config(str(config_path))
        args.exp_name = run_dir.parent.name
        apply_overrides(args, cli_args.override)

        _, algo_main = _import_algorithm(algo_name)

        print("\n" + "=" * 50)
        print(f"Resume — {algo_name}")
        print("=" * 50)
        print(f"Run directory: {run_dir}")
        print(f"Experiment name: {args.exp_name}")
        print(f"Env: {args.env_id} kwargs={args.env_kwargs}")
        print(f"Total timesteps: {args.total_timesteps:,}")
        latest = latest_checkpoint_path(run_dir)
        print(f"Latest checkpoint: {latest}")
        import torch
        print(f"Device: {'CUDA' if args.cuda and torch.cuda.is_available() else 'CPU'}")
        print("=" * 50 + "\n")
        algo_main(args, resume_run_dir=run_dir)
        return

    # Fresh run from configs/ name
    config_path = get_config_path(cli_args.config)

    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found")
        print("Available configs in configs folder:")
        configs_dir = Path("configs")
        if configs_dir.exists():
            for config_file in sorted(configs_dir.glob("*.yml")):
                print(f"  - {config_file.stem}")
            for config_file in sorted(configs_dir.glob("*.yaml")):
                print(f"  - {config_file.stem}")
        sys.exit(1)

    print(f"Loading configuration from {config_path}")
    args, algo_name = load_config(config_path)

    args.exp_name = get_experiment_name(cli_args.config)

    apply_overrides(args, cli_args.override)

    _, algo_main = _import_algorithm(algo_name)

    print("\n" + "=" * 50)
    print(f"Training — {algo_name}")
    print("=" * 50)
    print(f"Experiment name: {args.exp_name}")
    print(f"Env: {args.env_id} kwargs={args.env_kwargs}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Learning rate: {args.algo.learning_rate}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Seed: {args.seed}")

    import torch
    print(f"Device: {'CUDA' if args.cuda and torch.cuda.is_available() else 'CPU'}")
    print("=" * 50 + "\n")

    algo_main(args)


if __name__ == "__main__":
    main()
