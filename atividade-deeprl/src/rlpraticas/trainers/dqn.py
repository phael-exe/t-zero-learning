"""Laço de treinamento do DQN com registro no TensorBoard."""

from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from rlpraticas.agents.dqn_agent import DQNAgent
from rlpraticas.envs import make_env
from rlpraticas.evaluation import evaluate
from rlpraticas.utils.replay_buffer import ReplayBuffer


def linear_epsilon(step: int, total: int, hp: dict) -> float:
    start, end = hp["exploration_initial_eps"], hp["exploration_final_eps"]
    decay_steps = max(1, int(hp["exploration_fraction"] * total))
    frac = min(1.0, step / decay_steps)
    return start + frac * (end - start)


def train(cfg: dict, run_dir: Path) -> dict:
    hp = cfg["hyperparameters"]
    ev = cfg["evaluation"]
    seed = int(cfg["seed"])
    total = int(cfg["total_timesteps"])
    log_interval = int(cfg["logging"]["log_interval"])

    env = make_env(cfg, seed)
    eval_env = make_env(cfg, seed + ev["eval_seed_offset"])
    obs_dim = env.observation_space.shape[0]
    agent = DQNAgent(obs_dim, env.action_space.n, hp, seed)
    buffer = ReplayBuffer(int(hp["buffer_size"]), env.observation_space.shape, seed)

    writer = SummaryWriter(str(run_dir))
    eval_file = open(run_dir / "eval.csv", "w", newline="", encoding="utf-8")
    eval_csv = None

    def run_eval(step: int, n_episodes: int) -> dict:
        stats = evaluate(
            policy=lambda o: agent.act(o, epsilon=0.0),
            env=eval_env,
            n_episodes=n_episodes,
            seed=ev["eval_seed_offset"],  # mesmos estados iniciais em toda avaliação
            gamma=hp["gamma"],
            value_fn=lambda o: float(agent.q_values(o).max()),
        )
        for key, value in stats.items():
            writer.add_scalar(f"eval/{key}", value, step)
        return stats

    recent_returns: deque[float] = deque(maxlen=100)
    recent_lengths: deque[int] = deque(maxlen=100)
    recent_truncated: deque[int] = deque(maxlen=100)  # 1 se o episódio acabou por truncamento
    pending: dict[str, list[float]] = {}

    obs, _ = env.reset(seed=seed)
    for step in range(1, total + 1):
        epsilon = linear_epsilon(step, total, hp)
        action = agent.act(obs, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Ambiente único, sem autoreset: next_obs é a observação final real.
        buffer.add(obs, action, reward, next_obs, terminated, truncated)
        obs = next_obs

        if terminated or truncated:
            recent_returns.append(float(info["episode"]["r"]))
            recent_lengths.append(int(info["episode"]["l"]))
            recent_truncated.append(int(truncated and not terminated))
            obs, _ = env.reset()

        if step > hp["learning_starts"] and step % hp["train_freq"] == 0:
            for _ in range(int(hp["gradient_steps"])):
                for key, value in agent.update(buffer.sample(int(hp["batch_size"]))).items():
                    pending.setdefault(key, []).append(value)

        if step % hp["target_update_interval"] == 0:
            agent.sync_target()

        if step % log_interval == 0:
            writer.add_scalar("rollout/exploration_rate", epsilon, step)
            if recent_returns:
                writer.add_scalar("rollout/ep_rew_mean", np.mean(recent_returns), step)
                writer.add_scalar("rollout/ep_len_mean", np.mean(recent_lengths), step)
                writer.add_scalar("rollout/truncated_episode_frac", np.mean(recent_truncated), step)
            for key, values in pending.items():
                writer.add_scalar(f"train/{key}", float(np.mean(values)), step)
            pending.clear()

        if step % ev["eval_freq"] == 0:
            stats = run_eval(step, int(ev["n_eval_episodes"]))
            row = {"timestep": step, **stats}
            if eval_csv is None:
                eval_csv = csv.DictWriter(eval_file, fieldnames=list(row))
                eval_csv.writeheader()
            eval_csv.writerow(row)
            eval_file.flush()
            print(
                f"[{cfg['experiment']}/{cfg.get('variant')}/seed{seed}] "
                f"passo {step:>7d} | ε={epsilon:.2f} | "
                f"treino(média 100 ep)={np.mean(recent_returns) if recent_returns else float('nan'):8.1f} | "
                f"avaliação={stats['mean_return']:8.1f}",
                flush=True,
            )

    final = run_eval(total, int(ev["n_final_episodes"]))
    final["timestep"] = total
    with open(run_dir / "final.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    agent.save(run_dir / "model.pt")

    eval_file.close()
    writer.close()
    env.close()
    eval_env.close()
    return final
