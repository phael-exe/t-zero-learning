"""Treinamento de PPO com Stable-Baselines3, VecNormalize e o mesmo protocolo de avaliação do DQN."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from rlpraticas.envs import make_env
from rlpraticas.evaluation import evaluate


class LinearSchedule:
    """Decai linearmente de `start` até 0. Classe (e não lambda) para o modelo salvo ser portável."""

    def __init__(self, start: float):
        self.start = start

    def __call__(self, progress_remaining: float) -> float:
        return progress_remaining * self.start


def parse_schedule(value):
    if isinstance(value, str) and value.startswith("lin_"):
        return LinearSchedule(float(value[4:]))
    return float(value)


class PeriodicEval(BaseCallback):
    """Avalia em um ambiente separado usando as estatísticas de normalização CONGELADAS do treino."""

    def __init__(self, cfg: dict, run_dir: Path):
        super().__init__()
        self.cfg = cfg
        self.ev = cfg["evaluation"]
        self.eval_env = make_env(cfg, int(cfg["seed"]) + self.ev["eval_seed_offset"])
        self.last_eval = 0
        self.csv_file = open(run_dir / "eval.csv", "w", newline="", encoding="utf-8")
        self.csv_writer = None

    def policy(self, obs):
        venv = self.model.get_env()
        if isinstance(venv, VecNormalize):
            obs = venv.normalize_obs(obs)  # usa média/desvio atuais, sem atualizá-los
        action, _ = self.model.predict(obs, deterministic=self.ev["deterministic"])
        return int(action) if action.shape == () else action

    def run(self, n_episodes: int) -> dict:
        return evaluate(
            self.policy,
            self.eval_env,
            n_episodes=n_episodes,
            seed=self.ev["eval_seed_offset"],
            gamma=self.cfg["hyperparameters"]["gamma"],
        )

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval >= self.ev["eval_freq"]:
            self.last_eval = self.num_timesteps
            stats = self.run(int(self.ev["n_eval_episodes"]))
            for key, value in stats.items():
                self.logger.record(f"eval/{key}", value)
            self.logger.dump(self.num_timesteps)
            row = {"timestep": self.num_timesteps, **stats}
            if self.csv_writer is None:
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=list(row))
                self.csv_writer.writeheader()
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            print(
                f"[{self.cfg['experiment']}/{self.cfg.get('variant')}/seed{self.cfg['seed']}] "
                f"passo {self.num_timesteps:>7d} | avaliação={stats['mean_return']:8.1f}",
                flush=True,
            )
        return True

    def _on_training_end(self) -> None:
        self.csv_file.close()


def train(cfg: dict, run_dir: Path) -> dict:
    hp = dict(cfg["hyperparameters"])
    seed = int(cfg["seed"])
    norm = cfg.get("normalization", {})

    env_kwargs = {}
    if cfg.get("max_episode_steps"):
        env_kwargs["max_episode_steps"] = int(cfg["max_episode_steps"])
    venv = make_vec_env(cfg["env_id"], n_envs=int(hp.pop("n_envs")), seed=seed, env_kwargs=env_kwargs)
    if norm.get("norm_obs") or norm.get("norm_reward"):
        venv = VecNormalize(
            venv,
            norm_obs=bool(norm.get("norm_obs", True)),
            norm_reward=bool(norm.get("norm_reward", False)),
            clip_obs=float(norm.get("clip_obs", 10.0)),
            gamma=float(hp["gamma"]),
        )

    hidden = hp.pop("hidden_sizes", [64, 64])
    hp["learning_rate"] = parse_schedule(hp["learning_rate"])
    hp["clip_range"] = parse_schedule(hp["clip_range"])

    model = PPO(
        "MlpPolicy",
        venv,
        seed=seed,
        verbose=0,
        device="cpu",
        policy_kwargs={"net_arch": {"pi": hidden, "vf": hidden}},
        **hp,
    )
    model.set_logger(configure(str(run_dir), ["tensorboard", "csv"]))

    callback = PeriodicEval(cfg, run_dir)
    model.learn(total_timesteps=int(cfg["total_timesteps"]), callback=callback)

    final = callback.run(int(cfg["evaluation"]["n_final_episodes"]))
    final["timestep"] = int(model.num_timesteps)
    with open(run_dir / "final.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    model.save(run_dir / "model.zip")
    if isinstance(venv, VecNormalize):
        # Sem este arquivo o modelo salvo é inutilizável: ele espera observações normalizadas.
        venv.save(str(run_dir / "vecnormalize.pkl"))
    venv.close()
    callback.eval_env.close()
    return final
