"""Reavalia um modelo salvo, com opções que reproduzem erros comuns de avaliação.

Exemplos:
    python scripts/evaluate.py --run results/runs/ppo_cartpole/padrao/seed0
    python scripts/evaluate.py --run results/runs/ppo_cartpole/padrao/seed0 --sem-normalizacao
    python scripts/evaluate.py --run results/runs/ppo_cartpole/padrao/seed0 --estocastico
    python scripts/evaluate.py --run results/runs/dqn_acrobot/padrao/seed0 --epsilon 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
import yaml

from rlpraticas.envs import make_env
from rlpraticas.evaluation import evaluate


def build_policy(cfg: dict, run: Path, args):
    if cfg["algo"] == "dqn":
        from rlpraticas.agents.dqn_agent import DQNAgent

        env = make_env(cfg, 0)
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg["hyperparameters"], seed=0)
        agent.load(run / "model.pt")
        env.close()
        return lambda obs: agent.act(obs, epsilon=args.epsilon)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(run / "model.zip", device="cpu")
    stats_path = run / "vecnormalize.pkl"
    normalizer = None
    if stats_path.exists() and not args.sem_normalizacao:
        dummy = DummyVecEnv([lambda: make_env(cfg, 0)])
        normalizer = VecNormalize.load(str(stats_path), dummy)
        normalizer.training = False      # congela média e desvio
        normalizer.norm_reward = False   # avaliamos a recompensa original

    def policy(obs):
        if normalizer is not None:
            obs = normalizer.normalize_obs(obs)
        action, _ = model.predict(obs, deterministic=not args.estocastico)
        return int(action)

    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="pasta de uma execução (…/seedN)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345, help="semente dos episódios de avaliação")
    parser.add_argument("--epsilon", type=float, default=0.0, help="DQN: ε na avaliação")
    parser.add_argument("--estocastico", action="store_true", help="PPO: amostra ações em vez da moda")
    parser.add_argument(
        "--sem-normalizacao",
        action="store_true",
        help="PPO: ignora vecnormalize.pkl (reproduz o erro de não salvar as estatísticas)",
    )
    args = parser.parse_args()

    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    policy = build_policy(cfg, run, args)
    env = make_env(cfg, args.seed)
    stats = evaluate(policy, env, args.episodes, seed=args.seed, gamma=cfg["hyperparameters"]["gamma"])
    env.close()
    print(json.dumps({k: round(v, 2) for k, v in stats.items()}, indent=2))


if __name__ == "__main__":
    main()
